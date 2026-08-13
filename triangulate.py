"""Triangulate pheasant call coordinates from multi-recorder bearing detections.

This module implements a full end-to-end pipeline:
1. Read and validate input detections from multiple CSV files.
2. Parse WGS84 coordinate strings and project to EPSG:27700 (metres).
3. Collapse near-duplicate recorder placements into canonical site IDs.
4. Build call events observed by at least 3 unique sites.
5. Solve call locations using weighted, robust triangulation with bearing
   ambiguity resolution (bearing_1_deg vs bearing_2_deg).
6. Export canonical sites, event members, and triangulated call coordinates.

All geometric calculations are done in EPSG:27700 for metric stability.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.optimize import least_squares

# Input files with detector outputs.
INPUT_FILES = [
    "pheasant_results_sd1.csv",
    "pheasant_results_sd2.csv",
    "pheasant_results_sd3.csv",
]

# Filtering and grouping parameters.
TDOA_THRESHOLD_MS = 0.025
EVENT_WINDOW_SECONDS = 0.75
MIN_SITES_PER_EVENT = 3
EVENT_WINDOW_SWEEP_SECONDS = [0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 15.0]
MAX_PAIR_DELTA_SECONDS = 45.0

# Site clustering parameters in metres.
TARGET_SITE_COUNT = 6
SITE_CLUSTER_MIN_RADIUS_M = 1.0
SITE_CLUSTER_MAX_RADIUS_M = 20.0
SITE_CLUSTER_STEP_M = 0.25
MIN_RECORDS_PER_LOCATION = 2

# Weighting and fit-quality parameters.
TDOA_WEIGHT_LOW_MS = TDOA_THRESHOLD_MS
TDOA_WEIGHT_HIGH_MS = 0.25
MIN_WEIGHT = 0.05
ORIENTATION_SIGMA_DEG = 3.0

MAX_ACCEPTABLE_RMS_DEG = 18.0
MIN_GEOMETRY_SCORE = math.sin(math.radians(10.0))
MIN_BRANCH_MARGIN = 0.2

# Output files.
SITES_OUTPUT = "triangulation_sites.csv"
EVENT_MEMBERS_OUTPUT = "triangulation_event_members.csv"
SOLUTIONS_OUTPUT = "triangulated_calls.csv"
EVENT_DIAGNOSTICS_OUTPUT = "event_matching_diagnostics.csv"
PAIRWISE_OFFSET_OUTPUT = "site_pair_offset_estimates.csv"

# Local field time zone used to attach tzinfo to parsed local timestamps.
FIELD_TIMEZONE = ZoneInfo("Europe/London")


@dataclass
class ClusterSelection:
    """Result of choosing a site-clustering radius."""

    radius_m: float
    n_clusters: int
    labels: np.ndarray


@dataclass
class TriangulationResult:
    """Best triangulation candidate for one event."""

    x_m: float
    y_m: float
    chosen_bearings_deg: list[float]
    branch_bits: str
    objective: float
    rms_residual_deg: float
    geometry_score: float
    branch_margin: float


@dataclass
class OffsetCalibration:
    """Per-site time offsets relative to a reference site."""

    reference_site_id: int
    offsets_s: dict[int, float]
    site_pair_stats: pd.DataFrame


def extract_recorder_number(recorder_folder: str) -> int:
    """Extract recorder number from values like 'Recorder 3 - 1004 to 1704'."""

    if not isinstance(recorder_folder, str):
        raise ValueError(f"Invalid recorder_folder value: {recorder_folder!r}")

    match = re.search(r"Recorder\s+(\d+)", recorder_folder)
    if not match:
        raise ValueError(f"Could not parse recorder number from: {recorder_folder!r}")

    return int(match.group(1))


def parse_coordinate(coord: str) -> float:
    """Parse coordinate strings like 1.92444W or 50.93976N to signed degrees."""

    if not isinstance(coord, str) or len(coord) < 2:
        raise ValueError(f"Invalid coordinate value: {coord!r}")

    coord = coord.strip()
    hemi = coord[-1].upper()
    value = float(coord[:-1])

    if hemi in {"W", "S"}:
        return -value
    if hemi in {"E", "N"}:
        return value

    raise ValueError(f"Unsupported hemisphere suffix in coordinate: {coord!r}")


def parse_recording_start(date_str: str, time_str: str) -> datetime:
    """Parse DATE and TIME columns into a datetime.

    The detector files use mm-dd-YYYY in current datasets. A fallback parser for
    dd-mm-YYYY is included so malformed/legacy rows can still be interpreted.
    """

    combined = f"{date_str} {time_str}"
    fmts = ["%m-%d-%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S"]
    errors = []

    for fmt in fmts:
        try:
            return datetime.strptime(combined, fmt).replace(tzinfo=FIELD_TIMEZONE)
        except ValueError as exc:
            errors.append(str(exc))

    raise ValueError(
        f"Failed to parse DATE/TIME '{combined}'. Tried {fmts}. Errors: {errors}"
    )


def wrap_angle_deg(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-180, 180)."""

    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def bearing_from_points_deg(site_x: np.ndarray, site_y: np.ndarray, x: float, y: float) -> np.ndarray:
    """Compute bearing from site coordinates to candidate point.

    Bearing is clockwise from North to match the bearing convention used by
    the detector output.
    """

    dx = x - site_x
    dy = y - site_y
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def tdoa_weight(abs_tdoa_ms: np.ndarray) -> np.ndarray:
    """Map absolute TDOA to clipped reliability weights in [MIN_WEIGHT, 1]."""

    scaled = (abs_tdoa_ms - TDOA_WEIGHT_LOW_MS) / (TDOA_WEIGHT_HIGH_MS - TDOA_WEIGHT_LOW_MS)
    clipped = np.clip(scaled, 0.0, 1.0)
    return np.maximum(clipped, MIN_WEIGHT)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median of 1D values."""

    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    return float(v[np.searchsorted(cum, cutoff)])


def connected_components(points_xy: np.ndarray, radius_m: float) -> np.ndarray:
    """Cluster points by distance-threshold connectivity.

    Two points are connected if their Euclidean distance is <= radius_m.
    Clusters are connected components of this undirected graph.
    """

    n = points_xy.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    labels = -np.ones(n, dtype=int)
    cluster_id = 0

    # Pairwise distance matrix is practical here because unique recorder
    # placement count is small (dozens, not millions).
    diff = points_xy[:, None, :] - points_xy[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    adjacency = dist <= radius_m

    for start in range(n):
        if labels[start] != -1:
            continue

        stack = [start]
        labels[start] = cluster_id
        while stack:
            node = stack.pop()
            neighbors = np.where(adjacency[node])[0]
            for neigh in neighbors:
                if labels[neigh] == -1:
                    labels[neigh] = cluster_id
                    stack.append(neigh)

        cluster_id += 1

    return labels


def choose_site_clusters(unique_locs: pd.DataFrame, target_sites: int) -> ClusterSelection:
    """Sweep clustering radii and choose the best match to target site count."""

    points = unique_locs[["easting_m", "northing_m"]].to_numpy()

    radii = np.arange(
        SITE_CLUSTER_MIN_RADIUS_M,
        SITE_CLUSTER_MAX_RADIUS_M + SITE_CLUSTER_STEP_M,
        SITE_CLUSTER_STEP_M,
    )

    best: ClusterSelection | None = None
    for radius in radii:
        labels = connected_components(points, radius)
        n_clusters = len(np.unique(labels))
        current = ClusterSelection(radius_m=float(radius), n_clusters=n_clusters, labels=labels)

        if best is None:
            best = current
            continue

        best_distance = abs(best.n_clusters - target_sites)
        current_distance = abs(current.n_clusters - target_sites)

        # Prefer exact target if available; otherwise closest cluster count.
        # For ties, prefer smaller radius (more conservative merging).
        if current_distance < best_distance or (
            current_distance == best_distance and current.radius_m < best.radius_m
        ):
            best = current

    if best is None:
        raise RuntimeError("Failed to select site clusters; no clustering candidates evaluated.")

    return best


def load_and_prepare_data(files: Iterable[str]) -> pd.DataFrame:
    """Load detector CSVs, filter by TDOA threshold, and parse geometry/time columns."""

    frames = []
    for file_name in files:
        df = pd.read_csv(file_name)
        df["source_file"] = file_name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Rows before TDOA filter: {len(combined)}")

    # Keep rows with sufficient directional information.
    combined = combined[combined["tdoa_ms"].abs() >= TDOA_THRESHOLD_MS].copy()
    print(f"Rows after TDOA filter >= {TDOA_THRESHOLD_MS} ms: {len(combined)}")

    # Use recorder number as the canonical site identifier.
    combined["site_id"] = combined["recorder_folder"].map(extract_recorder_number)

    # Parse coordinates once into decimal degrees.
    combined["lon_deg"] = combined["LON"].map(parse_coordinate)
    combined["lat_deg"] = combined["LAT"].map(parse_coordinate)

    # Build absolute event timestamp from recording start + within-file peak time.
    starts = [
        parse_recording_start(d, t)
        for d, t in zip(combined["DATE"], combined["TIME"], strict=False)
    ]
    combined["recording_start"] = starts
    combined["event_time"] = [
        start + timedelta(seconds=float(offset_s))
        for start, offset_s in zip(starts, combined["peak_time_s"], strict=False)
    ]

    # Project WGS84 -> EPSG:27700 for all internal geometry.
    to_bng = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    eastings, northings = to_bng.transform(combined["lon_deg"].to_numpy(), combined["lat_deg"].to_numpy())
    combined["easting_m"] = eastings
    combined["northing_m"] = northings

    return combined


def estimate_nearest_neighbor_deltas(
    a_times: np.ndarray,
    b_times: np.ndarray,
    max_abs_delta_s: float,
) -> np.ndarray:
    """Estimate nearest-neighbor time deltas between two sorted timestamp arrays.

    Returned deltas are (b - a) in seconds for each timestamp in a where the
    closest b timestamp is within max_abs_delta_s.
    """

    if len(a_times) == 0 or len(b_times) == 0:
        return np.array([], dtype=float)

    b_ns = np.array([pd.Timestamp(ts).value for ts in b_times], dtype=np.int64)
    deltas = []

    for a in a_times:
        a_ns = int(pd.Timestamp(a).value)
        idx = np.searchsorted(b_ns, a_ns)

        candidates = []
        if idx > 0:
            candidates.append(b_ns[idx - 1])
        if idx < len(b_ns):
            candidates.append(b_ns[idx])

        if not candidates:
            continue

        cand = min(candidates, key=lambda x: abs(x - a_ns))
        delta_s = (cand - a_ns) / 1e9
        if abs(delta_s) <= max_abs_delta_s:
            deltas.append(delta_s)

    return np.array(deltas, dtype=float)


def calibrate_site_clock_offsets(df: pd.DataFrame) -> OffsetCalibration:
    """Estimate and apply per-site clock offsets from nearest-neighbor timing.

    Strategy:
    1. Choose the highest-support site as the reference site.
    2. For each other site, estimate delta = (site - reference) from nearest
       neighbor timestamp matches under MAX_PAIR_DELTA_SECONDS.
    3. Use the median delta as robust pairwise offset and set correction to
       negative delta so corrected time aligns with reference.
    """

    site_counts = df.groupby("site_id").size().sort_values(ascending=False)
    if site_counts.empty:
        raise RuntimeError("Cannot calibrate offsets: no site rows available.")

    reference_site = int(site_counts.index[0])
    offsets = {reference_site: 0.0}
    stats_rows = []

    # Match only within the same DATE/TIME minute block.
    # This prevents unrelated calls from being paired across different recordings.
    df_blocks = df.copy()
    df_blocks["time_block"] = [
        ts.strftime("%Y-%m-%d_%H:%M")
        for ts in df_blocks["recording_start"]
    ]

    ref_rows = df_blocks[df_blocks["site_id"] == reference_site].copy()

    for site_id in sorted(df_blocks["site_id"].unique()):
        if site_id == reference_site:
            continue

        site_rows = df_blocks[df_blocks["site_id"] == site_id].copy()

        ref_blocks = {
            block: grp["event_time"].sort_values().to_numpy()
            for block, grp in ref_rows.groupby("time_block")
        }
        site_blocks = {
            block: grp["event_time"].sort_values().to_numpy()
            for block, grp in site_rows.groupby("time_block")
        }

        common_blocks = sorted(set(ref_blocks).intersection(site_blocks))
        deltas = []
        for block in common_blocks:
            block_deltas = estimate_nearest_neighbor_deltas(
                ref_blocks[block],
                site_blocks[block],
                max_abs_delta_s=MAX_PAIR_DELTA_SECONDS,
            )
            if len(block_deltas):
                deltas.extend(block_deltas.tolist())

        deltas = np.array(deltas, dtype=float)

        if len(deltas) == 0:
            median_delta = 0.0
            mad_delta = np.nan
            p10 = np.nan
            p90 = np.nan
        else:
            median_delta = float(np.median(deltas))
            mad_delta = float(np.median(np.abs(deltas - median_delta)))
            p10, p90 = np.percentile(deltas, [10, 90])

        offsets[site_id] = -median_delta
        stats_rows.append(
            {
                "reference_site_id": reference_site,
                "other_site_id": site_id,
                "n_matches": len(deltas),
                "median_delta_s_other_minus_ref": median_delta,
                "mad_delta_s": mad_delta,
                "p10_delta_s": p10,
                "p90_delta_s": p90,
                "applied_correction_s": -median_delta,
            }
        )

    stats = pd.DataFrame(stats_rows)
    return OffsetCalibration(
        reference_site_id=reference_site,
        offsets_s=offsets,
        site_pair_stats=stats,
    )


def apply_site_clock_offsets(df: pd.DataFrame, calibration: OffsetCalibration) -> pd.DataFrame:
    """Apply per-site clock-offset corrections to produce corrected_event_time."""

    out = df.copy()
    out["clock_offset_s"] = out["site_id"].map(calibration.offsets_s).fillna(0.0)
    out["corrected_event_time"] = [
        ts + timedelta(seconds=float(offset_s))
        for ts, offset_s in zip(out["event_time"], out["clock_offset_s"], strict=False)
    ]
    return out


def assign_canonical_sites(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign recorder-based sites and compute per-site median coordinates."""

    site_rows = []
    for site_id, block in df.groupby("site_id"):
        site_rows.append(
            {
                "site_id": int(site_id),
                "n_detections": int(len(block)),
                "n_unique_locations": int(block[["LON", "LAT"]].drop_duplicates().shape[0]),
                "canonical_easting_m": float(block["easting_m"].median()),
                "canonical_northing_m": float(block["northing_m"].median()),
            }
        )

    sites = pd.DataFrame(site_rows).sort_values("site_id").reset_index(drop=True)

    # Add WGS84 coordinates for export readability.
    to_wgs84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(
        sites["canonical_easting_m"].to_numpy(),
        sites["canonical_northing_m"].to_numpy(),
    )
    sites["canonical_lon_deg"] = lon
    sites["canonical_lat_deg"] = lat

    df["site_id"] = df["site_id"].astype(int)

    # Replace per-row location with canonical site coordinates for solver stability.
    df = df.merge(
        sites[["site_id", "canonical_easting_m", "canonical_northing_m"]],
        on="site_id",
        how="left",
        validate="many_to_one",
    )

    print(f"Assigned {len(sites)} recorder-based site(s) with median coordinates.")

    return df, sites


def build_events(df: pd.DataFrame) -> pd.DataFrame:
    """Group detections into call events by temporal proximity and unique sites."""

    ordered = df.sort_values("corrected_event_time").copy().reset_index(drop=True)

    # Greedy temporal segmentation: rows within EVENT_WINDOW_SECONDS belong to
    # the same provisional event block.
    event_ids = []
    current_event = 0
    current_anchor = ordered.loc[0, "corrected_event_time"] if len(ordered) else None

    for row_time in ordered["corrected_event_time"]:
        if current_anchor is None:
            current_anchor = row_time
        elif (row_time - current_anchor).total_seconds() > EVENT_WINDOW_SECONDS:
            current_event += 1
            current_anchor = row_time
        event_ids.append(current_event)

    ordered["event_id"] = event_ids

    # Within each event and site, keep the strongest directional cue.
    ordered["abs_tdoa_ms"] = ordered["tdoa_ms"].abs()
    deduped = (
        ordered.sort_values("abs_tdoa_ms", ascending=False)
        .drop_duplicates(subset=["event_id", "site_id"], keep="first")
        .sort_values(["event_id", "corrected_event_time"])
        .reset_index(drop=True)
    )

    site_counts = deduped.groupby("event_id")["site_id"].nunique()
    valid_events = site_counts[site_counts >= MIN_SITES_PER_EVENT].index
    deduped = deduped[deduped["event_id"].isin(valid_events)].copy()

    # Renumber event IDs to a compact sequence for cleaner outputs.
    remap = {old: new for new, old in enumerate(sorted(deduped["event_id"].unique()))}
    deduped["event_id"] = deduped["event_id"].map(remap)

    print(f"Events with >= {MIN_SITES_PER_EVENT} unique sites: {deduped['event_id'].nunique()}")
    return deduped


def count_multi_site_events(df: pd.DataFrame, window_seconds: float, min_sites: int) -> int:
    """Count events with at least min_sites unique sites for a given window size."""

    ordered = df.sort_values("corrected_event_time").copy().reset_index(drop=True)
    if ordered.empty:
        return 0

    event_ids = []
    current_event = 0
    current_anchor = ordered.loc[0, "corrected_event_time"]

    for row_time in ordered["corrected_event_time"]:
        if (row_time - current_anchor).total_seconds() > window_seconds:
            current_event += 1
            current_anchor = row_time
        event_ids.append(current_event)

    ordered["event_id"] = event_ids
    ordered["abs_tdoa_ms"] = ordered["tdoa_ms"].abs()
    deduped = ordered.sort_values("abs_tdoa_ms", ascending=False).drop_duplicates(
        subset=["event_id", "site_id"],
        keep="first",
    )

    site_counts = deduped.groupby("event_id")["site_id"].nunique()
    return int((site_counts >= min_sites).sum())


def build_event_matching_diagnostics(
    raw_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    calibration: OffsetCalibration,
) -> pd.DataFrame:
    """Create diagnostics for timing alignment and event-window sensitivity."""

    rows = []

    for window_s in EVENT_WINDOW_SWEEP_SECONDS:
        raw_count = count_multi_site_events(raw_df.assign(corrected_event_time=raw_df["event_time"]), window_s, MIN_SITES_PER_EVENT)
        corrected_count = count_multi_site_events(corrected_df, window_s, MIN_SITES_PER_EVENT)
        rows.append(
            {
                "diagnostic_type": "window_sweep",
                "window_seconds": float(window_s),
                "multi_site_events_raw": raw_count,
                "multi_site_events_offset_corrected": corrected_count,
                "improvement": corrected_count - raw_count,
                "reference_site_id": calibration.reference_site_id,
            }
        )

    for _, row in calibration.site_pair_stats.iterrows():
        rows.append(
            {
                "diagnostic_type": "pair_offset",
                "window_seconds": np.nan,
                "multi_site_events_raw": np.nan,
                "multi_site_events_offset_corrected": np.nan,
                "improvement": np.nan,
                "reference_site_id": int(row["reference_site_id"]),
                "other_site_id": int(row["other_site_id"]),
                "n_matches": int(row["n_matches"]),
                "median_delta_s_other_minus_ref": row["median_delta_s_other_minus_ref"],
                "mad_delta_s": row["mad_delta_s"],
                "p10_delta_s": row["p10_delta_s"],
                "p90_delta_s": row["p90_delta_s"],
                "applied_correction_s": row["applied_correction_s"],
            }
        )

    return pd.DataFrame(rows)


def robust_residuals(
    xy: np.ndarray,
    site_x: np.ndarray,
    site_y: np.ndarray,
    chosen_bearings_deg: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Residual function for weighted robust least-squares triangulation."""

    pred = bearing_from_points_deg(site_x, site_y, x=xy[0], y=xy[1])
    residual_deg = wrap_angle_deg(chosen_bearings_deg - pred)

    # Orientation sigma scales residuals into a consistent robust-loss domain.
    scaled = residual_deg / ORIENTATION_SIGMA_DEG
    return np.sqrt(weights) * scaled


def geometry_score_from_bearings(bearings_deg: np.ndarray) -> float:
    """Compute a simple geometry score; lower values indicate near-parallel rays."""

    if len(bearings_deg) < 2:
        return 0.0

    min_score = 1.0
    for i in range(len(bearings_deg)):
        for j in range(i + 1, len(bearings_deg)):
            diff = abs(float(wrap_angle_deg(bearings_deg[i] - bearings_deg[j])))
            score = abs(math.sin(math.radians(diff)))
            min_score = min(min_score, score)
    return min_score


def triangulate_one_event(event_df: pd.DataFrame) -> TriangulationResult | None:
    """Triangulate a single event by searching all bearing-ambiguity branches."""

    event_df = event_df.sort_values("site_id").reset_index(drop=True)
    site_x = event_df["canonical_easting_m"].to_numpy(dtype=float)
    site_y = event_df["canonical_northing_m"].to_numpy(dtype=float)
    b1 = event_df["bearing_1_deg"].to_numpy(dtype=float)
    b2 = event_df["bearing_2_deg"].to_numpy(dtype=float)
    weights = tdoa_weight(event_df["abs_tdoa_ms"].to_numpy(dtype=float))

    x0 = np.array([np.mean(site_x), np.mean(site_y)], dtype=float)

    pad = 1000.0
    lower = np.array([site_x.min() - pad, site_y.min() - pad], dtype=float)
    upper = np.array([site_x.max() + pad, site_y.max() + pad], dtype=float)

    candidates = []
    for bits in itertools.product([0, 1], repeat=len(event_df)):
        chosen = np.where(np.array(bits, dtype=int) == 0, b1, b2)

        result = least_squares(
            robust_residuals,
            x0=x0,
            bounds=(lower, upper),
            loss="huber",
            f_scale=1.0,
            args=(site_x, site_y, chosen, weights),
        )

        if not result.success:
            continue

        pred = bearing_from_points_deg(site_x, site_y, x=result.x[0], y=result.x[1])
        residual_deg = wrap_angle_deg(chosen - pred)
        rms = float(np.sqrt(np.sum(weights * residual_deg**2) / np.sum(weights)))
        geom = geometry_score_from_bearings(chosen)

        candidates.append(
            {
                "xy": result.x,
                "chosen": chosen,
                "bits": "".join(str(int(v)) for v in bits),
                "objective": float(result.cost),
                "rms": rms,
                "geom": geom,
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["objective"])
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    margin = float((second["objective"] - best["objective"]) if second else np.inf)

    return TriangulationResult(
        x_m=float(best["xy"][0]),
        y_m=float(best["xy"][1]),
        chosen_bearings_deg=[float(v) for v in best["chosen"]],
        branch_bits=best["bits"],
        objective=best["objective"],
        rms_residual_deg=best["rms"],
        geometry_score=best["geom"],
        branch_margin=margin,
    )


def solve_all_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Solve triangulation for each event and return one row per event."""

    output_columns = [
        "event_id",
        "n_sites",
        "event_time_min",
        "event_time_max",
        "triangulation_success",
        "quality_pass",
        "solution_easting_m",
        "solution_northing_m",
        "solution_lon_deg",
        "solution_lat_deg",
        "branch_bits",
        "objective",
        "rms_residual_deg",
        "geometry_score",
        "branch_margin",
        "chosen_bearings_deg",
    ]

    if events_df.empty:
        return pd.DataFrame(columns=output_columns)

    rows = []
    for event_id, group in events_df.groupby("event_id"):
        solution = triangulate_one_event(group)
        if solution is None:
            rows.append(
                {
                    "event_id": int(event_id),
                    "n_sites": int(group["site_id"].nunique()),
                    "triangulation_success": False,
                    "quality_pass": False,
                }
            )
            continue

        # Convert solved BNG point back to WGS84 for map compatibility.
        to_wgs84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
        lon, lat = to_wgs84.transform(solution.x_m, solution.y_m)

        quality_pass = (
            solution.rms_residual_deg <= MAX_ACCEPTABLE_RMS_DEG
            and solution.geometry_score >= MIN_GEOMETRY_SCORE
            and solution.branch_margin >= MIN_BRANCH_MARGIN
        )

        rows.append(
            {
                "event_id": int(event_id),
                "n_sites": int(group["site_id"].nunique()),
                "event_time_min": group["event_time"].min(),
                "event_time_max": group["event_time"].max(),
                "triangulation_success": True,
                "quality_pass": bool(quality_pass),
                "solution_easting_m": solution.x_m,
                "solution_northing_m": solution.y_m,
                "solution_lon_deg": lon,
                "solution_lat_deg": lat,
                "branch_bits": solution.branch_bits,
                "objective": solution.objective,
                "rms_residual_deg": solution.rms_residual_deg,
                "geometry_score": solution.geometry_score,
                "branch_margin": solution.branch_margin,
                "chosen_bearings_deg": "|".join(f"{b:.3f}" for b in solution.chosen_bearings_deg),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=output_columns)

    out = out.sort_values("event_id").reset_index(drop=True)
    return out


def main() -> None:
    """Run the full triangulation workflow and write outputs."""

    root = Path.cwd()
    input_paths = [root / name for name in INPUT_FILES]
    missing = [str(p) for p in input_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {missing}")

    combined = load_and_prepare_data(INPUT_FILES)
    combined_with_sites, sites = assign_canonical_sites(combined)
    calibration = calibrate_site_clock_offsets(combined_with_sites)
    corrected = apply_site_clock_offsets(combined_with_sites, calibration)
    diagnostics = build_event_matching_diagnostics(combined_with_sites, corrected, calibration)
    event_members = build_events(corrected)
    solutions = solve_all_events(event_members)

    sites.to_csv(root / SITES_OUTPUT, index=False)
    calibration.site_pair_stats.to_csv(root / PAIRWISE_OFFSET_OUTPUT, index=False)
    diagnostics.to_csv(root / EVENT_DIAGNOSTICS_OUTPUT, index=False)
    event_members.to_csv(root / EVENT_MEMBERS_OUTPUT, index=False)
    solutions.to_csv(root / SOLUTIONS_OUTPUT, index=False)

    print(f"Wrote canonical sites to {SITES_OUTPUT} ({len(sites)} rows)")
    print(
        f"Wrote per-site offset estimates to {PAIRWISE_OFFSET_OUTPUT} "
        f"({len(calibration.site_pair_stats)} rows)"
    )
    print(
        f"Wrote event diagnostics to {EVENT_DIAGNOSTICS_OUTPUT} "
        f"({len(diagnostics)} rows)"
    )
    print(f"Wrote event members to {EVENT_MEMBERS_OUTPUT} ({len(event_members)} rows)")
    print(f"Wrote triangulated solutions to {SOLUTIONS_OUTPUT} ({len(solutions)} rows)")

    if len(solutions):
        n_pass = int(solutions["quality_pass"].sum())
        print(f"Quality-pass solutions: {n_pass}/{len(solutions)}")


if __name__ == "__main__":
    main()

