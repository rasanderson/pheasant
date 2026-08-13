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

import io
import itertools
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
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
EVENT_WINDOW_SECONDS = 1.5
MIN_SITES_PER_EVENT = 3
EVENT_WINDOW_SWEEP_SECONDS = [0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0]
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
# This short-baseline array can show several degrees of bearing drift from
# orientation and TDOA estimation errors, so the robust solver needs a wider
# residual scale than the original 3° assumption.
ORIENTATION_SIGMA_DEG = 20.0

MAX_ACCEPTABLE_RMS_DEG = 18.0
MIN_GEOMETRY_SCORE = math.sin(math.radians(10.0))
MIN_BRANCH_MARGIN = 0.2

# Output files.
SITES_OUTPUT = "triangulation_sites.csv"
EVENT_MEMBERS_OUTPUT = "triangulation_event_members.csv"
SOLUTIONS_OUTPUT = "triangulated_calls.csv"
EVENT_DIAGNOSTICS_OUTPUT = "event_matching_diagnostics.csv"
PAIRWISE_OFFSET_OUTPUT = "site_pair_offset_estimates.csv"
MAP_HTML_OUTPUT = "triangulation_sites_map.html"
MAP_PNG_OUTPUT = "triangulation_sites_map.png"

# Map rendering parameters.
MAP_TILE_SIZE = 256
MAP_STATIC_WIDTH = 1400
MAP_STATIC_HEIGHT = 1000
MAP_LABEL_PREFIX = "Recorder"
MAP_HTML_TITLE = "Pheasant recorder locations"
HTML_MAP_MARGIN_PX = 30
STATIC_MAP_MARGIN_FRACTION = 0.25
STATIC_MAP_MAX_ZOOM = 18
STATIC_MAP_MIN_ZOOM = 3
MAP_BEARING_HALF_LENGTH_DEFAULT_M = 100.0
MAP_BEARING_HALF_LENGTH_MIN_M = 100.0
MAP_BEARING_HALF_LENGTH_MAX_M = 750.0
MAP_BEARING_LENGTH_STEP_M = 25.0

OSM_TILE_SOURCE = {
    "name": "OpenStreetMap",
    "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "attribution": "&copy; OpenStreetMap contributors",
}
SATELLITE_TILE_SOURCE = {
    "name": "Esri World Imagery",
    "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "attribution": "Tiles &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community",
}

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


def _site_map_records(sites: pd.DataFrame) -> list[dict[str, float | int | str]]:
    """Convert canonical site rows into map-friendly records."""

    records = []
    for row in sites.sort_values("site_id").itertuples(index=False):
        records.append(
            {
                "site_id": int(row.site_id),
                "label": f"{MAP_LABEL_PREFIX} {int(row.site_id)}",
                "n_detections": int(row.n_detections),
                "lon": float(row.canonical_lon_deg),
                "lat": float(row.canonical_lat_deg),
            }
        )
    return records


def _bearing_line_records(detections: pd.DataFrame) -> list[dict[str, float | int]]:
    """Convert detection rows into bearing records centered on each recorder."""

    if detections.empty:
        return []

    to_wgs84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    records = []
    sort_columns = ["site_id"]
    if "corrected_event_time" in detections.columns:
        sort_columns.append("corrected_event_time")
    elif "event_time" in detections.columns:
        sort_columns.append("event_time")

    for det_id, row in enumerate(detections.sort_values(sort_columns).itertuples(index=False), start=1):
        site_x = float(row.canonical_easting_m)
        site_y = float(row.canonical_northing_m)
        bearing_values = [float(row.bearing_1_deg), float(row.bearing_2_deg)]

        for branch_index, bearing_deg in enumerate(bearing_values, start=1):
            lon, lat = to_wgs84.transform(site_x, site_y)

            records.append(
                {
                    "det_id": det_id,
                    "site_id": int(row.site_id),
                    "branch_index": branch_index,
                    "bearing_deg": bearing_deg,
                    "center_lon": float(lon),
                    "center_lat": float(lat),
                }
            )

    return records


def _site_bounds(records: list[dict[str, float | int | str]], pad_fraction: float = 0.25) -> tuple[float, float, float, float]:
    """Return lon/lat bounds expanded by a fractional margin."""

    lons = np.array([float(record["lon"]) for record in records], dtype=float)
    lats = np.array([float(record["lat"]) for record in records], dtype=float)

    lon_min = float(lons.min())
    lon_max = float(lons.max())
    lat_min = float(lats.min())
    lat_max = float(lats.max())

    lon_span = max(lon_max - lon_min, 0.0001)
    lat_span = max(lat_max - lat_min, 0.0001)
    lon_pad = lon_span * pad_fraction
    lat_pad = lat_span * pad_fraction

    return lon_min - lon_pad, lat_min - lat_pad, lon_max + lon_pad, lat_max + lat_pad


def _lonlat_to_world_px(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Convert lon/lat to Web Mercator world pixels."""

    lat = max(min(lat, 85.05112878), -85.05112878)
    world_size = MAP_TILE_SIZE * (2**zoom)
    x = (lon + 180.0) / 360.0 * world_size
    sin_lat = math.sin(math.radians(lat))
    y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * world_size
    return x, y


def _choose_static_zoom(bounds: tuple[float, float, float, float]) -> int:
    """Choose the highest zoom where the padded bounds fit the output canvas."""

    lon_min, lat_min, lon_max, lat_max = bounds
    for zoom in range(STATIC_MAP_MAX_ZOOM, STATIC_MAP_MIN_ZOOM - 1, -1):
        x1, y1 = _lonlat_to_world_px(lon_min, lat_max, zoom)
        x2, y2 = _lonlat_to_world_px(lon_max, lat_min, zoom)
        if abs(x2 - x1) <= MAP_STATIC_WIDTH * 0.9 and abs(y2 - y1) <= MAP_STATIC_HEIGHT * 0.9:
            return zoom
    return STATIC_MAP_MIN_ZOOM


def _tile_provider(style: str) -> dict[str, str]:
    """Return the tile source configuration for a given backdrop style."""

    if style == "satellite":
        return SATELLITE_TILE_SOURCE
    return OSM_TILE_SOURCE


def _download_tile(tile_url: str) -> Image.Image:
    """Fetch a single map tile image."""

    request = Request(tile_url, headers={"User-Agent": "pheasant-triangulation-map/1.0"})
    with urlopen(request, timeout=20) as response:
        return Image.open(io.BytesIO(response.read())).convert("RGB")


def _world_px_to_canvas_px(
    lon: float,
    lat: float,
    zoom: int,
    canvas_origin_x: float,
    canvas_origin_y: float,
) -> tuple[float, float]:
    """Project lon/lat into canvas pixel coordinates."""

    world_x, world_y = _lonlat_to_world_px(lon, lat, zoom)
    return world_x - canvas_origin_x, world_y - canvas_origin_y


def write_interactive_site_map(sites: pd.DataFrame, detections: pd.DataFrame, output_path: Path) -> None:
    """Write a lightweight Leaflet map showing recorder locations."""

    records = _site_map_records(sites)
    if not records:
        raise RuntimeError("Cannot write map: no sites provided.")

    bearing_records = _bearing_line_records(detections)
    bounds = _site_bounds(records)
    lon_min, lat_min, lon_max, lat_max = bounds
    site_json = json.dumps(records, indent=2)
    bearing_json = json.dumps(bearing_records, indent=2)

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>{MAP_HTML_TITLE}</title>
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
  <style>
    html, body {{ height: 100%; margin: 0; }}
    #map {{ width: 100%; height: 100%; }}
    body {{ font-family: system-ui, sans-serif; }}
        .bearing-length-control {{
            background: rgba(255, 255, 255, 0.95);
            padding: 10px 12px;
            border-radius: 8px;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
            min-width: 220px;
            color: #1f2937;
        }}
        .bearing-length-control label {{
            display: block;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 6px;
        }}
        .bearing-length-control input[type=range] {{
            width: 100%;
            margin: 6px 0 4px;
        }}
        .bearing-length-readout {{
            font-size: 12px;
            line-height: 1.3;
        }}
    .site-tooltip {{
      background: rgba(20, 20, 20, 0.88);
      color: white;
      border: 0;
      border-radius: 6px;
      box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
      padding: 4px 8px;
    }}
  </style>
</head>
<body>
  <div id=\"map\"></div>
  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
  <script>
    const sites = {site_json};
        const bearings = {bearing_json};
    const map = L.map('map', {{ scrollWheelZoom: true }});

    const osm = L.tileLayer('{OSM_TILE_SOURCE['url']}', {{
      attribution: '{OSM_TILE_SOURCE['attribution']}',
      maxZoom: 20,
    }});
    const satellite = L.tileLayer('{SATELLITE_TILE_SOURCE['url']}', {{
      attribution: '{SATELLITE_TILE_SOURCE['attribution']}',
      maxZoom: 20,
    }});

    osm.addTo(map);

    const markers = L.layerGroup();
        const bearingLines = L.layerGroup();

        function destinationPoint(lat, lon, bearingDeg, distanceM) {{
            const earthRadiusM = 6378137.0;
            const bearingRad = bearingDeg * Math.PI / 180.0;
            const latRad = lat * Math.PI / 180.0;
            const lonRad = lon * Math.PI / 180.0;
            const angularDistance = distanceM / earthRadiusM;

            const sinLat = Math.sin(latRad);
            const cosLat = Math.cos(latRad);
            const sinAngular = Math.sin(angularDistance);
            const cosAngular = Math.cos(angularDistance);

            const lat2 = Math.asin(
                sinLat * cosAngular + cosLat * sinAngular * Math.cos(bearingRad)
            );
            const lon2 = lonRad + Math.atan2(
                Math.sin(bearingRad) * sinAngular * cosLat,
                cosAngular - sinLat * Math.sin(lat2)
            );

            return [
                lat2 * 180.0 / Math.PI,
                ((lon2 * 180.0 / Math.PI + 540.0) % 360.0) - 180.0,
            ];
        }}

        function rebuildBearingLines(halfLengthM) {{
            bearingLines.clearLayers();
            bearings.forEach((segment) => {{
                const startPoint = destinationPoint(
                    segment.center_lat,
                    segment.center_lon,
                    segment.bearing_deg + 180.0,
                    halfLengthM
                );
                const endPoint = destinationPoint(
                    segment.center_lat,
                    segment.center_lon,
                    segment.bearing_deg,
                    halfLengthM
                );

                L.polyline(
                    [startPoint, endPoint],
                    {{
                        color: '#2b6cb0',
                        weight: 2,
                        opacity: 0.55,
                        lineCap: 'round',
                    }}
                ).addTo(bearingLines);
            }});
        }}

        rebuildBearingLines({MAP_BEARING_HALF_LENGTH_DEFAULT_M});

    sites.forEach((site) => {{
      L.circleMarker([site.lat, site.lon], {{
        radius: 8,
        color: '#ffffff',
        weight: 2,
        fillColor: '#e44',
        fillOpacity: 0.95,
      }})
        .bindTooltip(site.label, {{ permanent: true, direction: 'top', offset: [0, -8], className: 'site-tooltip' }})
        .bindPopup(`<strong>${{site.label}}</strong><br/>Detections: ${{site.n_detections}}<br/>Lon: ${{site.lon.toFixed(6)}}<br/>Lat: ${{site.lat.toFixed(6)}}`)
        .addTo(markers);
    }});

    markers.addTo(map);
        L.control.layers(
            {{ 'OpenStreetMap': osm, 'Satellite': satellite }},
            {{ 'Recorder locations': markers, 'Bearing lines': bearingLines }},
            {{ collapsed: false }}
        ).addTo(map);

        const bearingLengthControl = L.control({{ position: 'topright' }});
        bearingLengthControl.onAdd = function () {{
            const div = L.DomUtil.create('div', 'bearing-length-control leaflet-bar');
            div.innerHTML = `
                <label for="bearing-length-slider">Bearing line length</label>
                <input
                    id="bearing-length-slider"
                    type="range"
                    min="{MAP_BEARING_HALF_LENGTH_MIN_M}"
                    max="{MAP_BEARING_HALF_LENGTH_MAX_M}"
                    step="{MAP_BEARING_LENGTH_STEP_M}"
                    value="{MAP_BEARING_HALF_LENGTH_DEFAULT_M}"
                />
                <div id="bearing-length-readout" class="bearing-length-readout">
                    100 m each side / 200 m total
                </div>
            `;
            L.DomEvent.disableClickPropagation(div);
            L.DomEvent.disableScrollPropagation(div);
            return div;
        }};
        bearingLengthControl.addTo(map);

        const bearingLengthSlider = document.getElementById('bearing-length-slider');
        const bearingLengthReadout = document.getElementById('bearing-length-readout');
        const updateBearingLength = () => {{
            const halfLengthM = Number(bearingLengthSlider.value);
            rebuildBearingLines(halfLengthM);
            bearingLengthReadout.textContent = `${{halfLengthM}} m each side / ${{halfLengthM * 2}} m total`;
        }};
        bearingLengthSlider.addEventListener('input', updateBearingLength);
        updateBearingLength();

    map.fitBounds([[{lat_min}, {lon_min}], [{lat_max}, {lon_max}]], {{ padding: [{HTML_MAP_MARGIN_PX}, {HTML_MAP_MARGIN_PX}] }});
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def write_static_site_map(sites: pd.DataFrame, output_path: Path, style: str = "osm") -> None:
    """Write a stitched PNG map using public tile sources."""

    records = _site_map_records(sites)
    if not records:
        raise RuntimeError("Cannot write static map: no sites provided.")

    bounds = _site_bounds(records, pad_fraction=STATIC_MAP_MARGIN_FRACTION)
    zoom = _choose_static_zoom(bounds)
    lon_min, lat_min, lon_max, lat_max = bounds
    provider = _tile_provider(style)

    world_x1, world_y1 = _lonlat_to_world_px(lon_min, lat_max, zoom)
    world_x2, world_y2 = _lonlat_to_world_px(lon_max, lat_min, zoom)
    tile_x_min = math.floor(min(world_x1, world_x2) / MAP_TILE_SIZE)
    tile_x_max = math.floor(max(world_x1, world_x2) / MAP_TILE_SIZE)
    tile_y_min = math.floor(min(world_y1, world_y2) / MAP_TILE_SIZE)
    tile_y_max = math.floor(max(world_y1, world_y2) / MAP_TILE_SIZE)

    canvas_width = (tile_x_max - tile_x_min + 1) * MAP_TILE_SIZE
    canvas_height = (tile_y_max - tile_y_min + 1) * MAP_TILE_SIZE
    canvas = Image.new("RGB", (canvas_width, canvas_height), (243, 241, 234))

    for tile_x in range(tile_x_min, tile_x_max + 1):
        for tile_y in range(tile_y_min, tile_y_max + 1):
            tile_url = provider["url"].format(z=zoom, x=tile_x, y=tile_y)
            try:
                tile_image = _download_tile(tile_url)
            except (URLError, OSError, TimeoutError):
                continue

            canvas.paste(tile_image, ((tile_x - tile_x_min) * MAP_TILE_SIZE, (tile_y - tile_y_min) * MAP_TILE_SIZE))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    canvas_origin_x = tile_x_min * MAP_TILE_SIZE
    canvas_origin_y = tile_y_min * MAP_TILE_SIZE

    for record in records:
        px, py = _world_px_to_canvas_px(float(record["lon"]), float(record["lat"]), zoom, canvas_origin_x, canvas_origin_y)
        x = round(px)
        y = round(py)
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(220, 68, 62), outline=(255, 255, 255), width=3)

        text = f"{record['label']} ({record['n_detections']})"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        pad = 4
        tx = min(max(10, x + 12), canvas.width - text_w - 2 * pad - 10)
        ty = min(max(10, y - text_h - 2 * pad - 12), canvas.height - text_h - 2 * pad - 10)
        draw.rounded_rectangle([tx, ty, tx + text_w + 2 * pad, ty + text_h + 2 * pad], radius=4, fill=(25, 25, 25))
        draw.text((tx + pad, ty + pad), text, fill=(255, 255, 255), font=font)

    caption = f"{provider['name']} backdrop | zoom {zoom} | {len(records)} recorder locations"
    caption_bbox = draw.textbbox((0, 0), caption, font=font)
    caption_w = caption_bbox[2] - caption_bbox[0]
    caption_h = caption_bbox[3] - caption_bbox[1]
    draw.rounded_rectangle(
        [10, canvas.height - caption_h - 18, 20 + caption_w + 8, canvas.height - 8],
        radius=4,
        fill=(25, 25, 25),
    )
    draw.text((14, canvas.height - caption_h - 14), caption, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def extract_recorder_number(recorder_folder: str) -> int:
    """Extract recorder number from values like 'Recorder 3 - 1004 to 1704'."""

    if not isinstance(recorder_folder, str):
        raise TypeError(f"Invalid recorder_folder value: {recorder_folder!r}")

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
    print(f"Calls retained after TDOA filtering: {len(combined)}")

    # Use recorder number as the canonical site identifier.
    combined["site_id"] = combined["recorder_folder"].map(extract_recorder_number)

    # Parse coordinates once into decimal degrees.
    combined["lon_deg"] = combined["LON"].map(parse_coordinate)
    combined["lat_deg"] = combined["LAT"].map(parse_coordinate)

    # Build a single datetime field from the separate DATE and TIME columns,
    # then derive the event timestamp from the within-file peak time.
    starts = [
        parse_recording_start(d, t)
        for d, t in zip(combined["DATE"], combined["TIME"], strict=False)
    ]
    combined["recording_datetime"] = starts
    combined["recording_start"] = combined["recording_datetime"]
    combined["event_time"] = [
        start + timedelta(seconds=float(offset_s))
        for start, offset_s in zip(combined["recording_datetime"], combined["peak_time_s"], strict=False)
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
                "n_detections": len(block),
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

    # Print a readable per-site summary for quick field sanity checks.
    print("Recorder-site median coordinates and call counts:")
    for row in sites.itertuples(index=False):
        print(
            f"  Recorder {int(row.site_id)}: "
            f"lon={row.canonical_lon_deg:.8f}, "
            f"lat={row.canonical_lat_deg:.8f}, "
            f"calls={int(row.n_detections)}"
        )

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


def debug_event_candidates(df: pd.DataFrame, window_seconds: float, max_events: int = 10) -> None:
    """Print a compact diagnosis of provisional candidate clusters across sites."""

    if df.empty:
        print("Debug event candidates: no rows to inspect.")
        return

    ordered = df.sort_values("corrected_event_time").copy().reset_index(drop=True)
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
    deduped = (
        ordered.sort_values("abs_tdoa_ms", ascending=False)
        .drop_duplicates(subset=["event_id", "site_id"], keep="first")
        .sort_values(["event_id", "corrected_event_time"])
    )

    site_counts = deduped.groupby("event_id")["site_id"].nunique()
    by_site_count = site_counts.sort_values(ascending=False)
    if by_site_count.empty:
        print(f"Debug event candidates: no provisional clusters for window {window_seconds}s.")
        return

    print(f"Debug event candidates (window={window_seconds}s):")
    for event_id, n_sites in by_site_count.head(max_events).items():
        block = deduped[deduped["event_id"] == event_id].copy()
        site_list = sorted(block["site_id"].unique().tolist())
        print(
            f"  event {int(event_id)}: n_sites={len(site_list)}, sites={site_list}, "
            f"time_range=({block['corrected_event_time'].min()} -> {block['corrected_event_time'].max()})"
        )


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


def pairwise_time_alignment_matrix(
    df: pd.DataFrame,
    max_delta_s: float = 5.0,
) -> pd.DataFrame:
    """Count nearest-neighbor timestamps shared between recorder pairs."""

    if df.empty:
        return pd.DataFrame(columns=["site_a", "site_b", "n_matches", "median_delta_s", "max_delta_s"])

    site_ids = sorted(df["site_id"].unique().tolist())
    rows = []

    for site_a, site_b in itertools.combinations(site_ids, 2):
        a_times = df.loc[df["site_id"] == site_a, "corrected_event_time"].sort_values().to_numpy()
        b_times = df.loc[df["site_id"] == site_b, "corrected_event_time"].sort_values().to_numpy()
        if len(a_times) == 0 or len(b_times) == 0:
            continue

        a_ns = np.array([pd.Timestamp(ts).value for ts in a_times], dtype=np.int64)
        b_ns = np.array([pd.Timestamp(ts).value for ts in b_times], dtype=np.int64)

        deltas = []
        for a_ns_val in a_ns:
            idx = np.searchsorted(b_ns, a_ns_val)
            candidates = []
            if idx > 0:
                candidates.append(b_ns[idx - 1])
            if idx < len(b_ns):
                candidates.append(b_ns[idx])
            if not candidates:
                continue
            nearest = min(candidates, key=lambda v: abs(int(v) - int(a_ns_val)))
            delta_s = abs(float(nearest - a_ns_val) / 1e9)
            if delta_s <= max_delta_s:
                deltas.append(delta_s)

        if not deltas:
            continue

        rows.append(
            {
                "site_a": int(site_a),
                "site_b": int(site_b),
                "n_matches": int(len(deltas)),
                "median_delta_s": float(np.median(deltas)),
                "max_delta_s": float(np.max(deltas)),
                "min_delta_s": float(np.min(deltas)),
            }
        )

    return pd.DataFrame(rows).sort_values(["n_matches", "median_delta_s"], ascending=[False, True]).reset_index(drop=True)


def build_event_matching_diagnostics(
    raw_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    calibration: OffsetCalibration,
) -> pd.DataFrame:
    """Create diagnostics for timing alignment and event-window sensitivity."""

    rows = []

    print("\nEvent-window sweep diagnostics:")
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
        print(
            f"  window={window_s:.1f}s -> raw={raw_count}, corrected={corrected_count}, "
            f"delta={corrected_count - raw_count}"
        )

    pair_matrix = pairwise_time_alignment_matrix(corrected_df, max_delta_s=5.0)
    print("\nCross-recorder pair alignment matrix (nearest-neighbor matches within 5s):")
    if pair_matrix.empty:
        print("  no recorder pairs have any nearest-neighbor matches within 5s")
    else:
        print(pair_matrix.to_string(index=False))

    print("\nCandidate pair timestamps (recorder, datetime, delta):")
    pair_candidates = []
    for site_a, site_b in itertools.combinations(sorted(corrected_df["site_id"].unique()), 2):
        a_times = corrected_df.loc[corrected_df["site_id"] == site_a, ["corrected_event_time", "wav_file", "peak_time_s"]].sort_values("corrected_event_time").reset_index(drop=True)
        b_times = corrected_df.loc[corrected_df["site_id"] == site_b, ["corrected_event_time", "wav_file", "peak_time_s"]].sort_values("corrected_event_time").reset_index(drop=True)
        if a_times.empty or b_times.empty:
            continue

        for _, row_a in a_times.iterrows():
            best_delta = None
            best_row_b = None
            for _, row_b in b_times.iterrows():
                delta_s = abs((row_b["corrected_event_time"] - row_a["corrected_event_time"]).total_seconds())
                if best_delta is None or delta_s < best_delta:
                    best_delta = delta_s
                    best_row_b = row_b
            if best_delta is not None and best_delta <= 5.0:
                pair_candidates.append(
                    {
                        "site_a": int(site_a),
                        "site_b": int(site_b),
                        "time_a": row_a["corrected_event_time"],
                        "time_b": best_row_b["corrected_event_time"],
                        "delta_s": float(best_delta),
                        "wav_a": str(row_a["wav_file"]),
                        "wav_b": str(best_row_b["wav_file"]),
                        "peak_a_s": float(row_a["peak_time_s"]),
                        "peak_b_s": float(best_row_b["peak_time_s"]),
                    }
                )

    if not pair_candidates:
        print("  no candidate recorder pairs within 5s")
    else:
        for entry in sorted(pair_candidates, key=lambda x: x["delta_s"])[:20]:
            print(
                f"  recorder {entry['site_a']} vs {entry['site_b']} | "
                f"{entry['site_a']}: {entry['time_a']} ({entry['wav_a']}, peak={entry['peak_a_s']}) | "
                f"{entry['site_b']}: {entry['time_b']} ({entry['wav_b']}, peak={entry['peak_b_s']}) | "
                f"delta={entry['delta_s']:.3f}s"
            )

    matrix = pd.DataFrame(index=sorted(corrected_df["site_id"].unique()), columns=sorted(corrected_df["site_id"].unique()), dtype=int)
    for site_a, site_b in itertools.combinations(sorted(corrected_df["site_id"].unique()), 2):
        matches = pair_matrix[(pair_matrix["site_a"] == site_a) & (pair_matrix["site_b"] == site_b)]
        value = int(matches["n_matches"].iloc[0]) if not matches.empty else 0
        matrix.at[site_a, site_b] = value
        matrix.at[site_b, site_a] = value
    for site_id in matrix.index:
        matrix.at[site_id, site_id] = 0
    print("\nRecorder pair count matrix:")
    print(matrix.fillna(0).astype(int).to_string())

    debug_event_candidates(corrected_df, EVENT_WINDOW_SECONDS)

    if corrected_df.empty:
        return pd.DataFrame(rows)

    # Targeted inspection: find the most promising timestamp overlaps across site IDs.
    candidate_rows = []
    for site_id, block in corrected_df.groupby("site_id"):
        for ts, row in zip(block["corrected_event_time"], block.itertuples(index=False), strict=False):
            candidate_rows.append(
                {
                    "site_id": int(site_id),
                    "event_time": ts,
                    "tdoa_ms": float(getattr(row, "tdoa_ms")),
                }
            )

    if candidate_rows:
        print("\nSample multi-site timing candidates (first 10 by site/time):")
        probe = pd.DataFrame(candidate_rows).sort_values(["event_time", "site_id"]).head(10)
        for row in probe.itertuples(index=False):
            print(
                f"  site={int(row.site_id)} @ {row.event_time} | tdoa_ms={float(row.tdoa_ms):.3f}"
            )

    # The earlier tuple-based probe is not used here; this section keeps the debug
    # output stable while leaving the main workflow unchanged.

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

    # GPS-synced recorders are assumed to be time-aligned; disable offset
    # calibration unless evidence suggests drift between devices.
    calibration = OffsetCalibration(
        reference_site_id=int(sorted(combined_with_sites["site_id"].unique())[0]),
        offsets_s={int(site_id): 0.0 for site_id in sorted(combined_with_sites["site_id"].unique())},
        site_pair_stats=pd.DataFrame(columns=[
            "reference_site_id",
            "other_site_id",
            "n_matches",
            "median_delta_s_other_minus_ref",
            "mad_delta_s",
            "p10_delta_s",
            "p90_delta_s",
            "applied_correction_s",
        ]),
    )
    corrected = combined_with_sites.copy()
    corrected["corrected_event_time"] = corrected["event_time"].copy()
    diagnostics = build_event_matching_diagnostics(combined_with_sites, corrected, calibration)
    event_members = build_events(corrected)
    solutions = solve_all_events(event_members)

    sites.to_csv(root / SITES_OUTPUT, index=False)
    calibration.site_pair_stats.to_csv(root / PAIRWISE_OFFSET_OUTPUT, index=False)
    diagnostics.to_csv(root / EVENT_DIAGNOSTICS_OUTPUT, index=False)
    event_members.to_csv(root / EVENT_MEMBERS_OUTPUT, index=False)
    solutions.to_csv(root / SOLUTIONS_OUTPUT, index=False)
    write_interactive_site_map(sites, combined_with_sites, root / MAP_HTML_OUTPUT)
    write_static_site_map(sites, root / MAP_PNG_OUTPUT)

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
    print(f"Wrote interactive site map to {MAP_HTML_OUTPUT}")
    print(f"Wrote static site map to {MAP_PNG_OUTPUT}")

    if len(solutions):
        n_pass = int(solutions["quality_pass"].sum())
        print(f"Quality-pass solutions: {n_pass}/{len(solutions)}")


if __name__ == "__main__":
    main()

