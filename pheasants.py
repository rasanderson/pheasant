"""Detect pheasants from left and right channels."""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from birdnetlib import RecordingBuffer
from birdnetlib.analyzer import Analyzer
from scipy.fft import fft, ifft
from scipy.signal import butter, filtfilt


BASE_DIR = Path("subset_test")
OUTPUT_FILE = Path("pheasant_results.csv")

# Info files are CSV files with meta data about recordings. Header row is:
# DATE,TIME,LON,LAT,BAT,TEMP,HUMI,FILE,GAIN,SAMPLE RATE,CHANNEL,NOTE


# Calculate bearings from TDOA. Assumes left points West (270 degrees) and
# right points East (90 degrees). Bearing measured clockwise from North. Note
# that if call directly aligned then bearing is ambiguous and call could be
# from either side.
def tdoa_to_bearings(tdoa, mic_spacing=0.15, speed_of_sound=343.0):

    x = (speed_of_sound * tdoa) / mic_spacing
    x = np.clip(x, -1.0, 1.0)

    phi = np.degrees(np.arccos(x))

    # Bearing measured clockwise from North
    bearing1 = (90 - phi) % 360
    bearing2 = (90 + phi) % 360

    return bearing1, bearing2


# Helper function as pheasant calls in lower frequency range.
# Pheasant calls typically 1000 to 5500 kHz, but lower register very characteristic
def bandpass(signal, fs, low=500, high=2000, order=4):

    nyquist = fs / 2

    b, a = butter(
        order,
        [low / nyquist, high / nyquist],
        btype="band"
    )

    return filtfilt(b, a, signal)


# Function to compute the time delay of arrival (TDOA) using GCC-PHAT
# This normalises the cross-correlation to reduce impact of different volume
# levels between the two channels. Interpolation to improve TDOA accuracy.
def gcc_phat(sig, refsig, fs, max_tau=None, interp=16):

    n = len(sig) + len(refsig)

    SIG = fft(sig, n=n)
    REF = fft(refsig, n=n)

    # Cross-power spectrum
    R = SIG * np.conj(REF)

    # PHAT weighting
    R = R / (np.abs(R) + 1e-15)

    # GCC-PHAT correlation with interpolation
    cc = np.real(ifft(R, n=(interp * n)))

    max_shift = int(interp * n / 2)

    if max_tau is not None:
        max_shift = min(
            int(interp * fs * max_tau),
            max_shift
        )

    # Rearrange correlation so zero lag is in centre
    cc = np.concatenate(
        (cc[-max_shift:], cc[:max_shift + 1])
    )

    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(fs * interp)

    return tau, cc


def load_info_rows(info_path):

    with info_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def process_wav_file(wav_path, analyzer, metadata=None):

    metadata = metadata or {}

    try:
        data, samplerate = sf.read(wav_path)
    except Exception as exc:
        print(f"Skipping {wav_path}: unable to read audio ({exc})")
        return []

    if data.ndim != 2 or data.shape[1] < 2:
        print(f"Skipping {wav_path}: expected stereo audio, got shape {data.shape}")
        return []

    left = data[:, 0]
    right = data[:, 1]

    recording_left = RecordingBuffer(
        analyzer,
        left,
        samplerate,
        min_conf=0.25,
        overlap=0.0,  # This is the default. If too many calls near 3 s boundary, increase this value 2.0 (1.0 second step)
    )
    recording_right = RecordingBuffer(
        analyzer,
        right,
        samplerate,
        min_conf=0.25,
        overlap=0.0,  # This is the default. If too many calls near 3 s boundary, increase this value 2.0 (1.0 second step)
    )
    recording_left.analyze()
    recording_right.analyze()

    # Basic parameters for TDOA calculation
    speed_of_sound = 343.0  # m/s
    distance_between_mics = 0.15  # m
    max_tdoa = distance_between_mics / speed_of_sound  # seconds

    # If call only detected in one channel it is usually just audible in other.
    # Create a dictionary so that both channels can be used.
    all_windows = {}
    for det in recording_left.detections:
        key = (det["start_time"], det["end_time"])
        all_windows.setdefault(key, set()).add("left")
    for det in recording_right.detections:
        key = (det["start_time"], det["end_time"])
        all_windows.setdefault(key, set()).add("right")

    results = []

    for (start_time, end_time), detected_by in sorted(all_windows.items()):

        # Extract BirdNET's 3-second window from BOTH channels
        s0 = int(start_time * samplerate)
        s1 = int(end_time * samplerate)

        left_win = data[s0:s1, 0]
        right_win = data[s0:s1, 1]

        # Band-pass filter
        left_filt = bandpass(left_win, samplerate)
        right_filt = bandpass(right_win, samplerate)

        # Find centre of the vocalization by energy peak
        window = int(0.02 * samplerate)  # 20 ms
        energy = np.convolve(
            (left_filt**2 + right_filt**2),
            np.ones(window),
            mode="same"
        )
        peak_idx = np.argmax(energy)
        peak_time = start_time + peak_idx / samplerate

        # Extract ±0.25 s around peak; pheasant calls typically 0.5 s long
        half_window = int(0.25 * samplerate)

        a = max(0, peak_idx - half_window)
        b = min(len(left_filt), peak_idx + half_window)
        call_start = start_time + a / samplerate
        call_end = start_time + b / samplerate

        left_call = left_filt[a:b]
        right_call = right_filt[a:b]

        # Remove DC offset
        left_call = left_call - np.mean(left_call)
        right_call = right_call - np.mean(right_call)

        # GCC-PHAT to find TDOA
        tau, cc = gcc_phat(
            left_call,
            right_call,
            samplerate,
            max_tau=max_tdoa,
            interp=16  # Interpolation factor to improve TDOA accuracy
        )

        tdoa_ms = tau * 1000
        best_lag = tau * samplerate

        # Calculate bearings from TDOA
        bearing1, bearing2 = tdoa_to_bearings(tau, distance_between_mics, speed_of_sound)

        print(
            f"{wav_path.name}  "
            f"BirdNET={start_time:.0f}-{end_time:.0f}s  "
            f"Peak={peak_time:.3f}s  "
            f"GCC-PHAT={call_start:.3f}-{call_end:.3f}s  "
            f"TDOA={tdoa_ms:.3f} ms  "
            f"lag={best_lag:.3f}  "
            f"Bearings={bearing1:.1f}° or {bearing2:.1f}°"
        )

        result = dict(metadata)
        result.update({
            "recorder_folder": wav_path.parent.name,
            "wav_file": wav_path.name,
            "birdnet_start_s": start_time,
            "birdnet_end_s": end_time,
            "peak_time_s": peak_time,
            "gcc_start_s": call_start,
            "gcc_end_s": call_end,
            "tdoa_ms": tdoa_ms,
            "lag_samples": best_lag,
            "bearing_1_deg": bearing1,
            "bearing_2_deg": bearing2,
            "detected_by": ",".join(sorted(detected_by)),
        })
        results.append(result)

    return results


def process_recorder_folder(folder_path, analyzer):

    info_path = folder_path / "info.txt"
    if not info_path.exists():
        print(f"Skipping {folder_path}: missing info.txt")
        return []

    rows = load_info_rows(info_path)
    results = []

    print(f"Processing {folder_path.name}: {len(rows)} metadata rows")

    for row in rows:
        wav_name = (row.get("FILE") or "").strip()
        if not wav_name:
            print(f"Skipping row in {info_path}: missing FILE value")
            continue

        wav_path = folder_path / wav_name
        if not wav_path.exists():
            print(f"Skipping {wav_path}: file not found")
            continue

        results.extend(process_wav_file(wav_path, analyzer, row))

    return results


def write_results(results, output_path):

    if not results:
        print("No detections found; nothing to write.")
        return

    preferred_columns = [
        "recorder_folder",
        "DATE",
        "TIME",
        "LON",
        "LAT",
        "BAT",
        "TEMP",
        "HUMI",
        "FILE",
        "GAIN",
        "SAMPLE RATE",
        "CHANNEL",
        "NOTE",
        "wav_file",
        "birdnet_start_s",
        "birdnet_end_s",
        "peak_time_s",
        "gcc_start_s",
        "gcc_end_s",
        "tdoa_ms",
        "lag_samples",
        "bearing_1_deg",
        "bearing_2_deg",
        "detected_by",
    ]

    fieldnames = []
    for column in preferred_columns:
        if any(column in row for row in results):
            fieldnames.append(column)

    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} detections to {output_path}")


def main():

    custom_species = "species_list.txt"  # Path to the single species list file
    analyzer = Analyzer(custom_species_list_path=custom_species, version="2.4")

    recorder_dirs = sorted(
        folder for folder in BASE_DIR.glob("Recorder * - 1004 to 1704")
        if folder.is_dir()
    )

    if not recorder_dirs:
        print(f"No recorder folders found under {BASE_DIR}")
        return

    all_results = []
    for folder_path in recorder_dirs:
        all_results.extend(process_recorder_folder(folder_path, analyzer))

    write_results(all_results, OUTPUT_FILE)


if __name__ == "__main__":
    main()