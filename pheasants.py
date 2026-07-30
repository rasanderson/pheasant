# Detect pheasants from left and right channels

from birdnetlib import Recording
from birdnetlib import RecordingBuffer
from birdnetlib.analyzer import Analyzer
from datetime import datetime
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt, correlate
from scipy.fft import fft, ifft
# import matplotlib.pyplot as plt # import for optional plotting of GCC-PHAT results


# Folders Recorder 1 - 1004 to 1704 to Recorder 5 - 1004 to 1704 wav files
wav_file = "subset_test/Recorder 1 - 1004 to 1704/1_20260410_184202.wav"
# Info files are CSV files with meta data about recordings. Header row is:
# DATE,TIME,LON,LAT,BAT,TEMP,HUMI,FILE,GAIN,SAMPLE RATE,CHANNEL,NOTE
info_file = "subset_test/Recorder 1 - 1004 to 1704/info.txt"

# Calculate bearings from toda. Assumes left points West (270 degrees) and
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

# Load and initialize the BirdNET-Analyzer models.
custom_species = "species_list.txt"  # Path to the single species list file
analyzer = Analyzer(custom_species_list_path = custom_species, version="2.4")

# Split left and right channels into separate files
data, samplerate = sf.read(wav_file)
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
max_lag = int(max_tdoa * samplerate)  # samples

# If call only detected in one channel it is usually just audible in other.
# Create a dictionary so that both channels can be used.
all_windows = {}
for det in recording_left.detections:
    key = (det["start_time"], det["end_time"])
    all_windows.setdefault(key, set()).add("left")
for det in recording_right.detections:
    key = (det["start_time"], det["end_time"])
    all_windows.setdefault(key, set()).add("right")

print("Warning: with only 15 cm at 48 kHz 1 sample = 0.021 ms")
print("0.437 ms is the maximum TDOA for 15 cm distance at 343 m/s")
print("consider longer gap between microphones or interpolation (upsampling) to improve TDOA accuracy")
# Now go through each detection and calculate TDOA from both channels
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
 
    lags = np.arange(len(cc))
    lags = (lags - len(cc)//2) / 16

    # Optional: Plot the GCC-PHAT result for visual inspection
    # Plots with sharp peak indicate good TDOA estimation but this does not
    # guarantee that the call is from a pheasant. It could be a different species
    # plt.plot(lags, np.abs(cc))
    # plt.xlabel("Lag (samples)")
    # plt.ylabel("GCC-PHAT")
    # plt.title(f"{start_time:.0f}-{end_time:.0f}s")
    # plt.grid(True)
    # plt.show()

    tdoa_ms = tau * 1000
    best_lag = tau * samplerate

    # Calculate bearings from TDOA
    bearing1, bearing2 = tdoa_to_bearings(tau, distance_between_mics, speed_of_sound)

    print(
        f"BirdNET={start_time:.0f}-{end_time:.0f}s  "
        f"Peak={peak_time:.3f}s  "
        f"GCC-PHAT={call_start:.3f}-{call_end:.3f}s  "
        # f"Detected by: {detected_by}  "
        f"TDOA = {tdoa_ms:.3f} ms  "
        f"lag = {best_lag} "
        f"Bearings = {bearing1:.1f}° or {bearing2:.1f}°"
    )