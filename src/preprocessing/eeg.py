"""EEG preprocessing: Butterworth lowpass + 60 Hz notch filter.

As described in the paper:
  - Butterworth lowpass 0.4–128 Hz (high-pass 0.4 Hz + lowpass 128 Hz)
  - 60 Hz notch filter for power line noise
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, sosfilt


def preprocess(signal: np.ndarray, fs: int = 256) -> np.ndarray:
    """Apply bandpass (0.4-128 Hz) and 60 Hz notch filter.

    Args:
        signal: (n_samples, n_channels) EEG signal (AF7, AF8, TP9, TP10)
        fs: sampling frequency in Hz

    Returns:
        Filtered signal (n_samples, n_channels) as float32.
    """
    squeeze = signal.ndim == 1
    if squeeze:
        signal = signal[:, None]

    signal = signal.astype(np.float32)
    nyq = fs / 2

    # Bandpass 0.4 - min(128, nyq-1) Hz
    highcut = min(128.0, nyq - 1)
    bp_sos = butter(4, [0.4 / nyq, highcut / nyq], btype="band", output="sos")

    # 60 Hz notch
    b_notch, a_notch = iirnotch(60.0 / nyq, Q=30.0)
    notch_sos = np.array([[b_notch[0], b_notch[1], b_notch[2],
                           1.0, a_notch[1], a_notch[2]]])

    out = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        x = sosfiltfilt(bp_sos, signal[:, ch])
        x = sosfiltfilt(notch_sos, x)
        out[:, ch] = x

    return out[:, 0] if squeeze else out
