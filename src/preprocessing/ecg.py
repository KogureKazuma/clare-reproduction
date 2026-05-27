"""ECG preprocessing: Butterworth bandpass 5-15 Hz + z-score normalization."""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _butter_bandpass(lowcut: float, highcut: float, fs: int, order: int = 4):
    nyq = fs / 2
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sos


def preprocess(signal: np.ndarray, fs: int = 512) -> np.ndarray:
    """Apply bandpass filter (5-15 Hz) and z-score normalize per channel.

    Args:
        signal: (n_samples, n_channels) or (n_samples,)
        fs: sampling frequency in Hz

    Returns:
        Filtered and normalized signal of same shape.
    """
    squeeze = signal.ndim == 1
    if squeeze:
        signal = signal[:, None]

    sos = _butter_bandpass(5.0, 15.0, fs)
    out = np.zeros_like(signal, dtype=np.float32)
    for ch in range(signal.shape[1]):
        filtered = sosfiltfilt(sos, signal[:, ch])
        mu, sigma = filtered.mean(), filtered.std()
        out[:, ch] = (filtered - mu) / (sigma + 1e-8)

    return out[:, 0] if squeeze else out
