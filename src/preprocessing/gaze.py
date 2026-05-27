"""Gaze preprocessing: interpolate missing values and normalize pupil diameter."""

import numpy as np
import pandas as pd


def preprocess(signal: np.ndarray) -> np.ndarray:
    """Interpolate NaN / zero-padded values and z-score normalize each channel.

    Args:
        signal: (n_samples, n_channels) gaze data
                Typical channels: [x, y, pupil_left, pupil_right, ...]

    Returns:
        Cleaned signal of same shape as float32.
    """
    squeeze = signal.ndim == 1
    if squeeze:
        signal = signal[:, None]

    signal = signal.astype(np.float32)
    out = np.empty_like(signal)

    for ch in range(signal.shape[1]):
        col = signal[:, ch].copy()
        # treat 0 as missing for pupil-like columns (heuristic)
        col[col == 0] = np.nan
        s = pd.Series(col)
        s = s.interpolate(method="linear", limit_direction="both")
        s = s.fillna(method="bfill").fillna(method="ffill")
        arr = s.to_numpy(dtype=np.float32)
        mu, sigma = arr.mean(), arr.std()
        out[:, ch] = (arr - mu) / (sigma + 1e-8)

    return out[:, 0] if squeeze else out
