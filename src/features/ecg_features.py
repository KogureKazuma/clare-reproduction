"""ECG feature extraction.

Features per 10-second segment:
  Time-domain HRV: mean HR, SDNN, RMSSD, pNN50
  Frequency-domain HRV: ULF (<0.003 Hz), VLF (0.003-0.04), LF (0.04-0.15), HF (0.15-0.40)
  Statistical: mean, std, min, max per channel
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch, find_peaks


def _rr_intervals(signal: np.ndarray, fs: int) -> np.ndarray:
    """Detect R-peaks and return RR intervals in seconds (uses single channel)."""
    # use first channel if multi-channel
    x = signal[:, 0] if signal.ndim > 1 else signal
    # min_distance: 0.3 s between peaks (200 bpm max)
    peaks, _ = find_peaks(x, distance=int(0.3 * fs), height=0)
    if len(peaks) < 2:
        return np.array([])
    rr = np.diff(peaks) / fs
    # keep physiological range 0.3–2.0 s
    rr = rr[(rr > 0.3) & (rr < 2.0)]
    return rr


def _hrv_time_domain(rr: np.ndarray) -> dict[str, float]:
    if len(rr) < 2:
        return {"hr_mean": np.nan, "sdnn": np.nan, "rmssd": np.nan, "pnn50": np.nan}
    hr = 60.0 / rr
    diff_rr = np.diff(rr)
    pnn50 = 100.0 * np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr)
    return {
        "hr_mean": hr.mean(),
        "sdnn": rr.std(),
        "rmssd": np.sqrt(np.mean(diff_rr ** 2)),
        "pnn50": pnn50,
    }


def _hrv_freq_domain(rr: np.ndarray) -> dict[str, float]:
    bands = {
        "ulf": (0.0, 0.003),
        "vlf": (0.003, 0.04),
        "lf": (0.04, 0.15),
        "hf": (0.15, 0.40),
    }
    if len(rr) < 4:
        return {f"hrv_{k}": np.nan for k in bands}

    # resample RR to evenly spaced at 4 Hz
    fs_resamp = 4.0
    t_rr = np.cumsum(rr) - rr[0]
    t_uniform = np.arange(0, t_rr[-1], 1 / fs_resamp)
    if len(t_uniform) < 8:
        return {f"hrv_{k}": np.nan for k in bands}
    rr_interp = np.interp(t_uniform, t_rr, rr)

    freqs, psd = welch(rr_interp, fs=fs_resamp, nperseg=min(len(rr_interp), 256))
    result = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        result[f"hrv_{name}"] = float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0
    return result


def _channel_stats(signal: np.ndarray) -> dict[str, float]:
    feats = {}
    n_ch = signal.shape[1] if signal.ndim > 1 else 1
    x = signal if signal.ndim > 1 else signal[:, None]
    for ch in range(n_ch):
        col = x[:, ch]
        feats[f"ch{ch}_mean"] = float(col.mean())
        feats[f"ch{ch}_std"] = float(col.std())
        feats[f"ch{ch}_min"] = float(col.min())
        feats[f"ch{ch}_max"] = float(col.max())
    return feats


def extract(segment: np.ndarray, fs: int = 512) -> np.ndarray:
    """Extract ECG features from a single segment.

    Args:
        segment: (n_samples, n_channels) preprocessed ECG segment
        fs: sampling frequency

    Returns:
        1-D feature vector.
    """
    rr = _rr_intervals(segment, fs)
    feats = {}
    feats.update(_hrv_time_domain(rr))
    feats.update(_hrv_freq_domain(rr))
    feats.update(_channel_stats(segment))
    vec = np.array(list(feats.values()), dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0)
    return vec


def extract_batch(segments: np.ndarray, fs: int = 512) -> np.ndarray:
    """Extract features for a batch of segments.

    Args:
        segments: (N, n_samples, n_channels)

    Returns:
        (N, n_features) feature matrix.
    """
    return np.stack([extract(s, fs) for s in segments])
