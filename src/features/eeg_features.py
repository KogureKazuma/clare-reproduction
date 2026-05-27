"""EEG feature extraction.

Per channel features:
  - Spectral entropy
  - Hjorth parameters: activity, mobility, complexity
  - Band power: Delta (0.5-4), Theta (4-8), Alpha (8-13), Beta (13-30), Gamma (30-64) Hz
  - Statistical: mean, std
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 64.0),
}


def _spectral_entropy(psd: np.ndarray) -> float:
    psd_norm = psd / (psd.sum() + 1e-12)
    return float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))


def _hjorth(x: np.ndarray) -> tuple[float, float, float]:
    activity = float(np.var(x))
    d1 = np.diff(x)
    mobility = float(np.sqrt(np.var(d1) / (np.var(x) + 1e-12)))
    d2 = np.diff(d1)
    mob_d1 = np.sqrt(np.var(d2) / (np.var(d1) + 1e-12))
    complexity = float(mob_d1 / (mobility + 1e-12))
    return activity, mobility, complexity


def _band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _channel_features(x: np.ndarray, fs: int, ch_name: str) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats[f"{ch_name}_mean"] = float(x.mean())
    feats[f"{ch_name}_std"] = float(x.std())

    act, mob, comp = _hjorth(x)
    feats[f"{ch_name}_hjorth_act"] = act
    feats[f"{ch_name}_hjorth_mob"] = mob
    feats[f"{ch_name}_hjorth_comp"] = comp

    nperseg = min(len(x), 256)
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)
    feats[f"{ch_name}_spectral_entropy"] = _spectral_entropy(psd)

    for band_name, (lo, hi) in BANDS.items():
        feats[f"{ch_name}_{band_name}"] = _band_power(freqs, psd, lo, hi)

    return feats


def extract(segment: np.ndarray, fs: int = 256, ch_names: list[str] | None = None) -> np.ndarray:
    """Extract EEG features from a single preprocessed segment.

    Args:
        segment: (n_samples, n_channels) — channels: AF7, AF8, TP9, TP10
        fs: sampling frequency

    Returns:
        1-D feature vector.
    """
    if segment.ndim == 1:
        segment = segment[:, None]

    n_ch = segment.shape[1]
    if ch_names is None:
        ch_names = [f"ch{i}" for i in range(n_ch)]

    feats: dict[str, float] = {}
    for i, name in enumerate(ch_names):
        feats.update(_channel_features(segment[:, i], fs, name))

    vec = np.array(list(feats.values()), dtype=np.float32)
    return np.nan_to_num(vec, nan=0.0)


def extract_batch(segments: np.ndarray, fs: int = 256) -> np.ndarray:
    return np.stack([extract(s, fs) for s in segments])
