"""EDA feature extraction.

Features per 10-second segment from tonic (SCL) and phasic (SCR):
  Tonic: mean, std, min, max, slope
  Phasic: mean, std, min, max, n_peaks, mean_peak_amplitude, mean_rise_time, mean_recovery_time
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def _tonic_features(tonic: np.ndarray) -> dict[str, float]:
    slope = np.polyfit(np.arange(len(tonic)), tonic, 1)[0] if len(tonic) > 1 else 0.0
    return {
        "scl_mean": float(tonic.mean()),
        "scl_std": float(tonic.std()),
        "scl_min": float(tonic.min()),
        "scl_max": float(tonic.max()),
        "scl_slope": float(slope),
    }


def _phasic_features(phasic: np.ndarray, fs: int) -> dict[str, float]:
    feats: dict[str, float] = {
        "scr_mean": float(phasic.mean()),
        "scr_std": float(phasic.std()),
        "scr_min": float(phasic.min()),
        "scr_max": float(phasic.max()),
        "scr_n_peaks": 0.0,
        "scr_mean_amp": 0.0,
        "scr_mean_rise": 0.0,
        "scr_mean_recovery": 0.0,
    }

    # SCR onset detection: peaks above 0.01 µS with min distance 1 s
    peaks, props = find_peaks(phasic, height=0.01, distance=fs)
    if len(peaks) == 0:
        return feats

    feats["scr_n_peaks"] = float(len(peaks))
    amps = props["peak_heights"]
    feats["scr_mean_amp"] = float(amps.mean())

    # rise time: samples from previous trough to peak
    rise_times, recovery_times = [], []
    for pk in peaks:
        # look back up to 5 s for trough
        look_back = min(pk, 5 * fs)
        onset = pk - int(np.argmin(phasic[pk - look_back:pk][::-1]))
        rise_times.append((pk - onset) / fs)
        # recovery: samples until signal drops to half amplitude
        half_amp = phasic[pk] * 0.5
        after = phasic[pk:]
        crosses = np.where(after <= half_amp)[0]
        recovery_times.append(crosses[0] / fs if len(crosses) > 0 else np.nan)

    feats["scr_mean_rise"] = float(np.nanmean(rise_times))
    feats["scr_mean_recovery"] = float(np.nanmean(recovery_times))
    return feats


def extract(segment: np.ndarray, fs: int = 128) -> np.ndarray:
    """Extract EDA features from a preprocessed EDA segment.

    Args:
        segment: (n_samples,) or (n_samples, 1) EDA signal (full, undecomposed)
                 Will be decomposed internally.
        fs: sampling frequency

    Returns:
        1-D feature vector.
    """
    from src.preprocessing.eda import preprocess

    raw = segment.ravel()
    decomposed = preprocess(raw, fs)
    feats = {}
    feats.update(_tonic_features(decomposed["tonic"]))
    feats.update(_phasic_features(decomposed["phasic"], fs))
    vec = np.array(list(feats.values()), dtype=np.float32)
    return np.nan_to_num(vec, nan=0.0)


def extract_batch(segments: np.ndarray, fs: int = 128) -> np.ndarray:
    return np.stack([extract(s, fs) for s in segments])
