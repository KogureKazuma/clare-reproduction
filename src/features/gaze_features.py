"""Gaze feature extraction.

Features per 10-second segment:
  Pupil diameter: mean, std, min, max, slope (per eye if available)
  Blinks: count, mean duration
  Fixations: count, mean duration (via velocity threshold)
  Saccades: count, mean amplitude, mean peak velocity
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def _pupil_features(pupil: np.ndarray, fs: int) -> dict[str, float]:
    slope = np.polyfit(np.arange(len(pupil)), pupil, 1)[0] if len(pupil) > 1 else 0.0
    return {
        "pupil_mean": float(pupil.mean()),
        "pupil_std": float(pupil.std()),
        "pupil_min": float(pupil.min()),
        "pupil_max": float(pupil.max()),
        "pupil_slope": float(slope),
    }


def _blink_features(pupil: np.ndarray, fs: int) -> dict[str, float]:
    """Detect blinks as periods where pupil is near zero (or NaN before preprocessing)."""
    is_blink = pupil < (pupil.mean() - 2 * pupil.std())
    blink_runs = []
    in_blink = False
    start = 0
    for i, b in enumerate(is_blink):
        if b and not in_blink:
            in_blink = True
            start = i
        elif not b and in_blink:
            in_blink = False
            blink_runs.append(i - start)
    if in_blink:
        blink_runs.append(len(is_blink) - start)

    durations = [r / fs for r in blink_runs if r > 0]
    return {
        "blink_count": float(len(durations)),
        "blink_mean_dur": float(np.mean(durations)) if durations else 0.0,
    }


def _velocity_events(x: np.ndarray, y: np.ndarray, fs: int) -> dict[str, float]:
    """Classify gaze samples as fixation or saccade via velocity threshold."""
    if len(x) < 2:
        return {
            "fix_count": 0.0, "fix_mean_dur": 0.0,
            "sac_count": 0.0, "sac_mean_amp": 0.0, "sac_mean_vel": 0.0,
        }

    dx = np.diff(x)
    dy = np.diff(y)
    vel = np.sqrt(dx ** 2 + dy ** 2) * fs  # deg/s (assumes x,y in degrees)

    # I-VT threshold: 100 deg/s (common default)
    threshold = 100.0
    is_sac = vel > threshold

    def _runs(arr: np.ndarray):
        runs = []
        in_run, start = False, 0
        for i, v in enumerate(arr):
            if v and not in_run:
                in_run, start = True, i
            elif not v and in_run:
                in_run = False
                runs.append((start, i))
        if in_run:
            runs.append((start, len(arr)))
        return runs

    fix_runs = _runs(~is_sac)
    sac_runs = _runs(is_sac)

    fix_durs = [(e - s) / fs for s, e in fix_runs]
    sac_amps, sac_vels = [], []
    for s, e in sac_runs:
        amp = np.sqrt((x[e] - x[s]) ** 2 + (y[e] - y[s]) ** 2) if e < len(x) else 0.0
        sac_amps.append(float(amp))
        sac_vels.append(float(vel[s:e].mean()) if e > s else 0.0)

    return {
        "fix_count": float(len(fix_durs)),
        "fix_mean_dur": float(np.mean(fix_durs)) if fix_durs else 0.0,
        "sac_count": float(len(sac_amps)),
        "sac_mean_amp": float(np.mean(sac_amps)) if sac_amps else 0.0,
        "sac_mean_vel": float(np.mean(sac_vels)) if sac_vels else 0.0,
    }


def extract(segment: np.ndarray, fs: int = 50) -> np.ndarray:
    """Extract gaze features from a preprocessed gaze segment.

    Expected column layout (flexible): [gaze_x, gaze_y, pupil_left, pupil_right, ...]
    Fallback: treat all columns as pupil-like.

    Args:
        segment: (n_samples, n_channels)
        fs: sampling frequency

    Returns:
        1-D feature vector.
    """
    if segment.ndim == 1:
        segment = segment[:, None]

    n_ch = segment.shape[1]
    feats: dict[str, float] = {}

    if n_ch >= 3:
        x = segment[:, 0]
        y = segment[:, 1]
        pupil = segment[:, 2]
        feats.update(_velocity_events(x, y, fs))
    elif n_ch == 2:
        x, y = segment[:, 0], segment[:, 1]
        pupil = np.sqrt(x ** 2 + y ** 2)
        feats.update(_velocity_events(x, y, fs))
    else:
        pupil = segment[:, 0]

    feats.update(_pupil_features(pupil, fs))
    feats.update(_blink_features(pupil, fs))

    vec = np.array(list(feats.values()), dtype=np.float32)
    return np.nan_to_num(vec, nan=0.0)


def extract_batch(segments: np.ndarray, fs: int = 50) -> np.ndarray:
    return np.stack([extract(s, fs) for s in segments])
