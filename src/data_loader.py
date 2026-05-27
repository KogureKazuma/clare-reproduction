"""
CLARE dataset loader.

Expected directory layout under data_root:
    data_root/
    ├── ECG/   P01/ session1.csv ... session4.csv
    ├── EDA/   P01/ session1.csv ...
    ├── EEG/   P01/ session1.csv ...
    ├── Gaze/  P01/ session1.csv ...
    └── Labels/ P01/ session1.csv ...

Each signal CSV has a timestamp column followed by channel columns.
Each label CSV has columns: [timestamp, label] where label is 1-9.

Binary classification: label >= 5 → 1 (high), label < 5 → 0 (low).
Segments are 10 seconds long.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Sampling rates (Hz)
FS = {"ECG": 512, "EDA": 128, "EEG": 256, "Gaze": 50}

SEGMENT_SEC = 10
LABEL_THRESHOLD = 5  # >= threshold → high cognitive load (class 1)

MODALITIES = ["ECG", "EDA", "EEG", "Gaze"]


def _list_participants(data_root: Path) -> list[str]:
    ecg_dir = data_root / "ECG"
    return sorted(p.name for p in ecg_dir.iterdir() if p.is_dir())


def _list_sessions(participant_dir: Path) -> list[Path]:
    files = sorted(participant_dir.glob("*.csv"))
    # exclude baseline files
    return [f for f in files if "baseline" not in f.stem.lower()]


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def _segment_signal(signal: np.ndarray, fs: int, segment_sec: int = SEGMENT_SEC) -> np.ndarray:
    """Split signal into non-overlapping segments of segment_sec seconds.

    Returns array of shape (n_segments, segment_len, n_channels).
    """
    segment_len = fs * segment_sec
    n_channels = signal.shape[1] if signal.ndim > 1 else 1
    if signal.ndim == 1:
        signal = signal[:, None]

    n_full = len(signal) // segment_len
    signal = signal[: n_full * segment_len]
    segments = signal.reshape(n_full, segment_len, n_channels)
    return segments


def _load_modality_session(path: Path, modality: str) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    df = _load_csv(path)
    # drop timestamp/index columns (assume first column is time if dtype is float/int)
    ts_candidates = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
    if ts_candidates:
        df = df.drop(columns=ts_candidates)
    return df.values.astype(np.float32)


def _load_labels_session(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    df = _load_csv(path)
    # find the label column: last numeric column or named 'label'/'score'
    label_candidates = [c for c in df.columns if "label" in c.lower() or "score" in c.lower() or "load" in c.lower()]
    if label_candidates:
        labels = df[label_candidates[0]].values
    else:
        labels = df.iloc[:, -1].values
    return labels.astype(np.float32)


class CLAREDataset:
    """Loads and segments CLARE data for a given set of modalities.

    After calling load(), exposes:
        self.X       dict modality → (N, seg_len, n_ch) raw segments
        self.y       (N,) binary labels
        self.subject_ids  (N,) integer subject index per segment
    """

    def __init__(
        self,
        data_root: str | Path,
        modalities: list[str] = MODALITIES,
    ):
        self.data_root = Path(data_root)
        self.modalities = modalities

    def load(self) -> "CLAREDataset":
        participants = _list_participants(self.data_root)

        all_X: dict[str, list[np.ndarray]] = {m: [] for m in self.modalities}
        all_y: list[np.ndarray] = []
        all_subjects: list[np.ndarray] = []

        for subj_idx, pid in enumerate(participants):
            label_dir = self.data_root / "Labels" / pid
            label_sessions = sorted(label_dir.glob("*.csv")) if label_dir.exists() else []

            # match sessions across modalities by index
            mod_sessions: dict[str, list[Path]] = {}
            for mod in self.modalities:
                mod_dir = self.data_root / mod / pid
                mod_sessions[mod] = _list_sessions(mod_dir) if mod_dir.exists() else []

            n_sessions = min(
                len(label_sessions),
                *(len(v) for v in mod_sessions.values()),
            )

            for sess_idx in range(n_sessions):
                label_path = label_sessions[sess_idx]
                labels_raw = _load_labels_session(label_path)
                if labels_raw is None:
                    continue

                # segment each modality
                seg_counts = []
                mod_segments: dict[str, np.ndarray] = {}
                for mod in self.modalities:
                    if sess_idx >= len(mod_sessions[mod]):
                        break
                    sig = _load_modality_session(mod_sessions[mod][sess_idx], mod)
                    if sig is None:
                        break
                    segs = _segment_signal(sig, FS[mod])
                    mod_segments[mod] = segs
                    seg_counts.append(len(segs))
                else:
                    # only reached if no break
                    n_segs = min(seg_counts + [len(labels_raw)])
                    for mod in self.modalities:
                        all_X[mod].append(mod_segments[mod][:n_segs])
                    # binarize labels: one label per 10-sec segment
                    y_bin = (labels_raw[:n_segs] >= LABEL_THRESHOLD).astype(np.int64)
                    all_y.append(y_bin)
                    all_subjects.append(np.full(n_segs, subj_idx, dtype=np.int64))

        self.X = {m: np.concatenate(all_X[m], axis=0) for m in self.modalities}
        self.y = np.concatenate(all_y, axis=0)
        self.subject_ids = np.concatenate(all_subjects, axis=0)
        return self

    @property
    def n_subjects(self) -> int:
        return int(self.subject_ids.max()) + 1

    def __len__(self) -> int:
        return len(self.y)
