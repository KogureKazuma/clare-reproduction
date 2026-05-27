#!/usr/bin/env python3
"""Train and evaluate classical ML models on the CLARE dataset.

Usage:
    python train_classical.py --data_root ./data --modalities ECG EDA EEG Gaze
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.data_loader import CLAREDataset, FS
from src.preprocessing import ecg as ecg_pre, eda as eda_pre, eeg as eeg_pre, gaze as gaze_pre
from src.features import ecg_features, eda_features, eeg_features, gaze_features
from src.models.classical_ml import build_models
from src.evaluation.metrics import kfold_evaluate, loso_evaluate, print_results_table

PREPROCESSORS = {
    "ECG": lambda s: ecg_pre.preprocess(s, FS["ECG"]),
    "EDA": lambda s: s,  # EDA decomposition happens inside feature extractor
    "EEG": lambda s: eeg_pre.preprocess(s, FS["EEG"]),
    "Gaze": lambda s: gaze_pre.preprocess(s),
}

FEATURE_EXTRACTORS = {
    "ECG": lambda segs: ecg_features.extract_batch(segs, FS["ECG"]),
    "EDA": lambda segs: eda_features.extract_batch(segs, FS["EDA"]),
    "EEG": lambda segs: eeg_features.extract_batch(segs, FS["EEG"]),
    "Gaze": lambda segs: gaze_features.extract_batch(segs, FS["Gaze"]),
}


def build_feature_matrix(dataset: CLAREDataset, modalities: list[str]) -> np.ndarray:
    feature_blocks = []
    for mod in modalities:
        print(f"  Extracting {mod} features...")
        segs = dataset.X[mod]
        # preprocess each segment
        preprocessed = np.stack([PREPROCESSORS[mod](segs[i]) for i in range(len(segs))])
        feats = FEATURE_EXTRACTORS[mod](preprocessed)
        feature_blocks.append(feats)
    return np.concatenate(feature_blocks, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--modalities", nargs="+", default=["ECG", "EDA", "EEG", "Gaze"])
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = CLAREDataset(args.data_root, modalities=args.modalities).load()
    print(f"  Total segments: {len(dataset)}, subjects: {dataset.n_subjects}")
    print(f"  Label distribution: {np.bincount(dataset.y)}")

    print("Extracting features...")
    X = build_feature_matrix(dataset, args.modalities)
    y = dataset.y
    subjects = dataset.subject_ids
    print(f"  Feature matrix: {X.shape}")

    models = build_models()
    all_results: dict[str, dict] = {}

    for name, model_obj in models.items():
        print(f"\nEvaluating {name}...")

        # capture current model in closure
        def make_factory(m=model_obj):
            import copy
            return lambda: copy.deepcopy(m)

        factory = make_factory()
        r10 = kfold_evaluate(factory, X, y, n_splits=10)
        rloso = loso_evaluate(factory, X, y, subjects)
        all_results[name] = {"10-fold": r10, "LOSO": rloso}
        print(f"  10-fold: acc={r10['acc']*100:.2f}%, F1={r10['f1']*100:.2f}%")
        print(f"  LOSO:    acc={rloso['acc']*100:.2f}%, F1={rloso['f1']*100:.2f}%")

    print("\n=== Results ===")
    print_results_table(all_results)


if __name__ == "__main__":
    main()
