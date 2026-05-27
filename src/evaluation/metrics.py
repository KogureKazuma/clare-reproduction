"""Evaluation utilities: 10-fold CV and LOSO."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


def kfold_evaluate(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    random_state: int = 42,
) -> dict[str, float]:
    """Stratified K-fold cross-validation for sklearn-compatible models."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, f1s = [], []
    for train_idx, test_idx in skf.split(X, y):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], preds))
        f1s.append(f1_score(y[test_idx], preds, average="weighted", zero_division=0))
    return {"acc": float(np.mean(accs)), "f1": float(np.mean(f1s))}


def loso_evaluate(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
) -> dict[str, float]:
    """Leave-One-Subject-Out cross-validation for sklearn-compatible models."""
    subjects = np.unique(subject_ids)
    accs, f1s = [], []
    for subj in subjects:
        test_mask = subject_ids == subj
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        model = model_factory()
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        accs.append(accuracy_score(y[test_mask], preds))
        f1s.append(f1_score(y[test_mask], preds, average="weighted", zero_division=0))
    return {"acc": float(np.mean(accs)), "f1": float(np.mean(f1s))}


def print_results_table(results: dict[str, dict[str, dict[str, float]]]) -> None:
    """Print a formatted results table.

    Args:
        results: {model_name: {scheme: {"acc": ..., "f1": ...}}}
    """
    schemes = ["10-fold", "LOSO"]
    header = f"{'Model':<12}" + "".join(f"  {s} Acc   {s} F1 " for s in schemes)
    print(header)
    print("-" * len(header))
    for model_name, scheme_results in results.items():
        row = f"{model_name:<12}"
        for scheme in schemes:
            if scheme in scheme_results:
                r = scheme_results[scheme]
                row += f"  {r['acc']*100:6.2f}%  {r['f1']*100:6.2f}%"
            else:
                row += "  " + " " * 16
        print(row)
