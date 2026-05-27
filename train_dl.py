#!/usr/bin/env python3
"""Train and evaluate CNN / Transformer models on the CLARE dataset.

Usage:
    python train_dl.py --data_root ./data --model cnn
    python train_dl.py --data_root ./data --model transformer
"""

from __future__ import annotations

import argparse
import copy
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data_loader import CLAREDataset, FS
from src.preprocessing import ecg as ecg_pre, eeg as eeg_pre, gaze as gaze_pre
from src.models.cnn import build_cnn, FocalLoss
from src.models.transformer import build_transformer
from src.evaluation.metrics import print_results_table

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# number of channels after preprocessing
MODALITY_CHANNELS = {"ECG": 3, "EDA": 1, "EEG": 4, "Gaze": 3}


def preprocess_all(dataset: CLAREDataset, modalities: list[str]) -> dict[str, np.ndarray]:
    """Preprocess all segments, returns dict mod → (N, T, C)."""
    preprocessors = {
        "ECG": lambda s: ecg_pre.preprocess(s, FS["ECG"]),
        "EDA": lambda s: s,
        "EEG": lambda s: eeg_pre.preprocess(s, FS["EEG"]),
        "Gaze": lambda s: gaze_pre.preprocess(s),
    }
    result = {}
    for mod in modalities:
        segs = dataset.X[mod]
        processed = np.stack([preprocessors[mod](segs[i]) for i in range(len(segs))])
        if processed.ndim == 2:
            processed = processed[:, :, None]  # (N, T, 1)
        result[mod] = processed.astype(np.float32)
    return result


def to_tensors(X_dict: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Convert (N, T, C) arrays to (N, C, T) tensors for conv."""
    return {
        mod: torch.from_numpy(arr.transpose(0, 2, 1))
        for mod, arr in X_dict.items()
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    modalities: list[str],
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        *mod_tensors, labels = batch
        x_dict = {mod: t.to(DEVICE) for mod, t in zip(modalities, mod_tensors)}
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x_dict)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    modalities: list[str],
) -> tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        *mod_tensors, labels = batch
        x_dict = {mod: t.to(DEVICE) for mod, t in zip(modalities, mod_tensors)}
        logits = model(x_dict)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return acc, f1


def make_dataset_tensors(X_tensors: dict[str, torch.Tensor], y: np.ndarray) -> TensorDataset:
    mods = list(X_tensors.keys())
    return TensorDataset(*[X_tensors[m] for m in mods], torch.from_numpy(y).long())


def build_model(
    model_type: Literal["cnn", "transformer"],
    modalities: list[str],
    X_dict: dict[str, np.ndarray],
) -> nn.Module:
    mod_channels = {mod: X_dict[mod].shape[2] for mod in modalities}
    if model_type == "cnn":
        return build_cnn(mod_channels).to(DEVICE)
    else:
        # pick a unified sequence length (use shortest modality)
        seq_lens = {mod: X_dict[mod].shape[1] for mod in modalities}
        seq_len = min(seq_lens.values())
        return build_transformer(mod_channels, seq_len=seq_len).to(DEVICE)


def train_and_eval(
    model_type: str,
    X_np: dict[str, np.ndarray],
    y: np.ndarray,
    modalities: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int,
    batch_size: int,
) -> tuple[float, float]:
    X_tensors = to_tensors(X_np)
    full_ds = make_dataset_tensors(X_tensors, y)

    train_ds = Subset(full_ds, train_idx)
    test_ds = Subset(full_ds, test_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(model_type, modalities, X_np)
    criterion = FocalLoss(alpha=4.0, gamma=2.0)

    if model_type == "cnn":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=5e-3, rho=0.95)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, modalities)

    acc, f1 = evaluate(model, test_loader, modalities)
    return acc, f1


def kfold_dl(
    model_type: str,
    X_np: dict[str, np.ndarray],
    y: np.ndarray,
    modalities: list[str],
    n_splits: int = 10,
    epochs: int = 100,
    batch_size: int = 256,
) -> dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = np.arange(len(y))
    accs, f1s = [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, y)):
        print(f"  Fold {fold+1}/{n_splits}")
        acc, f1 = train_and_eval(
            model_type, X_np, y, modalities, train_idx, test_idx, epochs, batch_size
        )
        accs.append(acc)
        f1s.append(f1)
    return {"acc": float(np.mean(accs)), "f1": float(np.mean(f1s))}


def loso_dl(
    model_type: str,
    X_np: dict[str, np.ndarray],
    y: np.ndarray,
    subject_ids: np.ndarray,
    modalities: list[str],
    epochs: int = 100,
    batch_size: int = 256,
) -> dict[str, float]:
    subjects = np.unique(subject_ids)
    accs, f1s = [], []
    for subj in subjects:
        print(f"  LOSO subject {subj+1}/{len(subjects)}")
        test_idx = np.where(subject_ids == subj)[0]
        train_idx = np.where(subject_ids != subj)[0]
        acc, f1 = train_and_eval(
            model_type, X_np, y, modalities, train_idx, test_idx, epochs, batch_size
        )
        accs.append(acc)
        f1s.append(f1)
    return {"acc": float(np.mean(accs)), "f1": float(np.mean(f1s))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--model", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--modalities", nargs="+", default=["ECG", "EDA", "EEG", "Gaze"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--scheme", choices=["kfold", "loso", "both"], default="both")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print("Loading dataset...")
    dataset = CLAREDataset(args.data_root, modalities=args.modalities).load()
    print(f"  Segments: {len(dataset)}, subjects: {dataset.n_subjects}")

    print("Preprocessing signals...")
    X_np = preprocess_all(dataset, args.modalities)
    y = dataset.y
    subjects = dataset.subject_ids

    all_results: dict[str, dict] = {args.model: {}}

    if args.scheme in ("kfold", "both"):
        print(f"\n10-fold CV ({args.model.upper()})...")
        r10 = kfold_dl(args.model, X_np, y, args.modalities, epochs=args.epochs, batch_size=args.batch_size)
        all_results[args.model]["10-fold"] = r10
        print(f"  acc={r10['acc']*100:.2f}%, F1={r10['f1']*100:.2f}%")

    if args.scheme in ("loso", "both"):
        print(f"\nLOSO ({args.model.upper()})...")
        rloso = loso_dl(args.model, X_np, y, subjects, args.modalities, epochs=args.epochs, batch_size=args.batch_size)
        all_results[args.model]["LOSO"] = rloso
        print(f"  acc={rloso['acc']*100:.2f}%, F1={rloso['f1']*100:.2f}%")

    print("\n=== Results ===")
    print_results_table(all_results)


if __name__ == "__main__":
    main()
