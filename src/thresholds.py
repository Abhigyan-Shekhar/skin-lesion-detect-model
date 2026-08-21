from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def fit_per_class_thresholds(
    probs: np.ndarray,
    y_true: Sequence[int],
    num_classes: int,
    min_precision: float = 0.15,
    default_threshold: float = 0.33,
) -> dict[int, float]:
    """Per-class thresholds tuned to maximize recall subject to a precision floor."""
    y_true_arr = np.asarray(y_true)
    thresholds: dict[int, float] = {}
    for class_idx in range(num_classes):
        binary_y = (y_true_arr == class_idx).astype(int)
        if binary_y.sum() < 2:
            thresholds[class_idx] = default_threshold
            continue
        class_probs = probs[:, class_idx]
        candidates = np.unique(np.concatenate([[0.05, 0.95], class_probs]))
        best_threshold = default_threshold
        best_recall = -1.0
        for threshold in candidates:
            preds = (class_probs >= threshold).astype(int)
            precision, recall, _, _ = precision_recall_fscore_support(
                binary_y, preds, average="binary", zero_division=0
            )
            if precision >= min_precision and recall > best_recall:
                best_recall = float(recall)
                best_threshold = float(threshold)
        thresholds[class_idx] = best_threshold
    return thresholds


def predict_with_thresholds(probs: np.ndarray, thresholds: dict[int, float]) -> np.ndarray:
    """Argmax over classes meeting threshold; fallback to global argmax if none fire."""
    n_samples = probs.shape[0]
    preds = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        fired = [c for c, t in thresholds.items() if probs[i, c] >= t]
        if fired:
            preds[i] = max(fired, key=lambda c: probs[i, c])
        else:
            preds[i] = int(probs[i].argmax())
    return preds


def thresholds_to_json(
    thresholds: dict[int, float],
    class_names: Sequence[str],
) -> dict[str, float]:
    return {class_names[idx]: float(thresholds[idx]) for idx in sorted(thresholds.keys())}


def save_thresholds(path: str | Path, thresholds: dict[int, float], class_names: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(thresholds_to_json(thresholds, class_names), handle, indent=2)


def load_thresholds(path: str | Path, class_names: Sequence[str]) -> dict[int, float]:
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    return {name_to_idx[name]: float(value) for name, value in payload.items()}
