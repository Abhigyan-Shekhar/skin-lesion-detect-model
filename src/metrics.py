from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    max_k = min(k, logits.size(1))
    topk = torch.topk(logits, k=max_k, dim=1).indices
    matches = topk.eq(targets.unsqueeze(1))
    return float(matches.any(dim=1).float().mean().item())


def compute_epoch_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probs: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if probs is not None:
        targets = torch.tensor(y_true)
        logits = torch.tensor(probs)
        metrics["top3_accuracy"] = topk_accuracy(logits, targets, 3)
        metrics["top5_accuracy"] = topk_accuracy(logits, targets, 5)
    return metrics


def build_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: list[str],
) -> str:
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
