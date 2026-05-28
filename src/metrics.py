from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    max_k = min(k, logits.size(1))
    topk = torch.topk(logits, k=max_k, dim=1).indices
    matches = topk.eq(targets.unsqueeze(1))
    return float(matches.any(dim=1).float().mean().item())


def macro_recall(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


def min_class_recall(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> float:
    _, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    return float(np.min(recall))


def class_recall(y_true: Sequence[int], y_pred: Sequence[int], class_idx: int) -> float:
    _, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[class_idx], zero_division=0
    )
    return float(recall[0])


def recall_first_selection_score(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    urgent_class_name: str = "referral_urgent",
) -> float:
    macro_rec = macro_recall(y_true, y_pred)
    if urgent_class_name in class_names:
        urgent_idx = list(class_names).index(urgent_class_name)
        urgent_rec = class_recall(y_true, y_pred, urgent_idx)
    else:
        urgent_rec = macro_rec
    return 0.5 * macro_rec + 0.5 * urgent_rec


def selection_score_from_metrics(
    metrics: dict[str, float],
    checkpoint_metric: str = "macro_f1",
) -> float:
    if checkpoint_metric == "recall_first":
        return float(metrics.get("recall_first_score", metrics.get("macro_recall", 0.0)))
    return float(metrics.get("macro_f1", 0.0))


def compute_epoch_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probs: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, float]:
    n_classes = (
        len(class_names)
        if class_names
        else max(max(y_true, default=0), max(y_pred, default=0)) + 1
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_recall": macro_recall(y_true, y_pred),
        "min_class_recall": min_class_recall(y_true, y_pred, n_classes),
    }
    if class_names:
        metrics["recall_first_score"] = recall_first_selection_score(y_true, y_pred, class_names)
        if "referral_urgent" in class_names:
            urgent_idx = list(class_names).index("referral_urgent")
            metrics["referral_urgent_recall"] = class_recall(y_true, y_pred, urgent_idx)
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
