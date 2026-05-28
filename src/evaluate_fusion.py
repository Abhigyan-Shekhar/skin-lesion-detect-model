from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from coarse_fusion import predict_coarse_label, tune_alpha
from history_simulator import simulate_answers_batch, simulation_coverage
from metrics import compute_epoch_metrics, macro_recall, recall_first_selection_score
from utils import write_json


def probs_row_from_vector(probs: np.ndarray, class_names: list[str]) -> dict[str, float]:
    return {name: float(probs[i]) for i, name in enumerate(class_names)}


def run_fusion_experiments(
    model: torch.nn.Module,
    test_loader: DataLoader,
    test_df: pd.DataFrame,
    class_names: list[str],
    device: torch.device,
    temperature: float,
    output_dir: str | Path,
    alpha: float | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    model.eval()
    T = max(float(temperature), 0.05)

    image_prob_rows: list[dict[str, float]] = []
    y_true: list[str] = []
    image_preds: list[int] = []

    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Fusion eval", leave=False):
            images = images.to(device)
            logits = model(images).cpu()
            probs = torch.softmax(logits / T, dim=1).numpy()
            for i in range(len(labels)):
                row_probs = probs_row_from_vector(probs[i], class_names)
                image_prob_rows.append(row_probs)
                true_idx = int(labels[i].item())
                y_true.append(class_names[true_idx])
                image_preds.append(int(probs[i].argmax()))

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y_idx = [class_to_idx[y] for y in y_true]

    metrics_a = compute_epoch_metrics(y_idx, image_preds, class_names=class_names)

    simulated_answers = simulate_answers_batch(test_df.iloc[: len(y_true)])
    coverage = simulation_coverage(test_df.iloc[: len(y_true)])

    if alpha is None:
        alpha, _ = tune_alpha(image_prob_rows, simulated_answers, y_true, class_names)

    fused_preds_b: list[int] = []
    for probs, answers in zip(image_prob_rows, simulated_answers):
        label, _ = predict_coarse_label(probs, answers, alpha=alpha)
        fused_preds_b.append(class_to_idx[label])

    metrics_b = compute_epoch_metrics(y_idx, fused_preds_b, class_names=class_names)

    fused_preds_c: list[int] = []
    for probs, answers in zip(image_prob_rows, simulated_answers):
        label, _ = predict_coarse_label(probs, answers if answers else None, alpha=alpha)
        fused_preds_c.append(class_to_idx[label])
    metrics_c = compute_epoch_metrics(y_idx, fused_preds_c, class_names=class_names)

    lift = {
        "experiment_A_image_only": metrics_a,
        "experiment_B_image_plus_oracle_history": metrics_b,
        "experiment_C_image_plus_partial_history": metrics_c,
        "delta_macro_recall_B_minus_A": metrics_b["macro_recall"] - metrics_a["macro_recall"],
        "delta_macro_recall_C_minus_A": metrics_c["macro_recall"] - metrics_a["macro_recall"],
        "delta_urgent_recall_B_minus_A": (
            (metrics_b.get("referral_urgent_recall") or 0.0)
            - (metrics_a.get("referral_urgent_recall") or 0.0)
        ),
        "fusion_alpha": alpha,
        "simulation_coverage": coverage,
        "disclaimer": metrics_a.get("disclaimer", ""),
    }
    write_json(output_dir / "metrics" / "fusion_lift.json", lift)

    summary_df = pd.DataFrame(
        [
            {"experiment": "A_image_only", **{k: v for k, v in metrics_a.items() if isinstance(v, float)}},
            {"experiment": "B_oracle_history", **{k: v for k, v in metrics_b.items() if isinstance(v, float)}},
            {"experiment": "C_partial_history", **{k: v for k, v in metrics_c.items() if isinstance(v, float)}},
        ]
    )
    summary_df.to_csv(output_dir / "metrics" / "fusion_comparison.csv", index=False)
    return lift
