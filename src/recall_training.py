from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import LBFGS, AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import DermatologyDataset, build_transforms
from label_groups import load_class_loss_multipliers
from metrics import (
    build_classification_report,
    compute_epoch_metrics,
    selection_score_from_metrics,
)
from models import build_model, unfreeze_last_blocks
from thresholds import fit_per_class_thresholds, save_thresholds
from train import run_epoch, save_checkpoint
from utils import (
    DISCLAIMER_TEXT,
    choose_label_column,
    detect_columns,
    ensure_dir,
    read_table,
    resolve_device,
    set_seed,
    write_json,
)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)


def build_weighted_sampler(train_dataset: DermatologyDataset) -> WeightedRandomSampler:
    labels = train_dataset.df[train_dataset.label_column].astype(str).tolist()
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def make_loss_v3(
    train_dataset: DermatologyDataset,
    device: torch.device,
    config: dict,
    class_names: list[str],
) -> nn.Module:
    labels = train_dataset.df[train_dataset.label_column].astype(str).tolist()
    classes = np.array(sorted(train_dataset.class_to_idx.keys()))
    balanced = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    multipliers = load_class_loss_multipliers(config.get("class_map_path"))
    weights = []
    for cls_name, balanced_w in zip(classes, balanced):
        weights.append(float(balanced_w) * float(multipliers.get(str(cls_name), 1.0)))
    tensor_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    label_smoothing = float(config.get("label_smoothing", 0.1))
    if config.get("use_focal_loss"):
        return FocalLoss(
            gamma=float(config.get("focal_gamma", 2.0)),
            weight=tensor_weights,
            label_smoothing=label_smoothing,
        )
    return nn.CrossEntropyLoss(weight=tensor_weights, label_smoothing=label_smoothing)


def calibrate_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_iter: int = 50,
) -> float:
    model.eval()
    logits_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            logits_list.append(model(images.to(device)).cpu())
            targets_list.append(labels.cpu())
    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    scaler = TemperatureScaler(1.0)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled = scaler(logits)
        loss = F.cross_entropy(scaled, targets)
        loss.backward()
        return loss

    optimizer = LBFGS(scaler.parameters(), lr=0.01, max_iter=max_iter)
    optimizer.step(closure)
    return float(scaler.temperature.item())


def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[int] = []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Collecting logits", leave=False):
            all_logits.append(model(images.to(device)).cpu())
            all_targets.extend(labels.tolist())
    return torch.cat(all_logits, dim=0).numpy(), np.array(all_targets)


def evaluate_full_v3(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    image_column: str,
    temperature: float,
    device: torch.device,
    output_dir: str | Path,
    tag: str = "",
    thresholds: dict[int, float] | None = None,
) -> dict[str, Any]:
    from thresholds import predict_with_thresholds

    output_dir = Path(output_dir)
    metrics_dir = ensure_dir(output_dir / "metrics")
    plots_dir = ensure_dir(output_dir / "plots")
    predictions_dir = ensure_dir(output_dir / "predictions")
    n_classes = len(class_names)
    suffix = f"_{tag}" if tag else ""

    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[int] = []
    all_image_ids: list[str] = []

    with torch.no_grad():
        for images, labels, metadata in tqdm(test_loader, desc=f"Evaluating{suffix}", leave=False):
            images = images.to(device)
            logits = model(images).cpu()
            all_logits.append(logits)
            all_targets.extend(labels.tolist())
            batch_n = len(labels)
            ids = metadata.get(image_column, [str(i) for i in range(batch_n)])
            all_image_ids.extend(str(ids[i]) for i in range(batch_n))

    logits_t = torch.cat(all_logits, dim=0)
    targets_np = np.array(all_targets)
    targets_t = torch.tensor(targets_np)
    T = max(float(temperature), 0.05)
    cal_probs = torch.softmax(logits_t / T, dim=1).numpy()
    cal_preds = cal_probs.argmax(axis=1)
    if thresholds:
        cal_preds = predict_with_thresholds(cal_probs, thresholds)

    metrics = compute_epoch_metrics(
        targets_np.tolist(),
        cal_preds.tolist(),
        cal_probs,
        class_names=class_names,
    )
    metrics["tag"] = tag or "default"
    metrics["temperature"] = T
    metrics["calibrated_nll"] = float(F.cross_entropy(logits_t / T, targets_t).item())
    metrics["disclaimer"] = DISCLAIMER_TEXT

    try:
        metrics["macro_roc_auc"] = float(
            roc_auc_score(targets_np, cal_probs, multi_class="ovr", average="macro")
        )
    except Exception:
        metrics["macro_roc_auc"] = None

    write_json(metrics_dir / f"metrics{suffix}.json", metrics)

    precision, recall, f1_vals, support = precision_recall_fscore_support(
        targets_np, cal_preds, labels=list(range(n_classes)), zero_division=0
    )
    pd.DataFrame(
        {
            "class_name": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1_vals,
            "support": support,
        }
    ).to_csv(metrics_dir / f"per_class_metrics{suffix}.csv", index=False)

    report = build_classification_report(targets_np.tolist(), cal_preds.tolist(), class_names)
    with open(metrics_dir / f"classification_report{suffix}.txt", "w", encoding="utf-8") as fh:
        fh.write(report)
        fh.write(f"\n\n{DISCLAIMER_TEXT}\n")

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(targets_np, cal_preds, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix{suffix}")
    fig.tight_layout()
    fig.savefig(plots_dir / f"confusion_matrix_counts{suffix}.png", dpi=150)
    plt.close(fig)

    rows = []
    for i in range(len(targets_np)):
        top3_idx = np.argsort(cal_probs[i])[::-1][: min(3, n_classes)]
        rows.append(
            {
                "image_id": all_image_ids[i],
                "true_label": class_names[targets_np[i]],
                "predicted_label": class_names[cal_preds[i]],
                "top1_prob": round(float(cal_probs[i][top3_idx[0]]), 6),
            }
        )
    pd.DataFrame(rows).to_csv(predictions_dir / f"predictions_top3{suffix}.csv", index=False)

    print(f"\n=== Evaluation{suffix} ===")
    for key in ["macro_recall", "referral_urgent_recall", "recall_first_score", "macro_f1", "accuracy"]:
        val = metrics.get(key)
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")

    return metrics


def train_v3(
    config: dict,
    output_dir: str | Path,
    train_csv: str | Path,
    val_csv: str | Path,
    label_column_override: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    set_seed(int(config.get("seed", 42)))
    checkpoint_metric = config.get("checkpoint_metric", "recall_first")

    train_df = read_table(Path(train_csv))
    detected = detect_columns(list(train_df.columns))
    image_column = detected.get("image")
    if not image_column:
        raise ValueError(f"Cannot detect image column from: {list(train_df.columns)}")
    label_column = label_column_override or config.get("label_column") or choose_label_column(detected)
    if not label_column or label_column not in train_df.columns:
        raise ValueError(f"Label column missing. Columns: {list(train_df.columns)}")

    class_names = sorted(train_df[label_column].dropna().astype(str).unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    n_classes = len(class_names)
    print(f"Classes ({n_classes}): {class_names}")

    device = resolve_device(config.get("device", "auto"))
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    image_size = int(config.get("image_size", 224))
    image_dir = config["image_dir"]

    train_dataset = DermatologyDataset(
        csv_path=train_csv,
        image_dir=image_dir,
        image_column=image_column,
        label_column=label_column,
        class_to_idx=class_to_idx,
        transform=build_transforms(image_size=image_size, train=True),
    )
    val_dataset = DermatologyDataset(
        csv_path=val_csv,
        image_dir=image_dir,
        image_column=image_column,
        label_column=label_column,
        class_to_idx=class_to_idx,
        transform=build_transforms(image_size=image_size, train=False),
    )

    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": bool(config.get("pin_memory", True)),
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=build_weighted_sampler(train_dataset),
        drop_last=bool(config.get("drop_last", False)),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    model = build_model(
        model_name=config["model_name"],
        num_classes=n_classes,
        freeze_backbone=bool(config.get("freeze_backbone", True)),
    ).to(device)

    criterion = make_loss_v3(train_dataset, device, config, class_names)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")
    history: list[dict] = []
    best_score = -1.0
    patience = int(config.get("early_stopping_patience", 7))
    patience_counter = 0
    global_epoch = 0

    phases = [
        {"name": "head", "epochs": int(config["epochs_head"]), "lr": float(config["lr_head"]), "unfreeze": False},
        {
            "name": "finetune",
            "epochs": int(config["epochs_finetune"]),
            "lr": float(config["lr_finetune"]),
            "unfreeze": True,
        },
    ]

    for phase in phases:
        if phase["unfreeze"]:
            unfreeze_last_blocks(config["model_name"], model)
            patience_counter = 0
            print(f"Reset early-stopping patience for phase: {phase['name']}")
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=phase["lr"],
            weight_decay=float(config["weight_decay"]),
        )
        for phase_epoch in range(1, phase["epochs"] + 1):
            global_epoch += 1
            print(f"\n=== {config['model_name']} | {phase['name']} | epoch {phase_epoch}/{phase['epochs']} ===")
            train_loss, _train_metrics = run_epoch(
                model, train_loader, criterion, device, optimizer, scaler, amp_enabled
            )
            val_loss, _val_metrics = run_epoch(
                model, val_loader, criterion, device, None, None, amp_enabled
            )
            model.eval()
            val_targets: list[int] = []
            val_preds: list[int] = []
            val_probs: list[np.ndarray] = []
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images = images.to(device)
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    val_probs.extend(probs)
                    val_preds.extend(probs.argmax(axis=1).tolist())
                    val_targets.extend(labels.tolist())
            val_metrics = compute_epoch_metrics(
                val_targets, val_preds, np.asarray(val_probs), class_names=class_names
            )

            current_score = selection_score_from_metrics(val_metrics, checkpoint_metric)
            row = {
                "global_epoch": global_epoch,
                "phase": phase["name"],
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                **{f"val_{k}": round(v, 6) for k, v in val_metrics.items() if isinstance(v, float)},
            }
            history.append(row)
            with open(metrics_dir / "train_history.json", "w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=2)

            print(
                f"  val_{checkpoint_metric}={current_score:.4f} best={best_score:.4f} "
                f"urgent_recall={val_metrics.get('referral_urgent_recall', 'n/a')}"
            )

            if current_score > best_score:
                best_score = current_score
                patience_counter = 0
                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    model,
                    optimizer,
                    global_epoch,
                    config,
                    class_to_idx,
                    image_column,
                    label_column,
                    best_score,
                )
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break
        if patience_counter >= patience:
            break

    ckpt_path = checkpoints_dir / "best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    temperature = calibrate_temperature(model, val_loader, device)
    write_json(metrics_dir / "temperature.json", {"temperature": temperature})

    val_logits, val_targets = collect_logits(model, val_loader, device)
    val_probs = torch.softmax(torch.tensor(val_logits) / temperature, dim=1).numpy()
    thresholds = fit_per_class_thresholds(val_probs, val_targets, n_classes)
    save_thresholds(metrics_dir / "per_class_thresholds.json", thresholds, class_names)

    transform = build_transforms(image_size=image_size, train=False)
    return {
        "model": model,
        "device": device,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "image_column": image_column,
        "label_column": label_column,
        "config": config,
        "checkpoint_path": str(ckpt_path),
        "temperature": temperature,
        "thresholds": thresholds,
        "transform": transform,
        "best_score": best_score,
    }
