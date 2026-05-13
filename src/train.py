from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DermatologyDataset, build_transforms
from metrics import compute_epoch_metrics
from models import build_model, unfreeze_last_blocks
from utils import (
    choose_label_column,
    detect_columns,
    ensure_dir,
    read_table,
    read_yaml,
    resolve_device,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dermatology classifier.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--label_column", default=None, help="Optional explicit label column")
    return parser.parse_args()


def load_split_metadata(train_csv: Path, val_csv: Path, label_override: str | None) -> tuple[str, str, list[str]]:
    train_df = read_table(train_csv)
    val_df = read_table(val_csv)
    shared_columns = list(train_df.columns)
    detected = detect_columns(shared_columns)
    image_column = detected.get("image")
    if not image_column:
        raise ValueError(f"Could not detect image column from split columns: {shared_columns}")

    label_column = choose_label_column(detected, label_override)
    if not label_column or label_column not in train_df.columns:
        raise ValueError(
            "Could not detect label column automatically. "
            f"Available columns: {shared_columns}. Pass --label_column explicitly."
        )

    class_names = sorted(train_df[label_column].dropna().astype(str).unique().tolist())
    unseen_val = set(val_df[label_column].dropna().astype(str).unique()) - set(class_names)
    if unseen_val:
        raise ValueError(f"Validation split contains unseen labels: {sorted(unseen_val)}")
    return image_column, label_column, class_names


def create_dataloaders(config: dict, image_column: str, label_column: str, class_names: list[str]):
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    image_size = int(config["image_size"])
    image_dir = config["image_dir"]

    train_dataset = DermatologyDataset(
        csv_path=config["train_csv"],
        image_dir=image_dir,
        image_column=image_column,
        label_column=label_column,
        class_to_idx=class_to_idx,
        transform=build_transforms(image_size=image_size, train=True),
    )
    val_dataset = DermatologyDataset(
        csv_path=config["val_csv"],
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
        shuffle=True,
        drop_last=bool(config.get("drop_last", False)),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_dataset, val_dataset, train_loader, val_loader, class_to_idx


def make_loss(train_dataset: DermatologyDataset, use_class_weights: bool, device: torch.device) -> nn.Module:
    if not use_class_weights:
        return nn.CrossEntropyLoss()

    labels = train_dataset.df[train_dataset.label_column].astype(str).tolist()
    classes = np.array(sorted(train_dataset.class_to_idx.keys()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    tensor_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=tensor_weights)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp_enabled: bool = False,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    progress = tqdm(loader, leave=False)
    for images, labels, _metadata in progress:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=autocast_device, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

        if is_train:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        probabilities = torch.softmax(logits.detach().cpu(), dim=1).numpy()
        predictions = probabilities.argmax(axis=1)
        all_probs.extend(probabilities)
        all_preds.extend(predictions.tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    epoch_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_epoch_metrics(all_targets, all_preds, np.asarray(all_probs))
    return epoch_loss, metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    config: dict,
    class_to_idx: dict[str, int],
    image_column: str,
    label_column: str,
    best_score: float,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "class_to_idx": class_to_idx,
            "image_column": image_column,
            "label_column": label_column,
            "best_macro_f1": best_score,
        },
        path,
    )


def append_log_row(csv_path: Path, row: dict) -> None:
    ensure_dir(csv_path.parent)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    set_seed(int(config.get("seed", 42)))

    train_csv = Path(config["train_csv"])
    val_csv = Path(config["val_csv"])
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            "Train/val split CSVs were not found. Run src/prepare_splits.py first."
        )

    image_column, label_column, class_names = load_split_metadata(train_csv, val_csv, args.label_column)
    device = resolve_device(config.get("device", "auto"))
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"

    (
        train_dataset,
        _val_dataset,
        train_loader,
        val_loader,
        class_to_idx,
    ) = create_dataloaders(config, image_column, label_column, class_names)

    model = build_model(
        model_name=config["model_name"],
        num_classes=len(class_names),
        freeze_backbone=bool(config.get("freeze_backbone", True)),
    )
    model = model.to(device)

    criterion = make_loss(train_dataset, bool(config.get("use_class_weights", True)), device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    output_dir = Path(config["output_dir"])
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")
    history_json_path = metrics_dir / "train_history.json"
    history_csv_path = metrics_dir / "train_history.csv"

    history: list[dict] = []
    best_macro_f1 = -1.0
    patience = int(config.get("early_stopping_patience", 7))
    patience_counter = 0
    global_epoch = 0

    training_phases = [
        {
            "name": "head",
            "epochs": int(config["epochs_head"]),
            "lr": float(config["lr_head"]),
            "before_phase": None,
        },
        {
            "name": "finetune",
            "epochs": int(config["epochs_finetune"]),
            "lr": float(config["lr_finetune"]),
            "before_phase": lambda: unfreeze_last_blocks(config["model_name"], model),
        },
    ]

    for phase in training_phases:
        if phase["before_phase"] is not None:
            phase["before_phase"]()

        optimizer = AdamW(
            filter(lambda parameter: parameter.requires_grad, model.parameters()),
            lr=phase["lr"],
            weight_decay=float(config["weight_decay"]),
        )

        for phase_epoch in range(1, phase["epochs"] + 1):
            global_epoch += 1
            print(f"\n=== Phase: {phase['name']} | Epoch {phase_epoch}/{phase['epochs']} ===")

            train_loss, train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
            )
            val_loss, val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=None,
                amp_enabled=amp_enabled,
            )

            row = {
                "global_epoch": global_epoch,
                "phase": phase["name"],
                "phase_epoch": phase_epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                **{f"train_{key}": round(value, 6) for key, value in train_metrics.items()},
                **{f"val_{key}": round(value, 6) for key, value in val_metrics.items()},
            }
            history.append(row)
            append_log_row(history_csv_path, row)
            with open(history_json_path, "w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)

            save_checkpoint(
                checkpoints_dir / "latest.pt",
                model,
                optimizer,
                global_epoch,
                config,
                class_to_idx,
                image_column,
                label_column,
                best_macro_f1,
            )
            if bool(config.get("save_every_epoch", False)):
                save_checkpoint(
                    checkpoints_dir / f"epoch_{global_epoch:03d}.pt",
                    model,
                    optimizer,
                    global_epoch,
                    config,
                    class_to_idx,
                    image_column,
                    label_column,
                    best_macro_f1,
                )

            current_macro_f1 = val_metrics["macro_f1"]
            if current_macro_f1 > best_macro_f1:
                best_macro_f1 = current_macro_f1
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
                    best_macro_f1,
                )
            else:
                patience_counter += 1

            print(
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_macro_f1={current_macro_f1:.4f} best_macro_f1={best_macro_f1:.4f}"
            )

            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience_counter} non-improving epochs.")
                write_json(
                    metrics_dir / "training_summary.json",
                    {
                        "best_macro_f1": best_macro_f1,
                        "stopped_early": True,
                        "completed_epochs": global_epoch,
                        "label_column": label_column,
                        "image_column": image_column,
                        "class_names": class_names,
                    },
                )
                return

    write_json(
        metrics_dir / "training_summary.json",
        {
            "best_macro_f1": best_macro_f1,
            "stopped_early": False,
            "completed_epochs": global_epoch,
            "label_column": label_column,
            "image_column": image_column,
            "class_names": class_names,
        },
    )
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
