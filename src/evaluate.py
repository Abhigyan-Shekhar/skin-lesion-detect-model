from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DermatologyDataset, build_transforms
from metrics import build_classification_report, compute_epoch_metrics
from models import build_model
from utils import DISCLAMER_TEXT, ensure_dir, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dermatology classifier.")
    parser.add_argument("--checkpoint", required=True, help="Path to best checkpoint")
    parser.add_argument("--split", required=True, help="Path to test split CSV")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    split_path = Path(args.split)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    class_names = [idx_to_class[idx] for idx in range(len(idx_to_class))]

    device = resolve_device(config.get("device", "auto"))
    model = build_model(
        model_name=config["model_name"],
        num_classes=len(class_names),
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    dataset = DermatologyDataset(
        csv_path=split_path,
        image_dir=config["image_dir"],
        image_column=checkpoint["image_column"],
        label_column=checkpoint["label_column"],
        class_to_idx=class_to_idx,
        transform=build_transforms(image_size=int(config["image_size"]), train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    all_rows: list[dict] = []

    with torch.no_grad():
        for images, labels, metadata in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_targets.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

            batch_size = len(preds)
            for index in range(batch_size):
                top_indices = np.argsort(probs[index])[::-1][:5]
                image_id = metadata[checkpoint["image_column"]][index]
                all_rows.append(
                    {
                        "image_id": image_id,
                        "true_label": idx_to_class[int(labels[index].item())],
                        "predicted_label": idx_to_class[int(preds[index])],
                        "top5_labels": "|".join(idx_to_class[int(i)] for i in top_indices),
                        "top5_probs": "|".join(f"{float(probs[index][i]):.6f}" for i in top_indices),
                    }
                )

    probs_array = np.asarray(all_probs)
    metrics = compute_epoch_metrics(all_targets, all_preds, probs_array)
    metrics["disclaimer"] = DISCLAMER_TEXT

    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(len(class_names))), zero_division=0
    )
    per_class_df = pd.DataFrame(
        {
            "class_name": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    report = build_classification_report(all_targets, all_preds, class_names)
    matrix = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))

    output_dir = Path(args.output_dir)
    metrics_dir = ensure_dir(output_dir / "metrics")
    plots_dir = ensure_dir(output_dir / "plots")
    predictions_dir = ensure_dir(output_dir / "predictions")

    with open(metrics_dir / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(report)
        handle.write(f"\n\n{DISCLAMER_TEXT}\n")
    with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    per_class_df.to_csv(metrics_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(all_rows).to_csv(predictions_dir / "predictions.csv", index=False)

    fig_width = max(8, len(class_names) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    write_json(
        metrics_dir / "evaluation_summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "split": str(split_path),
            "metrics": metrics,
            "num_examples": len(dataset),
        },
    )

    print("\n=== Test Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")
    print(f"\nSaved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
