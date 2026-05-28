from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from recall_training import evaluate_full_v3, train_v3
from utils import read_yaml, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark backbones for 3-way recall-first training.")
    parser.add_argument("--config", default="configs/backbone_benchmark_3way.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = read_yaml(args.config)
    models = base_config.get("models", ["convnext_tiny"])
    output_root = Path(base_config.get("output_dir", "outputs_v3"))
    rows: list[dict] = []

    for model_name in models:
        config = dict(base_config)
        config["model_name"] = model_name
        out_dir = output_root / f"model_{model_name}"
        print(f"\n===== Training {model_name} =====")
        bundle = train_v3(
            config=config,
            output_dir=out_dir,
            train_csv=config["train_csv"],
            val_csv=config["val_csv"],
            label_column_override=config.get("label_column"),
        )
        from dataset import DermatologyDataset, build_transforms
        from torch.utils.data import DataLoader

        test_dataset = DermatologyDataset(
            csv_path=config["test_csv"],
            image_dir=config["image_dir"],
            image_column=bundle["image_column"],
            label_column=bundle["label_column"],
            class_to_idx=bundle["class_to_idx"],
            transform=build_transforms(int(config["image_size"]), train=False),
        )
        test_loader = DataLoader(test_dataset, batch_size=int(config["batch_size"]), shuffle=False)
        metrics = evaluate_full_v3(
            bundle["model"],
            test_loader,
            bundle["class_names"],
            bundle["image_column"],
            bundle["temperature"],
            bundle["device"],
            out_dir,
            tag="test",
            thresholds=bundle["thresholds"],
        )
        rows.append({"model_name": model_name, **{k: v for k, v in metrics.items() if isinstance(v, float)}})

    comparison = pd.DataFrame(rows).sort_values("recall_first_score", ascending=False)
    comparison_path = output_root / "benchmark_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    write_json(output_root / "benchmark_summary.json", {"best_model": comparison.iloc[0]["model_name"]})
    print(f"\nSaved benchmark to {comparison_path}")
    print(comparison)


if __name__ == "__main__":
    main()
