from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from dataset import build_transforms
from models import build_model
from utils import DISCLAIMER_TEXT, resolve_device


def confidence_level(max_prob: float) -> str:
    if max_prob >= 0.75:
        return "high"
    if max_prob >= 0.45:
        return "moderate"
    return "low"


def load_checkpoint_bundle(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}
    device = resolve_device(config.get("device", "auto"))

    model = build_model(
        model_name=config["model_name"],
        num_classes=len(class_to_idx),
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = build_transforms(image_size=int(config["image_size"]), train=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint": checkpoint,
        "config": config,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "device": device,
        "model": model,
        "transform": transform,
    }


def predict_with_bundle(
    bundle: dict[str, Any],
    image_path: str | Path,
    top_k: int = 5,
) -> dict[str, Any]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    tensor = bundle["transform"](image).unsqueeze(0).to(bundle["device"])

    with torch.no_grad():
        logits = bundle["model"](tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_k = min(top_k, probs.shape[0])
        values, indices = torch.topk(probs, k=top_k)

    predictions = [
        {
            "label": bundle["idx_to_class"][int(index.item())],
            "probability": round(float(value.item()), 6),
        }
        for value, index in zip(values, indices)
    ]
    max_prob = float(values[0].item()) if len(values) else 0.0
    return {
        "top_predictions": predictions,
        "confidence_level": confidence_level(max_prob),
        "max_probability": round(max_prob, 6),
        "checkpoint_path": bundle["checkpoint_path"],
        "model_name": bundle["config"]["model_name"],
        "task_name": bundle["config"].get("task_name"),
        "disclaimer": DISCLAIMER_TEXT,
    }


def predict_single_checkpoint(
    checkpoint_path: str | Path,
    image_path: str | Path,
    top_k: int = 5,
) -> dict[str, Any]:
    bundle = load_checkpoint_bundle(checkpoint_path)
    return predict_with_bundle(bundle=bundle, image_path=image_path, top_k=top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one dermatology image.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--top_k", type=int, default=5, help="Number of predictions to return")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_single_checkpoint(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
