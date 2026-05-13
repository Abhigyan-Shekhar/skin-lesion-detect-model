from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from dataset import build_transforms
from models import build_model
from utils import DISCLAMER_TEXT, resolve_device


def confidence_level(max_prob: float) -> str:
    if max_prob >= 0.75:
        return "high"
    if max_prob >= 0.45:
        return "moderate"
    return "low"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one dermatology image.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--top_k", type=int, default=5, help="Number of predictions to return")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    image_path = Path(args.image)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

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
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_k = min(args.top_k, probs.shape[0])
        values, indices = torch.topk(probs, k=top_k)

    predictions = [
        {
            "label": idx_to_class[int(index.item())],
            "probability": round(float(value.item()), 6),
        }
        for value, index in zip(values, indices)
    ]
    result = {
        "top_predictions": predictions,
        "confidence_level": confidence_level(float(values[0].item())),
        "disclaimer": DISCLAMER_TEXT,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
