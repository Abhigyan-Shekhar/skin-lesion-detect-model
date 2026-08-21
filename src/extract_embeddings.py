from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DermatologyDataset, build_transforms
from models import build_model
from utils import detect_columns, read_table, resolve_device


def _embedding_dim(model_name: str) -> int:
    name = model_name.lower()
    if name == "efficientnet_b0":
        return 1280
    if name == "resnet50":
        return 2048
    if name == "convnext_tiny":
        return 768
    if name == "swin_tiny":
        return 768
    return 512


def build_embedding_model(model_name: str, checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    model = build_model(model_name, num_classes=len(class_to_idx), freeze_backbone=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    name = model_name.lower()

    def forward_features(x: torch.Tensor) -> torch.Tensor:
        if name == "efficientnet_b0":
            x = model.features(x)
            x = model.avgpool(x)
            return torch.flatten(x, 1)
        if name == "resnet50":
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            return torch.flatten(x, 1)
        if name == "convnext_tiny":
            x = model.features(x)
            x = model.avgpool(x)
            return torch.flatten(x, 1)
        return model(x)

    model.forward_features = forward_features  # type: ignore[attr-defined]
    return model


def extract_embeddings(
    split_csv: str | Path,
    image_dir: str | Path,
    checkpoint_path: str | Path,
    model_name: str,
    label_column: str,
    image_column: str,
    class_to_idx: dict[str, int],
    output_path: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    device: str = "auto",
) -> pd.DataFrame:
    device_obj = resolve_device(device)
    model = build_embedding_model(model_name, checkpoint_path, device_obj)

    dataset = DermatologyDataset(
        csv_path=split_csv,
        image_dir=image_dir,
        image_column=image_column,
        label_column=label_column,
        class_to_idx=class_to_idx,
        transform=build_transforms(image_size=image_size, train=False),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    rows: list[list[float]] = []
    labels: list[int] = []
    image_ids: list[str] = []

    with torch.no_grad():
        for images, batch_labels, metadata in tqdm(loader, desc="Embeddings"):
            images = images.to(device_obj)
            feats = model.forward_features(images).cpu().numpy()  # type: ignore[attr-defined]
            rows.extend(feats.tolist())
            labels.extend(batch_labels.tolist())
            ids = metadata.get(image_column, [""] * len(batch_labels))
            image_ids.extend(str(ids[i]) for i in range(len(batch_labels)))

    dim = _embedding_dim(model_name)
    columns = [f"emb_{i}" for i in range(rows[0] if rows else range(dim))]
    df = pd.DataFrame(rows, columns=columns)
    df["image_id"] = image_ids
    df["label_idx"] = labels
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False) if out.suffix == ".parquet" else df.to_csv(out, index=False)
    return df


def build_tabular_frame(
    embeddings_df: pd.DataFrame,
    metadata_csv: str | Path,
    detected_columns: dict | None = None,
) -> pd.DataFrame:
    meta = read_table(metadata_csv)
    if detected_columns is None:
        detected_columns = detect_columns(meta.columns.tolist())
    image_col = detected_columns.get("image")
    if not image_col:
        raise ValueError("No image column in metadata")
    merged = embeddings_df.merge(meta, left_on="image_id", right_on=image_col, how="inner")
    return merged
