from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import IMAGENET_MEAN, IMAGENET_STD, read_table


def safe_metadata_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class DermatologyDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        image_column: str,
        label_column: str,
        class_to_idx: dict[str, int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.df = read_table(csv_path)
        self.image_dir = Path(image_dir)
        self.image_column = image_column
        self.label_column = label_column
        self.class_to_idx = class_to_idx
        self.transform = transform

        missing_cols = [column for column in [image_column, label_column] if column not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")

        self.df = self.df[self.df[label_column].notna()].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        row = self.df.iloc[index]
        raw_image_path = Path(str(row[self.image_column]))
        image_path = raw_image_path if raw_image_path.is_absolute() else self.image_dir / raw_image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label_name = str(row[self.label_column])
        label = self.class_to_idx[label_name]
        metadata = {key: safe_metadata_value(value) for key, value in row.to_dict().items()}
        metadata["image_path"] = str(image_path)
        return image, label, metadata
