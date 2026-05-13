from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import torch
except ModuleNotFoundError:  # Allows non-ML utilities to run before torch is installed.
    torch = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DISCLAIMER_TEXT = (
    "Research-only. Not a diagnosis. Doctor review required. "
    "Non-commercial use only. Not for clinical deployment."
)

# Backward-compatible alias for earlier Milestone 1 scripts.
DISCLAMER_TEXT = DISCLAIMER_TEXT

DEFAULT_COLUMN_ALIASES = {
    "image": [
        "image",
        "image_id",
        "image_name",
        "image_filename",
        "filename",
        "file_name",
        "img",
        "img_id",
        "img_name",
        "img_path",
        "path",
    ],
    "diagnosis": [
        "diagnosis",
        "dx",
        "label",
        "disease",
        "condition",
        "final_diagnosis",
        "diagnostic_label",
    ],
    "main_class": [
        "main_class",
        "main category",
        "main_category",
        "broad_class",
        "broad_category",
        "category",
        "super_class",
        "coarse_label",
    ],
    "subclass": [
        "subclass",
        "sub_class",
        "sub category",
        "sub_category",
        "fine_class",
        "fine_category",
        "disease_group",
    ],
    "age": ["age", "patient_age"],
    "sex": ["sex", "gender", "patient_sex"],
    "patient_id": [
        "patient_id",
        "patient",
        "patientid",
        "subject_id",
        "case_id",
        "person_id",
        "mrn",
        "study_id",
    ],
    "body_site": ["body_site", "site", "anatomical_site", "location", "body location"],
    "fitzpatrick": ["fitzpatrick", "fitzpatrick_type", "fitzpatrick_skin_type"],
    "monk_skin_tone": ["monk_skin_tone", "monk_tone", "monk_skin"],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV/TSV metadata while tolerating mislabeled .csv files from Dataverse."""
    path = Path(path)
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except pd.errors.ParserError:
        suffix = path.suffix.lower()
        if suffix in {".tab", ".tsv"}:
            return pd.read_csv(path, sep="\t")
        if suffix == ".csv":
            return pd.read_csv(path, sep="\t")
        raise


def normalize_column_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def detect_columns(columns: list[str]) -> dict[str, str | None]:
    normalized_map = {normalize_column_name(column): column for column in columns}
    detected: dict[str, str | None] = {}

    for field, aliases in DEFAULT_COLUMN_ALIASES.items():
        detected[field] = None
        for alias in aliases:
            candidate = normalized_map.get(normalize_column_name(alias))
            if candidate is not None:
                detected[field] = candidate
                break

    lesion_columns = [
        column
        for column in columns
        if any(
            token in normalize_column_name(column)
            for token in ["lesion", "morphology", "concept", "attribute"]
        )
    ]
    detected["lesion_concepts"] = lesion_columns
    return detected


def choose_label_column(detected: dict[str, Any], preferred: str | None = None) -> str | None:
    if preferred:
        return preferred
    for key in ("main_class", "subclass", "diagnosis"):
        value = detected.get(key)
        if value:
            return value
    return None


def resolve_device(requested: str = "auto") -> torch.device:
    if torch is None:
        raise ModuleNotFoundError(
            "torch is required for training, evaluation, and inference. "
            "Install project dependencies with: pip install -r requirements.txt"
        )
    requested = (requested or "auto").lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
