from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import normalize_column_name, read_yaml

COARSE_CLASS_NAMES = ("infectious", "non_urgent_dermatologic", "referral_urgent")

DEFAULT_MAIN_CLASS_TO_COARSE: dict[str, str] = {
    "Infectious Disorders": "infectious",
    "Inflammatory Disorders": "non_urgent_dermatologic",
    "Pigmentary Disorders": "non_urgent_dermatologic",
    "Skin Appendages Disorders": "non_urgent_dermatologic",
    "Keratanisation Disorders": "non_urgent_dermatologic",
    "Other skin disorders": "non_urgent_dermatologic",
    "Neoplasms and tumors": "referral_urgent",
    "No Definite Diagnosis": "referral_urgent",
}

MAIN_CLASS_ALIASES: dict[str, str] = {
    "infectious disorders": "Infectious Disorders",
    "inflammatory disorders": "Inflammatory Disorders",
    "pigmentary disorders": "Pigmentary Disorders",
    "skin appendages disorders": "Skin Appendages Disorders",
    "keratanisation disorders": "Keratanisation Disorders",
    "keratinisation disorders": "Keratanisation Disorders",
    "other skin disorders": "Other skin disorders",
    "neoplasms and tumors": "Neoplasms and tumors",
    "neoplasms and tumours": "Neoplasms and tumors",
    "no definite diagnosis": "No Definite Diagnosis",
}

COARSE_CLASS_ALIASES: dict[str, str] = {
    normalize_column_name(name): name for name in COARSE_CLASS_NAMES
}


def load_class_map(config_path: str | Path | None = None) -> dict[str, str]:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "class_map_3way.yaml"
    path = Path(config_path)
    if not path.exists():
        return dict(DEFAULT_MAIN_CLASS_TO_COARSE)
    payload = read_yaml(path)
    mapping = payload.get("main_class_to_coarse", payload)
    if not isinstance(mapping, dict):
        raise ValueError(f"Invalid class map in {path}")
    return {str(key): str(value) for key, value in mapping.items()}


def load_class_loss_multipliers(config_path: str | Path | None = None) -> dict[str, float]:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "class_map_3way.yaml"
    path = Path(config_path)
    if not path.exists():
        return {"infectious": 1.2, "non_urgent_dermatologic": 1.0, "referral_urgent": 3.0}
    payload = read_yaml(path)
    multipliers = payload.get("class_loss_multipliers", {})
    return {str(k): float(v) for k, v in multipliers.items()}


def canonical_main_class(label: str) -> str:
    normalized = normalize_column_name(str(label))
    if normalized in MAIN_CLASS_ALIASES:
        return MAIN_CLASS_ALIASES[normalized]
    return str(label).strip()


def map_main_class_to_coarse(label: str, mapping: dict[str, str] | None = None) -> str | None:
    mapping = mapping or DEFAULT_MAIN_CLASS_TO_COARSE
    canonical = canonical_main_class(label)
    if canonical in mapping:
        return mapping[canonical]
    normalized = normalize_column_name(canonical)
    for key, value in mapping.items():
        if normalize_column_name(key) == normalized:
            return value
    return None


def apply_coarse_label(
    df: pd.DataFrame,
    source_col: str,
    target_col: str = "coarse_class",
    mapping: dict[str, str] | None = None,
    drop_unmapped: bool = True,
) -> pd.DataFrame:
    mapping = mapping or load_class_map()
    out = df.copy()
    out[target_col] = out[source_col].astype(str).map(lambda value: map_main_class_to_coarse(value, mapping))
    if drop_unmapped:
        out = out[out[target_col].notna()].reset_index(drop=True)
    return out


def coarse_label_distribution(df: pd.DataFrame, label_col: str = "coarse_class") -> dict[str, int]:
    return {str(k): int(v) for k, v in df[label_col].value_counts().items()}
