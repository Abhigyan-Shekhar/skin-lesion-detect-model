from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from utils import choose_label_column, detect_columns, ensure_dir, read_table, set_seed, write_json


def load_split_file(split_path: Path, metadata: pd.DataFrame, image_column: str) -> pd.DataFrame:
    split_df = read_table(split_path)
    candidate_columns = [column for column in split_df.columns if column in metadata.columns]
    if image_column not in split_df.columns and image_column not in candidate_columns:
        raise ValueError(
            f"Split file {split_path} does not include the detected image column '{image_column}'."
        )
    join_column = image_column if image_column in split_df.columns else candidate_columns[0]
    merged = metadata.merge(split_df[[join_column]], left_on=image_column, right_on=join_column, how="inner")
    return merged[metadata.columns]


def patient_level_split(
    df: pd.DataFrame,
    patient_column: str,
    label_column: str | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df[patient_column].fillna("missing_patient")
    train_idx, temp_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed).split(df, groups=groups)
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    temp_groups = temp_df[patient_column].fillna("missing_patient")
    val_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1).split(
            temp_df, groups=temp_groups
        )
    )
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def patient_level_train_val_split(
    df: pd.DataFrame,
    patient_column: str,
    seed: int,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = df[patient_column].fillna("missing_patient")
    train_idx, val_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed).split(df, groups=groups)
    )
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def image_level_split(
    df: pd.DataFrame,
    label_column: str | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify = df[label_column] if label_column and df[label_column].nunique() > 1 else None
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=stratify
    )
    temp_stratify = (
        temp_df[label_column] if label_column and temp_df[label_column].nunique() > 1 else None
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed + 1, stratify=temp_stratify
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train/val/test splits.")
    parser.add_argument("--metadata", required=True, help="Path to Skin_Metadata.csv")
    parser.add_argument("--output_dir", default="data/splits", help="Output directory for split CSVs")
    parser.add_argument("--train_split", default=None, help="Optional official train split CSV")
    parser.add_argument("--test_split", default=None, help="Optional official test split CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--label_column", default=None, help="Optional explicit label column")
    return parser.parse_args()


def discover_official_split(metadata_path: Path, filename: str) -> Path | None:
    candidates = [
        metadata_path.parent / filename,
        metadata_path.parent.parent / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = read_table(metadata_path)
    detected = detect_columns(df.columns.tolist())
    image_column = detected.get("image")
    if not image_column:
        raise ValueError(
            "Could not detect image filename column automatically. "
            f"Available columns: {df.columns.tolist()}"
        )

    label_column = choose_label_column(detected, args.label_column)
    if label_column is None:
        print("WARNING: No label column was detected automatically. Split will not be stratified.")

    output_dir = ensure_dir(args.output_dir)
    train_split = Path(args.train_split) if args.train_split else discover_official_split(metadata_path, "train_split.csv")
    test_split = Path(args.test_split) if args.test_split else discover_official_split(metadata_path, "test_split.csv")
    patient_column = detected.get("patient_id")

    if train_split and test_split:
        train_base = load_split_file(train_split, df, image_column)
        test_df = load_split_file(test_split, df, image_column)
        if patient_column:
            train_df, val_df = patient_level_train_val_split(
                train_base,
                patient_column=patient_column,
                seed=args.seed,
                val_fraction=0.1765,
            )
            train_base_patient_ids = set(train_base[patient_column].fillna("missing_patient").tolist())
            official_test_patient_ids = set(test_df[patient_column].fillna("missing_patient").tolist())
            overlap = train_base_patient_ids & official_test_patient_ids
            if overlap:
                print(
                    "WARNING: Official train/test files appear to share patient IDs. "
                    "This may indicate leakage in the provided split files."
                )
        else:
            train_stratify = (
                train_base[label_column] if label_column and train_base[label_column].nunique() > 1 else None
            )
            train_df, val_df = train_test_split(
                train_base, test_size=0.1765, random_state=args.seed, stratify=train_stratify
            )
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        split_method = "official_train_test_plus_val_from_train"
    else:
        if patient_column:
            train_df, val_df, test_df = patient_level_split(df, patient_column, label_column, args.seed)
            split_method = "patient_level_group_split"
        else:
            print(
                "WARNING: No patient ID column detected. Falling back to stratified image-level split. "
                "This may introduce patient leakage if multiple images per patient exist."
            )
            train_df, val_df, test_df = image_level_split(df, label_column, args.seed)
            split_method = "image_level_stratified_split"

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "split_method": split_method,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "label_column": label_column,
        "detected_columns": detected,
    }
    write_json(output_dir / "split_summary.json", summary)

    print("\n=== Split Summary ===")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print(f"\nSaved splits to: {output_dir}")


if __name__ == "__main__":
    main()
