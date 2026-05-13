from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import ensure_dir, read_table, set_seed, write_json


HAM10000_LABELS = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def find_image_path(image_id: str, image_dirs: list[Path]) -> str | None:
    for image_dir in image_dirs:
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = image_dir / f"{image_id}{suffix}"
            if candidate.exists():
                return str(candidate)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HAM10000/ISIC-style splits.")
    parser.add_argument("--metadata", required=True, help="Path to HAM10000_metadata.csv")
    parser.add_argument(
        "--image_dirs",
        nargs="+",
        required=True,
        help="One or more image folders, e.g. HAM10000_images_part_1 HAM10000_images_part_2",
    )
    parser.add_argument("--output_dir", default="data/ham10000/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drop_missing_images",
        action="store_true",
        help="Drop metadata rows whose image file cannot be found.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    metadata_path = Path(args.metadata)
    image_dirs = [Path(path) for path in args.image_dirs]
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    missing_dirs = [str(path) for path in image_dirs if not path.exists()]
    if missing_dirs:
        raise FileNotFoundError(f"Image directories not found: {missing_dirs}")

    df = read_table(metadata_path)
    required_columns = {"image_id", "dx"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"HAM10000 metadata missing columns: {missing_columns}")

    df = df.copy()
    df["label_full"] = df["dx"].map(HAM10000_LABELS).fillna(df["dx"])
    df["image_path"] = df["image_id"].astype(str).map(lambda value: find_image_path(value, image_dirs))
    missing_images = int(df["image_path"].isna().sum())
    if missing_images:
        message = f"Missing image files for {missing_images} rows."
        if args.drop_missing_images:
            print(f"WARNING: {message} Dropping those rows.")
            df = df[df["image_path"].notna()].reset_index(drop=True)
        else:
            raise FileNotFoundError(f"{message} Pass --drop_missing_images to continue anyway.")

    # Reuse the generic Dataset by making image_id an absolute image path column.
    df["image_id"] = df["image_path"]
    df = df[df["dx"].notna()].reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=args.seed,
        stratify=df["dx"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=args.seed + 1,
        stratify=temp_df["dx"],
    )

    output_dir = ensure_dir(args.output_dir)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "dataset": "HAM10000",
        "metadata": str(metadata_path),
        "image_dirs": [str(path) for path in image_dirs],
        "split_method": "stratified_image_level_split",
        "warning": "HAM10000 has lesion_id; this script currently stratifies by image. Use lesion-level grouping for stricter leakage prevention.",
        "label_column": "dx",
        "image_column": "image_id",
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "missing_images": missing_images,
        "class_distribution": df["dx"].value_counts().to_dict(),
        "label_map": HAM10000_LABELS,
    }
    write_json(output_dir / "split_summary.json", summary)

    print("\n=== HAM10000 Split Summary ===")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print(f"\nSaved splits to: {output_dir}")


if __name__ == "__main__":
    main()
