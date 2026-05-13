from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import detect_columns, ensure_dir, read_table, write_json


def build_summary(metadata_path: Path, image_dir: Path) -> dict:
    df = read_table(metadata_path)
    detected = detect_columns(df.columns.tolist())
    summary: dict = {
        "metadata_path": str(metadata_path),
        "image_dir": str(image_dir),
        "num_rows": int(len(df)),
        "columns": df.columns.tolist(),
        "detected_columns": detected,
        "missing_percentage": {
            column: round(float(df[column].isna().mean() * 100), 2) for column in df.columns
        },
    }

    image_column = detected.get("image")
    if image_column:
        paths = df[image_column].astype(str).map(lambda value: image_dir / value)
        exists = paths.map(Path.exists)
        summary["num_images"] = int(paths.nunique())
        summary["num_existing_images"] = int(exists.sum())
        summary["num_missing_images"] = int((~exists).sum())
        summary["missing_image_examples"] = [
            str(path) for path in paths[~exists].head(10).tolist()
        ]
    else:
        summary["num_images"] = None
        summary["num_existing_images"] = None
        summary["num_missing_images"] = None
        summary["missing_image_examples"] = []

    for label_key in ("diagnosis", "main_class", "subclass"):
        column = detected.get(label_key)
        if column:
            summary[f"{label_key}_distribution"] = df[column].fillna("MISSING").value_counts().to_dict()
        else:
            summary[f"{label_key}_distribution"] = None

    patient_column = detected.get("patient_id")
    if patient_column:
        summary["num_unique_patients"] = int(df[patient_column].nunique(dropna=True))
    else:
        summary["num_unique_patients"] = None

    return summary


def print_summary(summary: dict) -> None:
    print("\n=== Metadata Inspection ===")
    print(f"Rows: {summary['num_rows']}")
    print(f"Image directory: {summary['image_dir']}")

    print("\nAvailable columns:")
    for column in summary["columns"]:
        print(f"- {column}")

    print("\nDetected columns:")
    for key, value in summary["detected_columns"].items():
        if key == "lesion_concepts":
            print(f"- {key}: {value}")
        else:
            print(f"- {key}: {value}")

    print("\nMissing value percentage:")
    for column, percent in summary["missing_percentage"].items():
        print(f"- {column}: {percent}%")

    if summary["num_unique_patients"] is not None:
        print(f"\nUnique patients: {summary['num_unique_patients']}")

    if summary["num_images"] is not None:
        print(f"Unique image references: {summary['num_images']}")
        print(f"Existing image files: {summary['num_existing_images']}")
        print(f"Missing image files: {summary['num_missing_images']}")

    for key in ("diagnosis", "main_class", "subclass"):
        distribution = summary.get(f"{key}_distribution")
        if distribution:
            print(f"\nLabel distribution for {key}:")
            for label, count in distribution.items():
                print(f"- {label}: {count}")

    if not summary["detected_columns"].get("image"):
        print(
            "\nWARNING: Could not confidently detect the image filename column. "
            "Review the printed columns and pass an explicit mapping in later steps."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect dermatology dataset metadata.")
    parser.add_argument("--metadata", required=True, help="Path to Skin_Metadata.csv")
    parser.add_argument("--image_dir", required=True, help="Path to DATASET image folder")
    parser.add_argument(
        "--output",
        default="outputs/metrics/metadata_summary.json",
        help="Path to save summary JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    image_dir = Path(args.image_dir)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    summary = build_summary(metadata_path, image_dir)
    print_summary(summary)

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    write_json(output_path, summary)
    print(f"\nSaved metadata summary to: {output_path}")


if __name__ == "__main__":
    main()
