from __future__ import annotations

import argparse
import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DATAVERSE_SERVER = "https://dataverse.harvard.edu"
DEFAULT_PERSISTENT_ID = "doi:10.7910/DVN/W7OUZM"


def request_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def dataset_api_url(persistent_id: str) -> str:
    query = urllib.parse.urlencode({"persistentId": persistent_id})
    return f"{DATAVERSE_SERVER}/api/datasets/:persistentId?{query}"


def download_file(file_id: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{DATAVERSE_SERVER}/api/access/datafile/{file_id}"
    with urllib.request.urlopen(url, timeout=120) as response:
        with open(destination, "wb") as handle:
            shutil.copyfileobj(response, handle)


def get_files(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    latest = metadata["data"]["latestVersion"]
    return latest.get("files", [])


def local_name(file_entry: dict[str, Any]) -> Path:
    data_file = file_entry["dataFile"]
    filename = data_file["filename"]
    directory = file_entry.get("directoryLabel") or data_file.get("directoryLabel")
    if directory:
        return Path(directory) / filename
    return Path(filename)


def wanted_file(file_entry: dict[str, Any], include_code: bool) -> bool:
    path = str(local_name(file_entry)).lower()
    filename = file_entry["dataFile"]["filename"].lower()
    content_type = file_entry["dataFile"].get("contentType", "").lower()

    if include_code:
        return True
    if filename.endswith((".jpg", ".jpeg", ".png", ".csv", ".tab", ".tsv")):
        return True
    if "image" in content_type or "csv" in content_type or "tab-separated" in content_type:
        return True
    if path.startswith("metadata/"):
        return True
    return False


def normalize_downloaded_layout(raw_dir: Path) -> None:
    metadata_dir = raw_dir / "METADATA"
    dataset_dir = raw_dir / "DATASET"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for file_path in raw_dir.rglob("*"):
        if not file_path.is_file():
            continue
        lower_name = file_path.name.lower()
        if lower_name in {"skin_metadata.tab", "skin_metadata.tsv", "skin_metadata.csv"}:
            target = metadata_dir / "Skin_Metadata.csv"
            if file_path != target:
                shutil.copy2(file_path, target)
        elif lower_name in {"train_split.tab", "train_split.tsv", "train_split.csv"}:
            target = raw_dir / "train_split.csv"
            if file_path != target:
                shutil.copy2(file_path, target)
        elif lower_name in {"test_split.tab", "test_split.tsv", "test_split.csv"}:
            target = raw_dir / "test_split.csv"
            if file_path != target:
                shutil.copy2(file_path, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download DermaCon-IN files from Harvard Dataverse.")
    parser.add_argument("--persistent_id", default=DEFAULT_PERSISTENT_ID)
    parser.add_argument("--output_dir", default="data/raw")
    parser.add_argument("--metadata_only", action="store_true", help="Only save Dataverse metadata JSON")
    parser.add_argument("--include_code", action="store_true", help="Also download code files from Dataverse")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = request_json(dataset_api_url(args.persistent_id))
    metadata_path = output_dir / "dataverse_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved Dataverse metadata to {metadata_path}")

    files = get_files(metadata)
    print(f"Dataset files in latest version: {len(files)}")
    if args.metadata_only:
        for entry in files:
            data_file = entry["dataFile"]
            print(f"{data_file['id']}\t{local_name(entry)}\t{data_file.get('filesize', '')}")
        return

    downloaded = 0
    skipped = 0
    for entry in files:
        if not wanted_file(entry, include_code=args.include_code):
            skipped += 1
            continue
        data_file = entry["dataFile"]
        destination = output_dir / local_name(entry)
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            print(f"Downloading {data_file['id']} -> {destination}")
            download_file(int(data_file["id"]), destination)
            downloaded += 1
        except urllib.error.HTTPError as exc:
            print(f"WARNING: failed to download {data_file['id']} ({destination}): {exc}")

    normalize_downloaded_layout(output_dir)
    print(f"Downloaded: {downloaded}; skipped: {skipped}")
    print(f"Expected metadata path: {output_dir / 'METADATA' / 'Skin_Metadata.csv'}")
    print(f"Expected image directory: {output_dir / 'DATASET'}")


if __name__ == "__main__":
    main()
