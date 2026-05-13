# Colab Training Guide

This is the current priority path for the project: train EfficientNet-B0 on Colab, evaluate it, and export artifacts.

## Runtime

Use Google Colab with:

- Runtime type: `Python 3`
- Hardware accelerator: `T4 GPU` or better

## Dataset Source

DermaCon-IN Dataset Release v1.0:

- URL: `https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/W7OUZM`
- DOI: `10.7910/DVN/W7OUZM`
- License: `CC BY-NC-SA 4.0`

The Dataverse web page may require JavaScript/browser verification. In Colab, use the Dataverse API downloader:

```bash
python src/download_dataverse.py \
  --persistent_id doi:10.7910/DVN/W7OUZM \
  --output_dir data/raw
```

To inspect the file list without downloading files:

```bash
python src/download_dataverse.py \
  --persistent_id doi:10.7910/DVN/W7OUZM \
  --output_dir data/raw \
  --metadata_only
```

If large image downloads fail or Dataverse throttles access, use the manual Drive layout below.

## GitHub Access In Colab

The notebook clones this repo into the Colab runtime. If the repo is private, Colab cannot clone it with the plain HTTPS URL and you will see an error like:

```text
fatal: could not read Username for 'https://github.com': No such device or address
```

Fix one of these ways:

- Make `https://github.com/Abhigyan-Shekhar/skin-lesion-detect-model` public while training.
- Or add a Colab Secret named `GITHUB_TOKEN` with read access to the repo, then rerun the clone cell.

The notebook creates `MyDrive/derm-opd-triage` automatically. Drive does not need to contain anything before the first run.

## Manual Dataset Layout In Google Drive

Create this folder in Drive:

```text
MyDrive/derm-opd-triage/
└── data/
    └── raw/
        ├── DATASET/
        │   └── *.jpg
        ├── METADATA/
        │   └── Skin_Metadata.csv
        ├── train_split.csv      # optional
        └── test_split.csv       # optional
```

Do not commit dataset files to GitHub. DermaCon-IN is licensed `CC BY-NC-SA 4.0`; keep usage non-commercial and research-only.

## Fast Colab Commands

After cloning the repo and mounting Drive:

```bash
pip install -r requirements.txt
```

Download from Dataverse into Colab runtime or Drive:

```bash
python src/download_dataverse.py \
  --persistent_id doi:10.7910/DVN/W7OUZM \
  --output_dir data/raw
```

Inspect metadata:

```bash
python src/inspect_metadata.py \
  --metadata data/raw/METADATA/Skin_Metadata.csv \
  --image_dir data/raw/DATASET
```

Prepare patient-safe splits:

```bash
python src/prepare_splits.py \
  --metadata data/raw/METADATA/Skin_Metadata.csv \
  --output_dir data/splits
```

Train EfficientNet-B0:

```bash
python src/train.py \
  --config configs/efficientnet_b0.yaml
```

Evaluate:

```bash
python src/evaluate.py \
  --checkpoint outputs/checkpoints/best.pt \
  --split data/splits/test.csv
```

## Outputs To Save

Copy these back to Drive after training:

- `outputs/checkpoints/best.pt`
- `outputs/checkpoints/latest.pt`
- `outputs/metrics/training_summary.json`
- `outputs/metrics/train_history.csv`
- `outputs/metrics/metrics.json`
- `outputs/metrics/classification_report.txt`
- `outputs/metrics/per_class_metrics.csv`
- `outputs/plots/confusion_matrix.png`
- `outputs/predictions/predictions.csv`

## Safety

This model is research-only. It is not for diagnosis, treatment, commercial use, or clinical deployment. All outputs require clinician review.
