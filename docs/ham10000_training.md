# HAM10000 / ISIC-Style Dermoscopy Workflow

This is a parallel workflow for dermoscopy lesion classification. It does not replace the DermaCon-IN OPD triage model.

## Scope

Use this path when the input is a dermoscopy or web-style pigmented lesion image and the expected labels are HAM10000-style lesion classes:

- `akiec`: Actinic keratoses and intraepithelial carcinoma
- `bcc`: Basal cell carcinoma
- `bkl`: Benign keratosis-like lesions
- `df`: Dermatofibroma
- `mel`: Melanoma
- `nv`: Melanocytic nevi
- `vasc`: Vascular lesions

This is still research-only. It is not a medical device and must not be used for autonomous diagnosis.

## Dataset

Recommended dataset:

```text
HAM10000: Human Against Machine with 10000 training images
```

Common Kaggle layout after download/unzip:

```text
data/ham10000/raw/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
│   └── ISIC_*.jpg
└── HAM10000_images_part_2/
    └── ISIC_*.jpg
```

HAM10000 contains dermoscopy images and `dx` labels. It is not the same task as DermaCon-IN OPD broad-category classification.

## Prepare Splits

```bash
python src/prepare_ham10000.py \
  --metadata data/ham10000/raw/HAM10000_metadata.csv \
  --image_dirs data/ham10000/raw/HAM10000_images_part_1 data/ham10000/raw/HAM10000_images_part_2 \
  --output_dir data/ham10000/splits
```

## Train EfficientNet-B0

```bash
python src/train.py \
  --config configs/efficientnet_b0_ham10000.yaml
```

## Evaluate

```bash
python src/evaluate.py \
  --checkpoint outputs_ham10000/checkpoints/best.pt \
  --split data/ham10000/splits/test.csv \
  --output_dir outputs_ham10000
```

## Inference

```bash
python src/inference.py \
  --checkpoint outputs_ham10000/checkpoints/best.pt \
  --image test_lesion.jpg \
  --top_k 5
```

## Important Limitation

This workflow uses image-level stratified splits by default. HAM10000 includes a `lesion_id` column; for stricter leakage prevention, lesion-level grouped splitting should be used before reporting final research metrics.
