# derm-opd-triage

Research-only prototype for:

`Image-Guided Adaptive History Taking for Dermatology OPD Triage in Indian Skin Disorders`

This repository is not a clinical product. It must not be used for autonomous diagnosis, treatment decisions, or clinical deployment.

## Warnings

- Research use only
- Non-commercial use only
- Not for medical diagnosis
- Not for clinical deployment
- Doctor review required

## Milestone 1

Implemented in this scaffold:

- Metadata inspection with automatic column detection
- Split preparation with official-split priority and patient-level fallback
- EfficientNet-B0 fine-tuning for broad class prediction
- Evaluation with top-k metrics and confusion matrix
- Inference script for top-k predictions with disclaimer output

## Milestone 2

Implemented:

- Rule-based adaptive OPD-style question engine
- Category score adjustment from model top-k plus patient answers
- Red-flag detection and urgency level assignment
- Doctor-facing structured summary generator

The question engine does not produce a final diagnosis. It produces a provisional differential and review priority for clinician interpretation.

## Milestone 3

Current focus:

- Colab training workflow for EfficientNet-B0
- Drive-based DermaCon-IN dataset layout
- Training, evaluation, and artifact export

Notebook:

- `notebooks/train_effnet_b0_colab.ipynb`

Supporting guide:

- `docs/colab_training.md`

Later demo work:

- Streamlit research demo with disclaimer gate
- Patient intake form
- Image upload and preview
- Optional checkpoint inference
- Manual prediction fallback for workflow testing before training
- Adaptive question collection
- Doctor-facing summary view
- JSON and TXT export
- Hugging Face model card draft

Train the model first. The Streamlit demo should use `outputs/checkpoints/best.pt` after Colab training.

## Train On Colab

Use the notebook:

```text
notebooks/train_effnet_b0_colab.ipynb
```

Dataset source:

```text
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/W7OUZM
```

The notebook uses DOI `10.7910/DVN/W7OUZM` and includes a Dataverse API download step:

```bash
python src/download_dataverse.py \
  --persistent_id doi:10.7910/DVN/W7OUZM \
  --output_dir data/raw
```

Recommended Drive layout:

```text
MyDrive/derm-opd-triage/
└── data/
    └── raw/
        ├── DATASET/
        │   └── *.jpg
        ├── METADATA/
        │   └── Skin_Metadata.csv
        ├── train_split.csv
        └── test_split.csv
```

In Colab:

1. Set runtime to GPU.
2. Mount Google Drive.
3. Clone this repo.
4. Install requirements.
5. Link Drive data into `data/raw`.
6. Run metadata inspection.
7. Prepare splits.
8. Train EfficientNet-B0.
9. Evaluate `outputs/checkpoints/best.pt`.
10. Copy outputs back to Drive.

## Optional HAM10000 / ISIC-Style Dermoscopy Model

This is a separate workflow for dermoscopy-style lesion labels such as `mel`, `nv`, `bcc`, `bkl`, `akiec`, `df`, and `vasc`. It does not replace the DermaCon-IN OPD triage model.

Use:

- `notebooks/train_ham10000_colab.ipynb`
- `configs/efficientnet_b0_ham10000.yaml`
- `src/prepare_ham10000.py`
- `docs/ham10000_training.md`

Expected Drive layout for Colab:

```text
MyDrive/derm-opd-triage-ham10000/
└── data/
    └── raw/
        ├── HAM10000_metadata.csv
        ├── HAM10000_images_part_1/
        │   └── ISIC_*.jpg
        └── HAM10000_images_part_2/
            └── ISIC_*.jpg
```

Prepare splits:

```bash
python src/prepare_ham10000.py \
  --metadata data/ham10000/raw/HAM10000_metadata.csv \
  --image_dirs data/ham10000/raw/HAM10000_images_part_1 data/ham10000/raw/HAM10000_images_part_2 \
  --output_dir data/ham10000/splits
```

Train:

```bash
python src/train.py \
  --config configs/efficientnet_b0_ham10000.yaml
```

Evaluate:

```bash
python src/evaluate.py \
  --checkpoint outputs_ham10000/checkpoints/best.pt \
  --split data/ham10000/splits/test.csv \
  --output_dir outputs_ham10000
```

## Expected data layout

Place the dataset like this:

```text
data/raw/
├── DATASET/
│   └── *.jpg
└── METADATA/
    └── Skin_Metadata.csv
```

Optional official split files may also be present:

- `data/raw/train_split.csv`
- `data/raw/test_split.csv`

## Setup

```bash
pip install -r requirements.txt
```

## Inspect metadata

```bash
python src/inspect_metadata.py \
  --metadata data/raw/METADATA/Skin_Metadata.csv \
  --image_dir data/raw/DATASET
```

If automatic column detection is ambiguous, the script prints all available columns and candidate mappings so you can pass explicit overrides in later steps.

## Prepare splits

```bash
python src/prepare_splits.py \
  --metadata data/raw/METADATA/Skin_Metadata.csv \
  --output_dir data/splits
```

Optional explicit split files:

```bash
python src/prepare_splits.py \
  --metadata data/raw/METADATA/Skin_Metadata.csv \
  --train_split data/raw/train_split.csv \
  --test_split data/raw/test_split.csv \
  --output_dir data/splits
```

## Train EfficientNet-B0

```bash
python src/train.py \
  --config configs/efficientnet_b0.yaml
```

Optional label override if auto-detection is wrong:

```bash
python src/train.py \
  --config configs/efficientnet_b0.yaml \
  --label_column main_class
```

## Evaluate

```bash
python src/evaluate.py \
  --checkpoint outputs/checkpoints/best.pt \
  --split data/splits/test.csv
```

## Run inference

```bash
python src/inference.py \
  --checkpoint outputs/checkpoints/best.pt \
  --image path/to/image.jpg
```

## Run adaptive question engine

Use model predictions from `src/inference.py` or provide equivalent JSON:

```bash
python src/question_engine.py \
  --predictions '{"top_predictions":[{"label":"tinea corporis","probability":0.42},{"label":"eczema","probability":0.21}]}' \
  --answers '{"ring_shaped":true,"scaling_border":true,"steroid_combination_cream":true,"fever_or_chills":false}' \
  --output outputs/predictions/question_engine_output.json
```

The output includes:

- adaptive questions
- updated broad-category differential
- red flags
- urgency level
- doctor review priority
- research-only disclaimer

## Generate doctor-facing summary

```bash
python src/summary_generator.py \
  --patient '{"patient_id":"P001","age":34,"sex":"female","region":"Karnataka","occupation":"teacher","education":"graduate","chief_complaint":"itchy circular rash for two weeks"}' \
  --predictions '{"top_predictions":[{"label":"tinea corporis","probability":0.42},{"label":"eczema","probability":0.21}],"confidence_level":"low"}' \
  --engine_output outputs/predictions/question_engine_output.json \
  --answers '{"ring_shaped":true,"scaling_border":true,"steroid_combination_cream":true,"fever_or_chills":false}' \
  --output_json outputs/predictions/opd_summary.json \
  --output_txt outputs/predictions/opd_summary.txt
```

## Launch Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

The app shows a required research-only disclaimer before the workflow. For a trained model, use the default checkpoint path `outputs/checkpoints/best.pt` or enter another checkpoint path in the app.

## Publish to Hugging Face

Review and complete before publishing:

- `model_card/README.md`
- dataset license obligations in `LICENSE_NOTICE.md`
- trained checkpoint and evaluation metrics under `outputs/`
- any redistribution constraints from DermaCon-IN / Harvard Dataverse

## Notes on leakage prevention

If a patient identifier exists, splitting is done at patient level so a patient does not appear in more than one split. If no patient identifier is available, the split script falls back to stratified image-level splitting and prints a warning.
