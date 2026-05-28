# Change Log — V3 Recall-First 3-Class Pipeline

## 2026-05-27

### Overview

V3 refactors the DermaCon-IN main-class task from **8 fine labels** to **3 triage buckets**, optimizes for **recall/sensitivity** (especially `referral_urgent`), benchmarks multiple CNN backbones, and measures whether **patient history** improves predictions beyond image-only.

Primary entry point: [`notebooks/train_main_class_v3_colab.ipynb`](../notebooks/train_main_class_v3_colab.ipynb).

Do **not** overwrite V2 artifacts; V3 writes under `outputs_v3/` (and `MyDrive/.../outputs_v3/` in Colab).

---

### What changed vs V2

| Area | V2 | V3 |
|------|----|----|
| Labels | 8 `Main_class` values | 3 `coarse_class` values (triage-urgency) |
| Checkpoint | Validation macro F1 | `recall_first_score` (macro recall + urgent recall) |
| Loss | Balanced CE + sampler | + per-class multipliers (urgent ×3), optional focal loss |
| Inference | Argmax + fixed uncertainty margins | Per-class thresholds tuned on validation |
| History | Question engine only (not fused into class head) | `coarse_fusion.py` + A/B/C lift metrics |
| Backbones | EffNet-B0 vs ConvNeXt-Tiny | + ResNet50, Swin-Tiny; comparison CSV |

### Three-class mapping (triage-urgency)

Defined in [`configs/class_map_3way.yaml`](../configs/class_map_3way.yaml) and [`src/label_groups.py`](../src/label_groups.py):

| `coarse_class` | Source `Main_class` values |
|----------------|----------------------------|
| `infectious` | Infectious Disorders |
| `non_urgent_dermatologic` | Inflammatory, Pigmentary, Skin Appendages, Keratanisation, Other |
| `referral_urgent` | Neoplasms and tumors, No Definite Diagnosis |

Edit the YAML to change grouping without code changes.

### Checkpoint criterion

```text
recall_first_score = 0.5 * macro_recall + 0.5 * referral_urgent_recall
```

Early stopping and best-checkpoint selection use this score when `checkpoint_metric: recall_first` (default in V3 config).

---

## Instructions — Google Colab (recommended)

### Prerequisites

1. Colab runtime: **GPU** (T4 or better).
2. DermaCon-IN data on Drive or downloadable via Dataverse (`doi:10.7910/DVN/W7OUZM`).
3. Suggested Drive layout:

```text
MyDrive/derm-opd-triage/
  data/raw/METADATA/Skin_Metadata.csv
  data/raw/DATASET/*.jpg
  outputs_v3/          # created by the notebook
```

### Steps

1. Open [`notebooks/train_main_class_v3_colab.ipynb`](../notebooks/train_main_class_v3_colab.ipynb) in Colab.
2. Run cells in order (sections 1–10).
3. In **section 2**, adjust flags if needed:

```python
OUTPUT_ROOT = 'outputs_v3'
RUN_H2O_BENCHMARK = False          # set True to run H2O AutoML (needs Java + h2o pip)
BACKBONES_TO_TRAIN = ['convnext_tiny', 'resnet50', 'efficientnet_b0']  # shorten for a quick run
```

4. **Section 5** builds coarse labels and patient-level splits (`data/splits/*.csv` with `coarse_class` column).
5. **Section 7** trains each backbone and writes `outputs_v3/benchmark_comparison.csv`.
6. **Section 8** runs fusion experiments A/B/C and writes `outputs_v3/model_<best>/metrics/fusion_lift.json`.
7. **Section 10** prints a summary and disclaimer.

### Expected outputs

```text
outputs_v3/
  metrics/coarse_label_distribution.json
  benchmark_comparison.csv
  best_model.json
  model_convnext_tiny/          # example
    checkpoints/best.pt
    metrics/temperature.json
    metrics/per_class_thresholds.json
    metrics/metrics_test.json
    metrics/fusion_lift.json
    metrics/fusion_comparison.csv
    plots/confusion_matrix_counts_test.png
    predictions/predictions_top3_test.csv
```

### Quick smoke run (one backbone)

In the notebook, set:

```python
BACKBONES_TO_TRAIN = ['convnext_tiny']
```

Reduce epochs in [`configs/backbone_benchmark_3way.yaml`](../configs/backbone_benchmark_3way.yaml) if needed (`epochs_head`, `epochs_finetune`).

---

## Instructions — local / CLI

Run from the repository root with `PYTHONPATH=src` or `cd src` as noted.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Place metadata and images under `data/raw/` (see [docs/colab_training.md](../docs/colab_training.md)).

### 3. Build 3-class splits

```bash
python src/prepare_splits.py \
  --metadata data/raw/METADATA/Skin_Metadata.csv \
  --output_dir data/splits \
  --coarse-labels \
  --class-map configs/class_map_3way.yaml
```

Verify `coarse_class` appears in `data/splits/train.csv`, `val.csv`, and `test.csv`.

### 4. Train + benchmark backbones (CLI)

```bash
PYTHONPATH=src python src/benchmark_backbones.py \
  --config configs/backbone_benchmark_3way.yaml
```

Ensure `image_dir` in the config points at your image folder (default: `data/raw/DATASET`).

### 5. Train a single model programmatically

```python
import sys
sys.path.insert(0, "src")

from recall_training import train_v3, evaluate_full_v3
from utils import read_yaml

cfg = read_yaml("configs/backbone_benchmark_3way.yaml")
cfg["model_name"] = "convnext_tiny"
cfg["image_dir"] = "data/raw/DATASET"

bundle = train_v3(cfg, "outputs_v3/model_convnext_tiny", cfg["train_csv"], cfg["val_csv"], "coarse_class")
```

### 6. Fusion evaluation (after training)

```python
import pandas as pd
from torch.utils.data import DataLoader
from dataset import DermatologyDataset, build_transforms
from evaluate_fusion import run_fusion_experiments

test_df = pd.read_csv("data/splits/test.csv")
test_ds = DermatologyDataset(
    "data/splits/test.csv", cfg["image_dir"],
    bundle["image_column"], bundle["label_column"], bundle["class_to_idx"],
    transform=build_transforms(224, train=False),
)
loader = DataLoader(test_ds, batch_size=32, shuffle=False)
lift = run_fusion_experiments(
    bundle["model"], loader, test_df, bundle["class_names"],
    bundle["device"], bundle["temperature"],
    "outputs_v3/model_convnext_tiny",
)
print(lift)
```

### 7. Inference with coarse fusion (dual-model CLI)

```bash
PYTHONPATH=src python src/unified_inference.py \
  --image path/to/lesion.jpg \
  --opd_checkpoint outputs_v3/model_convnext_tiny/checkpoints/best.pt \
  --lesion_checkpoint outputs/ham10000/checkpoints/best.pt \
  --coarse_fusion \
  --fusion_alpha 0.6 \
  --answers '{"fever": "yes"}'
```

### 8. Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

After scoring adaptive questions, the UI shows **image-only** vs **fused 3-class** triage labels when top-k predictions are available.

---

## Configuration reference

| File | Purpose |
|------|---------|
| [`configs/class_map_3way.yaml`](../configs/class_map_3way.yaml) | Main_class → coarse_class map and loss multipliers |
| [`configs/backbone_benchmark_3way.yaml`](../configs/backbone_benchmark_3way.yaml) | Training hyperparameters and model list |

Key YAML keys:

```yaml
checkpoint_metric: recall_first   # or macro_f1 for legacy behavior in src/train.py
label_column: coarse_class
use_focal_loss: false             # set true to enable focal loss
class_loss_multipliers:           # in class_map_3way.yaml
  referral_urgent: 3.0
```

---

## Fusion experiments (A / B / C)

Reported in `fusion_lift.json`:

| Experiment | Description |
|------------|-------------|
| **A** | Image-only (calibrated softmax argmax or thresholds) |
| **B** | Image + **oracle** history from metadata keyword rules (`history_simulator.py`) |
| **C** | Image + **partial** history (same simulator; empty answers when no rules match) |

Key metrics:

- `delta_macro_recall_B_minus_A`
- `delta_urgent_recall_B_minus_A`
- `simulation_coverage` — fraction of test rows with at least one simulated answer

---

## Optional H2O AutoML

1. In the Colab notebook, set `RUN_H2O_BENCHMARK = True`.
2. Section 9 extracts embeddings + metadata and runs `H2OAutoML`.
3. Compare test recall to the best CNN row in `benchmark_comparison.csv`.

Requires the `h2o` package and sufficient Colab RAM/Java heap.

---

## New files

- `src/label_groups.py` — Main_class → coarse_class mapping
- `src/thresholds.py` — Per-class recall-oriented thresholds
- `src/recall_training.py` — `train_v3()`, calibration, evaluation
- `src/coarse_fusion.py` — Image + history fusion for 3 classes
- `src/history_simulator.py` — Metadata → simulated questionnaire answers
- `src/evaluate_fusion.py` — Experiments A/B/C and `fusion_lift.json`
- `src/extract_embeddings.py` — Backbone embeddings for H2O
- `src/benchmark_backbones.py` — CLI backbone comparison
- `configs/class_map_3way.yaml`
- `configs/backbone_benchmark_3way.yaml`
- `notebooks/train_main_class_v3_colab.ipynb`

## Modified files

- `src/prepare_splits.py` — `--coarse-labels`
- `src/metrics.py` — Macro recall, recall-first selection score
- `src/train.py` — Optional `checkpoint_metric: recall_first`
- `src/evaluate.py` — Threshold-aware evaluation
- `src/question_engine.py` — Coarse/main-class → OPD candidate aliases
- `src/unified_inference.py` — `--coarse_fusion`
- `app/streamlit_app.py` — Image vs fused 3-class display
- `docs/colab_training.md`, `docs/research_plan.md`, `model_card/README.md`

---

## Trade-offs and safety

- **Precision and macro F1 may decrease** — acceptable when optimizing for sensitivity.
- **`referral_urgent` is rare (~89 images)** — treat metrics with caution; use fusion and thresholds to reduce false negatives.
- **Oracle history (B) upper-bounds** realistic adaptive flow (C).
- **Research only** — not for diagnosis, treatment, or clinical deployment. Doctor review required.

---

## Troubleshooting

| Issue | What to do |
|-------|------------|
| `Cannot detect image column` | Check `Skin_Metadata.csv` columns; see `src/inspect_metadata.py` |
| `Unseen labels in val split` | Regenerate splits with `--coarse-labels` after updating `class_map_3way.yaml` |
| OOM on Colab | Set `BACKBONES_TO_TRAIN = ['convnext_tiny']`, lower `batch_size` in config |
| Empty `referral_urgent` in test split | Check `coarse_label_distribution.json`; re-run splits with different `--seed` if needed |
| Fusion lift near zero | Low `simulation_coverage` — extend rules in `history_simulator.py` |

For V2 (8-class) instructions, see [v2-pipeline.md](v2-pipeline.md).
