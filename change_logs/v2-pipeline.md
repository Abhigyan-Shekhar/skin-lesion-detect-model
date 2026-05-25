# Change Log

## 2026-05-25 — Improved Main-Class Pipeline V2

### Overview

Added `notebooks/train_main_class_v2_colab.ipynb`, a new Colab notebook that upgrades the existing `train_effnet_b0_colab.ipynb` pipeline in four ordered steps. No existing `src/` files were modified.

---

### New File

**`notebooks/train_main_class_v2_colab.ipynb`**

Self-contained Colab notebook. All new logic is defined inline; shared utilities are imported from `src/`.

---

### Changes vs Baseline (`train_effnet_b0_colab.ipynb`)

#### 1. Imbalance Fix

- Replaced `DataLoader(shuffle=True)` with `WeightedRandomSampler` built from inverse class frequencies.
- Each sample receives weight `1 / class_count[class]`, so minority classes are sampled at the same expected rate as majority classes within each epoch.
- Added `label_smoothing=0.1` to the class-weighted `CrossEntropyLoss` to reduce overconfidence during training.

#### 2. Stronger Backbone

- Added **ConvNeXt-Tiny** as a second backbone candidate alongside EfficientNet-B0.
- Both backbones are trained under the identical recipe (same sampler, loss, LR schedule, early stopping) to enable a controlled comparison.
- Backbone is already supported in `src/models.py:69`; no model code changes required.
- Separate output directories: `outputs_v2_effnet/` and `outputs_v2_convnext/`.

#### 3. Post-Training Calibration

- Added `TemperatureScaler` — a single learnable scalar `T` that divides logits before softmax.
- `calibrate_temperature()` fits `T` on **validation logits only** using LBFGS to minimise NLL.
- Fitted temperature saved to `{output_dir}/metrics/temperature.json`.
- All downstream evaluation and inference uses `softmax(logits / T)`.

#### 4. Calibrated Top-3 Inference and Evaluation

**Inference contract changed** from raw top-k to calibrated top-3 shortlist:

| Field | Description |
|---|---|
| `top_predictions` | Top-3 labels sorted by calibrated probability |
| `confidence_level` | `high` / `moderate` / `low` based on top-1 calibrated prob |
| `uncertainty_flag` | `true` when top-1 prob < 0.45 or top-1/top-2 margin < 0.10 |
| `top1_top2_margin` | Calibrated probability difference between rank-1 and rank-2 |
| `temperature` | The fitted `T` applied at inference time |

**New evaluation outputs** per backbone:

| Artifact | Path |
|---|---|
| Raw-count confusion matrix | `plots/confusion_matrix_counts_{tag}.png` |
| Row-normalised confusion matrix | `plots/confusion_matrix_normalized_{tag}.png` |
| ROC curves (one-vs-rest per class + macro avg) | `plots/roc_curves_{tag}.png` |
| Per-class metrics with ROC-AUC | `metrics/per_class_metrics_{tag}.csv` |
| Calibration summary (NLL before/after) | `metrics/calibration_summary_{tag}.json` |
| All metrics including macro/weighted ROC-AUC | `metrics/metrics_{tag}.json` |
| Top-3 predictions with uncertainty flag | `predictions/predictions_top3_{tag}.csv` |
| Fitted temperature | `metrics/temperature.json` |

**Model selection guidance** (printed in comparison cell):
- Primary: `macro_f1` + `balanced_accuracy`
- Supporting: `top3_accuracy`, `macro_roc_auc`
- Do not select on top-1 accuracy alone
- Tie-break: lower `calibrated_nll`

---

### Helper Functions Added (inline in notebook)

| Function | Purpose |
|---|---|
| `build_weighted_sampler(train_dataset)` | Builds `WeightedRandomSampler` from inverse class frequencies |
| `make_loss_v2(train_dataset, device, label_smoothing)` | Class-weighted CE with label smoothing |
| `TemperatureScaler` | `nn.Module` with single temperature parameter |
| `collect_val_logits(model, val_loader, device)` | Collects raw logits from val set, no gradients |
| `calibrate_temperature(logits, targets, output_dir)` | Fits temperature via LBFGS on val logits |
| `evaluate_full(...)` | Full test evaluation: all metrics, both CMs, ROC curves, top-3 CSV |
| `predict_top3_calibrated(bundle, temperature, image_path)` | Calibrated top-3 inference with uncertainty flag |
| `train_v2(config, output_dir, ...)` | Two-phase training loop with sampler + label smoothing |

---

### Files Changed

| File | Change |
|---|---|
| `notebooks/train_main_class_v2_colab.ipynb` | **Created** — 42-cell Colab notebook |
| `change_logs/README.md` | **Created** — this file |

No existing files were modified.
