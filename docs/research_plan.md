# Research Plan

## Title

Image-Guided Adaptive History Taking for Dermatology OPD Triage in Indian Skin Disorders

## Research Question

Can an image model trained on Indian dermatology data guide adaptive OPD-style history-taking and produce structured doctor-facing triage summaries?

## Hypothesis

Image-guided adaptive questioning can improve triage workflow by focusing history-taking on likely differential diagnoses and red flags.

## Contributions

1. Fine-tuned visual classifier on Indian dermatology dataset.
2. Adaptive OPD-style question engine based on top-k differential.
3. Structured doctor-facing summary generator.
4. Research-only triage workflow for Indian outpatient dermatology.
5. Evaluation of confidence, top-k accuracy, and uncertainty-aware triage.

## Evaluation

### Image model (V3 — 3-class recall-first)

Primary metrics (see `notebooks/train_main_class_v3_colab.ipynb`):

- **macro recall** and **referral_urgent recall** (checkpoint selection)
- recall-first score: `0.5 * macro_recall + 0.5 * referral_urgent_recall`
- per-class precision/recall/F1 on `infectious`, `non_urgent_dermatologic`, `referral_urgent`
- balanced accuracy, macro F1 (reported; expected precision/F1 trade-off)
- per-class probability thresholds tuned on validation for sensitivity

### History fusion (V3)

Experiments A/B/C in `src/evaluate_fusion.py`:

- **A** — image-only 3-class prediction
- **B** — image + oracle simulated history from metadata
- **C** — image + partial simulated history

Report `fusion_lift.json`: delta macro recall and delta urgent recall vs image-only.

### Question engine

- simulate cases from metadata where possible (`src/history_simulator.py`)
- expert dermatologist review if available
- measure whether questions cover relevant HPI and red flags
- compare static questionnaire vs adaptive questionnaire
- coarse-class fusion via `src/coarse_fusion.py`

## Safety

- no autonomous diagnosis
- no treatment recommendation
- red-flag escalation
- doctor review required
