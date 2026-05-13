---
license: cc-by-nc-sa-4.0
tags:
- dermatology
- medical-ai
- image-classification
- efficientnet
- research-only
- non-commercial
datasets:
- DermaCon-IN
---

# dermacon-in-efficientnet-b0-opd-triage

## Disclaimer

This model is for non-commercial research use only. It is not a medical device. It must not be used for autonomous diagnosis, treatment decisions, or clinical deployment. All outputs require review by a qualified clinician.

## Model Details

- Model name: `dermacon-in-efficientnet-b0-opd-triage`
- Base architecture: `EfficientNet-B0`
- Initialization: ImageNet pretrained weights
- Intended milestone: broad dermatology main-class prediction for research triage support
- Interface: optional Streamlit research demo with adaptive question engine and doctor-facing summary

## Intended Use

- Research on image-guided adaptive dermatology OPD intake
- Educational experimentation with top-k differential support
- Doctor-facing prototype development with explicit clinician oversight

## Out-of-Scope Use

- Autonomous diagnosis
- Treatment recommendation
- Clinical deployment
- Emergency triage without clinician review
- Commercial use

## Dataset

- DermaCon-IN Dataset Release v1.0
- Harvard Dataverse
- DOI: `10.7910/DVN/W7OUZM`
- License: `CC BY-NC-SA 4.0`

## License and Restrictions

- Non-commercial research use only
- Respect dataset attribution and share-alike requirements
- Verify redistribution terms before publishing weights

## Training Procedure

Milestone 1 procedure:

- Task: broad main-class prediction
- Backbone: ImageNet-pretrained EfficientNet-B0
- Phase 1: frozen backbone, train classifier head
- Phase 2: partial fine-tuning with lower learning rate

Fill in the final training run details before publishing weights:

- Dataset version and access date:
- Train / validation / test split method:
- Number of classes:
- Number of images:
- Training date:
- Checkpoint hash:

## Evaluation Metrics

- Top-1 accuracy
- Top-3 accuracy
- Top-5 accuracy
- Macro F1
- Weighted F1
- Balanced accuracy
- Per-class precision, recall, and F1

Fill in final metrics before publishing:

| Metric | Value |
| --- | --- |
| Top-1 accuracy | TBD |
| Top-3 accuracy | TBD |
| Top-5 accuracy | TBD |
| Macro F1 | TBD |
| Weighted F1 | TBD |
| Balanced accuracy | TBD |

## Limitations

- Dataset-specific performance may not generalize outside the source distribution
- Image quality, framing, lighting, and occlusion can degrade performance
- Label imbalance can bias predictions toward frequent classes
- Top-k output is not equivalent to diagnosis

## Ethical Risks

- Medical overreliance risk if warnings are ignored
- Potential underperformance across skin tones, age groups, and body sites
- Risk of harmful misuse in clinical or commercial settings

## Safety Warnings

- Research use only
- Not for medical diagnosis
- Not for clinical deployment
- Doctor review required
- No treatment recommendation should be inferred

## Skin Tone Bias Warning

Model behavior may vary across Fitzpatrick skin types, Monk skin tones, acquisition devices, and local imaging conditions. Subgroup evaluation is required before drawing conclusions.

## Citation

Please cite the DermaCon-IN dataset source and this project appropriately in any derivative research output.

Dataset DOI: `10.7910/DVN/W7OUZM`
