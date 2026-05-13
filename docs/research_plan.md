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

### Image model

- accuracy
- top-3 accuracy
- top-5 accuracy
- macro F1
- balanced accuracy
- per-class F1

### Question engine

- simulate cases from metadata where possible
- expert dermatologist review if available
- measure whether questions cover relevant HPI and red flags
- compare static questionnaire vs adaptive questionnaire

## Safety

- no autonomous diagnosis
- no treatment recommendation
- red-flag escalation
- doctor review required
