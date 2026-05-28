from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from label_groups import COARSE_CLASS_NAMES

COARSE_QUESTION_BANKS: dict[str, list[dict[str, Any]]] = {
    "infectious": [
        {"id": "fever", "text": "Any fever?", "positive_weight": 0.18, "red_flag": True},
        {"id": "rapidly_spreading_redness", "text": "Is redness spreading rapidly?", "positive_weight": 0.20},
        {"id": "ring_shaped", "text": "Is it ring-shaped?", "positive_weight": 0.18},
        {"id": "honey_colored_crusting", "text": "Honey-colored crusting?", "positive_weight": 0.16},
        {"id": "pus_or_oozing", "text": "Pus or oozing?", "positive_weight": 0.16},
    ],
    "non_urgent_dermatologic": [
        {"id": "itching_main_symptom", "text": "Is itching the main symptom?", "positive_weight": 0.14},
        {"id": "scaling_plaques", "text": "Scaling plaques?", "positive_weight": 0.14},
        {"id": "new_contact_exposure", "text": "New soap/cosmetic exposure?", "positive_weight": 0.14},
    ],
    "referral_urgent": [
        {"id": "rapid_change_size_shape_color", "text": "Recent change in size/shape/color?", "positive_weight": 0.24, "red_flag": True},
        {"id": "asymmetry", "text": "Asymmetric appearance?", "positive_weight": 0.20},
        {"id": "irregular_border", "text": "Irregular border?", "positive_weight": 0.20},
        {"id": "bleeding_or_crusting_without_injury", "text": "Bleeding/crusting without injury?", "positive_weight": 0.18, "red_flag": True},
    ],
}


def initial_coarse_scores(image_probs: dict[str, float]) -> dict[str, float]:
    return {name: float(image_probs.get(name, 0.0)) for name in COARSE_CLASS_NAMES}


def score_coarse_history(
    answers: dict[str, Any],
    prior_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    scores = dict(prior_scores or {name: 1.0 / len(COARSE_CLASS_NAMES) for name in COARSE_CLASS_NAMES})
    for coarse_name, questions in COARSE_QUESTION_BANKS.items():
        for question in questions:
            qid = question["id"]
            raw = answers.get(qid)
            if raw is None:
                continue
            positive = str(raw).strip().lower() in {"yes", "y", "true", "1", "positive"}
            weight = float(question.get("positive_weight", 0.1))
            if positive:
                scores[coarse_name] += weight
            else:
                scores[coarse_name] = max(0.0, scores[coarse_name] - weight * 0.3)
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}


def fuse_image_and_history(
    image_probs: dict[str, float],
    history_scores: dict[str, float],
    alpha: float = 0.6,
) -> dict[str, float]:
    fused: dict[str, float] = {}
    for name in COARSE_CLASS_NAMES:
        fused[name] = alpha * float(image_probs.get(name, 0.0)) + (1.0 - alpha) * float(
            history_scores.get(name, 0.0)
        )
    total = sum(fused.values()) or 1.0
    return {k: v / total for k, v in fused.items()}


def predict_coarse_label(
    image_probs: dict[str, float],
    answers: dict[str, Any] | None = None,
    alpha: float = 0.6,
) -> tuple[str, dict[str, float]]:
    if not answers:
        best = max(image_probs, key=image_probs.get)
        return best, image_probs
    history = score_coarse_history(answers)
    fused = fuse_image_and_history(image_probs, history, alpha=alpha)
    best = max(fused, key=fused.get)
    return best, fused


def tune_alpha(
    image_prob_rows: Sequence[dict[str, float]],
    answer_rows: Sequence[dict[str, Any]],
    y_true: Sequence[str],
    class_names: Sequence[str],
    alpha_grid: Sequence[float] | None = None,
) -> tuple[float, float]:
    from metrics import recall_first_selection_score

    if alpha_grid is None:
        alpha_grid = [0.4, 0.5, 0.6, 0.7, 0.8]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y_idx = [class_to_idx[y] for y in y_true]
    best_alpha = 0.6
    best_score = -1.0
    for alpha in alpha_grid:
        preds: list[int] = []
        for probs, answers in zip(image_prob_rows, answer_rows):
            label, _ = predict_coarse_label(probs, answers, alpha=alpha)
            preds.append(class_to_idx[label])
        score = recall_first_selection_score(y_idx, preds, class_names)
        if score > best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha, best_score
