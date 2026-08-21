from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from question_engine import run_engine
from utils import DISCLAIMER_TEXT


STOP_URGENCY_LEVELS = {"emergency", "urgent dermatologist review"}


def _answered_questions(answers: dict[str, Any]) -> set[str]:
    return {key for key, value in answers.items() if value not in (None, "")}


def _uncertainty_from_differential(differential: list[dict[str, Any]]) -> float:
    scores = [max(float(row.get("score", 0.0)), 0.0) for row in differential]
    total = sum(scores)
    if total <= 0:
        return 1.0
    probs = [score / total for score in scores if score > 0]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    max_entropy = math.log(max(len(probs), 2))
    return round(entropy / max_entropy, 4)


def _top_margin(differential: list[dict[str, Any]]) -> float:
    if len(differential) < 2:
        return 1.0
    return round(float(differential[0].get("score", 0.0)) - float(differential[1].get("score", 0.0)), 4)


def _question_utility(question: dict[str, Any], uncertainty: float, top_margin: float) -> float:
    if "priority_score" in question:
        base = float(question["priority_score"])
    else:
        base = (
            float(question.get("clinical_relevance", 1)) * 2.0
            + float(question.get("discrimination_power", 1)) * 3.0
            + float(question.get("red_flag_importance", 0)) * 4.0
        )
        base += float(question.get("positive_weight", 0.08)) * 12.0
    safety_value = float(question.get("red_flag_weight", question.get("red_flag_importance", 0)))
    patient_burden = float(question.get("patient_burden", 1.0))
    information_gain = base * (0.5 + uncertainty) * (1.0 + max(0.0, 0.25 - top_margin))
    return round(information_gain + 3.0 * safety_value - 1.5 * patient_burden, 4)


def build_belief_state(
    predictions_payload: dict[str, Any] | list[dict[str, Any]],
    answers: dict[str, Any] | None = None,
    patient_context: dict[str, Any] | None = None,
    max_questions: int = 8,
    uncertainty_threshold: float = 0.34,
    stability_margin: float = 0.35,
    min_useful_utility: float = 1.0,
    image_quality: dict[str, Any] | None = None,
    modality: str | None = None,
) -> dict[str, Any]:
    answers = answers or {}
    engine_output = run_engine(
        predictions_payload,
        answers=answers,
        max_category_questions=max(12, max_questions * 2),
        patient_context=patient_context,
    )
    scoring = engine_output["scoring"]
    differential = scoring.get("updated_differential", [])
    uncertainty = _uncertainty_from_differential(differential)
    top_margin = _top_margin(differential)
    asked = _answered_questions(answers)

    question_rows: list[dict[str, Any]] = []
    for question in engine_output.get("questions", []):
        if question["id"] in asked:
            continue
        copied = deepcopy(question)
        copied["utility"] = _question_utility(copied, uncertainty, top_margin)
        question_rows.append(copied)
    question_rows.sort(key=lambda item: item["utility"], reverse=True)

    stop_reasons: list[str] = []
    if image_quality and not image_quality.get("accepted", True):
        stop_reasons.append("abstain_image_quality")
    if modality in {"unknown", "unsupported"}:
        stop_reasons.append("abstain_unsupported_modality")
    if scoring.get("urgency_level") in STOP_URGENCY_LEVELS and scoring.get("red_flags"):
        stop_reasons.append("urgent_red_flag")
    if len(asked) >= max_questions:
        stop_reasons.append("max_question_budget")
    if asked and uncertainty <= uncertainty_threshold:
        stop_reasons.append("uncertainty_threshold")
    if len(asked) >= 2 and top_margin >= stability_margin:
        stop_reasons.append("stable_differential")
    if question_rows and question_rows[0]["utility"] < min_useful_utility:
        stop_reasons.append("low_expected_utility")
    if not question_rows:
        stop_reasons.append("no_remaining_questions")

    should_continue = not stop_reasons
    return {
        "image_model_predictions": predictions_payload,
        "patient_demographics": patient_context or {},
        "lesion_characteristics": {
            key: (patient_context or {}).get(key)
            for key in [
                "body_part_affected",
                "lesion_color",
                "lesion_shape",
                "lesion_border",
                "lesion_size",
                "lesion_pattern",
                "lesion_pigmentation",
                "lesion_surface_change",
                "lesion_count",
            ]
        },
        "previous_answers": answers,
        "positive_findings": scoring.get("key_positive_answers", []),
        "negative_findings": scoring.get("key_negative_answers", []),
        "red_flags": scoring.get("red_flags", []),
        "current_differential": differential,
        "uncertainty": uncertainty,
        "top_differential_margin": top_margin,
        "questions_already_asked": sorted(asked),
        "question_budget": max_questions,
        "remaining_question_count": len(question_rows),
        "next_question": question_rows[0] if should_continue and question_rows else None,
        "candidate_questions": question_rows[:5],
        "should_continue": should_continue,
        "stop_reasons": stop_reasons,
        "engine_output": engine_output,
        "disclaimer": DISCLAIMER_TEXT,
    }
