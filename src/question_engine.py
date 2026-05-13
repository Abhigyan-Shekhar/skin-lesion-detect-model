from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils import DISCLAIMER_TEXT, write_json


QUESTION_BANKS: dict[str, dict[str, Any]] = {
    "superficial_bacterial_infection": {
        "display_name": "Superficial bacterial infection",
        "possible_labels": ["impetigo", "ecthyma", "folliculitis"],
        "questions": [
            {
                "id": "honey_colored_crusting",
                "section": "history_of_presenting_illness",
                "text": "Is there honey-colored crusting?",
                "positive_weight": 0.20,
            },
            {
                "id": "pus_or_oozing",
                "section": "history_of_presenting_illness",
                "text": "Is there pus or oozing?",
                "positive_weight": 0.18,
            },
            {
                "id": "painful_or_itchy",
                "section": "history_of_presenting_illness",
                "text": "Is the lesion painful or itchy?",
                "positive_weight": 0.08,
            },
            {
                "id": "after_bite_scratch_injury",
                "section": "history_of_presenting_illness",
                "text": "Did it start after an insect bite, scratch, or minor injury?",
                "positive_weight": 0.12,
            },
            {
                "id": "spreading_nearby",
                "section": "history_of_presenting_illness",
                "text": "Is it spreading to nearby areas?",
                "positive_weight": 0.10,
            },
            {
                "id": "siblings_family_similar_lesions",
                "section": "family_history",
                "text": "Are there similar lesions in siblings or family members?",
                "positive_weight": 0.08,
            },
            {
                "id": "fever",
                "section": "history_of_presenting_illness",
                "text": "Any fever?",
                "positive_weight": 0.06,
                "red_flag": True,
                "red_flag_weight": 1,
            },
        ],
    },
    "deep_bacterial_infection_urgent": {
        "display_name": "Deep bacterial infection / urgent",
        "possible_labels": ["cellulitis", "erysipelas", "abscess"],
        "questions": [
            {
                "id": "warm_to_touch",
                "section": "history_of_presenting_illness",
                "text": "Is the area warm to touch?",
                "positive_weight": 0.18,
            },
            {
                "id": "pain_or_tenderness",
                "section": "history_of_presenting_illness",
                "text": "Is there pain or tenderness?",
                "positive_weight": 0.14,
            },
            {
                "id": "rapidly_spreading_redness",
                "section": "history_of_presenting_illness",
                "text": "Is redness spreading rapidly?",
                "positive_weight": 0.18,
                "red_flag": True,
                "red_flag_weight": 2,
            },
            {
                "id": "fever_or_chills",
                "section": "history_of_presenting_illness",
                "text": "Do you have fever or chills?",
                "positive_weight": 0.14,
                "red_flag": True,
                "red_flag_weight": 2,
            },
            {
                "id": "swelling",
                "section": "history_of_presenting_illness",
                "text": "Is there swelling?",
                "positive_weight": 0.10,
            },
            {
                "id": "sharply_raised_border",
                "section": "history_of_presenting_illness",
                "text": "Is the border sharply raised?",
                "positive_weight": 0.08,
            },
            {
                "id": "diabetes",
                "section": "personal_history",
                "text": "Do you have diabetes?",
                "positive_weight": 0.08,
                "red_flag": True,
                "red_flag_weight": 1,
            },
            {
                "id": "wound_ulcer_cut_trauma_nearby",
                "section": "past_treatment_history",
                "text": "Any wound, ulcer, cut, or trauma nearby?",
                "positive_weight": 0.10,
            },
            {
                "id": "severe_pain_out_of_proportion",
                "section": "history_of_presenting_illness",
                "text": "Any severe pain out of proportion?",
                "positive_weight": 0.12,
                "red_flag": True,
                "red_flag_weight": 3,
            },
        ],
    },
    "fungal_infection": {
        "display_name": "Fungal infection",
        "possible_labels": ["tinea corporis", "tinea cruris", "candidiasis"],
        "questions": [
            {
                "id": "ring_shaped",
                "section": "history_of_presenting_illness",
                "text": "Is it ring-shaped?",
                "positive_weight": 0.20,
            },
            {
                "id": "scaling_border",
                "section": "history_of_presenting_illness",
                "text": "Is there scaling at the border?",
                "positive_weight": 0.20,
            },
            {
                "id": "itching_worse_sweating",
                "section": "history_of_presenting_illness",
                "text": "Is itching worse with sweating?",
                "positive_weight": 0.14,
            },
            {
                "id": "groin_or_folds",
                "section": "history_of_presenting_illness",
                "text": "Is it in the groin or skin folds?",
                "positive_weight": 0.12,
            },
            {
                "id": "steroid_combination_cream",
                "section": "past_treatment_history",
                "text": "Did you use a steroid combination cream?",
                "positive_weight": 0.16,
            },
            {
                "id": "family_similar_lesions",
                "section": "family_history",
                "text": "Any similar lesions in family members?",
                "positive_weight": 0.08,
            },
            {
                "id": "pets_or_contact_source",
                "section": "personal_history",
                "text": "Any pets or likely contact source?",
                "positive_weight": 0.06,
            },
        ],
    },
    "viral_infection": {
        "display_name": "Viral infection",
        "possible_labels": ["herpes zoster", "herpes simplex", "warts", "molluscum"],
        "questions": [
            {
                "id": "grouped_blisters",
                "section": "history_of_presenting_illness",
                "text": "Are there grouped blisters?",
                "positive_weight": 0.20,
            },
            {
                "id": "burning_pain",
                "section": "history_of_presenting_illness",
                "text": "Is there burning pain?",
                "positive_weight": 0.16,
            },
            {
                "id": "one_sided",
                "section": "history_of_presenting_illness",
                "text": "Is the lesion one-sided?",
                "positive_weight": 0.14,
            },
            {
                "id": "fever_before_rash",
                "section": "history_of_presenting_illness",
                "text": "Any fever before the rash?",
                "positive_weight": 0.10,
            },
            {
                "id": "recurrence",
                "section": "past_treatment_history",
                "text": "Any recurrence?",
                "positive_weight": 0.10,
            },
            {
                "id": "immune_suppression",
                "section": "personal_history",
                "text": "Any immune suppression?",
                "positive_weight": 0.08,
                "red_flag": True,
                "red_flag_weight": 1,
            },
        ],
    },
    "parasitic_infestation": {
        "display_name": "Parasitic / infestation",
        "possible_labels": ["scabies", "lice"],
        "questions": [
            {
                "id": "itching_worse_at_night",
                "section": "history_of_presenting_illness",
                "text": "Is itching worse at night?",
                "positive_weight": 0.24,
            },
            {
                "id": "family_members_itch",
                "section": "family_history",
                "text": "Do family members also itch?",
                "positive_weight": 0.22,
            },
            {
                "id": "classic_scabies_sites",
                "section": "history_of_presenting_illness",
                "text": "Are lesions in finger webs, wrist, waist, or genitals?",
                "positive_weight": 0.20,
            },
            {
                "id": "crowded_living_exposure",
                "section": "personal_history",
                "text": "Any hostel or crowded living exposure?",
                "positive_weight": 0.12,
            },
        ],
    },
    "inflammatory_mimics": {
        "display_name": "Inflammatory mimics",
        "possible_labels": ["eczema", "contact dermatitis", "psoriasis", "urticaria"],
        "questions": [
            {
                "id": "itching_main_symptom",
                "section": "history_of_presenting_illness",
                "text": "Is itching the main symptom?",
                "positive_weight": 0.14,
            },
            {
                "id": "new_contact_exposure",
                "section": "history_of_presenting_illness",
                "text": "Any new soap, cosmetic, detergent, metal, footwear, plant, or occupational exposure?",
                "positive_weight": 0.18,
            },
            {
                "id": "recurrent",
                "section": "past_treatment_history",
                "text": "Is it recurrent?",
                "positive_weight": 0.12,
            },
            {
                "id": "scaling_plaques",
                "section": "history_of_presenting_illness",
                "text": "Any scaling plaques?",
                "positive_weight": 0.14,
            },
            {
                "id": "allergy_or_asthma_history",
                "section": "personal_history",
                "text": "Any history of allergy or asthma?",
                "positive_weight": 0.12,
            },
            {
                "id": "improves_moisturizer_steroid",
                "section": "past_treatment_history",
                "text": "Does it improve with moisturizer or steroid?",
                "positive_weight": 0.10,
            },
        ],
    },
}

GENERAL_QUESTIONS: list[dict[str, str]] = [
    {"id": "onset", "section": "history_of_presenting_illness", "text": "When did it start?"},
    {"id": "duration", "section": "history_of_presenting_illness", "text": "How long has it been present?"},
    {"id": "progression", "section": "history_of_presenting_illness", "text": "Is it improving, worsening, or unchanged?"},
    {"id": "aggravating_factors", "section": "history_of_presenting_illness", "text": "What makes it worse?"},
    {"id": "relieving_factors", "section": "history_of_presenting_illness", "text": "What relieves it?"},
    {"id": "associated_symptoms", "section": "history_of_presenting_illness", "text": "Any associated symptoms?"},
    {"id": "negative_history", "section": "history_of_presenting_illness", "text": "Any symptoms specifically absent that the doctor should know?"},
    {"id": "diet", "section": "personal_history", "text": "Any notable diet changes?"},
    {"id": "sleep", "section": "personal_history", "text": "Any sleep disturbance?"},
    {"id": "appetite", "section": "personal_history", "text": "Any appetite change?"},
    {"id": "hypertension", "section": "personal_history", "text": "Do you have hypertension?"},
    {"id": "substance_use", "section": "personal_history", "text": "Any tobacco, alcohol, or substance use?"},
    {"id": "previous_similar_episode", "section": "past_treatment_history", "text": "Any previous similar episode?"},
    {"id": "previous_hospital_admission", "section": "past_treatment_history", "text": "Any previous hospital admission for this problem?"},
    {"id": "previous_treatment", "section": "past_treatment_history", "text": "Any previous treatment taken?"},
    {"id": "antibiotic_use", "section": "past_treatment_history", "text": "Any antibiotic use?"},
    {"id": "antifungal_use", "section": "past_treatment_history", "text": "Any antifungal use?"},
    {"id": "similar_issue_family", "section": "family_history", "text": "Any similar issue in the family?"},
    {"id": "contagious_spread", "section": "family_history", "text": "Does it seem to be spreading among close contacts?"},
]

GLOBAL_RED_FLAGS: dict[str, dict[str, Any]] = {
    "facial_eye_involvement": {"text": "Facial or eye involvement", "weight": 3},
    "black_discoloration": {"text": "Black discoloration", "weight": 3},
    "confusion": {"text": "Confusion", "weight": 3},
    "low_blood_pressure_symptoms": {"text": "Low blood pressure symptoms", "weight": 3},
    "immunosuppression": {"text": "Immunosuppression", "weight": 2},
}


@dataclass
class Prediction:
    label: str
    probability: float


def normalize_label(value: str) -> str:
    return value.strip().lower().replace("_", " ").replace("-", " ")


def answer_is_positive(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "present", "positive"}:
        return True
    if text in {"no", "n", "false", "0", "absent", "negative"}:
        return False
    return None


def category_for_label(label: str) -> str | None:
    normalized = normalize_label(label)
    for category, bank in QUESTION_BANKS.items():
        if normalized in {normalize_label(item) for item in bank["possible_labels"]}:
            return category
    return None


def parse_predictions(payload: dict[str, Any] | list[dict[str, Any]]) -> list[Prediction]:
    rows = payload.get("top_predictions", payload) if isinstance(payload, dict) else payload
    predictions: list[Prediction] = []
    for row in rows:
        label = str(row["label"])
        probability = float(row.get("probability", row.get("prob", 0.0)))
        predictions.append(Prediction(label=label, probability=probability))
    return predictions


def initial_category_scores(predictions: list[Prediction]) -> dict[str, float]:
    scores = {category: 0.0 for category in QUESTION_BANKS}
    unmatched = []
    for prediction in predictions:
        category = category_for_label(prediction.label)
        if category:
            scores[category] += prediction.probability
        else:
            unmatched.append(prediction.label)

    if unmatched and not any(scores.values()):
        shared = 1.0 / len(scores)
        scores = {category: shared for category in QUESTION_BANKS}
    return scores


def get_adaptive_questions(
    predictions: list[Prediction] | list[dict[str, Any]],
    max_category_questions: int = 12,
    include_general: bool = True,
) -> list[dict[str, Any]]:
    parsed = [
        item if isinstance(item, Prediction) else Prediction(label=str(item["label"]), probability=float(item.get("probability", 0.0)))
        for item in predictions
    ]
    scores = initial_category_scores(parsed)
    ranked_categories = sorted(scores, key=scores.get, reverse=True)

    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    if include_general:
        for question in GENERAL_QUESTIONS:
            copied = deepcopy(question)
            copied["type"] = "general"
            questions.append(copied)
            seen_ids.add(copied["id"])

    for category in ranked_categories:
        if scores[category] <= 0 and any(scores.values()):
            continue
        bank = QUESTION_BANKS[category]
        for question in bank["questions"]:
            if question["id"] in seen_ids:
                continue
            copied = deepcopy(question)
            copied["type"] = "adaptive"
            copied["category"] = category
            copied["category_display_name"] = bank["display_name"]
            questions.append(copied)
            seen_ids.add(copied["id"])
            adaptive_count = sum(1 for item in questions if item.get("type") == "adaptive")
            if adaptive_count >= max_category_questions:
                return questions
    return questions


def score_answers(
    predictions: list[Prediction] | list[dict[str, Any]],
    answers: dict[str, Any],
) -> dict[str, Any]:
    parsed = [
        item if isinstance(item, Prediction) else Prediction(label=str(item["label"]), probability=float(item.get("probability", 0.0)))
        for item in predictions
    ]
    scores = initial_category_scores(parsed)
    red_flags: list[dict[str, Any]] = []
    key_positive: list[str] = []
    key_negative: list[str] = []

    question_lookup = {
        question["id"]: (category, question)
        for category, bank in QUESTION_BANKS.items()
        for question in bank["questions"]
    }

    for answer_id, raw_answer in answers.items():
        positive = answer_is_positive(raw_answer)
        if positive is None:
            continue

        if answer_id in question_lookup:
            category, question = question_lookup[answer_id]
            weight = float(question.get("positive_weight", 0.08))
            if positive:
                scores[category] += weight
                key_positive.append(question["text"])
                if question.get("red_flag"):
                    red_flags.append(
                        {
                            "id": answer_id,
                            "text": question["text"],
                            "weight": int(question.get("red_flag_weight", 1)),
                        }
                    )
            else:
                scores[category] = max(0.0, scores[category] - weight * 0.4)
                key_negative.append(question["text"])

        global_flag = GLOBAL_RED_FLAGS.get(answer_id)
        if positive and global_flag:
            red_flags.append(
                {"id": answer_id, "text": global_flag["text"], "weight": int(global_flag["weight"])}
            )
            key_positive.append(global_flag["text"])

    total = sum(scores.values())
    normalized_scores = (
        {category: score / total for category, score in scores.items()} if total > 0 else scores
    )
    differential = [
        {
            "category": category,
            "display_name": QUESTION_BANKS[category]["display_name"],
            "score": round(score, 4),
            "possible_labels": QUESTION_BANKS[category]["possible_labels"],
        }
        for category, score in sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True)
    ]

    urgency = determine_urgency(red_flags, answers, differential)
    return {
        "updated_differential": differential,
        "urgency_level": urgency["level"],
        "doctor_review_priority": urgency["review_priority"],
        "red_flags": red_flags,
        "key_positive_answers": key_positive,
        "key_negative_answers": key_negative,
        "disclaimer": DISCLAIMER_TEXT,
    }


def determine_urgency(
    red_flags: list[dict[str, Any]],
    answers: dict[str, Any],
    differential: list[dict[str, Any]],
) -> dict[str, str]:
    red_flag_score = sum(int(flag.get("weight", 1)) for flag in red_flags)
    top_category = differential[0]["category"] if differential else None

    has_severe_pain = answer_is_positive(answers.get("severe_pain_out_of_proportion")) is True
    has_black = answer_is_positive(answers.get("black_discoloration")) is True
    has_confusion = answer_is_positive(answers.get("confusion")) is True
    has_rapid_fever_diabetes = all(
        answer_is_positive(answers.get(item)) is True
        for item in ["rapidly_spreading_redness", "fever_or_chills", "diabetes"]
    )

    if has_black or has_confusion or has_severe_pain or red_flag_score >= 6:
        return {
            "level": "emergency",
            "review_priority": "Immediate emergency evaluation by clinician.",
        }
    if has_rapid_fever_diabetes or red_flag_score >= 3 or top_category == "deep_bacterial_infection_urgent":
        return {
            "level": "urgent dermatologist review",
            "review_priority": "Same-day or urgent clinician review recommended.",
        }
    if red_flag_score > 0:
        return {
            "level": "routine dermatologist review",
            "review_priority": "Clinician review required; prioritize if symptoms progress.",
        }
    return {
        "level": "non-urgent / monitor with doctor advice",
        "review_priority": "Routine clinician review required for interpretation.",
    }


def run_engine(
    predictions_payload: dict[str, Any] | list[dict[str, Any]],
    answers: dict[str, Any] | None = None,
    max_category_questions: int = 12,
) -> dict[str, Any]:
    predictions = parse_predictions(predictions_payload)
    questions = get_adaptive_questions(predictions, max_category_questions=max_category_questions)
    scoring = score_answers(predictions, answers or {})
    return {
        "questions": questions,
        "scoring": scoring,
        "disclaimer": DISCLAIMER_TEXT,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive OPD-style question engine.")
    parser.add_argument("--predictions", required=True, help="JSON file or JSON string with top_predictions")
    parser.add_argument("--answers", default=None, help="Optional JSON file or JSON string with answers")
    parser.add_argument("--output", default=None, help="Optional path to save engine output JSON")
    parser.add_argument("--max_category_questions", type=int, default=12)
    return parser.parse_args()


def load_json_arg(value: str) -> Any:
    path = Path(value)
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(value)


def main() -> None:
    args = parse_args()
    predictions = load_json_arg(args.predictions)
    answers = load_json_arg(args.answers) if args.answers else {}
    result = run_engine(
        predictions,
        answers=answers,
        max_category_questions=args.max_category_questions,
    )
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
