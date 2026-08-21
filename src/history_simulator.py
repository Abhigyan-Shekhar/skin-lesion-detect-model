from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Keyword rules: question_id -> list of regex patterns on subclass/diagnosis text.
SUBCLASS_ANSWER_RULES: dict[str, list[str]] = {
    "fever": [r"fever", r"pyrexia", r"chills"],
    "rapidly_spreading_redness": [r"cellulitis", r"erysipelas", r"spreading"],
    "ring_shaped": [r"tinea", r"ringworm", r"dermatophyt"],
    "scaling_at_border": [r"tinea", r"pityriasis"],
    "itching_worse_at_night": [r"scabies"],
    "family_members_itch": [r"scabies"],
    "rapid_change_size_shape_color": [r"melanoma", r"carcinoma", r"malignan", r"neoplasm"],
    "asymmetry": [r"melanoma", r"dysplastic"],
    "irregular_border": [r"melanoma", r"basal cell"],
    "bleeding_or_crusting_without_injury": [r"melanoma", r"carcinoma", r"ulcer"],
    "honey_colored_crusting": [r"impetigo", r"pyoderma"],
    "pus_or_oozing": [r"impetigo", r"furuncle", r"abscess", r"bacterial"],
    "warm_to_touch": [r"cellulitis", r"furuncle", r"carbuncle"],
}


def _text_fields(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ("Sub_class", "subclass", "diagnosis", "Diagnosis", "Main_class", "main_class"):
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]).lower())
    return " ".join(parts)


def simulate_answers_from_row(row: pd.Series) -> dict[str, str]:
    text = _text_fields(row)
    answers: dict[str, str] = {}
    for question_id, patterns in SUBCLASS_ANSWER_RULES.items():
        if any(re.search(pat, text, re.IGNORECASE) for pat in patterns):
            answers[question_id] = "yes"
    return answers


def simulate_answers_batch(df: pd.DataFrame) -> list[dict[str, str]]:
    return [simulate_answers_from_row(row) for _, row in df.iterrows()]


def simulation_coverage(df: pd.DataFrame) -> dict[str, Any]:
    answers_list = simulate_answers_batch(df)
    with_answers = sum(1 for a in answers_list if a)
    return {
        "total_rows": len(df),
        "rows_with_simulated_answers": with_answers,
        "coverage_fraction": round(with_answers / max(len(df), 1), 4),
    }
