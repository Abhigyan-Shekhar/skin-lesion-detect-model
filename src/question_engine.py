from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils import DISCLAIMER_TEXT, write_json


OPD_QUESTION_BANKS: dict[str, dict[str, Any]] = {
    "superficial_bacterial_infection": {
        "display_name": "Superficial bacterial infection",
        "possible_labels": ["impetigo", "ecthyma", "folliculitis"],
        "questions": [
            {"id": "honey_colored_crusting", "section": "history_of_presenting_illness", "text": "Is there honey-colored crusting?", "positive_weight": 0.20},
            {"id": "pus_or_oozing", "section": "history_of_presenting_illness", "text": "Is there pus or oozing?", "positive_weight": 0.18},
            {"id": "painful_or_itchy", "section": "history_of_presenting_illness", "text": "Is the lesion painful or itchy?", "positive_weight": 0.08},
            {"id": "after_bite_scratch_injury", "section": "history_of_presenting_illness", "text": "Did it start after an insect bite, scratch, or minor injury?", "positive_weight": 0.12},
            {"id": "spreading_nearby", "section": "history_of_presenting_illness", "text": "Is it spreading to nearby areas?", "positive_weight": 0.10},
            {"id": "siblings_family_similar_lesions", "section": "family_history", "text": "Are there similar lesions in siblings or family members?", "positive_weight": 0.08},
            {"id": "fever", "section": "history_of_presenting_illness", "text": "Any fever?", "positive_weight": 0.06, "red_flag": True, "red_flag_weight": 1},
        ],
    },
    "deep_bacterial_infection_urgent": {
        "display_name": "Deep bacterial infection / urgent",
        "possible_labels": ["cellulitis", "erysipelas", "abscess"],
        "questions": [
            {"id": "warm_to_touch", "section": "history_of_presenting_illness", "text": "Is the area warm to touch?", "positive_weight": 0.18},
            {"id": "pain_or_tenderness", "section": "history_of_presenting_illness", "text": "Is there pain or tenderness?", "positive_weight": 0.14},
            {"id": "rapidly_spreading_redness", "section": "history_of_presenting_illness", "text": "Is redness spreading rapidly?", "positive_weight": 0.18, "red_flag": True, "red_flag_weight": 2},
            {"id": "fever_or_chills", "section": "history_of_presenting_illness", "text": "Do you have fever or chills?", "positive_weight": 0.14, "red_flag": True, "red_flag_weight": 2},
            {"id": "swelling", "section": "history_of_presenting_illness", "text": "Is there swelling?", "positive_weight": 0.10},
            {"id": "sharply_raised_border", "section": "history_of_presenting_illness", "text": "Is the border sharply raised?", "positive_weight": 0.08},
            {"id": "diabetes", "section": "personal_history", "text": "Do you have diabetes?", "positive_weight": 0.08, "red_flag": True, "red_flag_weight": 1},
            {"id": "wound_ulcer_cut_trauma_nearby", "section": "past_treatment_history", "text": "Any wound, ulcer, cut, or trauma nearby?", "positive_weight": 0.10},
            {"id": "severe_pain_out_of_proportion", "section": "history_of_presenting_illness", "text": "Any severe pain out of proportion?", "positive_weight": 0.12, "red_flag": True, "red_flag_weight": 3},
        ],
    },
    "fungal_infection": {
        "display_name": "Fungal infection",
        "possible_labels": ["tinea corporis", "tinea cruris", "candidiasis"],
        "questions": [
            {"id": "ring_shaped", "section": "history_of_presenting_illness", "text": "Is it ring-shaped?", "positive_weight": 0.20},
            {"id": "scaling_border", "section": "history_of_presenting_illness", "text": "Is there scaling at the border?", "positive_weight": 0.20},
            {"id": "itching_worse_sweating", "section": "history_of_presenting_illness", "text": "Is itching worse with sweating?", "positive_weight": 0.14},
            {"id": "groin_or_folds", "section": "history_of_presenting_illness", "text": "Is it in the groin or skin folds?", "positive_weight": 0.12},
            {"id": "steroid_combination_cream", "section": "past_treatment_history", "text": "Did you use a steroid combination cream?", "positive_weight": 0.16},
            {"id": "family_similar_lesions", "section": "family_history", "text": "Any similar lesions in family members?", "positive_weight": 0.08},
            {"id": "pets_or_contact_source", "section": "personal_history", "text": "Any pets or likely contact source?", "positive_weight": 0.06},
        ],
    },
    "viral_infection": {
        "display_name": "Viral infection",
        "possible_labels": ["herpes zoster", "herpes simplex", "warts", "molluscum"],
        "questions": [
            {"id": "grouped_blisters", "section": "history_of_presenting_illness", "text": "Are there grouped blisters?", "positive_weight": 0.20},
            {"id": "burning_pain", "section": "history_of_presenting_illness", "text": "Is there burning pain?", "positive_weight": 0.16},
            {"id": "one_sided", "section": "history_of_presenting_illness", "text": "Is the lesion one-sided?", "positive_weight": 0.14},
            {"id": "fever_before_rash", "section": "history_of_presenting_illness", "text": "Any fever before the rash?", "positive_weight": 0.10},
            {"id": "recurrence", "section": "past_treatment_history", "text": "Any recurrence?", "positive_weight": 0.10},
            {"id": "immune_suppression", "section": "personal_history", "text": "Any immune suppression?", "positive_weight": 0.08, "red_flag": True, "red_flag_weight": 1},
        ],
    },
    "parasitic_infestation": {
        "display_name": "Parasitic / infestation",
        "possible_labels": ["scabies", "lice"],
        "questions": [
            {"id": "itching_worse_at_night", "section": "history_of_presenting_illness", "text": "Is itching worse at night?", "positive_weight": 0.24},
            {"id": "family_members_itch", "section": "family_history", "text": "Do family members also itch?", "positive_weight": 0.22},
            {"id": "classic_scabies_sites", "section": "history_of_presenting_illness", "text": "Are lesions in finger webs, wrist, waist, or genitals?", "positive_weight": 0.20},
            {"id": "crowded_living_exposure", "section": "personal_history", "text": "Any hostel or crowded living exposure?", "positive_weight": 0.12},
        ],
    },
    "inflammatory_mimics": {
        "display_name": "Inflammatory mimics",
        "possible_labels": ["eczema", "contact dermatitis", "psoriasis", "urticaria"],
        "questions": [
            {"id": "itching_main_symptom", "section": "history_of_presenting_illness", "text": "Is itching the main symptom?", "positive_weight": 0.14},
            {"id": "new_contact_exposure", "section": "history_of_presenting_illness", "text": "Any new soap, cosmetic, detergent, metal, footwear, plant, or occupational exposure?", "positive_weight": 0.18},
            {"id": "recurrent", "section": "past_treatment_history", "text": "Is it recurrent?", "positive_weight": 0.12},
            {"id": "scaling_plaques", "section": "history_of_presenting_illness", "text": "Any scaling plaques?", "positive_weight": 0.14},
            {"id": "allergy_or_asthma_history", "section": "personal_history", "text": "Any history of allergy or asthma?", "positive_weight": 0.12},
            {"id": "improves_moisturizer_steroid", "section": "past_treatment_history", "text": "Does it improve with moisturizer or steroid?", "positive_weight": 0.10},
        ],
    },
}

HAM10000_QUESTION_BANKS: dict[str, dict[str, Any]] = {
    "mel": {
        "display_name": "Melanoma",
        "possible_labels": ["mel", "melanoma"],
        "questions": [
            {"id": "rapid_change_size_shape_color", "section": "history_of_presenting_illness", "text": "Has it changed in size, shape, or color recently?", "positive_weight": 0.24, "red_flag": True, "red_flag_weight": 2},
            {"id": "asymmetry", "section": "history_of_presenting_illness", "text": "Does it look asymmetric?", "positive_weight": 0.18},
            {"id": "irregular_border", "section": "history_of_presenting_illness", "text": "Does it have an irregular or notched border?", "positive_weight": 0.18},
            {"id": "multiple_colors_dark_areas", "section": "history_of_presenting_illness", "text": "Are there multiple colors or very dark areas in it?", "positive_weight": 0.18},
            {"id": "bleeding_or_crusting_without_injury", "section": "history_of_presenting_illness", "text": "Has it bled or crusted without a clear injury?", "positive_weight": 0.16, "red_flag": True, "red_flag_weight": 2},
            {"id": "new_lesion_in_adulthood", "section": "history_of_presenting_illness", "text": "Did it appear as a new mole or lesion in adulthood?", "positive_weight": 0.10},
            {"id": "family_history_skin_cancer", "section": "family_history", "text": "Any family history of melanoma or skin cancer?", "positive_weight": 0.08},
            {"id": "heavy_sun_exposure_or_sunburns", "section": "personal_history", "text": "Any heavy sun exposure or repeated sunburns?", "positive_weight": 0.06},
        ],
    },
    "nv": {
        "display_name": "Melanocytic nevus",
        "possible_labels": ["nv", "melanocytic nevi", "melanocytic nevus", "nevus", "mole"],
        "questions": [
            {"id": "longstanding_stable_for_years", "section": "history_of_presenting_illness", "text": "Has it been present and largely stable for years?", "positive_weight": 0.24},
            {"id": "uniform_single_color", "section": "history_of_presenting_illness", "text": "Is it mostly a single uniform color?", "positive_weight": 0.18},
            {"id": "symmetric_shape", "section": "history_of_presenting_illness", "text": "Is it fairly symmetric in shape?", "positive_weight": 0.16},
            {"id": "present_since_childhood", "section": "history_of_presenting_illness", "text": "Was it present since childhood or adolescence?", "positive_weight": 0.14},
            {"id": "no_recent_change", "section": "history_of_presenting_illness", "text": "Has there been no important recent change?", "positive_weight": 0.22},
        ],
    },
    "bcc": {
        "display_name": "Basal cell carcinoma",
        "possible_labels": ["bcc", "basal cell carcinoma"],
        "questions": [
            {"id": "pearly_shiny_bump", "section": "history_of_presenting_illness", "text": "Does it look pearly, shiny, or translucent?", "positive_weight": 0.22},
            {"id": "non_healing_sore", "section": "history_of_presenting_illness", "text": "Is it a sore that does not heal properly?", "positive_weight": 0.20, "red_flag": True, "red_flag_weight": 1},
            {"id": "bleeds_with_minor_trauma", "section": "history_of_presenting_illness", "text": "Does it bleed with minor rubbing or trauma?", "positive_weight": 0.16, "red_flag": True, "red_flag_weight": 1},
            {"id": "visible_tiny_blood_vessels", "section": "history_of_presenting_illness", "text": "Are tiny blood vessels visible on the surface?", "positive_weight": 0.16},
            {"id": "sun_exposed_site", "section": "history_of_presenting_illness", "text": "Is it on a sun-exposed area such as the face, scalp, or neck?", "positive_weight": 0.12},
            {"id": "slowly_enlarging_months", "section": "history_of_presenting_illness", "text": "Has it slowly enlarged over months?", "positive_weight": 0.14},
        ],
    },
    "bkl": {
        "display_name": "Benign keratosis-like lesion",
        "possible_labels": ["bkl", "benign keratosis like lesions", "benign keratosis-like lesion", "seborrheic keratosis"],
        "questions": [
            {"id": "rough_warty_surface", "section": "history_of_presenting_illness", "text": "Does it feel rough or warty?", "positive_weight": 0.22},
            {"id": "stuck_on_appearance", "section": "history_of_presenting_illness", "text": "Does it look stuck on to the skin surface?", "positive_weight": 0.24},
            {"id": "flaky_or_scaly_surface", "section": "history_of_presenting_illness", "text": "Is the surface flaky or scaly?", "positive_weight": 0.14},
            {"id": "multiple_similar_spots", "section": "history_of_presenting_illness", "text": "Are there multiple similar spots elsewhere?", "positive_weight": 0.12},
            {"id": "slow_change_without_bleeding", "section": "history_of_presenting_illness", "text": "Has it changed slowly without repeated bleeding?", "positive_weight": 0.10},
        ],
    },
    "akiec": {
        "display_name": "Actinic keratosis / intraepithelial carcinoma",
        "possible_labels": ["akiec", "actinic keratoses", "actinic keratosis", "bowen disease", "intraepithelial carcinoma"],
        "questions": [
            {"id": "rough_scaly_patch", "section": "history_of_presenting_illness", "text": "Is it a rough scaly patch or plaque?", "positive_weight": 0.22},
            {"id": "persistent_crusting", "section": "history_of_presenting_illness", "text": "Is there persistent crusting?", "positive_weight": 0.18, "red_flag": True, "red_flag_weight": 1},
            {"id": "tender_or_painful_lesion", "section": "history_of_presenting_illness", "text": "Is it tender or painful?", "positive_weight": 0.12},
            {"id": "chronically_sun_exposed_area", "section": "history_of_presenting_illness", "text": "Is it on a chronically sun-exposed area?", "positive_weight": 0.14},
            {"id": "non_healing_or_growing_patch", "section": "history_of_presenting_illness", "text": "Is it a non-healing or enlarging patch?", "positive_weight": 0.18, "red_flag": True, "red_flag_weight": 2},
            {"id": "older_age_group", "section": "personal_history", "text": "Is the patient in an older age group?", "positive_weight": 0.06},
        ],
    },
    "df": {
        "display_name": "Dermatofibroma",
        "possible_labels": ["df", "dermatofibroma"],
        "questions": [
            {"id": "firm_raised_bump", "section": "history_of_presenting_illness", "text": "Does it feel like a firm raised bump?", "positive_weight": 0.18},
            {"id": "dimple_sign", "section": "history_of_presenting_illness", "text": "Does it dimple inward when pinched from the sides?", "positive_weight": 0.24},
            {"id": "longstanding_stable_small", "section": "history_of_presenting_illness", "text": "Has it stayed small and stable for a long time?", "positive_weight": 0.18},
            {"id": "on_legs", "section": "history_of_presenting_illness", "text": "Is it on the leg?", "positive_weight": 0.12},
            {"id": "after_insect_bite_or_minor_trauma", "section": "history_of_presenting_illness", "text": "Did it appear after an insect bite or minor trauma?", "positive_weight": 0.08},
        ],
    },
    "vasc": {
        "display_name": "Vascular lesion",
        "possible_labels": ["vasc", "vascular lesion", "angioma", "hemangioma"],
        "questions": [
            {"id": "bright_red_or_purple", "section": "history_of_presenting_illness", "text": "Is it bright red, maroon, or purple?", "positive_weight": 0.22},
            {"id": "changes_with_pressure", "section": "history_of_presenting_illness", "text": "Does it blanch or change a bit with pressure?", "positive_weight": 0.16},
            {"id": "bleeds_easily", "section": "history_of_presenting_illness", "text": "Does it bleed easily?", "positive_weight": 0.16, "red_flag": True, "red_flag_weight": 1},
            {"id": "cluster_of_vascular_spots", "section": "history_of_presenting_illness", "text": "Does it look like a cluster of blood-vessel spots?", "positive_weight": 0.16},
            {"id": "longstanding_without_major_change", "section": "history_of_presenting_illness", "text": "Has it been there a long time without major change?", "positive_weight": 0.10},
        ],
    },
}

OPD_GENERAL_QUESTIONS: list[dict[str, str]] = [
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

HAM10000_GENERAL_QUESTIONS: list[dict[str, str]] = [
    {"id": "lesion_onset", "section": "history_of_presenting_illness", "text": "When was the lesion first noticed?"},
    {"id": "lesion_duration", "section": "history_of_presenting_illness", "text": "How long has this lesion been present?"},
    {"id": "lesion_evolution", "section": "history_of_presenting_illness", "text": "Has the lesion changed over time?"},
    {"id": "itch_or_pain", "section": "history_of_presenting_illness", "text": "Is it itchy, painful, or otherwise symptomatic?"},
    {"id": "lesion_size_change_note", "section": "history_of_presenting_illness", "text": "How has the size changed, if at all?"},
    {"id": "lesion_color_change_note", "section": "history_of_presenting_illness", "text": "How has the color changed, if at all?"},
    {"id": "prior_biopsy_or_treatment", "section": "past_treatment_history", "text": "Any prior biopsy, freezing, laser, or other treatment for this lesion?"},
    {"id": "history_skin_cancer", "section": "past_treatment_history", "text": "Any past history of skin cancer or precancerous lesions?"},
    {"id": "family_history_skin_cancer_note", "section": "family_history", "text": "Any family history of skin cancer or melanoma?"},
    {"id": "immunosuppression_note", "section": "personal_history", "text": "Any immunosuppression or transplant history?"},
    {"id": "occupational_sun_exposure", "section": "personal_history", "text": "Any major occupational or chronic sun exposure?"},
]

OPD_GLOBAL_RED_FLAGS: dict[str, dict[str, Any]] = {
    "facial_eye_involvement": {"text": "Facial or eye involvement", "weight": 3},
    "black_discoloration": {"text": "Black discoloration", "weight": 3},
    "confusion": {"text": "Confusion", "weight": 3},
    "low_blood_pressure_symptoms": {"text": "Low blood pressure symptoms", "weight": 3},
    "immunosuppression": {"text": "Immunosuppression", "weight": 2},
}

HAM10000_GLOBAL_RED_FLAGS: dict[str, dict[str, Any]] = {
    "rapid_change_size_shape_color": {"text": "Recent size/shape/color change", "weight": 2},
    "bleeding_or_crusting_without_injury": {"text": "Bleeding or crusting without injury", "weight": 2},
    "non_healing_sore": {"text": "Non-healing sore", "weight": 1},
    "non_healing_or_growing_patch": {"text": "Non-healing or enlarging patch", "weight": 2},
    "immunosuppression_note": {"text": "Immunosuppression history", "weight": 1},
    "history_skin_cancer": {"text": "Past skin cancer history", "weight": 1},
}

QUESTION_PROFILES: dict[str, dict[str, Any]] = {
    "opd": {
        "display_name": "Broad OPD dermatology triage",
        "question_banks": OPD_QUESTION_BANKS,
        "general_questions": OPD_GENERAL_QUESTIONS,
        "global_red_flags": OPD_GLOBAL_RED_FLAGS,
        "candidate_label": "category",
    },
    "ham10000": {
        "display_name": "Lesion-level dermoscopy triage",
        "question_banks": HAM10000_QUESTION_BANKS,
        "general_questions": HAM10000_GENERAL_QUESTIONS,
        "global_red_flags": HAM10000_GLOBAL_RED_FLAGS,
        "candidate_label": "lesion",
    },
}


LABEL_ALIASES: dict[str, str] = {
    "mel": "Melanoma",
    "melanoma": "Melanoma",
    "nv": "Melanocytic nevus",
    "nevus": "Melanocytic nevus",
    "mole": "Melanocytic nevus",
    "melanocytic nevus": "Melanocytic nevus",
    "melanocytic nevi": "Melanocytic nevus",
    "bcc": "Basal cell carcinoma",
    "basal cell carcinoma": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "seborrheic keratosis": "Benign keratosis-like lesion",
    "benign keratosis like lesions": "Benign keratosis-like lesion",
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "actinic keratosis": "Actinic keratosis / intraepithelial carcinoma",
    "bowen disease": "Actinic keratosis / intraepithelial carcinoma",
    "df": "Dermatofibroma",
    "dermatofibroma": "Dermatofibroma",
    "vasc": "Vascular lesion",
    "vascular lesion": "Vascular lesion",
    "angioma": "Vascular lesion",
    "hemangioma": "Vascular lesion",
    "tinea corporis": "Tinea corporis",
    "tinea capitis": "Tinea capitis",
    "tinea cruris": "Tinea cruris",
    "furuncle": "Furuncle",
    "carbuncle": "Carbuncle",
    "bacterial infection": "Bacterial infection",
    "vitiligo": "Vitiligo",
    "pityriasis versicolor": "Pityriasis versicolor",
    "eczema": "Eczema",
    "contact dermatitis": "Contact dermatitis",
    "psoriasis": "Psoriasis",
}

DISEASE_GROUPS: dict[str, set[str]] = {
    "infective": {
        "Tinea corporis",
        "Tinea capitis",
        "Tinea cruris",
        "Furuncle",
        "Carbuncle",
        "Bacterial infection",
        "Pityriasis versicolor",
    },
    "pigmented": {
        "Melanoma",
        "Melanocytic nevus",
        "Basal cell carcinoma",
        "Benign keratosis-like lesion",
        "Actinic keratosis / intraepithelial carcinoma",
        "Dermatofibroma",
        "Vascular lesion",
    },
    "depigmented": {
        "Vitiligo",
        "Pityriasis versicolor",
    },
    "inflammatory": {
        "Eczema",
        "Contact dermatitis",
        "Psoriasis",
    },
}

DIAGNOSIS_GUIDED_COMMON_QUESTIONS: list[dict[str, Any]] = [
    {"id": "lesion_duration", "section": "history_of_presenting_illness", "text": "Since when is the lesion present?", "clinical_relevance": 3, "discrimination_power": 2, "red_flag_importance": 1, "patient_burden": 1, "groups": ["general"]},
    {"id": "lesion_onset_pattern", "section": "history_of_presenting_illness", "text": "Was the onset sudden or gradual?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 0, "patient_burden": 1, "groups": ["general"]},
    {"id": "lesion_progression", "section": "history_of_presenting_illness", "text": "Is the lesion increasing, decreasing, or stable?", "clinical_relevance": 3, "discrimination_power": 3, "red_flag_importance": 2, "patient_burden": 1, "groups": ["general"]},
    {"id": "color_change", "section": "history_of_presenting_illness", "text": "Has the color changed?", "clinical_relevance": 3, "discrimination_power": 3, "red_flag_importance": 2, "patient_burden": 1, "groups": ["general", "pigmented", "depigmented"]},
    {"id": "shape_change", "section": "history_of_presenting_illness", "text": "Has the shape changed?", "clinical_relevance": 3, "discrimination_power": 3, "red_flag_importance": 2, "patient_burden": 1, "groups": ["general", "pigmented"]},
    {"id": "size_change", "section": "history_of_presenting_illness", "text": "Has the size changed?", "clinical_relevance": 3, "discrimination_power": 3, "red_flag_importance": 2, "patient_burden": 1, "groups": ["general"]},
    {"id": "itching", "section": "history_of_presenting_illness", "text": "Is there itching?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 0, "patient_burden": 1, "groups": ["infective", "inflammatory"]},
    {"id": "pain", "section": "history_of_presenting_illness", "text": "Is there pain?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 2, "patient_burden": 1, "groups": ["infective", "pigmented"]},
    {"id": "burning_sensation", "section": "history_of_presenting_illness", "text": "Is there burning sensation?", "clinical_relevance": 1, "discrimination_power": 1, "red_flag_importance": 0, "patient_burden": 1, "groups": ["infective", "inflammatory"]},
    {"id": "discharge_pus_bleeding_crusting_ulceration", "section": "history_of_presenting_illness", "text": "Is there discharge, pus, bleeding, crusting, or ulceration?", "clinical_relevance": 3, "discrimination_power": 3, "red_flag_importance": 3, "patient_burden": 1, "groups": ["general"]},
    {"id": "single_or_multiple", "section": "history_of_presenting_illness", "text": "Is it present at one site or multiple sites?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 0, "patient_burden": 1, "groups": ["general"]},
    {"id": "fever", "section": "history_of_presenting_illness", "text": "Any fever?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 2, "patient_burden": 1, "groups": ["infective"]},
    {"id": "family_similar_lesions", "section": "family_history", "text": "Any similar lesion in family members?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 0, "patient_burden": 1, "groups": ["infective", "depigmented"]},
    {"id": "previous_similar_episode", "section": "past_treatment_history", "text": "Any previous similar episode?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 0, "patient_burden": 1, "groups": ["general"]},
    {"id": "diabetes", "section": "personal_history", "text": "Any history of diabetes?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 2, "patient_burden": 1, "groups": ["infective"]},
    {"id": "immune_suppression", "section": "personal_history", "text": "Any immune suppression, HIV, steroid use, or chemotherapy?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 2, "patient_burden": 1, "groups": ["general"]},
]

DIAGNOSIS_GUIDED_SPECIFIC_QUESTIONS: list[dict[str, Any]] = [
    {"id": "ring_shaped", "section": "history_of_presenting_illness", "text": "Is the lesion ring-shaped?", "clinical_relevance": 3, "discrimination_power": 3, "red_flag_importance": 0, "patient_burden": 1, "groups": ["infective"], "labels": ["Tinea corporis", "Tinea cruris", "Tinea capitis"]},
    {"id": "central_clearing", "section": "history_of_presenting_illness", "text": "Is there central clearing?", "clinical_relevance": 3, "discrimination_power": 3, "red_flag_importance": 0, "patient_burden": 1, "groups": ["infective"], "labels": ["Tinea corporis", "Tinea cruris"]},
    {"id": "scaling_at_border", "section": "history_of_presenting_illness", "text": "Is there scaling at the border?", "clinical_relevance": 3, "discrimination_power": 3, "red_flag_importance": 0, "patient_burden": 1, "groups": ["infective", "depigmented"], "labels": ["Tinea corporis", "Tinea cruris", "Pityriasis versicolor"]},
    {"id": "spreading_outward", "section": "history_of_presenting_illness", "text": "Is it spreading outward?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 1, "patient_burden": 1, "groups": ["infective"], "labels": ["Tinea corporis", "Tinea cruris", "Tinea capitis"]},
    {"id": "shared_towels_clothes", "section": "personal_history", "text": "Any sharing of towels or clothes?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 0, "patient_burden": 1, "groups": ["infective"], "labels": ["Tinea corporis", "Tinea cruris", "Tinea capitis"]},
    {"id": "excessive_sweating", "section": "personal_history", "text": "Any excessive sweating?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 0, "patient_burden": 1, "groups": ["infective"], "labels": ["Tinea corporis", "Tinea cruris", "Tinea capitis"]},
    {"id": "steroid_cream_use", "section": "past_treatment_history", "text": "Any steroid cream use?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 0, "patient_burden": 1, "groups": ["infective"], "labels": ["Tinea corporis", "Tinea cruris", "Tinea capitis"]},
    {"id": "swelling", "section": "history_of_presenting_illness", "text": "Is there swelling?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 2, "patient_burden": 1, "groups": ["infective"], "labels": ["Furuncle", "Carbuncle", "Bacterial infection"]},
    {"id": "pus_or_discharge", "section": "history_of_presenting_illness", "text": "Is there pus or discharge?", "clinical_relevance": 3, "discrimination_power": 3, "red_flag_importance": 2, "patient_burden": 1, "groups": ["infective"], "labels": ["Furuncle", "Carbuncle", "Bacterial infection"]},
    {"id": "warm_to_touch", "section": "history_of_presenting_illness", "text": "Is the area warm to touch?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 2, "patient_burden": 1, "groups": ["infective"], "labels": ["Furuncle", "Carbuncle", "Bacterial infection"]},
    {"id": "recurrent_boils", "section": "past_treatment_history", "text": "Any recurrent boils?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 1, "patient_burden": 1, "groups": ["infective"], "labels": ["Furuncle", "Carbuncle", "Bacterial infection"]},
    {"id": "recent_trauma_shaving_friction", "section": "past_treatment_history", "text": "Any recent trauma, shaving, or friction?", "clinical_relevance": 1, "discrimination_power": 1, "red_flag_importance": 0, "patient_burden": 1, "groups": ["infective"], "labels": ["Furuncle", "Carbuncle", "Bacterial infection"]},
    {"id": "present_since_childhood_or_recent", "section": "history_of_presenting_illness", "text": "Was it present since childhood or did it appear recently?", "clinical_relevance": 2, "discrimination_power": 3, "red_flag_importance": 1, "patient_burden": 1, "groups": ["pigmented"], "labels": ["Melanoma", "Melanocytic nevus"]},
    {"id": "asymmetry", "section": "history_of_presenting_illness", "text": "Is it asymmetrical?", "clinical_relevance": 3, "discrimination_power": 4, "red_flag_importance": 2, "patient_burden": 1, "groups": ["pigmented"], "labels": ["Melanoma"]},
    {"id": "irregular_border", "section": "history_of_presenting_illness", "text": "Are the borders irregular?", "clinical_relevance": 3, "discrimination_power": 4, "red_flag_importance": 2, "patient_burden": 1, "groups": ["pigmented"], "labels": ["Melanoma"]},
    {"id": "multiple_colors_dark_areas", "section": "history_of_presenting_illness", "text": "Is there color variation or very dark areas?", "clinical_relevance": 3, "discrimination_power": 4, "red_flag_importance": 2, "patient_burden": 1, "groups": ["pigmented"], "labels": ["Melanoma"]},
    {"id": "diameter_gt_6mm", "section": "history_of_presenting_illness", "text": "Is the diameter greater than 6 mm?", "clinical_relevance": 2, "discrimination_power": 3, "red_flag_importance": 1, "patient_burden": 1, "groups": ["pigmented"], "labels": ["Melanoma"]},
    {"id": "bleeding_or_ulceration", "section": "history_of_presenting_illness", "text": "Is there bleeding or ulceration?", "clinical_relevance": 3, "discrimination_power": 3, "red_flag_importance": 3, "patient_burden": 1, "groups": ["pigmented"], "labels": ["Melanoma", "Basal cell carcinoma", "Actinic keratosis / intraepithelial carcinoma"]},
    {"id": "family_history_skin_cancer", "section": "family_history", "text": "Any family history of melanoma or skin cancer?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 1, "patient_burden": 1, "groups": ["pigmented"], "labels": ["Melanoma"]},
    {"id": "excessive_sun_exposure", "section": "personal_history", "text": "Any excessive sun exposure, tanning, or radiation exposure?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 1, "patient_burden": 1, "groups": ["pigmented"], "labels": ["Melanoma", "Basal cell carcinoma", "Actinic keratosis / intraepithelial carcinoma"]},
    {"id": "white_hair_over_patch", "section": "history_of_presenting_illness", "text": "Any white hair over the patch?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 0, "patient_burden": 1, "groups": ["depigmented"], "labels": ["Vitiligo"]},
    {"id": "thyroid_or_autoimmune_disease", "section": "personal_history", "text": "Any thyroid or autoimmune disease?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 0, "patient_burden": 1, "groups": ["depigmented"], "labels": ["Vitiligo"]},
    {"id": "loss_of_sensation", "section": "history_of_presenting_illness", "text": "Is there loss of sensation over the patch?", "clinical_relevance": 3, "discrimination_power": 4, "red_flag_importance": 2, "patient_burden": 1, "groups": ["depigmented"], "labels": ["Vitiligo", "Pityriasis versicolor"]},
    {"id": "scaling_over_patch", "section": "history_of_presenting_illness", "text": "Is there scaling over the patch?", "clinical_relevance": 2, "discrimination_power": 3, "red_flag_importance": 0, "patient_burden": 1, "groups": ["depigmented"], "labels": ["Pityriasis versicolor", "Vitiligo"]},
    {"id": "new_contact_exposure", "section": "history_of_presenting_illness", "text": "Any new soap, cosmetic, detergent, metal, footwear, plant, or occupational exposure?", "clinical_relevance": 2, "discrimination_power": 3, "red_flag_importance": 0, "patient_burden": 1, "groups": ["inflammatory"], "labels": ["Eczema", "Contact dermatitis"]},
    {"id": "scaling_plaques", "section": "history_of_presenting_illness", "text": "Any scaling plaques?", "clinical_relevance": 2, "discrimination_power": 2, "red_flag_importance": 0, "patient_burden": 1, "groups": ["inflammatory"], "labels": ["Psoriasis", "Eczema"]},
]

RED_FLAG_ANSWER_IDS: dict[str, str] = {
    "lesion_progression": "Rapidly spreading lesion",
    "pain": "Severe pain",
    "fever": "Fever associated with lesion",
    "pus_or_discharge": "Pus or discharge with swelling",
    "bleeding_or_ulceration": "Bleeding or ulceration",
    "loss_of_sensation": "Loss of sensation over patch",
    "immune_suppression": "Immunocompromised patient",
}


@dataclass
class Prediction:
    label: str
    probability: float


def normalize_label(value: str) -> str:
    return value.strip().lower().replace("_", " ").replace("-", " ")


def canonical_label(label: str) -> str:
    normalized = normalize_label(label)
    return LABEL_ALIASES.get(normalized, str(label).strip())


def diagnosis_group_for_label(label: str) -> str:
    canonical = canonical_label(label)
    for group_name, labels in DISEASE_GROUPS.items():
        if canonical in labels:
            return group_name
    return "general"


def flatten_prediction_payload(predictions_payload: dict[str, Any] | list[dict[str, Any]]) -> list[Prediction]:
    if is_combined_payload(predictions_payload):
        rows: list[Prediction] = []
        for branch_payload in predictions_payload["branches"].values():
            rows.extend(parse_predictions(branch_payload))
        return rows
    return parse_predictions(predictions_payload)


def diagnosis_guided_supported(predictions_payload: dict[str, Any] | list[dict[str, Any]]) -> bool:
    predictions = flatten_prediction_payload(predictions_payload)
    return any(diagnosis_group_for_label(prediction.label) != "general" for prediction in predictions)


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


def profile_alias_map(profile_name: str) -> dict[str, str]:
    banks = QUESTION_PROFILES[profile_name]["question_banks"]
    alias_map: dict[str, str] = {}
    for candidate_id, bank in banks.items():
        alias_map[normalize_label(candidate_id)] = candidate_id
        for alias in bank["possible_labels"]:
            alias_map[normalize_label(alias)] = candidate_id
    return alias_map


def candidate_for_label(label: str, profile_name: str) -> str | None:
    return profile_alias_map(profile_name).get(normalize_label(label))


def parse_predictions(payload: dict[str, Any] | list[dict[str, Any]]) -> list[Prediction]:
    rows = payload.get("top_predictions", payload) if isinstance(payload, dict) else payload
    return [
        Prediction(label=str(row["label"]), probability=float(row.get("probability", row.get("prob", 0.0))))
        for row in rows
    ]


def coerce_predictions(
    payload: dict[str, Any] | list[dict[str, Any]] | list[Prediction],
) -> list[Prediction]:
    if isinstance(payload, list) and all(isinstance(item, Prediction) for item in payload):
        return payload
    return parse_predictions(payload)  # type: ignore[arg-type]


def is_combined_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("branches"), dict)


def merge_questions(*question_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for question_list in question_lists:
        for question in question_list:
            question_id = str(question.get("id"))
            if question_id in seen_ids:
                continue
            merged.append(question)
            seen_ids.add(question_id)
    return merged


def unique_strings(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def prediction_score_map(predictions_payload: dict[str, Any] | list[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for prediction in flatten_prediction_payload(predictions_payload):
        label = canonical_label(prediction.label)
        scores[label] = scores.get(label, 0.0) + float(prediction.probability)
    return scores


def diagnosis_groups_from_predictions(predictions_payload: dict[str, Any] | list[dict[str, Any]]) -> list[str]:
    groups = [diagnosis_group_for_label(prediction.label) for prediction in flatten_prediction_payload(predictions_payload)]
    ordered: list[str] = []
    for group_name in groups:
        if group_name not in ordered:
            ordered.append(group_name)
    return ordered or ["general"]


def diagnosis_group_score_map(predictions_payload: dict[str, Any] | list[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label, score in prediction_score_map(predictions_payload).items():
        group_name = diagnosis_group_for_label(label)
        scores[group_name] = scores.get(group_name, 0.0) + score
    return scores


def model_relevance_for_question(question: dict[str, Any], label_scores: dict[str, float], active_groups: list[str]) -> float:
    relevance = 0.0
    question_groups = set(question.get("groups", []))
    question_labels = {canonical_label(label) for label in question.get("labels", [])}
    has_specific_labels = bool(question_labels)
    for label, score in label_scores.items():
        if label in question_labels:
            relevance += score
            continue
        if has_specific_labels:
            if diagnosis_group_for_label(label) in question_groups:
                relevance += score * 0.15
            continue
        if "general" in question_groups:
            relevance += score * 0.15
            continue
        if diagnosis_group_for_label(label) in question_groups:
            relevance += score
    if question_groups.intersection(active_groups):
        relevance += 0.1
    return relevance


def score_diagnosis_guided_question(
    question: dict[str, Any],
    label_scores: dict[str, float],
    active_groups: list[str],
    asked_questions: set[str],
) -> float:
    score = 0.0
    score += float(question.get("clinical_relevance", 0)) * 2.0
    score += float(question.get("discrimination_power", 0)) * 3.0
    score += float(question.get("red_flag_importance", 0)) * 4.0
    score += model_relevance_for_question(question, label_scores, active_groups) * 8.0
    if question["id"] in asked_questions:
        score -= 100.0
    score -= float(question.get("patient_burden", 0))
    return score


def diagnosis_guided_question_candidates(
    predictions_payload: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    active_groups = diagnosis_groups_from_predictions(predictions_payload)
    label_scores = prediction_score_map(predictions_payload)
    group_scores = diagnosis_group_score_map(predictions_payload)
    dominant_label = max(label_scores.items(), key=lambda item: item[1])[0] if label_scores else None
    ranked_common: list[tuple[float, dict[str, Any]]] = []
    ranked_specific: list[tuple[float, dict[str, Any]]] = []

    for question in DIAGNOSIS_GUIDED_COMMON_QUESTIONS + DIAGNOSIS_GUIDED_SPECIFIC_QUESTIONS:
        groups = set(question.get("groups", []))
        if "general" not in groups and not groups.intersection(active_groups):
            continue
        model_relevance = model_relevance_for_question(question, label_scores, active_groups)
        if "general" not in groups:
            branch_weight = max((group_scores.get(group_name, 0.0) for group_name in groups), default=0.0)
            if branch_weight < 0.12:
                continue
            model_relevance += branch_weight
        rank = score_diagnosis_guided_question(question, label_scores, active_groups, set()) + model_relevance * 6.0
        if question in DIAGNOSIS_GUIDED_COMMON_QUESTIONS:
            ranked_common.append((rank, question))
        else:
            ranked_specific.append((rank, question))

    ranked_common.sort(key=lambda item: item[0], reverse=True)
    ranked_specific.sort(key=lambda item: item[0], reverse=True)

    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    dominant_specific = []
    if dominant_label:
        for rank, question in ranked_specific:
            if dominant_label in {canonical_label(label) for label in question.get("labels", [])}:
                dominant_specific.append((rank + 4.0, question))
        dominant_specific.sort(key=lambda item: item[0], reverse=True)

    ordered_specific = dominant_specific[:4] + ranked_specific
    for rank, question in ranked_common[:8] + ordered_specific:
        if question["id"] in seen_ids:
            continue
        copied = deepcopy(question)
        copied["type"] = "adaptive"
        copied["profile"] = "diagnosis_guided"
        copied["priority_score"] = round(rank, 4)
        questions.append(copied)
        seen_ids.add(copied["id"])
    return questions


def apply_metadata_bias(scores: dict[str, float], patient_context: dict[str, Any] | None) -> dict[str, float]:
    if not patient_context:
        return scores
    updated = dict(scores)
    body_part = normalize_label(str(patient_context.get("body_part_affected", "")))
    color = normalize_label(str(patient_context.get("lesion_color", "")))
    pattern = normalize_label(str(patient_context.get("lesion_pattern", "")))
    scaling = normalize_label(str(patient_context.get("lesion_surface_change", "")))
    pigmentation = normalize_label(str(patient_context.get("lesion_pigmentation", "")))

    if "groin" in body_part or "thigh" in body_part:
        updated["Tinea cruris"] = updated.get("Tinea cruris", 0.0) + 0.15
    if "scalp" in body_part:
        updated["Tinea capitis"] = updated.get("Tinea capitis", 0.0) + 0.15
    if any(token in body_part for token in ["body", "forearm", "arm", "trunk"]):
        updated["Tinea corporis"] = updated.get("Tinea corporis", 0.0) + 0.1
    if any(token in pattern for token in ["ring", "annular", "central clearing"]):
        updated["Tinea corporis"] = updated.get("Tinea corporis", 0.0) + 0.2
    if "scale" in scaling or "crust" in scaling:
        updated["Pityriasis versicolor"] = updated.get("Pityriasis versicolor", 0.0) + 0.08
        updated["Psoriasis"] = updated.get("Psoriasis", 0.0) + 0.08
    if any(token in pigmentation for token in ["depigmented", "white", "hypopigmented"]):
        updated["Vitiligo"] = updated.get("Vitiligo", 0.0) + 0.15
    if any(token in color for token in ["black", "dark brown", "variegated"]):
        updated["Melanoma"] = updated.get("Melanoma", 0.0) + 0.12
    return updated


def update_diagnosis_guided_scores(
    base_scores: dict[str, float],
    answers: dict[str, Any],
    patient_context: dict[str, Any] | None,
) -> tuple[dict[str, float], list[str], list[str]]:
    scores = apply_metadata_bias(base_scores, patient_context)
    question_lookup = {
        question["id"]: question
        for question in (DIAGNOSIS_GUIDED_COMMON_QUESTIONS + DIAGNOSIS_GUIDED_SPECIFIC_QUESTIONS)
    }
    key_positive: list[str] = []
    key_negative: list[str] = []

    for answer_id, raw_answer in answers.items():
        positive = answer_is_positive(raw_answer)
        if positive is None:
            continue
        question = question_lookup.get(answer_id)
        if not question:
            continue
        weight = 0.04 * (
            float(question.get("clinical_relevance", 0))
            + float(question.get("discrimination_power", 0))
            + float(question.get("red_flag_importance", 0))
        )
        target_labels = [canonical_label(label) for label in question.get("labels", [])]
        target_groups = question.get("groups", [])

        if positive:
            key_positive.append(question["text"])
            if target_labels:
                for label in target_labels:
                    scores[label] = scores.get(label, 0.0) + weight
            for label in list(scores):
                if diagnosis_group_for_label(label) in target_groups:
                    scores[label] += weight * 0.4
        else:
            key_negative.append(question["text"])
            if target_labels:
                for label in target_labels:
                    scores[label] = max(0.0, scores.get(label, 0.0) - weight * 0.6)
    return scores, unique_strings(key_positive), unique_strings(key_negative)


def melanoma_rule_score(answers: dict[str, Any]) -> int:
    score = 0
    if answer_is_positive(answers.get("asymmetry")) is True:
        score += 1
    if answer_is_positive(answers.get("irregular_border")) is True:
        score += 1
    if answer_is_positive(answers.get("multiple_colors_dark_areas")) is True or answer_is_positive(answers.get("color_change")) is True:
        score += 1
    if answer_is_positive(answers.get("diameter_gt_6mm")) is True:
        score += 1
    if answer_is_positive(answers.get("shape_change")) is True or answer_is_positive(answers.get("size_change")) is True:
        score += 2
    if answer_is_positive(answers.get("bleeding_or_ulceration")) is True:
        score += 2
    return score


def diagnosis_guided_red_flags(answers: dict[str, Any]) -> list[dict[str, Any]]:
    red_flags: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(answers.get("lesion_progression"), str) and "rapid" in str(answers.get("lesion_progression")).lower():
        red_flags.append({"id": "lesion_progression", "text": "Rapidly spreading lesion", "weight": 2})
        seen.add("lesion_progression")
    for answer_id, flag_text in RED_FLAG_ANSWER_IDS.items():
        positive = answer_is_positive(answers.get(answer_id))
        if positive is True and answer_id not in seen:
            weight = 2 if answer_id in {"fever", "bleeding_or_ulceration", "loss_of_sensation", "immune_suppression"} else 1
            red_flags.append({"id": answer_id, "text": flag_text, "weight": weight})
            seen.add(answer_id)
    if answer_is_positive(answers.get("diabetes")) is True and (
        answer_is_positive(answers.get("pus_or_discharge")) is True
        or answer_is_positive(answers.get("fever")) is True
        or answer_is_positive(answers.get("swelling")) is True
    ):
        red_flags.append({"id": "diabetes_infection", "text": "Diabetic patient with possible skin infection", "weight": 2})
    return red_flags


def diagnosis_guided_urgency(
    answers: dict[str, Any],
    red_flags: list[dict[str, Any]],
    differential: list[dict[str, Any]],
) -> dict[str, str]:
    melanoma_score = melanoma_rule_score(answers)
    red_flag_weight = sum(int(flag.get("weight", 1)) for flag in red_flags)
    if melanoma_score >= 4:
        return {"level": "urgent dermatologist review", "review_priority": "Urgent dermatologist referral is recommended because the lesion may be suspicious."}
    if red_flag_weight >= 3:
        return {"level": "urgent dermatologist review", "review_priority": "Urgent in-person clinician review is recommended because red flags were detected."}
    if red_flags:
        return {"level": "routine dermatologist review", "review_priority": "Clinician review is required sooner if symptoms worsen or new warning signs appear."}
    top_label = differential[0]["display_name"] if differential else "skin lesion"
    return {"level": "routine dermatologist review", "review_priority": f"Routine dermatologist review recommended for {top_label.lower()} interpretation."}


def diagnosis_guided_differential_rows(scores: dict[str, float]) -> list[dict[str, Any]]:
    normalized = normalize_scores(scores)
    rows: list[dict[str, Any]] = []
    for label, score in sorted(normalized.items(), key=lambda item: item[1], reverse=True):
        rows.append(
            {
                "diagnosis": label,
                "display_name": label,
                "group": diagnosis_group_for_label(label),
                "score": round(score, 4),
            }
        )
    return rows


def score_diagnosis_guided_answers(
    predictions_payload: dict[str, Any] | list[dict[str, Any]],
    answers: dict[str, Any],
    patient_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_scores = prediction_score_map(predictions_payload)
    updated_scores, key_positive, key_negative = update_diagnosis_guided_scores(base_scores, answers, patient_context)
    differential = diagnosis_guided_differential_rows(updated_scores)
    red_flags = diagnosis_guided_red_flags(answers)
    urgency = diagnosis_guided_urgency(answers, red_flags, differential)
    return {
        "profile": "diagnosis_guided",
        "profile_display_name": "Abhigyan Algorithm",
        "candidate_type": "diagnosis",
        "updated_differential": differential,
        "urgency_level": urgency["level"],
        "doctor_review_priority": urgency["review_priority"],
        "red_flags": red_flags,
        "key_positive_answers": key_positive,
        "key_negative_answers": key_negative,
        "disease_groups": diagnosis_groups_from_predictions(predictions_payload),
        "melanoma_rule_score": melanoma_rule_score(answers),
        "disclaimer": DISCLAIMER_TEXT,
    }


def detect_profile(predictions: list[Prediction]) -> str:
    profile_hits = {name: 0 for name in QUESTION_PROFILES}
    for prediction in predictions:
        for profile_name in QUESTION_PROFILES:
            if candidate_for_label(prediction.label, profile_name):
                profile_hits[profile_name] += 1
    if profile_hits["ham10000"] > profile_hits["opd"]:
        return "ham10000"
    return "opd"


def initial_candidate_scores(predictions: list[Prediction], profile_name: str) -> dict[str, float]:
    banks = QUESTION_PROFILES[profile_name]["question_banks"]
    scores = {candidate_id: 0.0 for candidate_id in banks}
    for prediction in predictions:
        candidate_id = candidate_for_label(prediction.label, profile_name)
        if candidate_id:
            scores[candidate_id] += prediction.probability

    if not any(scores.values()):
        shared = 1.0 / len(scores)
        return {candidate_id: shared for candidate_id in scores}
    return scores


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        return scores
    return {candidate_id: score / total for candidate_id, score in scores.items()}


def get_adaptive_questions(
    predictions: list[Prediction] | list[dict[str, Any]],
    max_category_questions: int = 12,
    include_general: bool = True,
    profile_name: str | None = None,
) -> list[dict[str, Any]]:
    parsed = coerce_predictions(predictions)
    resolved_profile = profile_name or detect_profile(parsed)
    profile = QUESTION_PROFILES[resolved_profile]
    scores = initial_candidate_scores(parsed, resolved_profile)
    normalized = normalize_scores(scores)

    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    if include_general:
        for question in profile["general_questions"]:
            copied = deepcopy(question)
            copied["type"] = "general"
            copied["profile"] = resolved_profile
            questions.append(copied)
            seen_ids.add(copied["id"])

    ranked_candidates = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    ranked_question_rows: list[tuple[float, str, dict[str, Any]]] = []
    for candidate_id, score in ranked_candidates:
        if score <= 0 and any(normalized.values()):
            continue
        bank = profile["question_banks"][candidate_id]
        for question in bank["questions"]:
            if question["id"] in seen_ids:
                continue
            priority = score * float(question.get("positive_weight", 0.08))
            ranked_question_rows.append((priority, candidate_id, question))

    ranked_question_rows.sort(key=lambda row: row[0], reverse=True)
    adaptive_count = 0
    for _priority, candidate_id, question in ranked_question_rows:
        if question["id"] in seen_ids:
            continue
        copied = deepcopy(question)
        copied["type"] = "adaptive"
        copied["profile"] = resolved_profile
        copied["candidate"] = candidate_id
        copied["candidate_display_name"] = profile["question_banks"][candidate_id]["display_name"]
        questions.append(copied)
        seen_ids.add(copied["id"])
        adaptive_count += 1
        if adaptive_count >= max_category_questions:
            break
    return questions


def differential_rows(scores: dict[str, float], profile_name: str) -> list[dict[str, Any]]:
    profile = QUESTION_PROFILES[profile_name]
    normalized = normalize_scores(scores)
    candidate_key = profile["candidate_label"]
    rows = []
    for candidate_id, score in sorted(normalized.items(), key=lambda item: item[1], reverse=True):
        bank = profile["question_banks"][candidate_id]
        rows.append(
            {
                candidate_key: candidate_id,
                "display_name": bank["display_name"],
                "score": round(score, 4),
                "possible_labels": bank["possible_labels"],
            }
        )
    return rows


def determine_opd_urgency(
    red_flags: list[dict[str, Any]],
    answers: dict[str, Any],
    differential: list[dict[str, Any]],
) -> dict[str, str]:
    red_flag_score = sum(int(flag.get("weight", 1)) for flag in red_flags)
    top_category = differential[0].get("category") if differential else None
    has_severe_pain = answer_is_positive(answers.get("severe_pain_out_of_proportion")) is True
    has_black = answer_is_positive(answers.get("black_discoloration")) is True
    has_confusion = answer_is_positive(answers.get("confusion")) is True
    has_rapid_fever_diabetes = all(
        answer_is_positive(answers.get(item)) is True
        for item in ["rapidly_spreading_redness", "fever_or_chills", "diabetes"]
    )

    if has_black or has_confusion or has_severe_pain or red_flag_score >= 6:
        return {"level": "emergency", "review_priority": "Immediate emergency evaluation by clinician."}
    if has_rapid_fever_diabetes or red_flag_score >= 3 or top_category == "deep_bacterial_infection_urgent":
        return {"level": "urgent dermatologist review", "review_priority": "Same-day or urgent clinician review recommended."}
    if red_flag_score > 0:
        return {"level": "routine dermatologist review", "review_priority": "Clinician review required; prioritize if symptoms progress."}
    return {"level": "non-urgent / monitor with doctor advice", "review_priority": "Routine clinician review required for interpretation."}


def determine_ham10000_urgency(
    red_flags: list[dict[str, Any]],
    answers: dict[str, Any],
    differential: list[dict[str, Any]],
) -> dict[str, str]:
    red_flag_score = sum(int(flag.get("weight", 1)) for flag in red_flags)
    top_candidate = differential[0].get("lesion") if differential else None
    top_score = float(differential[0].get("score", 0.0)) if differential else 0.0

    has_change = answer_is_positive(answers.get("rapid_change_size_shape_color")) is True
    has_bleeding = (
        answer_is_positive(answers.get("bleeding_or_crusting_without_injury")) is True
        or answer_is_positive(answers.get("bleeds_with_minor_trauma")) is True
        or answer_is_positive(answers.get("bleeds_easily")) is True
    )
    has_non_healing = (
        answer_is_positive(answers.get("non_healing_sore")) is True
        or answer_is_positive(answers.get("non_healing_or_growing_patch")) is True
    )

    if (has_change and has_bleeding) or (has_non_healing and has_bleeding) or red_flag_score >= 4:
        return {"level": "urgent dermatologist review", "review_priority": "Prompt in-person dermatologist review recommended for suspicious lesion change."}
    if top_candidate in {"mel", "bcc", "akiec"} and (red_flag_score > 0 or top_score >= 0.45):
        return {"level": "urgent dermatologist review", "review_priority": "Dermatologist review should be prioritized because the lesion pattern may be concerning."}
    if red_flag_score > 0:
        return {"level": "routine dermatologist review", "review_priority": "Clinician review required, with earlier review if the lesion continues to change."}
    return {"level": "non-urgent / monitor with doctor advice", "review_priority": "Routine dermatologist review required for interpretation; do not self-diagnose from image alone."}


def determine_urgency(
    profile_name: str,
    red_flags: list[dict[str, Any]],
    answers: dict[str, Any],
    differential: list[dict[str, Any]],
) -> dict[str, str]:
    if profile_name == "ham10000":
        return determine_ham10000_urgency(red_flags, answers, differential)
    return determine_opd_urgency(red_flags, answers, differential)


def urgency_rank(level: str) -> int:
    order = {
        "non-urgent / monitor with doctor advice": 1,
        "routine dermatologist review": 2,
        "urgent dermatologist review": 3,
        "emergency": 4,
    }
    return order.get(level, 0)


def score_answers(
    predictions: list[Prediction] | list[dict[str, Any]],
    answers: dict[str, Any],
    profile_name: str | None = None,
) -> dict[str, Any]:
    parsed = coerce_predictions(predictions)
    resolved_profile = profile_name or detect_profile(parsed)
    profile = QUESTION_PROFILES[resolved_profile]
    scores = initial_candidate_scores(parsed, resolved_profile)
    red_flags: list[dict[str, Any]] = []
    seen_red_flag_ids: set[str] = set()
    key_positive: list[str] = []
    key_negative: list[str] = []

    question_lookup = {
        question["id"]: (candidate_id, question)
        for candidate_id, bank in profile["question_banks"].items()
        for question in bank["questions"]
    }

    for answer_id, raw_answer in answers.items():
        positive = answer_is_positive(raw_answer)
        if positive is None:
            continue

        if answer_id in question_lookup:
            candidate_id, question = question_lookup[answer_id]
            weight = float(question.get("positive_weight", 0.08))
            if positive:
                scores[candidate_id] += weight
                key_positive.append(question["text"])
                if question.get("red_flag") and answer_id not in seen_red_flag_ids:
                    red_flags.append({"id": answer_id, "text": question["text"], "weight": int(question.get("red_flag_weight", 1))})
                    seen_red_flag_ids.add(answer_id)
            else:
                scores[candidate_id] = max(0.0, scores[candidate_id] - weight * 0.4)
                key_negative.append(question["text"])

        global_flag = profile["global_red_flags"].get(answer_id)
        if positive and global_flag and answer_id not in seen_red_flag_ids:
            red_flags.append({"id": answer_id, "text": global_flag["text"], "weight": int(global_flag["weight"])})
            seen_red_flag_ids.add(answer_id)
            if global_flag["text"] not in key_positive:
                key_positive.append(global_flag["text"])

    differential = differential_rows(scores, resolved_profile)
    urgency = determine_urgency(resolved_profile, red_flags, answers, differential)
    return {
        "profile": resolved_profile,
        "profile_display_name": profile["display_name"],
        "candidate_type": profile["candidate_label"],
        "updated_differential": differential,
        "urgency_level": urgency["level"],
        "doctor_review_priority": urgency["review_priority"],
        "red_flags": red_flags,
        "key_positive_answers": key_positive,
        "key_negative_answers": key_negative,
        "disclaimer": DISCLAIMER_TEXT,
    }


def score_combined_answers(
    combined_payload: dict[str, Any],
    answers: dict[str, Any],
) -> dict[str, Any]:
    branches = combined_payload.get("branches", {})
    branch_outputs: dict[str, dict[str, Any]] = {}
    branch_questions: dict[str, list[dict[str, Any]]] = {}

    for branch_name, payload in branches.items():
        profile_name = "ham10000" if branch_name == "ham10000" else "opd"
        branch_outputs[branch_name] = score_answers(payload, answers, profile_name=profile_name)
        branch_questions[branch_name] = get_adaptive_questions(
            parse_predictions(payload),
            include_general=False,
            max_category_questions=8,
            profile_name=profile_name,
        )

    combined_differential: list[dict[str, Any]] = []
    for branch_name, branch_output in branch_outputs.items():
        branch_title = "Lesion branch" if branch_name == "ham10000" else "Broad OPD branch"
        for row in branch_output["updated_differential"][:4]:
            combined_differential.append(
                {
                    "branch": branch_name,
                    "display_name": f"{branch_title}: {row['display_name']}",
                    "score": row["score"],
                    "possible_labels": row.get("possible_labels", []),
                }
            )

    combined_differential.sort(key=lambda item: float(item["score"]), reverse=True)
    merged_red_flags: list[dict[str, Any]] = []
    seen_red_flags: set[tuple[str, str]] = set()
    for branch_output in branch_outputs.values():
        for flag in branch_output["red_flags"]:
            marker = (str(flag.get("id")), str(flag.get("text")))
            if marker in seen_red_flags:
                continue
            merged_red_flags.append(flag)
            seen_red_flags.add(marker)

    strongest_branch = max(
        branch_outputs.values(),
        key=lambda item: urgency_rank(item["urgency_level"]),
        default={
            "urgency_level": "routine dermatologist review",
            "doctor_review_priority": "Clinician review required.",
        },
    )
    return {
        "profile": "combined",
        "profile_display_name": "Combined clinical + lesion reasoning",
        "candidate_type": "combined",
        "updated_differential": combined_differential,
        "branch_differentials": {
            branch_name: branch_output["updated_differential"]
            for branch_name, branch_output in branch_outputs.items()
        },
        "urgency_level": strongest_branch["urgency_level"],
        "doctor_review_priority": strongest_branch["doctor_review_priority"],
        "red_flags": merged_red_flags,
        "key_positive_answers": unique_strings(
            [item for branch_output in branch_outputs.values() for item in branch_output["key_positive_answers"]]
        ),
        "key_negative_answers": unique_strings(
            [item for branch_output in branch_outputs.values() for item in branch_output["key_negative_answers"]]
        ),
        "branch_outputs": branch_outputs,
        "branch_questions": branch_questions,
        "disclaimer": DISCLAIMER_TEXT,
    }


def run_engine(
    predictions_payload: dict[str, Any] | list[dict[str, Any]],
    answers: dict[str, Any] | None = None,
    max_category_questions: int = 12,
    patient_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if diagnosis_guided_supported(predictions_payload):
        questions = diagnosis_guided_question_candidates(predictions_payload)[:max_category_questions]
        scoring = score_diagnosis_guided_answers(
            predictions_payload,
            answers or {},
            patient_context=patient_context,
        )
        return {
            "profile": "diagnosis_guided",
            "profile_display_name": "Abhigyan Algorithm",
            "questions": questions,
            "scoring": scoring,
            "disclaimer": DISCLAIMER_TEXT,
        }

    if is_combined_payload(predictions_payload):
        branches = predictions_payload["branches"]
        opd_questions = get_adaptive_questions(
            parse_predictions(branches["opd"]),
            profile_name="opd",
            max_category_questions=max(4, max_category_questions // 2),
        )
        lesion_questions = get_adaptive_questions(
            parse_predictions(branches["ham10000"]),
            profile_name="ham10000",
            max_category_questions=max(4, max_category_questions // 2),
        )
        scoring = score_combined_answers(predictions_payload, answers or {})
        return {
            "profile": "combined",
            "profile_display_name": "Combined clinical + lesion reasoning",
            "questions": merge_questions(opd_questions, lesion_questions),
            "branch_questions": scoring.get("branch_questions", {}),
            "scoring": scoring,
            "disclaimer": DISCLAIMER_TEXT,
        }

    predictions = parse_predictions(predictions_payload)
    profile_name = detect_profile(predictions)
    questions = get_adaptive_questions(
        predictions,
        max_category_questions=max_category_questions,
        profile_name=profile_name,
    )
    scoring = score_answers(predictions, answers or {}, profile_name=profile_name)
    return {
        "profile": profile_name,
        "profile_display_name": QUESTION_PROFILES[profile_name]["display_name"],
        "questions": questions,
        "scoring": scoring,
        "disclaimer": DISCLAIMER_TEXT,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive OPD-style question engine.")
    parser.add_argument("--predictions", required=True, help="JSON file or JSON string with top_predictions")
    parser.add_argument("--answers", default=None, help="Optional JSON file or JSON string with answers")
    parser.add_argument("--patient_context", default=None, help="Optional JSON file or JSON string with patient and lesion context")
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
    patient_context = load_json_arg(args.patient_context) if args.patient_context else None
    result = run_engine(
        predictions,
        answers=answers,
        max_category_questions=args.max_category_questions,
        patient_context=patient_context,
    )
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
