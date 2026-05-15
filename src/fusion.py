from __future__ import annotations

from typing import Any

from utils import DISCLAIMER_TEXT


CONFIDENCE_RANK = {
    "low": 1,
    "moderate": 2,
    "high": 3,
}


def branch_strength(branch_payload: dict[str, Any]) -> float:
    confidence = str(branch_payload.get("confidence_level", "low")).lower()
    confidence_score = CONFIDENCE_RANK.get(confidence, 0)
    max_probability = float(branch_payload.get("max_probability", 0.0))
    return confidence_score + max_probability


def dominant_branch(branches: dict[str, dict[str, Any]]) -> str:
    ranked = sorted(branches.items(), key=lambda item: branch_strength(item[1]), reverse=True)
    return ranked[0][0] if ranked else "opd"


def build_combined_payload(dual_model_result: dict[str, Any]) -> dict[str, Any]:
    branches = dual_model_result.get("branches", {})
    dominant = dominant_branch(branches)
    return {
        "mode": "combined",
        "image_path": dual_model_result.get("image_path"),
        "branches": branches,
        "fusion": {
            "dominant_branch": dominant,
            "branch_confidence_levels": {
                branch_name: payload.get("confidence_level")
                for branch_name, payload in branches.items()
            },
            "branch_max_probabilities": {
                branch_name: payload.get("max_probability")
                for branch_name, payload in branches.items()
            },
            "reasoning_note": (
                "Both models were run on the same image. Broad OPD context and lesion-specific "
                "signals should be interpreted together rather than averaging probabilities."
            ),
        },
        "disclaimer": DISCLAIMER_TEXT,
    }
