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


def build_combined_payload(model_result: dict[str, Any]) -> dict[str, Any]:
    branches = model_result.get("branches", {})
    dominant = dominant_branch(branches)
    mode = model_result.get("mode", "combined")
    if len(branches) == 1:
        branch_name, payload = next(iter(branches.items()))
        return {
            "mode": "single_model",
            "modality": model_result.get("modality"),
            "selected_branch": branch_name,
            "image_path": model_result.get("image_path"),
            "top_predictions": payload.get("top_predictions", []),
            "confidence_level": payload.get("confidence_level"),
            "max_probability": payload.get("max_probability"),
            "branches": branches,
            "fusion": {
                "dominant_branch": branch_name,
                "reasoning_note": "Only the modality-appropriate model branch was run.",
            },
            "disclaimer": DISCLAIMER_TEXT,
        }
    return {
        "mode": "combined" if mode == "dual_model" else mode,
        "modality": model_result.get("modality"),
        "image_path": model_result.get("image_path"),
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
                "Both model branches were run only because valid clinical and dermoscopic "
                "images were provided. Probability spaces are kept separate and are not averaged."
            ),
        },
        "disclaimer": DISCLAIMER_TEXT,
    }
