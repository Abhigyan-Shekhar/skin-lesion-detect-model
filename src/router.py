from __future__ import annotations

from pathlib import Path
from typing import Any

from inference import predict_single_checkpoint
from utils import DISCLAIMER_TEXT


def normalize_modality(value: str | None) -> str:
    normalized = (value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "clinical_photo": "clinical",
        "clinical_photograph": "clinical",
        "smartphone": "clinical",
        "dermoscopy": "dermoscopic",
        "dermoscopic_image": "dermoscopic",
        "both_images": "both",
    }
    return aliases.get(normalized, normalized)


def run_modality_gated_inference(
    *,
    modality: str,
    clinical_image_path: str | Path | None = None,
    dermoscopic_image_path: str | Path | None = None,
    opd_checkpoint: str | Path | None = None,
    lesion_checkpoint: str | Path | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    resolved = normalize_modality(modality)
    if resolved == "unknown":
        return {
            "mode": "abstain",
            "modality": resolved,
            "reason": "Image modality is unknown. Request a clinical photograph or dermoscopic image before model inference.",
            "disclaimer": DISCLAIMER_TEXT,
        }
    if resolved == "clinical":
        if not clinical_image_path or not opd_checkpoint:
            raise ValueError("clinical_image_path and opd_checkpoint are required for clinical photographs.")
        return {
            "mode": "single_model",
            "modality": resolved,
            "selected_branch": "opd",
            "image_path": str(Path(clinical_image_path)),
            "branches": {
                "opd": predict_single_checkpoint(opd_checkpoint, clinical_image_path, top_k=top_k),
            },
            "disclaimer": DISCLAIMER_TEXT,
        }
    if resolved == "dermoscopic":
        if not dermoscopic_image_path or not lesion_checkpoint:
            raise ValueError("dermoscopic_image_path and lesion_checkpoint are required for dermoscopic images.")
        return {
            "mode": "single_model",
            "modality": resolved,
            "selected_branch": "ham10000",
            "image_path": str(Path(dermoscopic_image_path)),
            "branches": {
                "ham10000": predict_single_checkpoint(lesion_checkpoint, dermoscopic_image_path, top_k=top_k),
            },
            "disclaimer": DISCLAIMER_TEXT,
        }
    if resolved == "both":
        if not clinical_image_path or not dermoscopic_image_path or not opd_checkpoint or not lesion_checkpoint:
            raise ValueError("Both image paths and both checkpoints are required for both-images inference.")
        return {
            "mode": "dual_valid_modalities",
            "modality": resolved,
            "image_path": {
                "clinical": str(Path(clinical_image_path)),
                "dermoscopic": str(Path(dermoscopic_image_path)),
            },
            "branches": {
                "opd": predict_single_checkpoint(opd_checkpoint, clinical_image_path, top_k=top_k),
                "ham10000": predict_single_checkpoint(lesion_checkpoint, dermoscopic_image_path, top_k=top_k),
            },
            "disclaimer": DISCLAIMER_TEXT,
        }
    return {
        "mode": "abstain",
        "modality": "unsupported",
        "reason": f"Unsupported modality: {modality}.",
        "disclaimer": DISCLAIMER_TEXT,
    }


def run_dual_model_inference(
    image_path: str | Path,
    opd_checkpoint: str | Path,
    lesion_checkpoint: str | Path,
    top_k_opd: int = 5,
    top_k_lesion: int = 5,
) -> dict[str, Any]:
    """Legacy helper. Prefer run_modality_gated_inference for new triage flows."""
    opd_result = predict_single_checkpoint(
        checkpoint_path=opd_checkpoint,
        image_path=image_path,
        top_k=top_k_opd,
    )
    lesion_result = predict_single_checkpoint(
        checkpoint_path=lesion_checkpoint,
        image_path=image_path,
        top_k=top_k_lesion,
    )
    return {
        "mode": "dual_model",
        "image_path": str(Path(image_path)),
        "branches": {
            "opd": opd_result,
            "ham10000": lesion_result,
        },
    }
