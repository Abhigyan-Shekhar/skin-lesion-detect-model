from __future__ import annotations

from pathlib import Path
from typing import Any

from inference import predict_single_checkpoint


def run_dual_model_inference(
    image_path: str | Path,
    opd_checkpoint: str | Path,
    lesion_checkpoint: str | Path,
    top_k_opd: int = 5,
    top_k_lesion: int = 5,
) -> dict[str, Any]:
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
