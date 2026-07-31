from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageStat


@dataclass(frozen=True)
class QualityThresholds:
    min_edge_variance: float = 70.0
    min_brightness: float = 35.0
    max_brightness: float = 230.0
    min_saturation: float = 8.0
    min_dimension_px: int = 160
    extreme_aspect_ratio: float = 3.0


def _laplacian_variance(gray: np.ndarray) -> float:
    center = gray[1:-1, 1:-1] * -4.0
    response = center + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(np.var(response))


def assess_image_quality(
    image_path: str | Path,
    thresholds: QualityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or QualityThresholds()
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    resized = image.resize((224, 224))
    gray = np.asarray(resized.convert("L"), dtype=np.float32)
    hsv = resized.convert("HSV")
    brightness = float(ImageStat.Stat(resized.convert("L")).mean[0])
    saturation = float(ImageStat.Stat(hsv.getchannel("S")).mean[0])
    edge_variance = _laplacian_variance(gray)

    warnings: list[str] = []
    if min(width, height) < thresholds.min_dimension_px:
        warnings.append("Image resolution is very low.")
    if edge_variance < thresholds.min_edge_variance:
        warnings.append("Image may be blurred.")
    if brightness < thresholds.min_brightness:
        warnings.append("Image appears poorly lit.")
    if brightness > thresholds.max_brightness:
        warnings.append("Image appears overexposed.")
    if saturation < thresholds.min_saturation:
        warnings.append("Image has very low color information; lesion visibility may be poor.")
    if max(width, height) / max(min(width, height), 1) > thresholds.extreme_aspect_ratio:
        warnings.append("Image framing is unusually narrow and may reflect extreme zoom or cropping.")

    return {
        "accepted": not warnings,
        "warnings": warnings,
        "measurements": {
            "width": width,
            "height": height,
            "brightness": round(brightness, 3),
            "saturation": round(saturation, 3),
            "edge_variance": round(edge_variance, 3),
        },
    }


def image_quality_warning_text(quality: dict[str, Any] | None) -> str:
    if not quality:
        return "Not assessed"
    warnings = quality.get("warnings", [])
    if not warnings:
        return "No automatic quality warnings detected."
    return " ".join(str(item) for item in warnings)
