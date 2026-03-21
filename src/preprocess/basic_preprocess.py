from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PreprocessResult:
    """Minimal structured output for the preprocess stage."""

    image: Any
    meta: dict[str, Any] = field(default_factory=dict)


def preprocess_image(image: Any, config: dict[str, Any] | None = None) -> PreprocessResult:
    """Run a minimal preprocess stage that defaults to passthrough."""
    config = config or {}
    processed_image = image
    applied_steps: list[str] = []

    if bool(config.get("to_grayscale", False)) and _is_color_image(image):
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for grayscale preprocessing when 'to_grayscale' is enabled."
            ) from exc

        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        applied_steps.append("to_grayscale")
    else:
        applied_steps.append("passthrough")

    meta: dict[str, Any] = {
        "stage": "basic_preprocess",
        "status": "ready",
        "applied_steps": applied_steps,
        "input_shape": _shape_of(image),
        "output_shape": _shape_of(processed_image),
    }
    if "image_size" in config:
        meta["requested_image_size"] = config["image_size"]

    return PreprocessResult(image=processed_image, meta=meta)


def _is_color_image(image: Any) -> bool:
    shape = getattr(image, "shape", None)
    return isinstance(shape, tuple) and len(shape) == 3 and shape[2] >= 3


def _shape_of(image: Any) -> tuple[int, ...] | None:
    shape = getattr(image, "shape", None)
    if not isinstance(shape, tuple):
        return None
    return tuple(int(dim) for dim in shape)
