from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


SUPPORTED_COLOR_MODES = ("keep", "gray")


@dataclass(slots=True)
class PreprocessResult:
    """Structured output for the basic preprocess stage."""

    image: np.ndarray
    original_shape: tuple[int, ...]
    processed_shape: tuple[int, ...]
    color_mode: str
    meta: dict[str, Any] = field(default_factory=dict)



def preprocess_image(image: Any, config: dict[str, Any] | None = None) -> PreprocessResult:
    """Validate and minimally standardize an OpenCV image for later stages."""
    preprocess_config = config or {}
    validated_image = _validate_image(image)
    resize_config = _resolve_resize_config(preprocess_config)
    requested_color_mode = _resolve_color_mode(preprocess_config)

    processed_image = validated_image
    applied_steps: list[str] = []

    if resize_config["enabled"]:
        processed_image = _resize_image(
            processed_image,
            width=resize_config["width"],
            height=resize_config["height"],
        )
        applied_steps.append("resize")

    if requested_color_mode == "gray":
        processed_image = _convert_to_gray(processed_image)
        applied_steps.append("color_mode:gray")

    if not applied_steps:
        applied_steps.append("passthrough")

    original_shape = _shape_of(validated_image)
    processed_shape = _shape_of(processed_image)
    final_color_mode = "gray" if processed_image.ndim == 2 else "keep"

    meta: dict[str, Any] = {
        "stage": "basic_preprocess",
        "status": "ready",
        "applied_steps": applied_steps,
        "input_shape": original_shape,
        "output_shape": processed_shape,
        "requested_color_mode": requested_color_mode,
        "resize": resize_config,
    }

    return PreprocessResult(
        image=processed_image,
        original_shape=original_shape,
        processed_shape=processed_shape,
        color_mode=final_color_mode,
        meta=meta,
    )



def _validate_image(image: Any) -> np.ndarray:
    if image is None:
        raise ValueError("Preprocess input image is None.")
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Preprocess input must be a numpy.ndarray returned by OpenCV, "
            f"got {type(image).__name__}."
        )
    if image.size == 0:
        raise ValueError("Preprocess input image is empty.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Preprocess input must be 2D or 3D, got ndim={image.ndim}.")
    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise ValueError(
            "Preprocess input channel count must be 1 or 3, "
            f"got shape={tuple(int(dim) for dim in image.shape)}."
        )
    if any(int(dim) <= 0 for dim in image.shape):
        raise ValueError(f"Preprocess input has invalid shape: {image.shape}.")
    return image



def _resolve_resize_config(config: dict[str, Any]) -> dict[str, Any]:
    resize_config = config.get("resize")
    if resize_config is None:
        legacy_image_size = config.get("image_size")
        if legacy_image_size is None:
            return {"enabled": False, "width": None, "height": None, "source": "default"}

        height, width = _parse_size_pair(legacy_image_size, "preprocess.image_size")
        return {
            "enabled": True,
            "width": width,
            "height": height,
            "source": "legacy_image_size",
        }

    if not isinstance(resize_config, dict):
        raise ValueError("Config section 'preprocess.resize' must be a mapping.")

    enabled = bool(resize_config.get("enabled", False))
    width_value = resize_config.get("width")
    height_value = resize_config.get("height")

    if not enabled:
        return {"enabled": False, "width": None, "height": None, "source": "resize_config"}

    width = _parse_positive_int(width_value, "preprocess.resize.width")
    height = _parse_positive_int(height_value, "preprocess.resize.height")
    return {"enabled": True, "width": width, "height": height, "source": "resize_config"}



def _resolve_color_mode(config: dict[str, Any]) -> str:
    color_mode = config.get("color_mode")
    if color_mode is None:
        return "gray" if bool(config.get("to_grayscale", False)) else "keep"

    if not isinstance(color_mode, str):
        raise ValueError("Config field 'preprocess.color_mode' must be a string.")

    normalized = color_mode.strip().lower()
    if normalized not in SUPPORTED_COLOR_MODES:
        raise ValueError(
            "Unsupported preprocess.color_mode: "
            f"{color_mode}. Supported values: {', '.join(SUPPORTED_COLOR_MODES)}"
        )
    return normalized



def _resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for resize preprocessing.") from exc

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)



def _convert_to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for grayscale preprocessing.") from exc

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def _parse_size_pair(value: Any, field_name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Config field '{field_name}' must be a [height, width] pair.")

    height = _parse_positive_int(value[0], f"{field_name}[0]")
    width = _parse_positive_int(value[1], f"{field_name}[1]")
    return height, width



def _parse_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Config field '{field_name}' must be a positive integer.")

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config field '{field_name}' must be a positive integer.") from exc

    if parsed <= 0:
        raise ValueError(f"Config field '{field_name}' must be greater than zero, got {parsed}.")
    return parsed



def _shape_of(image: np.ndarray) -> tuple[int, ...]:
    return tuple(int(dim) for dim in image.shape)
