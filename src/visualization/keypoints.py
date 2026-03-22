from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np



def render_keypoint_overlay(
    image: Any,
    keypoints: Any,
    label: str | None = None,
) -> np.ndarray:
    """Render serialized keypoints on top of an image."""
    canvas = _prepare_canvas(image)
    keypoint_array = _validate_keypoints(keypoints)

    for keypoint in keypoint_array:
        x = int(round(float(keypoint[0])))
        y = int(round(float(keypoint[1])))
        size = float(keypoint[2]) if keypoint.shape[0] >= 3 else 0.0
        radius = max(2, int(round(size / 6.0))) if size > 0 else 2
        cv2.circle(canvas, (x, y), radius, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    overlay_label = label or f"keypoints={keypoint_array.shape[0]}"
    cv2.putText(
        canvas,
        overlay_label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (40, 40, 40),
        2,
        lineType=cv2.LINE_AA,
    )
    return canvas



def save_keypoint_visualization(
    sample_id: str,
    image: Any,
    keypoints: Any,
    output_dir: str | Path,
    label: str | None = None,
) -> Path:
    """Save a rendered keypoint overlay as a PNG figure."""
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    figure = render_keypoint_overlay(image, keypoints, label=label)
    save_path = target_dir / f"keypoints_{_make_safe_name(sample_id)}.png"
    success = cv2.imwrite(str(save_path), figure)
    if not success:
        raise OSError(f"Failed to save keypoint visualization: {save_path}")
    return save_path



def _prepare_canvas(image: Any) -> np.ndarray:
    if image is None:
        raise ValueError("Visualization input image is None.")
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Visualization input must be a numpy.ndarray, "
            f"got {type(image).__name__}."
        )
    if image.size == 0:
        raise ValueError("Visualization input image is empty.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    raise ValueError(
        "Visualization input image must be 2D grayscale or 3-channel BGR, "
        f"got shape={tuple(int(dim) for dim in image.shape)}."
    )



def _validate_keypoints(keypoints: Any) -> np.ndarray:
    if keypoints is None:
        return np.empty((0, 2), dtype=np.float32)

    keypoint_array = np.asarray(keypoints, dtype=np.float32)
    if keypoint_array.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if keypoint_array.ndim != 2:
        raise ValueError(
            "Serialized keypoints for visualization must be a 2D array, "
            f"got ndim={keypoint_array.ndim}."
        )
    if keypoint_array.shape[1] < 2:
        raise ValueError(
            "Serialized keypoints for visualization must provide at least x and y columns, "
            f"got shape={keypoint_array.shape}."
        )
    return keypoint_array



def _make_safe_name(sample_id: str) -> str:
    normalized = str(sample_id).replace("\\", "/")
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "__", normalized)
    normalized = normalized.strip("._")
    return normalized or "sample"
