from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


KEYPOINT_FIELDS = ("x", "y", "size", "angle", "response", "octave", "class_id")
SUPPORTED_LOCAL_FEATURE_METHODS = ("SIFT", "ORB")


@dataclass(slots=True)
class LocalFeatureResult:
    """Structured output for the local feature extraction stage."""

    keypoints: np.ndarray = field(default_factory=lambda: np.empty((0, len(KEYPOINT_FIELDS)), dtype=np.float32))
    descriptors: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)



def extract_local_features(image: Any, config: dict[str, Any] | None = None) -> LocalFeatureResult:
    """Extract serializable local features using a configured OpenCV backend."""
    feature_config = config or {}
    input_image = _validate_image(image)
    method = _resolve_method(feature_config)
    grayscale_image = _to_grayscale(input_image)
    extractor = _create_extractor(method, feature_config)

    keypoints, descriptors = extractor.detectAndCompute(grayscale_image, None)
    serialized_keypoints = _serialize_keypoints(keypoints)

    if descriptors is not None:
        descriptors = np.asarray(descriptors)

    descriptor_shape = None if descriptors is None else tuple(int(dim) for dim in descriptors.shape)
    descriptor_dtype = None if descriptors is None else str(descriptors.dtype)

    return LocalFeatureResult(
        keypoints=serialized_keypoints,
        descriptors=descriptors,
        meta={
            "stage": "local_feature_extraction",
            "status": "ready",
            "method": method,
            "num_keypoints": int(serialized_keypoints.shape[0]),
            "descriptor_shape": descriptor_shape,
            "descriptor_dtype": descriptor_dtype,
            "input_shape": _shape_of(input_image),
            "working_image_shape": _shape_of(grayscale_image),
            "keypoint_fields": list(KEYPOINT_FIELDS),
        },
    )



def save_local_feature_result(
    sample_id: str,
    feature_result: LocalFeatureResult,
    output_dir: str | Path,
) -> Path:
    """Persist a local feature result as a compressed .npz file."""
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    file_name = _make_safe_feature_name(sample_id)
    save_path = target_dir / f"{file_name}.npz"

    descriptors_present = feature_result.descriptors is not None
    descriptors_to_save = (
        feature_result.descriptors
        if descriptors_present
        else np.empty((0, 0), dtype=np.float32)
    )

    np.savez_compressed(
        save_path,
        sample_id=np.array(sample_id),
        method=np.array(feature_result.meta.get("method", "")),
        num_keypoints=np.array(int(feature_result.meta.get("num_keypoints", 0)), dtype=np.int32),
        keypoints=feature_result.keypoints,
        descriptors=descriptors_to_save,
        descriptors_present=np.array(int(descriptors_present), dtype=np.uint8),
        descriptor_shape=np.array(feature_result.meta.get("descriptor_shape") or (), dtype=np.int32),
        descriptor_dtype=np.array(feature_result.meta.get("descriptor_dtype") or ""),
        keypoint_fields=np.array(feature_result.meta.get("keypoint_fields") or KEYPOINT_FIELDS),
    )
    return save_path



def _resolve_method(config: dict[str, Any]) -> str:
    raw_method = config.get("method", config.get("local_method", "sift"))
    if not isinstance(raw_method, str) or not raw_method.strip():
        raise ValueError("Local feature method must be a non-empty string.")

    method = raw_method.strip().upper()
    if method not in SUPPORTED_LOCAL_FEATURE_METHODS:
        raise ValueError(
            "Unsupported local feature method: "
            f"{raw_method}. Supported methods: {', '.join(SUPPORTED_LOCAL_FEATURE_METHODS)}"
        )
    return method



def _create_extractor(method: str, config: dict[str, Any]) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for local feature extraction.") from exc

    if method == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("OpenCV in this environment does not provide SIFT_create.")
        nfeatures = _optional_positive_int(config.get("sift_nfeatures"), "local_feature.sift_nfeatures")
        return cv2.SIFT_create() if nfeatures is None else cv2.SIFT_create(nfeatures=nfeatures)

    if method == "ORB":
        if not hasattr(cv2, "ORB_create"):
            raise RuntimeError("OpenCV in this environment does not provide ORB_create.")
        nfeatures = _optional_positive_int(
            config.get("orb_nfeatures", 500),
            "local_feature.orb_nfeatures",
        )
        return cv2.ORB_create(nfeatures=nfeatures)

    raise ValueError(f"Unsupported local feature method: {method}")



def _validate_image(image: Any) -> np.ndarray:
    if image is None:
        raise ValueError("Local feature input image is None.")
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "Local feature input must be a numpy.ndarray, "
            f"got {type(image).__name__}."
        )
    if image.size == 0:
        raise ValueError("Local feature input image is empty.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Local feature input must be 2D or 3D, got ndim={image.ndim}.")
    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise ValueError(
            "Local feature input channel count must be 1 or 3, "
            f"got shape={tuple(int(dim) for dim in image.shape)}."
        )
    return image



def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required to convert local feature input to grayscale.") from exc

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def _serialize_keypoints(keypoints: list[Any] | tuple[Any, ...] | None) -> np.ndarray:
    if not keypoints:
        return np.empty((0, len(KEYPOINT_FIELDS)), dtype=np.float32)

    serialized = np.empty((len(keypoints), len(KEYPOINT_FIELDS)), dtype=np.float32)
    for index, keypoint in enumerate(keypoints):
        serialized[index] = (
            float(keypoint.pt[0]),
            float(keypoint.pt[1]),
            float(keypoint.size),
            float(keypoint.angle),
            float(keypoint.response),
            float(keypoint.octave),
            float(keypoint.class_id),
        )
    return serialized



def _make_safe_feature_name(sample_id: str) -> str:
    normalized = sample_id.replace("\\", "/")
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "__", normalized)
    normalized = normalized.strip("._")
    return normalized or "feature"



def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
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
