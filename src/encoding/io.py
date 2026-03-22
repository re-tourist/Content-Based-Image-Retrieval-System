from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.npyio import NpzFile

from src.features.local import KEYPOINT_FIELDS, LocalFeatureResult

from .types import EncodingInput, FeatureFileRecord


@dataclass(frozen=True, slots=True)
class _MethodSpec:
    dtype: str
    dim: int


_METHOD_SPECS = {
    "SIFT": _MethodSpec(dtype="float32", dim=128),
    "ORB": _MethodSpec(dtype="uint8", dim=32),
}

_FEATURE_FILE_REQUIRED_KEYS = (
    "sample_id",
    "method",
    "num_keypoints",
    "keypoints",
    "descriptors",
    "descriptors_present",
    "descriptor_shape",
    "descriptor_dtype",
)


def load_feature_npz(feature_path: str | Path) -> FeatureFileRecord:
    """Load and validate one saved local-feature artifact."""
    resolved_path = Path(feature_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Feature file not found: {resolved_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Feature path is not a file: {resolved_path}")

    with np.load(resolved_path, allow_pickle=False) as data:
        _validate_required_npz_keys(data, resolved_path)

        sample_id = _read_required_string(data, "sample_id", resolved_path)
        method = _normalize_method(_read_required_string(data, "method", resolved_path))
        num_keypoints = _read_required_int(data, "num_keypoints", resolved_path)
        keypoints = _validate_keypoints_array(np.array(data["keypoints"], copy=True), num_keypoints, resolved_path)
        descriptors_present = _read_descriptors_present(data, resolved_path)
        descriptor_shape = _read_descriptor_shape(data, "descriptor_shape", resolved_path)
        descriptor_dtype = _read_optional_dtype(data, "descriptor_dtype", resolved_path)
        descriptors_array = np.array(data["descriptors"], copy=True)

    descriptors = _validate_file_descriptors(
        descriptors=descriptors_array,
        descriptors_present=descriptors_present,
        method=method,
        num_keypoints=num_keypoints,
        descriptor_shape=descriptor_shape,
        descriptor_dtype=descriptor_dtype,
        source=resolved_path,
    )

    return FeatureFileRecord(
        sample_id=sample_id,
        method=method,
        num_keypoints=num_keypoints,
        keypoints=keypoints,
        descriptors=descriptors,
        descriptors_present=descriptors_present,
        descriptor_shape=descriptor_shape,
        descriptor_dtype=descriptor_dtype,
    )


def build_encoding_input_from_feature_record(record: FeatureFileRecord) -> EncodingInput:
    """Build normalized encoding input from a validated feature file record."""
    if not isinstance(record, FeatureFileRecord):
        raise TypeError(
            "Encoding input must be built from a FeatureFileRecord, "
            f"got {type(record).__name__}."
        )

    _validate_keypoints_array(record.keypoints, record.num_keypoints, "FeatureFileRecord")
    _validate_descriptor_count(record.num_keypoints, record.descriptors, "FeatureFileRecord")

    return _build_encoding_input(
        sample_id=record.sample_id,
        method=record.method,
        descriptors=record.descriptors,
        descriptor_shape=record.descriptor_shape,
        descriptor_dtype=record.descriptor_dtype,
        descriptors_present=record.descriptors_present,
        source="FeatureFileRecord",
    )


def build_encoding_input_from_local_feature(
    feature_result: LocalFeatureResult,
    method: str | None = None,
    sample_id: str | None = None,
) -> EncodingInput:
    """Build normalized encoding input from an in-memory LocalFeatureResult."""
    if not isinstance(feature_result, LocalFeatureResult):
        raise TypeError(
            "Encoding input must be built from a LocalFeatureResult, "
            f"got {type(feature_result).__name__}."
        )

    meta = feature_result.meta
    if not isinstance(meta, dict):
        raise TypeError("LocalFeatureResult.meta must be a dictionary.")

    resolved_sample_id = _normalize_optional_sample_id(sample_id, "sample_id")
    meta_method_raw = meta.get("method")
    if method is None:
        resolved_method = _normalize_method(meta_method_raw)
    else:
        resolved_method = _normalize_method(method)
        if meta_method_raw is not None and _normalize_method(meta_method_raw) != resolved_method:
            raise ValueError(
                "LocalFeatureResult method does not match the requested method: "
                f"meta={meta_method_raw!r}, method={method!r}."
            )

    num_keypoints = _read_optional_non_negative_int(meta.get("num_keypoints"), "LocalFeatureResult.meta['num_keypoints']")
    keypoints = _validate_keypoints_array(feature_result.keypoints, num_keypoints, "LocalFeatureResult")
    descriptors = _normalize_descriptor_array(feature_result.descriptors, "LocalFeatureResult.descriptors")
    descriptor_shape = _normalize_descriptor_shape(meta.get("descriptor_shape"), "LocalFeatureResult.meta['descriptor_shape']")
    descriptor_dtype = _normalize_optional_dtype_value(meta.get("descriptor_dtype"), "LocalFeatureResult.meta['descriptor_dtype']")
    descriptors_present = descriptors is not None

    if num_keypoints is not None:
        _validate_descriptor_count(num_keypoints, descriptors, "LocalFeatureResult")
    else:
        _validate_descriptor_count(int(keypoints.shape[0]), descriptors, "LocalFeatureResult")

    return _build_encoding_input(
        sample_id=resolved_sample_id,
        method=resolved_method,
        descriptors=descriptors,
        descriptor_shape=descriptor_shape,
        descriptor_dtype=descriptor_dtype,
        descriptors_present=descriptors_present,
        source="LocalFeatureResult",
    )


def _build_encoding_input(
    sample_id: str | None,
    method: str,
    descriptors: np.ndarray | None,
    descriptor_shape: tuple[int, ...],
    descriptor_dtype: str | None,
    descriptors_present: bool,
    source: Any,
) -> EncodingInput:
    resolved_method = _normalize_method(method)
    spec = _METHOD_SPECS[resolved_method]
    resolved_sample_id = _normalize_optional_sample_id(sample_id, "sample_id")

    if not descriptors_present:
        if descriptors is not None:
            raise ValueError(f"{source}: descriptors_present is False but descriptors are not empty.")
        _validate_empty_descriptor_metadata(descriptor_shape, descriptor_dtype, resolved_method, source)
        return EncodingInput(
            sample_id=resolved_sample_id,
            method=resolved_method,
            descriptors=None,
            descriptors_present=False,
            descriptor_shape=(0, spec.dim),
            descriptor_dtype=spec.dtype,
            descriptor_dim=spec.dim,
            num_descriptors=0,
        )

    if descriptors is None:
        raise ValueError(f"{source}: descriptors_present is True but descriptors are missing.")
    if descriptors.ndim != 2:
        raise ValueError(f"{source}: descriptors must be a 2D array, got shape={descriptors.shape}.")
    if descriptors.shape[0] <= 0:
        raise ValueError(f"{source}: descriptors_present is True but descriptor rows={descriptors.shape[0]}.")

    actual_shape = tuple(int(dim) for dim in descriptors.shape)
    actual_dtype = str(descriptors.dtype)

    if actual_dtype != spec.dtype:
        raise ValueError(
            f"{source}: {resolved_method} descriptors must have dtype {spec.dtype}, got {actual_dtype}."
        )
    if actual_shape[1] != spec.dim:
        raise ValueError(
            f"{source}: {resolved_method} descriptors must have dimension {spec.dim}, got {actual_shape[1]}."
        )
    if descriptor_shape and descriptor_shape != actual_shape:
        raise ValueError(
            f"{source}: descriptor_shape metadata {descriptor_shape} does not match descriptors.shape {actual_shape}."
        )
    if descriptor_dtype is not None and descriptor_dtype != actual_dtype:
        raise ValueError(
            f"{source}: descriptor_dtype metadata {descriptor_dtype!r} does not match descriptors dtype {actual_dtype!r}."
        )

    return EncodingInput(
        sample_id=resolved_sample_id,
        method=resolved_method,
        descriptors=descriptors,
        descriptors_present=True,
        descriptor_shape=actual_shape,
        descriptor_dtype=actual_dtype,
        descriptor_dim=spec.dim,
        num_descriptors=actual_shape[0],
    )


def _validate_required_npz_keys(data: NpzFile, feature_path: Path) -> None:
    missing_keys = [key for key in _FEATURE_FILE_REQUIRED_KEYS if key not in data.files]
    if missing_keys:
        raise ValueError(f"{feature_path}: missing required feature keys: {', '.join(missing_keys)}.")


def _read_required_string(data: NpzFile, key: str, source: Any) -> str:
    value = data[key]
    if value.ndim != 0:
        raise ValueError(f"{source}: key '{key}' must be a scalar string value.")

    scalar = value.item()
    if not isinstance(scalar, str) or not scalar.strip():
        raise ValueError(f"{source}: key '{key}' must be a non-empty string.")
    return scalar.strip()


def _read_required_int(data: NpzFile, key: str, source: Any) -> int:
    value = data[key]
    if value.ndim != 0:
        raise ValueError(f"{source}: key '{key}' must be a scalar integer value.")

    scalar = value.item()
    if isinstance(scalar, bool):
        raise ValueError(f"{source}: key '{key}' must be a non-negative integer.")

    try:
        parsed = int(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: key '{key}' must be a non-negative integer.") from exc

    if parsed < 0:
        raise ValueError(f"{source}: key '{key}' must be non-negative, got {parsed}.")
    return parsed


def _read_descriptors_present(data: NpzFile, source: Any) -> bool:
    parsed = _read_required_int(data, "descriptors_present", source)
    if parsed not in (0, 1):
        raise ValueError(f"{source}: key 'descriptors_present' must be 0 or 1, got {parsed}.")
    return bool(parsed)


def _read_descriptor_shape(data: NpzFile, key: str, source: Any) -> tuple[int, ...]:
    return _normalize_descriptor_shape(data[key], f"{source}: key '{key}'")


def _read_optional_dtype(data: NpzFile, key: str, source: Any) -> str | None:
    value = data[key]
    if value.ndim != 0:
        raise ValueError(f"{source}: key '{key}' must be a scalar string value.")
    return _normalize_optional_dtype_value(value.item(), f"{source}: key '{key}'")


def _normalize_method(method: Any) -> str:
    if not isinstance(method, str) or not method.strip():
        raise ValueError("Feature encoding method must be a non-empty string.")

    normalized = method.strip().upper()
    if normalized not in _METHOD_SPECS:
        raise ValueError(
            "Unsupported encoding method: "
            f"{method!r}. Supported methods: {', '.join(sorted(_METHOD_SPECS))}."
        )
    return normalized


def _normalize_optional_sample_id(sample_id: Any, field_name: str) -> str | None:
    if sample_id is None:
        return None
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"{field_name} must be a non-empty string when provided.")
    return sample_id.strip()


def _validate_keypoints_array(keypoints: Any, num_keypoints: int | None, source: Any) -> np.ndarray:
    if not isinstance(keypoints, np.ndarray):
        raise TypeError(f"{source}: keypoints must be a numpy.ndarray, got {type(keypoints).__name__}.")
    if keypoints.ndim != 2:
        raise ValueError(f"{source}: keypoints must be a 2D array, got shape={keypoints.shape}.")
    if keypoints.shape[1] != len(KEYPOINT_FIELDS):
        raise ValueError(
            f"{source}: keypoints must have shape (N, {len(KEYPOINT_FIELDS)}), got {keypoints.shape}."
        )
    if num_keypoints is not None and int(keypoints.shape[0]) != int(num_keypoints):
        raise ValueError(
            f"{source}: num_keypoints={num_keypoints} does not match keypoints rows={keypoints.shape[0]}."
        )
    return keypoints


def _normalize_descriptor_array(descriptors: Any, source: Any) -> np.ndarray | None:
    if descriptors is None:
        return None
    if not isinstance(descriptors, np.ndarray):
        raise TypeError(f"{source}: descriptors must be a numpy.ndarray or None, got {type(descriptors).__name__}.")

    if descriptors.size == 0:
        if descriptors.ndim == 1 and descriptors.shape == (0,):
            return None
        if descriptors.ndim == 2 and descriptors.shape[0] == 0:
            return None
        raise ValueError(f"{source}: empty descriptors must use shape (0,) or (0, D), got {descriptors.shape}.")

    return descriptors


def _normalize_descriptor_shape(value: Any, source: Any) -> tuple[int, ...]:
    if value is None:
        return ()

    array = np.asarray(value)
    if array.ndim > 1:
        raise ValueError(f"{source} must be a 1D shape array or empty, got shape={array.shape}.")

    shape: list[int] = []
    for item in array.tolist():
        if isinstance(item, bool):
            raise ValueError(f"{source} must contain integers, got boolean value {item}.")
        try:
            parsed = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source} must contain integers, got {item!r}.") from exc
        if parsed < 0:
            raise ValueError(f"{source} must contain non-negative integers, got {parsed}.")
        shape.append(parsed)
    return tuple(shape)


def _normalize_optional_dtype_value(value: Any, source: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{source} must be a string or empty.")

    stripped = value.strip()
    if not stripped:
        return None

    try:
        return str(np.dtype(stripped))
    except TypeError as exc:
        raise ValueError(f"{source} contains an invalid numpy dtype string: {value!r}.") from exc


def _read_optional_non_negative_int(value: Any, source: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{source} must be a non-negative integer.")

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a non-negative integer.") from exc

    if parsed < 0:
        raise ValueError(f"{source} must be non-negative, got {parsed}.")
    return parsed


def _validate_descriptor_count(expected_count: int, descriptors: np.ndarray | None, source: Any) -> None:
    actual_count = 0 if descriptors is None else int(descriptors.shape[0])
    if expected_count != actual_count:
        raise ValueError(
            f"{source}: descriptor rows={actual_count} do not match num_keypoints={expected_count}."
        )


def _validate_file_descriptors(
    descriptors: np.ndarray,
    descriptors_present: bool,
    method: str,
    num_keypoints: int,
    descriptor_shape: tuple[int, ...],
    descriptor_dtype: str | None,
    source: Any,
) -> np.ndarray | None:
    normalized_descriptors = _normalize_descriptor_array(descriptors, f"{source}: descriptors")
    if not descriptors_present:
        if normalized_descriptors is not None:
            raise ValueError(f"{source}: descriptors_present is 0 but descriptors are not empty.")
        _validate_empty_descriptor_metadata(descriptor_shape, descriptor_dtype, method, source)
        return None

    if normalized_descriptors is None:
        raise ValueError(f"{source}: descriptors_present is 1 but descriptors are empty.")
    if normalized_descriptors.ndim != 2:
        raise ValueError(f"{source}: descriptors must be a 2D array, got shape={normalized_descriptors.shape}.")
    if normalized_descriptors.shape[0] != num_keypoints:
        raise ValueError(
            f"{source}: descriptors rows={normalized_descriptors.shape[0]} do not match num_keypoints={num_keypoints}."
        )

    spec = _METHOD_SPECS[method]
    actual_shape = tuple(int(dim) for dim in normalized_descriptors.shape)
    actual_dtype = str(normalized_descriptors.dtype)

    if actual_dtype != spec.dtype:
        raise ValueError(f"{source}: {method} descriptors must have dtype {spec.dtype}, got {actual_dtype}.")
    if actual_shape[1] != spec.dim:
        raise ValueError(f"{source}: {method} descriptors must have dimension {spec.dim}, got {actual_shape[1]}.")
    if descriptor_shape != actual_shape:
        raise ValueError(
            f"{source}: descriptor_shape metadata {descriptor_shape} does not match descriptors.shape {actual_shape}."
        )
    if descriptor_dtype != actual_dtype:
        raise ValueError(
            f"{source}: descriptor_dtype metadata {descriptor_dtype!r} does not match descriptors dtype {actual_dtype!r}."
        )

    return normalized_descriptors


def _validate_empty_descriptor_metadata(
    descriptor_shape: tuple[int, ...],
    descriptor_dtype: str | None,
    method: str,
    source: Any,
) -> None:
    spec = _METHOD_SPECS[method]

    if descriptor_dtype is not None and descriptor_dtype != spec.dtype:
        raise ValueError(
            f"{source}: empty {method} descriptor metadata must use dtype {spec.dtype} when provided, "
            f"got {descriptor_dtype!r}."
        )

    if not descriptor_shape:
        return
    if descriptor_shape == (0,):
        return
    if descriptor_shape == (0, 0):
        return
    if descriptor_shape == (0, spec.dim):
        return

    raise ValueError(
        f"{source}: empty {method} descriptor_shape metadata must be empty, (0,), (0, 0), or (0, {spec.dim}), "
        f"got {descriptor_shape}."
    )
