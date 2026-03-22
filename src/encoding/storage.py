from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.npyio import NpzFile

from .io import _normalize_method


_REQUIRED_ENCODED_KEYS = (
    "sample_id",
    "method",
    "encoding_type",
    "num_visual_words",
    "histogram",
    "histogram_dtype",
    "num_descriptors",
    "descriptors_present",
    "codebook_path",
    "normalized",
)


@dataclass(slots=True)
class EncodedFeatureResult:
    """Stable persisted representation for one encoded BoW feature."""

    sample_id: str | None
    method: str
    encoding_type: str
    num_visual_words: int
    histogram: np.ndarray
    histogram_dtype: str
    num_descriptors: int
    descriptors_present: bool
    codebook_path: str | None
    normalized: bool = False



def save_encoded_feature(encoded_feature: EncodedFeatureResult, output_dir: str | Path) -> Path:
    """Save one encoded feature artifact as a compressed `.npz` file."""
    _validate_encoded_feature(encoded_feature, "save_encoded_feature")
    if encoded_feature.sample_id is None:
        raise ValueError("save_encoded_feature: sample_id must be provided for persistence.")

    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    save_path = target_dir / f"{_make_safe_encoded_name(encoded_feature.sample_id)}.npz"

    np.savez_compressed(
        save_path,
        sample_id=np.array(encoded_feature.sample_id),
        method=np.array(encoded_feature.method),
        encoding_type=np.array(encoded_feature.encoding_type),
        num_visual_words=np.array(encoded_feature.num_visual_words, dtype=np.int32),
        histogram=np.asarray(encoded_feature.histogram),
        histogram_dtype=np.array(encoded_feature.histogram_dtype),
        num_descriptors=np.array(encoded_feature.num_descriptors, dtype=np.int32),
        descriptors_present=np.array(int(encoded_feature.descriptors_present), dtype=np.uint8),
        codebook_path=np.array(encoded_feature.codebook_path or ""),
        normalized=np.array(int(encoded_feature.normalized), dtype=np.uint8),
    )
    return save_path



def load_encoded_feature(encoded_path: str | Path) -> EncodedFeatureResult:
    """Load and validate one saved encoded feature artifact."""
    resolved_path = Path(encoded_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Encoded feature artifact not found: {resolved_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Encoded feature path is not a file: {resolved_path}")

    with np.load(resolved_path, allow_pickle=False) as data:
        _validate_required_keys(data, resolved_path)
        result = EncodedFeatureResult(
            sample_id=_read_optional_string(data, "sample_id", resolved_path),
            method=_normalize_method(_read_required_string(data, "method", resolved_path)),
            encoding_type=_read_required_string(data, "encoding_type", resolved_path),
            num_visual_words=_read_non_negative_int(data, "num_visual_words", resolved_path),
            histogram=np.array(data["histogram"], copy=True),
            histogram_dtype=_read_required_string(data, "histogram_dtype", resolved_path),
            num_descriptors=_read_non_negative_int(data, "num_descriptors", resolved_path),
            descriptors_present=_read_required_flag(data, "descriptors_present", resolved_path),
            codebook_path=_read_optional_string(data, "codebook_path", resolved_path),
            normalized=_read_required_flag(data, "normalized", resolved_path),
        )

    _validate_encoded_feature(result, resolved_path)
    return result



def _validate_encoded_feature(encoded_feature: EncodedFeatureResult, source: Any) -> None:
    if not isinstance(encoded_feature, EncodedFeatureResult):
        raise TypeError(
            f"{source}: encoded_feature must be an EncodedFeatureResult, got {type(encoded_feature).__name__}."
        )

    if encoded_feature.sample_id is not None:
        if not isinstance(encoded_feature.sample_id, str) or not encoded_feature.sample_id.strip():
            raise ValueError(f"{source}: sample_id must be a non-empty string when provided.")
        encoded_feature.sample_id = encoded_feature.sample_id.strip()

    encoded_feature.method = _normalize_method(encoded_feature.method)
    if encoded_feature.encoding_type != "bow":
        raise ValueError(f"{source}: encoding_type must be 'bow', got {encoded_feature.encoding_type!r}.")
    encoded_feature.num_visual_words = _require_non_negative_int(encoded_feature.num_visual_words, "num_visual_words")
    encoded_feature.num_descriptors = _require_non_negative_int(encoded_feature.num_descriptors, "num_descriptors")
    encoded_feature.descriptors_present = bool(encoded_feature.descriptors_present)
    encoded_feature.normalized = bool(encoded_feature.normalized)

    histogram = encoded_feature.histogram
    if not isinstance(histogram, np.ndarray):
        raise TypeError(f"{source}: histogram must be a numpy.ndarray, got {type(histogram).__name__}.")
    if histogram.ndim != 1:
        raise ValueError(f"{source}: histogram must be a 1D array, got shape={histogram.shape}.")
    if histogram.shape[0] != encoded_feature.num_visual_words:
        raise ValueError(
            f"{source}: histogram length {histogram.shape[0]} does not match num_visual_words={encoded_feature.num_visual_words}."
        )

    actual_dtype = str(histogram.dtype)
    if encoded_feature.histogram_dtype != actual_dtype:
        raise ValueError(
            f"{source}: histogram_dtype {encoded_feature.histogram_dtype!r} does not match histogram dtype {actual_dtype!r}."
        )

    if encoded_feature.normalized:
        if actual_dtype != "float32":
            raise ValueError(f"{source}: normalized BoW histograms must use dtype float32, got {actual_dtype}.")
    else:
        if actual_dtype != "int32":
            raise ValueError(f"{source}: raw-count BoW histograms must use dtype int32, got {actual_dtype}.")

    if encoded_feature.codebook_path is not None:
        if not isinstance(encoded_feature.codebook_path, str) or not encoded_feature.codebook_path.strip():
            raise ValueError(f"{source}: codebook_path must be a non-empty string when provided.")
        encoded_feature.codebook_path = Path(encoded_feature.codebook_path).expanduser().as_posix()



def _validate_required_keys(data: NpzFile, source: Path) -> None:
    missing_keys = [key for key in _REQUIRED_ENCODED_KEYS if key not in data.files]
    if missing_keys:
        raise ValueError(f"{source}: missing required encoded feature keys: {', '.join(missing_keys)}.")



def _read_required_string(data: NpzFile, key: str, source: Any) -> str:
    value = data[key]
    if value.ndim != 0:
        raise ValueError(f"{source}: key '{key}' must be a scalar string value.")
    scalar = value.item()
    if not isinstance(scalar, str) or not scalar.strip():
        raise ValueError(f"{source}: key '{key}' must be a non-empty string.")
    return scalar.strip()



def _read_optional_string(data: NpzFile, key: str, source: Any) -> str | None:
    value = data[key]
    if value.ndim != 0:
        raise ValueError(f"{source}: key '{key}' must be a scalar string value.")
    scalar = value.item()
    if not isinstance(scalar, str):
        raise ValueError(f"{source}: key '{key}' must be a string value.")
    stripped = scalar.strip()
    return stripped or None



def _read_non_negative_int(data: NpzFile, key: str, source: Any) -> int:
    value = data[key]
    if value.ndim != 0:
        raise ValueError(f"{source}: key '{key}' must be a scalar integer value.")
    return _require_non_negative_int(value.item(), key)



def _read_required_flag(data: NpzFile, key: str, source: Any) -> bool:
    parsed = _read_non_negative_int(data, key, source)
    if parsed not in (0, 1):
        raise ValueError(f"{source}: key '{key}' must be 0 or 1, got {parsed}.")
    return bool(parsed)



def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative, got {parsed}.")
    return parsed



def _make_safe_encoded_name(sample_id: str) -> str:
    normalized = sample_id.replace("\\", "/")
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "__", normalized)
    normalized = normalized.strip("._")
    return normalized or "encoded"

