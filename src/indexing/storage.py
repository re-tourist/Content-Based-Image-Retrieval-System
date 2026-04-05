from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.npyio import NpzFile

from src.encoding.io import _normalize_method

from .types import InvertedIndexArtifact, TfidfStatsArtifact


_TFIDF_STATS_FILENAME = "tfidf_stats.npz"
_INDEX_FILENAME_PREFIX = "index_"
_SUPPORTED_WEIGHTING_MODES = ("tf", "tfidf")


def resolve_method_index_dir(index_root: str | Path, corpus_split: str, method: str) -> Path:
    """Return the canonical output directory for one method-specific index bundle."""
    resolved_root = _resolve_path_root(index_root)
    resolved_corpus_split = _normalize_label(corpus_split, "corpus_split")
    normalized_method = _normalize_method(method)
    return resolved_root / resolved_corpus_split / normalized_method


def locate_method_index_dir(index_dir: str | Path, method: str) -> Path:
    """Locate a method-specific index directory under an explicit index root."""
    resolved_dir = Path(index_dir).expanduser()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {resolved_dir}")
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"Index path is not a directory: {resolved_dir}")

    normalized_method = _normalize_method(method)
    if _has_index_payload(resolved_dir):
        return resolved_dir.resolve()

    direct_candidate = resolved_dir / normalized_method
    if direct_candidate.exists():
        if not direct_candidate.is_dir():
            raise NotADirectoryError(
                f"Index path for method {normalized_method} is not a directory: {direct_candidate}"
            )
        if _has_index_payload(direct_candidate):
            return direct_candidate.resolve()

    nested_candidates = [
        path
        for path in resolved_dir.rglob(normalized_method)
        if path.is_dir() and _has_index_payload(path)
    ]
    if len(nested_candidates) == 1:
        return nested_candidates[0].resolve()
    if len(nested_candidates) > 1:
        candidate_list = ", ".join(str(path) for path in nested_candidates)
        raise ValueError(
            f"Multiple index directories found for method {normalized_method} under {resolved_dir}: {candidate_list}."
        )

    raise FileNotFoundError(
        f"No saved index directory found for method {normalized_method} under {resolved_dir}."
    )


def resolve_tfidf_stats_path(index_dir: str | Path) -> Path:
    """Return the standard TF-IDF stats file path inside a method directory."""
    return _resolve_path_root(index_dir) / _TFIDF_STATS_FILENAME


def resolve_inverted_index_path(index_dir: str | Path, weighting_mode: str) -> Path:
    """Return the standard inverted-index file path inside a method directory."""
    normalized_weighting_mode = _normalize_weighting_mode(weighting_mode)
    return _resolve_path_root(index_dir) / f"{_INDEX_FILENAME_PREFIX}{normalized_weighting_mode}.npz"


def save_tfidf_stats(stats: TfidfStatsArtifact, output_dir: str | Path) -> Path:
    """Save TF-IDF corpus statistics as a compressed `.npz` artifact."""
    _validate_tfidf_stats_artifact(stats, "save_tfidf_stats")
    target_dir = _resolve_path_root(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    save_path = target_dir / _TFIDF_STATS_FILENAME

    np.savez_compressed(
        save_path,
        method=np.array(stats.method),
        corpus_split=np.array(stats.corpus_split),
        num_docs=np.array(stats.num_docs, dtype=np.int32),
        num_visual_words=np.array(stats.num_visual_words, dtype=np.int32),
        df=np.asarray(stats.df, dtype=np.int32),
        idf=np.asarray(stats.idf, dtype=np.float32),
    )
    return save_path


def load_tfidf_stats(stats_path: str | Path) -> TfidfStatsArtifact:
    """Load and validate TF-IDF corpus statistics."""
    resolved_path = Path(stats_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"TF-IDF stats artifact not found: {resolved_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"TF-IDF stats path is not a file: {resolved_path}")

    with np.load(resolved_path, allow_pickle=False) as data:
        _validate_required_keys(data, resolved_path, _required_tfidf_keys())
        stats = TfidfStatsArtifact(
            method=_read_required_string(data, "method", resolved_path),
            corpus_split=_read_required_string(data, "corpus_split", resolved_path),
            num_docs=_read_non_negative_int(data, "num_docs", resolved_path),
            num_visual_words=_read_positive_int(data, "num_visual_words", resolved_path),
            df=np.array(data["df"], copy=True),
            idf=np.array(data["idf"], copy=True),
        )

    _validate_tfidf_stats_artifact(stats, resolved_path)
    return stats


def save_inverted_index(index: InvertedIndexArtifact, output_dir: str | Path) -> Path:
    """Save one inverted index artifact as a compressed `.npz` file."""
    _validate_inverted_index_artifact(index, "save_inverted_index")
    target_dir = _resolve_path_root(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    save_path = target_dir / f"{_INDEX_FILENAME_PREFIX}{index.weighting_mode}.npz"

    np.savez_compressed(
        save_path,
        method=np.array(index.method),
        corpus_split=np.array(index.corpus_split),
        weighting_mode=np.array(index.weighting_mode),
        num_docs=np.array(index.num_docs, dtype=np.int32),
        num_visual_words=np.array(index.num_visual_words, dtype=np.int32),
        sample_ids=np.asarray(index.sample_ids, dtype=np.str_),
        indptr=np.asarray(index.indptr, dtype=np.int32),
        indices=np.asarray(index.indices, dtype=np.int32),
        data=np.asarray(index.data, dtype=np.float32),
        doc_norms=np.asarray(index.doc_norms, dtype=np.float32),
    )
    return save_path


def load_inverted_index(index_path: str | Path) -> InvertedIndexArtifact:
    """Load and validate one saved inverted index artifact."""
    resolved_path = Path(index_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Inverted index artifact not found: {resolved_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Inverted index path is not a file: {resolved_path}")

    with np.load(resolved_path, allow_pickle=False) as data:
        _validate_required_keys(data, resolved_path, _required_index_keys())
        index = InvertedIndexArtifact(
            method=_read_required_string(data, "method", resolved_path),
            corpus_split=_read_required_string(data, "corpus_split", resolved_path),
            weighting_mode=_normalize_weighting_mode(_read_required_string(data, "weighting_mode", resolved_path)),
            num_docs=_read_non_negative_int(data, "num_docs", resolved_path),
            num_visual_words=_read_positive_int(data, "num_visual_words", resolved_path),
            sample_ids=_read_string_sequence(data["sample_ids"], "sample_ids", resolved_path),
            indptr=np.array(data["indptr"], copy=True),
            indices=np.array(data["indices"], copy=True),
            data=np.array(data["data"], copy=True),
            doc_norms=np.array(data["doc_norms"], copy=True),
        )

    _validate_inverted_index_artifact(index, resolved_path)
    return index


def _validate_tfidf_stats_artifact(stats: TfidfStatsArtifact, source: Any) -> None:
    if not isinstance(stats, TfidfStatsArtifact):
        raise TypeError(f"{source}: stats must be a TfidfStatsArtifact, got {type(stats).__name__}.")

    stats.method = _normalize_method(stats.method)
    stats.corpus_split = _normalize_label(stats.corpus_split, "corpus_split")
    stats.num_docs = _read_non_negative_int_value(stats.num_docs, "num_docs")
    stats.num_visual_words = _read_positive_int_value(stats.num_visual_words, "num_visual_words")

    df = _coerce_int32_array(stats.df, f"{source}: df")
    idf = _coerce_float32_array(stats.idf, f"{source}: idf")

    if df.ndim != 1:
        raise ValueError(f"{source}: df must be a 1D array, got shape={df.shape}.")
    if idf.ndim != 1:
        raise ValueError(f"{source}: idf must be a 1D array, got shape={idf.shape}.")
    if df.shape[0] != stats.num_visual_words:
        raise ValueError(
            f"{source}: df length {df.shape[0]} does not match num_visual_words={stats.num_visual_words}."
        )
    if idf.shape[0] != stats.num_visual_words:
        raise ValueError(
            f"{source}: idf length {idf.shape[0]} does not match num_visual_words={stats.num_visual_words}."
        )
    if np.any(df < 0):
        raise ValueError(f"{source}: df values must be non-negative.")
    if stats.num_docs > 0 and np.any(df > stats.num_docs):
        raise ValueError(f"{source}: df values cannot exceed num_docs={stats.num_docs}.")
    if np.any(idf < 0.0):
        raise ValueError(f"{source}: idf values must be non-negative.")
    if not np.all(np.isfinite(idf)):
        raise ValueError(f"{source}: idf values must be finite.")

    stats.df = df
    stats.idf = idf


def _validate_inverted_index_artifact(index: InvertedIndexArtifact, source: Any) -> None:
    if not isinstance(index, InvertedIndexArtifact):
        raise TypeError(f"{source}: index must be an InvertedIndexArtifact, got {type(index).__name__}.")

    index.method = _normalize_method(index.method)
    index.corpus_split = _normalize_label(index.corpus_split, "corpus_split")
    index.weighting_mode = _normalize_weighting_mode(index.weighting_mode)
    index.num_docs = _read_non_negative_int_value(index.num_docs, "num_docs")
    index.num_visual_words = _read_positive_int_value(index.num_visual_words, "num_visual_words")

    sample_ids = _coerce_sample_ids(index.sample_ids, index.num_docs, source)
    indptr = _coerce_int32_array(index.indptr, f"{source}: indptr")
    indices = _coerce_int32_array(index.indices, f"{source}: indices")
    data = _coerce_float32_array(index.data, f"{source}: data")
    doc_norms = _coerce_float32_array(index.doc_norms, f"{source}: doc_norms")

    if indptr.ndim != 1:
        raise ValueError(f"{source}: indptr must be a 1D array, got shape={indptr.shape}.")
    if indices.ndim != 1:
        raise ValueError(f"{source}: indices must be a 1D array, got shape={indices.shape}.")
    if data.ndim != 1:
        raise ValueError(f"{source}: data must be a 1D array, got shape={data.shape}.")
    if doc_norms.ndim != 1:
        raise ValueError(f"{source}: doc_norms must be a 1D array, got shape={doc_norms.shape}.")

    if indptr.shape[0] != index.num_visual_words + 1:
        raise ValueError(
            f"{source}: indptr length {indptr.shape[0]} does not match num_visual_words+1={index.num_visual_words + 1}."
        )
    if indices.shape[0] != data.shape[0]:
        raise ValueError(f"{source}: indices length {indices.shape[0]} does not match data length {data.shape[0]}.")
    if doc_norms.shape[0] != index.num_docs:
        raise ValueError(f"{source}: doc_norms length {doc_norms.shape[0]} does not match num_docs={index.num_docs}.")

    if indptr.size > 0:
        if indptr[0] != 0:
            raise ValueError(f"{source}: indptr must start at 0.")
        if np.any(np.diff(indptr) < 0):
            raise ValueError(f"{source}: indptr must be non-decreasing.")
        if int(indptr[-1]) != indices.shape[0]:
            raise ValueError(
                f"{source}: indptr last value {int(indptr[-1])} does not match nnz={indices.shape[0]}."
            )

    if index.num_docs == 0:
        if indices.size != 0 or data.size != 0:
            raise ValueError(f"{source}: empty corpus index must not contain postings.")
    elif indices.size > 0:
        if np.any(indices < 0) or np.any(indices >= index.num_docs):
            raise ValueError(f"{source}: indices must reference doc rows in [0, {index.num_docs - 1}].")

    if np.any(data < 0.0):
        raise ValueError(f"{source}: posting weights must be non-negative.")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{source}: posting weights must be finite.")
    if np.any(doc_norms < 0.0):
        raise ValueError(f"{source}: doc_norms must be non-negative.")
    if not np.all(np.isfinite(doc_norms)):
        raise ValueError(f"{source}: doc_norms must be finite.")

    index.sample_ids = sample_ids
    index.indptr = indptr
    index.indices = indices
    index.data = data
    index.doc_norms = doc_norms


def _normalize_label(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string label.")

    normalized = value.strip()
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError(f"{field_name} must not contain path separators, got {value!r}.")
    return normalized


def _normalize_weighting_mode(weighting_mode: Any) -> str:
    if not isinstance(weighting_mode, str) or not weighting_mode.strip():
        raise ValueError("weighting_mode must be a non-empty string.")

    normalized = weighting_mode.strip().lower()
    if normalized not in _SUPPORTED_WEIGHTING_MODES:
        raise ValueError(
            f"Unsupported weighting_mode: {weighting_mode!r}. Supported modes: {', '.join(_SUPPORTED_WEIGHTING_MODES)}."
        )
    return normalized


def _required_tfidf_keys() -> tuple[str, ...]:
    return ("method", "corpus_split", "num_docs", "num_visual_words", "df", "idf")


def _required_index_keys() -> tuple[str, ...]:
    return (
        "method",
        "corpus_split",
        "weighting_mode",
        "num_docs",
        "num_visual_words",
        "sample_ids",
        "indptr",
        "indices",
        "data",
        "doc_norms",
    )


def _validate_required_keys(data: NpzFile, source: Path, required_keys: tuple[str, ...]) -> None:
    missing_keys = [key for key in required_keys if key not in data.files]
    if missing_keys:
        raise ValueError(f"{source}: missing required keys: {', '.join(missing_keys)}.")


def _read_required_string(data: NpzFile, key: str, source: Any) -> str:
    value = data[key]
    if value.ndim != 0:
        raise ValueError(f"{source}: key '{key}' must be a scalar string value.")

    scalar = value.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    if not isinstance(scalar, str) or not scalar.strip():
        raise ValueError(f"{source}: key '{key}' must be a non-empty string.")
    return scalar.strip()


def _read_non_negative_int(data: NpzFile, key: str, source: Any) -> int:
    value = data[key]
    if value.ndim != 0:
        raise ValueError(f"{source}: key '{key}' must be a scalar integer value.")
    return _read_non_negative_int_value(value.item(), f"{source}: key '{key}'")


def _read_positive_int(data: NpzFile, key: str, source: Any) -> int:
    value = data[key]
    if value.ndim != 0:
        raise ValueError(f"{source}: key '{key}' must be a scalar integer value.")
    return _read_positive_int_value(value.item(), f"{source}: key '{key}'")


def _read_non_negative_int_value(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative, got {parsed}.")
    return parsed


def _read_positive_int_value(value: Any, field_name: str) -> int:
    parsed = _read_non_negative_int_value(value, field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than zero, got {parsed}.")
    return parsed


def _coerce_int32_array(value: Any, source: Any) -> np.ndarray:
    array = np.array(value, copy=True)
    if array.dtype != np.int32:
        raise ValueError(f"{source} must use dtype int32, got {array.dtype}.")
    return array


def _coerce_float32_array(value: Any, source: Any) -> np.ndarray:
    array = np.array(value, copy=True)
    if array.dtype != np.float32:
        raise ValueError(f"{source} must use dtype float32, got {array.dtype}.")
    return array


def _read_string_sequence(value: Any, field_name: str, source: Any) -> tuple[str, ...]:
    array = np.array(value, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{source}: key '{field_name}' must be a 1D string array, got shape={array.shape}.")

    sample_ids: list[str] = []
    for item in array.tolist():
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{source}: key '{field_name}' must contain non-empty strings.")
        sample_ids.append(item.strip())
    return tuple(sample_ids)


def _coerce_sample_ids(sample_ids: Any, num_docs: int, source: Any) -> tuple[str, ...]:
    if isinstance(sample_ids, tuple):
        resolved = sample_ids
    elif isinstance(sample_ids, list):
        resolved = tuple(sample_ids)
    else:
        array = np.array(sample_ids, copy=True)
        if array.ndim != 1:
            raise ValueError(f"{source}: sample_ids must be a 1D sequence, got shape={array.shape}.")
        resolved = tuple(array.tolist())

    if len(resolved) != num_docs:
        raise ValueError(f"{source}: sample_ids length {len(resolved)} does not match num_docs={num_docs}.")

    normalized: list[str] = []
    for sample_id in resolved:
        if isinstance(sample_id, bytes):
            sample_id = sample_id.decode("utf-8")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"{source}: sample_ids must contain non-empty strings.")
        normalized.append(sample_id.strip())
    return tuple(normalized)


def _resolve_path_root(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = resolved.resolve()
    return resolved


def _has_index_payload(index_dir: Path) -> bool:
    if not index_dir.exists() or not index_dir.is_dir():
        return False
    if (index_dir / _TFIDF_STATS_FILENAME).is_file():
        return True
    for weighting_mode in _SUPPORTED_WEIGHTING_MODES:
        if (index_dir / f"{_INDEX_FILENAME_PREFIX}{weighting_mode}.npz").is_file():
            return True
    return False

