from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np


def compute_document_frequency(histograms: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Count in how many documents each visual word appears."""
    matrix = _coerce_histogram_matrix(histograms)
    if matrix.shape[0] == 0:
        raise ValueError("compute_document_frequency: at least one histogram is required.")

    return np.count_nonzero(matrix > 0, axis=0).astype(np.int32, copy=False)


def compute_idf(df: np.ndarray, num_docs: int) -> np.ndarray:
    """Compute smooth IDF with log((N + 1) / (df + 1)) + 1."""
    df_array = np.asarray(df)
    if df_array.ndim != 1:
        raise ValueError(f"compute_idf: df must be a 1D array, got shape={df_array.shape}.")
    if isinstance(num_docs, bool):
        raise ValueError("compute_idf: num_docs must be a non-negative integer.")

    try:
        resolved_num_docs = int(num_docs)
    except (TypeError, ValueError) as exc:
        raise ValueError("compute_idf: num_docs must be a non-negative integer.") from exc

    if resolved_num_docs < 0:
        raise ValueError(f"compute_idf: num_docs must be non-negative, got {resolved_num_docs}.")
    if df_array.size == 0:
        return np.empty((0,), dtype=np.float32)

    if np.any(df_array < 0):
        raise ValueError("compute_idf: df values must be non-negative.")
    if np.any(df_array > resolved_num_docs):
        raise ValueError("compute_idf: df values cannot exceed num_docs.")

    idf = np.log((resolved_num_docs + 1.0) / (df_array.astype(np.float64) + 1.0)) + 1.0
    return idf.astype(np.float32, copy=False)


def apply_tfidf_weighting(histogram: np.ndarray, idf: np.ndarray) -> np.ndarray:
    """Apply raw-count TF weighting followed by IDF scaling."""
    tf_vector = np.asarray(histogram)
    idf_vector = np.asarray(idf)
    if tf_vector.ndim != 1:
        raise ValueError(f"apply_tfidf_weighting: histogram must be 1D, got shape={tf_vector.shape}.")
    if idf_vector.ndim != 1:
        raise ValueError(f"apply_tfidf_weighting: idf must be 1D, got shape={idf_vector.shape}.")
    if tf_vector.shape != idf_vector.shape:
        raise ValueError(
            "apply_tfidf_weighting: histogram and idf must have the same shape, "
            f"got {tf_vector.shape} and {idf_vector.shape}."
        )

    counts = tf_vector.astype(np.float32, copy=False)
    scaling = idf_vector.astype(np.float32, copy=False)
    return counts * scaling


def normalize_vector(vector: np.ndarray, mode: Literal["none", "l1", "l2"] = "l2") -> np.ndarray:
    """Normalize a vector for retrieval scoring."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"normalize_vector: vector must be 1D, got shape={arr.shape}.")

    normalized_mode = mode.lower()
    if normalized_mode == "none":
        return arr.copy()
    if normalized_mode == "l1":
        denom = float(np.sum(np.abs(arr)))
    elif normalized_mode == "l2":
        denom = float(np.linalg.norm(arr))
    else:
        raise ValueError("normalize_vector: mode must be one of 'none', 'l1', or 'l2'.")

    if denom == 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / denom


def _coerce_histogram_matrix(histograms: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    if isinstance(histograms, np.ndarray):
        matrix = np.asarray(histograms)
        if matrix.ndim != 2:
            raise ValueError(
                f"compute_document_frequency: histograms must be a 2D array, got shape={matrix.shape}."
            )
        return matrix

    histogram_list = [np.asarray(histogram) for histogram in histograms]
    if not histogram_list:
        raise ValueError("compute_document_frequency: histograms cannot be empty.")

    first_shape = histogram_list[0].shape
    if histogram_list[0].ndim != 1:
        raise ValueError(
            f"compute_document_frequency: histograms must be 1D arrays, got shape={histogram_list[0].shape}."
        )
    for index, histogram in enumerate(histogram_list[1:], start=1):
        if histogram.ndim != 1:
            raise ValueError(
                f"compute_document_frequency: histogram at position {index} must be 1D, got shape={histogram.shape}."
            )
        if histogram.shape != first_shape:
            raise ValueError(
                "compute_document_frequency: all histograms must have the same length, "
                f"got {first_shape} and {histogram.shape}."
            )

    return np.stack(histogram_list, axis=0)

