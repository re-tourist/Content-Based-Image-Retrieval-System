from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import PRCurvePoint, RetrievedResult, SummaryPRCurvePoint


_CANONICAL_PREFIXES: tuple[str, ...] = (
    "data/",
    "train/image/",
    "train/",
    "test/",
    "gallery/",
    "query/",
)

DEFAULT_PR_RECALL_GRID = np.linspace(0.0, 1.0, 101, dtype=np.float32)


def normalize_canonical_sample_id(sample_id: str) -> str:
    """Normalize a sample id into the canonical evaluation key.

    The retrieval evaluation layer matches split-file entries and encoded-artifact
    sample ids through this key. It strips known dataset prefixes such as
    ``data/``, ``train/``, ``test/``, ``gallery/`` and ``query/`` while keeping the
    class folder and file path intact.
    """
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id must be a non-empty string.")

    normalized = sample_id.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    while normalized.startswith("/"):
        normalized = normalized[1:]

    changed = True
    while changed:
        changed = False
        for prefix in _CANONICAL_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
                break

    normalized = normalized.strip("/")
    if not normalized:
        raise ValueError(f"Canonical sample id normalization produced an empty value for {sample_id!r}.")
    return normalized


def extract_class_label(sample_id: str) -> str:
    """Return the class-folder label from a canonical sample id."""
    normalized = normalize_canonical_sample_id(sample_id)
    label = normalized.split("/", 1)[0].strip()
    if not label:
        raise ValueError(f"Could not extract a class label from sample id {sample_id!r}.")
    return label


def compute_precision_at_k(relevance_flags: Sequence[bool], requested_k: int) -> float:
    """Compute precision at k using raw relevance flags."""
    effective_k = _validate_requested_k(requested_k)
    if effective_k == 0:
        return 0.0

    top_flags = list(relevance_flags[:effective_k])
    return float(np.count_nonzero(top_flags) / effective_k)


def compute_recall_at_k(
    relevance_flags: Sequence[bool],
    requested_k: int,
    relevant_total: int,
) -> float | None:
    """Compute recall at k using raw relevance flags.

    Returns ``None`` when no relevant items exist for the query.
    """
    if relevant_total < 0:
        raise ValueError(f"relevant_total must be non-negative, got {relevant_total}.")
    if relevant_total == 0:
        return None

    effective_k = _validate_requested_k(requested_k)
    top_flags = list(relevance_flags[:effective_k])
    return float(np.count_nonzero(top_flags) / relevant_total)


def compute_average_precision(
    relevance_flags: Sequence[bool],
    relevant_total: int | None = None,
) -> float | None:
    """Compute average precision over the full ranked list.

    Returns ``None`` when the query has no relevant gallery items.
    """
    if relevant_total is None:
        relevant_total = int(np.count_nonzero(list(relevance_flags)))
    if relevant_total < 0:
        raise ValueError(f"relevant_total must be non-negative, got {relevant_total}.")
    if relevant_total == 0:
        return None

    hit_count = 0
    precision_sum = 0.0
    for rank, is_relevant in enumerate(relevance_flags, start=1):
        if not bool(is_relevant):
            continue
        hit_count += 1
        precision_sum += hit_count / rank
    return float(precision_sum / relevant_total)


def build_prefix_pr_curve(
    ranked_results: Sequence[RetrievedResult],
    relevant_total: int,
) -> list[PRCurvePoint]:
    """Build a prefix precision/recall curve for one query."""
    if relevant_total < 0:
        raise ValueError(f"relevant_total must be non-negative, got {relevant_total}.")
    if relevant_total == 0:
        return []

    hit_count = 0
    curve_points: list[PRCurvePoint] = []
    for rank, result in enumerate(ranked_results, start=1):
        if result.relevant:
            hit_count += 1
        curve_points.append(
            PRCurvePoint(
                rank=rank,
                precision=float(hit_count / rank),
                recall=float(hit_count / relevant_total),
                relevant=bool(result.relevant),
                sample_id=result.sample_id,
                artifact_sample_id=result.artifact_sample_id,
            )
        )
    return curve_points


def build_macro_pr_curve(
    query_pr_curves: Sequence[Sequence[PRCurvePoint]],
    recall_grid: Sequence[float] | np.ndarray | None = None,
) -> list[SummaryPRCurvePoint]:
    """Build a macro-averaged interpolated PR curve.

    The interpolation follows the standard retrieval convention: for each recall
    threshold, the precision is the maximum precision achieved at any point whose
    recall is at least that threshold.
    """
    if recall_grid is None:
        grid = DEFAULT_PR_RECALL_GRID
    else:
        grid = np.asarray(list(recall_grid), dtype=np.float32)
        if grid.ndim != 1:
            raise ValueError(f"recall_grid must be one-dimensional, got shape={grid.shape}.")

    valid_curves = [list(curve) for curve in query_pr_curves if curve]
    support = len(valid_curves)

    summary_points: list[SummaryPRCurvePoint] = []
    for threshold in grid.tolist():
        precision_values = [
            _interpolated_precision_at_recall(curve, float(threshold))
            for curve in valid_curves
        ]
        precision = float(np.mean(precision_values)) if precision_values else 0.0
        summary_points.append(
            SummaryPRCurvePoint(
                recall=float(threshold),
                precision=precision,
                support=support,
            )
        )
    return summary_points


def _interpolated_precision_at_recall(curve_points: Sequence[PRCurvePoint], threshold: float) -> float:
    precisions = [point.precision for point in curve_points if point.recall >= threshold]
    if not precisions:
        return 0.0
    return float(max(precisions))


def _validate_requested_k(requested_k: int) -> int:
    if isinstance(requested_k, bool):
        raise ValueError("requested_k must be a positive integer.")
    try:
        effective_k = int(requested_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("requested_k must be a positive integer.") from exc
    if effective_k <= 0:
        raise ValueError(f"requested_k must be greater than zero, got {effective_k}.")
    return effective_k
