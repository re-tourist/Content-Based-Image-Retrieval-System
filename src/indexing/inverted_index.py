from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from src.encoding import EncodedFeatureResult, load_encoded_feature
from src.encoding.io import _normalize_method

from .tfidf import apply_tfidf_weighting, compute_document_frequency, compute_idf
from .types import (
    InvertedIndexArtifact,
    MethodIndexArtifacts,
    PostingEntry,
    RankedResult,
    ScoringMode,
    TfidfStatsArtifact,
    WeightingMode,
)


_DEFAULT_WEIGHTING_MODES: tuple[WeightingMode, ...] = ("tf", "tfidf")
_SUPPORTED_SCORING_MODES: tuple[str, ...] = ("dot", "cosine")


def iter_encoded_feature_paths(encoded_dir: str | Path) -> Iterable[Path]:
    """Yield encoded artifact paths in a stable sorted order."""
    resolved_dir = Path(encoded_dir).expanduser()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Encoded feature directory not found: {resolved_dir}")
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"Encoded feature path is not a directory: {resolved_dir}")

    yield from sorted(path for path in resolved_dir.glob("*.npz") if path.is_file())


def load_encoded_feature_records(
    encoded_dir: str | Path,
    methods: Iterable[str] | None = None,
) -> list[EncodedFeatureResult]:
    """Load encoded feature artifacts from a directory."""
    normalized_methods = _normalize_method_filter(methods)
    records: list[EncodedFeatureResult] = []

    for encoded_path in iter_encoded_feature_paths(encoded_dir):
        record = load_encoded_feature(encoded_path)
        if normalized_methods is not None and record.method not in normalized_methods:
            continue
        records.append(record)

    if not records:
        resolved_dir = Path(encoded_dir).expanduser()
        requested = "all methods" if normalized_methods is None else ", ".join(sorted(normalized_methods))
        raise ValueError(f"No encoded feature artifacts matched for {requested} under {resolved_dir}.")

    return records


def build_inverted_index_from_records(
    records: Sequence[EncodedFeatureResult],
    *,
    corpus_split: str,
    weighting_modes: Iterable[WeightingMode] | None = None,
) -> MethodIndexArtifacts:
    """Build TF / TF-IDF inverted indexes for one method-specific record set."""
    record_list = list(records)
    if not record_list:
        raise ValueError("build_inverted_index_from_records: records cannot be empty.")

    normalized_weighting_modes = _normalize_weighting_modes(weighting_modes)
    normalized_corpus_split = _normalize_corpus_split(corpus_split)

    method: str | None = None
    num_visual_words: int | None = None
    sample_ids: list[str] = []
    histograms: list[np.ndarray] = []

    for position, record in enumerate(record_list):
        validated = _validate_index_record(
            record,
            expected_method=method,
            expected_num_visual_words=num_visual_words,
            source=f"build_inverted_index_from_records[{position}]",
        )
        if method is None:
            method = validated.method
        if num_visual_words is None:
            num_visual_words = validated.num_visual_words

        sample_ids.append(validated.sample_id)
        histograms.append(np.asarray(validated.histogram, dtype=np.int32))

    assert method is not None
    assert num_visual_words is not None

    df = compute_document_frequency(histograms)
    idf = compute_idf(df, len(histograms))
    tfidf_stats = TfidfStatsArtifact(
        method=method,
        corpus_split=normalized_corpus_split,
        num_docs=len(histograms),
        num_visual_words=num_visual_words,
        df=df,
        idf=idf,
    )

    indexes: dict[WeightingMode, InvertedIndexArtifact] = {}
    for weighting_mode in normalized_weighting_modes:
        indexes[weighting_mode] = _build_weighted_index(
            sample_ids=tuple(sample_ids),
            histograms=histograms,
            tfidf_stats=tfidf_stats,
            weighting_mode=weighting_mode,
        )

    return MethodIndexArtifacts(
        method=method,
        corpus_split=normalized_corpus_split,
        tfidf_stats=tfidf_stats,
        indexes=indexes,
    )


def build_inverted_index_from_encoded_dir(
    encoded_dir: str | Path,
    *,
    corpus_split: str,
    methods: Iterable[str] | None = None,
    weighting_modes: Iterable[WeightingMode] | None = None,
) -> dict[str, MethodIndexArtifacts]:
    """Build method-specific inverted indexes from an encoded-artifact directory."""
    records = load_encoded_feature_records(encoded_dir, methods=methods)
    grouped: dict[str, list[EncodedFeatureResult]] = defaultdict(list)
    for record in records:
        grouped[record.method].append(record)

    normalized_methods = _normalize_method_filter(methods)
    if normalized_methods is not None:
        missing_methods = sorted(method for method in normalized_methods if method not in grouped)
        if missing_methods:
            raise FileNotFoundError(
                f"Missing encoded feature artifact(s) for method(s): {', '.join(missing_methods)} under {Path(encoded_dir).expanduser()}."
            )

    return {
        method: build_inverted_index_from_records(
            method_records,
            corpus_split=corpus_split,
            weighting_modes=weighting_modes,
        )
        for method, method_records in sorted(grouped.items())
    }


def score_query_against_index(
    query: EncodedFeatureResult,
    index: InvertedIndexArtifact,
    *,
    tfidf_stats: TfidfStatsArtifact | None = None,
    scoring_mode: ScoringMode = "dot",
) -> np.ndarray:
    """Compute dense document scores for one query artifact."""
    normalized_scoring_mode = _normalize_scoring_mode(scoring_mode)
    validated_query = _validate_query_record(query, index, source="score_query_against_index")
    if index.weighting_mode == "tf" and normalized_scoring_mode != "dot":
        raise ValueError("score_query_against_index: tf weighting only supports dot scoring.")
    if index.weighting_mode == "tfidf" and normalized_scoring_mode != "cosine":
        raise ValueError("score_query_against_index: tfidf weighting only supports cosine scoring.")

    if index.weighting_mode == "tfidf":
        if tfidf_stats is None:
            raise ValueError("score_query_against_index: tfidf_stats is required for tfidf + cosine scoring.")
        _validate_tfidf_stats_for_query(tfidf_stats, index, source="score_query_against_index")

    query_weights = _build_query_weights(validated_query, index, tfidf_stats)
    scores = np.zeros(index.num_docs, dtype=np.float64)
    query_histogram = np.asarray(validated_query.histogram, dtype=np.int32)
    nonzero_terms = np.flatnonzero(query_histogram)

    for term_index in nonzero_terms:
        start = int(index.indptr[term_index])
        end = int(index.indptr[term_index + 1])
        if start == end:
            continue

        query_weight = float(query_weights[term_index])
        if query_weight == 0.0:
            continue

        doc_indices = index.indices[start:end].astype(np.int64, copy=False)
        posting_weights = index.data[start:end].astype(np.float64, copy=False)
        scores[doc_indices] += query_weight * posting_weights

    if index.weighting_mode == "tfidf" and normalized_scoring_mode == "cosine":
        query_norm = float(np.linalg.norm(query_weights.astype(np.float64, copy=False)))
        if query_norm == 0.0:
            return np.zeros(index.num_docs, dtype=np.float64)
        denom = index.doc_norms.astype(np.float64, copy=False) * query_norm
        scores = np.divide(scores, denom, out=np.zeros_like(scores), where=denom > 0)

    return scores


def search_inverted_index(
    query: EncodedFeatureResult,
    index: InvertedIndexArtifact,
    *,
    tfidf_stats: TfidfStatsArtifact | None = None,
    scoring_mode: ScoringMode = "dot",
    top_k: int = 10,
) -> list[RankedResult]:
    """Return the top-k ranked results for one query artifact."""
    if isinstance(top_k, bool):
        raise ValueError("search_inverted_index: top_k must be a positive integer.")
    try:
        resolved_top_k = int(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("search_inverted_index: top_k must be a positive integer.") from exc
    if resolved_top_k <= 0:
        raise ValueError(f"search_inverted_index: top_k must be greater than zero, got {resolved_top_k}.")

    scores = score_query_against_index(query, index, tfidf_stats=tfidf_stats, scoring_mode=scoring_mode)
    if scores.size == 0:
        return []

    order = np.lexsort((np.arange(scores.size), -scores))
    limit = min(resolved_top_k, scores.size)
    ranked_results: list[RankedResult] = []
    for rank, doc_index in enumerate(order[:limit], start=1):
        ranked_results.append(
            RankedResult(
                sample_id=index.sample_ids[int(doc_index)],
                score=float(scores[int(doc_index)]),
                rank=rank,
                doc_index=int(doc_index),
            )
        )
    return ranked_results


def _build_weighted_index(
    *,
    sample_ids: Sequence[str],
    histograms: Sequence[np.ndarray],
    tfidf_stats: TfidfStatsArtifact,
    weighting_mode: WeightingMode,
) -> InvertedIndexArtifact:
    normalized_weighting_mode = _normalize_weighting_mode(weighting_mode)
    num_docs = len(histograms)
    num_visual_words = tfidf_stats.num_visual_words

    postings_by_term: list[list[PostingEntry]] = [[] for _ in range(num_visual_words)]
    doc_norms = np.zeros(num_docs, dtype=np.float32)

    for doc_index, histogram in enumerate(histograms):
        if histogram.shape[0] != num_visual_words:
            raise ValueError(
                "_build_weighted_index: histogram length mismatch against num_visual_words="
                f"{num_visual_words}."
            )

        if normalized_weighting_mode == "tf":
            weights = histogram.astype(np.float32, copy=False)
        else:
            weights = apply_tfidf_weighting(histogram, tfidf_stats.idf)

        doc_norms[doc_index] = float(np.linalg.norm(weights.astype(np.float32, copy=False)))
        nonzero_terms = np.flatnonzero(histogram)
        for term_index in nonzero_terms:
            postings_by_term[term_index].append(
                PostingEntry(
                    sample_id=sample_ids[doc_index],
                    doc_index=doc_index,
                    weight=float(weights[term_index]),
                    term_frequency=int(histogram[term_index]),
                )
            )

    indptr = np.zeros(num_visual_words + 1, dtype=np.int32)
    indices: list[int] = []
    data: list[float] = []
    offset = 0
    for term_index, postings in enumerate(postings_by_term):
        indptr[term_index] = offset
        for posting in postings:
            indices.append(posting.doc_index)
            data.append(posting.weight)
        offset += len(postings)
    indptr[num_visual_words] = offset

    return InvertedIndexArtifact(
        method=tfidf_stats.method,
        corpus_split=tfidf_stats.corpus_split,
        weighting_mode=normalized_weighting_mode,
        num_docs=num_docs,
        num_visual_words=num_visual_words,
        sample_ids=tuple(sample_ids),
        indptr=indptr,
        indices=np.asarray(indices, dtype=np.int32),
        data=np.asarray(data, dtype=np.float32),
        doc_norms=doc_norms,
    )


def _build_query_weights(
    query: EncodedFeatureResult,
    index: InvertedIndexArtifact,
    tfidf_stats: TfidfStatsArtifact | None,
) -> np.ndarray:
    histogram = np.asarray(query.histogram, dtype=np.int32)
    if index.weighting_mode == "tf":
        return histogram.astype(np.float32, copy=False)
    assert tfidf_stats is not None
    return apply_tfidf_weighting(histogram, tfidf_stats.idf)


def _validate_index_record(
    record: EncodedFeatureResult,
    *,
    expected_method: str | None,
    expected_num_visual_words: int | None,
    source: str,
) -> EncodedFeatureResult:
    if not isinstance(record, EncodedFeatureResult):
        raise TypeError(f"{source}: record must be an EncodedFeatureResult, got {type(record).__name__}.")

    if record.sample_id is None or not isinstance(record.sample_id, str) or not record.sample_id.strip():
        raise ValueError(f"{source}: sample_id must be a non-empty string.")
    if record.encoding_type != "bow":
        raise ValueError(f"{source}: encoding_type must be 'bow', got {record.encoding_type!r}.")
    if record.normalized:
        raise ValueError(
            f"{source}: normalized BoW artifacts are not supported in Milestone 5. "
            "Please use raw-count BoW artifacts with normalized == False."
        )
    if not isinstance(record.histogram, np.ndarray):
        raise TypeError(f"{source}: histogram must be a numpy.ndarray, got {type(record.histogram).__name__}.")
    if record.histogram.ndim != 1:
        raise ValueError(f"{source}: histogram must be a 1D array, got shape={record.histogram.shape}.")
    if record.histogram_dtype != "int32":
        raise ValueError(
            f"{source}: histogram_dtype must be 'int32' for raw-count BoW artifacts, got {record.histogram_dtype!r}."
        )
    if record.histogram.dtype != np.int32:
        raise ValueError(
            f"{source}: histogram dtype must be int32 for raw-count BoW artifacts, got {record.histogram.dtype}."
        )
    if record.num_visual_words <= 0:
        raise ValueError(f"{source}: num_visual_words must be greater than zero, got {record.num_visual_words}.")
    if record.histogram.shape[0] != record.num_visual_words:
        raise ValueError(
            f"{source}: histogram length {record.histogram.shape[0]} does not match num_visual_words={record.num_visual_words}."
        )

    if expected_method is not None and record.method != expected_method:
        raise ValueError(
            f"{source}: mixed methods are not supported in one build: expected {expected_method}, got {record.method}."
        )
    if expected_num_visual_words is not None and record.num_visual_words != expected_num_visual_words:
        raise ValueError(
            f"{source}: histogram size mismatch: expected num_visual_words={expected_num_visual_words}, "
            f"got {record.num_visual_words}."
        )

    return record


def _validate_query_record(
    query: EncodedFeatureResult,
    index: InvertedIndexArtifact,
    *,
    source: str,
) -> EncodedFeatureResult:
    if not isinstance(query, EncodedFeatureResult):
        raise TypeError(f"{source}: query must be an EncodedFeatureResult, got {type(query).__name__}.")

    if query.sample_id is None or not isinstance(query.sample_id, str) or not query.sample_id.strip():
        raise ValueError(f"{source}: query sample_id must be a non-empty string.")
    if query.method != index.method:
        raise ValueError(f"{source}: method mismatch: query method {query.method} does not match index method {index.method}.")
    if query.normalized:
        raise ValueError(
            f"{source}: normalized BoW queries are not supported in Milestone 5. "
            "Please use raw-count BoW artifacts with normalized == False."
        )
    if not isinstance(query.histogram, np.ndarray):
        raise TypeError(f"{source}: query histogram must be a numpy.ndarray, got {type(query.histogram).__name__}.")
    if query.histogram.ndim != 1:
        raise ValueError(f"{source}: query histogram must be a 1D array, got shape={query.histogram.shape}.")
    if query.histogram_dtype != "int32":
        raise ValueError(
            f"{source}: query histogram_dtype must be 'int32' for raw-count BoW artifacts, got {query.histogram_dtype!r}."
        )
    if query.histogram.dtype != np.int32:
        raise ValueError(
            f"{source}: query histogram dtype must be int32 for raw-count BoW artifacts, got {query.histogram.dtype}."
        )
    if query.num_visual_words != index.num_visual_words:
        raise ValueError(
            f"{source}: histogram size mismatch: query has num_visual_words={query.num_visual_words}, "
            f"index expects {index.num_visual_words}."
        )
    return query


def _validate_tfidf_stats_for_query(
    tfidf_stats: TfidfStatsArtifact,
    index: InvertedIndexArtifact,
    *,
    source: str,
) -> None:
    if not isinstance(tfidf_stats, TfidfStatsArtifact):
        raise TypeError(f"{source}: tfidf_stats must be a TfidfStatsArtifact, got {type(tfidf_stats).__name__}.")
    if tfidf_stats.method != index.method:
        raise ValueError(
            f"{source}: TF-IDF stats method {tfidf_stats.method} does not match index method {index.method}."
        )
    if tfidf_stats.num_visual_words != index.num_visual_words:
        raise ValueError(
            f"{source}: TF-IDF stats num_visual_words={tfidf_stats.num_visual_words} does not match index num_visual_words={index.num_visual_words}."
        )


def _normalize_method_filter(methods: Iterable[str] | None) -> tuple[str, ...] | None:
    if methods is None:
        return None

    normalized_methods: list[str] = []
    for method in methods:
        normalized = _normalize_method(method)
        if normalized not in normalized_methods:
            normalized_methods.append(normalized)
    return tuple(sorted(normalized_methods))


def _normalize_weighting_modes(weighting_modes: Iterable[WeightingMode] | None) -> tuple[WeightingMode, ...]:
    if weighting_modes is None:
        return _DEFAULT_WEIGHTING_MODES

    normalized: list[WeightingMode] = []
    for mode in weighting_modes:
        normalized_mode = _normalize_weighting_mode(mode)
        if normalized_mode not in normalized:
            normalized.append(normalized_mode)
    if not normalized:
        return _DEFAULT_WEIGHTING_MODES
    return tuple(sorted(normalized))


def _normalize_weighting_mode(weighting_mode: str) -> WeightingMode:
    if not isinstance(weighting_mode, str) or not weighting_mode.strip():
        raise ValueError("weighting_mode must be a non-empty string.")

    normalized = weighting_mode.strip().lower()
    if normalized not in _DEFAULT_WEIGHTING_MODES:
        raise ValueError(f"Unsupported weighting_mode: {weighting_mode!r}. Supported modes: tf, tfidf.")
    return normalized  # type: ignore[return-value]


def _normalize_scoring_mode(scoring_mode: ScoringMode | str) -> ScoringMode:
    if not isinstance(scoring_mode, str) or not scoring_mode.strip():
        raise ValueError("scoring_mode must be a non-empty string.")

    normalized = scoring_mode.strip().lower()
    if normalized not in _SUPPORTED_SCORING_MODES:
        raise ValueError(
            f"Unsupported scoring_mode: {scoring_mode!r}. Supported modes: {', '.join(_SUPPORTED_SCORING_MODES)}."
        )
    return normalized  # type: ignore[return-value]


def _normalize_corpus_split(corpus_split: str) -> str:
    if not isinstance(corpus_split, str) or not corpus_split.strip():
        raise ValueError("corpus_split must be a non-empty string label.")
    normalized = corpus_split.strip()
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError(f"corpus_split must not contain path separators, got {corpus_split!r}.")
    return normalized
