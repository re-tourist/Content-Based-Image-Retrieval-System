from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

from src.datasets import ImageDatasetLoader, ImageSample
from src.encoding import EncodedFeatureResult
from src.encoding.io import _normalize_method
from src.indexing import (
    InvertedIndexArtifact,
    MethodIndexArtifacts,
    WeightingMode,
    ScoringMode,
    build_inverted_index_from_records,
    load_encoded_feature_records,
    load_inverted_index,
    load_tfidf_stats,
    resolve_inverted_index_path,
    resolve_method_index_dir,
    resolve_tfidf_stats_path,
    save_inverted_index,
    save_tfidf_stats,
    score_query_against_index,
)

from .metrics import (
    build_macro_pr_curve,
    build_prefix_pr_curve,
    compute_average_precision,
    compute_precision_at_k,
    compute_recall_at_k,
    extract_class_label,
    normalize_canonical_sample_id,
)
from .types import (
    MethodEvaluation,
    QueryEvaluation,
    QueryMetrics,
    PRCurvePoint,
    RetrievalEvalRunManifest,
    RetrievalSummary,
    RetrievedResult,
    VariantEvaluation,
)


@dataclass(slots=True)
class _MatchedRecord:
    split_sample: ImageSample
    encoded_record: EncodedFeatureResult
    canonical_sample_id: str
    label: str


@dataclass(slots=True)
class _VariantSpec:
    weighting_mode: WeightingMode
    scoring_mode: ScoringMode
    variant_name: str


_SUPPORTED_SINGLE_VARIANT_PAIRS: tuple[tuple[WeightingMode, ScoringMode], ...] = (
    ("tf", "dot"),
    ("tfidf", "cosine"),
)


def load_canonical_split_samples(
    gallery_split_file: str | Path,
    query_split_file: str | Path,
    data_root: str | Path,
) -> tuple[list[ImageSample], list[ImageSample]]:
    """Load gallery/query samples from canonical split files."""
    gallery_loader = ImageDatasetLoader(
        split_file=gallery_split_file,
        data_root=data_root,
        split="gallery",
        verbose=False,
    )
    query_loader = ImageDatasetLoader(
        split_file=query_split_file,
        data_root=data_root,
        split="query",
        verbose=False,
    )
    gallery_samples = list(gallery_loader)
    query_samples = list(query_loader)
    if not gallery_samples:
        raise ValueError(f"Gallery split is empty: {Path(gallery_split_file).expanduser()}")
    if not query_samples:
        raise ValueError(f"Query split is empty: {Path(query_split_file).expanduser()}")
    return gallery_samples, query_samples


def evaluate_canonical_retrieval(
    *,
    encoded_dir: str | Path,
    gallery_split_file: str | Path,
    query_split_file: str | Path,
    data_root: str | Path,
    index_root: str | Path,
    output_root: str | Path,
    corpus_split: str = "gallery",
    methods: Iterable[str] | None = None,
    compare: bool = False,
    requested_top_k: int = 10,
    weighting_mode: WeightingMode = "tfidf",
    scoring_mode: ScoringMode = "cosine",
    rebuild_index: bool = False,
) -> tuple[list[MethodEvaluation], RetrievalEvalRunManifest]:
    """Run canonical split-driven retrieval evaluation for one encoded corpus."""
    gallery_samples, query_samples = load_canonical_split_samples(gallery_split_file, query_split_file, data_root)

    all_encoded_records = load_encoded_feature_records(encoded_dir, methods=methods)
    records_by_method = _group_records_by_method(all_encoded_records)

    evaluated_methods = _resolve_requested_methods(methods, records_by_method)
    if not evaluated_methods:
        raise ValueError(f"No encoded artifacts were found under {Path(encoded_dir).expanduser()}.")

    variant_specs = _resolve_variant_specs(compare, weighting_mode, scoring_mode)

    method_evaluations: list[MethodEvaluation] = []
    for method in evaluated_methods:
        method_records = records_by_method.get(method)
        if not method_records:
            raise FileNotFoundError(f"No encoded artifacts found for method {method} under {Path(encoded_dir).expanduser()}.")

        gallery_matches = _match_split_samples_to_records(
            gallery_samples,
            method_records,
            split_name="gallery",
        )
        query_matches = _match_split_samples_to_records(
            query_samples,
            method_records,
            split_name="query",
        )

        required_weightings = tuple(sorted({spec.weighting_mode for spec in variant_specs}))
        bundle = _load_or_build_method_bundle(
            method=method,
            gallery_records=[match.encoded_record for match in gallery_matches],
            index_root=index_root,
            corpus_split=corpus_split,
            required_weightings=required_weightings,
            rebuild_index=rebuild_index,
        )

        variants = [
            _evaluate_variant(
                method=method,
                bundle=bundle,
                gallery_matches=gallery_matches,
                query_matches=query_matches,
                corpus_split=corpus_split,
                requested_top_k=requested_top_k,
                variant_spec=variant_spec,
            )
            for variant_spec in variant_specs
        ]
        method_evaluations.append(MethodEvaluation(method=method, variants=variants))

    manifest = RetrievalEvalRunManifest(
        gallery_split_file=Path(gallery_split_file).expanduser().resolve().as_posix(),
        query_split_file=Path(query_split_file).expanduser().resolve().as_posix(),
        data_root=Path(data_root).expanduser().resolve().as_posix(),
        encoded_dir=Path(encoded_dir).expanduser().resolve().as_posix(),
        index_root=Path(index_root).expanduser().resolve().as_posix(),
        output_root=Path(output_root).expanduser().resolve().as_posix(),
        corpus_split=corpus_split,
        methods=tuple(evaluated_methods),
        compare=bool(compare),
        requested_top_k=int(requested_top_k),
        weighting_mode=weighting_mode,
        scoring_mode=scoring_mode,
        rebuild_index=bool(rebuild_index),
    )
    return method_evaluations, manifest


def _resolve_requested_methods(
    methods: Iterable[str] | None,
    records_by_method: dict[str, list[EncodedFeatureResult]],
) -> list[str]:
    if methods is None:
        return sorted(records_by_method)

    normalized_methods: list[str] = []
    for method in methods:
        normalized = _normalize_method(method)
        if normalized not in normalized_methods:
            normalized_methods.append(normalized)

    missing = [method for method in normalized_methods if method not in records_by_method]
    if missing:
        available = ", ".join(sorted(records_by_method)) or "<none>"
        raise FileNotFoundError(
            f"Requested method(s) not found in encoded_dir: {', '.join(missing)}. Available methods: {available}."
        )
    return normalized_methods


def _resolve_variant_specs(
    compare: bool,
    weighting_mode: WeightingMode,
    scoring_mode: ScoringMode,
) -> list[_VariantSpec]:
    if compare:
        return [
            _VariantSpec(weighting_mode="tf", scoring_mode="dot", variant_name="tf_dot"),
            _VariantSpec(weighting_mode="tfidf", scoring_mode="cosine", variant_name="tfidf_cosine"),
        ]

    requested_pair = (weighting_mode, scoring_mode)
    if requested_pair not in _SUPPORTED_SINGLE_VARIANT_PAIRS:
        raise ValueError(
            "Only the following weighting/scoring pairs are supported in Milestone 5: "
            "tf + dot, tfidf + cosine."
        )

    return [
        _VariantSpec(
            weighting_mode=weighting_mode,
            scoring_mode=scoring_mode,
            variant_name=f"{weighting_mode}_{scoring_mode}",
        )
    ]


def _group_records_by_method(records: Sequence[EncodedFeatureResult]) -> dict[str, list[EncodedFeatureResult]]:
    grouped: dict[str, list[EncodedFeatureResult]] = defaultdict(list)
    for record in records:
        grouped[record.method].append(record)
    return dict(grouped)


def _match_split_samples_to_records(
    split_samples: Sequence[ImageSample],
    method_records: Sequence[EncodedFeatureResult],
    *,
    split_name: str,
) -> list[_MatchedRecord]:
    record_lookup: dict[str, EncodedFeatureResult] = {}
    for record in method_records:
        if record.sample_id is None:
            raise ValueError(f"{split_name}: encoded artifact is missing sample_id.")
        canonical_key = normalize_canonical_sample_id(record.sample_id)
        if canonical_key in record_lookup:
            existing = record_lookup[canonical_key]
            raise ValueError(
                f"Duplicate encoded artifact for canonical sample id {canonical_key!r}: "
                f"{existing.sample_id!r} and {record.sample_id!r}."
            )
        record_lookup[canonical_key] = record

    matched_records: list[_MatchedRecord] = []
    missing_sample_ids: list[str] = []
    seen_split_keys: set[str] = set()
    for sample in split_samples:
        canonical_sample_id = normalize_canonical_sample_id(sample.sample_id)
        if canonical_sample_id in seen_split_keys:
            raise ValueError(f"Duplicate sample id in {split_name} split after normalization: {sample.sample_id!r}.")
        seen_split_keys.add(canonical_sample_id)

        record = record_lookup.get(canonical_sample_id)
        if record is None:
            missing_sample_ids.append(sample.sample_id)
            continue

        matched_records.append(
            _MatchedRecord(
                split_sample=sample,
                encoded_record=record,
                canonical_sample_id=canonical_sample_id,
                label=extract_class_label(sample.sample_id),
            )
        )

    if missing_sample_ids:
        raise FileNotFoundError(
            f"{split_name.capitalize()} split has no matching encoded artifact(s) under the provided encoded_dir: "
            f"{', '.join(missing_sample_ids)}"
        )
    if not matched_records:
        raise ValueError(f"{split_name.capitalize()} split produced no matched encoded artifacts.")
    return matched_records


def _load_or_build_method_bundle(
    *,
    method: str,
    gallery_records: Sequence[EncodedFeatureResult],
    index_root: str | Path,
    corpus_split: str,
    required_weightings: Sequence[WeightingMode],
    rebuild_index: bool,
) -> MethodIndexArtifacts:
    method_dir = resolve_method_index_dir(index_root, corpus_split, method)
    tfidf_stats_path = resolve_tfidf_stats_path(method_dir)
    required_index_paths = {
        weighting_mode: resolve_inverted_index_path(method_dir, weighting_mode)
        for weighting_mode in required_weightings
    }

    should_build = bool(rebuild_index) or not tfidf_stats_path.is_file()
    if not should_build:
        should_build = any(not path.is_file() for path in required_index_paths.values())

    if should_build:
        bundle = build_inverted_index_from_records(
            gallery_records,
            corpus_split=corpus_split,
            weighting_modes=required_weightings,
        )
        method_dir.mkdir(parents=True, exist_ok=True)
        save_tfidf_stats(bundle.tfidf_stats, method_dir)
        for index in bundle.indexes.values():
            save_inverted_index(index, method_dir)
        return bundle

    tfidf_stats = load_tfidf_stats(tfidf_stats_path)
    indexes = {
        weighting_mode: load_inverted_index(path)
        for weighting_mode, path in required_index_paths.items()
    }
    return MethodIndexArtifacts(
        method=tfidf_stats.method,
        corpus_split=tfidf_stats.corpus_split,
        tfidf_stats=tfidf_stats,
        indexes=indexes,
    )


def _evaluate_variant(
    *,
    method: str,
    bundle: MethodIndexArtifacts,
    gallery_matches: Sequence[_MatchedRecord],
    query_matches: Sequence[_MatchedRecord],
    corpus_split: str,
    requested_top_k: int,
    variant_spec: _VariantSpec,
) -> VariantEvaluation:
    index = _require_index(bundle, variant_spec.weighting_mode)
    tfidf_stats = bundle.tfidf_stats if variant_spec.weighting_mode == "tfidf" else None

    gallery_count = len(gallery_matches)
    query_count = len(query_matches)
    if gallery_count == 0:
        raise ValueError("Gallery split produced no encoded artifacts.")
    if query_count == 0:
        raise ValueError("Query split produced no encoded artifacts.")

    gallery_labels = [match.label for match in gallery_matches]
    query_results: list[QueryEvaluation] = []
    query_pr_curves: list[list[PRCurvePoint]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    ap_values: list[float] = []
    timings_ms: list[float] = []
    query_count_with_relevant_items = 0

    for query_match in query_matches:
        start = perf_counter()
        scores = score_query_against_index(
            query_match.encoded_record,
            index,
            tfidf_stats=tfidf_stats,
            scoring_mode=variant_spec.scoring_mode,
        )
        ranking = np.lexsort((np.arange(scores.size), -scores))
        elapsed_ms = (perf_counter() - start) * 1000.0

        relevance_flags_full = [label == query_match.label for label in gallery_labels]
        relevant_total = int(np.count_nonzero(relevance_flags_full))
        if relevant_total > 0:
            query_count_with_relevant_items += 1

        effective_top_k = min(int(requested_top_k), gallery_count)
        top_indices = ranking[:effective_top_k]
        top_relevance = [bool(relevance_flags_full[int(index)]) for index in top_indices.tolist()]
        retrieved_relevant_at_k = int(np.count_nonzero(top_relevance))

        ranked_results = [
            RetrievedResult(
                rank=rank,
                sample_id=gallery_matches[int(doc_index)].split_sample.sample_id,
                artifact_sample_id=gallery_matches[int(doc_index)].encoded_record.sample_id or "",
                label=gallery_matches[int(doc_index)].label,
                score=float(scores[int(doc_index)]),
                relevant=bool(relevance_flags_full[int(doc_index)]),
                doc_index=int(doc_index),
            )
            for rank, doc_index in enumerate(top_indices.tolist(), start=1)
        ]

        full_ranked_results = [
            RetrievedResult(
                rank=rank,
                sample_id=gallery_matches[int(doc_index)].split_sample.sample_id,
                artifact_sample_id=gallery_matches[int(doc_index)].encoded_record.sample_id or "",
                label=gallery_matches[int(doc_index)].label,
                score=float(scores[int(doc_index)]),
                relevant=bool(relevance_flags_full[int(doc_index)]),
                doc_index=int(doc_index),
            )
            for rank, doc_index in enumerate(ranking.tolist(), start=1)
        ]

        precision_at_k = compute_precision_at_k(top_relevance, effective_top_k)
        recall_at_k = compute_recall_at_k(top_relevance, effective_top_k, relevant_total)
        average_precision = compute_average_precision(relevance_flags_full, relevant_total)
        pr_curve_points = build_prefix_pr_curve(full_ranked_results, relevant_total)
        if relevant_total > 0:
            query_pr_curves.append(pr_curve_points)

        precision_values.append(precision_at_k)
        if recall_at_k is not None:
            recall_values.append(recall_at_k)
        if average_precision is not None:
            ap_values.append(average_precision)
        timings_ms.append(elapsed_ms)

        query_metrics = QueryMetrics(
            query_sample_id=query_match.split_sample.sample_id,
            query_artifact_sample_id=query_match.encoded_record.sample_id or "",
            query_label=query_match.label,
            method=method,
            weighting_mode=variant_spec.weighting_mode,
            scoring_mode=variant_spec.scoring_mode,
            requested_top_k=int(requested_top_k),
            effective_top_k=effective_top_k,
            gallery_count=gallery_count,
            relevant_total=relevant_total,
            retrieved_relevant_at_k=retrieved_relevant_at_k,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            average_precision=average_precision,
            included_in_mean_recall=relevant_total > 0,
            included_in_mean_average_precision=relevant_total > 0,
            retrieval_time_ms=elapsed_ms,
        )
        query_results.append(
            QueryEvaluation(
                query_sample_id=query_match.split_sample.sample_id,
                query_artifact_sample_id=query_match.encoded_record.sample_id or "",
                query_label=query_match.label,
                method=method,
                weighting_mode=variant_spec.weighting_mode,
                scoring_mode=variant_spec.scoring_mode,
                requested_top_k=int(requested_top_k),
                effective_top_k=effective_top_k,
                ranked_results=ranked_results,
                metrics=query_metrics,
                pr_curve_points=pr_curve_points,
            )
        )

    summary = RetrievalSummary(
        method=method,
        weighting_mode=variant_spec.weighting_mode,
        scoring_mode=variant_spec.scoring_mode,
        corpus_split=corpus_split,
        gallery_split="gallery",
        query_split="query",
        requested_top_k=int(requested_top_k),
        effective_top_k=min(int(requested_top_k), gallery_count),
        gallery_count=gallery_count,
        query_count=query_count,
        query_count_with_relevant_items=query_count_with_relevant_items,
        query_count_without_relevant_items=query_count - query_count_with_relevant_items,
        mean_precision_at_k=float(np.mean(precision_values)) if precision_values else 0.0,
        mean_recall_at_k=float(np.mean(recall_values)) if recall_values else None,
        mean_average_precision=float(np.mean(ap_values)) if ap_values else None,
        total_retrieval_time_ms=float(np.sum(timings_ms)),
        avg_retrieval_time_ms=float(np.mean(timings_ms)) if timings_ms else 0.0,
        relevance_definition="same class-folder name after canonical sample-id normalization",
        canonical_sample_id_rule=(
            "strip known dataset prefixes such as data/, train/, train/image/, test/, gallery/ and query/ "
            "before extracting the class-folder label"
        ),
    )

    return VariantEvaluation(
        variant_name=variant_spec.variant_name,
        method=method,
        weighting_mode=variant_spec.weighting_mode,
        scoring_mode=variant_spec.scoring_mode,
        corpus_split=corpus_split,
        gallery_split="gallery",
        query_split="query",
        requested_top_k=int(requested_top_k),
        effective_top_k=summary.effective_top_k,
        gallery_count=gallery_count,
        query_count=query_count,
        query_results=query_results,
        summary=summary,
        summary_pr_curve=build_macro_pr_curve(query_pr_curves),
    )


def _require_index(bundle: MethodIndexArtifacts, weighting_mode: WeightingMode) -> InvertedIndexArtifact:
    index = bundle.indexes.get(weighting_mode)
    if index is None:
        raise FileNotFoundError(
            f"Missing inverted index for weighting_mode={weighting_mode} under method={bundle.method}."
        )
    return index
