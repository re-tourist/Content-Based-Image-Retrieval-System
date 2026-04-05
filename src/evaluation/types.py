from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from src.indexing.types import ScoringMode, WeightingMode


@dataclass(slots=True)
class RetrievedResult:
    """One ranked gallery hit for a query."""

    rank: int
    sample_id: str
    artifact_sample_id: str
    label: str
    score: float
    relevant: bool
    doc_index: int | None = None


@dataclass(slots=True)
class QueryMetrics:
    """Per-query metric summary."""

    query_sample_id: str
    query_artifact_sample_id: str
    query_label: str
    method: str
    weighting_mode: WeightingMode
    scoring_mode: ScoringMode
    requested_top_k: int
    effective_top_k: int
    gallery_count: int
    relevant_total: int
    retrieved_relevant_at_k: int
    precision_at_k: float
    recall_at_k: float | None
    average_precision: float | None
    included_in_mean_recall: bool
    included_in_mean_average_precision: bool
    retrieval_time_ms: float


@dataclass(slots=True)
class PRCurvePoint:
    """A single prefix-rank precision/recall point for one query."""

    rank: int
    precision: float
    recall: float
    relevant: bool
    sample_id: str
    artifact_sample_id: str


@dataclass(slots=True)
class SummaryPRCurvePoint:
    """A single point on the macro-averaged PR curve."""

    recall: float
    precision: float
    support: int


@dataclass(slots=True)
class QueryEvaluation:
    """Full evaluation payload for one query."""

    query_sample_id: str
    query_artifact_sample_id: str
    query_label: str
    method: str
    weighting_mode: WeightingMode
    scoring_mode: ScoringMode
    requested_top_k: int
    effective_top_k: int
    ranked_results: list[RetrievedResult]
    metrics: QueryMetrics
    pr_curve_points: list[PRCurvePoint]


@dataclass(slots=True)
class RetrievalSummary:
    """Run-level summary metrics for one evaluation variant."""

    method: str
    weighting_mode: WeightingMode
    scoring_mode: ScoringMode
    corpus_split: str
    gallery_split: str
    query_split: str
    requested_top_k: int
    effective_top_k: int
    gallery_count: int
    query_count: int
    query_count_with_relevant_items: int
    query_count_without_relevant_items: int
    mean_precision_at_k: float
    mean_recall_at_k: float | None
    mean_average_precision: float | None
    total_retrieval_time_ms: float
    avg_retrieval_time_ms: float
    relevance_definition: str
    canonical_sample_id_rule: str


@dataclass(slots=True)
class VariantEvaluation:
    """Evaluation result for one method / weighting / scoring variant."""

    variant_name: str
    method: str
    weighting_mode: WeightingMode
    scoring_mode: ScoringMode
    corpus_split: str
    gallery_split: str
    query_split: str
    requested_top_k: int
    effective_top_k: int
    gallery_count: int
    query_count: int
    query_results: list[QueryEvaluation]
    summary: RetrievalSummary
    summary_pr_curve: list[SummaryPRCurvePoint]


@dataclass(slots=True)
class MethodEvaluation:
    """All evaluation variants for one method."""

    method: str
    variants: list[VariantEvaluation]


@dataclass(slots=True)
class RetrievalEvalRunManifest:
    """Stable metadata describing one canonical retrieval evaluation run."""

    gallery_split_file: str
    query_split_file: str
    data_root: str
    encoded_dir: str
    index_root: str
    output_root: str
    corpus_split: str
    methods: tuple[str, ...]
    compare: bool
    requested_top_k: int
    weighting_mode: WeightingMode
    scoring_mode: ScoringMode
    rebuild_index: bool


RunArtifactList: TypeAlias = list[VariantEvaluation]
