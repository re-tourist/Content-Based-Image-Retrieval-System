from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np


WeightingMode: TypeAlias = Literal["tf", "tfidf"]
ScoringMode: TypeAlias = Literal["dot", "cosine"]


@dataclass(slots=True)
class PostingEntry:
    """One posting entry for a visual word."""

    sample_id: str
    doc_index: int
    weight: float
    term_frequency: int | None = None


@dataclass(slots=True)
class TfidfStatsArtifact:
    """Persisted DF / IDF statistics for one method-specific corpus."""

    method: str
    corpus_split: str
    num_docs: int
    num_visual_words: int
    df: np.ndarray
    idf: np.ndarray


@dataclass(slots=True)
class InvertedIndexArtifact:
    """CSR-style inverted index for one method and one weighting mode."""

    method: str
    corpus_split: str
    weighting_mode: WeightingMode
    num_docs: int
    num_visual_words: int
    sample_ids: tuple[str, ...]
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    doc_norms: np.ndarray


@dataclass(slots=True)
class MethodIndexArtifacts:
    """In-memory bundle of all saved index artifacts for one method."""

    method: str
    corpus_split: str
    tfidf_stats: TfidfStatsArtifact
    indexes: dict[WeightingMode, InvertedIndexArtifact]


@dataclass(slots=True)
class RankedResult:
    """One ranked retrieval result."""

    sample_id: str
    score: float
    rank: int
    doc_index: int | None = None


SearchResult = RankedResult

