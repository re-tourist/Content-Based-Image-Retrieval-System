from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import RetrievalEvalRunManifest, VariantEvaluation


_VARIANT_FILENAME = "variant.json"
_PER_QUERY_RESULTS_FILENAME = "per_query_results.json"
_PER_QUERY_METRICS_FILENAME = "per_query_metrics.json"
_SUMMARY_METRICS_FILENAME = "summary_metrics.json"
_PR_CURVE_FILENAME = "pr_curve.json"
_RUN_MANIFEST_FILENAME = "run_manifest.json"


def save_variant_evaluation(variant: VariantEvaluation, output_dir: str | Path) -> dict[str, Path]:
    """Persist one evaluation variant to a stable JSON layout."""
    target_dir = _resolve_output_dir(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    payloads = {
        _VARIANT_FILENAME: variant,
        _PER_QUERY_RESULTS_FILENAME: [query for query in variant.query_results],
        _PER_QUERY_METRICS_FILENAME: [query.metrics for query in variant.query_results],
        _SUMMARY_METRICS_FILENAME: variant.summary,
        _PR_CURVE_FILENAME: {
            "variant_name": variant.variant_name,
            "method": variant.method,
            "weighting_mode": variant.weighting_mode,
            "scoring_mode": variant.scoring_mode,
            "corpus_split": variant.corpus_split,
            "gallery_split": variant.gallery_split,
            "query_split": variant.query_split,
            "summary_pr_curve": variant.summary_pr_curve,
            "query_pr_curves": [
                {
                    "query_sample_id": query.query_sample_id,
                    "query_artifact_sample_id": query.query_artifact_sample_id,
                    "query_label": query.query_label,
                    "points": query.pr_curve_points,
                }
                for query in variant.query_results
            ],
        },
    }

    saved_paths: dict[str, Path] = {}
    for filename, payload in payloads.items():
        path = target_dir / filename
        _write_json(path, payload)
        saved_paths[filename] = path
    return saved_paths


def save_run_manifest(manifest: RetrievalEvalRunManifest, output_dir: str | Path) -> Path:
    """Persist a run-level manifest for the canonical retrieval evaluation."""
    target_dir = _resolve_output_dir(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / _RUN_MANIFEST_FILENAME
    _write_json(path, manifest)
    return path


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        result: dict[str, Any] = {}
        for field in fields(value):
            result[field.name] = _to_jsonable(getattr(value, field.name))
        return result

    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_to_jsonable(item) for item in sorted(value, key=str)]
    return value


def _resolve_output_dir(output_dir: str | Path) -> Path:
    resolved = Path(output_dir).expanduser()
    if not resolved.is_absolute():
        resolved = resolved.resolve()
    return resolved

