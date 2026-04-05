from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.encoding import EncodedFeatureResult, load_encoded_feature
from src.indexing import (
    InvertedIndexArtifact,
    TfidfStatsArtifact,
    load_inverted_index,
    load_tfidf_stats,
    locate_method_index_dir,
    resolve_inverted_index_path,
    resolve_tfidf_stats_path,
    search_inverted_index,
    iter_encoded_feature_paths,
)
from src.utils import get_default_config_path, load_config


DEFAULT_TOP_K = 5
SUPPORTED_WEIGHTING_SCORING_PAIRS = {
    ("tf", "dot"),
    ("tfidf", "cosine"),
}


@dataclass(slots=True)
class _MethodSearchBundle:
    method: str
    index_dir: Path
    tfidf_stats: TfidfStatsArtifact
    indexes: dict[str, InvertedIndexArtifact]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search encoded queries against method-specific inverted indexes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=get_default_config_path(),
        help="Path to the runtime config file. Defaults to configs/base.yaml.",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Directory containing inverted-index artifacts. Can be a corpus root or a method directory.",
    )
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--query-encoded",
        help="Path to one encoded query artifact.",
    )
    query_group.add_argument(
        "--query-dir",
        help="Directory containing encoded query artifacts for batch search.",
    )
    parser.add_argument(
        "--query-sample-id",
        help="Optional sample_id selector when using --query-dir.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieval results to print per query.",
    )
    parser.add_argument(
        "--weighting",
        choices=("tf", "tfidf"),
        default="tfidf",
        help="Weighting mode for single-query search. Ignored in --compare mode.",
    )
    parser.add_argument(
        "--scoring",
        choices=("dot", "cosine"),
        default="cosine",
        help="Scoring mode for single-query search. Ignored in --compare mode.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both supported paths: tf+dot and tfidf+cosine.",
    )
    return parser.parse_args()


def load_runtime_config(config_path: str) -> tuple[dict[str, Any], Path]:
    resolved_path = Path(config_path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = (PROJECT_ROOT / resolved_path).resolve()
    else:
        resolved_path = resolved_path.resolve()

    config = load_config(str(resolved_path))
    return config, resolved_path


def run_search(
    *,
    index_root: Path,
    queries: list[EncodedFeatureResult],
    top_k: int,
    compare: bool,
    weighting: str,
    scoring: str,
) -> None:
    bundle_cache: dict[str, _MethodSearchBundle] = {}
    for query in queries:
        method_bundle = _load_method_bundle(index_root, query.method, bundle_cache)
        if compare:
            _require_weighting_mode(method_bundle, "tf")
            _require_weighting_mode(method_bundle, "tfidf")
            tf_results = search_inverted_index(
                query,
                method_bundle.indexes["tf"],
                scoring_mode="dot",
                top_k=top_k,
            )
            tfidf_results = search_inverted_index(
                query,
                method_bundle.indexes["tfidf"],
                tfidf_stats=method_bundle.tfidf_stats,
                scoring_mode="cosine",
                top_k=top_k,
            )
            _print_compare_results(query, tf_results, tfidf_results, top_k)
            continue

        if (weighting, scoring) not in SUPPORTED_WEIGHTING_SCORING_PAIRS:
            raise ValueError(
                "search_images.py only supports the following weighting/scoring pairs in Milestone 5: "
                "tf + dot, tfidf + cosine."
            )

        selected_index = _require_weighting_mode(method_bundle, weighting)
        selected_stats = method_bundle.tfidf_stats if weighting == "tfidf" else None
        results = search_inverted_index(
            query,
            selected_index,
            tfidf_stats=selected_stats,
            scoring_mode=scoring,
            top_k=top_k,
        )
        _print_single_results(query, weighting, scoring, results)


def _load_method_bundle(
    index_root: Path,
    method: str,
    cache: dict[str, _MethodSearchBundle],
) -> _MethodSearchBundle:
    normalized_method = method.strip().upper()
    if normalized_method in cache:
        return cache[normalized_method]

    method_dir = locate_method_index_dir(index_root, normalized_method)
    tfidf_stats_path = resolve_tfidf_stats_path(method_dir)
    tfidf_stats = load_tfidf_stats(tfidf_stats_path)
    if tfidf_stats.method != normalized_method:
        raise ValueError(
            f"Index directory {method_dir} contains method {tfidf_stats.method}, "
            f"but query method {normalized_method} was requested."
        )
    indexes: dict[str, InvertedIndexArtifact] = {}
    for weighting_mode in ("tf", "tfidf"):
        index_path = resolve_inverted_index_path(method_dir, weighting_mode)
        if index_path.is_file():
            indexes[weighting_mode] = load_inverted_index(index_path)

    if not indexes:
        raise FileNotFoundError(f"No inverted index artifacts found under {method_dir}.")

    bundle = _MethodSearchBundle(
        method=normalized_method,
        index_dir=method_dir,
        tfidf_stats=tfidf_stats,
        indexes=indexes,
    )
    cache[normalized_method] = bundle
    return bundle


def _require_weighting_mode(bundle: _MethodSearchBundle, weighting_mode: str) -> InvertedIndexArtifact:
    index = bundle.indexes.get(weighting_mode)
    if index is None:
        raise FileNotFoundError(
            f"Missing index artifact for weighting_mode={weighting_mode} under {bundle.index_dir}."
        )
    return index


def _print_single_results(
    query: EncodedFeatureResult,
    weighting: str,
    scoring: str,
    results: list[Any],
) -> None:
    print(f"Query: {query.sample_id} method={query.method} weighting={weighting} scoring={scoring}")
    if not results:
        print("  no results")
        return

    for result in results:
        print(f"  {result.rank:>2}. {result.sample_id}  score={result.score:.6f}  doc_index={result.doc_index}")


def _print_compare_results(
    query: EncodedFeatureResult,
    tf_results: list[Any],
    tfidf_results: list[Any],
    top_k: int,
) -> None:
    print(f"Query: {query.sample_id} method={query.method} compare=tf+dot vs tfidf+cosine")
    print("  rank | tf+dot                         | tfidf+cosine")
    limit = max(min(top_k, len(tf_results)), min(top_k, len(tfidf_results)))
    for rank in range(limit):
        tf_cell = _format_result_cell(tf_results[rank]) if rank < len(tf_results) else "-"
        tfidf_cell = _format_result_cell(tfidf_results[rank]) if rank < len(tfidf_results) else "-"
        print(f"  {rank + 1:>4} | {tf_cell:<30} | {tfidf_cell}")


def _format_result_cell(result: Any) -> str:
    return f"{result.sample_id} ({result.score:.6f})"


def _resolve_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_default_index_root(config: dict[str, Any]) -> Path:
    output_config = _get_mapping(config, "output")
    raw_index_root = output_config.get("index_dir", "outputs/indices")
    resolved = _resolve_path(raw_index_root, "output.index_dir")
    return resolved / "inverted"


def _get_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return value


def _load_queries_from_single_path(query_path: Path, query_sample_id: str | None) -> list[EncodedFeatureResult]:
    query = load_encoded_feature(query_path)
    if query_sample_id is not None and query.sample_id != query_sample_id:
        raise ValueError(
            f"query sample_id mismatch: requested {query_sample_id!r} but artifact contains {query.sample_id!r}."
        )
    return [query]


def _load_queries_from_dir(query_dir: Path, query_sample_id: str | None) -> list[EncodedFeatureResult]:
    query_paths = list(iter_encoded_feature_paths(query_dir))
    if not query_paths:
        raise FileNotFoundError(f"No encoded query artifacts found under: {query_dir}")

    queries: list[EncodedFeatureResult] = []
    matched_query: EncodedFeatureResult | None = None
    for query_path in query_paths:
        query = load_encoded_feature(query_path)
        if query_sample_id is None:
            queries.append(query)
            continue

        if query.sample_id == query_sample_id:
            if matched_query is not None:
                raise ValueError(
                    f"Multiple query artifacts matched sample_id {query_sample_id!r} under {query_dir}."
                )
            matched_query = query

    if query_sample_id is not None:
        if matched_query is None:
            raise FileNotFoundError(f"No encoded query artifact matched sample_id {query_sample_id!r} under {query_dir}.")
        return [matched_query]

    return queries


def main() -> int:
    args = parse_args()

    try:
        config, config_path = load_runtime_config(args.config)
        print(f"Loaded config from {config_path}")

        index_root = _resolve_path(args.index_dir, "index_dir") if args.index_dir is not None else _resolve_default_index_root(config)
        top_k = int(args.top_k)
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than zero, got {top_k}.")

        if args.query_encoded is not None:
            queries = _load_queries_from_single_path(_resolve_path(args.query_encoded, "query_encoded"), args.query_sample_id)
        else:
            if args.query_sample_id is not None and not args.query_dir:
                raise ValueError("--query-sample-id can only be used with --query-dir.")
            queries = _load_queries_from_dir(_resolve_path(args.query_dir, "query_dir"), args.query_sample_id)

        print(f"Using index root: {index_root}")
        print(f"Loaded {len(queries)} query artifact(s)")
        run_search(
            index_root=index_root,
            queries=queries,
            top_k=top_k,
            compare=bool(args.compare),
            weighting=str(args.weighting),
            scoring=str(args.scoring),
        )
        return 0
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError, ImportError, RuntimeError, TypeError) as exc:
        print(f"Search failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
