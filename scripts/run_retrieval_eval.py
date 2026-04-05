from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_canonical_retrieval, save_run_manifest, save_variant_evaluation
from src.evaluation.types import MethodEvaluation, VariantEvaluation
from src.utils import get_default_config_path, load_config


DEFAULT_CORPUS_SPLIT = "gallery"
DEFAULT_WEIGHTING = "tfidf"
DEFAULT_SCORING = "cosine"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run canonical retrieval evaluation on gallery/query split files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=get_default_config_path(),
        help="Path to the runtime config file. Defaults to configs/base.yaml.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional override for the dataset root. Defaults to dataset.root from config.",
    )
    parser.add_argument(
        "--gallery-split-file",
        default=None,
        help="Optional override for the canonical gallery split file. Defaults to data/splits/gallery.txt.",
    )
    parser.add_argument(
        "--query-split-file",
        default=None,
        help="Optional override for the canonical query split file. Defaults to data/splits/query.txt.",
    )
    parser.add_argument(
        "--encoded-dir",
        default=None,
        help="Directory containing encoded BoW artifacts. Defaults to encoding.input.encoded_dir from config.",
    )
    parser.add_argument(
        "--index-root",
        default=None,
        help="Root output directory for inverted indexes. Defaults to output.index_dir/inverted from config.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root output directory for retrieval evaluation artifacts. Defaults to outputs/evaluations/retrieval.",
    )
    parser.add_argument(
        "--corpus-split",
        default=DEFAULT_CORPUS_SPLIT,
        help="Naming/logging label for the indexed corpus. This is metadata only.",
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Optional method filter. Can be provided multiple times, e.g. --method sift --method orb.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K retrieval size for exported ranked results and P@K / R@K.",
    )
    parser.add_argument(
        "--weighting",
        choices=("tf", "tfidf"),
        default=DEFAULT_WEIGHTING,
        help="Weighting mode for a single-variant run. Ignored when --compare is used.",
    )
    parser.add_argument(
        "--scoring",
        choices=("dot", "cosine"),
        default=DEFAULT_SCORING,
        help="Scoring mode for a single-variant run. Ignored when --compare is used.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run the two canonical Milestone 5 variants: tf+dot and tfidf+cosine.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuilding of sparse indexes even if saved artifacts already exist.",
    )
    return parser.parse_args(argv)


def load_runtime_config(config_path: str) -> tuple[dict[str, Any], Path]:
    resolved_path = Path(config_path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = (PROJECT_ROOT / resolved_path).resolve()
    else:
        resolved_path = resolved_path.resolve()

    config = load_config(str(resolved_path))
    return config, resolved_path


def run_retrieval_eval(
    *,
    encoded_dir: Path,
    gallery_split_file: Path,
    query_split_file: Path,
    data_root: Path,
    index_root: Path,
    output_root: Path,
    corpus_split: str,
    methods: list[str] | None,
    compare: bool,
    top_k: int,
    weighting: str,
    scoring: str,
    rebuild_index: bool,
) -> tuple[list[MethodEvaluation], Path]:
    method_evaluations, manifest = evaluate_canonical_retrieval(
        encoded_dir=encoded_dir,
        gallery_split_file=gallery_split_file,
        query_split_file=query_split_file,
        data_root=data_root,
        index_root=index_root,
        output_root=output_root,
        corpus_split=corpus_split,
        methods=methods,
        compare=compare,
        requested_top_k=top_k,
        weighting_mode=weighting,
        scoring_mode=scoring,
        rebuild_index=rebuild_index,
    )

    run_root = output_root / corpus_split
    manifest_path = save_run_manifest(manifest, run_root)
    print(f"Saved run manifest to {manifest_path}")

    for method_evaluation in method_evaluations:
        for variant in method_evaluation.variants:
            variant_dir = run_root / method_evaluation.method / variant.variant_name
            saved_paths = save_variant_evaluation(variant, variant_dir)
            _print_variant_summary(method_evaluation.method, variant, variant_dir, saved_paths)

    return method_evaluations, manifest_path


def _print_variant_summary(
    method: str,
    variant: VariantEvaluation,
    variant_dir: Path,
    saved_paths: dict[str, Path],
) -> None:
    summary = variant.summary
    print(
        f"Evaluated method={method} variant={variant.variant_name} "
        f"weighting={variant.weighting_mode} scoring={variant.scoring_mode}"
    )
    print(
        f"  queries={summary.query_count} relevant_queries={summary.query_count_with_relevant_items} "
        f"gallery={summary.gallery_count} top_k={summary.effective_top_k}"
    )
    print(
        f"  P@K={summary.mean_precision_at_k:.4f} "
        f"R@K={(summary.mean_recall_at_k if summary.mean_recall_at_k is not None else float('nan')):.4f} "
        f"mAP={(summary.mean_average_precision if summary.mean_average_precision is not None else float('nan')):.4f}"
    )
    print(f"  output_dir={variant_dir}")
    print(f"  exported={', '.join(sorted(saved_paths))}")


def _resolve_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_default_data_root(config: dict[str, Any]) -> Path:
    dataset_config = _get_mapping(config, "dataset")
    raw_root = dataset_config.get("root", "data")
    return _resolve_path(raw_root, "dataset.root")


def _resolve_default_gallery_split_file(config: dict[str, Any], data_root: Path) -> Path:
    dataset_config = _get_mapping(config, "dataset")
    splits_dir = dataset_config.get("splits_dir", "splits")
    gallery_split = dataset_config.get("gallery_split", "gallery.txt")
    return (data_root / splits_dir / gallery_split).resolve()


def _resolve_default_query_split_file(config: dict[str, Any], data_root: Path) -> Path:
    dataset_config = _get_mapping(config, "dataset")
    splits_dir = dataset_config.get("splits_dir", "splits")
    query_split = dataset_config.get("query_split", "query.txt")
    return (data_root / splits_dir / query_split).resolve()


def _resolve_default_encoded_dir(config: dict[str, Any]) -> Path:
    encoding_config = _get_mapping(config, "encoding")
    input_config = _get_mapping(encoding_config, "input")
    raw_encoded_dir = input_config.get("encoded_dir", "outputs/encoded")
    return _resolve_path(raw_encoded_dir, "encoding.input.encoded_dir")


def _resolve_default_index_root(config: dict[str, Any]) -> Path:
    output_config = _get_mapping(config, "output")
    raw_index_root = output_config.get("index_dir", "outputs/indices")
    return _resolve_path(raw_index_root, "output.index_dir") / "inverted"


def _resolve_default_output_root() -> Path:
    return (PROJECT_ROOT / "outputs" / "evaluations" / "retrieval").resolve()


def _get_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        config, config_path = load_runtime_config(args.config)
        print(f"Loaded config from {config_path}")

        data_root = _resolve_path(args.data_root, "data_root") if args.data_root is not None else _resolve_default_data_root(config)
        gallery_split_file = (
            _resolve_path(args.gallery_split_file, "gallery_split_file")
            if args.gallery_split_file is not None
            else _resolve_default_gallery_split_file(config, data_root)
        )
        query_split_file = (
            _resolve_path(args.query_split_file, "query_split_file")
            if args.query_split_file is not None
            else _resolve_default_query_split_file(config, data_root)
        )
        encoded_dir = _resolve_path(args.encoded_dir, "encoded_dir") if args.encoded_dir is not None else _resolve_default_encoded_dir(config)
        index_root = _resolve_path(args.index_root, "index_root") if args.index_root is not None else _resolve_default_index_root(config)
        output_root = _resolve_path(args.output_root, "output_root") if args.output_root is not None else _resolve_default_output_root()

        if args.top_k <= 0:
            raise ValueError(f"top_k must be greater than zero, got {args.top_k}.")

        print(f"Using data root: {data_root}")
        print(f"Gallery split file: {gallery_split_file}")
        print(f"Query split file: {query_split_file}")
        print(f"Encoded dir: {encoded_dir}")
        print(f"Index root: {index_root}")
        print(f"Output root: {output_root}")

        method_evaluations, manifest_path = run_retrieval_eval(
            encoded_dir=encoded_dir,
            gallery_split_file=gallery_split_file,
            query_split_file=query_split_file,
            data_root=data_root,
            index_root=index_root,
            output_root=output_root,
            corpus_split=args.corpus_split.strip(),
            methods=args.method,
            compare=bool(args.compare),
            top_k=int(args.top_k),
            weighting=str(args.weighting),
            scoring=str(args.scoring),
            rebuild_index=bool(args.rebuild_index),
        )

        total_variants = sum(len(method_eval.variants) for method_eval in method_evaluations)
        print(f"Completed retrieval evaluation for {len(method_evaluations)} method(s) and {total_variants} variant(s).")
        print(f"Run manifest: {manifest_path}")
        return 0
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError, ImportError, RuntimeError, TypeError) as exc:
        print(f"Retrieval evaluation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

