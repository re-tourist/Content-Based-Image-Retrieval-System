from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing import (
    MethodIndexArtifacts,
    build_inverted_index_from_encoded_dir,
    load_inverted_index,
    resolve_inverted_index_path,
    resolve_method_index_dir,
    resolve_tfidf_stats_path,
    save_inverted_index,
    save_tfidf_stats,
)
from src.utils import get_default_config_path, load_config


DEFAULT_CORPUS_SPLIT = "gallery"
DEFAULT_WEIGHTING_MODES = ("tf", "tfidf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TF-IDF statistics and inverted indexes from encoded BoW artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=get_default_config_path(),
        help="Path to the runtime config file. Defaults to configs/base.yaml.",
    )
    parser.add_argument(
        "--encoded-dir",
        required=True,
        help="Directory containing encoded BoW artifacts to index.",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Root output directory for inverted index artifacts. Defaults to output.index_dir/inverted from config.",
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
        "--weighting",
        action="append",
        choices=DEFAULT_WEIGHTING_MODES,
        help="Optional weighting filter. By default both tf and tfidf indexes are built.",
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


def run_build_index(
    *,
    encoded_dir: Path,
    index_root: Path,
    corpus_split: str,
    methods: list[str] | None = None,
    weighting_modes: list[str] | None = None,
) -> list[Path]:
    build_results = build_inverted_index_from_encoded_dir(
        encoded_dir,
        corpus_split=corpus_split,
        methods=methods,
        weighting_modes=weighting_modes,
    )

    saved_paths: list[Path] = []
    for method, bundle in build_results.items():
        method_dir = resolve_method_index_dir(index_root, corpus_split, method)
        method_dir.mkdir(parents=True, exist_ok=True)

        tfidf_stats_path = save_tfidf_stats(bundle.tfidf_stats, method_dir)
        saved_paths.append(tfidf_stats_path)

        for weighting_mode, index_artifact in sorted(bundle.indexes.items()):
            index_path = save_inverted_index(index_artifact, method_dir)
            saved_paths.append(index_path)

        _print_method_summary(method, bundle, method_dir)

    return saved_paths


def _print_method_summary(method: str, bundle: MethodIndexArtifacts, method_dir: Path) -> None:
    tfidf_stats = bundle.tfidf_stats
    print(
        f"Built inverted index for method={method} corpus_split={tfidf_stats.corpus_split} "
        f"docs={tfidf_stats.num_docs} visual_words={tfidf_stats.num_visual_words}"
    )
    print(f"Saved TF-IDF stats to {resolve_tfidf_stats_path(method_dir)}")
    for weighting_mode in sorted(bundle.indexes):
        index_path = resolve_inverted_index_path(method_dir, weighting_mode)
        loaded_index = load_inverted_index(index_path)
        nnz = int(loaded_index.indices.size)
        non_empty_terms = int(np.count_nonzero(np.diff(loaded_index.indptr) > 0))
        print(
            f"Saved {weighting_mode} index to {index_path} "
            f"(nnz={nnz}, non_empty_terms={non_empty_terms}, doc_norms={loaded_index.doc_norms.size})"
        )


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


def main() -> int:
    args = parse_args()

    try:
        config, config_path = load_runtime_config(args.config)
        print(f"Loaded config from {config_path}")

        encoded_dir = _resolve_path(args.encoded_dir, "encoded_dir")
        index_root = _resolve_path(args.index_dir, "index_dir") if args.index_dir is not None else _resolve_default_index_root(config)
        corpus_split = args.corpus_split.strip()
        if not corpus_split:
            raise ValueError("corpus_split must be a non-empty string label.")

        print(f"Building indexes from encoded_dir={encoded_dir}")
        print(f"Writing index artifacts under {index_root}")
        print(f"Corpus split label: {corpus_split}")

        saved_paths = run_build_index(
            encoded_dir=encoded_dir,
            index_root=index_root,
            corpus_split=corpus_split,
            methods=args.method,
            weighting_modes=args.weighting,
        )
        print(f"Index build completed. Saved {len(saved_paths)} artifact(s).")
        return 0
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError, ImportError, RuntimeError, TypeError) as exc:
        print(f"Index build failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
