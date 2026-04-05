from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.encoding import sample_descriptors_by_method, save_codebook_artifact, train_codebook
from src.utils import get_default_config_path, load_config



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train method-specific codebooks from saved feature artifacts.")
    parser.add_argument(
        "--config",
        default=get_default_config_path(),
        help="Path to the runtime config file. Defaults to configs/base.yaml.",
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Optional method filter. Can be provided multiple times, e.g. --method sift --method orb.",
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



def run_codebook_training(config: dict[str, Any], methods: list[str] | None = None) -> list[Path]:
    encoding_config = _get_mapping(config, "encoding")
    input_config = _get_nested_mapping(encoding_config, "input", "encoding.input")
    codebook_config = _get_nested_mapping(encoding_config, "codebook", "encoding.codebook")
    output_config = _get_mapping(config, "output")

    if not bool(codebook_config.get("enabled", False)):
        raise ValueError("Config field 'encoding.codebook.enabled' must be true to train codebooks.")

    feature_dir = _resolve_path(
        input_config.get("feature_dir", output_config.get("feature_dir", "outputs/features")),
        "encoding.input.feature_dir",
    )
    output_dir = _resolve_path(
        codebook_config.get("output_dir", "outputs/indices/codebooks"),
        "encoding.codebook.output_dir",
    )
    trainer = _require_non_empty_string(codebook_config.get("trainer", "minibatch_kmeans"), "encoding.codebook.trainer")
    n_clusters = _require_positive_int(codebook_config.get("n_clusters", 256), "encoding.codebook.n_clusters")
    max_descriptors = _require_positive_int(
        codebook_config.get("max_descriptors", 50000),
        "encoding.codebook.max_descriptors",
    )
    batch_size = _require_positive_int(codebook_config.get("batch_size", 1024), "encoding.codebook.batch_size")
    random_state = _optional_int(codebook_config.get("random_state", 42), "encoding.codebook.random_state")

    descriptor_samples = sample_descriptors_by_method(
        feature_dir=feature_dir,
        max_descriptors=max_descriptors,
        random_state=random_state,
        methods=methods,
    )
    if not descriptor_samples:
        raise ValueError(f"No descriptor samples were collected from {feature_dir}.")

    saved_paths: list[Path] = []
    for method, sample in descriptor_samples.items():
        if sample.feature_file_count == 0:
            raise ValueError(f"No feature files found for requested method {method} under {feature_dir}.")
        if sample.descriptors is None or sample.sampled_descriptor_count == 0:
            raise ValueError(
                f"No non-empty descriptors available for method {method}. "
                f"Checked {sample.feature_file_count} feature files and skipped {sample.empty_record_count} empty records."
            )

        codebook = train_codebook(
            sample.descriptors,
            method=method,
            trainer=trainer,
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=random_state,
        )
        save_path = save_codebook_artifact(codebook, output_dir)
        saved_paths.append(save_path)

        print(
            f"Method={method} files={sample.feature_file_count} empty={sample.empty_record_count} "
            f"seen={sample.total_descriptor_count} sampled={sample.sampled_descriptor_count} "
            f"clusters={codebook.n_clusters} saved={save_path}"
        )

    return saved_paths



def _get_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return value



def _get_nested_mapping(parent: dict[str, Any], key: str, field_name: str) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{field_name}' must be a mapping.")
    return value



def _resolve_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field '{field_name}' must be a non-empty path string.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path



def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field '{field_name}' must be a non-empty string.")
    return value.strip()



def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Config field '{field_name}' must be a positive integer.")

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config field '{field_name}' must be a positive integer.") from exc

    if parsed <= 0:
        raise ValueError(f"Config field '{field_name}' must be greater than zero, got {parsed}.")
    return parsed



def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Config field '{field_name}' must be an integer or null.")

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config field '{field_name}' must be an integer or null.") from exc



def main() -> int:
    args = parse_args()

    try:
        config, config_path = load_runtime_config(args.config)
        print(f"Loaded config from {config_path}")
        saved_paths = run_codebook_training(config, methods=args.method)
        print(f"Codebook training completed. Saved {len(saved_paths)} artifact(s).")
        return 0
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError, ImportError, RuntimeError, TypeError) as exc:
        print(f"Codebook training failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
