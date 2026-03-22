from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.encoding import encode_feature_directory
from src.utils import get_default_config_path, load_config



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode saved feature artifacts into BoW histograms.")
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



def run_feature_encoding(config: dict[str, Any], methods: list[str] | None = None) -> list[Path]:
    encoding_config = _get_mapping(config, "encoding")
    input_config = _get_nested_mapping(encoding_config, "input", "encoding.input")
    codebook_config = _get_nested_mapping(encoding_config, "codebook", "encoding.codebook")
    bow_config = _get_nested_mapping(encoding_config, "bow", "encoding.bow")
    output_config = _get_mapping(config, "output")

    if not bool(bow_config.get("enabled", False)):
        raise ValueError("Config field 'encoding.bow.enabled' must be true to encode features.")

    feature_dir = _resolve_path(
        input_config.get("feature_dir", output_config.get("feature_dir", "outputs/features")),
        "encoding.input.feature_dir",
    )
    codebook_dir = _resolve_path(
        bow_config.get("codebook_dir", codebook_config.get("output_dir", "outputs/indices/codebooks")),
        "encoding.bow.codebook_dir",
    )
    encoded_dir = _resolve_path(
        bow_config.get("output_dir", "outputs/encoded"),
        "encoding.bow.output_dir",
    )
    normalize = _require_bool(bow_config.get("normalize", False), "encoding.bow.normalize")

    saved_paths = encode_feature_directory(
        feature_dir=feature_dir,
        codebook_dir=codebook_dir,
        output_dir=encoded_dir,
        normalize=normalize,
        methods=methods,
    )
    print(
        f"Encoded {len(saved_paths)} feature artifact(s) from {feature_dir} using codebooks in {codebook_dir}."
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



def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Config field '{field_name}' must be a boolean.")
    return value



def main() -> int:
    args = parse_args()

    try:
        config, config_path = load_runtime_config(args.config)
        print(f"Loaded config from {config_path}")
        saved_paths = run_feature_encoding(config, methods=args.method)
        print(f"BoW encoding completed. Saved {len(saved_paths)} artifact(s).")
        return 0
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError, ImportError, RuntimeError, TypeError) as exc:
        print(f"BoW encoding failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
