from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"



def get_default_config_path() -> str:
    """Return the default config path for project scripts."""
    return str(DEFAULT_CONFIG_PATH)



def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML config file and return a plain Python dictionary."""
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Config path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config '{path}': {exc}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read config file '{path}': {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file '{path}' must contain a top-level mapping, got {type(config).__name__}."
        )

    return _resolve_paths(config)



def _resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve configured relative paths into project-friendly absolute strings."""
    dataset = config.get("dataset")
    if isinstance(dataset, dict):
        dataset_root = _to_absolute_path(dataset.get("root"), PROJECT_ROOT)
        if dataset_root is not None:
            dataset["root"] = dataset_root

            for key in (
                "raw_dir",
                "processed_dir",
                "splits_dir",
                "train_source_dir",
                "test_source_dir",
            ):
                resolved = _to_absolute_path(dataset.get(key), Path(dataset_root))
                if resolved is not None:
                    dataset[key] = resolved

            splits_dir = dataset.get("splits_dir")
            if isinstance(splits_dir, str):
                for key in ("train_split", "gallery_split", "query_split"):
                    resolved = _to_absolute_path(dataset.get(key), Path(splits_dir))
                    if resolved is not None:
                        dataset[key] = resolved

    output = config.get("output")
    if isinstance(output, dict):
        for key in ("feature_dir", "index_dir", "figure_dir", "log_dir"):
            resolved = _to_absolute_path(output.get(key), PROJECT_ROOT)
            if resolved is not None:
                output[key] = resolved

    return config



def _to_absolute_path(value: Any, base_dir: Path) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected a path string, got {type(value).__name__}.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return str(path)
