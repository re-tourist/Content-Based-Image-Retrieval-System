from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import ImageDatasetLoader
from src.utils import get_default_config_path, load_config


DEFAULT_PREVIEW_COUNT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the stage-1 pipeline skeleton.")
    parser.add_argument(
        "--config",
        default=get_default_config_path(),
        help="Path to the runtime config file. Defaults to configs/base.yaml.",
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


def build_dataset_loader(config: dict[str, Any]) -> tuple[ImageDatasetLoader, Path, str]:
    dataset_config = config.get("dataset")
    if not isinstance(dataset_config, dict):
        raise ValueError("Config must contain a 'dataset' mapping.")

    dataset_root = _require_path(dataset_config.get("root"), "dataset.root")
    image_dir, resolved_from = resolve_dataset_image_dir(dataset_config, dataset_root)
    split = infer_split_name(dataset_root, image_dir)
    loader = ImageDatasetLoader(root_dir=image_dir, split=split, verbose=False)
    return loader, image_dir, resolved_from


def resolve_dataset_image_dir(dataset_config: dict[str, Any], dataset_root: Path) -> tuple[Path, str]:
    explicit_keys = ("image_dir", "images_dir", "input_dir")
    for key in explicit_keys:
        value = dataset_config.get(key)
        if value is None:
            continue
        image_dir = _require_path(value, f"dataset.{key}", base_dir=dataset_root)
        if not image_dir.exists():
            raise FileNotFoundError(
                f"Configured dataset image directory not found: {image_dir} (from dataset.{key})"
            )
        if not image_dir.is_dir():
            raise NotADirectoryError(
                f"Configured dataset image path is not a directory: {image_dir} (from dataset.{key})"
            )
        return image_dir, f"dataset.{key}"

    fallback_candidates = (
        (dataset_root / "train" / "image", "fallback: dataset.root/train/image"),
        (dataset_root / "test", "fallback: dataset.root/test"),
        (dataset_root / "raw", "fallback: dataset.root/raw"),
    )
    for candidate_path, source in fallback_candidates:
        if candidate_path.exists() and candidate_path.is_dir():
            return candidate_path.resolve(), source

    checked = ", ".join(str(path) for path, _ in fallback_candidates)
    raise FileNotFoundError(
        "Could not resolve a dataset image directory from config. "
        f"Checked: {checked}"
    )


def infer_split_name(dataset_root: Path, image_dir: Path) -> str:
    train_dir = (dataset_root / "train" / "image").resolve()
    test_dir = (dataset_root / "test").resolve()
    if image_dir == train_dir:
        return "train"
    if image_dir == test_dir:
        return "test"
    return "unspecified"


def print_dataset_summary(loader: ImageDatasetLoader, image_dir: Path, resolved_from: str) -> None:
    stats = loader.stats()
    print(f"Resolved dataset image directory: {image_dir} ({resolved_from})")
    print(f"Loaded {stats['total_images']} images from {image_dir}")
    print(f"Dataset split: {stats['split']}")
    print(f"Supported extensions: {', '.join(stats['supported_extensions'])}")


def run_dataset_preview(loader: ImageDatasetLoader, preview_count: int = DEFAULT_PREVIEW_COUNT) -> None:
    samples = loader.preview(min(preview_count, len(loader)))
    print(f"Previewing first {len(samples)} samples")
    for sample in samples:
        print(
            f"Sample: id={sample.sample_id} "
            f"file={sample.file_name} split={sample.split}"
        )

    for sample in samples:
        image = loader.load_image(sample)
        print(f"Read image: {sample.file_name} shape={tuple(int(dim) for dim in image.shape)}")

        # Future hook: preprocess
        # Future hook: local feature extraction
        # Future hook: feature saving
        # Future hook: visualization


def _require_path(value: Any, field_name: str, base_dir: Path | None = None) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field '{field_name}' must be a non-empty path string.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ((base_dir or PROJECT_ROOT) / path).resolve()
    else:
        path = path.resolve()
    return path


def main() -> int:
    args = parse_args()

    try:
        config, config_path = load_runtime_config(args.config)
        print(f"Loaded config from {config_path}")

        loader, image_dir, resolved_from = build_dataset_loader(config)
        print_dataset_summary(loader, image_dir, resolved_from)
        run_dataset_preview(loader)
        print("Pipeline skeleton run completed")
        return 0
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError, ImportError) as exc:
        print(f"Pipeline skeleton run failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
