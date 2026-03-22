from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import ImageDatasetLoader, ImageSample
from src.features.local import LocalFeatureResult, extract_local_features
from src.preprocess import PreprocessResult, preprocess_image
from src.utils import get_default_config_path, load_config


DEFAULT_PREVIEW_COUNT = 3



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the current pipeline skeleton.")
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



def run_pipeline_skeleton(loader: ImageDatasetLoader, config: dict[str, Any]) -> None:
    preprocess_config = _get_mapping(config, "preprocess")
    feature_config = _get_mapping(config, "feature")
    output_config = _get_mapping(config, "output")
    samples = loader.preview(min(DEFAULT_PREVIEW_COUNT, len(loader)))

    print(f"Running pipeline skeleton on first {len(samples)} samples")
    for sample in samples:
        print(f"Sample: id={sample.sample_id} file={sample.file_name} split={sample.split}")

        image = loader.load_image(sample)
        print(f"Loaded image with shape={_shape_of(image)}")

        preprocess_result = preprocess_image(image, preprocess_config)
        print_preprocess_stage(sample, preprocess_result)

        feature_result = extract_local_features(preprocess_result.image, feature_config)
        print_local_feature_stage(sample, feature_result)

        save_feature_result(sample, feature_result, output_config)
        visualize_keypoints(sample, preprocess_result.image, feature_result, output_config)

        # Future hooks: feature encoding -> tf-idf -> inverted index -> retrieval
        # -> rerank -> query expansion -> dense global retrieval -> hybrid fusion



def print_preprocess_stage(sample: ImageSample, result: PreprocessResult) -> None:
    resize_meta = result.meta.get("resize", {})
    resize_text = "disabled"
    if isinstance(resize_meta, dict) and resize_meta.get("enabled"):
        resize_text = f"enabled -> ({resize_meta.get('height')}, {resize_meta.get('width')})"

    print(
        "Preprocess stage completed for "
        f"{sample.file_name} original_shape={result.original_shape} "
        f"processed_shape={result.processed_shape} color_mode={result.color_mode} "
        f"resize={resize_text} steps={result.meta.get('applied_steps')}"
    )



def print_local_feature_stage(sample: ImageSample, result: LocalFeatureResult) -> None:
    print(
        "Local feature stage placeholder executed for "
        f"{sample.file_name} method={result.meta.get('method')} "
        f"keypoints={len(result.keypoints)}"
    )



def save_feature_result(
    sample: ImageSample,
    feature_result: LocalFeatureResult,
    output_config: dict[str, Any],
) -> None:
    feature_dir = output_config.get("feature_dir", "outputs/features")
    print(
        "Save stage placeholder executed for "
        f"{sample.file_name} target={feature_dir} descriptors={feature_result.descriptors is not None}"
    )



def visualize_keypoints(
    sample: ImageSample,
    image: Any,
    feature_result: LocalFeatureResult,
    output_config: dict[str, Any],
) -> None:
    figure_dir = output_config.get("figure_dir", "outputs/figures")
    print(
        "Visualization stage placeholder executed for "
        f"{sample.file_name} target={figure_dir} input_shape={_shape_of(image)} "
        f"keypoints={len(feature_result.keypoints)}"
    )



def _require_path(value: Any, field_name: str, base_dir: Path | None = None) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field '{field_name}' must be a non-empty path string.")

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ((base_dir or PROJECT_ROOT) / path).resolve()
    else:
        path = path.resolve()
    return path



def _get_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return value



def _shape_of(image: Any) -> tuple[int, ...] | None:
    shape = getattr(image, "shape", None)
    if not isinstance(shape, tuple):
        return None
    return tuple(int(dim) for dim in shape)



def main() -> int:
    args = parse_args()

    try:
        config, config_path = load_runtime_config(args.config)
        print(f"Loaded config from {config_path}")

        loader, image_dir, resolved_from = build_dataset_loader(config)
        print_dataset_summary(loader, image_dir, resolved_from)
        run_pipeline_skeleton(loader, config)
        print("Pipeline skeleton run completed")
        return 0
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError, ImportError, TypeError) as exc:
        print(f"Pipeline skeleton run failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
