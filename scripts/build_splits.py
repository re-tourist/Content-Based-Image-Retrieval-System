from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset_loader import DEFAULT_IMAGE_EXTENSIONS
from src.utils import get_default_config_path, load_config



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standard dataset split files.")
    parser.add_argument(
        "--config",
        default=get_default_config_path(),
        help="Path to the runtime config file. Defaults to configs/base.yaml.",
    )
    return parser.parse_args()



def collect_images(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset source directory not found: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Dataset source path is not a directory: {root_dir}")

    images = sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(
            "No supported image files were found in "
            f"{root_dir}. Supported extensions: {', '.join(DEFAULT_IMAGE_EXTENSIONS)}"
        )
    return images



def build_train_split(train_root: Path, data_root: Path) -> list[str]:
    return [path.relative_to(data_root).as_posix() for path in collect_images(train_root)]



def build_query_and_gallery_splits(test_root: Path, data_root: Path) -> tuple[list[str], list[str]]:
    query_paths: list[str] = []
    gallery_paths: list[str] = []

    class_directories = sorted(path for path in test_root.iterdir() if path.is_dir())
    if class_directories:
        for class_directory in class_directories:
            class_images = collect_images(class_directory)
            query_paths.append(class_images[0].relative_to(data_root).as_posix())
            gallery_paths.extend(
                image_path.relative_to(data_root).as_posix() for image_path in class_images[1:]
            )
        return query_paths, gallery_paths

    test_images = collect_images(test_root)
    query_paths.append(test_images[0].relative_to(data_root).as_posix())
    gallery_paths.extend(image_path.relative_to(data_root).as_posix() for image_path in test_images[1:])
    return query_paths, gallery_paths



def write_split_file(split_path: Path, entries: list[str]) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text("\n".join(entries) + "\n", encoding="utf-8")



def build_splits(config_path: str) -> int:
    config = load_config(config_path)
    dataset_config = config.get("dataset")
    if not isinstance(dataset_config, dict):
        raise ValueError("Config must contain a 'dataset' mapping.")

    data_root = require_directory(dataset_config.get("root"), "dataset.root")
    raw_dir = require_path(dataset_config.get("raw_dir"), "dataset.raw_dir")
    processed_dir = require_path(dataset_config.get("processed_dir"), "dataset.processed_dir")
    splits_dir = require_path(dataset_config.get("splits_dir"), "dataset.splits_dir")
    train_source_dir = require_directory(
        dataset_config.get("train_source_dir", data_root / "train" / "image"),
        "dataset.train_source_dir",
    )
    test_source_dir = require_directory(
        dataset_config.get("test_source_dir", data_root / "test"),
        "dataset.test_source_dir",
    )
    train_split_path = require_path(
        dataset_config.get("train_split", splits_dir / "train.txt"),
        "dataset.train_split",
    )
    gallery_split_path = require_path(
        dataset_config.get("gallery_split", splits_dir / "gallery.txt"),
        "dataset.gallery_split",
    )
    query_split_path = require_path(
        dataset_config.get("query_split", splits_dir / "query.txt"),
        "dataset.query_split",
    )

    for directory in (raw_dir, processed_dir, splits_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_entries = build_train_split(train_source_dir, data_root)
    query_entries, gallery_entries = build_query_and_gallery_splits(test_source_dir, data_root)

    write_split_file(train_split_path, train_entries)
    write_split_file(query_split_path, query_entries)
    write_split_file(gallery_split_path, gallery_entries)

    print(f"Data root: {data_root}")
    print(f"Train source: {train_source_dir}")
    print(f"Test source: {test_source_dir}")
    print(f"Generated {train_split_path.name} with {len(train_entries)} entries")
    print(f"Generated {query_split_path.name} with {len(query_entries)} entries")
    print(f"Generated {gallery_split_path.name} with {len(gallery_entries)} entries")
    print(
        "Compatibility note: split files standardize access while the current source images "
        "remain under data/train/image and data/test."
    )
    return 0



def require_directory(value: str | Path | None, field_name: str) -> Path:
    path = require_path(value, field_name)
    if not path.exists():
        raise FileNotFoundError(f"Configured directory not found: {path} (from {field_name})")
    if not path.is_dir():
        raise NotADirectoryError(f"Configured path is not a directory: {path} (from {field_name})")
    return path



def require_path(value: str | Path | None, field_name: str) -> Path:
    if value is None:
        raise ValueError(f"Config field '{field_name}' must be set.")
    return Path(value).expanduser().resolve()



def main() -> int:
    args = parse_args()
    try:
        return build_splits(args.config)
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as exc:
        print(f"Split build failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
