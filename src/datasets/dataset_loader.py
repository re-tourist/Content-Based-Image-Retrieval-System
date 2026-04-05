from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(slots=True)
class ImageSample:
    """A minimal sample record used by the early project pipeline."""

    sample_id: str
    image_path: Path
    file_name: str
    split: str = "unspecified"
    meta: dict[str, Any] = field(default_factory=dict)


class ImageDatasetLoader:
    """Scan an image directory or load a split file and expose stable sample records."""

    def __init__(
        self,
        root_dir: str | Path | None = None,
        split: str = "unspecified",
        supported_extensions: tuple[str, ...] | None = None,
        split_file: str | Path | None = None,
        data_root: str | Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.split = split
        self.supported_extensions = self._normalize_extensions(
            supported_extensions or DEFAULT_IMAGE_EXTENSIONS
        )

        if split_file is not None:
            self.root_dir = self._resolve_directory_path(data_root, "data_root")
            self.split_file = self._resolve_path(split_file, "split_file")
            self.samples = self._load_samples_from_split()
        else:
            self.root_dir = self._resolve_directory_path(root_dir, "root_dir")
            self.split_file = None
            self.samples = self._scan_samples()

        if verbose:
            if self.split_file is None:
                print(f"Loaded {len(self.samples)} images from {self.root_dir}")
            else:
                print(
                    "Loaded "
                    f"{len(self.samples)} images from split file {self.split_file} "
                    f"with data root {self.root_dir}"
                )
            print(f"Supported extensions: {', '.join(self.supported_extensions)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ImageSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[ImageSample]:
        return iter(self.samples)

    def preview(self, n: int = 5) -> list[ImageSample]:
        """Return the first n sample records for quick inspection."""
        if n < 0:
            raise ValueError(f"Preview size must be non-negative, got {n}.")
        return self.samples[:n]

    def stats(self) -> dict[str, Any]:
        """Return lightweight dataset statistics for the current scan."""
        return {
            "root_dir": str(self.root_dir),
            "split": self.split,
            "total_images": len(self.samples),
            "supported_extensions": list(self.supported_extensions),
            "split_file": None if self.split_file is None else str(self.split_file),
        }

    def load_image(self, sample: ImageSample | str | Path, grayscale: bool = False) -> Any:
        """Read an image from a sample record or direct path using OpenCV."""
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required to load images. Install a package that provides 'cv2'."
            ) from exc

        image_path = self._resolve_image_path(sample)
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(str(image_path), flag)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        return image

    def _scan_samples(self) -> list[ImageSample]:
        self._validate_root_dir()

        candidate_paths = sorted(
            path
            for path in self.root_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.supported_extensions
        )

        if not candidate_paths:
            if not any(self.root_dir.iterdir()):
                raise ValueError(f"Dataset directory is empty: {self.root_dir}")
            raise ValueError(
                "No supported image files were found in "
                f"{self.root_dir}. Supported extensions: {', '.join(self.supported_extensions)}"
            )

        return [self._build_sample(path) for path in candidate_paths]

    def _load_samples_from_split(self) -> list[ImageSample]:
        self._validate_root_dir()
        self._validate_split_file()

        try:
            lines = self.split_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise OSError(f"Failed to read split file '{self.split_file}': {exc}") from exc

        relative_paths = [
            line.strip() for line in lines if line.strip() and not line.strip().startswith("#")
        ]
        if not relative_paths:
            raise ValueError(f"Split file is empty: {self.split_file}")

        samples: list[ImageSample] = []
        for relative_path in relative_paths:
            image_path = self._resolve_split_image_path(relative_path)
            samples.append(self._build_sample(image_path, relative_path=Path(relative_path)))
        return samples

    def _build_sample(self, image_path: Path, relative_path: Path | None = None) -> ImageSample:
        if relative_path is None:
            relative_path = image_path.relative_to(self.root_dir)

        sample_id = relative_path.as_posix()
        return ImageSample(
            sample_id=sample_id,
            image_path=image_path.resolve(),
            file_name=image_path.name,
            split=self.split,
            meta={"relative_path": sample_id},
        )

    def _validate_root_dir(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"Dataset path is not a directory: {self.root_dir}")

    def _validate_split_file(self) -> None:
        if self.split_file is None:
            raise ValueError("Split file is not configured.")
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        if not self.split_file.is_file():
            raise FileNotFoundError(f"Split path is not a file: {self.split_file}")

    def _resolve_split_image_path(self, relative_path: str) -> Path:
        image_path = Path(relative_path).expanduser()
        if not image_path.is_absolute():
            image_path = (self.root_dir / image_path).resolve()
        else:
            image_path = image_path.resolve()

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image path '{relative_path}' listed in split file was not found under {self.root_dir}."
            )
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image path '{relative_path}' listed in split file is not a file: {image_path}"
            )
        if image_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(
                "Image path listed in split file uses an unsupported extension: "
                f"{image_path}"
            )
        return image_path

    @staticmethod
    def _normalize_extensions(extensions: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        for extension in extensions:
            value = extension.strip().lower()
            if not value:
                continue
            if not value.startswith("."):
                value = f".{value}"
            if value not in normalized:
                normalized.append(value)

        if not normalized:
            raise ValueError("At least one supported image extension must be provided.")

        return tuple(normalized)

    @staticmethod
    def _resolve_directory_path(value: str | Path | None, field_name: str) -> Path:
        if value is None:
            raise ValueError(f"'{field_name}' is required.")
        return Path(value).expanduser().resolve()

    @staticmethod
    def _resolve_path(value: str | Path | None, field_name: str) -> Path:
        if value is None:
            raise ValueError(f"'{field_name}' is required.")
        return Path(value).expanduser().resolve()

    @staticmethod
    def _resolve_image_path(sample: ImageSample | str | Path) -> Path:
        if isinstance(sample, ImageSample):
            return sample.image_path

        path = Path(sample).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image path not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Image path is not a file: {path}")
        return path
