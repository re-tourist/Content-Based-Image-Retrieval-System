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
    """Scan an image directory and expose stable sample records."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "unspecified",
        supported_extensions: tuple[str, ...] | None = None,
        verbose: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.split = split
        self.supported_extensions = self._normalize_extensions(
            supported_extensions or DEFAULT_IMAGE_EXTENSIONS
        )
        self.samples = self._scan_samples()

        if verbose:
            print(f"Loaded {len(self.samples)} images from {self.root_dir}")
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

    def _build_sample(self, image_path: Path) -> ImageSample:
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
    def _resolve_image_path(sample: ImageSample | str | Path) -> Path:
        if isinstance(sample, ImageSample):
            return sample.image_path

        path = Path(sample).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image path not found: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Image path is not a file: {path}")
        return path
