from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LocalFeatureResult:
    """Structured placeholder output for the local feature stage."""

    keypoints: list[Any] = field(default_factory=list)
    descriptors: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def extract_local_features(image: Any, config: dict[str, Any] | None = None) -> LocalFeatureResult:
    """Return a placeholder feature result while keeping the interface stable."""
    config = config or {}
    method = str(config.get("local_method", "placeholder")).upper()

    return LocalFeatureResult(
        keypoints=[],
        descriptors=None,
        meta={
            "stage": "local_feature_extraction",
            "status": "placeholder",
            "method": method,
            "input_shape": _shape_of(image),
            "message": "Local feature extraction is intentionally a placeholder in Issue 1.3.",
        },
    )


def _shape_of(image: Any) -> tuple[int, ...] | None:
    shape = getattr(image, "shape", None)
    if not isinstance(shape, tuple):
        return None
    return tuple(int(dim) for dim in shape)
