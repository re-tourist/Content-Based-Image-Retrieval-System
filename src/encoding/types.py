from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FeatureFileRecord:
    """Validated view of one saved local-feature artifact."""

    sample_id: str
    method: str
    num_keypoints: int
    keypoints: np.ndarray
    descriptors: np.ndarray | None
    descriptors_present: bool
    descriptor_shape: tuple[int, ...]
    descriptor_dtype: str | None


@dataclass(slots=True)
class EncodingInput:
    """Normalized encoding-stage input for one sample."""

    sample_id: str | None
    method: str
    descriptors: np.ndarray | None
    descriptors_present: bool
    descriptor_shape: tuple[int, int]
    descriptor_dtype: str
    descriptor_dim: int
    num_descriptors: int
