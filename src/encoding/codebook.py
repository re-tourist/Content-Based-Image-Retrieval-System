from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.npyio import NpzFile
from sklearn.cluster import KMeans, MiniBatchKMeans

from .io import _METHOD_SPECS, _normalize_method


_SUPPORTED_TRAINERS = ("minibatch_kmeans", "kmeans")
_REQUIRED_CODEBOOK_KEYS = (
    "method",
    "trainer",
    "n_clusters",
    "descriptor_dim",
    "descriptor_dtype",
    "training_descriptor_count",
    "random_state",
    "batch_size",
    "cluster_centers",
)


@dataclass(slots=True)
class CodebookArtifact:
    """Reusable method-specific codebook artifact."""

    method: str
    trainer: str
    n_clusters: int
    descriptor_dim: int
    descriptor_dtype: str
    training_descriptor_count: int
    random_state: int | None
    batch_size: int | None
    cluster_centers: np.ndarray



def train_codebook(
    descriptors: np.ndarray,
    method: str,
    *,
    trainer: str = "minibatch_kmeans",
    n_clusters: int = 256,
    batch_size: int = 1024,
    random_state: int | None = 42,
) -> CodebookArtifact:
    """Train one method-specific codebook from a validated descriptor matrix."""
    normalized_method = _normalize_method(method)
    normalized_trainer = _normalize_trainer(trainer)
    validated_descriptors = _validate_training_descriptors(descriptors, normalized_method, "train_codebook")
    resolved_clusters = _require_positive_int(n_clusters, "n_clusters")
    resolved_random_state = _normalize_optional_int(random_state, "random_state")

    if validated_descriptors.shape[0] < resolved_clusters:
        raise ValueError(
            f"train_codebook: n_clusters={resolved_clusters} exceeds descriptor rows={validated_descriptors.shape[0]}."
        )

    training_data = validated_descriptors.astype(np.float32, copy=False)
    resolved_batch_size: int | None = None
    if normalized_trainer == "minibatch_kmeans":
        resolved_batch_size = _require_positive_int(batch_size, "batch_size")
        estimator = MiniBatchKMeans(
            n_clusters=resolved_clusters,
            batch_size=resolved_batch_size,
            random_state=resolved_random_state,
            n_init=3,
        )
    else:
        estimator = KMeans(
            n_clusters=resolved_clusters,
            random_state=resolved_random_state,
            n_init=3,
        )

    estimator.fit(training_data)
    artifact = CodebookArtifact(
        method=normalized_method,
        trainer=normalized_trainer,
        n_clusters=resolved_clusters,
        descriptor_dim=int(validated_descriptors.shape[1]),
        descriptor_dtype=str(validated_descriptors.dtype),
        training_descriptor_count=int(validated_descriptors.shape[0]),
        random_state=resolved_random_state,
        batch_size=resolved_batch_size,
        cluster_centers=np.asarray(estimator.cluster_centers_, dtype=np.float32),
    )
    _validate_codebook_artifact(artifact, "trained codebook")
    return artifact



def save_codebook_artifact(codebook: CodebookArtifact, output_dir: str | Path) -> Path:
    """Save a codebook artifact as a compressed `.npz` file."""
    _validate_codebook_artifact(codebook, "save_codebook_artifact")

    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    save_path = target_dir / _make_codebook_name(codebook)

    np.savez_compressed(
        save_path,
        method=np.array(codebook.method),
        trainer=np.array(codebook.trainer),
        n_clusters=np.array(codebook.n_clusters, dtype=np.int32),
        descriptor_dim=np.array(codebook.descriptor_dim, dtype=np.int32),
        descriptor_dtype=np.array(codebook.descriptor_dtype),
        training_descriptor_count=np.array(codebook.training_descriptor_count, dtype=np.int32),
        random_state=np.array(-1 if codebook.random_state is None else codebook.random_state, dtype=np.int32),
        batch_size=np.array(-1 if codebook.batch_size is None else codebook.batch_size, dtype=np.int32),
        cluster_centers=np.asarray(codebook.cluster_centers, dtype=np.float32),
    )
    return save_path



def load_codebook_artifact(codebook_path: str | Path) -> CodebookArtifact:
    """Load and validate a saved codebook artifact."""
    resolved_path = Path(codebook_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Codebook artifact not found: {resolved_path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Codebook path is not a file: {resolved_path}")

    with np.load(resolved_path, allow_pickle=False) as data:
        _validate_required_keys(data, resolved_path)

        artifact = CodebookArtifact(
            method=_normalize_method(_read_required_string(data, "method", resolved_path)),
            trainer=_normalize_trainer(_read_required_string(data, "trainer", resolved_path)),
            n_clusters=_read_required_int(data, "n_clusters", resolved_path),
            descriptor_dim=_read_required_int(data, "descriptor_dim", resolved_path),
            descriptor_dtype=_read_required_string(data, "descriptor_dtype", resolved_path),
            training_descriptor_count=_read_required_int(data, "training_descriptor_count", resolved_path),
            random_state=_read_optional_int(data, "random_state", resolved_path),
            batch_size=_read_optional_int(data, "batch_size", resolved_path),
            cluster_centers=np.array(data["cluster_centers"], copy=True),
        )

    _validate_codebook_artifact(artifact, resolved_path)
    return artifact



def _validate_training_descriptors(descriptors: Any, method: str, source: str) -> np.ndarray:
    if not isinstance(descriptors, np.ndarray):
        raise TypeError(f"{source}: descriptors must be a numpy.ndarray, got {type(descriptors).__name__}.")
    if descriptors.ndim != 2:
        raise ValueError(f"{source}: descriptors must be a 2D array, got shape={descriptors.shape}.")
    if descriptors.shape[0] <= 0:
        raise ValueError(f"{source}: descriptors must contain at least one row.")

    spec = _METHOD_SPECS[method]
    actual_dtype = str(descriptors.dtype)
    actual_dim = int(descriptors.shape[1])
    if actual_dtype != spec.dtype:
        raise ValueError(f"{source}: {method} descriptors must have dtype {spec.dtype}, got {actual_dtype}.")
    if actual_dim != spec.dim:
        raise ValueError(f"{source}: {method} descriptors must have dimension {spec.dim}, got {actual_dim}.")
    return descriptors



def _validate_codebook_artifact(codebook: CodebookArtifact, source: Any) -> None:
    if not isinstance(codebook, CodebookArtifact):
        raise TypeError(f"{source}: codebook must be a CodebookArtifact, got {type(codebook).__name__}.")

    method = _normalize_method(codebook.method)
    trainer = _normalize_trainer(codebook.trainer)
    n_clusters = _require_positive_int(codebook.n_clusters, "n_clusters")
    descriptor_dim = _require_positive_int(codebook.descriptor_dim, "descriptor_dim")
    training_descriptor_count = _require_positive_int(codebook.training_descriptor_count, "training_descriptor_count")
    random_state = _normalize_optional_int(codebook.random_state, "random_state")
    batch_size = None if codebook.batch_size is None else _require_positive_int(codebook.batch_size, "batch_size")

    spec = _METHOD_SPECS[method]
    if descriptor_dim != spec.dim:
        raise ValueError(f"{source}: {method} codebook descriptor_dim must be {spec.dim}, got {descriptor_dim}.")
    if codebook.descriptor_dtype != spec.dtype:
        raise ValueError(
            f"{source}: {method} codebook descriptor_dtype must be {spec.dtype}, got {codebook.descriptor_dtype!r}."
        )
    if training_descriptor_count < n_clusters:
        raise ValueError(
            f"{source}: training_descriptor_count={training_descriptor_count} must be >= n_clusters={n_clusters}."
        )
    if trainer == "minibatch_kmeans" and batch_size is None:
        raise ValueError(f"{source}: batch_size is required for trainer 'minibatch_kmeans'.")

    centers = codebook.cluster_centers
    if not isinstance(centers, np.ndarray):
        raise TypeError(f"{source}: cluster_centers must be a numpy.ndarray, got {type(centers).__name__}.")
    if centers.ndim != 2:
        raise ValueError(f"{source}: cluster_centers must be a 2D array, got shape={centers.shape}.")
    if centers.shape != (n_clusters, descriptor_dim):
        raise ValueError(
            f"{source}: cluster_centers shape {centers.shape} does not match "
            f"(n_clusters={n_clusters}, descriptor_dim={descriptor_dim})."
        )
    if not np.issubdtype(centers.dtype, np.floating):
        raise ValueError(f"{source}: cluster_centers must use a floating dtype, got {centers.dtype}.")

    codebook.method = method
    codebook.trainer = trainer
    codebook.n_clusters = n_clusters
    codebook.descriptor_dim = descriptor_dim
    codebook.training_descriptor_count = training_descriptor_count
    codebook.random_state = random_state
    codebook.batch_size = batch_size
    codebook.cluster_centers = np.asarray(centers, dtype=np.float32)



def _make_codebook_name(codebook: CodebookArtifact) -> str:
    return f"codebook_{codebook.method.lower()}_k{codebook.n_clusters}_{codebook.trainer}.npz"



def _validate_required_keys(data: NpzFile, source: Path) -> None:
    missing_keys = [key for key in _REQUIRED_CODEBOOK_KEYS if key not in data.files]
    if missing_keys:
        raise ValueError(f"{source}: missing required codebook keys: {', '.join(missing_keys)}.")



def _normalize_trainer(trainer: str) -> str:
    if not isinstance(trainer, str) or not trainer.strip():
        raise ValueError("Codebook trainer must be a non-empty string.")

    normalized = trainer.strip().lower()
    if normalized not in _SUPPORTED_TRAINERS:
        raise ValueError(
            f"Unsupported codebook trainer: {trainer!r}. Supported trainers: {', '.join(_SUPPORTED_TRAINERS)}."
        )
    return normalized



def _read_required_string(data: NpzFile, key: str, source: Any) -> str:
    value = data[key]
    if value.ndim != 0:
        raise ValueError(f"{source}: key '{key}' must be a scalar string value.")

    scalar = value.item()
    if not isinstance(scalar, str) or not scalar.strip():
        raise ValueError(f"{source}: key '{key}' must be a non-empty string.")
    return scalar.strip()



def _read_required_int(data: NpzFile, key: str, source: Any) -> int:
    value = data[key]
    if value.ndim != 0:
        raise ValueError(f"{source}: key '{key}' must be a scalar integer value.")

    return _require_positive_int(value.item(), f"{source}: key '{key}'")



def _read_optional_int(data: NpzFile, key: str, source: Any) -> int | None:
    value = data[key]
    if value.ndim != 0:
        raise ValueError(f"{source}: key '{key}' must be a scalar integer value.")

    parsed = int(value.item())
    if parsed == -1:
        return None
    if parsed < 0:
        raise ValueError(f"{source}: key '{key}' must be >= 0 or -1 for None, got {parsed}.")
    return parsed



def _normalize_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer or None.")

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer or None.") from exc



def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc

    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than zero, got {parsed}.")
    return parsed

