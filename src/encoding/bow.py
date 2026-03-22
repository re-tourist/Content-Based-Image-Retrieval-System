from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import pairwise_distances_argmin

from .codebook import CodebookArtifact, load_codebook_artifact
from .io import _normalize_method, build_encoding_input_from_feature_record, load_feature_npz
from .sampling import iter_feature_paths
from .storage import EncodedFeatureResult, save_encoded_feature
from .types import EncodingInput, FeatureFileRecord



def encode_bow_from_input(
    encoding_input: EncodingInput,
    codebook: CodebookArtifact,
    *,
    normalize: bool = False,
    codebook_path: str | Path | None = None,
) -> EncodedFeatureResult:
    """Encode one validated encoding input into a BoW histogram."""
    if not isinstance(encoding_input, EncodingInput):
        raise TypeError(
            "BoW encoding input must be an EncodingInput, "
            f"got {type(encoding_input).__name__}."
        )

    _validate_input_and_codebook(encoding_input, codebook, "encode_bow_from_input")
    resolved_codebook_path = _normalize_optional_codebook_path(codebook_path)
    histogram_dtype = np.float32 if normalize else np.int32
    histogram = np.zeros(codebook.n_clusters, dtype=histogram_dtype)

    if encoding_input.descriptors_present and encoding_input.descriptors is not None:
        visual_words = assign_visual_words(encoding_input.descriptors, codebook)
        counts = np.bincount(visual_words, minlength=codebook.n_clusters)
        if normalize:
            histogram = counts.astype(np.float32, copy=False)
            total = float(histogram.sum())
            if total > 0.0:
                histogram /= total
        else:
            histogram = counts.astype(np.int32, copy=False)

    return EncodedFeatureResult(
        sample_id=encoding_input.sample_id,
        method=encoding_input.method,
        encoding_type="bow",
        num_visual_words=codebook.n_clusters,
        histogram=histogram,
        histogram_dtype=str(histogram.dtype),
        num_descriptors=encoding_input.num_descriptors,
        descriptors_present=encoding_input.descriptors_present,
        codebook_path=resolved_codebook_path,
        normalized=bool(normalize),
    )



def encode_bow_from_feature_record(
    record: FeatureFileRecord,
    codebook: CodebookArtifact,
    *,
    normalize: bool = False,
    codebook_path: str | Path | None = None,
) -> EncodedFeatureResult:
    """Encode one validated feature-file record into a BoW histogram."""
    encoding_input = build_encoding_input_from_feature_record(record)
    return encode_bow_from_input(
        encoding_input,
        codebook,
        normalize=normalize,
        codebook_path=codebook_path,
    )



def encode_bow_from_feature_path(
    feature_path: str | Path,
    codebook: CodebookArtifact,
    *,
    normalize: bool = False,
    codebook_path: str | Path | None = None,
) -> EncodedFeatureResult:
    """Load a feature artifact and encode it into a BoW histogram."""
    record = load_feature_npz(feature_path)
    return encode_bow_from_feature_record(
        record,
        codebook,
        normalize=normalize,
        codebook_path=codebook_path,
    )



def encode_feature_file(
    feature_path: str | Path,
    codebook_dir: str | Path,
    output_dir: str | Path,
    *,
    normalize: bool = False,
) -> Path:
    """Encode one saved feature artifact with its matching method-specific codebook."""
    record = load_feature_npz(feature_path)
    codebook, codebook_path = _load_required_codebook_for_method(codebook_dir, record.method)
    encoded_result = encode_bow_from_feature_record(
        record,
        codebook,
        normalize=normalize,
        codebook_path=codebook_path,
    )
    return save_encoded_feature(encoded_result, output_dir)



def assign_visual_words(descriptors: np.ndarray, codebook: CodebookArtifact) -> np.ndarray:
    """Assign each descriptor to its nearest visual word index."""
    if not isinstance(descriptors, np.ndarray):
        raise TypeError(f"Descriptors must be a numpy.ndarray, got {type(descriptors).__name__}.")
    if descriptors.ndim != 2:
        raise ValueError(f"Descriptors must be a 2D array, got shape={descriptors.shape}.")
    if descriptors.shape[0] == 0:
        return np.empty((0,), dtype=np.int32)
    if descriptors.shape[1] != codebook.descriptor_dim:
        raise ValueError(
            f"Descriptor dimension {descriptors.shape[1]} does not match codebook descriptor_dim={codebook.descriptor_dim}."
        )

    descriptor_matrix = descriptors.astype(np.float32, copy=False)
    assignments = pairwise_distances_argmin(descriptor_matrix, codebook.cluster_centers, metric="euclidean")
    return np.asarray(assignments, dtype=np.int32)



def encode_feature_directory(
    feature_dir: str | Path,
    codebook_dir: str | Path,
    output_dir: str | Path,
    *,
    normalize: bool = False,
    methods: Iterable[str] | None = None,
) -> list[Path]:
    """Batch-encode saved feature artifacts with method-matched codebooks."""
    allowed_methods = _normalize_method_filter(methods)
    codebooks = _load_codebooks_by_method(codebook_dir, allowed_methods)
    if allowed_methods is not None:
        missing_methods = sorted(method for method in allowed_methods if method not in codebooks)
        if missing_methods:
            raise FileNotFoundError(
                f"Missing codebook artifact(s) for method(s): {', '.join(missing_methods)} under {Path(codebook_dir).expanduser()}."
            )

    saved_paths: list[Path] = []
    matched_feature_count = 0
    for feature_path in iter_feature_paths(feature_dir):
        record = load_feature_npz(feature_path)
        method = record.method
        if allowed_methods is not None and method not in allowed_methods:
            continue

        matched_feature_count += 1
        codebook, codebook_path = _require_codebook_entry(codebooks, method, codebook_dir)
        encoded_result = encode_bow_from_feature_record(
            record,
            codebook,
            normalize=normalize,
            codebook_path=codebook_path,
        )
        saved_paths.append(save_encoded_feature(encoded_result, output_dir))

    if matched_feature_count == 0:
        requested = "all methods" if allowed_methods is None else ", ".join(allowed_methods)
        raise ValueError(f"No feature artifacts matched for {requested} under {Path(feature_dir).expanduser()}.")

    return saved_paths



def _validate_input_and_codebook(encoding_input: EncodingInput, codebook: CodebookArtifact, source: str) -> None:
    if not isinstance(codebook, CodebookArtifact):
        raise TypeError(f"{source}: codebook must be a CodebookArtifact, got {type(codebook).__name__}.")

    method = _normalize_method(encoding_input.method)
    if method != codebook.method:
        raise ValueError(
            f"{source}: encoding method {method} does not match codebook method {codebook.method}."
        )
    if encoding_input.descriptor_dim != codebook.descriptor_dim:
        raise ValueError(
            f"{source}: descriptor_dim {encoding_input.descriptor_dim} does not match codebook descriptor_dim {codebook.descriptor_dim}."
        )
    if encoding_input.descriptor_dtype != codebook.descriptor_dtype:
        raise ValueError(
            f"{source}: descriptor_dtype {encoding_input.descriptor_dtype!r} does not match codebook descriptor_dtype {codebook.descriptor_dtype!r}."
        )



def _normalize_optional_codebook_path(codebook_path: str | Path | None) -> str | None:
    if codebook_path is None:
        return None
    path = Path(codebook_path).expanduser()
    return path.as_posix()



def _normalize_method_filter(methods: Iterable[str] | None) -> tuple[str, ...] | None:
    if methods is None:
        return None
    normalized = [_normalize_method(method) for method in methods]
    return tuple(sorted(set(normalized)))



def _load_required_codebook_for_method(
    codebook_dir: str | Path,
    method: str,
) -> tuple[CodebookArtifact, Path]:
    normalized_method = _normalize_method(method)
    codebooks = _load_codebooks_by_method(codebook_dir, methods=(normalized_method,))
    return _require_codebook_entry(codebooks, normalized_method, codebook_dir)



def _require_codebook_entry(
    codebooks: dict[str, tuple[CodebookArtifact, Path]],
    method: str,
    codebook_dir: str | Path,
) -> tuple[CodebookArtifact, Path]:
    codebook_entry = codebooks.get(method)
    if codebook_entry is None:
        raise FileNotFoundError(
            f"No codebook artifact available for method {method} under {Path(codebook_dir).expanduser()}."
        )
    return codebook_entry



def _load_codebooks_by_method(
    codebook_dir: str | Path,
    methods: tuple[str, ...] | None = None,
) -> dict[str, tuple[CodebookArtifact, Path]]:
    resolved_dir = Path(codebook_dir).expanduser()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Codebook directory not found: {resolved_dir}")
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"Codebook path is not a directory: {resolved_dir}")

    loaded: dict[str, tuple[CodebookArtifact, Path]] = {}
    for codebook_path in sorted(path for path in resolved_dir.glob("*.npz") if path.is_file()):
        artifact = load_codebook_artifact(codebook_path)
        if methods is not None and artifact.method not in methods:
            continue
        if artifact.method in loaded:
            previous_path = loaded[artifact.method][1]
            raise ValueError(
                f"Multiple codebook artifacts found for method {artifact.method}: {previous_path} and {codebook_path}."
            )
        loaded[artifact.method] = (artifact, codebook_path.resolve())

    return loaded
