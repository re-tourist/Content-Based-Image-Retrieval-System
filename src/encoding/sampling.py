from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .io import _METHOD_SPECS, _normalize_method, build_encoding_input_from_feature_record, load_feature_npz


@dataclass(slots=True)
class DescriptorSample:
    """Bounded method-specific descriptor sample collected from saved feature files."""

    method: str
    descriptors: np.ndarray | None
    descriptor_dim: int
    descriptor_dtype: str
    total_descriptor_count: int
    sampled_descriptor_count: int
    feature_file_count: int
    empty_record_count: int


@dataclass(slots=True)
class _SamplingState:
    method: str
    descriptor_dim: int
    descriptor_dtype: str
    capacity: int
    reservoir: np.ndarray | None = None
    total_descriptor_count: int = 0
    sampled_descriptor_count: int = 0
    feature_file_count: int = 0
    empty_record_count: int = 0



def iter_feature_paths(feature_dir: str | Path) -> Iterator[Path]:
    """Yield saved feature artifact paths in a stable sorted order."""
    resolved_dir = Path(feature_dir).expanduser()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {resolved_dir}")
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"Feature path is not a directory: {resolved_dir}")

    yield from sorted(path for path in resolved_dir.glob("*.npz") if path.is_file())



def sample_descriptors_by_method(
    feature_dir: str | Path,
    max_descriptors: int,
    random_state: int | None = 42,
    methods: Iterable[str] | None = None,
) -> dict[str, DescriptorSample]:
    """Collect a bounded descriptor sample per method from saved feature artifacts."""
    capacity = _require_positive_int(max_descriptors, "max_descriptors")
    rng = np.random.default_rng(random_state)
    feature_paths = list(iter_feature_paths(feature_dir))
    if not feature_paths:
        raise FileNotFoundError(f"No feature files found under: {Path(feature_dir).expanduser()}")

    allowed_methods = _normalize_method_filter(methods)
    states: dict[str, _SamplingState] = {}
    if allowed_methods is not None:
        for method in allowed_methods:
            states[method] = _make_sampling_state(method, capacity)

    for feature_path in feature_paths:
        record = load_feature_npz(feature_path)
        encoding_input = build_encoding_input_from_feature_record(record)
        method = encoding_input.method

        if allowed_methods is not None and method not in allowed_methods:
            continue

        state = states.get(method)
        if state is None:
            state = _make_sampling_state(method, capacity)
            states[method] = state

        state.feature_file_count += 1
        if not encoding_input.descriptors_present or encoding_input.descriptors is None:
            state.empty_record_count += 1
            continue

        _validate_sampling_input(state, encoding_input, feature_path)
        _update_reservoir(state, encoding_input.descriptors, rng)

    return {method: _finalize_sampling_state(state) for method, state in sorted(states.items())}



def _normalize_method_filter(methods: Iterable[str] | None) -> tuple[str, ...] | None:
    if methods is None:
        return None

    normalized = []
    for method in methods:
        normalized.append(_normalize_method(method))
    return tuple(sorted(set(normalized)))



def _make_sampling_state(method: str, capacity: int) -> _SamplingState:
    spec = _METHOD_SPECS[method]
    return _SamplingState(
        method=method,
        descriptor_dim=spec.dim,
        descriptor_dtype=spec.dtype,
        capacity=capacity,
    )



def _validate_sampling_input(state: _SamplingState, encoding_input: Any, source: Path) -> None:
    if encoding_input.descriptor_dim != state.descriptor_dim:
        raise ValueError(
            f"{source}: descriptor_dim {encoding_input.descriptor_dim} does not match "
            f"existing {state.method} sampling dim {state.descriptor_dim}."
        )
    if encoding_input.descriptor_dtype != state.descriptor_dtype:
        raise ValueError(
            f"{source}: descriptor_dtype {encoding_input.descriptor_dtype!r} does not match "
            f"existing {state.method} sampling dtype {state.descriptor_dtype!r}."
        )



def _update_reservoir(state: _SamplingState, descriptors: np.ndarray, rng: np.random.Generator) -> None:
    if state.reservoir is None:
        state.reservoir = np.empty((state.capacity, state.descriptor_dim), dtype=descriptors.dtype)

    for row in descriptors:
        state.total_descriptor_count += 1
        if state.sampled_descriptor_count < state.capacity:
            state.reservoir[state.sampled_descriptor_count] = row
            state.sampled_descriptor_count += 1
            continue

        replace_index = int(rng.integers(0, state.total_descriptor_count))
        if replace_index < state.capacity:
            state.reservoir[replace_index] = row



def _finalize_sampling_state(state: _SamplingState) -> DescriptorSample:
    descriptors: np.ndarray | None = None
    if state.reservoir is not None and state.sampled_descriptor_count > 0:
        descriptors = state.reservoir[: state.sampled_descriptor_count].copy()

    return DescriptorSample(
        method=state.method,
        descriptors=descriptors,
        descriptor_dim=state.descriptor_dim,
        descriptor_dtype=state.descriptor_dtype,
        total_descriptor_count=state.total_descriptor_count,
        sampled_descriptor_count=state.sampled_descriptor_count,
        feature_file_count=state.feature_file_count,
        empty_record_count=state.empty_record_count,
    )



def _require_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc

    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than zero, got {parsed}.")
    return parsed

