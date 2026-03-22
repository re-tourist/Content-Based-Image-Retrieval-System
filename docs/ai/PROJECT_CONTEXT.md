# Project Context

## Purpose of This File

This file is the current handoff context for engineers or coding agents joining the repository.
It summarizes the actual implemented system state and the next safe integration point.

## Repository State

Completed work at the time of writing:

- Milestone 1: System Skeleton
- Milestone 2: Data Preparation
- Milestone 3: Local Feature Extraction and Keypoint Visualization

The main runnable pipeline is no longer only a skeleton. It now performs:

- dataset loading
- preprocess
- real local feature extraction
- feature saving
- keypoint figure generation

## Main Entry Points

### `scripts/run_pipeline.py`

Primary engineering entry for the current pipeline.
It currently:

1. loads `configs/base.yaml`
2. resolves a dataset image directory
3. previews a small number of samples
4. preprocesses each image
5. extracts local features
6. saves `.npz` feature files
7. saves `.png` keypoint figures

Important note:

- the script currently uses directory-scan dataset loading, not split-file loading, even though split support exists in the loader

### `scripts/build_splits.py`

Builds `train.txt`, `gallery.txt`, and `query.txt` under `data/splits/`.
This is the current standardization tool for dataset manifests.

### `scripts/run_demo.py`

Launches the minimal Gradio demo.
The demo reuses current preprocess and local feature extraction logic, but its Top-K display is still a placeholder.

## Current Data Flow

Actual per-sample data flow:

`ImageSample -> load_image -> PreprocessResult -> LocalFeatureResult -> .npz feature file -> keypoint figure`

### Dataset stage

Primary type:

- `ImageSample(sample_id, image_path, file_name, split, meta)`

Key points:

- the loader supports both directory mode and split-file mode
- the main pipeline currently demonstrates directory mode

### Preprocess stage

Primary type:

- `PreprocessResult(image, original_shape, processed_shape, color_mode, meta)`

Current supported operations:

- input validation
- resize
- grayscale conversion

### Local feature stage

Primary type:

- `LocalFeatureResult(keypoints, descriptors, meta)`

Current supported methods:

- SIFT
- ORB

Serialized keypoint contract:

- `keypoints` is always `N x 7`
- columns are `x, y, size, angle, response, octave, class_id`

Descriptor contract:

- SIFT descriptors are typically `float32`, shape `N x 128`
- ORB descriptors are typically `uint8`, shape `N x 32`
- empty results are valid and must be handled downstream

## On-Disk Contracts

### Feature files

Location:

- `outputs/features/*.npz`

Stable keys:

- `sample_id`
- `method`
- `num_keypoints`
- `keypoints`
- `descriptors`
- `descriptors_present`
- `descriptor_shape`
- `descriptor_dtype`
- `keypoint_fields`

This is the primary handoff contract for the next stage.

### Visualization files

Location:

- `outputs/figures/keypoints_*.png`

Current visualization scope:

- single-image keypoint overlays only
- no matching lines
- no query-gallery figure composition

## Config Sections in Active Use

Current `configs/base.yaml` sections used by the implemented pipeline:

- `dataset`
- `preprocess`
- `local_feature`
- `visualization`
- `output`

Important operational fields:

- `preprocess.resize.enabled`
- `preprocess.resize.width`
- `preprocess.resize.height`
- `preprocess.color_mode`
- `local_feature.method`
- `local_feature.save`
- `local_feature.max_samples`
- `visualization.enabled`
- `visualization.save_keypoints`

## What Is Implemented

- dataset enumeration and image loading
- split manifest generation and loader compatibility
- preprocess result standardization
- real SIFT / ORB extraction
- `.npz` feature persistence
- `.png` keypoint visualization
- minimal demo reuse of preprocess and local feature extraction

## What Is Not Implemented

- feature encoding
- codebook creation
- descriptor aggregation / pooling beyond raw extraction
- TF-IDF
- indexing
- retrieval
- reranking
- dense retrieval
- hybrid fusion

## Next Stage: Feature Encoding

Feature Encoding should be the next inserted stage.
It should not modify the existing extraction contract.

### Expected encoding input

Encoding should consume descriptor matrices from either:

- `LocalFeatureResult.descriptors`
- or `outputs/features/*.npz`

It should also read:

- `method`
- `descriptors_present`
- `descriptor_dtype`
- `descriptor_shape`

### Integration point

The correct insertion point is after feature extraction and feature saving.
In practical terms, new encoding modules should be wired after:

- `extract_local_features(...)`
- `save_local_feature_result(...)`

and before any future indexing or retrieval modules.

### Design constraints for the next stage

A future encoding implementation must account for:

- empty descriptor cases
- SIFT and ORB using different descriptor dtypes and dimensionalities
- the current preview-oriented `run_pipeline.py` behavior
- the already-established `.npz` contract

The next stage should extend the pipeline by adding a new module and config section.
It should not replace or refactor the current local feature extraction stage.
