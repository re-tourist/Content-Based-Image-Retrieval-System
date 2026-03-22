# Pipeline Skeleton

## Purpose

This document defines the runnable pipeline skeleton for the main project.
The goal is to keep the call chain explicit and stable while the system moves from
Stage 1 skeleton work into Stage 2 data preparation and Stage 3 local feature extraction
and visualization.

## Current Stage Boundaries

Current runnable pipeline:

`dataset loader -> basic preprocess -> local feature extraction -> feature save -> keypoint visualization`

Current non-goals:

- No image matching workflow
- No retrieval, ranking, or reranking pipeline
- No codebook / TF-IDF / inverted index logic
- No query expansion or dense retrieval logic
- No query-gallery comparison figures yet

## Stage Overview

| Stage | Module | Responsibility | Input | Output | Status |
| --- | --- | --- | --- | --- | --- |
| Dataset | `src/datasets/dataset_loader.py` | Scan image directory, build sample records, load images | dataset directory, sample record | `ImageSample`, loaded image | Connected |
| Basic Preprocess | `src/preprocess/basic_preprocess.py` | Validate image input and apply minimal resize or color conversion | image, preprocess config | `PreprocessResult` | Connected |
| Local Features | `src/features/local/local_feature_extractor.py` | Extract real local features with OpenCV and return serializable results | processed image, local feature config | `LocalFeatureResult` | Connected |
| Feature Save | `scripts/run_pipeline.py` + `src/features/local/local_feature_extractor.py` | Save extracted features as `.npz` files | sample id, feature result, output config | `outputs/features/*.npz` | Connected |
| Visualization | `src/visualization/keypoints.py` + `scripts/run_pipeline.py` | Render and save single-image keypoint overlays | sample id, processed image, keypoints, output config | `outputs/figures/keypoints_*.png` | Connected |

## Module Interfaces

### Dataset

Primary types and functions:

- `ImageSample`
- `ImageDatasetLoader`
- `ImageDatasetLoader.load_image(sample)`

Behavior:

- Provides ordered sample records
- Provides image loading for downstream stages
- Keeps sample metadata stable for future split/query/gallery extensions

### Basic Preprocess

Primary types and functions:

- `PreprocessResult`
- `preprocess_image(image, config)`

Current behavior:

- Validates that the input is a non-empty OpenCV-style `numpy.ndarray`
- Supports minimal resize through `preprocess.resize.enabled`, `width`, and `height`
- Supports `color_mode=keep` and `color_mode=gray`
- Returns structured metadata including original shape, processed shape, color mode and applied steps
- Does not perform augmentation, denoising, caching or deep learning style normalization

### Local Feature Extraction

Primary types and functions:

- `LocalFeatureResult`
- `extract_local_features(image, config)`
- `save_local_feature_result(sample_id, feature_result, output_dir)`

Current behavior:

- Supports `SIFT` and `ORB`
- Converts OpenCV keypoints into a serializable `Nx7` array with the columns:
  - `x`
  - `y`
  - `size`
  - `angle`
  - `response`
  - `octave`
  - `class_id`
- Returns descriptor arrays exactly as produced by OpenCV
- Handles empty feature results without crashing
- Does not perform matching, encoding, indexing or retrieval

### Feature Save

Saved file format:

- compressed `.npz`

Saved content includes at least:

- `sample_id`
- `method`
- `num_keypoints`
- `keypoints`
- `descriptors`
- descriptor presence and descriptor metadata

### Keypoint Visualization

Primary types and functions:

- `render_keypoint_overlay(image, keypoints, label=None)`
- `save_keypoint_visualization(sample_id, image, keypoints, output_dir, label=None)`

Current behavior:

- draws serialized keypoints directly from the `LocalFeatureResult.keypoints` array
- saves single-image overlays as `outputs/figures/keypoints_*.png`
- handles empty keypoint arrays by saving a valid image with a keypoint count label
- does not implement matching lines, RANSAC views, or retrieval result layouts

## Runnable Call Order

The current skeleton is executed in this order:

1. Load config from `configs/base.yaml`
2. Resolve dataset image directory
3. Build `ImageDatasetLoader`
4. Select a few sample records for demonstration
5. Load image from dataset stage
6. Run `preprocess_image(...)`
7. Run `extract_local_features(...)`
8. Save features to `outputs/features/*.npz`
9. Save keypoint figures to `outputs/figures/keypoints_*.png`
10. Exit cleanly

Pseudo-shape:

```python
image = loader.load_image(sample)
preprocess_result = preprocess_image(image, preprocess_config)
feature_result = extract_local_features(preprocess_result.image, local_feature_config)
save_path = save_local_feature_result(sample.sample_id, feature_result, output_dir)
figure_path = save_keypoint_visualization(
    sample.sample_id,
    preprocess_result.image,
    feature_result.keypoints,
    figure_dir,
)
```

## Future Hook Positions

The current skeleton intentionally reserves the following future hook chain after local feature extraction, feature saving, and single-image keypoint visualization:

1. Feature encoding
2. TF-IDF representation
3. Inverted index
4. Retrieval
5. Rerank
6. Query expansion
7. Dense global retrieval
8. Hybrid fusion

These are only reserved as hook positions in code comments and documentation.
They are not implemented in Stage 3.2.

## Implementation Status Summary

Already connected:

- Config loading
- Dataset loading
- Basic preprocess stage with resize and grayscale support
- Real local feature extraction with SIFT and ORB
- Feature saving to `.npz`
- Keypoint visualization saving to `.png`
- Pipeline stage sequencing from the main script

Placeholder by design:

- Query-gallery comparison visualization
- All later retrieval modules

## Why This Skeleton Exists

This skeleton is not the final retrieval system architecture.
Its purpose is to:

- keep the main call chain runnable
- make module boundaries explicit
- produce reusable feature files and figures for later stages
- prevent premature implementation of matching and retrieval logic
