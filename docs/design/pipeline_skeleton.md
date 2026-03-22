# Pipeline Skeleton

## Purpose

This document defines the runnable Stage 1 pipeline skeleton for the main project.
The goal is to keep the call chain explicit and stable before Stage 2 data preparation
and Stage 3 local feature extraction are implemented in depth.

## Current Stage Boundaries

Current runnable pipeline:

`dataset loader -> basic preprocess -> local feature extraction -> feature save -> keypoint visualization`

Current non-goals:

- No full preprocessing workflow
- No real SIFT / ORB extraction in the mainline
- No real feature file serialization
- No real keypoint drawing
- No encoding / TF-IDF / indexing / retrieval / rerank / QE / dense retrieval logic

## Stage Overview

| Stage | Module | Responsibility | Input | Output | Status |
| --- | --- | --- | --- | --- | --- |
| Dataset | `src/datasets/dataset_loader.py` | Scan image directory, build sample records, load images | dataset directory, sample record | `ImageSample`, loaded image | Connected |
| Basic Preprocess | `src/preprocess/basic_preprocess.py` | Validate image input and apply minimal resize or color conversion | image, preprocess config | `PreprocessResult` | Connected |
| Local Features | `src/features/local/local_feature_extractor.py` | Define local feature extraction result structure | processed image, feature config | `LocalFeatureResult` | Placeholder interface |
| Feature Save | `scripts/run_pipeline.py` | Reserve feature saving hook | sample, feature result, output config | no-op console placeholder | Placeholder hook |
| Visualization | `scripts/run_pipeline.py` | Reserve keypoint visualization hook | sample, image, feature result, output config | no-op console placeholder | Placeholder hook |

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

Current behavior:

- Returns a structured placeholder result
- Fixes the output shape for later real implementations:
  - `keypoints`
  - `descriptors`
  - `meta`
- Does not perform real SIFT / ORB extraction yet

### Feature Save

Current interface location:

- `save_feature_result(sample, feature_result, output_config)` in `scripts/run_pipeline.py`

Current behavior:

- Placeholder only
- Prints target feature directory and placeholder status

### Visualization

Current interface location:

- `visualize_keypoints(sample, image, feature_result, output_config)` in `scripts/run_pipeline.py`

Current behavior:

- Placeholder only
- Prints target figure directory and placeholder status

## Runnable Call Order

The current skeleton is executed in this order:

1. Load config from `configs/base.yaml`
2. Resolve dataset image directory
3. Build `ImageDatasetLoader`
4. Select a few sample records for demonstration
5. Load image from dataset stage
6. Run `preprocess_image(...)`
7. Run `extract_local_features(...)`
8. Run `save_feature_result(...)`
9. Run `visualize_keypoints(...)`
10. Exit cleanly

Pseudo-shape:

```python
image = loader.load_image(sample)
preprocess_result = preprocess_image(image, preprocess_config)
feature_result = extract_local_features(preprocess_result.image, feature_config)
save_feature_result(sample, feature_result, output_config)
visualize_keypoints(sample, preprocess_result.image, feature_result, output_config)
```

## Future Hook Positions

The current skeleton intentionally reserves the following future hook chain after local feature extraction:

1. Feature encoding
2. TF-IDF representation
3. Inverted index
4. Retrieval
5. Rerank
6. Query expansion
7. Dense global retrieval
8. Hybrid fusion

These are only reserved as hook positions in code comments and documentation.
They are not implemented in Stage 1.

## Implementation Status Summary

Already connected:

- Config loading
- Dataset loading
- Basic preprocess stage with resize and grayscale support
- Pipeline stage sequencing from the main script

Placeholder by design:

- Local feature extraction result generation
- Feature save stage
- Visualization stage
- All later retrieval modules

## Why This Skeleton Exists

This skeleton is not the final system architecture.
Its purpose is to:

- keep the main call chain runnable
- make module boundaries explicit
- prevent premature implementation of later stages
- give Stage 2 and Stage 3 a stable integration point
