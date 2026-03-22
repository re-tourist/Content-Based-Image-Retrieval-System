# Pipeline Skeleton

## Scope

Companion document: docs/ai/PROJECT_CONTEXT.md.

This document describes the pipeline that is actually implemented in the repository today.
It is intended to be code-aligned and usable as the handoff point for the next stage:
Feature Encoding.

## Actual End-to-End Flow

The current runnable path in `scripts/run_pipeline.py` is:

`raw image -> dataset loader -> preprocess -> local feature extraction -> feature save -> keypoint visualization`

In concrete terms:

1. Resolve config from `configs/base.yaml`
2. Resolve a dataset image directory
3. Build `ImageDatasetLoader`
4. Read a few sample images
5. Preprocess each image
6. Extract local features
7. Save features to `outputs/features/*.npz`
8. Save keypoint figures to `outputs/figures/keypoints_*.png`

## Stage-by-Stage Description

| Stage | Main Module | Input | Output | Responsibility | Status |
| --- | --- | --- | --- | --- | --- |
| Dataset Loader | `src/datasets/dataset_loader.py` | dataset directory or split file | `ImageSample`, loaded `numpy.ndarray` image | enumerate samples and load image bytes through OpenCV | Implemented |
| Preprocess | `src/preprocess/basic_preprocess.py` | OpenCV image, preprocess config | `PreprocessResult` | validate image, resize, optional grayscale conversion | Implemented |
| Local Feature Extraction | `src/features/local/local_feature_extractor.py` | preprocessed image, local feature config | `LocalFeatureResult` | run SIFT or ORB and return serializable feature data | Implemented |
| Feature Save | `src/features/local/local_feature_extractor.py` + `scripts/run_pipeline.py` | `sample_id`, `LocalFeatureResult`, output dir | `.npz` file under `outputs/features/` | persist extracted features in a stable on-disk format | Implemented |
| Keypoint Visualization | `src/visualization/keypoints.py` + `scripts/run_pipeline.py` | preprocessed image, serialized keypoints, output dir | `.png` figure under `outputs/figures/` | draw single-image keypoint overlays for inspection and reporting | Implemented |
| Feature Encoding | not implemented yet | saved or in-memory descriptors | encoded representation | convert descriptors into the next-stage representation | Not implemented |
| Retrieval / Indexing / Matching | not implemented yet | encoded features or descriptors | retrieval outputs | ranking, indexing, matching, and search logic | Not implemented |

## Important Runtime Notes

- `scripts/run_pipeline.py` currently uses directory-scan mode when it builds `ImageDatasetLoader`.
- The loader already supports split-file mode, but the main pipeline entry does not yet instantiate it through `train.txt`, `gallery.txt`, or `query.txt`.
- `scripts/build_splits.py` is the tool that currently builds the split manifests.
- Keypoint visualization is drawn on the preprocessed image, not on the original raw-resolution image. This is intentional because the extracted keypoint coordinates correspond to the preprocessed image space.

## Module Interfaces

### Dataset Sample Structure

`ImageSample` fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `sample_id` | `str` | stable sample identifier, usually a relative path-like id |
| `image_path` | `pathlib.Path` | resolved filesystem path to the image |
| `file_name` | `str` | basename of the image file |
| `split` | `str` | current split label such as `train`, `test`, or `unspecified` |
| `meta` | `dict[str, Any]` | lightweight metadata, currently including at least `relative_path` |

Loader behaviors that matter for downstream work:

- directory mode: recursively scans a directory and builds ordered samples
- split-file mode: resolves each line in a split manifest relative to a data root
- `load_image(...)`: returns an OpenCV image as `numpy.ndarray`

### PreprocessResult Structure

`PreprocessResult` fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `image` | `numpy.ndarray` | processed image passed to later stages |
| `original_shape` | `tuple[int, ...]` | shape before preprocessing |
| `processed_shape` | `tuple[int, ...]` | shape after preprocessing |
| `color_mode` | `str` | final output color mode, currently `keep` or `gray` |
| `meta` | `dict[str, Any]` | preprocessing metadata |

Current `meta` keys used by the pipeline:

- `stage`
- `status`
- `applied_steps`
- `input_shape`
- `output_shape`
- `requested_color_mode`
- `resize`

### LocalFeatureResult Structure

`LocalFeatureResult` fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `keypoints` | `numpy.ndarray` | serialized keypoints with shape `N x 7` |
| `descriptors` | `numpy.ndarray | None` | descriptor matrix or `None` when no descriptors exist |
| `meta` | `dict[str, Any]` | extraction metadata |

#### `keypoints` format

`LocalFeatureResult.keypoints` uses shape `N x 7`.
Column order is fixed as:

1. `x`
2. `y`
3. `size`
4. `angle`
5. `response`
6. `octave`
7. `class_id`

This array is serialized from OpenCV keypoints and is the only keypoint representation used by downstream code.

#### `descriptors` format

Current descriptor behavior depends on the extractor:

- `SIFT`: descriptor shape is typically `N x 128`, dtype is `float32`
- `ORB`: descriptor shape is typically `N x 32`, dtype is `uint8`
- empty-result case: `descriptors` is `None` in memory

The next stage must not assume a single descriptor dtype across methods.

#### `meta` fields

Current extraction metadata includes:

- `stage`
- `status`
- `method`
- `num_keypoints`
- `descriptor_shape`
- `descriptor_dtype`
- `input_shape`
- `working_image_shape`
- `keypoint_fields`

## Feature File Format: `outputs/features/*.npz`

Each extracted feature file is saved as a compressed `.npz` file.
The file name is derived from `sample_id` by replacing path separators and other non-safe characters with `__`.

Example output name:

- `A03Z78__A03Z78_20151127145344_6753243118.jpg.npz`

### Saved keys

| Key | Shape / Type | Meaning |
| --- | --- | --- |
| `sample_id` | scalar string array | original sample id |
| `method` | scalar string array | extractor method, currently `SIFT` or `ORB` |
| `num_keypoints` | scalar `int32` array | number of serialized keypoints |
| `keypoints` | `N x 7` float array | serialized keypoints |
| `descriptors` | descriptor matrix or empty array | saved descriptor matrix |
| `descriptors_present` | scalar `uint8` array | `1` if descriptors existed in memory, else `0` |
| `descriptor_shape` | shape array | saved descriptor shape metadata |
| `descriptor_dtype` | scalar string array | saved descriptor dtype metadata |
| `keypoint_fields` | string array | keypoint column names |

### How future modules should read `.npz`

The next stage should read feature files with `allow_pickle=False` and use `descriptors_present` to decide whether a descriptor matrix is meaningful.

Example read pattern:

```python
import numpy as np

with np.load(feature_path, allow_pickle=False) as data:
    descriptors_present = bool(int(data["descriptors_present"]))
    descriptors = data["descriptors"] if descriptors_present else None
    keypoints = data["keypoints"]
    method = str(data["method"])
```

Encoding code should treat these cases explicitly:

- `descriptors_present == 0`: skip or mark as empty
- `method == "SIFT"`: descriptors are float-valued vectors
- `method == "ORB"`: descriptors are binary-style `uint8` vectors

## What Is Implemented

- config loading from `configs/base.yaml`
- dataset loading from directory scan mode in the main pipeline
- split manifest generation through `scripts/build_splits.py`
- split-file support inside `ImageDatasetLoader`
- minimal preprocess with validation, resize, and grayscale conversion
- real local feature extraction with SIFT and ORB
- `.npz` feature saving
- single-image keypoint visualization to `.png`
- minimal Gradio demo entry point

## What Is NOT Implemented

- feature encoding
- codebook generation
- TF-IDF
- inverted index construction
- query-gallery retrieval
- feature matching workflows
- query-gallery comparison figures
- RANSAC-based visualization
- reranking, query expansion, dense retrieval, or hybrid fusion

## Next Stage: Feature Encoding

The next stage should consume descriptor matrices, not raw images and not OpenCV keypoint objects.

### Encoding input

Encoding should accept either:

- in-memory `LocalFeatureResult.descriptors`
- or descriptors loaded from `outputs/features/*.npz`

The stable handoff contract is the descriptor matrix plus method metadata.

### Where descriptors come from

Descriptors are produced in:

- `src/features/local/local_feature_extractor.py`

They are persisted by:

- `save_local_feature_result(...)`

They are saved under:

- `outputs/features/*.npz`

### Where encoding should connect

The clean integration point is after feature saving and before any retrieval or indexing logic.
In the current pipeline, that means inserting a new stage after:

- `extract_local_features(...)`
- `save_local_feature_result(...)`

and before any future retrieval-specific logic.

A stage-correct next-step shape is:

```python
feature_result = extract_local_features(preprocess_result.image, local_feature_config)
save_path = save_local_feature_result(sample.sample_id, feature_result, feature_dir)
encoding_result = encode_local_features(feature_result.descriptors, encoding_config)
```

or, for offline processing:

```python
with np.load(feature_path, allow_pickle=False) as data:
    descriptors = ...
encoding_result = encode_local_features(descriptors, encoding_config)
```

### Constraints for the next stage

The encoding stage should account for:

- empty descriptor cases
- method-dependent descriptor dtype and shape
- the fact that `run_pipeline.py` currently processes only a small sample preview for demonstration
- the fact that feature files already exist as the on-disk contract

Encoding should be added as a new stage, not by rewriting the current local feature extraction stage.
