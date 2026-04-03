# Project Context

## Purpose of This File

This is the canonical handoff document for engineers or coding agents joining the repository.
It records the actual implemented system state, the stable data contracts, and the next safe integration point.

Use this file together with the following code-aligned documents:

- `docs/design/pipeline_skeleton.md`: current pipeline stages, interfaces, and `.npz` contract
- `docs/design/dataset_structure.md`: dataset layout and split manifest rules
- `docs/design/indexing_usage.md`: sparse retrieval contract, CLI usage, and output layout
- `docs/design/web_demo.md`: current Gradio demo scope and limitations

## Project Identity

This repository is a coursework-driven and engineering-driven Hybrid Image Retrieval System.

Long-term project direction:

- traditional retrieval line: local features -> encoding -> TF-IDF -> inverted index
- later enhancement line: dense global retrieval and hybrid fusion

Current development priority:

- keep the implemented pipeline runnable
- keep interfaces and output contracts stable
- prepare the next stage without overbuilding future modules

## Repository State

Verified completed work at the time of writing:

- Milestone 1: System Skeleton
- Milestone 2: Data Preparation
- Milestone 3: Local Feature Extraction and Keypoint Visualization
- Milestone 4: Feature Encoding and Codebook Wiring
- Milestone 5: TF-IDF and Inverted Index

The main runnable engineering path is no longer only a skeleton. It now performs:

- dataset loading
- preprocess
- real local feature extraction
- feature saving
- keypoint figure generation

The repository also now contains the offline sparse retrieval stages:

- batch BoW encoding from saved local features
- TF-IDF statistics
- method-specific inverted indexes
- query-time top-k search over encoded artifacts

## Main Entry Points

### `scripts/run_pipeline.py`

Primary engineering entry for the current pipeline. It currently:

1. loads `configs/base.yaml`
2. resolves a dataset image directory
3. previews a small number of samples
4. preprocesses each image
5. extracts local features
6. saves `.npz` feature files
7. saves `.png` keypoint figures

Important runtime note:

- the script currently uses directory-scan dataset loading, not split-file loading, even though split support exists in the loader
- the script remains preview-oriented; batch BoW encoding and sparse retrieval are handled by dedicated scripts

### `scripts/encode_features.py`

Batch-encodes saved local-feature artifacts into BoW histograms under `outputs/encoded/`.
This is the offline entrypoint for Milestone 4 and is the stable handoff into indexing.

### `scripts/build_inverted_index.py`

Builds TF-IDF statistics and method-specific inverted indexes from an explicit encoded directory.
The script consumes `outputs/encoded/*.npz` raw-count BoW artifacts and writes `outputs/indices/inverted/<corpus_split>/<method>/...`.

### `scripts/search_images.py`

Loads encoded query artifacts and searches them against a method-specific inverted index.
It supports the Milestone 5 scoring paths:

- `tf + dot`
- `tfidf + cosine`

### `scripts/build_splits.py`

Builds `train.txt`, `gallery.txt`, and `query.txt` under `data/splits/`.
This is the current standardization tool for dataset manifests.

### `scripts/run_demo.py`

Launches the minimal Gradio demo.
The demo reuses the current preprocess and local feature extraction logic, reports keypoint statistics, and keeps the Top-K result area as a placeholder.

## Current Implemented Data Flow

Actual per-sample pipeline flow:

`ImageSample -> load_image -> PreprocessResult -> LocalFeatureResult -> .npz feature file -> keypoint figure`

## Core Module Contracts

### Dataset stage

Primary type:

- `ImageSample(sample_id, image_path, file_name, split, meta)`

What matters downstream:

- the loader supports both directory mode and split-file mode
- the main pipeline currently demonstrates directory mode
- `load_image(...)` returns an OpenCV image as `numpy.ndarray`

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

### Split manifests

Location:

- `data/splits/train.txt`
- `data/splits/gallery.txt`
- `data/splits/query.txt`

Current format:

- one `data/`-relative image path per line
- blank lines and `#` comment lines are ignored

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

### Encoded BoW files

Location:

- `outputs/encoded/*.npz`

Stable keys:

- `sample_id`
- `method`
- `encoding_type`
- `num_visual_words`
- `histogram`
- `histogram_dtype`
- `num_descriptors`
- `descriptors_present`
- `codebook_path`
- `normalized`

Important indexing constraint:

- Milestone 5 indexing only accepts raw-count encoded artifacts with `normalized == False`
- normalized BoW artifacts are valid encoding outputs, but they are rejected by the indexing stage

### Indexing artifacts

Location:

- `outputs/indices/inverted/<corpus_split>/<method>/tfidf_stats.npz`
- `outputs/indices/inverted/<corpus_split>/<method>/index_tf.npz`
- `outputs/indices/inverted/<corpus_split>/<method>/index_tfidf.npz`

The `corpus_split` segment is a naming and logging label only.
The actual corpus source is the explicit encoded directory passed to the build script.

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
- `local_feature.orb_nfeatures`
- `visualization.enabled`
- `visualization.save_keypoints`

## What Is Implemented

- dataset enumeration and image loading
- split manifest generation and loader compatibility
- preprocess result standardization
- real SIFT / ORB extraction
- `.npz` feature persistence
- BoW encoding from saved local features
- TF-IDF statistics for encoded BoW corpora
- method-specific inverted index construction
- top-k sparse search over encoded artifacts
- `.png` keypoint visualization
- minimal demo reuse of preprocess and local feature extraction

## What Is Not Implemented

- codebook training automation beyond the current scripts
- retrieval evaluation metrics such as mAP / PR curves
- reranking
- dense retrieval
- hybrid fusion
- query expansion
- web demo overhaul

## Next Stage: Retrieval Evaluation

The next safe stage is not feature encoding or indexing anymore.
Those are now implemented and should be treated as stable handoff points.

The next intended work is:

- retrieval evaluation on top of the new sparse search stage
- gallery/query experiment wiring
- metrics such as PR curves and mAP

That work should consume `outputs/indices/inverted/<corpus_split>/<method>/...` and should not change the raw-count BoW contract.

## Recommended Document Set for Next-Stage Planning

If a new engineer or ChatGPT is asked to plan Feature Encoding, the minimum reliable input set is:

- `docs/ai/PROJECT_CONTEXT.md`
- `docs/design/pipeline_skeleton.md`
- `configs/base.yaml`
- `scripts/run_pipeline.py`
- `src/features/local/local_feature_extractor.py`
- `docs/design/dataset_structure.md`

Optionally include one real `.npz` sample summary to make descriptor shapes and dtypes explicit.
