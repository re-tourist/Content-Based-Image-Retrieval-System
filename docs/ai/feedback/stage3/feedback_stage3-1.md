# Issue 3.1 Feedback

## Changed

- `src/features/local/local_feature_extractor.py` now performs real local feature extraction instead of returning a placeholder result.
- `src/features/local/__init__.py` now exports the real extraction and saving helpers.
- `scripts/run_pipeline.py` now reads the local feature config, runs real extraction, and saves `.npz` feature files.
- `configs/base.yaml` now provides a minimal `local_feature` config section.
- `src/demo/web_demo.py` was updated for minimal compatibility so the demo bridge reports real local feature statistics.
- `docs/design/pipeline_skeleton.md` was updated to reflect the Stage 3.1 local feature and feature-save responsibilities.

## Supported Methods

The current implementation supports:

- `SIFT`
- `ORB`

Method selection is config-driven. Unsupported methods raise a clear error. The implementation does not silently fall back to a different extractor.

## LocalFeatureResult

The in-memory result structure now contains:

- `keypoints`: a serializable `Nx7` array with the fields `x, y, size, angle, response, octave, class_id`
- `descriptors`: the descriptor matrix returned by OpenCV, or `None` when no descriptors exist
- `meta`: method, number of keypoints, descriptor shape, descriptor dtype, input shape, working image shape, and keypoint field names

Empty-feature cases are handled without crashing. The result simply reports zero keypoints and `descriptors=None`.

## Saved Feature Files

Feature files are saved to:

- `outputs/features/*.npz`

Each `.npz` file contains at least:

- `sample_id`
- `method`
- `num_keypoints`
- `keypoints`
- `descriptors`
- `descriptors_present`
- `descriptor_shape`
- `descriptor_dtype`
- `keypoint_fields`

The save path uses a safe filename derived from `sample_id` so path separators do not leak into the output filename.

## Validation

Commands verified:

- `python -m py_compile src/features/local/local_feature_extractor.py src/features/local/__init__.py scripts/run_pipeline.py src/demo/web_demo.py`
- `python scripts/run_pipeline.py`

Observed extraction check on one preprocessed sample:

- `method=sift keypoints=118 descriptor_shape=(118, 128) dtype=float32`
- `method=orb keypoints=384 descriptor_shape=(384, 32) dtype=uint8`

Observed saved file check:

- `.npz` keys include `keypoints`, `descriptors`, `method`, and `num_keypoints`
- saved `keypoints` shape is `(118, 7)`
- saved `descriptors` shape is `(118, 128)` for a real SIFT run

Demo compatibility check:

- `run_demo_bridge(...)` still works with the updated local feature stage
- the demo now reports real keypoint counts and descriptor shapes while keeping the Top-K area as a placeholder

## Acceptance Alignment

This issue now satisfies the intended Stage 3.1 scope because:

- local feature extraction is real rather than placeholder
- both SIFT and ORB are supported
- extracted keypoints and descriptors are structured and serializable
- `run_pipeline.py` performs real extraction and saves feature files
- empty-feature cases are handled safely
- no matching, retrieval, encoding, or indexing logic was added
- preprocess, dataset loading, and demo behavior remain usable
