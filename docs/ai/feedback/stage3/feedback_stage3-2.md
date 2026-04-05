# Issue 3.2 Feedback

## Changed

- `src/visualization/keypoints.py` now implements real single-image keypoint visualization based on serialized keypoints.
- `src/visualization/__init__.py` exports the visualization helpers.
- `scripts/run_pipeline.py` now calls the visualization stage after local feature extraction and feature saving.
- `configs/base.yaml` now provides a minimal `visualization` config section.
- `docs/design/pipeline_skeleton.md` now reflects that keypoint visualization is connected and saves `.png` files.

## Visualization Design

The visualization stage now works directly from:

- the image already available in the current pipeline
- `LocalFeatureResult.keypoints`

The implementation draws keypoints from the serialized `Nx7` array and saves a PNG figure under `outputs/figures/`.
It does not rebuild a parallel feature-reading flow and does not depend on OpenCV `KeyPoint` objects.

Key behavior:

- grayscale inputs are converted to a 3-channel canvas for saving
- each keypoint is rendered as a small circle plus center point
- a small text label is written onto the figure with the keypoint count or provided label
- empty keypoint arrays still produce a valid saved figure

## Output Naming

Saved figures follow the stable format:

- `outputs/figures/keypoints_<safe_sample_name>.png`

The filename is derived from `sample_id` through a minimal safe-name conversion so path separators do not leak into the filename.

## Validation

Commands verified:

- `python -m py_compile src/visualization/keypoints.py src/visualization/__init__.py scripts/run_pipeline.py`
- `python scripts/run_pipeline.py`

Observed pipeline output pattern:

- `Keypoint visualization enabled`
- `Visualized 118 keypoints for sample ...`
- `Saved keypoint figure to outputs/figures/keypoints_...png`

Observed figure check:

- a real saved figure exists under `outputs/figures/`
- loaded figure shape is `(256, 256, 3)` for a pipeline-generated sample

Observed empty-keypoint check:

- saving with an empty `(0, 7)` keypoint array succeeds
- a valid `keypoints_empty__case.png` file is created instead of crashing

Demo compatibility check:

- `run_demo_bridge(...)` still works after this change
- the demo remains minimal and no UI expansion was added

## Acceptance Alignment

This issue now satisfies the intended Stage 3.2 scope because:

- keypoint visualization is implemented as a real module
- it works directly from the current local feature output structure
- figures are saved to `outputs/figures/keypoints_*.png`
- `run_pipeline.py` calls the visualization stage successfully
- empty-keypoint cases do not crash
- no matching, retrieval, RANSAC, or other out-of-scope logic was added
