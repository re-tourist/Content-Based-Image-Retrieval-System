# Issue 2.2 Feedback

## Changed

- `src/preprocess/basic_preprocess.py` now provides a structured `PreprocessResult` and implements input validation, resize, and `color_mode=keep|gray`.
- `src/preprocess/__init__.py` re-exports the updated preprocess interface.
- `configs/base.yaml` now uses the minimal preprocess config shape:
  - `resize.enabled`
  - `resize.width`
  - `resize.height`
  - `color_mode`
- `scripts/run_pipeline.py` now prints original shape, processed shape, color mode, and resize status for the preprocess stage.
- `src/demo/web_demo.py` was updated for minimal compatibility so the demo bridge reports preprocess shapes and steps.
- `docs/design/pipeline_skeleton.md` was updated to reflect the preprocess stage responsibilities and supported scope.

## Core Design

- The preprocess output is fixed to a structured result:
  - `image`
  - `original_shape`
  - `processed_shape`
  - `color_mode`
  - `meta`
- The preprocess module currently supports only the minimal input-standardization steps:
  - resize
  - grayscale conversion
- Basic validation now checks:
  - the input is not `None`
  - the input is a `numpy.ndarray`
  - the image is not empty
  - the shape is 2D or 3D
  - the channel count is acceptable
- The implementation does not add augmentation, tensor normalization, denoising, CLAHE, caching, or any Stage 3 logic.
- The code keeps backward compatibility with the earlier config style through fallback parsing of:
  - `image_size`
  - `to_grayscale`

## Validation

Commands verified:

- `python -m py_compile src/preprocess/basic_preprocess.py src/preprocess/__init__.py scripts/run_pipeline.py src/demo/web_demo.py`
- `python scripts/run_pipeline.py`

Observed preprocess result on a real sample:

- `original_shape=(450, 450, 3)`
- `processed_shape=(256, 256)`
- `color_mode=gray`
- `steps=['resize', 'color_mode:gray']`

Observed pipeline output pattern:

```text
Loaded image with shape=(450, 450, 3)
Preprocess stage completed for ... original_shape=(450, 450, 3) processed_shape=(256, 256) color_mode=gray resize=enabled -> (256, 256) steps=['resize', 'color_mode:gray']
```

Demo compatibility check:

- `gradio` imports successfully in the current environment
- `run_demo_bridge(...)` imports successfully
- calling `run_demo_bridge(...)` on a local sample image returns preprocess status and 4 placeholder gallery items

## Acceptance Alignment

This issue now satisfies the intended scope because:

- a minimal preprocess module exists
- resize is supported
- basic color conversion is supported
- input format and shape checks are enforced
- `run_pipeline.py` integrates preprocess without breaking later placeholder stages
- the output structure is clear for the future local feature module
- Stage 1 and Stage 2.1 behavior remain usable
