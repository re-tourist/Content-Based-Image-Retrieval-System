# Web Demo

## Goal

This document describes the current minimal Gradio demo for the repository.
The demo is intended for local coursework presentation and for validating that the implemented pipeline can be triggered from a lightweight web UI.

## Why Gradio

Gradio is still the correct choice for the current stage:

- Python-native integration with the existing pipeline modules
- built-in image upload and preview support
- simple gallery output for placeholder Top-K results
- low setup cost for a coursework demo

The project does not need a separate frontend/backend stack at this stage.

## Current Demo Components

The current demo contains:

1. title and short project description
2. query image upload component
3. built-in uploaded image preview
4. `Run Pipeline Skeleton` button
5. pipeline status area
6. `Top-K Results` gallery area

## Relationship to the Current Pipeline

The web demo does not import `scripts/run_pipeline.py` as an application module.
Instead, it reuses the same underlying module boundaries:

- config loading via `src/utils/config.py`
- preprocess via `src/preprocess/basic_preprocess.py`
- local feature extraction via `src/features/local/local_feature_extractor.py`

This keeps the CLI path and the demo aligned to the same preprocessing and feature extraction contracts.

## Demo Runtime Flow

Current flow:

1. user uploads a query image
2. user clicks `Run Pipeline Skeleton`
3. demo loads config
4. demo reads the uploaded query image
5. demo runs preprocess
6. demo runs real local feature extraction
7. demo returns status text including preprocess and keypoint statistics
8. demo returns placeholder Top-K gallery cards

## What Is Real vs Placeholder

Already connected:

- query image upload
- query image preview
- config loading
- query image loading
- preprocess stage
- real local feature extraction
- keypoint count and descriptor shape reporting

Placeholder by design:

- real retrieval over the gallery
- real Top-K ranking output
- keypoint visualization display inside the web UI
- local feature visualization display inside the web UI
- retrieval result explanation

## Output and Persistence Behavior

The current demo is an inspection bridge, not a batch pipeline runner.
It does not currently:

- save `.npz` feature files
- save keypoint figures
- query a gallery or ranking module

Persistent outputs remain the responsibility of `scripts/run_pipeline.py`.

## Future Extension Hooks

The current demo reserves extension points for:

- real retrieval output integration
- keypoint visualization integration
- local feature visualization integration
- future ranked Top-K results

These are intentionally not implemented yet.

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the demo:

```bash
python scripts/run_demo.py
```

Then open the local Gradio URL shown in the terminal.

## Notes

This demo is not a final web product.
It is a minimal interactive shell that demonstrates the current preprocess and local feature stages while keeping retrieval output as an honest placeholder.
