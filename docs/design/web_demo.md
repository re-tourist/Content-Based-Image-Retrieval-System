# Web Demo

## Goal

This document describes the minimal Stage 1 web demo for the project.  
The demo is intended for local coursework presentation and for validating that the current pipeline skeleton can be triggered from a lightweight web UI.

## Why Gradio

Gradio is used because it matches the current project stage:

- Python-native integration with the existing pipeline modules
- Built-in image upload and preview support
- Simple gallery output for placeholder Top-K results
- Low setup cost for a coursework demo

At Stage 1, a lightweight demo shell is more appropriate than a full frontend/backend stack.

## Current Demo Components

The current demo contains:

1. Title and short project description
2. Query image upload component
3. Built-in uploaded image preview
4. `Run Pipeline Skeleton` button
5. Pipeline status area
6. `Top-K Results` gallery area

## Relationship to the Current Pipeline Skeleton

The web demo does not import `scripts/run_pipeline.py` as an application module.  
Instead, it reuses the same underlying module boundaries:

- config loading via `src/utils/config.py`
- preprocess stage via `src/preprocess/basic_preprocess.py`
- local feature placeholder stage via `src/features/local/local_feature_extractor.py`

This keeps the CLI skeleton and the web demo aligned to the same stage boundaries.

## Demo Runtime Flow

Current flow:

1. User uploads a query image
2. User clicks `Run Pipeline Skeleton`
3. Demo loads config
4. Demo reads the uploaded query image
5. Demo runs the basic preprocess stage
6. Demo runs the local feature placeholder stage
7. Demo returns status text
8. Demo returns placeholder Top-K gallery cards

## What Is Real vs Placeholder

Already connected:

- Query image upload
- Query image preview
- Config loading
- Query image loading
- Basic preprocess stage
- Local feature interface invocation

Placeholder by design:

- Real retrieval over the gallery
- Real Top-K ranking output
- Keypoint visualization
- Local feature visualization
- Retrieval result explanation

## Future Extension Hooks

The current demo reserves extension points for:

- real retrieval output integration
- keypoint visualization integration
- local feature visualization integration
- future ranked Top-K results

These are intentionally not implemented in Issue 1.4.

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
It is a minimal interactive shell for Milestone 1 so that the current pipeline skeleton can be demonstrated in a browser before later retrieval stages are implemented.
