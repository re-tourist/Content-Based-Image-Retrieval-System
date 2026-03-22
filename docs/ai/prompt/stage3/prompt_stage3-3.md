You are working on a research-engineering repository for a Hybrid Image Retrieval System.

The project has completed:

- Milestone 1: System Skeleton
- Milestone 2: Data Preparation
- Milestone 3: Local Feature Extraction

Your task is NOT to implement new features.

Your task is to consolidate the current system state into clean, accurate, and usable documentation, and prepare for the next stage: Feature Encoding.

==================================================
1. GOAL
==================================================

Produce a clear, up-to-date, and code-aligned documentation of:

1) The actual pipeline (as implemented, not as originally planned)
2) Module interfaces and data flow
3) Feature output format (.npz)
4) Integration point for the next stage (Feature Encoding)

The output should allow a new engineer (or Codex) to continue development without reading old conversations.

==================================================
2. REQUIRED OUTPUT FILES
==================================================

You must create or update:

1) docs/design/pipeline_skeleton.md
2) docs/ai/PROJECT_CONTEXT.md

Do NOT create many new files.
Focus on these two.

==================================================
3. PIPELINE DOCUMENTATION REQUIREMENTS
==================================================

In pipeline_skeleton.md, clearly describe:

- End-to-end flow:

  raw image
  → dataset loader
  → preprocess
  → local feature extraction
  → feature save
  → keypoint visualization

- For each stage:

  INPUT
  OUTPUT
  RESPONSIBILITY

- Which stages are fully implemented vs placeholder

==================================================
4. MODULE INTERFACES
==================================================

You must explicitly document:

- Dataset sample structure
- PreprocessResult structure
- LocalFeatureResult structure

For LocalFeatureResult, include:

- keypoints shape and meaning (Nx7)
- descriptors shape and dtype
- meta fields

==================================================
5. FEATURE FILE FORMAT (.npz)
==================================================

Document:

- What is saved in outputs/features/*.npz
- Key names
- Shapes
- How future modules should read it

This is CRITICAL for the next stage.

==================================================
6. CURRENT SYSTEM CAPABILITIES
==================================================

Add a section:

## What is implemented

## What is NOT implemented

Be explicit. Do not guess.

==================================================
7. NEXT STAGE PREPARATION (VERY IMPORTANT)
==================================================

Add a section:

## Next Stage: Feature Encoding

Explain:

- What input encoding will consume (descriptors)
- Where descriptors come from
- How encoding should connect to current pipeline
- Where new modules should be inserted

DO NOT implement encoding.

Only define interfaces and integration points.

==================================================
8. STYLE REQUIREMENTS
==================================================

- Markdown
- Clear structure
- No fluff
- No speculative design beyond current stage
- Reflect actual code, not assumptions

==================================================
9. OUTPUT FORMAT
==================================================

At the end, also provide:

- Summary of files updated
- Key design decisions
- Any assumptions made

==================================================
IMPORTANT
==================================================

- Do NOT modify core pipeline logic
- Do NOT implement feature encoding
- Do NOT refactor the whole repo

This is a documentation and system consolidation task.