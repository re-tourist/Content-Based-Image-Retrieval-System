You are assisting an existing research-engineering repository for a Hybrid Image Retrieval System.

This is NOT a fresh project.
Do NOT redesign the system.
You must continue from the current implemented state.

==================================================
0. BRANCH CONTEXT
==================================================

Suggested branch: feat/encoding

==================================================
1. READ FIRST
==================================================

Before making any changes, read these files first and use them as the source of truth:

1. docs/ai/PROJECT_CONTEXT.md
2. docs/design/pipeline_skeleton.md
3. docs/design/dataset_structure.md
4. configs/base.yaml
5. scripts/run_pipeline.py
6. src/features/local/local_feature_extractor.py
7. src/encoding/types.py
8. src/encoding/io.py
9. tests/test_encoding_io.py

If any exact path differs slightly in the repository, find the closest exact file and proceed carefully.
Do not assume structure without checking the repo.

==================================================
2. CURRENT SYSTEM STATE
==================================================

The current implemented pipeline is:

raw image
→ dataset loader
→ preprocess
→ local feature extraction (SIFT / ORB)
→ feature save (.npz)
→ keypoint visualization (.png)

Feature Encoding stage has begun.

Issue 4-1 is assumed completed:
- src/encoding/ exists
- encoding input contracts exist
- .npz feature loading exists
- in-memory LocalFeatureResult -> EncodingInput conversion exists

Existing contracts that MUST remain unchanged:
- LocalFeatureResult structure
- outputs/features/*.npz contract
- support for BOTH SIFT and ORB
- explicit handling of empty descriptors

Existing upstream feature file keys include:
- sample_id
- method
- num_keypoints
- keypoints
- descriptors
- descriptors_present
- descriptor_shape
- descriptor_dtype
- keypoint_fields

SIFT descriptors:
- float32
- Nx128

ORB descriptors:
- uint8
- Nx32

==================================================
3. TASK
==================================================

Implement Issue 4-2:

Descriptor Sampling and Reusable Codebook Training

Goal:
Build the next minimal encoding-stage layer that:
1. reads existing local feature artifacts
2. samples descriptors safely
3. trains reusable method-specific codebooks
4. saves codebook artifacts to disk for later BoW encoding

This issue is ONLY about:
- descriptor collection / sampling
- codebook training
- codebook persistence
- minimal config extension
- a standalone training script

==================================================
4. IN SCOPE
==================================================

You should implement:

1. New encoding modules:
   - src/encoding/sampling.py
   - src/encoding/codebook.py

2. A standalone training script:
   - scripts/train_codebook.py

3. Minimal config extension in configs/base.yaml for encoding/codebook settings

4. Method-aware descriptor grouping and codebook training:
   - SIFT codebook trained only from SIFT descriptors
   - ORB codebook trained only from ORB descriptors

5. Explicit handling of empty descriptors:
   - skip safely
   - do not crash
   - do not contaminate sampling

6. Reusable on-disk codebook artifacts with metadata

==================================================
5. OUT OF SCOPE
==================================================

Do NOT implement any of the following in this issue:

- BoW encoding
- visual word histogram generation
- TF-IDF
- inverted index
- retrieval
- pipeline integration into run_pipeline.py
- batch encoding of features into outputs/encoded
- redesign of Issue 4-1 interfaces
- changes to LocalFeatureResult
- changes to outputs/features/*.npz contract

Keep this issue focused and independently executable.

==================================================
6. DESIGN REQUIREMENTS
==================================================

A. Reuse Issue 4-1 interfaces

Use the encoding foundation already introduced in src/encoding/types.py and src/encoding/io.py.
Do not bypass them with ad hoc descriptor loading logic unless absolutely necessary.

B. Method-specific codebooks

Do NOT mix SIFT and ORB descriptors into one codebook.

At minimum:
- train one codebook per method
- preserve method metadata in the saved artifact

C. Sampling behavior

Implement descriptor sampling from existing feature files.

Requirements:
- iterate over saved feature files
- ignore empty descriptor records safely
- validate descriptor compatibility within each method
- allow configurable cap on total sampled descriptors
- avoid loading or concatenating unbounded data blindly

Use a minimal, robust approach.
A simple first-pass solution is preferred over overengineering.

D. Training algorithm

Use a classical codebook trainer:
- KMeans or MiniBatchKMeans

Prefer MiniBatchKMeans if it simplifies scalability and keeps dependencies reasonable.

Training must be config-driven.

E. Reusable persistence

Saved codebook artifacts must be reusable by later issues.

At minimum store:
- method
- n_clusters
- descriptor_dim
- training_descriptor_count
- random_state
- cluster centers

Use a simple, explicit artifact format.
A compressed .npz artifact is preferred unless the repo already has a strong serialization convention.

F. Minimal config extension

Extend configs/base.yaml with a new encoding section, but do not disturb current active pipeline behavior.

Recommended minimal shape:

encoding:
  enabled: false
  input:
    feature_dir: outputs/features
  codebook:
    enabled: false
    output_dir: outputs/indices/codebooks
    trainer: minibatch_kmeans
    n_clusters: 256
    max_descriptors: 50000
    batch_size: 1024
    random_state: 42

You may adjust field names slightly if needed, but keep the structure minimal and obvious.

==================================================
7. IMPLEMENTATION GUIDANCE
==================================================

Use minimal abstraction.
Do not anticipate BoW or retrieval logic yet.

Recommended responsibilities:

src/encoding/sampling.py
- iterate feature files
- load FeatureFileRecord / EncodingInput through Issue 4-1 interfaces
- collect descriptors by method
- perform capped sampling
- expose clear sampling outputs for training

src/encoding/codebook.py
- fit method-specific codebook
- validate descriptor matrix shape/dtype
- save codebook artifact
- load codebook artifact
- define a lightweight codebook metadata/container if useful

scripts/train_codebook.py
- load config
- locate feature_dir
- call sampling
- train codebook(s)
- save outputs
- print concise summary logs

Keep module boundaries clean and obvious.

==================================================
8. VALIDATION RULES
==================================================

Implement clear validation rules.

Examples:

1. Empty input handling
- if a feature record has descriptors_present == 0, skip it safely
- if all records for a method are empty, fail clearly with a useful error message

2. Descriptor consistency
- all descriptors within a training run for one method must have the same dimensionality
- SIFT expected dim = 128
- ORB expected dim = 32

3. Method separation
- never concatenate SIFT and ORB descriptors together
- artifacts must remain method-specific

4. Metadata integrity
- saved codebook metadata must agree with the trained centers

Do not silently fix corrupted or inconsistent data.

==================================================
9. TESTS / VERIFICATION
==================================================

Add at least minimal verification.

Preferred:
- tests/test_codebook.py
or another small, repo-consistent test file

Cover at least:
1. sampling skips empty descriptor records
2. sampling groups by method correctly
3. codebook training works on small synthetic SIFT-like descriptors
4. codebook training works on small synthetic ORB-like descriptors
5. saved codebook artifact can be loaded back
6. inconsistent dimensionality raises a clear error

Do not create an oversized test suite.
Keep it focused and meaningful.

==================================================
10. OUTPUT EXPECTATION
==================================================

Deliver production-quality code with:
- clear typing
- concise docstrings
- minimal but robust validation
- reusable codebook artifact format
- no regression to existing behavior
- no BoW / TF-IDF / retrieval logic

==================================================
11. FINAL RESPONSE FORMAT
==================================================

At the end, provide:

1. A short summary of what was added
2. The exact files created / modified
3. The codebook artifact format you chose
4. Any assumptions made
5. How the implementation satisfies the acceptance criteria
6. What is intentionally left for Issue 4-3

Do not overclaim.
If something could not be completed, say so clearly.