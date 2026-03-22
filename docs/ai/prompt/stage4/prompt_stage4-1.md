Please make the smallest correct implementation for Issue 4-1.
Do not anticipate future abstractions beyond this issue unless they directly improve interface clarity.

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

If any path differs slightly in the repository, find the closest exact file and proceed carefully.
Do not assume missing structure without checking the repo.

==================================================
2. CURRENT SYSTEM STATE
==================================================

The implemented pipeline is:

raw image
→ dataset loader
→ preprocess
→ local feature extraction (SIFT / ORB)
→ feature save (.npz)
→ keypoint visualization (.png)

The current feature artifacts already exist and are the required upstream input for this task.

Existing data contract that MUST remain unchanged:

LocalFeatureResult:
- keypoints: Nx7
- descriptors:
  - SIFT: float32, Nx128
  - ORB: uint8, Nx32
  - may be None (empty case)

Saved feature format in outputs/features/*.npz includes:
- sample_id
- method
- num_keypoints
- keypoints
- descriptors
- descriptors_present
- descriptor_shape
- descriptor_dtype

Critical constraints:
- Do NOT change existing pipeline behavior
- Do NOT modify LocalFeatureResult structure
- Do NOT break the .npz contract
- Must support BOTH SIFT and ORB
- Must handle empty descriptors explicitly

==================================================
3. TASK
==================================================

Implement Issue 4-1:

Encoding Foundation: Feature File Reader and Encoding Contracts

Goal:
Build the minimal infrastructure under src/encoding/ so that later stages
(codebook training and BoW encoding) can consume features through a stable,
validated, method-aware interface.

This issue is ONLY about:
- encoding-stage data structures
- feature file reading
- building a unified encoding input from either:
  1) LocalFeatureResult in memory
  2) saved .npz feature files on disk

==================================================
4. IN SCOPE
==================================================

You should implement:

1. A new package:
   - src/encoding/

2. Files:
   - src/encoding/__init__.py
   - src/encoding/types.py
   - src/encoding/io.py

3. Lightweight dataclasses or equivalent typed structures, such as:
   - FeatureFileRecord
   - EncodingInput

4. A function to load feature artifacts from disk, e.g.:
   - load_feature_npz(feature_path)

5. A function to build encoding input from in-memory local feature output, e.g.:
   - build_encoding_input_from_local_feature(feature_result, method, sample_id=None)
   or an equivalent interface that is simple and stable

6. Explicit validation for:
   - method
   - descriptors_present
   - descriptor_dtype
   - descriptor_shape
   - compatibility between descriptors and metadata

7. Explicit handling of empty descriptor cases:
   - descriptors_present == 0
   - descriptors is None or empty
   These must return a clear valid state, not crash implicitly.

8. Minimal tests or a verification script:
   - tests/test_encoding_io.py
   or an equivalent minimal, runnable validation path

Optional but recommended:
- docs/design/encoding_contract.md

==================================================
5. OUT OF SCOPE
==================================================

Do NOT implement any of the following in this issue:

- codebook training
- descriptor sampling for k-means
- BoW encoding
- TF-IDF
- indexing
- retrieval
- pipeline redesign
- config redesign
- changes to upstream feature extraction or save logic

This issue must remain small, precise, and reusable.

==================================================
6. DESIGN REQUIREMENTS
==================================================

A. Interface requirements

The encoding layer must support two input sources:

1. In-memory local features
   - from LocalFeatureResult.descriptors

2. Persisted local features
   - from outputs/features/*.npz

Both paths should end up in a unified encoding-stage input structure.

B. Method awareness

Do NOT assume a single descriptor type.

Support:
- SIFT: float32, Nx128
- ORB: uint8, Nx32

Method-specific metadata must be preserved.

C. Empty-safe behavior

You must distinguish:
- no descriptors present
- descriptors present with valid metadata

Do not silently coerce invalid cases.
Prefer explicit validation errors for inconsistent metadata,
and explicit empty-state objects for truly empty inputs.

D. Stability

This is a foundation issue.
The interfaces introduced here should be simple, minimal, and stable enough
for later reuse by:
- codebook training
- BoW encoding

E. File reading safety

When loading .npz:
- use allow_pickle=False
- validate required keys
- fail clearly if contract is violated

==================================================
7. IMPLEMENTATION GUIDANCE
==================================================

Use minimal abstraction.
Do not overengineer.

Recommended structure:

src/encoding/types.py
- dataclasses / typed containers for:
  - FeatureFileRecord
  - EncodingInput

src/encoding/io.py
- load_feature_npz(feature_path)
- build_encoding_input_from_feature_record(record)
- build_encoding_input_from_local_feature(...)
- validation helpers as needed

The exact names can differ slightly if you have a better minimal design,
but keep the module boundary clean and obvious.

Recommended design direction:

FeatureFileRecord should represent what is read from disk, including:
- sample_id
- method
- num_keypoints
- keypoints
- descriptors
- descriptors_present
- descriptor_shape
- descriptor_dtype

EncodingInput should represent the normalized encoding-facing view, including:
- sample_id
- method
- descriptors
- descriptors_present
- descriptor_shape
- descriptor_dtype
- descriptor_dim
- num_descriptors

Keep it lightweight.
Avoid prematurely adding codebook or histogram fields.

==================================================
8. VALIDATION RULES
==================================================

Implement clear validation rules.

Examples:

1. If descriptors_present == 0:
- descriptors may be None or empty
- descriptor_shape / descriptor_dtype should still be interpreted safely
- result must remain valid and empty-safe

2. If descriptors_present == 1:
- descriptors must not be None
- descriptors must be 2D
- method-specific constraints must hold:
  - SIFT => dtype float32, dim 128
  - ORB => dtype uint8, dim 32

3. If metadata says one thing but descriptors imply another:
- raise a clear ValueError or custom exception
- do not silently fix corrupted metadata

==================================================
9. TESTS / VERIFICATION
==================================================

Add at least minimal verification.

Preferred:
- tests/test_encoding_io.py

Cover at least:
1. load valid SIFT feature npz
2. load valid ORB feature npz
3. load empty-descriptor feature npz
4. build EncodingInput from LocalFeatureResult-like input
5. detect inconsistent metadata / invalid dtype / invalid shape

If repository testing conventions already exist, follow them.
If not, add a minimal, repo-consistent test file.

Do not create a huge test suite.
Keep it focused and meaningful.

==================================================
10. OUTPUT EXPECTATION
==================================================

Deliver production-quality code with:
- clear typing
- concise docstrings
- minimal but robust validation
- no unnecessary abstractions
- no regression to existing behavior

==================================================
11. FINAL RESPONSE FORMAT
==================================================

At the end, provide:

1. A short summary of what was added
2. The exact files created / modified
3. Any assumptions made
4. How the implementation satisfies the acceptance criteria
5. Anything that should be handled in Issue 4-2 instead of now

Do not overclaim.
If something could not be completed, say so clearly.