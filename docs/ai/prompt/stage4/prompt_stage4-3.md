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
5. src/encoding/types.py
6. src/encoding/io.py
7. src/encoding/sampling.py
8. src/encoding/codebook.py
9. scripts/train_codebook.py
10. tests/test_encoding_io.py
11. tests/test_codebook.py

If any exact path differs slightly in the repository, find the closest exact file and proceed carefully.
Do not assume missing structure without checking the repo.

==================================================
2. CURRENT SYSTEM STATE
==================================================

The implemented pipeline is currently:

raw image
→ dataset loader
→ preprocess
→ local feature extraction (SIFT / ORB)
→ feature save (.npz)
→ keypoint visualization (.png)

Stage 4.1 and Stage 4.2 are assumed completed:

- src/encoding/types.py and src/encoding/io.py exist
- feature `.npz` loading and EncodingInput conversion exist
- src/encoding/sampling.py exists
- src/encoding/codebook.py exists
- method-specific codebook training and save/load exist
- scripts/train_codebook.py exists

Existing contracts that MUST remain unchanged:
- LocalFeatureResult structure
- outputs/features/*.npz contract
- support for BOTH SIFT and ORB
- explicit handling of empty descriptors

Upstream feature file keys include:
- sample_id
- method
- num_keypoints
- keypoints
- descriptors
- descriptors_present
- descriptor_shape
- descriptor_dtype
- keypoint_fields

Current codebook artifact is reusable and method-specific.
Do not redesign it unless a very small compatibility-safe change is absolutely necessary.

==================================================
3. TASK
==================================================

Implement Issue 4-3:

BoW Encoding and Encoded Feature Persistence

Goal:
Given a feature artifact or encoding input plus the matching method-specific codebook,
map local descriptors to visual words and produce an image-level BoW histogram.
Then save the encoded feature artifact in a stable format for later TF-IDF and indexing stages.

This issue is ONLY about:
- loading codebooks
- descriptor quantization / visual word assignment
- image-level BoW histogram construction
- encoded feature persistence
- minimal batch offline encoding support

==================================================
4. IN SCOPE
==================================================

You should implement:

1. New modules:
   - src/encoding/bow.py
   - src/encoding/storage.py

2. A lightweight result structure for encoded outputs, e.g.:
   - EncodedFeatureResult

3. BoW encoding logic:
   - input: EncodingInput or feature `.npz` derived input
   - input: matching method-specific codebook
   - output: image-level histogram over visual words

4. Explicit empty-descriptor behavior:
   - if descriptors are absent, return a valid empty-safe BoW result
   - histogram must still have the correct vocabulary size

5. Encoded artifact persistence:
   - save encoded outputs to outputs/encoded/*.npz
   - add load helpers for later stages

6. A minimal batch offline encoding function or script-level callable path
   - enough to encode saved feature artifacts in batch
   - do NOT integrate into run_pipeline.py yet

7. Minimal tests or verification coverage

Optional but recommended:
- docs/design/encoded_feature_contract.md

==================================================
5. OUT OF SCOPE
==================================================

Do NOT implement any of the following in this issue:

- DF statistics
- IDF computation
- TF-IDF vectors
- inverted index
- retrieval / ranking
- reranking
- run_pipeline.py integration
- pipeline redesign
- deep features / dense retrieval

This issue must remain focused on BoW encoding and encoded artifact storage only.

==================================================
6. DESIGN REQUIREMENTS
==================================================

A. Reuse prior interfaces

Use the existing Issue 4.1 and 4.2 interfaces:
- feature loading through src/encoding/io.py
- codebook loading through src/encoding/codebook.py

Do not add parallel ad hoc loaders unless absolutely necessary.

B. Method-aware encoding

Do NOT assume a single descriptor type.

Support:
- SIFT descriptors with SIFT codebook
- ORB descriptors with ORB codebook

Never allow mismatched method/codebook pairing silently.
Fail clearly if a SIFT feature is paired with an ORB codebook or vice versa.

C. BoW output semantics

This stage is still feature encoding, not TF-IDF.

So the saved representation should be:
- raw term counts
or
- optionally normalized histogram

But it must NOT include DF / IDF weighting.

Keep the design simple and explicit.

D. Empty-safe behavior

If descriptors_present == 0 or descriptors are empty:
- return a valid histogram of zeros
- preserve metadata
- do not crash
- do not skip the sample silently

E. Stable encoded artifact format

The encoded artifact must be suitable for later TF-IDF and indexing stages.

At minimum persist:
- sample_id
- method
- encoding_type ("bow")
- num_visual_words
- histogram
- histogram_dtype
- num_descriptors
- descriptors_present
- codebook_path or codebook_id

You may include a few additional minimal metadata fields if they improve stability,
but do not overdesign the format.

F. Simple module boundaries

Recommended responsibilities:

src/encoding/bow.py
- visual word assignment
- histogram generation
- empty-safe encoding
- validation of codebook / descriptor compatibility

src/encoding/storage.py
- EncodedFeatureResult
- save_encoded_feature(...)
- load_encoded_feature(...)

Keep boundaries obvious and minimal.

==================================================
7. IMPLEMENTATION GUIDANCE
==================================================

Use a minimal classical BoW implementation.

Recommended approach:
1. load EncodingInput
2. load matching codebook artifact
3. compute nearest codeword index for each descriptor
4. build histogram of size n_clusters
5. store result and metadata

Acceptable implementation choices:
- nearest-cluster assignment based on cluster centers
- numpy/scikit-learn utilities if already appropriate

Prefer the smallest correct solution.

Do not introduce VLAD, Fisher Vector, soft assignment, or ANN search.

If you support optional histogram normalization, make it config-free or very lightly parameterized,
and keep raw count semantics clearly available.

==================================================
8. VALIDATION RULES
==================================================

Implement clear validation rules.

Examples:

1. Method compatibility
- feature method must match codebook method
- otherwise raise a clear error

2. Descriptor dimensionality
- descriptor dim must match codebook descriptor_dim
- otherwise raise a clear error

3. Empty input
- empty descriptors must yield a zero histogram of length n_clusters
- metadata must remain valid

4. Histogram integrity
- histogram length must equal n_clusters
- histogram dtype must be explicit and stable

5. Persistence integrity
- a saved encoded artifact must be loadable back with matching metadata

Do not silently repair corrupted data.

==================================================
9. TESTS / VERIFICATION
==================================================

Add at least minimal verification.

Preferred:
- tests/test_bow.py

Cover at least:
1. BoW encoding works for small synthetic SIFT-like descriptors with a matching codebook
2. BoW encoding works for small synthetic ORB-like descriptors with a matching codebook
3. empty descriptors produce a zero histogram with correct length
4. mismatched method/codebook raises a clear error
5. mismatched descriptor dimensionality raises a clear error
6. encoded artifact save/load roundtrip works

If repository conventions support a small helper for batch encoding verification, include it.
Do not create a large suite. Keep it focused.

==================================================
10. OUTPUT EXPECTATION
==================================================

Deliver production-quality code with:
- clear typing
- concise docstrings
- minimal but robust validation
- stable encoded artifact format
- no TF-IDF / indexing / retrieval logic
- no regression to existing behavior

==================================================
11. FINAL RESPONSE FORMAT
==================================================

At the end, provide:

1. A short summary of what was added
2. The exact files created / modified
3. The encoded artifact format you chose
4. Any assumptions made
5. How the implementation satisfies the acceptance criteria
6. What is intentionally left for Issue 4-4 and later stages

Do not overclaim.
If something could not be completed, say so clearly.

Please keep the encoded artifact format minimal and stable.
Do not anticipate TF-IDF fields yet.