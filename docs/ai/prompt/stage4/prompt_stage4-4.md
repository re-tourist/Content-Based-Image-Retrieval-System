You are assisting an existing research-engineering repository for a Hybrid Image Retrieval System.

This is NOT a fresh project.
Do NOT redesign the system.
You must continue from the current implemented state.

==================================================
0. BRANCH CONTEXT
==================================================

Suggested branch: feat/scripts

==================================================
1. READ FIRST
==================================================

Before making any changes, read these files first and use them as the source of truth:

1. docs/ai/PROJECT_CONTEXT.md
2. docs/design/pipeline_skeleton.md
3. docs/design/dataset_structure.md
4. configs/base.yaml
5. scripts/run_pipeline.py
6. scripts/train_codebook.py
7. scripts/encode_features.py
8. src/encoding/types.py
9. src/encoding/io.py
10. src/encoding/codebook.py
11. src/encoding/bow.py
12. src/encoding/storage.py
13. tests/test_encoding_io.py
14. tests/test_codebook.py
15. tests/test_bow.py

If any exact path differs slightly in the repository, find the closest exact file and proceed carefully.
Do not assume missing structure without checking the repo.

==================================================
2. CURRENT SYSTEM STATE
==================================================

The implemented system currently supports:

raw image
→ dataset loader
→ preprocess
→ local feature extraction (SIFT / ORB)
→ feature save (.npz)
→ keypoint visualization (.png)

Stage 4.1 completed:
- encoding input contracts exist
- feature `.npz` loading exists

Stage 4.2 completed:
- method-specific descriptor sampling and codebook training exist
- reusable codebook artifacts exist
- scripts/train_codebook.py exists

Stage 4.3 completed:
- method-aware BoW encoding exists
- encoded feature persistence exists
- scripts/encode_features.py exists as a standalone offline batch entry
- outputs/encoded/*.npz is the encoded artifact target

Existing contracts that MUST remain unchanged:
- LocalFeatureResult structure
- outputs/features/*.npz contract
- codebook artifact compatibility from Stage 4.2
- encoded artifact compatibility from Stage 4.3
- support for BOTH SIFT and ORB
- explicit handling of empty descriptors

==================================================
3. TASK
==================================================

Implement Issue 4-4:

Encoding Stage Integration and Offline Batch Encoding Entry

Goal:
Integrate the encoding stage into the current system in the least invasive way:
1. add a config-driven optional encoding hook to scripts/run_pipeline.py
2. keep the current preview-oriented behavior unchanged by default
3. provide a stable offline batch entry for full encoded-feature generation
4. prepare a clean handoff for the later TF-IDF / indexing stage

This issue is about integration and entrypoints, not new encoding algorithms.

==================================================
4. IN SCOPE
==================================================

You should implement:

1. Extend and regularize the `encoding` section in configs/base.yaml
   - keep it minimal
   - keep backward compatibility with what Stage 4.3 already introduced
   - avoid creating a second parallel config scheme

2. Add a minimal optional encoding hook in scripts/run_pipeline.py
   - default behavior must remain unchanged
   - only if `encoding.enabled: true`, attempt encoding after feature save
   - the hook should use the existing Stage 4.3 interfaces, not duplicate logic

3. Ensure scripts/encode_features.py is a proper standalone offline batch entry
   - iterate over outputs/features/*.npz
   - load the matching method-specific codebook
   - generate encoded outputs under the configured directory
   - provide clear logs and failure messages

4. Add minimal usage documentation or run instructions
   - enough for a developer to run:
     - codebook training
     - preview-time optional encoding
     - offline batch encoding

==================================================
5. OUT OF SCOPE
==================================================

Do NOT implement any of the following in this issue:

- TF-IDF
- DF / IDF statistics
- inverted index
- retrieval / ranking
- reranking
- new codebook algorithms
- new encoding algorithms
- pipeline redesign
- deep feature extraction
- dense retrieval

Do not turn this into a new architecture pass.
This issue is integration only.

==================================================
6. DESIGN REQUIREMENTS
==================================================

A. Default no-regression behavior

This is the most important rule.

When `encoding.enabled` is false or absent:
- scripts/run_pipeline.py must behave exactly as before
- existing preview-oriented feature extraction flow must remain intact

B. Minimal integration point

The encoding hook should happen only after feature save succeeds.

Recommended pattern:
- run existing extraction and save flow
- if encoding is enabled, resolve the correct codebook
- encode the just-produced feature artifact
- save encoded output
- log success or clear failure

Do not spread encoding logic throughout the pipeline.

C. Reuse existing Stage 4.3 code

Use the existing modules:
- src/encoding/io.py
- src/encoding/codebook.py
- src/encoding/bow.py
- src/encoding/storage.py

Do not reimplement BoW in run_pipeline.py.

D. Config-driven design

Unify around a minimal `encoding` section in configs/base.yaml.

Recommended minimal shape:

encoding:
  enabled: false

  input:
    feature_dir: outputs/features
    encoded_dir: outputs/encoded

  codebook:
    output_dir: outputs/indices/codebooks

  bow:
    enabled: true
    normalized: false

You may adjust names slightly if needed, but:
- keep compatibility with existing Stage 4.3 config usage
- do not overdesign
- do not add future TF-IDF fields now

E. Offline batch path

`scripts/encode_features.py` should remain the main full-dataset batch path.

It should:
- read saved feature artifacts
- locate matching codebooks by method
- write encoded artifacts to disk
- produce concise summary logs

F. Clear errors

Typical failure modes should be explained clearly:
- codebook missing for a method
- method mismatch
- invalid encoded output path
- no feature files found

==================================================
7. IMPLEMENTATION GUIDANCE
==================================================

Use the smallest correct integration.

Recommended work items:

1. configs/base.yaml
- normalize the encoding section
- preserve compatibility with Stage 4.3 fields if they already exist

2. scripts/run_pipeline.py
- add a very small encoding-enabled conditional block after feature saving
- do not change the main flow structure unless necessary
- do not make encoding mandatory

3. scripts/encode_features.py
- refine only as needed so it is a clear standalone entrypoint
- avoid moving core logic into the script; keep core logic in src/encoding/

4. Documentation
- add a short run note, README subsection, or docs note
- explain:
  - how to train codebooks
  - how to run preview pipeline with encoding enabled
  - how to batch encode a feature directory

==================================================
8. VALIDATION RULES
==================================================

Implement or verify behavior for at least these cases:

1. Default path
- with encoding disabled, run_pipeline.py preserves existing behavior

2. Preview encoding path
- with encoding enabled and matching codebook available, preview samples can be encoded after feature save

3. Missing codebook path
- clear error message, not silent failure

4. Offline batch path
- scripts/encode_features.py can encode existing outputs/features/*.npz files into outputs/encoded/*.npz

5. Contract stability
- no changes to existing LocalFeatureResult or feature `.npz` contract
- no breaking changes to existing codebook or encoded artifact format

==================================================
9. TESTS / VERIFICATION
==================================================

Add only minimal verification if needed.

Preferred:
- a small test or lightweight integration verification
- or update existing tests minimally if that is more repo-consistent

Do NOT build a large integration suite.

If practical, validate at least:
1. encoding-disabled path leaves pipeline behavior unchanged
2. encoding-enabled path triggers encoded artifact generation when prerequisites exist
3. offline script still works with current config structure

==================================================
10. OUTPUT EXPECTATION
==================================================

Deliver production-quality changes with:
- minimal intrusion
- stable config-driven integration
- clear logs
- no regression by default
- no TF-IDF / indexing / retrieval logic

==================================================
11. FINAL RESPONSE FORMAT
==================================================

At the end, provide:

1. A short summary of what was added or changed
2. The exact files created / modified
3. The final `encoding` config structure
4. How default no-regression behavior was preserved
5. How preview-time encoding works
6. How offline batch encoding works
7. What is intentionally left for Stage 5 / later issues

Do not overclaim.
If something could not be completed, say so clearly.