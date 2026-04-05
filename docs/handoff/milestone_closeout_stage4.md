# milestone_closeout

## Milestone Info

- milestone_id: M4
- milestone_title: Milestone 4: Feature Encoding and Codebook Wiring
- milestone_type: implementation
- priority: P1
- status:
  - [x] complete
  - [ ] partially complete
  - [ ] blocked
  - [ ] handed off with risks
- date: 2026-04-05
- related_plan: docs/plan/stage_plan/plan_stage4.md
- related_issue_plan: docs/ai/prompt/stage4/prompt_stage4-1.md, docs/ai/prompt/stage4/prompt_stage4-2.md, docs/ai/prompt/stage4/prompt_stage4-3.md, docs/ai/prompt/stage4/prompt_stage4-4.md
- related_issues: #26, #27, #28, #29
- related_prs: #30, #31, #32, #33
- related_contract_freeze: none (no formal freeze doc recorded)

---

## 1. Executive Summary

Milestone 4 aimed to add a minimal, method-aware feature-encoding layer on top of the local-feature pipeline. The delivered work includes encoding contracts and IO, descriptor sampling + codebook training, BoW encoding + encoded artifact persistence, and optional pipeline/batch integration. The milestone is safe to move on from because the encoding artifacts are stable, method-aware, and explicitly validated. The biggest remaining gap is that indexing and sparse retrieval are not part of this milestone; they are handled in Milestone 5.

---

## 2. Planned vs Delivered

### Planned
- encoding contracts and feature file reader
- descriptor sampling and reusable codebook training
- BoW encoding and encoded artifact persistence
- optional pipeline hook and offline batch encoding entry

### Delivered
- encoding contracts and feature file reader
- descriptor sampling and reusable codebook training
- BoW encoding and encoded artifact persistence
- optional pipeline hook and offline batch encoding entry

### Partially Delivered
- none

### Not Delivered
- none (indexing and retrieval were explicitly out of scope)

---

## 3. Key Files and Changes

### Code
- `src/encoding/types.py`
  - role: encoding-stage data contracts
  - responsibility in this milestone: stable, validated encoding inputs
- `src/encoding/io.py`
  - role: feature `.npz` reader + validation
  - responsibility in this milestone: safe loading and normalized encoding inputs
- `src/encoding/sampling.py`
  - role: method-aware descriptor sampling
  - responsibility in this milestone: bounded sampling for codebook training
- `src/encoding/codebook.py`
  - role: codebook training + save/load
  - responsibility in this milestone: method-specific codebook artifacts
- `src/encoding/bow.py`
  - role: BoW assignment + histogram building
  - responsibility in this milestone: encoded feature generation
- `src/encoding/storage.py`
  - role: encoded artifact persistence
  - responsibility in this milestone: stable encoded `.npz` contract

### Docs
- `docs/design/encoding_usage.md`
  - role: encoding-stage usage and CLI guidance
  - whether it is aligned with implementation: yes

### Config / Scripts / Assets
- `configs/base.yaml`
  - role: encoding config normalization
  - whether it is reusable as-is: yes
- `scripts/train_codebook.py`
  - role: offline codebook training entry
  - whether it is reusable as-is: yes
- `scripts/encode_features.py`
  - role: offline batch encoding entry
  - whether it is reusable as-is: yes
- `scripts/run_pipeline.py`
  - role: optional encoding hook after feature save
  - whether it is reusable as-is: yes (disabled by default)

---

## 4. Validation Summary

### Validation Run
- `python -m unittest discover -s tests -p "test_*.py" -v` (last known run during milestone work)
- `python scripts/train_codebook.py --config <config>` (manual spot check)
- `python scripts/encode_features.py --config <config>` (manual spot check)

### What These Checks Actually Prove
- proved: encoding IO validation, codebook training, BoW encoding, and encoded artifact save/load work for SIFT/ORB
- did not prove: scalability or indexing / retrieval correctness

### Missing or Weak Validation
- no large-scale performance test
- no end-to-end indexing/retrieval test in Milestone 4 (by design)

---

## 5. Risks and Known Limitations

### P0 / blocking
- none

### P1 / serious but not blocking
- none

### P2 / should improve later
- KMeans training and query assignment are CPU-only and may be slow at scale
- joblib/loky warnings can appear on some Windows environments; they do not affect correctness

---

## 6. Contract / Scope Notes

- [x] completely within the frozen contract
- [ ] small deviation, explained
- [ ] contract-level issue occurred
- [ ] actual scope drift occurred
- [x] no formal contract freeze existed

Notes:

- touched boundaries: encoding contracts, codebook artifacts, encoded BoW artifacts
- whether re-freeze is needed before the next milestone: yes, if Milestone 5 introduces new index contracts

---

## 7. Recommended Next Entry Point

### Recommended first task
- begin Milestone 5 indexing and sparse retrieval on top of encoded artifacts

### Recommended first files to read
- `docs/design/indexing_usage.md`
- `docs/design/retrieval_evaluation.md`
- `scripts/build_inverted_index.py`
- `scripts/search_images.py`
- `scripts/run_retrieval_eval.py`

### Recommended first checks
- verify encoded BoW artifacts are raw-count (`normalized == False`)
- verify method-specific codebooks exist under `outputs/indices/codebooks`
- run a smoke chain using `scripts/build_inverted_index.py` and `scripts/search_images.py`

### Recommended decision to make before coding
- confirm the canonical evaluation output format and whether any additional metrics are required

---

## 8. Handoff Guidance

- do not repeat: encoding contracts, BoW histogram generation, or codebook training logic
- do not misunderstand: encoded artifacts are raw-count BoW, not TF-IDF
- known limits that look like bugs: empty descriptors yield zero histograms by design
- high-value confirmations before continuing: check for method mismatch handling and `normalized == False` before indexing

- Note 1: index and retrieval modules must consume encoded artifacts via `src/encoding/storage.py`
- Note 2: do not alter the `.npz` feature or encoded artifact contracts without a contract freeze
- Note 3: keep indexing logic method-aware (no SIFT/ORB mixing)

---

## 9. Final Verdict

- [x] milestone can be cleanly closed
- [ ] milestone can be closed with documented limitations
- [ ] milestone should remain open pending one last validation
- [ ] milestone should not be closed because the result is not yet reliable

Current conclusion:
Milestone 4 is complete and closed. The encoding layer is stable and ready for Milestone 5 indexing work.
