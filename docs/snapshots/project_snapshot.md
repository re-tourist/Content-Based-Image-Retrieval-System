# project_snapshot

## 1. Repository Overview

- Repository name: Content-Based-Image-Retrieval-System
- Apparent purpose: coursework-driven and engineering-driven hybrid image retrieval system
- Repository type:
  - [ ] single-package
  - [ ] monorepo
  - [ ] research repo
  - [x] coursework repo
  - [ ] app/service repo
  - [x] library/tooling repo
- Primary language(s): Python, Markdown, TOML, shell scripts
- Main framework(s): OpenCV, NumPy, Matplotlib, GitHub templates, Codex workflow docs
- Current documentation quality:
  - [ ] strong
  - [x] usable
  - [ ] weak
  - [ ] unclear

---

## 2. Top-Level Structure

- `configs/`
  - runtime configuration, currently anchored by `configs/base.yaml`
- `data/`
  - raw images, dataset manifests, split files, and processed inputs
- `src/`
  - implementation modules for datasets, features, encoding, indexing, retrieval, evaluation, and utilities
- `scripts/`
  - runnable entrypoints for preprocessing, encoding, indexing, retrieval, and evaluation
- `outputs/`
  - generated artifacts such as features, encoded histograms, indices, figures, and logs
- `docs/`
  - project context, workflow rules, design docs, plans, reviews, handoff notes, and snapshots
- `templates/`
  - reusable issue / PR / report templates
- `.github/`
  - GitHub issue and PR templates

Also present:

- `.codex/`
  - Codex project configuration
- `.agents/`
  - agent-related local assets
- `coursework/`
  - course-aligned reference material and supporting artifacts

High-frequency code area:

- `src/`
- `scripts/`

High-frequency docs area:

- `docs/ai/`
- `docs/design/`
- `docs/plan/`

Stable area that should not be edited casually:

- `src/encoding/storage.py`
- `src/evaluation/`
- `docs/ai/PROJECT_CONTEXT.md`
- `docs/ai/WORKFLOW_GUIDE.md`

---

## 3. Key Entry Points

### Runtime / App entrypoints
- `scripts/run_pipeline.py`
- `scripts/run_demo.py`

### Training / Experiment entrypoints
- `scripts/train_codebook.py`
- `scripts/encode_features.py`
- `scripts/build_inverted_index.py`
- `scripts/run_retrieval_eval.py`

### Inference / Evaluation entrypoints
- `scripts/search_images.py`
- `scripts/run_retrieval_eval.py`

### Main configs
- `configs/base.yaml`
- `.codex/config.toml`

### Main docs
- `docs/ai/PROJECT_CONTEXT.md`
- `docs/ai/WORKFLOW_GUIDE.md`
- `docs/design/pipeline_skeleton.md`
- `docs/design/indexing_usage.md`
- `docs/design/retrieval_evaluation.md`

If a new agent cannot confirm one of these, it should mark the gap explicitly.

---

## 4. Tooling Signals

### Dependency / package manager
- signal: Python project with script-based execution
- file(s): `requirements.txt` if present, plus imported modules in `scripts/` and `src/`

### Build system
- signal: none clearly discoverable
- file(s): not clearly discoverable

### Lint / format
- signal: none clearly discoverable
- file(s): not clearly discoverable

### Type checking
- signal: none clearly discoverable
- file(s): not clearly discoverable

### Test framework
- signal: `unittest`
- file(s): `tests/`

### CI / GitHub Actions
- signal: templates exist; CI wiring not clearly discoverable from the snapshot
- file(s): `.github/`, repository workflow files if added later

### Docker / environment / devcontainer
- signal: not clearly discoverable
- file(s): not clearly discoverable

---

## 5. Validation Signals

### Install
- command / evidence: not clearly discoverable from the repo snapshot

### Run
- command / evidence:
  - `python scripts/run_pipeline.py`
  - `python scripts/encode_features.py`
  - `python scripts/build_inverted_index.py`
  - `python scripts/search_images.py`
  - `python scripts/run_retrieval_eval.py`

### Test
- command / evidence:
  - `python -m unittest discover -s tests`

### Lint
- command / evidence: not clearly discoverable

### Build
- command / evidence: not clearly discoverable

### Focused / local checks
- command / evidence:
  - `git diff --check`
  - smoke retrieval evaluation on canonical splits

---

## 6. Risk Zones

### Sensitive directories/files
- `src/encoding/storage.py`
- `src/evaluation/`
- `configs/base.yaml`
- `docs/ai/PROJECT_CONTEXT.md`
- `docs/design/pipeline_skeleton.md`

### Potentially generated files
- `outputs/features/`
- `outputs/encoded/`
- `outputs/indices/`
- `outputs/evaluations/`
- `outputs/figures/`

### Infra / deployment / secrets-related areas
- `.codex/`
- `.github/`

### Public interfaces / schemas / stable outputs
- `outputs/*.npz`
- `outputs/evaluations/retrieval/*`
- `docs/design/indexing_usage.md`
- `docs/design/retrieval_evaluation.md`

### Places where small edits could have repo-wide impact
- `scripts/run_pipeline.py`
- `scripts/encode_features.py`
- `scripts/build_inverted_index.py`
- `scripts/run_retrieval_eval.py`

---

## 7. Documentation State

### Existing docs
- `README.md`: not confirmed in this snapshot
- `AGENTS.md`: present and should be treated as the root workflow manual
- `docs/...`: substantial design, plan, workflow, and handoff documentation is present
- other relevant docs: `templates/`, `.github/`

### Likely missing docs
- [ ] AGENTS.md
- [ ] PROJECT_CONTEXT.md
- [ ] stage plan
- [ ] issue breakdown
- [ ] review checklist
- [ ] contract freeze
- [ ] closeout doc

### Suspected stale docs
- `docs/ai/prompt/stage4/prompt_stage4-4.md` may still be stale relative to the current implementation
- any older notes that still say indexing or evaluation is not implemented

---

## 8. Working Recommendations

### Read first
- `AGENTS.md`
- `docs/ai/PROJECT_CONTEXT.md`
- `docs/ai/WORKFLOW_GUIDE.md`

### Avoid touching casually
- `src/encoding/storage.py`
- `src/evaluation/`
- `docs/design/pipeline_skeleton.md`

### Validate first
- `python -m unittest discover -s tests`
- `git diff --check`
- a smoke retrieval evaluation on canonical splits

### Document before coding if missing
- milestone spec
- issue plan draft
- contract freeze
- closeout note

---

## 9. Open Uncertainties

- whether GitHub issue / milestone creation is actually available in the current session
- whether a package manager manifest is present and canonical
- whether older prompt files under `docs/ai/prompt/` need a final archive pass

These must be verified explicitly rather than assumed.
