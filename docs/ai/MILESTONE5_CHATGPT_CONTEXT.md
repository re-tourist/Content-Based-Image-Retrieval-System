# Milestone 5 Final Summary

## Status

Milestone 5 is complete at the repository level.

The implemented scope now includes:

- TF-IDF statistics
- method-specific inverted indexes
- sparse retrieval with `tf + dot` and `tfidf + cosine`
- canonical retrieval evaluation closure
- split-driven metrics export for `precision@k`, `recall@k`, `AP`, `mAP`, and PR curve data

## Delivered

### Sparse retrieval

- raw-count BoW encoded artifacts are accepted as input
- normalized BoW artifacts are rejected by design
- TF-IDF statistics are saved and loaded as stable `.npz` artifacts
- inverted indexes are saved per method and per weighting mode

### Canonical retrieval evaluation

- evaluation is driven by `data/splits/gallery.txt` and `data/splits/query.txt`
- relevance is defined by class-folder equality after canonical sample-id normalization
- per-query ranked results, per-query metrics, summary metrics, and PR data are exported as JSON
- the canonical entrypoint is `scripts/run_retrieval_eval.py`

### Documentation

- project context is aligned with the current code state
- sparse indexing usage is documented
- canonical retrieval evaluation is documented separately

## Verification

Completed validation:

- full test suite: passed
- synthetic metric tests: passed
- indexing storage tests: passed
- retrieval evaluation workflow tests: passed
- real smoke test with canonical splits and real encoded artifacts: passed

Observed smoke result:

- both canonical paths executed successfully
  - `tf + dot`
  - `tfidf + cosine`

## Known Limits

- GitHub issue / milestone creation could not be completed from this environment because the current toolchain does not expose a usable issue/milestone creation path.
- `scripts/run_pipeline.py` remains preview-oriented and is intentionally not turned into a canonical retrieval orchestrator.
- The project still does not include reranking, query expansion, dense retrieval, or hybrid fusion in Milestone 5.

## Notes For Future Work

If you want to extend beyond Milestone 5, the next natural step is retrieval evaluation reporting polish:

- PR curve plotting
- report-ready tables
- optional comparison summaries
- later, `mAP`-driven experiment tracking

