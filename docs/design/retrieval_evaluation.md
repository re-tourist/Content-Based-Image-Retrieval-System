# Retrieval Evaluation

## Scope

This document describes the canonical retrieval evaluation closure added in Milestone 5.
It sits on top of the sparse retrieval layer:

`canonical split files -> encoded BoW artifacts -> sparse index -> batch retrieval -> metrics -> exported results`

The goal is not to build a general benchmark platform. The goal is to provide a
reproducible, course-report-friendly evaluation flow with stable inputs and outputs.

## Canonical Inputs

The evaluation flow is split-file driven and uses the following canonical manifests:

- `data/splits/gallery.txt`
- `data/splits/query.txt`

The evaluation script reads those split files explicitly and matches them against
encoded artifacts through canonical sample-id normalization.

Important rules:

- the real corpus source is the explicit `--encoded-dir`
- the script does not infer split membership from sample ids or directory names
- `corpus_split` is only a naming / logging label for the saved artifacts

## Relevance Contract

Relevance is defined by the same class-folder semantics used in the coursework:

- remove known dataset prefixes from the sample id
- extract the first path component as the class label
- any gallery item with the same class label as the query is relevant

Examples:

- `test/A0C573/A0C573_20151103074304_6595543738.jpg` -> class label `A0C573`
- `A0C573/A0C573_20151103073308_3029240562.jpg` -> class label `A0C573`

Queries with no relevant gallery items are reported explicitly and are excluded from
the summary `mean_recall_at_k` and `mAP` aggregates.

## Output Contract

Each run saves a stable artifact tree under:

```text
outputs/evaluations/retrieval/<corpus_split>/<method>/<variant>/
```

The evaluation script writes the following JSON files:

- `variant.json`
- `per_query_results.json`
- `per_query_metrics.json`
- `summary_metrics.json`
- `pr_curve.json`

And one run-level manifest:

- `outputs/evaluations/retrieval/<corpus_split>/run_manifest.json`

### `per_query_results.json`

Contains one entry per query with:

- canonical query id and artifact id
- ranked top-k results
- per-query metrics
- prefix PR curve points for queries with relevant gallery items

### `per_query_metrics.json`

Contains only the metric summary for each query:

- `precision_at_k`
- `recall_at_k`
- `average_precision`
- inclusion flags for summary aggregation

### `summary_metrics.json`

Contains the run-level summary:

- query and gallery counts
- mean `precision@k`
- mean `recall@k`
- `mAP`
- retrieval timings
- relevance metadata

### `pr_curve.json`

Contains:

- the macro-averaged PR curve
- per-query PR curve data

## Script

The canonical entry point is:

```bash
python scripts/run_retrieval_eval.py \
  --data-root data \
  --gallery-split-file data/splits/gallery.txt \
  --query-split-file data/splits/query.txt \
  --encoded-dir coursework/week02/outputs/sift_k64_minibatch_kmeans/encoded/test \
  --index-root outputs/indices/inverted \
  --output-root outputs/evaluations/retrieval_smoke \
  --compare \
  --top-k 10
```

Behavior:

- loads the canonical gallery/query split files
- loads encoded artifacts from the explicit `--encoded-dir`
- builds or loads the sparse index for the method being evaluated
- runs the canonical comparison paths:
  - `tf + dot`
  - `tfidf + cosine`
- exports ranked results, metrics, and PR data

## Failure Modes

The evaluation layer raises explicit errors for:

- missing split files
- empty gallery or query split files
- missing encoded artifacts for split entries
- normalized BoW artifacts
- histogram size mismatches
- method mismatches
- unsupported weighting / scoring combinations

## Smoke Procedure

The smallest reproducible smoke path is:

1. prepare encoded BoW artifacts for the gallery and query split entries
2. ensure the canonical split files exist under `data/splits/`
3. run `scripts/run_retrieval_eval.py`
4. inspect the exported JSON files under `outputs/evaluations/retrieval/...`

