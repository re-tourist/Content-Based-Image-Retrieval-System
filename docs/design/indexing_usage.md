# Indexing Usage

## Scope

This document describes the Milestone 5 sparse retrieval layer:

- TF-IDF statistics
- method-specific inverted indexes
- top-k search over encoded BoW artifacts

The indexing layer consumes `outputs/encoded/*.npz` and does not read raw local descriptors directly.

## Input Contract

Only raw-count encoded BoW artifacts are accepted for indexing and search.

Required constraints:

- `encoding_type == "bow"`
- `normalized == False`
- `histogram_dtype == "int32"`
- method-aware artifacts only
- one method per index bundle

The `corpus_split` field is a naming and logging label only.
The actual corpus source is the explicit `--encoded-dir` passed to the build script.

## Output Layout

The canonical sparse retrieval layout is:

```text
outputs/indices/inverted/<corpus_split>/<method>/
  |- tfidf_stats.npz
  |- index_tf.npz
  `- index_tfidf.npz
```

Artifact semantics:

- `tfidf_stats.npz` stores `df` and `idf`
- `index_tf.npz` stores raw TF posting weights
- `index_tfidf.npz` stores TF-IDF posting weights

## Weighting and Scoring

Milestone 5 supports only two paths:

1. `tf + dot`
2. `tfidf + cosine`

The weighting mode and scoring mode are intentionally separate:

- `tf` means raw term counts with no hidden normalization
- `tfidf` means raw counts multiplied by smooth IDF
- `dot` means plain dot-product scoring
- `cosine` means dot-product scoring divided by query and document norms

## Build Command

Example:

```bash
python scripts/build_inverted_index.py \
  --encoded-dir outputs/encoded/gallery_smoke \
  --index-dir outputs/indices/inverted \
  --corpus-split gallery_smoke \
  --method sift
```

Notes:

- `--encoded-dir` is required and selects the actual corpus source
- `--corpus-split` only labels the saved artifacts
- `--method` may be repeated to filter methods
- the build command saves both weighting modes by default

## Search Command

Single query example:

```bash
python scripts/search_images.py \
  --query-encoded outputs/encoded/query_smoke/A03Z78__A03Z78_20151127145544_6753254871.jpg.npz \
  --index-dir outputs/indices/inverted/gallery_smoke/SIFT \
  --weighting tfidf \
  --scoring cosine
```

Batch query example:

```bash
python scripts/search_images.py \
  --query-dir outputs/encoded/query_smoke \
  --index-dir outputs/indices/inverted/gallery_smoke/SIFT \
  --compare
```

Notes:

- `--query-dir` is the minimal batch-query entrypoint
- `--query-sample-id` can select one artifact from a batch directory
- `--compare` runs both supported scoring paths and prints side-by-side output

## Failure Modes

The indexing layer raises explicit errors for the following cases:

- missing encoded or index files
- normalized BoW input
- method mismatch between query and index
- histogram dimension mismatch
- invalid weighting / scoring combination

These failures are intended to be descriptive and stable so they can be surfaced in CLI output and tests.

## Smoke Test

The smallest reproducible smoke chain is:

1. use the checked-in feature artifacts under `outputs/features/`
2. encode a gallery subset into `outputs/encoded/gallery_smoke/`
3. encode a query subset into `outputs/encoded/query_smoke/`
4. build the index into `outputs/indices/inverted/gallery_smoke/SIFT/`
5. search the query encoded directory against the saved index
6. compare `tf + dot` and `tfidf + cosine`

