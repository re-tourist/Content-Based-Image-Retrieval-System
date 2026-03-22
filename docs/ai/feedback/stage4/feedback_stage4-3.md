# Issue 4.3 Feedback

## Changed

- `src/encoding/bow.py` now implements method-aware visual word assignment, BoW histogram construction, empty-safe encoding, and minimal batch offline encoding from saved feature files.
- `src/encoding/storage.py` now defines `EncodedFeatureResult` and the stable encoded artifact save/load helpers.
- `src/encoding/__init__.py` now exports the BoW and encoded-storage interfaces.
- `scripts/encode_features.py` now provides a standalone offline entry point for batch BoW encoding.
- `configs/base.yaml` now includes the minimal `encoding.bow` config section for offline encoded-feature generation.
- `tests/test_bow.py` now verifies BoW encoding, empty-descriptor handling, mismatch validation, and encoded artifact roundtrip behavior.

## BoW Design

The BoW stage now reuses the existing encoding and codebook interfaces instead of adding parallel loaders.

Encoding flow:

- load feature artifact through `src/encoding/io.py`
- build `EncodingInput`
- load the matching method-specific codebook through `src/encoding/codebook.py`
- assign each descriptor to its nearest cluster center
- build an image-level histogram of length `n_clusters`

Key behavior:

- SIFT features only pair with SIFT codebooks
- ORB features only pair with ORB codebooks
- empty descriptor inputs produce a valid zero histogram instead of being skipped
- raw BoW counts remain the default representation
- optional histogram normalization is supported but no TF-IDF fields are introduced

## Encoded Artifact Contract

The encoded artifact is saved as a compressed `.npz` under `outputs/encoded/`.

Stored fields:

- `sample_id`
- `method`
- `encoding_type`
- `num_visual_words`
- `histogram`
- `histogram_dtype`
- `num_descriptors`
- `descriptors_present`
- `codebook_path`
- `normalized`

This keeps the persisted format minimal and stable while providing the metadata needed for the next TF-IDF and indexing stages.

## Validation

Commands verified:

- `python -m unittest discover -s tests -p "test_*.py" -v`
- `python scripts/encode_features.py --config outputs/logs/test_encode_features.yaml`

Observed coverage:

- BoW encoding works for small synthetic SIFT-like descriptors
- BoW encoding works for small synthetic ORB-like descriptors
- empty descriptors produce zero histograms with the correct vocabulary size
- mismatched method / codebook pairs fail clearly
- mismatched descriptor dimensionality fails clearly
- encoded feature artifacts save and load back with matching metadata

Observed batch behavior:

- the standalone script reads saved feature artifacts and method-specific codebooks from disk
- batch encoding writes stable `.npz` encoded outputs under the configured directory
- the batch path does not modify `run_pipeline.py`

## Acceptance Alignment

This issue now satisfies the intended Stage 4.3 scope because:

- codebooks can be loaded and used for visual word assignment
- image-level BoW histograms are produced from valid encoding inputs
- empty-descriptor samples are encoded into valid zero histograms
- encoded feature artifacts are persisted and loadable back
- a minimal offline batch encoding path exists
- no DF, IDF, TF-IDF, indexing, retrieval, or pipeline redesign was added
