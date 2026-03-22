# Encoding Stage Usage

This note covers the current Stage 4 encoding entrypoints.

## Config Shape

The active config lives under `encoding` in `configs/base.yaml`:

```yaml
encoding:
  enabled: false
  input:
    feature_dir: outputs/features
    encoded_dir: outputs/encoded
  codebook:
    enabled: false
    output_dir: outputs/indices/codebooks
    trainer: minibatch_kmeans
    n_clusters: 256
    max_descriptors: 50000
    batch_size: 1024
    random_state: 42
  bow:
    enabled: true
    normalized: false
```

Notes:

- `encoding.enabled` controls the optional preview-time hook in `scripts/run_pipeline.py`.
- `encoding.codebook.enabled` controls whether `scripts/train_codebook.py` is allowed to run.
- `encoding.bow.enabled` controls whether `scripts/encode_features.py` is allowed to run.
- `encoding.input.encoded_dir` is the default save location for encoded BoW artifacts.
- Legacy Stage 4.3 fields `encoding.bow.codebook_dir`, `encoding.bow.output_dir`, and `encoding.bow.normalize` are still accepted for compatibility.

## 1. Train Codebooks

Enable codebook training in a config copy, then run:

```bash
python scripts/train_codebook.py --config path/to/codebook_config.yaml
```

Expected inputs and outputs:

- reads local feature artifacts from `encoding.input.feature_dir`
- writes method-specific codebooks to `encoding.codebook.output_dir`

## 2. Preview-Time Encoding From `run_pipeline.py`

To encode each preview sample right after its feature artifact is saved:

1. set `encoding.enabled: true`
2. keep `encoding.bow.enabled: true`
3. point `encoding.codebook.output_dir` to a directory that already contains the matching method-specific codebook

Then run:

```bash
python scripts/run_pipeline.py --config path/to/preview_encoding_config.yaml
```

Behavior:

- when `encoding.enabled` is `false`, `run_pipeline.py` behaves as before
- when `encoding.enabled` is `true`, the script encodes the just-saved feature artifact and writes the BoW result to `encoding.input.encoded_dir`
- if the matching codebook is missing, the run fails with a clear error instead of skipping encoding silently

## 3. Offline Batch Encoding

To encode an existing feature directory in batch:

```bash
python scripts/encode_features.py --config path/to/encoding_config.yaml
```

Optional method filters are supported:

```bash
python scripts/encode_features.py --config path/to/encoding_config.yaml --method sift --method orb
```

Behavior:

- reads saved feature artifacts from `encoding.input.feature_dir`
- loads the matching method-specific codebook from `encoding.codebook.output_dir`
- writes encoded BoW artifacts to `encoding.input.encoded_dir`
