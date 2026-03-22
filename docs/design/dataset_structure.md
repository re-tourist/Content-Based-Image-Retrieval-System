# Dataset Structure

## Goal

Milestone 2 standardizes how the project organizes image data and split lists without
introducing a heavy data engineering layer. The current goal is to make the dataset
layout stable enough for preprocessing, local feature extraction, and later indexing
modules to read from a shared convention.

## Directory Convention

The project now treats `data/` as the dataset root and reserves three standardized
subdirectories:

```text
data/
|- raw/
|- processed/
`- splits/
   |- train.txt
   |- gallery.txt
   `- query.txt
```

### `raw/`

`data/raw/` is the long-term home for raw input images. In the current repository,
the historical source images are still stored under `data/train/image` and `data/test`.
This issue does not force a full migration. Instead, it standardizes access through
split files first and keeps the legacy-compatible layout working.

### `processed/`

`data/processed/` is reserved for later preprocessing outputs. It may remain empty
during this issue, but the directory is now part of the official project layout.

### `splits/`

`data/splits/` stores plain-text split manifests. These files are the main output of
Issue 2.1 and define which images belong to `train`, `gallery`, and `query`.

## Split Semantics

- `train`: images used for the main database-side preparation flow.
- `gallery`: candidate retrieval images that future query images will be matched against.
- `query`: query-side images used to trigger retrieval.

The current minimal compatibility rule is:

- `train.txt` includes all images found under `data/train/image`
- `query.txt` includes the first sorted image from each class directory under `data/test`
- `gallery.txt` includes the remaining sorted images from each class directory under `data/test`

This keeps the existing repository data usable without introducing random sampling or
complex partition logic.

## Split File Format

Each split file uses one `data/`-relative image path per line:

```text
train/image/A03Z78/A03Z78_20151127145344_6753243118.jpg
test/A0C573/A0C573_20151103073308_3029240562.jpg
```

Rules:

- one line represents one image
- paths are relative to `data/`
- blank lines are ignored by the loader
- comment lines starting with `#` are ignored by the loader

The format is intentionally minimal so it can be read by scripts, the dataset loader,
and later pipeline modules without extra parsing layers.

## Build Script

Use the split generation script from the project root:

```bash
python scripts/build_splits.py
```

The script:

1. loads `configs/base.yaml`
2. resolves `dataset.root`, `dataset.splits_dir`, and the train/test source directories
3. creates `data/raw/`, `data/processed/`, and `data/splits/` if needed
4. generates `train.txt`, `gallery.txt`, and `query.txt`
5. prints concise generation statistics

Minimal config fields used by the script:

- `dataset.root`
- `dataset.raw_dir`
- `dataset.processed_dir`
- `dataset.splits_dir`
- `dataset.train_source_dir`
- `dataset.test_source_dir`
- `dataset.train_split`
- `dataset.gallery_split`
- `dataset.query_split`

## Dataset Loader Integration

`ImageDatasetLoader` now supports two compatible modes:

1. directory scan mode: recursively scan a directory exactly as in Stage 1
2. split file mode: read image paths from a split file and resolve them relative to
   the dataset root

Example:

```python
from src.datasets import ImageDatasetLoader

loader = ImageDatasetLoader(
    split="query",
    split_file="data/splits/query.txt",
    data_root="data",
    verbose=False,
)
```

This keeps the Stage 1 pipeline working while allowing Milestone 2 code to depend on
standardized split manifests.

## Compatibility Assumptions

The current repository does not yet store all source images under `data/raw/`. The
implementation therefore uses a minimal compatibility strategy:

- keep `data/train/image` and `data/test` as the active source image locations
- standardize dataset access through `data/splits/*.txt`
- reserve `data/raw/` and `data/processed/` as the official long-term locations

This is enough for Milestone 2 without forcing a risky bulk data migration.
