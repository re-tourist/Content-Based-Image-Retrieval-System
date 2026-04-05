# Issue 4.1 Feedback

## Changed

- `src/encoding/types.py` now defines the minimal encoding-stage data containers: `FeatureFileRecord` and `EncodingInput`.
- `src/encoding/io.py` now implements validated feature artifact loading from `.npz` and unified encoding-input construction from both disk and in-memory local features.
- `src/encoding/__init__.py` now exports the encoding-stage public interface.
- `tests/test_encoding_io.py` now provides focused verification for valid SIFT inputs, valid ORB inputs, empty-descriptor inputs, in-memory conversion, and invalid metadata detection.

## Encoding Contract

The new encoding layer now supports two input paths without changing the upstream local feature contract:

- saved feature files under `outputs/features/*.npz`
- in-memory `LocalFeatureResult`

The implementation keeps the encoding interface method-aware:

- `SIFT` descriptors must be `float32` with dimension `128`
- `ORB` descriptors must be `uint8` with dimension `32`

Empty descriptor cases are handled explicitly instead of failing implicitly.
When the input is truly empty, the loader/builder returns a valid empty encoding state rather than crashing.

## Validation Behavior

The `.npz` loader now uses `allow_pickle=False` and validates:

- required keys
- method
- `descriptors_present`
- `descriptor_shape`
- `descriptor_dtype`
- compatibility between metadata and the actual descriptor array

If metadata and descriptor content disagree, the implementation raises a clear `ValueError` instead of silently correcting the file.

For normalized encoding input:

- non-empty inputs keep the real descriptor array and validated metadata
- empty inputs are normalized into a stable method-aware encoding-facing representation

## Validation

Commands verified:

- `python -m unittest discover -s tests -p "test_encoding_io.py" -v`

Observed test coverage:

- valid SIFT feature `.npz` loading succeeds
- valid ORB feature `.npz` loading succeeds
- empty-descriptor feature `.npz` loading succeeds
- in-memory `LocalFeatureResult` to `EncodingInput` conversion succeeds
- invalid dtype / invalid descriptor shape metadata is rejected

Real artifact compatibility check:

- a real existing feature file under `outputs/features/` was loaded successfully
- the resulting `FeatureFileRecord` and `EncodingInput` matched the saved SIFT descriptor shape and dtype

## Acceptance Alignment

This issue now satisfies the intended Stage 4.1 scope because:

- the encoding package exists under `src/encoding/`
- stable encoding-stage data structures are defined
- feature files can be read safely and validated against the current `.npz` contract
- both disk-based and in-memory local feature inputs are converted into one normalized encoding-facing structure
- empty descriptor cases are handled explicitly
- no codebook training, BoW encoding, TF-IDF, indexing, retrieval, or upstream pipeline redesign was added
