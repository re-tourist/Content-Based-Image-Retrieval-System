1. Added the minimal encoding foundation for Issue 4-1: a new encoding package with a validated file-reader path and a unified encoding-input path for both saved feature artifacts and in-memory `LocalFeatureResult`. The implementation is in [src/encoding/types.py](d:/MyProject/Content-Based-Image-Retrieval-System/src/encoding/types.py), [src/encoding/io.py](d:/MyProject/Content-Based-Image-Retrieval-System/src/encoding/io.py), and [src/encoding/__init__.py](d:/MyProject/Content-Based-Image-Retrieval-System/src/encoding/__init__.py). It does not change the existing extractor, `LocalFeatureResult`, or `.npz` save contract.

2. Files created / modified:
   Created [src/encoding/__init__.py](d:/MyProject/Content-Based-Image-Retrieval-System/src/encoding/__init__.py), [src/encoding/types.py](d:/MyProject/Content-Based-Image-Retrieval-System/src/encoding/types.py), [src/encoding/io.py](d:/MyProject/Content-Based-Image-Retrieval-System/src/encoding/io.py), and [tests/test_encoding_io.py](d:/MyProject/Content-Based-Image-Retrieval-System/tests/test_encoding_io.py). No existing pipeline files were modified.

3. Assumptions made:
   Empty descriptor inputs are valid and normalize to a method-aware empty `EncodingInput`, so empty SIFT becomes shape `(0, 128)` / dtype `float32` and empty ORB becomes `(0, 32)` / `uint8`, even when the saved `.npz` metadata is empty.
   When descriptors are present, descriptor row count must match `num_keypoints`, which matches the current extractor/save behavior.

4. Acceptance criteria coverage:
   `FeatureFileRecord` and `EncodingInput` are implemented; `load_feature_npz(...)` reads with `allow_pickle=False`, checks required keys, validates method, `descriptors_present`, descriptor dtype/shape, and rejects inconsistent metadata. `build_encoding_input_from_feature_record(...)` and `build_encoding_input_from_local_feature(...)` both produce the same stable encoding-facing structure. Empty descriptors are handled explicitly instead of failing implicitly. Verification was added in [tests/test_encoding_io.py](d:/MyProject/Content-Based-Image-Retrieval-System/tests/test_encoding_io.py) and passes with `python -m unittest discover -s tests -p "test_encoding_io.py" -v`. I also verified the loader against one real existing saved SIFT artifact in `outputs/features/`.

5. Leave for Issue 4-2:
   Codebook training, descriptor sampling, BoW encoding, any histogram/output artifact format, and any pipeline/config wiring that actually runs encoding. I also left `docs/design/encoding_contract.md` out to keep this issue minimal.