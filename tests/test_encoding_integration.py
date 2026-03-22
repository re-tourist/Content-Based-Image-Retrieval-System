from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from scripts.encode_features import run_feature_encoding
from scripts.run_pipeline import maybe_encode_saved_feature
from src.encoding import CodebookArtifact, load_encoded_feature, save_codebook_artifact


class EncodingIntegrationTests(unittest.TestCase):
    def test_preview_encoding_disabled_keeps_default_path_inactive(self) -> None:
        result = maybe_encode_saved_feature(None, {"encoding": {"enabled": False}})
        self.assertIsNone(result)

    def test_preview_encoding_enabled_saves_encoded_artifact(self) -> None:
        workspace = self._make_workspace()
        feature_path = self._write_feature_npz(
            workspace / "features" / "preview_sift.npz",
            method="SIFT",
            descriptors=np.vstack([
                np.zeros((1, 128), dtype=np.float32),
                np.full((1, 128), 10.0, dtype=np.float32),
            ]),
        )
        save_codebook_artifact(self._make_sift_codebook(), workspace / "codebooks")

        encoded_path = maybe_encode_saved_feature(
            feature_path,
            {
                "encoding": {
                    "enabled": True,
                    "input": {"encoded_dir": str(workspace / "encoded")},
                    "codebook": {"output_dir": str(workspace / "codebooks")},
                    "bow": {"enabled": True, "normalized": False},
                }
            },
        )
        loaded = load_encoded_feature(encoded_path)

        self.assertTrue(encoded_path.is_file())
        self.assertEqual(loaded.method, "SIFT")
        self.assertEqual(loaded.num_visual_words, 2)
        self.assertEqual(loaded.histogram.tolist(), [1, 1])
        self.assertFalse(loaded.normalized)

    def test_preview_encoding_missing_codebook_raises_clear_error(self) -> None:
        workspace = self._make_workspace()
        feature_path = self._write_feature_npz(
            workspace / "features" / "missing_codebook.npz",
            method="SIFT",
            descriptors=np.ones((1, 128), dtype=np.float32),
        )

        with self.assertRaises(FileNotFoundError) as exc:
            maybe_encode_saved_feature(
                feature_path,
                {
                    "encoding": {
                        "enabled": True,
                        "input": {"encoded_dir": str(workspace / "encoded")},
                        "codebook": {"output_dir": str(workspace / "codebooks")},
                        "bow": {"enabled": True, "normalized": False},
                    }
                },
            )

        self.assertIn("No codebook artifact available", str(exc.exception))

    def test_offline_batch_encoding_supports_stage43_legacy_fields(self) -> None:
        workspace = self._make_workspace()
        self._write_feature_npz(
            workspace / "features" / "legacy_orb.npz",
            method="ORB",
            descriptors=np.vstack([
                np.zeros((1, 32), dtype=np.uint8),
                np.full((2, 32), 255, dtype=np.uint8),
            ]),
        )
        save_codebook_artifact(self._make_orb_codebook(), workspace / "codebooks")

        saved_paths = run_feature_encoding(
            {
                "encoding": {
                    "input": {"feature_dir": str(workspace / "features")},
                    "bow": {
                        "enabled": True,
                        "codebook_dir": str(workspace / "codebooks"),
                        "output_dir": str(workspace / "encoded"),
                        "normalize": False,
                    },
                },
                "output": {},
            }
        )
        loaded = load_encoded_feature(saved_paths[0])

        self.assertEqual(len(saved_paths), 1)
        self.assertEqual(loaded.method, "ORB")
        self.assertEqual(loaded.histogram.tolist(), [1, 2])
        self.assertEqual(loaded.histogram_dtype, "int32")

    def _make_workspace(self) -> Path:
        root = Path("outputs") / "logs" / "test_tmp" / uuid.uuid4().hex
        (root / "features").mkdir(parents=True, exist_ok=True)
        (root / "codebooks").mkdir(parents=True, exist_ok=True)
        (root / "encoded").mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def _make_sift_codebook(self) -> CodebookArtifact:
        return CodebookArtifact(
            method="SIFT",
            trainer="minibatch_kmeans",
            n_clusters=2,
            descriptor_dim=128,
            descriptor_dtype="float32",
            training_descriptor_count=2,
            random_state=0,
            batch_size=2,
            cluster_centers=np.vstack([
                np.zeros((1, 128), dtype=np.float32),
                np.full((1, 128), 10.0, dtype=np.float32),
            ]),
        )

    def _make_orb_codebook(self) -> CodebookArtifact:
        return CodebookArtifact(
            method="ORB",
            trainer="minibatch_kmeans",
            n_clusters=2,
            descriptor_dim=32,
            descriptor_dtype="uint8",
            training_descriptor_count=2,
            random_state=0,
            batch_size=2,
            cluster_centers=np.vstack([
                np.zeros((1, 32), dtype=np.float32),
                np.full((1, 32), 255.0, dtype=np.float32),
            ]),
        )

    def _write_feature_npz(self, path: Path, method: str, descriptors: np.ndarray | None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        num_keypoints = 0 if descriptors is None else int(descriptors.shape[0])
        keypoints = np.zeros((num_keypoints, 7), dtype=np.float32)
        descriptors_present = descriptors is not None
        descriptors_to_save = descriptors if descriptors is not None else np.empty((0, 0), dtype=np.float32)

        np.savez_compressed(
            path,
            sample_id=np.array(path.stem),
            method=np.array(method),
            num_keypoints=np.array(num_keypoints, dtype=np.int32),
            keypoints=keypoints,
            descriptors=descriptors_to_save,
            descriptors_present=np.array(int(descriptors_present), dtype=np.uint8),
            descriptor_shape=np.array(() if descriptors is None else descriptors.shape, dtype=np.int32),
            descriptor_dtype=np.array("" if descriptors is None else str(descriptors.dtype)),
            keypoint_fields=np.array(("x", "y", "size", "angle", "response", "octave", "class_id")),
        )
        return path


if __name__ == "__main__":
    unittest.main()
