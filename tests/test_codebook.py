from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from src.encoding import (
    load_codebook_artifact,
    sample_descriptors_by_method,
    save_codebook_artifact,
    train_codebook,
)


class CodebookTests(unittest.TestCase):
    def test_sampling_skips_empty_records_and_applies_cap(self) -> None:
        feature_dir = self._make_temp_dir("features")
        self._write_feature_npz(feature_dir / "empty_sift.npz", method="SIFT", descriptors=None)
        self._write_feature_npz(
            feature_dir / "non_empty_sift.npz",
            method="SIFT",
            descriptors=np.arange(3 * 128, dtype=np.float32).reshape(3, 128),
        )

        samples = sample_descriptors_by_method(feature_dir, max_descriptors=2, random_state=0)

        self.assertEqual(set(samples), {"SIFT"})
        sift_sample = samples["SIFT"]
        self.assertEqual(sift_sample.feature_file_count, 2)
        self.assertEqual(sift_sample.empty_record_count, 1)
        self.assertEqual(sift_sample.total_descriptor_count, 3)
        self.assertEqual(sift_sample.sampled_descriptor_count, 2)
        self.assertEqual(sift_sample.descriptors.shape, (2, 128))
        self.assertEqual(str(sift_sample.descriptors.dtype), "float32")

    def test_sampling_groups_methods_separately(self) -> None:
        feature_dir = self._make_temp_dir("features")
        self._write_feature_npz(
            feature_dir / "sample_sift.npz",
            method="SIFT",
            descriptors=np.ones((2, 128), dtype=np.float32),
        )
        self._write_feature_npz(
            feature_dir / "sample_orb.npz",
            method="ORB",
            descriptors=np.full((4, 32), 9, dtype=np.uint8),
        )

        samples = sample_descriptors_by_method(feature_dir, max_descriptors=10, random_state=0)

        self.assertEqual(set(samples), {"ORB", "SIFT"})
        self.assertEqual(samples["SIFT"].descriptors.shape, (2, 128))
        self.assertEqual(str(samples["SIFT"].descriptors.dtype), "float32")
        self.assertEqual(samples["ORB"].descriptors.shape, (4, 32))
        self.assertEqual(str(samples["ORB"].descriptors.dtype), "uint8")

    def test_train_codebook_on_sift_descriptors(self) -> None:
        descriptors = self._make_sift_descriptors()

        codebook = train_codebook(
            descriptors,
            method="SIFT",
            trainer="minibatch_kmeans",
            n_clusters=2,
            batch_size=2,
            random_state=0,
        )

        self.assertEqual(codebook.method, "SIFT")
        self.assertEqual(codebook.n_clusters, 2)
        self.assertEqual(codebook.descriptor_dim, 128)
        self.assertEqual(codebook.descriptor_dtype, "float32")
        self.assertEqual(codebook.cluster_centers.shape, (2, 128))
        self.assertEqual(str(codebook.cluster_centers.dtype), "float32")

    def test_train_codebook_on_orb_descriptors(self) -> None:
        descriptors = self._make_orb_descriptors()

        codebook = train_codebook(
            descriptors,
            method="ORB",
            trainer="minibatch_kmeans",
            n_clusters=2,
            batch_size=2,
            random_state=0,
        )

        self.assertEqual(codebook.method, "ORB")
        self.assertEqual(codebook.n_clusters, 2)
        self.assertEqual(codebook.descriptor_dim, 32)
        self.assertEqual(codebook.descriptor_dtype, "uint8")
        self.assertEqual(codebook.cluster_centers.shape, (2, 32))

    def test_saved_codebook_can_be_loaded(self) -> None:
        output_dir = self._make_temp_dir("indices")
        codebook = train_codebook(
            self._make_sift_descriptors(),
            method="SIFT",
            trainer="minibatch_kmeans",
            n_clusters=2,
            batch_size=2,
            random_state=0,
        )

        save_path = save_codebook_artifact(codebook, output_dir)
        loaded = load_codebook_artifact(save_path)

        self.assertEqual(loaded.method, "SIFT")
        self.assertEqual(loaded.trainer, "minibatch_kmeans")
        self.assertEqual(loaded.n_clusters, 2)
        self.assertEqual(loaded.descriptor_dim, 128)
        self.assertEqual(loaded.training_descriptor_count, codebook.training_descriptor_count)
        self.assertEqual(loaded.cluster_centers.shape, (2, 128))

    def test_inconsistent_descriptor_dim_raises(self) -> None:
        with self.assertRaises(ValueError):
            train_codebook(
                np.ones((4, 64), dtype=np.float32),
                method="SIFT",
                trainer="minibatch_kmeans",
                n_clusters=2,
                batch_size=2,
                random_state=0,
            )

    def _make_temp_dir(self, kind: str) -> Path:
        root = Path("outputs") / kind / "test_tmp"
        root.mkdir(parents=True, exist_ok=True)
        temp_dir = root / uuid.uuid4().hex
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir

    def _make_sift_descriptors(self) -> np.ndarray:
        cluster_a = np.zeros((3, 128), dtype=np.float32)
        cluster_b = np.full((3, 128), 5.0, dtype=np.float32)
        return np.vstack([cluster_a, cluster_b])

    def _make_orb_descriptors(self) -> np.ndarray:
        cluster_a = np.zeros((3, 32), dtype=np.uint8)
        cluster_b = np.full((3, 32), 255, dtype=np.uint8)
        return np.vstack([cluster_a, cluster_b])

    def _write_feature_npz(self, path: Path, method: str, descriptors: np.ndarray | None) -> Path:
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
