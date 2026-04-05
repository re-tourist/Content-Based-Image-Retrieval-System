from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from src.encoding import (
    CodebookArtifact,
    EncodingInput,
    encode_bow_from_input,
    load_encoded_feature,
    save_encoded_feature,
)


class BowTests(unittest.TestCase):
    def test_bow_encoding_sift_descriptors(self) -> None:
        codebook = self._make_sift_codebook()
        encoding_input = EncodingInput(
            sample_id="sample/sift.jpg",
            method="SIFT",
            descriptors=np.vstack([
                np.zeros((2, 128), dtype=np.float32),
                np.full((3, 128), 10.0, dtype=np.float32),
            ]),
            descriptors_present=True,
            descriptor_shape=(5, 128),
            descriptor_dtype="float32",
            descriptor_dim=128,
            num_descriptors=5,
        )

        result = encode_bow_from_input(encoding_input, codebook, normalize=False)

        self.assertEqual(result.method, "SIFT")
        self.assertEqual(result.encoding_type, "bow")
        self.assertEqual(result.num_visual_words, 2)
        self.assertEqual(result.histogram.tolist(), [2, 3])
        self.assertEqual(result.histogram_dtype, "int32")
        self.assertEqual(result.num_descriptors, 5)
        self.assertTrue(result.descriptors_present)

    def test_bow_encoding_orb_descriptors(self) -> None:
        codebook = self._make_orb_codebook()
        encoding_input = EncodingInput(
            sample_id="sample/orb.jpg",
            method="ORB",
            descriptors=np.vstack([
                np.zeros((1, 32), dtype=np.uint8),
                np.full((4, 32), 255, dtype=np.uint8),
            ]),
            descriptors_present=True,
            descriptor_shape=(5, 32),
            descriptor_dtype="uint8",
            descriptor_dim=32,
            num_descriptors=5,
        )

        result = encode_bow_from_input(encoding_input, codebook, normalize=False)

        self.assertEqual(result.method, "ORB")
        self.assertEqual(result.histogram.tolist(), [1, 4])
        self.assertEqual(result.histogram_dtype, "int32")

    def test_empty_descriptors_produce_zero_histogram(self) -> None:
        codebook = self._make_sift_codebook()
        encoding_input = EncodingInput(
            sample_id="sample/empty.jpg",
            method="SIFT",
            descriptors=None,
            descriptors_present=False,
            descriptor_shape=(0, 128),
            descriptor_dtype="float32",
            descriptor_dim=128,
            num_descriptors=0,
        )

        result = encode_bow_from_input(encoding_input, codebook, normalize=False)

        self.assertEqual(result.histogram.tolist(), [0, 0])
        self.assertEqual(result.num_visual_words, 2)
        self.assertFalse(result.descriptors_present)
        self.assertEqual(result.num_descriptors, 0)

    def test_mismatched_method_raises(self) -> None:
        encoding_input = EncodingInput(
            sample_id="sample/mismatch.jpg",
            method="SIFT",
            descriptors=np.ones((1, 128), dtype=np.float32),
            descriptors_present=True,
            descriptor_shape=(1, 128),
            descriptor_dtype="float32",
            descriptor_dim=128,
            num_descriptors=1,
        )

        with self.assertRaises(ValueError):
            encode_bow_from_input(encoding_input, self._make_orb_codebook())

    def test_mismatched_descriptor_dim_raises(self) -> None:
        encoding_input = EncodingInput(
            sample_id="sample/bad_dim.jpg",
            method="SIFT",
            descriptors=np.ones((1, 64), dtype=np.float32),
            descriptors_present=True,
            descriptor_shape=(1, 64),
            descriptor_dtype="float32",
            descriptor_dim=64,
            num_descriptors=1,
        )

        with self.assertRaises(ValueError):
            encode_bow_from_input(encoding_input, self._make_sift_codebook())

    def test_encoded_artifact_roundtrip(self) -> None:
        output_dir = self._make_temp_dir()
        codebook = self._make_sift_codebook()
        encoding_input = EncodingInput(
            sample_id="sample/roundtrip.jpg",
            method="SIFT",
            descriptors=np.vstack([
                np.zeros((1, 128), dtype=np.float32),
                np.full((1, 128), 10.0, dtype=np.float32),
            ]),
            descriptors_present=True,
            descriptor_shape=(2, 128),
            descriptor_dtype="float32",
            descriptor_dim=128,
            num_descriptors=2,
        )

        result = encode_bow_from_input(
            encoding_input,
            codebook,
            normalize=False,
            codebook_path="outputs/indices/codebooks/codebook_sift_k2_minibatch_kmeans.npz",
        )
        save_path = save_encoded_feature(result, output_dir)
        loaded = load_encoded_feature(save_path)

        self.assertEqual(loaded.sample_id, "sample/roundtrip.jpg")
        self.assertEqual(loaded.method, "SIFT")
        self.assertEqual(loaded.encoding_type, "bow")
        self.assertEqual(loaded.num_visual_words, 2)
        self.assertEqual(loaded.histogram.tolist(), [1, 1])
        self.assertEqual(loaded.histogram_dtype, "int32")
        self.assertTrue(loaded.descriptors_present)
        self.assertEqual(loaded.num_descriptors, 2)
        self.assertEqual(
            loaded.codebook_path,
            "outputs/indices/codebooks/codebook_sift_k2_minibatch_kmeans.npz",
        )

    def _make_temp_dir(self) -> Path:
        root = Path("outputs") / "encoded" / "test_tmp"
        root.mkdir(parents=True, exist_ok=True)
        temp_dir = root / uuid.uuid4().hex
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir

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


if __name__ == "__main__":
    unittest.main()
