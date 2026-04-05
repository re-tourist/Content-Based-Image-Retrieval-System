from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from src.encoding import (
    build_encoding_input_from_feature_record,
    build_encoding_input_from_local_feature,
    load_feature_npz,
)
from src.features.local import KEYPOINT_FIELDS, LocalFeatureResult


class EncodingIoTests(unittest.TestCase):
    def test_load_valid_sift_feature_npz(self) -> None:
        temp_dir = self._make_temp_dir()
        feature_path = self._write_feature_npz(
            temp_dir / "sample_sift.npz",
            method="SIFT",
            descriptors=np.ones((3, 128), dtype=np.float32),
        )

        record = load_feature_npz(feature_path)
        encoding_input = build_encoding_input_from_feature_record(record)

        self.assertEqual(record.method, "SIFT")
        self.assertTrue(record.descriptors_present)
        self.assertEqual(record.descriptor_shape, (3, 128))
        self.assertEqual(record.descriptor_dtype, "float32")
        self.assertEqual(encoding_input.descriptor_shape, (3, 128))
        self.assertEqual(encoding_input.descriptor_dtype, "float32")
        self.assertEqual(encoding_input.descriptor_dim, 128)
        self.assertEqual(encoding_input.num_descriptors, 3)

    def test_load_valid_orb_feature_npz(self) -> None:
        temp_dir = self._make_temp_dir()
        feature_path = self._write_feature_npz(
            temp_dir / "sample_orb.npz",
            method="ORB",
            descriptors=np.full((2, 32), 7, dtype=np.uint8),
        )

        record = load_feature_npz(feature_path)
        encoding_input = build_encoding_input_from_feature_record(record)

        self.assertEqual(record.method, "ORB")
        self.assertTrue(record.descriptors_present)
        self.assertEqual(record.descriptor_shape, (2, 32))
        self.assertEqual(record.descriptor_dtype, "uint8")
        self.assertEqual(encoding_input.descriptor_shape, (2, 32))
        self.assertEqual(encoding_input.descriptor_dtype, "uint8")
        self.assertEqual(encoding_input.descriptor_dim, 32)
        self.assertEqual(encoding_input.num_descriptors, 2)

    def test_load_empty_descriptor_feature_npz(self) -> None:
        temp_dir = self._make_temp_dir()
        feature_path = self._write_feature_npz(
            temp_dir / "sample_empty.npz",
            method="SIFT",
            descriptors=None,
        )

        record = load_feature_npz(feature_path)
        encoding_input = build_encoding_input_from_feature_record(record)

        self.assertFalse(record.descriptors_present)
        self.assertIsNone(record.descriptors)
        self.assertEqual(record.descriptor_shape, ())
        self.assertIsNone(record.descriptor_dtype)
        self.assertFalse(encoding_input.descriptors_present)
        self.assertIsNone(encoding_input.descriptors)
        self.assertEqual(encoding_input.descriptor_shape, (0, 128))
        self.assertEqual(encoding_input.descriptor_dtype, "float32")
        self.assertEqual(encoding_input.num_descriptors, 0)

    def test_build_encoding_input_from_local_feature(self) -> None:
        descriptors = np.arange(256, dtype=np.float32).reshape(2, 128)
        feature_result = LocalFeatureResult(
            keypoints=self._make_keypoints(2),
            descriptors=descriptors,
            meta={
                "method": "SIFT",
                "num_keypoints": 2,
                "descriptor_shape": (2, 128),
                "descriptor_dtype": "float32",
            },
        )

        encoding_input = build_encoding_input_from_local_feature(
            feature_result,
            sample_id="train/sample.jpg",
        )

        self.assertEqual(encoding_input.sample_id, "train/sample.jpg")
        self.assertEqual(encoding_input.method, "SIFT")
        self.assertTrue(encoding_input.descriptors_present)
        self.assertEqual(encoding_input.descriptor_shape, (2, 128))
        self.assertEqual(encoding_input.descriptor_dtype, "float32")
        self.assertEqual(encoding_input.num_descriptors, 2)

    def test_detect_invalid_metadata(self) -> None:
        temp_dir = self._make_temp_dir()
        feature_path = self._write_feature_npz(
            temp_dir / "invalid_dtype.npz",
            method="ORB",
            descriptors=np.ones((2, 32), dtype=np.uint8),
            descriptor_dtype="float32",
        )

        with self.assertRaises(ValueError):
            load_feature_npz(feature_path)

        invalid_local_result = LocalFeatureResult(
            keypoints=self._make_keypoints(1),
            descriptors=np.ones((1, 64), dtype=np.float32),
            meta={
                "method": "SIFT",
                "num_keypoints": 1,
                "descriptor_shape": (1, 64),
                "descriptor_dtype": "float32",
            },
        )

        with self.assertRaises(ValueError):
            build_encoding_input_from_local_feature(invalid_local_result)

    def _make_temp_dir(self) -> Path:
        temp_root = Path("outputs/features/test_tmp")
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = temp_root / uuid.uuid4().hex
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir

    def _write_feature_npz(
        self,
        path: Path,
        method: str,
        descriptors: np.ndarray | None,
        descriptor_shape: tuple[int, ...] | None = None,
        descriptor_dtype: str | None = None,
    ) -> Path:
        num_keypoints = 0 if descriptors is None else int(descriptors.shape[0])
        keypoints = self._make_keypoints(num_keypoints)
        descriptors_present = descriptors is not None
        descriptors_to_save = descriptors if descriptors is not None else np.empty((0, 0), dtype=np.float32)

        np.savez_compressed(
            path,
            sample_id=np.array("sample/image.jpg"),
            method=np.array(method),
            num_keypoints=np.array(num_keypoints, dtype=np.int32),
            keypoints=keypoints,
            descriptors=descriptors_to_save,
            descriptors_present=np.array(int(descriptors_present), dtype=np.uint8),
            descriptor_shape=np.array(
                descriptor_shape if descriptor_shape is not None else (() if descriptors is None else descriptors.shape),
                dtype=np.int32,
            ),
            descriptor_dtype=np.array(
                descriptor_dtype if descriptor_dtype is not None else ("" if descriptors is None else str(descriptors.dtype))
            ),
        )
        return path

    def _make_keypoints(self, count: int) -> np.ndarray:
        if count == 0:
            return np.empty((0, len(KEYPOINT_FIELDS)), dtype=np.float32)
        return np.arange(count * len(KEYPOINT_FIELDS), dtype=np.float32).reshape(count, len(KEYPOINT_FIELDS))


if __name__ == "__main__":
    unittest.main()
