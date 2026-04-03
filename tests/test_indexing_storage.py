from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.indexing import (
    InvertedIndexArtifact,
    TfidfStatsArtifact,
    load_inverted_index,
    load_tfidf_stats,
    resolve_inverted_index_path,
    resolve_method_index_dir,
    resolve_tfidf_stats_path,
    save_inverted_index,
    save_tfidf_stats,
)


class IndexingStorageTests(unittest.TestCase):
    def test_tfidf_stats_and_index_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            method_dir = resolve_method_index_dir(root, "gallery", "SIFT")
            tfidf_stats = TfidfStatsArtifact(
                method="SIFT",
                corpus_split="gallery",
                num_docs=3,
                num_visual_words=3,
                df=np.array([2, 1, 0], dtype=np.int32),
                idf=np.array([1.1, 1.2, 1.3], dtype=np.float32),
            )
            index = InvertedIndexArtifact(
                method="SIFT",
                corpus_split="gallery",
                weighting_mode="tfidf",
                num_docs=3,
                num_visual_words=3,
                sample_ids=("doc1", "doc2", "doc3"),
                indptr=np.array([0, 2, 3, 3], dtype=np.int32),
                indices=np.array([0, 2, 1], dtype=np.int32),
                data=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                doc_norms=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            )

            tfidf_stats_path = save_tfidf_stats(tfidf_stats, method_dir)
            index_path = save_inverted_index(index, method_dir)

            self.assertEqual(tfidf_stats_path, resolve_tfidf_stats_path(method_dir))
            self.assertEqual(index_path, resolve_inverted_index_path(method_dir, "tfidf"))

            loaded_stats = load_tfidf_stats(tfidf_stats_path)
            loaded_index = load_inverted_index(index_path)

            self.assertEqual(loaded_stats.method, "SIFT")
            self.assertEqual(loaded_stats.corpus_split, "gallery")
            self.assertEqual(loaded_stats.num_docs, 3)
            np.testing.assert_array_equal(loaded_stats.df, np.array([2, 1, 0], dtype=np.int32))
            np.testing.assert_allclose(loaded_stats.idf, np.array([1.1, 1.2, 1.3], dtype=np.float32))

            self.assertEqual(loaded_index.method, "SIFT")
            self.assertEqual(loaded_index.corpus_split, "gallery")
            self.assertEqual(loaded_index.weighting_mode, "tfidf")
            self.assertEqual(loaded_index.sample_ids, ("doc1", "doc2", "doc3"))
            np.testing.assert_array_equal(loaded_index.indptr, np.array([0, 2, 3, 3], dtype=np.int32))
            np.testing.assert_array_equal(loaded_index.indices, np.array([0, 2, 1], dtype=np.int32))
            np.testing.assert_allclose(loaded_index.data, np.array([1.0, 2.0, 3.0], dtype=np.float32))
            np.testing.assert_allclose(loaded_index.doc_norms, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_missing_file_errors_are_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                load_tfidf_stats(root / "missing_tfidf_stats.npz")
            with self.assertRaises(FileNotFoundError):
                load_inverted_index(root / "missing_index.npz")


if __name__ == "__main__":
    unittest.main()

