from __future__ import annotations

import unittest

import numpy as np

from src.encoding import EncodedFeatureResult
from src.indexing import build_inverted_index_from_records, score_query_against_index, search_inverted_index


class IndexingInvertedIndexTests(unittest.TestCase):
    def test_build_index_and_search_paths(self) -> None:
        records = [
            self._make_record("doc1", [20, 0]),
            self._make_record("doc2", [1, 1]),
            self._make_record("doc3", [1, 0]),
            self._make_record("doc4", [0, 0], descriptors_present=False),
        ]
        bundle = build_inverted_index_from_records(records, corpus_split="gallery", weighting_modes=("tf", "tfidf"))
        query = self._make_record("query", [1, 1])

        self.assertEqual(bundle.method, "SIFT")
        self.assertEqual(bundle.corpus_split, "gallery")
        np.testing.assert_array_equal(bundle.tfidf_stats.df, np.array([3, 1], dtype=np.int32))
        self.assertIn("tf", bundle.indexes)
        self.assertIn("tfidf", bundle.indexes)

        tf_index = bundle.indexes["tf"]
        np.testing.assert_array_equal(tf_index.indptr, np.array([0, 3, 4], dtype=np.int32))
        np.testing.assert_array_equal(tf_index.indices, np.array([0, 1, 2, 1], dtype=np.int32))
        np.testing.assert_allclose(tf_index.data, np.array([20.0, 1.0, 1.0, 1.0], dtype=np.float32))
        self.assertEqual(float(tf_index.doc_norms[3]), 0.0)

        tf_scores = score_query_against_index(query, tf_index, scoring_mode="dot")
        tfidf_scores = score_query_against_index(
            query,
            bundle.indexes["tfidf"],
            tfidf_stats=bundle.tfidf_stats,
            scoring_mode="cosine",
        )

        tf_results = search_inverted_index(query, tf_index, scoring_mode="dot", top_k=4)
        tfidf_results = search_inverted_index(
            query,
            bundle.indexes["tfidf"],
            tfidf_stats=bundle.tfidf_stats,
            scoring_mode="cosine",
            top_k=4,
        )

        np.testing.assert_allclose(tf_scores, np.array([20.0, 2.0, 1.0, 0.0], dtype=np.float64))
        self.assertEqual(tf_results[0].sample_id, "doc1")
        self.assertEqual(tfidf_results[0].sample_id, "doc2")
        self.assertNotEqual(tf_results[0].sample_id, tfidf_results[0].sample_id)
        self.assertGreater(tf_results[0].score, tf_results[1].score)
        self.assertGreater(tfidf_results[0].score, tfidf_results[1].score)

    def test_reject_normalized_encoded_artifact(self) -> None:
        records = [self._make_record("doc1", [1, 0], normalized=True, histogram_dtype="float32")]

        with self.assertRaises(ValueError) as exc:
            build_inverted_index_from_records(records, corpus_split="gallery")

        self.assertIn("normalized BoW artifacts are not supported", str(exc.exception))

    def test_reject_histogram_dimension_mismatch(self) -> None:
        records = [
            self._make_record("doc1", [1, 0]),
            self._make_record("doc2", [1, 0, 1], num_visual_words=3),
        ]

        with self.assertRaises(ValueError) as exc:
            build_inverted_index_from_records(records, corpus_split="gallery")

        self.assertIn("histogram size mismatch", str(exc.exception))

    def test_reject_method_mismatch_in_search(self) -> None:
        records = [self._make_record("doc1", [1, 0])]
        bundle = build_inverted_index_from_records(records, corpus_split="gallery")
        query = self._make_record("query", [1, 0], method="ORB", descriptor_dim=32)

        with self.assertRaises(ValueError) as exc:
            search_inverted_index(query, bundle.indexes["tf"], scoring_mode="dot")

        self.assertIn("method mismatch", str(exc.exception))

    def test_reject_query_dimension_mismatch(self) -> None:
        records = [self._make_record("doc1", [1, 0])]
        bundle = build_inverted_index_from_records(records, corpus_split="gallery")
        query = self._make_record("query", [1, 0, 1], num_visual_words=3)

        with self.assertRaises(ValueError) as exc:
            search_inverted_index(query, bundle.indexes["tf"], scoring_mode="dot")

        self.assertIn("histogram size mismatch", str(exc.exception))

    def test_reject_missing_tfidf_stats_for_cosine(self) -> None:
        records = [self._make_record("doc1", [1, 0]), self._make_record("doc2", [1, 1])]
        bundle = build_inverted_index_from_records(records, corpus_split="gallery")
        query = self._make_record("query", [1, 1])

        with self.assertRaises(ValueError) as exc:
            search_inverted_index(query, bundle.indexes["tfidf"], scoring_mode="cosine")

        self.assertIn("tfidf_stats is required", str(exc.exception))

    def _make_record(
        self,
        sample_id: str,
        histogram_values: list[int],
        *,
        method: str = "SIFT",
        normalized: bool = False,
        num_visual_words: int | None = None,
        descriptors_present: bool | None = None,
        histogram_dtype: str = "int32",
        descriptor_dim: int | None = None,
    ) -> EncodedFeatureResult:
        histogram = np.asarray(histogram_values, dtype=np.int32 if histogram_dtype == "int32" else np.float32)
        resolved_num_visual_words = len(histogram_values) if num_visual_words is None else num_visual_words
        resolved_descriptors_present = bool(np.sum(histogram_values) > 0) if descriptors_present is None else descriptors_present
        resolved_descriptor_dim = 128 if method == "SIFT" else 32 if descriptor_dim is None else descriptor_dim
        descriptors = None
        if resolved_descriptors_present:
            descriptor_dtype = np.float32 if method == "SIFT" else np.uint8
            descriptors = np.zeros((int(np.sum(histogram_values)), resolved_descriptor_dim), dtype=descriptor_dtype)

        return EncodedFeatureResult(
            sample_id=sample_id,
            method=method,
            encoding_type="bow",
            num_visual_words=resolved_num_visual_words,
            histogram=histogram,
            histogram_dtype=histogram_dtype,
            num_descriptors=int(np.sum(histogram_values)),
            descriptors_present=resolved_descriptors_present,
            codebook_path="outputs/indices/codebooks/codebook_sift_k2_minibatch_kmeans.npz",
            normalized=normalized,
        )


if __name__ == "__main__":
    unittest.main()

