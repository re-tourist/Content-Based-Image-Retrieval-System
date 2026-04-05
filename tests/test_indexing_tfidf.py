from __future__ import annotations

import unittest

import numpy as np

from src.indexing import apply_tfidf_weighting, compute_document_frequency, compute_idf, normalize_vector


class IndexingTfidfTests(unittest.TestCase):
    def test_compute_document_frequency_and_idf(self) -> None:
        histograms = np.array(
            [
                [1, 0, 2],
                [0, 3, 0],
                [4, 0, 5],
            ],
            dtype=np.int32,
        )

        df = compute_document_frequency(histograms)
        idf = compute_idf(df, num_docs=3)

        np.testing.assert_array_equal(df, np.array([2, 1, 2], dtype=np.int32))
        np.testing.assert_allclose(
            idf,
            np.array(
                [
                    np.log(4.0 / 3.0) + 1.0,
                    np.log(4.0 / 2.0) + 1.0,
                    np.log(4.0 / 3.0) + 1.0,
                ],
                dtype=np.float32,
            ),
        )

    def test_apply_tfidf_weighting(self) -> None:
        histogram = np.array([2, 0, 1], dtype=np.int32)
        idf = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        weighted = apply_tfidf_weighting(histogram, idf)

        np.testing.assert_allclose(weighted, np.array([2.0, 0.0, 3.0], dtype=np.float32))

    def test_normalize_vector(self) -> None:
        vector = np.array([3.0, 4.0], dtype=np.float32)

        l2_normalized = normalize_vector(vector, mode="l2")
        none_normalized = normalize_vector(vector, mode="none")

        np.testing.assert_allclose(l2_normalized, np.array([0.6, 0.8], dtype=np.float32))
        np.testing.assert_allclose(none_normalized, np.array([3.0, 4.0], dtype=np.float32))

    def test_zero_histogram_document_frequency(self) -> None:
        histograms = np.array(
            [
                [0, 0],
                [0, 0],
            ],
            dtype=np.int32,
        )

        df = compute_document_frequency(histograms)
        idf = compute_idf(df, num_docs=2)

        np.testing.assert_array_equal(df, np.array([0, 0], dtype=np.int32))
        np.testing.assert_allclose(
            idf,
            np.array(
                [
                    np.log(3.0 / 1.0) + 1.0,
                    np.log(3.0 / 1.0) + 1.0,
                ],
                dtype=np.float32,
            ),
        )


if __name__ == "__main__":
    unittest.main()

