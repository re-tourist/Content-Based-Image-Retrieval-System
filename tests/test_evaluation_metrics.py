from __future__ import annotations

import unittest

import numpy as np

from src.evaluation import (
    build_macro_pr_curve,
    build_prefix_pr_curve,
    compute_average_precision,
    compute_precision_at_k,
    compute_recall_at_k,
    extract_class_label,
    normalize_canonical_sample_id,
)
from src.evaluation.types import RetrievedResult


class EvaluationMetricsTests(unittest.TestCase):
    def test_precision_recall_ap_and_pr_curve(self) -> None:
        flags = [True, False, True, False]

        self.assertAlmostEqual(compute_precision_at_k(flags, 3), 2 / 3)
        self.assertAlmostEqual(compute_recall_at_k(flags, 3, relevant_total=2), 1.0)
        self.assertAlmostEqual(compute_average_precision(flags, relevant_total=2), (1.0 + (2 / 3)) / 2)

        ranked_results = [
            RetrievedResult(rank=1, sample_id="test/A/img1.jpg", artifact_sample_id="A/img1.jpg", label="A", score=3.0, relevant=True),
            RetrievedResult(rank=2, sample_id="test/B/img2.jpg", artifact_sample_id="B/img2.jpg", label="B", score=2.0, relevant=False),
            RetrievedResult(rank=3, sample_id="test/A/img3.jpg", artifact_sample_id="A/img3.jpg", label="A", score=1.0, relevant=True),
            RetrievedResult(rank=4, sample_id="test/C/img4.jpg", artifact_sample_id="C/img4.jpg", label="C", score=0.0, relevant=False),
        ]

        curve_points = build_prefix_pr_curve(ranked_results, relevant_total=2)
        self.assertEqual(len(curve_points), 4)
        self.assertAlmostEqual(curve_points[0].precision, 1.0)
        self.assertAlmostEqual(curve_points[0].recall, 0.5)
        self.assertAlmostEqual(curve_points[2].precision, 2 / 3)
        self.assertAlmostEqual(curve_points[2].recall, 1.0)

    def test_macro_pr_curve_ignores_empty_queries(self) -> None:
        query_curve = build_prefix_pr_curve(
            [
                RetrievedResult(rank=1, sample_id="test/A/img1.jpg", artifact_sample_id="A/img1.jpg", label="A", score=3.0, relevant=True),
                RetrievedResult(rank=2, sample_id="test/B/img2.jpg", artifact_sample_id="B/img2.jpg", label="B", score=2.0, relevant=False),
            ],
            relevant_total=1,
        )

        summary_curve = build_macro_pr_curve([query_curve, []], recall_grid=np.array([0.0, 0.5, 1.0], dtype=np.float32))

        self.assertEqual(len(summary_curve), 3)
        self.assertEqual(summary_curve[0].support, 1)
        self.assertAlmostEqual(summary_curve[0].precision, 1.0)
        self.assertAlmostEqual(summary_curve[1].precision, 1.0)
        self.assertAlmostEqual(summary_curve[2].precision, 1.0)

    def test_canonical_sample_id_normalization(self) -> None:
        self.assertEqual(normalize_canonical_sample_id("test/A0C573/img.jpg"), "A0C573/img.jpg")
        self.assertEqual(normalize_canonical_sample_id("data/test/A0C573/img.jpg"), "A0C573/img.jpg")
        self.assertEqual(extract_class_label("gallery/A1B2C3/foo/bar.jpg"), "A1B2C3")

    def test_no_relevant_items_return_none_metrics(self) -> None:
        flags = [False, False, False]

        self.assertIsNone(compute_recall_at_k(flags, 2, relevant_total=0))
        self.assertIsNone(compute_average_precision(flags, relevant_total=0))
        self.assertEqual(build_prefix_pr_curve([], relevant_total=0), [])


if __name__ == "__main__":
    unittest.main()

