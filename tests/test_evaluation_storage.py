from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation import (
    PRCurvePoint,
    QueryEvaluation,
    QueryMetrics,
    RetrievalEvalRunManifest,
    RetrievalSummary,
    RetrievedResult,
    SummaryPRCurvePoint,
    VariantEvaluation,
    save_run_manifest,
    save_variant_evaluation,
)


class EvaluationStorageTests(unittest.TestCase):
    def test_variant_and_manifest_roundtrip_layout(self) -> None:
        variant = self._make_variant()
        manifest = self._make_manifest()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variant_dir = root / "gallery" / "SIFT" / "tfidf_cosine"
            saved_paths = save_variant_evaluation(variant, variant_dir)
            manifest_path = save_run_manifest(manifest, root / "gallery")

            self.assertTrue((variant_dir / "variant.json").is_file())
            self.assertTrue((variant_dir / "per_query_results.json").is_file())
            self.assertTrue((variant_dir / "per_query_metrics.json").is_file())
            self.assertTrue((variant_dir / "summary_metrics.json").is_file())
            self.assertTrue((variant_dir / "pr_curve.json").is_file())
            self.assertTrue(manifest_path.is_file())
            self.assertIn("variant.json", saved_paths)

            with (variant_dir / "per_query_results.json").open("r", encoding="utf-8") as handle:
                per_query_results = json.load(handle)
            with (variant_dir / "summary_metrics.json").open("r", encoding="utf-8") as handle:
                summary_metrics = json.load(handle)
            with (variant_dir / "pr_curve.json").open("r", encoding="utf-8") as handle:
                pr_curve = json.load(handle)
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest_json = json.load(handle)

            self.assertEqual(len(per_query_results), 1)
            self.assertEqual(per_query_results[0]["query_sample_id"], "test/A/img1.jpg")
            self.assertEqual(per_query_results[0]["ranked_results"][0]["sample_id"], "test/A/img2.jpg")
            self.assertEqual(summary_metrics["method"], "SIFT")
            self.assertEqual(pr_curve["variant_name"], "tfidf_cosine")
            self.assertEqual(len(pr_curve["summary_pr_curve"]), 2)
            self.assertEqual(manifest_json["corpus_split"], "gallery")

    def _make_variant(self) -> VariantEvaluation:
        query_metrics = QueryMetrics(
            query_sample_id="test/A/img1.jpg",
            query_artifact_sample_id="A/img1.jpg",
            query_label="A",
            method="SIFT",
            weighting_mode="tfidf",
            scoring_mode="cosine",
            requested_top_k=2,
            effective_top_k=2,
            gallery_count=2,
            relevant_total=1,
            retrieved_relevant_at_k=1,
            precision_at_k=0.5,
            recall_at_k=1.0,
            average_precision=1.0,
            included_in_mean_recall=True,
            included_in_mean_average_precision=True,
            retrieval_time_ms=1.23,
        )
        query_eval = QueryEvaluation(
            query_sample_id="test/A/img1.jpg",
            query_artifact_sample_id="A/img1.jpg",
            query_label="A",
            method="SIFT",
            weighting_mode="tfidf",
            scoring_mode="cosine",
            requested_top_k=2,
            effective_top_k=2,
            ranked_results=[
                RetrievedResult(
                    rank=1,
                    sample_id="test/A/img2.jpg",
                    artifact_sample_id="A/img2.jpg",
                    label="A",
                    score=0.9,
                    relevant=True,
                    doc_index=0,
                )
            ],
            metrics=query_metrics,
            pr_curve_points=[
                PRCurvePoint(
                    rank=1,
                    precision=1.0,
                    recall=1.0,
                    relevant=True,
                    sample_id="test/A/img2.jpg",
                    artifact_sample_id="A/img2.jpg",
                )
            ],
        )
        summary = RetrievalSummary(
            method="SIFT",
            weighting_mode="tfidf",
            scoring_mode="cosine",
            corpus_split="gallery",
            gallery_split="gallery",
            query_split="query",
            requested_top_k=2,
            effective_top_k=2,
            gallery_count=2,
            query_count=1,
            query_count_with_relevant_items=1,
            query_count_without_relevant_items=0,
            mean_precision_at_k=0.5,
            mean_recall_at_k=1.0,
            mean_average_precision=1.0,
            total_retrieval_time_ms=1.23,
            avg_retrieval_time_ms=1.23,
            relevance_definition="same class-folder name after canonical sample-id normalization",
            canonical_sample_id_rule="strip known dataset prefixes",
        )
        return VariantEvaluation(
            variant_name="tfidf_cosine",
            method="SIFT",
            weighting_mode="tfidf",
            scoring_mode="cosine",
            corpus_split="gallery",
            gallery_split="gallery",
            query_split="query",
            requested_top_k=2,
            effective_top_k=2,
            gallery_count=2,
            query_count=1,
            query_results=[query_eval],
            summary=summary,
            summary_pr_curve=[
                SummaryPRCurvePoint(recall=0.0, precision=1.0, support=1),
                SummaryPRCurvePoint(recall=1.0, precision=1.0, support=1),
            ],
        )

    def _make_manifest(self) -> RetrievalEvalRunManifest:
        return RetrievalEvalRunManifest(
            gallery_split_file="data/splits/gallery.txt",
            query_split_file="data/splits/query.txt",
            data_root="data",
            encoded_dir="outputs/encoded",
            index_root="outputs/indices/inverted",
            output_root="outputs/evaluations/retrieval",
            corpus_split="gallery",
            methods=("SIFT",),
            compare=True,
            requested_top_k=10,
            weighting_mode="tfidf",
            scoring_mode="cosine",
            rebuild_index=False,
        )


if __name__ == "__main__":
    unittest.main()
