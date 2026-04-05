from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_retrieval_eval import main as run_retrieval_eval_main
from src.evaluation import evaluate_canonical_retrieval, load_canonical_split_samples
from src.encoding import EncodedFeatureResult
from src.encoding import save_encoded_feature


class RetrievalEvaluationTests(unittest.TestCase):
    def test_cli_compare_run_exports_variants_and_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root, gallery_split, query_split = self._prepare_dataset(root, query_label="A")
            encoded_dir = root / "encoded"
            index_root = root / "indices"
            output_root = root / "evaluations"

            self._save_encoded_artifacts(encoded_dir, query_label="A")

            exit_code = run_retrieval_eval_main(
                [
                    "--data-root",
                    str(data_root),
                    "--gallery-split-file",
                    str(gallery_split),
                    "--query-split-file",
                    str(query_split),
                    "--encoded-dir",
                    str(encoded_dir),
                    "--index-root",
                    str(index_root),
                    "--output-root",
                    str(output_root),
                    "--compare",
                    "--top-k",
                    "2",
                ]
            )
            self.assertEqual(exit_code, 0)

            tf_variant = output_root / "gallery" / "SIFT" / "tf_dot"
            tfidf_variant = output_root / "gallery" / "SIFT" / "tfidf_cosine"
            self.assertTrue((tf_variant / "summary_metrics.json").is_file())
            self.assertTrue((tfidf_variant / "summary_metrics.json").is_file())

            with (tf_variant / "per_query_results.json").open("r", encoding="utf-8") as handle:
                tf_results = json.load(handle)
            with (tfidf_variant / "per_query_results.json").open("r", encoding="utf-8") as handle:
                tfidf_results = json.load(handle)
            with (tf_variant / "summary_metrics.json").open("r", encoding="utf-8") as handle:
                tf_summary = json.load(handle)
            with (tfidf_variant / "summary_metrics.json").open("r", encoding="utf-8") as handle:
                tfidf_summary = json.load(handle)

            self.assertEqual(tf_results[0]["query_sample_id"], "test/A/q1.jpg")
            self.assertEqual(tf_results[0]["ranked_results"][0]["sample_id"], "test/A/g1.jpg")
            self.assertEqual(tfidf_results[0]["ranked_results"][0]["sample_id"], "test/A/g2.jpg")
            self.assertNotEqual(tf_summary["mean_precision_at_k"], tfidf_summary["mean_precision_at_k"])

    def test_no_relevant_gallery_items_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root, gallery_split, query_split = self._prepare_dataset(root, query_label="C")
            encoded_dir = root / "encoded"
            index_root = root / "indices"
            output_root = root / "evaluations"

            self._save_encoded_artifacts(encoded_dir, query_label="C")

            method_evaluations, _ = evaluate_canonical_retrieval(
                encoded_dir=encoded_dir,
                gallery_split_file=gallery_split,
                query_split_file=query_split,
                data_root=data_root,
                index_root=index_root,
                output_root=output_root,
                compare=False,
                requested_top_k=2,
                weighting_mode="tfidf",
                scoring_mode="cosine",
            )

            summary = method_evaluations[0].variants[0].summary
            query_metrics = method_evaluations[0].variants[0].query_results[0].metrics

            self.assertEqual(summary.query_count_with_relevant_items, 0)
            self.assertEqual(summary.query_count_without_relevant_items, 1)
            self.assertIsNone(summary.mean_recall_at_k)
            self.assertIsNone(summary.mean_average_precision)
            self.assertFalse(query_metrics.included_in_mean_recall)
            self.assertFalse(query_metrics.included_in_mean_average_precision)
            self.assertIsNone(query_metrics.recall_at_k)
            self.assertIsNone(query_metrics.average_precision)
            self.assertEqual(method_evaluations[0].variants[0].query_results[0].pr_curve_points, [])

    def test_missing_split_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root, gallery_split, query_split = self._prepare_dataset(root, query_label="A")

            with self.assertRaises(FileNotFoundError):
                load_canonical_split_samples(gallery_split, query_split.with_name("missing_query.txt"), data_root)

    def test_normalized_bow_rejection_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root, gallery_split, query_split = self._prepare_dataset(root, query_label="A")
            encoded_dir = root / "encoded"
            index_root = root / "indices"
            output_root = root / "evaluations"

            self._save_encoded_artifacts(
                encoded_dir,
                query_label="A",
                normalized_gallery=True,
                mismatched_query_dims=True,
            )

            with self.assertRaises(ValueError) as normalized_exc:
                evaluate_canonical_retrieval(
                    encoded_dir=encoded_dir,
                    gallery_split_file=gallery_split,
                    query_split_file=query_split,
                    data_root=data_root,
                    index_root=index_root,
                    output_root=output_root,
                    compare=False,
                    requested_top_k=2,
                    weighting_mode="tfidf",
                    scoring_mode="cosine",
                )
            self.assertIn("normalized BoW artifacts are not supported", str(normalized_exc.exception))

    def test_histogram_size_mismatch_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root, gallery_split, query_split = self._prepare_dataset(root, query_label="A")
            encoded_dir = root / "encoded"
            index_root = root / "indices"
            output_root = root / "evaluations"

            self._save_encoded_artifacts(
                encoded_dir,
                query_label="A",
                normalized_gallery=False,
                mismatched_query_dims=True,
            )

            with self.assertRaises(ValueError) as mismatch_exc:
                evaluate_canonical_retrieval(
                    encoded_dir=encoded_dir,
                    gallery_split_file=gallery_split,
                    query_split_file=query_split,
                    data_root=data_root,
                    index_root=index_root,
                    output_root=output_root,
                    compare=False,
                    requested_top_k=2,
                    weighting_mode="tfidf",
                    scoring_mode="cosine",
                )
            self.assertIn("histogram size mismatch", str(mismatch_exc.exception))

    def _prepare_dataset(self, root: Path, *, query_label: str) -> tuple[Path, Path, Path]:
        data_root = root / "data"
        gallery_split = data_root / "splits" / "gallery.txt"
        query_split = data_root / "splits" / "query.txt"

        gallery_entries = [
            "test/A/g1.jpg",
            "test/A/g2.jpg",
            "test/B/g3.jpg",
        ]
        query_entries = [
            f"test/{query_label}/q1.jpg",
        ]

        self._write_split_and_files(data_root, gallery_entries, gallery_split)
        self._write_split_and_files(data_root, query_entries, query_split)
        return data_root, gallery_split, query_split

    def _write_split_and_files(self, data_root: Path, entries: list[str], split_file: Path) -> None:
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text("\n".join(entries) + "\n", encoding="utf-8")
        for entry in entries:
            file_path = data_root / entry
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(b"")

    def _save_encoded_artifacts(
        self,
        encoded_dir: Path,
        *,
        query_label: str,
        normalized_gallery: bool = False,
        mismatched_query_dims: bool = False,
    ) -> None:
        encoded_dir.mkdir(parents=True, exist_ok=True)

        self._save_encoded_feature(encoded_dir, "A/g1.jpg", [20, 0], normalized=normalized_gallery)
        self._save_encoded_feature(encoded_dir, "A/g2.jpg", [1, 1], normalized=normalized_gallery)
        self._save_encoded_feature(encoded_dir, "B/g3.jpg", [0, 5], normalized=normalized_gallery)
        query_histogram = [1, 1, 0] if mismatched_query_dims else [1, 1]
        self._save_encoded_feature(
            encoded_dir,
            f"{query_label}/q1.jpg",
            query_histogram,
            normalized=False,
            num_visual_words=3 if mismatched_query_dims else 2,
        )

    def _save_encoded_feature(
        self,
        encoded_dir: Path,
        sample_id: str,
        histogram_values: list[int],
        *,
        normalized: bool,
        num_visual_words: int | None = None,
    ) -> None:
        histogram_dtype = "float32" if normalized else "int32"
        histogram = np.asarray(histogram_values, dtype=np.float32 if normalized else np.int32)
        resolved_num_visual_words = len(histogram_values) if num_visual_words is None else num_visual_words
        encoded_feature = EncodedFeatureResult(
            sample_id=sample_id,
            method="SIFT",
            encoding_type="bow",
            num_visual_words=resolved_num_visual_words,
            histogram=histogram,
            histogram_dtype=histogram_dtype,
            num_descriptors=int(np.sum(histogram_values)),
            descriptors_present=True,
            codebook_path="outputs/indices/codebooks/codebook_sift_test.npz",
            normalized=normalized,
        )
        save_encoded_feature(encoded_feature, encoded_dir)


if __name__ == "__main__":
    unittest.main()
