from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np

from src.features.local import extract_local_features
from src.preprocess import preprocess_image
from src.utils import get_default_config_path, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER_RESULT_COUNT = 4


def load_demo_config(config_path: str | None = None) -> tuple[dict[str, Any], Path]:
    resolved_path = Path(config_path or get_default_config_path()).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = (PROJECT_ROOT / resolved_path).resolve()
    else:
        resolved_path = resolved_path.resolve()

    return load_config(str(resolved_path)), resolved_path


def run_demo_bridge(query_image_path: str | None) -> tuple[str, list[tuple[np.ndarray, str]]]:
    if not query_image_path:
        return (
            "### Pipeline Status\n- No query image provided.\n- Upload an image and click **Run Pipeline Skeleton**.",
            build_placeholder_gallery(),
        )

    config, config_path = load_demo_config()
    image = load_query_image(query_image_path)
    preprocess_result = preprocess_image(image, _get_mapping(config, "preprocess"))
    feature_result = extract_local_features(preprocess_result.image, _get_mapping(config, "feature"))

    status = "\n".join(
        [
            "### Pipeline Status",
            f"- Config loaded from `{config_path}`",
            f"- Query image loaded: `{Path(query_image_path).name}` shape={shape_of(image)}",
            f"- Preprocess stage completed: steps={preprocess_result.meta.get('applied_steps')}",
            "- Local feature stage executed as placeholder",
            f"- Placeholder method: `{feature_result.meta.get('method')}`",
            "- Top-K results below are placeholders for future retrieval outputs",
            "- Future hooks reserved: retrieval, keypoint visualization, local feature visualization",
        ]
    )
    return status, build_placeholder_gallery()


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Hybrid Image Retrieval System Demo") as demo:
        gr.Markdown(
            "# Hybrid Image Retrieval System Demo\n"
            "Milestone 1 web demo. Upload one query image, trigger the current pipeline skeleton, "
            "and inspect the placeholder Top-K result area."
        )

        with gr.Row():
            with gr.Column(scale=1):
                query_image = gr.Image(
                    type="filepath",
                    label="Query Image Upload",
                    sources=["upload"],
                )
                run_button = gr.Button("Run Pipeline Skeleton", variant="primary")
                gr.Markdown(
                    "This demo currently runs image loading, basic preprocess and local feature placeholder logic."
                )

            with gr.Column(scale=1):
                pipeline_status = gr.Markdown(
                    "### Pipeline Status\n- Waiting for a query image.\n- Retrieval results are placeholders in Stage 1."
                )
                topk_results = gr.Gallery(
                    value=build_placeholder_gallery(),
                    label="Top-K Results",
                    columns=2,
                    height="auto",
                )

        run_button.click(
            fn=run_demo_bridge,
            inputs=query_image,
            outputs=[pipeline_status, topk_results],
        )

    return demo


def load_query_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Query image not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Query image path is not a file: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read query image: {path}")
    return image


def build_placeholder_gallery() -> list[tuple[np.ndarray, str]]:
    return [
        (
            create_placeholder_card(rank=index + 1),
            f"Top-{index + 1} placeholder - future retrieval result",
        )
        for index in range(PLACEHOLDER_RESULT_COUNT)
    ]


def create_placeholder_card(rank: int, size: tuple[int, int] = (320, 320)) -> np.ndarray:
    height, width = size
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    accent = 210 - rank * 12
    canvas[:, :, 0] = accent
    canvas[:, :, 1] = min(255, accent + 18)
    canvas[:, :, 2] = 235

    cv2.putText(canvas, f"Top-{rank}", (36, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (35, 35, 35), 2)
    cv2.putText(canvas, "Placeholder", (36, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (35, 35, 35), 2)
    cv2.putText(
        canvas,
        "Future retrieval result",
        (36, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (55, 55, 55),
        2,
    )
    return canvas


def _get_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return value


def shape_of(image: Any) -> tuple[int, ...] | None:
    shape = getattr(image, "shape", None)
    if not isinstance(shape, tuple):
        return None
    return tuple(int(dim) for dim in shape)
