from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
OUTPUT_DIR = DEMO_DIR / "outputs"
LOCAL_IMAGE_DIR = DEMO_DIR / "images"
DEFAULT_FALLBACK_DIRS = [
    REPO_ROOT / "data" / "train" / "image",
    REPO_ROOT / "data" / "test",
]


@dataclass
class CandidateScore:
    image_path: Path
    good_matches: int
    raw_matches: int
    average_distance: float
    elapsed_ms: float
    keypoints: int


@dataclass
class RetrievalResult:
    requested_algorithm: str
    actual_algorithm: str
    algorithm_note: str
    query_path: Path
    best_match_path: Path
    visualization_path: Path
    gallery_count: int
    total_elapsed_ms: float
    best_candidate_elapsed_ms: float
    query_keypoints: int
    best_keypoints: int
    raw_matches: int
    good_matches: int

    def to_text(self) -> str:
        lines = [
            f"算法: {self.actual_algorithm}",
            f"请求算法: {self.requested_algorithm}",
            f"Query: {self.query_path}",
            f"最佳匹配: {self.best_match_path}",
            f"good matches: {self.good_matches}",
            f"原始 matches: {self.raw_matches}",
            f"Query 关键点: {self.query_keypoints}",
            f"最佳图关键点: {self.best_keypoints}",
            f"最佳候选单张耗时: {self.best_candidate_elapsed_ms:.2f} ms",
            f"总检索耗时: {self.total_elapsed_ms:.2f} ms",
            f"候选图数量: {self.gallery_count}",
            f"结果图: {self.visualization_path}",
        ]
        if self.algorithm_note:
            lines.append(f"说明: {self.algorithm_note}")
        return "\n".join(lines)


def ensure_demo_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def get_default_gallery_dirs() -> list[Path]:
    ensure_demo_dirs()
    local_images = list_images(LOCAL_IMAGE_DIR)
    if local_images:
        return [LOCAL_IMAGE_DIR]
    return [path for path in DEFAULT_FALLBACK_DIRS if path.exists()]


def collect_gallery_images(
    query_path: Path,
    gallery_dirs: Iterable[Path] | None = None,
    max_candidates: int | None = None,
) -> list[Path]:
    chosen_dirs = list(gallery_dirs) if gallery_dirs else get_default_gallery_dirs()
    query_resolved = query_path.resolve()
    collected: list[Path] = []
    seen: set[Path] = set()

    for root in chosen_dirs:
        for image_path in list_images(root):
            resolved = image_path.resolve()
            if resolved == query_resolved or resolved in seen:
                continue
            seen.add(resolved)
            collected.append(image_path)
            if max_candidates and len(collected) >= max_candidates:
                return collected

    return collected


def find_sample_query_image() -> Path | None:
    search_roots = [LOCAL_IMAGE_DIR, REPO_ROOT / "data" / "test", REPO_ROOT / "data" / "train" / "image"]
    for root in search_roots:
        images = list_images(root)
        if images:
            return images[0]
    return None


def available_algorithms() -> list[str]:
    names: list[str] = []
    if hasattr(cv2, "SIFT_create"):
        names.append("SIFT")
    if hasattr(cv2, "ORB_create"):
        names.append("ORB")
    if hasattr(cv2, "AKAZE_create"):
        names.append("AKAZE")
    return names


def default_benchmark_algorithms() -> list[str]:
    names = available_algorithms()
    if "SIFT" in names and "ORB" in names:
        return ["SIFT", "ORB"]
    if "ORB" in names and "AKAZE" in names:
        return ["ORB", "AKAZE"]
    return names[:2]


def resolve_algorithm(requested: str) -> tuple[str, str]:
    requested = requested.upper()
    names = available_algorithms()
    if not names:
        raise RuntimeError("当前 OpenCV 环境没有可用的局部特征算法。")
    if requested in names:
        return requested, ""

    fallback_order = {
        "SIFT": ["ORB", "AKAZE"],
        "SURF": ["SIFT", "ORB", "AKAZE"],
        "ORB": ["AKAZE", "SIFT"],
        "AKAZE": ["ORB", "SIFT"],
    }
    for candidate in fallback_order.get(requested, names):
        if candidate in names:
            return candidate, f"{requested} 不可用，已自动回退到 {candidate}。"

    return names[0], f"{requested} 不可用，已自动回退到 {names[0]}。"


def create_extractor(name: str):
    if name == "SIFT":
        return cv2.SIFT_create(nfeatures=800)
    if name == "ORB":
        return cv2.ORB_create(nfeatures=1500)
    if name == "AKAZE":
        return cv2.AKAZE_create()
    raise ValueError(f"不支持的算法: {name}")


def load_image(path: Path, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    return image


def resize_for_feature(image: np.ndarray, max_side: int = 960) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def extract_features(image_path: Path, algorithm: str):
    gray = load_image(image_path, grayscale=True)
    gray = resize_for_feature(gray)
    extractor = create_extractor(algorithm)
    keypoints, descriptors = extractor.detectAndCompute(gray, None)
    keypoints = keypoints or []
    return gray, keypoints, descriptors


def descriptor_norm(descriptors: np.ndarray | None, algorithm: str) -> int:
    if algorithm == "SIFT":
        return cv2.NORM_L2
    if descriptors is not None and descriptors.dtype == np.uint8:
        return cv2.NORM_HAMMING
    return cv2.NORM_L2


def ratio_threshold(algorithm: str) -> float:
    if algorithm == "SIFT":
        return 0.75
    if algorithm == "ORB":
        return 0.78
    return 0.80


def match_descriptors(
    query_descriptors: np.ndarray | None,
    candidate_descriptors: np.ndarray | None,
    algorithm: str,
) -> tuple[list[cv2.DMatch], int]:
    if query_descriptors is None or candidate_descriptors is None:
        return [], 0
    if len(query_descriptors) < 2 or len(candidate_descriptors) < 2:
        return [], 0

    matcher = cv2.BFMatcher(descriptor_norm(query_descriptors, algorithm), crossCheck=False)
    raw_pairs = matcher.knnMatch(query_descriptors, candidate_descriptors, k=2)
    threshold = ratio_threshold(algorithm)
    good_matches: list[cv2.DMatch] = []

    for pair in raw_pairs:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < threshold * second.distance:
            good_matches.append(first)

    good_matches.sort(key=lambda match: match.distance)
    return good_matches, len(raw_pairs)


def choose_better_candidate(current: CandidateScore | None, challenger: CandidateScore) -> CandidateScore:
    if current is None:
        return challenger
    if challenger.good_matches > current.good_matches:
        return challenger
    if challenger.good_matches == current.good_matches and challenger.average_distance < current.average_distance:
        return challenger
    return current


def save_match_visualization(
    query_path: Path,
    best_path: Path,
    algorithm: str,
    query_keypoints,
    best_keypoints,
    matches: list[cv2.DMatch],
) -> Path:
    query_color = resize_for_feature(load_image(query_path))
    best_color = resize_for_feature(load_image(best_path))
    top_matches = matches[:50]
    canvas = cv2.drawMatches(
        query_color,
        query_keypoints,
        best_color,
        best_keypoints,
        top_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = OUTPUT_DIR / f"{algorithm.lower()}_{query_path.stem}_{timestamp}.jpg"
    cv2.imwrite(str(save_path), canvas)
    return save_path


def _fit_image_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width == target_width:
        return image
    scale = target_width / width
    new_height = max(1, int(height * scale))
    return cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)


def create_benchmark_comparison_figure(results: Iterable[RetrievalResult]) -> Path:
    result_list = list(results)
    if not result_list:
        raise ValueError("没有可用于合并展示的 benchmark 结果。")

    panels: list[np.ndarray] = []
    target_width = 1300
    margin = 24
    title_height = 80
    caption_height = 70
    font = cv2.FONT_HERSHEY_SIMPLEX

    for result in result_list:
        image = load_image(result.visualization_path)
        image = _fit_image_to_width(image, target_width)
        caption = np.full((caption_height, target_width, 3), 245, dtype=np.uint8)
        best_name = result.best_match_path.name
        line1 = f"{result.actual_algorithm}  good matches={result.good_matches}  total={result.total_elapsed_ms:.2f} ms"
        line2 = f"Best match: {best_name}"
        cv2.putText(caption, line1, (18, 28), font, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(caption, line2, (18, 56), font, 0.62, (50, 50, 50), 1, cv2.LINE_AA)
        panels.append(caption)
        panels.append(image)

    canvas_height = title_height + margin * (len(panels) + 1) + sum(panel.shape[0] for panel in panels)
    canvas_width = target_width + margin * 2
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    cv2.putText(
        canvas,
        "Local Feature Matching Comparison",
        (margin, 34),
        font,
        1.0,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )
    query_name = result_list[0].query_path.name
    cv2.putText(
        canvas,
        f"Query: {query_name}",
        (margin, 64),
        font,
        0.65,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )

    y = title_height
    for panel in panels:
        canvas[y : y + panel.shape[0], margin : margin + panel.shape[1]] = panel
        y += panel.shape[0] + margin

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = OUTPUT_DIR / f"comparison_{result_list[0].query_path.stem}_{timestamp}.jpg"
    cv2.imwrite(str(save_path), canvas)
    return save_path


def retrieve_best_match(
    query_path: str | Path,
    algorithm: str = "SIFT",
    gallery_dirs: Iterable[Path] | None = None,
    max_candidates: int | None = None,
) -> RetrievalResult:
    ensure_demo_dirs()
    query_path = Path(query_path).expanduser().resolve()
    if not query_path.exists():
        raise FileNotFoundError(f"待匹配图像不存在: {query_path}")

    actual_algorithm, algorithm_note = resolve_algorithm(algorithm)
    gallery_images = collect_gallery_images(query_path, gallery_dirs, max_candidates=max_candidates)
    if not gallery_images:
        roots = ", ".join(str(path) for path in (gallery_dirs or get_default_gallery_dirs()))
        raise FileNotFoundError(f"未找到可用图库图像。当前搜索目录: {roots}")

    total_start = time.perf_counter()
    _, query_keypoints, query_descriptors = extract_features(query_path, actual_algorithm)
    if query_descriptors is None or not query_keypoints:
        raise RuntimeError(f"Query 图像无法提取到 {actual_algorithm} 特征: {query_path}")

    best_candidate: CandidateScore | None = None
    best_matches: list[cv2.DMatch] = []
    best_candidate_keypoints = []
    best_candidate_descriptors = None

    for candidate_path in gallery_images:
        single_start = time.perf_counter()
        try:
            _, candidate_keypoints, candidate_descriptors = extract_features(candidate_path, actual_algorithm)
        except Exception:
            continue

        good_matches, raw_count = match_descriptors(query_descriptors, candidate_descriptors, actual_algorithm)
        elapsed_ms = (time.perf_counter() - single_start) * 1000
        average_distance = float(np.mean([item.distance for item in good_matches])) if good_matches else float("inf")
        candidate_score = CandidateScore(
            image_path=candidate_path,
            good_matches=len(good_matches),
            raw_matches=raw_count,
            average_distance=average_distance,
            elapsed_ms=elapsed_ms,
            keypoints=len(candidate_keypoints),
        )

        previous_best = best_candidate
        best_candidate = choose_better_candidate(best_candidate, candidate_score)
        if best_candidate is not previous_best:
            best_matches = good_matches
            best_candidate_keypoints = candidate_keypoints
            best_candidate_descriptors = candidate_descriptors

    if best_candidate is None or best_candidate_descriptors is None:
        raise RuntimeError("图库图像未能提取有效特征，无法完成匹配。")

    total_elapsed_ms = (time.perf_counter() - total_start) * 1000
    visualization_path = save_match_visualization(
        query_path=query_path,
        best_path=best_candidate.image_path,
        algorithm=actual_algorithm,
        query_keypoints=query_keypoints,
        best_keypoints=best_candidate_keypoints,
        matches=best_matches,
    )

    return RetrievalResult(
        requested_algorithm=algorithm.upper(),
        actual_algorithm=actual_algorithm,
        algorithm_note=algorithm_note,
        query_path=query_path,
        best_match_path=best_candidate.image_path.resolve(),
        visualization_path=visualization_path.resolve(),
        gallery_count=len(gallery_images),
        total_elapsed_ms=total_elapsed_ms,
        best_candidate_elapsed_ms=best_candidate.elapsed_ms,
        query_keypoints=len(query_keypoints),
        best_keypoints=best_candidate.keypoints,
        raw_matches=best_candidate.raw_matches,
        good_matches=best_candidate.good_matches,
    )


def benchmark_algorithms(
    query_path: str | Path,
    gallery_dirs: Iterable[Path] | None = None,
    algorithms: Iterable[str] | None = None,
    max_candidates: int | None = None,
) -> list[RetrievalResult]:
    chosen = list(algorithms) if algorithms else default_benchmark_algorithms()
    return [
        retrieve_best_match(query_path, algorithm=name, gallery_dirs=gallery_dirs, max_candidates=max_candidates)
        for name in chosen
    ]


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Week01 局部特征图像匹配 demo")
    parser.add_argument("--query", help="待匹配图像路径")
    parser.add_argument("--algorithm", default="SIFT", help="算法名称: SIFT / ORB / AKAZE")
    parser.add_argument("--gallery", nargs="*", help="图库目录，可传多个")
    parser.add_argument("--max-candidates", type=int, default=None, help="限制候选图数量，便于快速测试")
    parser.add_argument("--benchmark", action="store_true", help="运行两种算法并输出时间对比")
    parser.add_argument("--json", action="store_true", help="以 JSON 形式输出结果")
    args = parser.parse_args()

    query_path = Path(args.query).expanduser() if args.query else find_sample_query_image()
    if query_path is None:
        parser.error("未找到可用的 query 图像，请通过 --query 指定。")

    gallery_dirs = [Path(path).expanduser() for path in args.gallery] if args.gallery else None

    if args.benchmark:
        results = benchmark_algorithms(
            query_path=query_path,
            gallery_dirs=gallery_dirs,
            max_candidates=args.max_candidates,
        )
        comparison_path = create_benchmark_comparison_figure(results)
        if args.json:
            payload = {
                "results": [_result_to_dict(item) for item in results],
                "comparison_figure": str(comparison_path),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for item in results:
                print(item.to_text())
                print("-" * 60)
            print(f"对比图: {comparison_path}")
        return 0

    result = retrieve_best_match(
        query_path=query_path,
        algorithm=args.algorithm,
        gallery_dirs=gallery_dirs,
        max_candidates=args.max_candidates,
    )
    if args.json:
        print(json.dumps(_result_to_dict(result), ensure_ascii=False, indent=2))
    else:
        print(result.to_text())
    return 0


def _result_to_dict(result: RetrievalResult) -> dict:
    payload = asdict(result)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


if __name__ == "__main__":
    raise SystemExit(_cli())
