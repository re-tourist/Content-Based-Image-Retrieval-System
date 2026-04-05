from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.demo import build_demo


def main() -> None:
    print("Launching Gradio web demo...")
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()
