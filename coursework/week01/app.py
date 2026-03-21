from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageTk

from core import (
    LOCAL_IMAGE_DIR,
    benchmark_algorithms,
    create_benchmark_comparison_figure,
    default_benchmark_algorithms,
    find_sample_query_image,
    get_default_gallery_dirs,
    retrieve_best_match,
)


class MatchDemoApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Week01 局部特征匹配 Demo")
        self.root.geometry("1320x860")

        self.query_var = tk.StringVar()
        self.algorithm_var = tk.StringVar(value="SIFT")
        self.query_photo = None
        self.best_photo = None
        self.match_photo = None

        self._build_ui()
        self._load_default_query()
        self._append_info(self._default_intro())

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(main)
        controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls, text="待匹配图像").grid(row=0, column=0, padx=(0, 8), sticky="w")
        ttk.Entry(controls, textvariable=self.query_var, width=90).grid(row=0, column=1, padx=(0, 8), sticky="ew")
        self.select_button = ttk.Button(controls, text="选择图像", command=self.select_query_image)
        self.select_button.grid(row=0, column=2, padx=(0, 8))

        ttk.Label(controls, text="算法").grid(row=0, column=3, padx=(10, 8))
        self.algorithm_box = ttk.Combobox(
            controls,
            textvariable=self.algorithm_var,
            values=["SIFT", "ORB", "AKAZE"],
            state="readonly",
            width=10,
        )
        self.algorithm_box.grid(row=0, column=4, padx=(0, 8))

        self.match_button = ttk.Button(controls, text="开始匹配", command=self.start_match)
        self.match_button.grid(row=0, column=5, padx=(0, 8))
        self.benchmark_button = ttk.Button(controls, text="两算法对比", command=self.start_benchmark)
        self.benchmark_button.grid(row=0, column=6)
        controls.columnconfigure(1, weight=1)

        images_frame = ttk.Frame(main)
        images_frame.pack(fill=tk.BOTH, expand=True)

        self.query_panel = self._create_image_panel(images_frame, "待匹配图像", 0)
        self.best_panel = self._create_image_panel(images_frame, "最佳匹配图像", 1)
        self.match_panel = self._create_image_panel(images_frame, "匹配连线可视化图", 2)

        info_frame = ttk.LabelFrame(main, text="运行信息", padding=8)
        info_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 0))
        self.info_text = scrolledtext.ScrolledText(info_frame, height=14, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.info_text.configure(state=tk.DISABLED)

    def _create_image_panel(self, parent: ttk.Frame, title: str, column: int) -> ttk.Label:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.grid(row=0, column=column, padx=6, sticky="nsew")
        label = ttk.Label(frame, text="暂无图像", anchor="center")
        label.pack(fill=tk.BOTH, expand=True)
        parent.columnconfigure(column, weight=1)
        parent.rowconfigure(0, weight=1)
        return label

    def _default_intro(self) -> str:
        gallery_dirs = "\n".join(str(path) for path in get_default_gallery_dirs())
        return (
            "默认图库目录:\n"
            f"{gallery_dirs or '未找到图库目录'}\n\n"
            "说明:\n"
            "1. 若 coursework/week01/images/ 中有图片，程序优先使用该目录作为图库。\n"
            "2. 若该目录为空，程序会回退到仓库内已有的 data/train/image 和 data/test。\n"
            "3. 匹配结果图会自动保存到 coursework/week01/outputs/。"
        )

    def _load_default_query(self) -> None:
        sample = find_sample_query_image()
        if sample:
            self.query_var.set(str(sample))
            self._set_panel_image(self.query_panel, sample, "query")

    def select_query_image(self) -> None:
        initial_dir = LOCAL_IMAGE_DIR if LOCAL_IMAGE_DIR.exists() else Path.cwd()
        file_path = filedialog.askopenfilename(
            title="选择待匹配图像",
            initialdir=str(initial_dir),
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if file_path:
            self.query_var.set(file_path)
            self._set_panel_image(self.query_panel, Path(file_path), "query")

    def start_match(self) -> None:
        query_path = self.query_var.get().strip()
        if not query_path:
            messagebox.showwarning("缺少输入", "请先选择一张待匹配图像。")
            return
        self._set_busy(True)
        self._append_info(f"开始匹配: {query_path}\n算法: {self.algorithm_var.get()}")
        threading.Thread(target=self._run_match_worker, daemon=True).start()

    def start_benchmark(self) -> None:
        query_path = self.query_var.get().strip()
        if not query_path:
            messagebox.showwarning("缺少输入", "请先选择一张待匹配图像。")
            return
        self._set_busy(True)
        self._append_info(f"开始两算法对比: {query_path}")
        threading.Thread(target=self._run_benchmark_worker, daemon=True).start()

    def _run_match_worker(self) -> None:
        try:
            result = retrieve_best_match(self.query_var.get(), self.algorithm_var.get())
        except Exception as exc:
            self.root.after(0, lambda: self._handle_error(exc))
            return
        self.root.after(0, lambda: self._show_match_result(result))

    def _run_benchmark_worker(self) -> None:
        try:
            results = benchmark_algorithms(self.query_var.get())
            comparison_path = create_benchmark_comparison_figure(results)
        except Exception as exc:
            self.root.after(0, lambda: self._handle_error(exc))
            return
        self.root.after(0, lambda: self._show_benchmark_results(results, comparison_path))

    def _show_match_result(self, result) -> None:
        self._set_busy(False)
        self._set_panel_image(self.query_panel, result.query_path, "query")
        self._set_panel_image(self.best_panel, result.best_match_path, "best")
        self._set_panel_image(self.match_panel, result.visualization_path, "match")
        self._append_info(result.to_text())

    def _show_benchmark_results(self, results, comparison_path: Path) -> None:
        self._set_busy(False)
        if results:
            self._set_panel_image(self.query_panel, results[0].query_path, "query")
            self._set_panel_image(self.best_panel, results[0].best_match_path, "best")
        self._set_panel_image(self.match_panel, comparison_path, "match")
        lines = ["两算法时间对比结果:"]
        for item in results:
            lines.append(
                f"{item.actual_algorithm}: 总耗时 {item.total_elapsed_ms:.2f} ms, "
                f"good matches {item.good_matches}, 最佳匹配 {item.best_match_path.name}"
            )
        lines.append(f"合并对比图: {comparison_path}")
        lines.append("建议: 截图时保留当前窗口和 outputs 中的结果图。")
        self._append_info("\n".join(lines))

    def _handle_error(self, exc: Exception) -> None:
        self._set_busy(False)
        messagebox.showerror("运行失败", str(exc))
        self._append_info(f"错误: {exc}")

    def _set_panel_image(self, panel: ttk.Label, image_path: Path, slot: str) -> None:
        try:
            image = Image.open(image_path)
            image.thumbnail((400, 300))
            photo = ImageTk.PhotoImage(image)
        except Exception as exc:
            panel.configure(text=f"图像加载失败\n{exc}", image="")
            return

        panel.configure(image=photo, text="")
        if slot == "query":
            self.query_photo = photo
        elif slot == "best":
            self.best_photo = photo
        else:
            self.match_photo = photo

    def _append_info(self, message: str) -> None:
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.insert(tk.END, message + "\n" + "=" * 72 + "\n")
        self.info_text.see(tk.END)
        self.info_text.configure(state=tk.DISABLED)

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        self.select_button.configure(state=state)
        self.match_button.configure(state=state)
        self.benchmark_button.configure(state=state)
        self.algorithm_box.configure(state="disabled" if busy else "readonly")


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    MatchDemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
