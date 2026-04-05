# Codex Prompt — Issue 1.1 Implement Dataset Loader

你正在协助实现一个课程驱动、工程化推进的 **Hybrid Image Retrieval System**。
当前项目处于 **Milestone 1: System Skeleton**，目标是建立最小系统骨架，而不是提前实现复杂检索系统。

## 1. Current Milestone Context

当前项目采用“宏观规划驱动 + 阶段闭环驱动”的方式开发。
目前最重要的是支撑这一最小闭环：

**原始图像 → 数据读取 → 基础预处理 → 局部特征提取 → 特征保存 → 关键点可视化**

因此，这个 Issue 的重点不是做训练型 DataLoader，也不是做高性能并行加载，而是建立一个：

* 稳定
* 清晰
* 最小可用
* 易扩展

的数据入口层，供后续预处理、局部特征提取、可视化模块使用。

---

## 2. Issue Target

实现 **Issue 1.1 — Implement Dataset Loader**

目标是提供一个最小可用的数据加载模块，使系统能够：

1. 扫描指定图像目录；
2. 过滤支持的图像格式；
3. 生成统一样本记录，而不是裸路径列表；
4. 提供顺序访问能力；
5. 提供基本长度统计和样本预览能力；
6. 提供图像读取辅助接口；
7. 为未来 `query / gallery / split` 扩展保留字段。

---

## 3. Scope Boundaries

### In scope

只实现当前阶段所需的最小统一数据入口，包括：

* 扫描图像目录
* 过滤图像文件格式
* 生成样本清单
* 定义单样本记录结构
* 提供基础访问接口
* 输出基本统计信息

### Out of scope

本次 **不要实现** 下列内容：

* 多进程 / 多线程并行加载
* batch tensor 化
* PyTorch 训练型 Dataset / DataLoader 复杂适配
* 数据增强
* 缓存机制
* query/gallery 专门子类
* 复杂元数据解析系统
* 递归扫描以外的复杂数据组织逻辑
* 任何“为了以后可能用到”而提前加入的重量设计

原则：**只做当前最小闭环需要的最小实现。**

---

## 4. Required File Changes

请优先在以下路径下实现：

### Must create / update

* `src/datasets/dataset_loader.py`

### If necessary, may also create

* `src/datasets/__init__.py`

但不要无故新增很多文件。
本 Issue 应尽量集中在 `dataset_loader.py` 中完成。

---

## 5. Implementation Requirements

请实现一个简洁、可维护的数据加载模块，建议包含以下内容（你可以合理命名，但不要过度拆分）：

### 5.1 Sample record definition

定义统一样本记录，建议使用 `dataclass`。

每个样本至少包含以下字段：

* `sample_id: str`
* `image_path: Path` 或 `str`
* `file_name: str`
* `split: str`
* `meta: dict`

约束：

* `split` 当前可默认 `"unspecified"` 或类似安全默认值
* `meta` 当前可为空字典，但必须预留
* `sample_id` 应稳定且可读，至少在当前扫描结果内唯一
* 不要把 sample record 设计得过重

---

### 5.2 Dataset loader class

实现一个最小类，例如可命名为：

* `ImageDatasetLoader`
* 或 `DatasetLoader`

其职责：

1. 接收一个图像目录路径；
2. 扫描其中支持的图像文件；
3. 生成样本记录列表；
4. 支持：

   * `__len__`
   * `__getitem__`
   * `__iter__`
5. 提供基础统计信息；
6. 提供少量样本预览方法。

建议支持的图像格式至少包括：

* `.jpg`
* `.jpeg`
* `.png`
* `.bmp`
* `.webp`

扩展名匹配请大小写不敏感。

---

### 5.3 Image reading helper

提供一个基础图像读取辅助函数或方法，例如：

* `load_image(sample)`
* `read_image(path)`

要求：

* 使用 OpenCV 或 PIL 之一即可，优先选与你项目后续局部特征流程更顺手的方案
* 出错时给出清晰报错信息
* 不要在这里提前做复杂预处理
* 当前只需要保证后续模块能基于 sample record 读到图像

如果使用 OpenCV，请注意返回值为空时的错误处理。

---

### 5.4 Statistics / preview methods

至少提供：

* 数据集长度统计
* 支持查看前若干个样本记录的方法，例如 `preview(n=5)`

并在适当位置提供简洁输出，例如：

* `Loaded 100 images from data/train/image`
* `Supported extensions: .jpg, .jpeg, .png, .bmp, .webp`

输出风格保持简洁，不要做复杂 logging 系统。

---

## 6. Design Constraints

请遵守以下工程约束：

### 6.1 Keep it minimal

只实现当前阶段最小可用能力。
不要为了“未来更强”提前做复杂抽象。

### 6.2 Clear interface

后续模块应该可以很自然地这样使用：

1. 初始化 loader
2. 获取 sample records
3. 基于 sample record 读取图像

### 6.3 Stable and readable

代码应：

* 命名清楚
* 注释克制但必要
* 类型标注尽量完整
* 错误信息清楚
* 不要魔法行为

### 6.4 Robust basic behavior

至少处理这些情况：

* 数据目录不存在
* 目录为空
* 没有支持的图像文件
* 图像读取失败

这些情况请使用清晰、直接的异常或提示。

---

## 7. Suggested API Shape

你不必完全照抄，但最终接口风格应接近下面这种简单程度：

```python
from pathlib import Path

loader = ImageDatasetLoader(root_dir=Path("data/train/image"))
print(len(loader))

sample = loader[0]
print(sample)

image = loader.load_image(sample)

for item in loader:
    ...
```

或者：

```python
samples = loader.samples
preview = loader.preview(5)
```

总之，目标是让后续 pipeline 很容易接上。

---

## 8. Acceptance Criteria

完成后应满足：

1. 可以从指定目录扫描得到图像样本；
2. 返回结果不是裸路径列表，而是统一样本记录；
3. 后续模块可以基于该记录读取图像；
4. 可进行长度统计和简单样本预览；
5. 运行时可输出类似：

   * `Loaded 100 images from ...`
6. 数据入口足以支撑当前阶段最小闭环；
7. 没有引入训练型 DataLoader 或其他不必要复杂度。

---

## 9. Expected Output From You

请直接完成代码修改，并同时提供：

1. 你修改/新增了哪些文件；
2. 核心设计说明（简短即可）；
3. 如何使用该 loader 的示例；
4. 说明它如何满足上述 acceptance criteria。

---

## 10. Important Non-Goals

请再次注意，这个 Issue **不是**：

* 深度学习训练数据管线
* 高性能加载器
* 最终版数据系统
* query/gallery/split 全量设计

它只是 **Milestone 1 / System Skeleton** 下的最小数据入口。

宁可简单清晰，也不要提前复杂化。

