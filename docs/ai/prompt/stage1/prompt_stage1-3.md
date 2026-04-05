# Codex Prompt — Issue 1.3 Build Pipeline Skeleton and Define Module Interfaces

你正在协助实现一个课程驱动、工程化推进的 **Hybrid Image Retrieval System**。
当前项目处于 **Milestone 1: System Skeleton**。这一阶段的重点不是实现完整图像检索系统，而是建立一个**可运行、可扩展、边界清晰的系统骨架**，为后续数据预处理、局部特征提取、特征保存、可视化、特征编码和索引模块提供稳定挂接点。

## 1. Current Project Context

当前项目遵循“宏观规划驱动 + 阶段闭环驱动”的方式开发。当前阶段最小闭环应围绕以下流程建立：

**raw image → dataset loader → basic preprocess → local feature extraction → feature save → keypoint visualization**

当前项目在工期1阶段最重要的目标不是性能，也不是提前实现复杂检索系统，而是：

* 把主流程阶段划分清楚
* 把模块职责与输入输出边界约定清楚
* 让主脚本具备清晰的 pipeline 调用链
* 为后续工期2（数据预处理）和工期3（局部特征提取）做好结构准备

后续模块如 `encoding / indexing / retrieval / rerank / expansion / dense global retrieval / hybrid fusion` 只需要预留接口位置，不要提前实现。

## 2. Issue Target

实现 **Issue 1.3 — Build Pipeline Skeleton and Define Module Interfaces**

目标是：

1. 明确当前阶段 pipeline stages；
2. 定义 dataset / preprocess / local features / save / visualize 的模块输入输出；
3. 串联一个清晰的最小调用链；
4. 为未来模块留出清楚的挂接位置；
5. 在 docs 中补充一版当前阶段 pipeline skeleton 说明。

## 3. Scope Boundaries

### In scope

本次只做以下内容：

* 定义当前阶段 pipeline stages
* 实现/整理当前阶段主调用骨架
* 约定模块输入输出接口
* 为未来模块创建最小 stub / placeholder
* 补一份当前阶段 pipeline 说明文档

### Out of scope

本次 **不要实现** 下列内容：

* 真正完整的 preprocess 算法
* 真正完整的 local feature extraction 算法
* 真正的特征保存格式体系
* 真正的关键点可视化实现
* 任何 codebook / tf-idf / inverted index / retrieval / rerank / query expansion / dense retrieval 的实质逻辑
* 复杂的插件系统、注册表系统、依赖注入框架
* “最终版”系统架构

原则：

> **本 Issue 只建立骨架、接口、阶段边界和未来挂接位置。**

## 4. Required File Changes

请优先在以下路径完成最小改动：

### Must create or update

* `scripts/run_pipeline.py`
* `src/preprocess/__init__.py`
* `src/preprocess/basic_preprocess.py`
* `src/features/local/__init__.py`
* `src/features/local/local_feature_extractor.py`
* `src/utils/io.py` 或同等最小工具文件（仅当确实需要）
* `docs/design/pipeline_skeleton.md`

### May create minimal placeholder files if necessary

只在确实有助于边界清晰时，才可增加：

* `src/encoding/__init__.py`
* `src/indexing/__init__.py`
* `src/retrieval/__init__.py`
* `src/rerank/__init__.py`
* `src/expansion/__init__.py`

但这些未来模块**只允许最小占位，不允许实现真实逻辑**。

不要无故新增大量文件。
这次的核心是把当前阶段最小骨架立住，而不是扩张目录。

## 5. Required Skeleton Design

请围绕当前最小闭环建立骨架：

### Stage A — Dataset

职责：

* 提供统一样本记录
* 提供图像读取入口

输入：

* dataset root / image directory

输出：

* sample records
* loaded image

### Stage B — Basic Preprocess

职责：

* 提供最小预处理接口
* 当前阶段只允许做非常轻量的保留型操作，或者直接 passthrough

输入：

* image
* optional preprocess config

输出：

* processed image
* optional preprocess metadata

要求：

* 当前默认可直接返回原图，或仅做极轻量的格式/颜色处理
* 不要做复杂增强

### Stage C — Local Feature Extraction

职责：

* 定义局部特征提取模块接口
* 当前阶段允许只返回占位结果，不要求实现真实 SIFT / ORB

输入：

* processed image
* feature config

输出建议统一成结构化结果，例如：

* `keypoints`
* `descriptors`
* `meta`

要求：

* 输出格式要为后续真实实现留好位置
* 当前可返回空列表 / `None` / 占位结构，但结构必须清楚

### Stage D — Feature Save

职责：

* 定义特征保存接口位置
* 当前阶段不需要真正落盘复杂特征文件

输入：

* sample
* feature result

输出：

* 当前可为 no-op 或打印占位信息

### Stage E — Visualization

职责：

* 定义关键点可视化接口位置
* 当前阶段不需要真正画图

输入：

* sample
* image
* feature result

输出：

* 当前可为 no-op 或打印占位信息

## 6. Interface Design Requirements

### 6.1 Keep interfaces explicit

请为当前阶段模块定义清晰、简单的接口，不要搞抽象层套抽象层。

例如可以接近：

```python
processed_image = preprocess_image(image, config)
feature_result = extract_local_features(processed_image, config)
save_feature_result(sample, feature_result, config)
visualize_keypoints(sample, image, feature_result, config)
```

或者使用极轻量类，但不要过度设计。

### 6.2 Use structured outputs

对于局部特征结果，建议定义一个轻量数据结构，例如 dataclass：

* `LocalFeatureResult`

  * `keypoints`
  * `descriptors`
  * `meta`

即使当前还是 placeholder，也要把后续接口形状先固定。

### 6.3 Preserve future extensibility

未来模块只需要清楚挂接点，例如在 `run_pipeline.py` 或 docs 中明确：

* feature encoding hook
* tf-idf hook
* inverted index hook
* retrieval hook
* rerank hook
* query expansion hook
* dense global retrieval hook
* hybrid fusion hook

但只允许注释、stub 或文档说明，不能实现真实功能。

## 7. Changes Required in run_pipeline.py

请在已有 `scripts/run_pipeline.py` 基础上升级为**当前阶段 pipeline skeleton**，要求：

1. 保留 Issue 1.2 已完成的 config + dataset 入口
2. 增加对 preprocess / local feature / save / visualize 的调用链
3. 当前只需处理前少量样本进行最小演示
4. 输出每个阶段的简洁状态信息
5. 整体仍然必须可以直接：

```bash
python scripts/run_pipeline.py
```

运行后，用户应能清楚看到主流程已经按阶段串起来，即使其中部分阶段还是 placeholder。

## 8. Console Output Style

输出保持简洁明确，例如：

* `Loaded config from ...`
* `Loaded 946 images from ...`
* `Running pipeline skeleton on first 3 samples`
* `Preprocess stage completed for ...`
* `Local feature stage placeholder executed for ...`
* `Save stage placeholder executed for ...`
* `Visualization stage placeholder executed for ...`
* `Pipeline skeleton run completed`

不要引入 logging 框架。
不要输出大段花哨文本。

## 9. Documentation Requirement

请新增或补充：

* `docs/design/pipeline_skeleton.md`

内容至少包括：

1. 当前阶段 pipeline stages
2. 每个模块职责
3. 每个模块输入输出
4. 当前最小闭环的调用顺序
5. 未来模块挂接位置
6. 明确说明哪些模块当前只是 placeholder，哪些是已真实接通

文档风格要求：

* Markdown
* 结构清楚
* 工程语气
* 可作为后续工期2/3的接口说明基础

## 10. Design Constraints

### 10.1 Do not overbuild

不要把这一步做成最终版框架。
它只是工期1的系统骨架。

### 10.2 Do not fake implementation depth

可以有 placeholder，但必须诚实清楚标明 placeholder。
不要写一堆看起来复杂、实际上没有意义的空壳体系。

### 10.3 Keep the call chain runnable

这次最重要的一点：

> **主调用链必须仍然真实可运行。**

即便 preprocess / feature / save / visualize 还是最小占位，也要能从入口脚本顺序走通。

### 10.4 Respect current issue boundaries

不要提前实现：

* 真实 preprocess 细节
* 真实 SIFT / ORB 提取
* 真实特征保存格式
* 真实关键点可视化
* 任何 encoding/indexing/retrieval 算法

这些属于后续 issue，不属于这次。

## 11. Acceptance Criteria

完成后应满足：

1. 当前阶段 pipeline 结构可被清晰说明；
2. 系统可从入口调用到 dataset / preprocess / features / save / visualize；
3. 各模块的输入输出边界明确；
4. 后续模块具有清晰挂接位置；
5. 文档里清楚说明当前阶段骨架与未来扩展关系；
6. `python scripts/run_pipeline.py` 仍可运行；
7. 没有提前引入复杂算法实现或最终版框架。

## 12. Expected Output From You

请直接完成代码修改，并同时提供：

1. 你新增/修改了哪些文件；
2. 当前 pipeline skeleton 的调用链说明；
3. 你定义了哪些模块接口；
4. 哪些模块是已真实接通，哪些是 placeholder；
5. 示例运行输出；
6. 它如何满足 acceptance criteria。

## 13. Important Non-Goals

请再次注意，这个 Issue 不是：

* 工期2的数据预处理实现
* 工期3的局部特征提取实现
* 工期4之后的任何检索算法实现
* 最终版系统框架

它只是：

> **在工期1中，把“流程怎么串、模块边界是什么、未来往哪接”这件事彻底钉清楚。**
