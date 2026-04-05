# Codex Prompt — Issue 3.1 Implement Local Feature Extraction

你正在协助实现一个课程驱动、工程化推进的 **Hybrid Image Retrieval System**。
当前项目已经完成：

* Milestone 1: System Skeleton
* Milestone 2: Data Preparation（包括 split 机制与最小 preprocess）

现在进入 **Issue 3.1 — Implement Local Feature Extraction**。
这一阶段的目标，是把当前 pipeline skeleton 中的 **local feature extraction placeholder** 替换成**真实可运行的局部特征提取模块**，为后续关键点可视化、特征保存、特征编码与倒排检索主线打基础。

---

## 1. Current Project Context

当前项目的主链路已经具备：

**sample → image loading → preprocess → local feature extraction → save placeholder → visualize placeholder**

其中：

* dataset loader 已能稳定读样本
* preprocess 已能统一输入图像
* `LocalFeatureResult` 等接口骨架已经存在
* `run_pipeline.py` 和 demo 已可运行

本 Issue 的职责不是实现完整检索，也不是提前做编码/匹配/倒排，而是：

> **先把局部特征提取本身做真。**

当前阶段建议优先支持：

* `SIFT`
* `ORB`

因为它们与课程主线、OpenCV 工程实现和你现有原型验证都比较对齐。

---

## 2. Issue Target

实现 **Issue 3.1 — Implement Local Feature Extraction**

目标是：

1. 在主工程中实现真实局部特征提取模块；
2. 至少支持 `SIFT` 和 `ORB`；
3. 输出结构化的：

   * `keypoints`
   * `descriptors`
   * `meta`
4. 将结果保存到 `outputs/features/*.npz`
5. 接入当前 `run_pipeline.py`
6. 为下一步关键点可视化（Issue 3.2）提供稳定输入。

---

## 3. Scope Boundaries

### In scope

本次只做以下内容：

* 在主工程中实现真实 local feature extraction
* 支持 `SIFT`
* 支持 `ORB`
* 统一特征输出结构
* 最小特征保存逻辑（`.npz`）
* 接入当前 pipeline
* 输出简洁统计信息

### Out of scope

本次 **不要实现** 以下内容：

* 图像匹配
* query-gallery 检索
* BFMatcher / FLANN 全库匹配流程
* RANSAC
* 特征编码 / codebook / tf-idf
* 倒排索引
* 重排序
* Web Demo 的复杂特征可视化 UI
* 任何性能优化（多进程、批处理等）

原则：

> **本 Issue 只做“从图像中提取并保存局部特征”这件事。**

---

## 4. Required File Changes

请优先在以下路径完成最小改动：

### Must create or update

* `src/features/local/local_feature_extractor.py`
* `src/features/local/__init__.py`
* `scripts/run_pipeline.py`

### May create if necessary

* `src/features/local/extract_local_features.py`（仅当确实有助于职责清晰）
* `src/utils/io.py` 或同等最小工具文件（仅在保存 `.npz` 时确实需要）
* `docs/design/pipeline_skeleton.md`（如需补充 local feature 阶段说明）
* `configs/base.yaml`

### Output location

请确保最小保存逻辑使用：

* `outputs/features/*.npz`

不要无故扩散到很多新文件。
本 Issue 的核心交付应尽量集中在 `local_feature_extractor.py` 与现有 pipeline 上。

---

## 5. Functional Requirements

### 5.1 Supported algorithms

至少支持两种算法：

* `SIFT`
* `ORB`

要求：

* 从配置中选择算法
* 若环境不支持某算法，应给出清晰错误信息
* 不要 silently fallback 到别的算法

### 5.2 Input

输入应为 preprocess 后的图像，即来自当前 `PreprocessResult.image`。

要求：

* 能处理 grayscale 输入
* 若传入 3 通道图像，也应稳妥处理（必要时转换为算法需要的格式）
* 不要依赖额外深度学习框架

### 5.3 Structured output

请沿用或补全当前骨架中的 `LocalFeatureResult`，建议至少包含：

* `keypoints`
* `descriptors`
* `meta`

其中：

#### keypoints

不要直接保存 OpenCV `KeyPoint` 对象列表作为唯一可序列化结果。
请转换成可保存、可复用的结构，例如 `Nx2` 或更丰富的数组/列表，至少包括位置；如你认为必要，也可包含：

* x
* y
* size
* angle
* response
* octave
* class_id

#### descriptors

直接保存 OpenCV 输出的 descriptor 矩阵。

#### meta

至少记录：

* `method`
* `num_keypoints`
* `descriptor_shape`
* `descriptor_dtype`
* 其他必要说明

### 5.4 Empty-result handling

必须清楚处理这类情况：

* 图像无有效关键点
* descriptor 为 `None`

要求：

* 不要崩溃
* 返回结构化空结果
* `meta` 中应能看出关键点数量为 0

---

## 6. Saving Requirement

请实现最小特征保存逻辑，将结果保存为：

* `outputs/features/<sample_id_or_safe_name>.npz`

保存内容建议至少包括：

* `keypoints`
* `descriptors`
* `method`
* `num_keypoints`

要求：

* 文件名要安全，不要直接使用包含路径分隔符的原始路径
* 若你需要从 sample_id 生成安全文件名，请做最小、可解释的转换
* 不要引入复杂数据库或缓存系统

---

## 7. Pipeline Integration Requirement

请将真实 local feature extraction 接入当前 `scripts/run_pipeline.py`。

要求：

1. 从 config 读取特征提取方法
2. 对前若干个样本执行真实特征提取
3. 打印简洁统计，例如：

   * method
   * number of keypoints
   * descriptor shape
   * save path
4. 不要打坏现有 preprocess、save placeholder 之外的主链路

运行：

```bash
python scripts/run_pipeline.py
```

时，用户应能看到：

* preprocess 已执行
* local feature 已真实提取
* 特征已保存到 `outputs/features/*.npz`

---

## 8. Config Requirements

如有必要，请在 `configs/base.yaml` 中补充最小字段，例如：

```yaml
local_feature:
  method: sift   # or orb
  save: true
  max_samples: 3
```

如果你认为还需要少量算法参数，也可以加，但必须保持最小，例如：

* `nfeatures`（ORB）
* `contrast_threshold`（SIFT）

要求：

* 只加入当前阶段真正需要的最少字段
* 不要做成庞大配置树
* 默认值要稳妥

---

## 9. Console Output Style

输出保持简洁明确，例如：

* `Local feature method: sift`
* `Extracted 128 keypoints for sample ...`
* `Descriptor shape=(128, 128)`
* `Saved features to outputs/features/...npz`

不要引入 logging 框架。
不要输出大段花哨文本。

---

## 10. Documentation Requirement

如果需要，请补充当前设计文档，说明：

1. local feature 阶段职责
2. 支持的算法
3. `LocalFeatureResult` 的输出格式
4. `.npz` 保存内容
5. 当前不做哪些后续逻辑（匹配、编码、检索等）

优先补已有文档，不要为了这一步再新增很多文档。

---

## 11. Design Constraints

### 11.1 Keep it stage-correct

不要把这一步扩展成图像匹配或检索。
当前只是提取和保存局部特征。

### 11.2 Respect existing skeleton

优先复用当前：

* `PreprocessResult`
* `LocalFeatureResult`
* `run_pipeline.py`

不要另起一套并行 pipeline。

### 11.3 Save serializable results

请确保输出是后续模块可读、可保存、可视化的，而不是只能在内存里存在的 OpenCV 对象。

### 11.4 Preserve demo compatibility

如果 demo 需要最小兼容更新，可以做，但不要大改 UI。
关键是不要破坏 Stage 1.4 的可运行性。

---

## 12. Acceptance Criteria

完成后应满足：

1. 已实现真实 local feature extraction 模块；
2. 至少支持 `SIFT` 和 `ORB`；
3. 可成功输出：

   * `keypoints`
   * `descriptors`
4. `run_pipeline.py` 可真实调用该模块；
5. 特征可保存到 `outputs/features/*.npz`；
6. 遇到空特征结果时不会崩溃；
7. 没有提前引入匹配、检索或编码逻辑；
8. 不破坏已有 preprocess / loader / demo 能力。

---

## 13. Expected Output From You

请直接完成代码修改，并同时提供：

1. 你新增/修改了哪些文件
2. 当前支持的局部特征算法
3. `LocalFeatureResult` 的结构说明
4. `.npz` 保存内容说明
5. `run_pipeline.py` 如何接入 local feature
6. 示例运行输出
7. 任意你采用的最小兼容或环境假设
8. 它如何满足 acceptance criteria

---

## 14. Important Non-Goals

请再次注意，这个 Issue 不是：

* 图像匹配实现
* query-gallery 检索
* 结果排序
* 倒排索引
* codebook / tf-idf
* Web 前端增强

它只是：

> **把“局部特征提取”从 placeholder 变成主工程中的真实可运行模块，并产出标准化特征文件。**
