## Issue 3.2 — Keypoint Visualization

下面这份你可以直接复制给 Codex。

---

# Codex Prompt — Issue 3.2 Keypoint Visualization

你正在协助实现一个课程驱动、工程化推进的 **Hybrid Image Retrieval System**。
当前项目已经完成：

* Milestone 1: System Skeleton
* Milestone 2: Data Preparation
* Issue 3.1: Real local feature extraction with SIFT / ORB and `.npz` feature saving

现在进入 **Issue 3.2 — Keypoint Visualization**。
这一阶段的目标，是把已经真实提取出来的局部特征转换为**可展示、可检查、可用于课程演示和调试**的关键点可视化结果。

---

## 1. Current Project Context

当前项目主链路已经具备：

**sample → image loading → preprocess → local feature extraction → feature save**

其中：

* preprocess 已经提供统一输入
* local feature extraction 已经真实支持 `SIFT` 和 `ORB`
* `LocalFeatureResult` 已可提供序列化的 `keypoints` 和 `descriptors`
* `.npz` 特征文件已落到 `outputs/features/*.npz`

本 Issue 的职责不是实现匹配可视化，也不是做复杂 UI，而是：

> **把单张图像上的关键点直观画出来，并保存为标准 figure 输出。**

这一步既服务于课程展示，也服务于你自己检查特征提取是否合理。

---

## 2. Issue Target

实现 **Issue 3.2 — Keypoint Visualization**

目标是：

1. 基于当前真实 local feature extraction 结果实现关键点可视化；
2. 支持将关键点绘制到图像上；
3. 将可视化结果保存到：

   * `outputs/figures/keypoints_*.png`
4. 接入当前 `run_pipeline.py`
5. 为后续 demo 或报告展示提供基础素材。

---

## 3. Scope Boundaries

### In scope

本次只做以下内容：

* 单图关键点可视化
* 从当前提取结果中读取关键点信息
* 生成带关键点覆盖的图像
* 保存 `.png` 结果到 `outputs/figures/`
* 在 pipeline 中最小接入
* 输出简洁统计信息

### Out of scope

本次 **不要实现** 以下内容：

* 图像匹配连线可视化
* query-gallery 对比图
* RANSAC 内点可视化
* Web Demo 中复杂交互式特征显示
* 复杂前端展示增强
* 特征编码、检索、排序
* 批量大规模可视化系统
* 高级绘图风格系统

原则：

> **本 Issue 只做“把单张图像的关键点画出来并保存”的最小可视化模块。**

---

## 4. Required File Changes

请优先在以下路径完成最小改动：

### Must create or update

* `src/visualization/keypoints.py`
* `scripts/run_pipeline.py`

### May create if necessary

* `src/visualization/__init__.py`
* `docs/design/pipeline_skeleton.md`（如需补充 visualize 阶段说明）
* `src/demo/web_demo.py`（仅当你能以极小改动兼容展示结果时，否则不要强行改）

### Output location

请确保输出图像保存到：

* `outputs/figures/keypoints_*.png`

不要无故新增很多绘图文件。
本 Issue 的核心交付应集中在 `keypoints.py` 与现有 pipeline 接入。

---

## 5. Functional Requirements

### 5.1 Input source

可视化应基于当前主工程已有输入：

* 原始图像或 preprocess 后图像
* `LocalFeatureResult.keypoints`

优先使用当前 pipeline 内已经拿到的结果，不要另起一套读取流程。

### 5.2 Visualization behavior

请实现单张图像关键点绘制，要求：

* 在图像上绘制关键点位置
* 输出直观可检查
* 对关键点较多的图像也能正常显示
* 空关键点情况也能生成合理结果，不崩溃

### 5.3 Implementation choice

优先使用 OpenCV 提供的关键点绘制能力，或基于 matplotlib / OpenCV 做最小实现。

如果你直接使用 OpenCV 绘制，请确保能够从当前保存的 `Nx7` 关键点数组恢复出绘制所需信息；如果恢复 `cv2.KeyPoint` 麻烦，也可以采用“只画点/圆”的最小实现，但要满足：

* 能清楚看到关键点分布
* 不依赖不可序列化对象
* 与当前 `LocalFeatureResult.keypoints` 兼容

### 5.4 Empty-feature handling

如果关键点数量为 0，要求：

* 不要崩溃
* 仍可保存一张图
* 可在输出中保留原图，或附加简洁提示
* console 输出应明确说明 keypoints=0

### 5.5 Save output

请将结果保存为：

* `outputs/figures/keypoints_<safe_sample_name>.png`

要求：

* 文件名安全
* 与 `sample_id` 或样本名有稳定对应关系
* 不要引入复杂命名系统

---

## 6. Pipeline Integration Requirement

请将关键点可视化接入当前 `scripts/run_pipeline.py`。

要求：

1. 在 local feature extraction 之后调用 visualization
2. 对前若干个样本执行可视化
3. 输出简洁信息，例如：

   * `Visualized 118 keypoints for sample ...`
   * `Saved figure to outputs/figures/keypoints_xxx.png`
4. 不要打坏现有：

   * preprocess
   * local feature extraction
   * feature save

运行：

```bash
python scripts/run_pipeline.py
```

时，用户应能看到：

* 特征已提取
* 可视化已执行
* figure 文件已保存

---

## 7. Config Requirements

如有必要，请在 `configs/base.yaml` 中加入最少字段，例如：

```yaml
visualization:
  enabled: true
  save_keypoints: true
```

如果你认为还需要极少量配置，例如是否使用 preprocess 图像作为底图，也可以加，但必须保持最小。

要求：

* 只增加当前阶段真正需要的字段
* 不要扩成复杂可视化配置树

---

## 8. Console Output Style

输出保持简洁明确，例如：

* `Keypoint visualization enabled`
* `Visualized 118 keypoints for sample sample_xxx`
* `Saved keypoint figure to outputs/figures/keypoints_sample_xxx.png`

不要引入 logging 框架。
不要输出大段花哨文本。

---

## 9. Documentation Requirement

如果需要，请补充当前设计文档，说明：

1. keypoint visualization 阶段职责
2. 输入输出格式
3. 可视化基于什么数据结构
4. 输出文件命名方式
5. 当前不做哪些内容（匹配可视化、RANSAC、检索结果展示等）

优先补已有文档，不要为了这一步新增很多文档。

---

## 10. Design Constraints

### 10.1 Keep it stage-correct

不要把这一步扩展成匹配可视化或检索展示。
当前只是单图关键点可视化。

### 10.2 Reuse current local feature outputs

优先复用当前 `LocalFeatureResult.keypoints`，不要另起并行格式。

### 10.3 Save usable figures

输出必须是真正可打开、可检查、可放进报告/演示中的图片，而不是仅在内存中显示。

### 10.4 Preserve existing skeleton and demo usability

不要破坏：

* `run_pipeline.py`
* local feature extraction
* feature saving
* demo 的最小可运行能力

如果 demo 不需要更新，就不要为了“更完整”而强行改它。

---

## 11. Acceptance Criteria

完成后应满足：

1. 已实现关键点可视化模块；
2. 能基于当前 local feature extraction 结果绘制关键点；
3. 能保存：

   * `outputs/figures/keypoints_*.png`
4. `run_pipeline.py` 可调用该模块并正常运行；
5. 空关键点情况不会崩溃；
6. 没有引入匹配、检索或其他超出阶段范围的逻辑；
7. 不破坏已有 preprocess / loader / demo / feature save 能力。

---

## 12. Expected Output From You

请直接完成代码修改，并同时提供：

1. 你新增/修改了哪些文件
2. 当前关键点可视化的实现方式
3. 输入使用了哪些数据
4. 输出文件命名与保存方式
5. `run_pipeline.py` 如何接入 visualize
6. 示例运行输出
7. 任意你采用的最小兼容假设
8. 它如何满足 acceptance criteria

---

## 13. Important Non-Goals

请再次注意，这个 Issue 不是：

* 图像匹配可视化
* query-gallery 检索结果展示
* RANSAC 内点图
* Web 前端增强
* 编码 / 倒排 / 检索系统实现

它只是：

> **把已经真实提取出来的关键点变成可检查、可保存、可展示的图像结果。**

