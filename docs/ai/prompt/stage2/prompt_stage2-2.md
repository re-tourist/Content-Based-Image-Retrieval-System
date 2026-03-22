## Issue 2.2 — Image Preprocessing Pipeline

下面这份可以直接复制给 Codex。

---

# Codex Prompt — Issue 2.2 Image Preprocessing Pipeline

你正在协助实现一个课程驱动、工程化推进的 **Hybrid Image Retrieval System**。
当前项目已经完成：

* Milestone 1: System Skeleton
* Issue 2.1: Dataset Structure Standardization

现在进入 **Issue 2.2 — Image Preprocessing Pipeline**。
这一阶段的目标不是做复杂图像增强，也不是做训练型预处理，而是建立一个**最小、稳定、可配置、可复用**的图像预处理模块，为后续局部特征提取提供统一输入。

---

## 1. Current Project Context

当前项目的最小主线已经是：

**raw image / split-based sample → image loading → preprocess → local feature extraction**

Stage 2.1 已经把数据读取标准化为：

* 统一 split 文件
* dataset loader 可兼容 split 读取
* `run_pipeline.py` 仍可正常运行

现在 Issue 2.2 的任务，是把“图像进入局部特征模块前的输入格式”钉清楚。
这一步只需要解决**最基础的输入统一问题**，例如：

* resize
* color conversion
* format / shape check

不要把它扩成复杂预处理系统。

---

## 2. Issue Target

实现 **Issue 2.2 — Image Preprocessing Pipeline**

目标是：

1. 提供一个最小可用的图像预处理模块；
2. 支持基础 resize；
3. 支持颜色空间转换；
4. 支持基本格式检查；
5. 让后续局部特征提取能够拿到更稳定、统一的输入；
6. 与当前 `run_pipeline.py` 和 Web Demo 保持兼容。

---

## 3. Scope Boundaries

### In scope

本次只做以下内容：

* resize
* color conversion
* format / dimension check
* 最小配置支持
* 预处理结果结构化输出
* 接入当前 pipeline skeleton

### Out of scope

本次 **不要实现** 以下内容：

* 数据增强
* 归一化到 tensor / 深度学习训练预处理
* 复杂缓存
* 批处理优化
* 去噪、锐化、CLAHE 等高级图像处理
* 大规模离线预处理落盘系统
* 任何局部特征提取逻辑
* 任何检索算法逻辑

原则：

> **只做“统一图像输入”的最小预处理模块。**

---

## 4. Required File Changes

请优先在以下路径完成最小改动：

### Must create or update

* `src/preprocess/basic_preprocess.py`
* `src/preprocess/__init__.py`
* `scripts/run_pipeline.py`

### May update if necessary

* `configs/base.yaml`
* `src/demo/web_demo.py`
* `docs/design/pipeline_skeleton.md`

如果当前仓库里已有与 preprocess 相关的骨架文件，请优先在原位置补齐，而不是新造并行文件。

**不要把实现放到 `src/utils/preprocess.py`，优先沿用当前 pipeline skeleton 的 `src/preprocess/` 模块边界。**

---

## 5. Functional Requirements

### 5.1 Input

输入应支持当前 pipeline 已有的图像格式，即 OpenCV 读取后的 `numpy.ndarray` 图像。

### 5.2 Resize

实现基础 resize，要求：

* 支持配置目标尺寸，例如：

  * `height`
  * `width`
* 支持“不开 resize”的最小兼容模式
* 默认行为应清楚，不要隐式做危险改动

建议支持的最小模式：

* 指定 `(height, width)` 直接 resize
* 或 `enabled: false` 时保持原图

### 5.3 Color conversion

实现基础颜色空间转换，至少支持：

* 保持原样
* 转为 grayscale

如果你认为有必要，也可以支持：

* BGR → RGB

但请保持最小，不要扩太多。

当前更推荐明确支持：

* `color_mode = "keep"`
* `color_mode = "gray"`

### 5.4 Format / shape check

请做最基本的输入检查，例如：

* 图像是否为空
* 是否为 ndarray
* shape 是否合理
* 通道数是否可接受

错误信息应清晰直接。

### 5.5 Structured output

预处理结果不要只返回裸图像，建议继续沿用结构化输出，类似：

* `PreprocessResult`

  * `image`
  * `original_shape`
  * `processed_shape`
  * `color_mode`
  * `meta`

如果项目里已有 `PreprocessResult` 骨架，请在原有基础上补齐，不要重复定义一套。

---

## 6. Config Requirements

如果需要，请在 `configs/base.yaml` 中加入最少必要字段，例如：

```yaml
preprocess:
  resize:
    enabled: true
    width: 256
    height: 256
  color_mode: gray
```

要求：

* 只增加当前阶段真正需要的字段
* 不要设计成最终版复杂配置系统
* 默认值要稳妥

---

## 7. Pipeline Integration Requirement

请把预处理模块接入当前 `scripts/run_pipeline.py`，要求：

1. 从 config 读取 preprocess 配置
2. 对读入图像执行 preprocess
3. 输出简洁的阶段状态信息，例如：

   * original shape
   * processed shape
   * color mode
4. 不要打坏后续 placeholder 的 local feature / save / visualize 调用链

也就是说，运行：

```bash
python scripts/run_pipeline.py
```

时，用户应能明确看到：

* 图像已被预处理
* 预处理结果形状/模式已统一
* pipeline 主链路仍然可跑通

---

## 8. Demo Compatibility Requirement

如果当前 Web Demo 使用到了 preprocess，请做最小兼容更新，保证：

* Demo 仍可运行
* 上传图像后可以走新的 preprocess 模块
* 不需要大改 UI

如果 Demo 当前没有明确显示 preprocess 细节，也没关系，只要不打坏 Stage 1.4 已有能力即可。

---

## 9. Console Output Style

输出保持简洁明确，例如：

* `Loaded image with shape=(450, 450, 3)`
* `Preprocess: resize enabled -> (256, 256)`
* `Preprocess: color_mode=gray`
* `Processed image shape=(256, 256)`

不要引入 logging 框架。
不要输出大段花哨文本。

---

## 10. Documentation Requirement

如果确有必要，请补充或更新现有设计文档，说明：

1. 当前 preprocess 模块职责
2. 支持的预处理步骤
3. 输入输出格式
4. 与 pipeline skeleton 的关系
5. 当前不做哪些高级处理

优先补已有文档，不要为了这一步再额外造很多文档。

---

## 11. Design Constraints

### 11.1 Keep it minimal

不要把这一步做成完整图像处理库。
当前只是统一输入。

### 11.2 Respect stage boundaries

不要提前实现：

* Stage 3 的局部特征算法
* 特征保存格式
* 可视化逻辑
* 检索算法逻辑

### 11.3 Preserve current skeleton usability

不要破坏已有：

* split-based dataset loading
* `scripts/run_pipeline.py`
* `scripts/run_demo.py`

### 11.4 Honest preprocessing

如果当前只做了 resize + gray + check，就明确保持这个范围。
不要为了“看起来强”而加很多无关处理。

---

## 12. Acceptance Criteria

完成后应满足：

1. 已实现最小预处理模块；
2. 支持 resize；
3. 支持基础颜色转换；
4. 支持基本格式检查；
5. `run_pipeline.py` 可调用 preprocess 并正常运行；
6. 预处理输出结构清晰，便于后续 local feature 模块接入；
7. 没有引入不必要复杂度；
8. 不破坏已有 Stage 1 与 Stage 2.1 的能力。

---

## 13. Expected Output From You

请直接完成代码修改，并同时提供：

1. 你新增/修改了哪些文件
2. preprocess 模块的核心设计
3. 支持哪些预处理步骤
4. 配置字段说明
5. `run_pipeline.py` 如何接入 preprocess
6. 示例运行输出
7. 是否对 demo 做了最小兼容处理
8. 它如何满足 acceptance criteria

---

## 14. Important Non-Goals

请再次注意，这个 Issue 不是：

* 完整图像增强系统
* 深度学习预处理系统
* 离线批处理数据工程
* 局部特征提取实现
* 检索系统实现

它只是：

> **在进入 Stage 3 之前，把“图像进入特征模块前应该长什么样、如何被统一处理”这件事钉清楚。**
