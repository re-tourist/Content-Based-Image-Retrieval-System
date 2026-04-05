## Issue 1.4 — Build Minimal Web Demo Interface

下面这份你可以直接复制给 Codex。

---

# Codex Prompt — Issue 1.4 Build Minimal Web Demo Interface

你正在协助实现一个课程驱动、工程化推进的 **Hybrid Image Retrieval System**。
当前项目处于 **Milestone 1: System Skeleton**。这一阶段的目标不是开发完整前端系统，也不是实现完整检索服务，而是建立一个**最小可运行的 Web Demo 界面骨架**，用于展示当前 pipeline skeleton、支持课程演示、并为后续真实检索结果可视化预留入口。

## 1. Current Project Context

当前项目已经具备以下骨架能力：

* config 读取
* dataset loader
* image loading
* basic preprocess stage
* local feature / save / visualize placeholder stages
* `scripts/run_pipeline.py` 可运行

当前最小闭环是：

**query image input → pipeline skeleton trigger → placeholder retrieval result area**

因此，本 Issue 的重点不是做完整前后端架构，而是提供一个**轻量、可运行、易演示**的 Web UI，让系统第一次具备基础的图像交互界面。

## 2. Issue Target

实现 **Issue 1.4 — Build Minimal Web Demo Interface**

目标是：

1. 提供 query image 上传入口；
2. 显示用户上传图片的预览；
3. 通过按钮触发当前 pipeline skeleton 或对应 stub；
4. 展示占位的 Top-K 结果区域；
5. 为后续真实检索结果和关键点/特征可视化预留扩展接口；
6. 在 README 或 docs 中补充最小运行说明。

## 3. Technology Choice

优先使用 **Gradio**，不要使用重量级 Web 技术栈。

选择理由：

* 与 Python pipeline 集成简单
* 适合图片输入与图片结果展示
* 很适合科研 demo / 课程展示
* 当前阶段不需要复杂前后端分离

除非仓库已有更明确的技术约束，否则请默认用 **Gradio**。

## 4. Scope Boundaries

### In scope

本次只做最小 Demo 界面骨架，包括：

* 一个可启动的 Gradio 界面
* query image 上传组件
* 上传图片预览
* 一个 “Run Retrieval” 或同等按钮
* 调用当前 pipeline skeleton 的最小桥接函数
* 一个 placeholder Top-K 结果展示区域
* 少量说明文字
* 最小运行说明文档

### Out of scope

本次 **不要实现** 以下内容：

* 完整前端工程
* React / Vue / 前后端分离
* 用户系统
* 上传历史
* 数据库存储
* 真正的检索后端服务
* 真实 Top-K 检索逻辑
* 复杂 CSS/UI 美化
* 生产级部署方案
* API server 架构重构

原则：

> **只做一个最小、能运行、能演示、能接后续真实检索结果的 Web Demo 骨架。**

## 5. Required File Changes

请优先在以下路径完成：

### Must create or update

* `scripts/run_demo.py`
* `src/demo/__init__.py`
* `src/demo/web_demo.py`
* `docs/design/web_demo.md`

### May update if necessary

* `requirements.txt`（仅当当前仓库还没有 Gradio 时，再补充最小依赖）
* `README.md`（仅补最小 demo 启动说明，不要大改）

不要无故新增很多 demo 文件。
本 Issue 应尽量保持轻量。

## 6. Functional Requirements

### 6.1 Upload input

界面必须支持上传一张 query image。

要求：

* 支持常见图像格式
* 上传后可显示预览
* 不需要复杂校验 UI，但应有基本异常提示

### 6.2 Query image preview

上传后，页面应能显示 query image 本身。

### 6.3 Run button

界面应提供一个按钮，例如：

* `Run Retrieval`
* 或 `Run Pipeline Skeleton`

点击后触发当前阶段的最小处理逻辑。

### 6.4 Pipeline bridge

请实现一个轻量桥接函数，将 Web Demo 与当前 pipeline skeleton 接起来。

要求：

* 不要直接把 `scripts/run_pipeline.py` 当模块乱 import 使用，如果结构不适合，请做一个最小可复用桥接函数
* 可以在 `src/demo/web_demo.py` 中调用现有模块能力，完成：

  * 读取上传图片
  * 执行最小 preprocess
  * 执行 local feature placeholder
  * 返回 demo 结果对象

注意：

* 当前不要求真正检索整个图库
* 当前更像是“以 query image 为输入，跑通当前 skeleton，并向 UI 返回占位结果”

### 6.5 Placeholder Top-K results

页面必须有一个 **Top-K Results** 区域，但当前允许是 placeholder。

建议行为：

* 返回 3~5 个占位结果卡片/图片位
* 或返回固定的 placeholder 图像/文本
* 或返回“future retrieval results will appear here”风格的占位信息

但要注意：

* 这个区域必须在界面上真实存在
* 用户能明显看出这里就是未来检索结果展示区

### 6.6 Future extensibility hooks

请在代码中通过清晰注释或极轻量接口预留：

* real retrieval output integration
* keypoint visualization integration
* local feature visualization integration
* future Top-K ranking results

但只允许预留位置，不要实现真实逻辑。

## 7. Suggested UI Layout

界面建议尽量简洁，例如：

* 标题：Hybrid Image Retrieval System Demo
* 左侧：

  * Query Image Upload
  * Image Preview
  * Run button
* 右侧：

  * Pipeline status / placeholder summary
  * Top-K Results area

或者用更简单的一列式布局也可以。
只要清楚、可演示即可。

## 8. Runtime Behavior

用户运行：

```bash
python scripts/run_demo.py
```

后应能：

1. 在本地启动 demo
2. 打开 Web UI
3. 上传图片
4. 看到图片预览
5. 点击按钮
6. 看到 pipeline skeleton 被触发后的状态输出
7. 看到 placeholder 的 Top-K 结果区域

## 9. Design Constraints

### 9.1 Keep it minimal

不要把这一步做成最终版产品 UI。
它只是工期1的最小演示界面。

### 9.2 Do not bypass current architecture

不要重新发明一套与当前 pipeline skeleton 平行的流程。
要尽量复用当前已有模块边界。

### 9.3 Do not overbuild backend

不要引入 Flask/FastAPI + 前端分离，除非仓库当前已经强依赖这些。
当前阶段 Gradio 足够。

### 9.4 Honest placeholders

如果结果区是 placeholder，就明确写清楚是 placeholder。
不要伪装成真实检索结果。

## 10. Documentation Requirement

请新增：

* `docs/design/web_demo.md`

内容至少包括：

1. 当前 demo 的目标
2. 为什么选择 Gradio
3. 页面包含哪些组件
4. demo 与当前 pipeline skeleton 的关系
5. 哪些部分是真实接通的
6. 哪些部分仍是 placeholder
7. 如何本地运行 demo

文档风格要求：

* Markdown
* 结构清楚
* 工程说明风格
* 可作为后续真实检索展示扩展的基础说明

## 11. Acceptance Criteria

完成后应满足：

1. Demo 界面可以在本地启动
2. 用户可以通过界面上传图片
3. 页面能够显示上传图片预览
4. 点击按钮可以触发当前 pipeline skeleton 或 stub
5. 页面能够展示占位的 Top-K 检索结果区域
6. 代码结构为未来真实检索结果与可视化模块预留扩展接口
7. 没有引入与当前阶段不相称的复杂前后端架构

## 12. Expected Output From You

请直接完成代码修改，并同时提供：

1. 你新增/修改了哪些文件
2. Web Demo 的技术选型说明
3. UI 主要组件说明
4. Demo 如何与当前 pipeline skeleton 对接
5. 哪些部分是真实接通的，哪些是 placeholder
6. 本地运行方式
7. 示例启动输出或示例交互说明
8. 它如何满足 acceptance criteria

## 13. Important Non-Goals

请再次注意，这个 Issue 不是：

* 完整前端系统
* 完整后端服务
* 真正的图像检索网页产品
* 真实 Top-K 检索实现
* 工期2/3/4之后的功能提前实现

它只是：

> **在工期1中，为当前系统骨架提供一个最小、可演示、可交互的 Web Demo 外壳。**



