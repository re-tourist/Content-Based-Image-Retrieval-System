（GPT-5.4 Thinking）

结论先说：

> **Stage 1.4 可以通过，进入 Issue 2.1，不需要先修补。**

原因有两部分。

第一，反馈文本表明这次实现已经覆盖了 Issue 1.4 的核心目标：

* 已创建 `scripts/run_demo.py`、`src/demo/web_demo.py` 等最小 Demo 骨架文件。
* 采用了 Gradio，且复用了当前 skeleton 的 `load_config`、`preprocess_image(...)`、`extract_local_features(...)`，没有另起一套平行架构。
* 已有上传、预览、按钮、状态区、Top-K placeholder Gallery，这和 1.4 的范围完全对齐。
* 也明确区分了哪些是真实接通、哪些仍是 placeholder，没有偷跑去做真实 retrieval。

第二，你给的界面截图本身说明 **Demo 已经实际启动并在本地跑起来了**：页面标题、上传区、按钮、状态区和 Top-K placeholder 区域都真实存在，而且运行在 `localhost:7860`。这正好补上了反馈里“因本地 `gradio` 安装阻塞，未能完成启动验证”的那一点不确定性。也就是说：

> **从最终证据看，Stage 1.4 的本地启动验收已经被你的截图补全了。**

所以这一步不该再回头打磨 M1，而应该正式进入 **Milestone 2 / Issue 2.1 — Dataset Structure Standardization**。

---

## 一、我对 Stage 1.4 的审核判断

### 通过点

这一步通过，不是因为“看起来像个网页”，而是因为它满足了工期1真正要做的事情：

* 给当前 skeleton 提供了一个**最小可演示外壳**
* 没有把 demo 做成重型前后端
* 没有绕开已有模块边界
* 没有提前偷跑 M2/M3 的算法内容

这就符合工期1“先把系统骨架立住，再进入数据层与局部特征主线”的原则。你前面的项目规划也是这么定的：工期1是 system skeleton，工期2才是 data preparation，工期3才是 local feature extraction。

### 仅有的小问题

唯一的小瑕疵是：
反馈文本里提到 `gradio` 安装当时被本机环境阻塞。
但这个问题已经被你上传的界面截图实际覆盖了，所以它现在不构成阻塞 issue，只是一个**环境记录点**，以后可写进问题记录或实验记录里。

---

## 二、现在为什么应该进入 Issue 2.1

因为 M1 的四个 issue 本质上已经形成闭环了：

* 1.1 数据入口
* 1.2 主脚本入口
* 1.3 pipeline skeleton 和接口
* 1.4 最小 Web Demo

到这里，**“系统骨架阶段”已经够用了**。
再继续修补，只会让你停留在骨架层反复打磨。

真正合理的下一步，就是进入 **M2：Data Preparation**，先把数据组织方式标准化。你给的 Issue 2.1 目标也很清楚：统一 `data/raw`、`data/processed`、`data/splits`，并生成 `train.txt / gallery.txt / query.txt` 这类清单文件，让 dataset loader 后续可以稳定读取。这个方向是对的。

---

# 给 Codex 的下一个 Prompt

## Issue 2.1 — Dataset Structure Standardization

下面这份可以直接复制给 Codex。

---

# Codex Prompt — Issue 2.1 Dataset Structure Standardization

你正在协助实现一个课程驱动、工程化推进的 **Hybrid Image Retrieval System**。
当前项目已经完成 **Milestone 1: System Skeleton** 的基础骨架，包括：

* dataset loader
* main script entry
* pipeline skeleton
* minimal web demo

现在进入 **Milestone 2: Data Preparation**。
这一阶段的目标不是做复杂算法，而是先把**数据层整理干净、统一、可被后续模块稳定读取**。

## 1. Current Project Context

当前项目后续的传统检索主线需要依赖稳定的数据组织方式。
在进入真实预处理、局部特征提取、编码和索引之前，必须先明确：

* 原始数据放在哪里
* 标准化后的数据结构是什么
* `train / gallery / query` 如何组织
* dataset loader 如何读取 split 清单

本 Issue 的目标不是重新设计完整数据平台，而是建立一个**最小、清晰、稳定的数据目录约定和 split 清单机制**，为工期2和工期3做准备。

---

## 2. Issue Target

实现 **Issue 2.1 — Dataset Structure Standardization**

目标是：

1. 明确并建立标准数据目录结构
2. 定义 `train / gallery / query` 的 split 清单文件格式
3. 提供生成这些清单文件的最小脚本或工具
4. 让现有 dataset loader 可以基于标准化结构继续工作
5. 在 docs 中写清楚当前项目的数据组织约定

---

## 3. Scope Boundaries

### In scope

本次只做以下内容：

* 标准化 `data/` 目录结构
* 明确 `raw / processed / splits` 三层约定
* 生成最小 split 文件：

  * `train.txt`
  * `gallery.txt`
  * `query.txt`
* 定义 split 文件中每行记录的格式
* 让 dataset loader 至少能够读取这些 split 文件中的路径信息，或为此预留清楚接口
* 补充数据组织说明文档

### Out of scope

本次 **不要实现** 以下内容：

* 真正复杂的数据清洗
* 去重算法
* 损坏图片检测全套系统
* 真实图像预处理流水线
* 数据增强
* 深度学习训练 DataLoader
* 复杂标注系统
* 复杂数据库式元数据管理

原则：

> **本 Issue 只统一数据结构与 split 清单机制，不做复杂数据工程。**

---

## 4. Required File Changes

请优先在以下路径完成最小改动：

### Must create or update

* `data/raw/`（如缺少则补齐目录约定）
* `data/processed/`（如缺少则补齐目录约定）
* `data/splits/`
* `data/splits/train.txt`
* `data/splits/gallery.txt`
* `data/splits/query.txt`
* `scripts/build_splits.py`
* `docs/design/dataset_structure.md`

### May update if necessary

* `src/datasets/dataset_loader.py`
* `configs/base.yaml`

但注意：

* 只在确有必要时修改 loader 和 config
* 不要把这个 issue 扩展成大型重构

---

## 5. Required Data Structure

请明确并实现当前项目的数据目录约定，至少包括：

```text
data/
├─ raw/
├─ processed/
└─ splits/
   ├─ train.txt
   ├─ gallery.txt
   └─ query.txt
```

要求：

### 5.1 raw

用于放原始图像数据。
当前阶段可以只做目录约定和最小兼容，不要求移动所有历史文件到完美状态。

### 5.2 processed

用于后续保存预处理后的产物。
当前阶段可以为空，但必须建立约定。

### 5.3 splits

用于存放数据清单文件。
这是本 Issue 的核心交付之一。

---

## 6. Split File Requirements

请定义 `train.txt / gallery.txt / query.txt` 的最小格式。

建议每行至少记录：

* 图像相对路径

例如：

```text
raw/train/image_001.jpg
raw/train/image_002.jpg
```

或者相对于某个 data root 的统一相对路径格式。
关键要求：

* 格式清晰
* 简单稳定
* 后续 dataset loader 易于读取

如果你认为有必要加入额外字段，例如：

* `sample_id`
* `label`
* `split`

也可以，但必须保持最小、直接，不要过度设计。

**推荐优先采用：每行一个相对路径。**

---

## 7. Split Building Script

请实现：

* `scripts/build_splits.py`

其职责是：

1. 扫描指定原始数据目录
2. 按最小规则生成：

   * `train.txt`
   * `gallery.txt`
   * `query.txt`
3. 打印生成统计信息

### Requirements

* 支持最小可配置的数据根目录
* 输出简洁统计，例如：

  * `Generated train.txt with 900 entries`
  * `Generated gallery.txt with 50 entries`
  * `Generated query.txt with 20 entries`
* 行为简单可解释
* 不要引入复杂随机切分逻辑，除非当前数据组织确实需要
* 如果当前仓库数据现状不完全匹配理想结构，请做**最小兼容实现**，并在说明中明确假设

---

## 8. Dataset Loader Integration

请让现有 dataset loader 与 split 机制形成最小兼容。可以采用以下两种方式之一：

### Option A（推荐）

在现有 loader 中增加“从 split 文件加载样本路径”的最小支持。

### Option B

暂时不改 loader 主逻辑，但提供清楚的辅助函数/接口，让后续能够基于 split 文件创建样本记录。

要求：

* 不要破坏 Stage 1 已有可运行能力
* 不要把 loader 重构成复杂系统
* 重点是让“dataset loader 可读取标准化 split 信息”这件事成立

---

## 9. Documentation Requirement

请新增：

* `docs/design/dataset_structure.md`

内容至少包括：

1. 当前项目的数据目录约定
2. `raw / processed / splits` 的职责
3. `train / gallery / query` 的含义
4. split 文件格式说明
5. `scripts/build_splits.py` 如何使用
6. dataset loader 如何与 split 对接
7. 当前实现有哪些最小假设或兼容策略

文档风格要求：

* Markdown
* 工程说明风格
* 结构清楚
* 可直接作为后续 issue 的数据层参考

---

## 10. Config Update Requirement

如果当前项目需要，请在 `configs/base.yaml` 中加入最小必要字段，例如：

* `data.root`
* `data.raw_dir`
* `data.processed_dir`
* `data.splits_dir`
* `data.train_split`
* `data.gallery_split`
* `data.query_split`

要求：

* 只加当前阶段真正需要的最少字段
* 不要设计成最终版配置中心

---

## 11. Design Constraints

### 11.1 Keep it minimal

不要把这一步做成完整数据平台。
当前只是统一目录结构和 split 文件。

### 11.2 Respect current stage boundaries

不要提前实现：

* 工期2.2 的完整预处理逻辑
* 工期3 的局部特征提取逻辑
* 工期4 之后的任何检索算法逻辑

### 11.3 Be honest about assumptions

如果当前仓库已有数据分布与理想结构不完全一致，请采用最小兼容方案，并在输出说明和文档中明确写清楚。

### 11.4 Preserve existing skeleton usability

不要破坏已有的：

* `scripts/run_pipeline.py`
* `scripts/run_demo.py`
* dataset loader 最小可运行逻辑

---

## 12. Acceptance Criteria

完成后应满足：

1. 项目具备清晰的数据目录结构约定：

   * `data/raw`
   * `data/processed`
   * `data/splits`
2. 已生成：

   * `train.txt`
   * `gallery.txt`
   * `query.txt`
3. split 文件格式清晰简单，可被后续模块读取
4. dataset loader 可以读取这些 split 信息，或至少具备清晰兼容接口
5. `scripts/build_splits.py` 可运行并输出基本统计
6. 有对应文档说明数据组织方式
7. 没有引入与当前阶段不相称的复杂数据工程系统

---

## 13. Expected Output From You

请直接完成代码修改，并同时提供：

1. 你新增/修改了哪些文件
2. 当前采用的数据目录结构
3. split 文件格式说明
4. `build_splits.py` 的使用方式
5. dataset loader 如何与 split 对接
6. 示例生成输出
7. 任何你采用的最小兼容假设
8. 它如何满足 acceptance criteria

---

## 14. Important Non-Goals

请再次注意，这个 Issue 不是：

* 完整数据清洗系统
* 完整预处理 pipeline
* 标注与元数据系统
* 深度学习数据管线
* 检索算法实现

它只是：

> **为后续工期2和工期3，把“数据怎么放、怎么列清单、loader 怎么按标准读取”这件事彻底钉清楚。**

---

## 我对你下一轮审核的重点

等 Codex 做完 Issue 2.1，我会重点帮你审：

* 它是不是把 split 机制做得**简单、稳定、可读**
* 它有没有**最小兼容当前仓库已有数据现实**
* 它是不是改坏了 Stage 1 已经能跑的 loader / pipeline / demo
* 它有没有偷跑去做 2.2 的预处理逻辑
* 文档是否已经足够支撑后面工期推进

如果你愿意，下一条我还可以顺手给你一份 **Issue 2.1 的 review checklist**，方便你审 Codex 输出。
