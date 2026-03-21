## 一、对 Stage1-1 的审核结论

### 1. 通过点

从反馈看，以下方面是对齐的：

* 定义了统一样本记录 `ImageSample`，字段也符合预期。
* `ImageDatasetLoader` 具备你要求的最小接口：扫描、过滤、`__len__`、`__getitem__`、`__iter__`、`preview()`、`stats()`。
* 提供了 `load_image()`，并且和后续 OpenCV 局部特征路线保持一致。
* 运行时有基本统计输出，且没有扩展成训练型 DataLoader。
* 本地最小验证通过。

### 2. 可能的小瑕疵

有两个地方值得你在 code review 时顺手看一眼，但它们**不像必须卡住 Issue 1.1 的问题**：

第一，反馈里说支持“递归扫描”。这未必是坏事，但你原 issue 只说“扫描指定目录”，没有强制递归。递归本身可以接受，只要行为清楚、默认逻辑不迷惑即可。

第二，初始化时“默认打印加载数量和支持格式”。这在当前阶段能接受，但后面进入脚本入口和 pipeline 时，最好把“是否打印”交给脚本层控制，而不是 dataset loader 过于主动输出。

### 3. 我的决策

**结论：进入 Issue 1.2，不必先做专门修补。**

理由很简单：
Issue 1.1 的目标是建立最小数据入口，不是把数据层打磨成最终版。现在它已经足以支撑下一个闭环步骤：由主脚本加载 config、构造 loader、顺序读取样本并打印/验证基本信息。这个正好和 Issue 1.2 对接。

---

## 二、你现在该做什么

最合理的推进顺序是：

**直接进入 Issue 1.2 — Create Main Script Entry**

因为从工程依赖关系上讲：

* Issue 1.1 解决“数据入口”
* Issue 1.2 解决“统一运行入口”
* Issue 1.3 再解决“pipeline skeleton / module interfaces”

这条顺序是顺的。你前面的 milestone 拆分也是这个逻辑。

---

## 三、给 Codex 的下一个优质 Prompt（Issue 1.2）

下面这份你可以直接复制给 Codex。

---

# Codex Prompt — Issue 1.2 Create Main Script Entry

你正在协助实现一个课程驱动、工程化推进的 **Hybrid Image Retrieval System**。
当前项目处于 **Milestone 1: System Skeleton**，目标是建立最小系统骨架，而不是提前实现复杂检索系统。

## 1. Current Milestone Context

当前阶段的最小闭环是：

**原始图像 → 数据读取 → 基础预处理（当前可留空占位）→ 局部特征提取（当前未实现）→ 特征保存（当前未实现）→ 关键点可视化（当前未实现）**

当前已经有一个最小数据入口模块（dataset loader）。
本 Issue 的目标不是实现完整 pipeline，而是建立一个**统一、清晰、可运行的主脚本入口**，让项目具备“从 config 启动并顺序跑通前半段流程”的骨架能力。

---

## 2. Issue Target

实现 **Issue 1.2 — Create Main Script Entry**

目标是创建一个基础运行脚本：

* 能加载配置；
* 能初始化 dataset loader；
* 能顺序迭代图像样本；
* 能输出基本运行信息；
* 为后续 pipeline 模块接入保留清晰位置。

---

## 3. Scope Boundaries

### In scope

本次只做最小主脚本入口，包括：

* 创建 `scripts/run_pipeline.py`
* 加载基础配置
* 读取数据目录配置
* 初始化 dataset loader
* 输出数据集基本统计
* 顺序迭代少量样本做最小验证
* 预留后续 preprocess / feature extraction / visualization 挂接位置

### Out of scope

本次 **不要实现** 下列内容：

* 真正的局部特征提取
* 特征保存逻辑
* 可视化逻辑
* 完整日志系统
* CLI 的复杂参数覆盖系统
* 多阶段 pipeline orchestration
* web demo / GUI
* 任何深度学习训练入口
* 复杂异常恢复机制

原则：**只建立最小、稳定、清晰的统一脚本入口。**

---

## 4. Required File Changes

### Must create

* `scripts/run_pipeline.py`

### May update if necessary

* `src/datasets/__init__.py`
* `README.md`（仅在确有必要时补一小段运行示例，否则不要乱改）
* 与 config 读取直接相关的最少文件

不要无故新增很多脚本文件。
本 Issue 的核心交付物就是 `scripts/run_pipeline.py`。

---

## 5. Expected Runtime Flow

运行逻辑应尽量简单清楚，接近下面这样：

1. load config
2. resolve dataset root
3. create dataset loader
4. print basic dataset stats
5. iterate a few images
6. read image successfully
7. print sample preview / image shape
8. exit cleanly

也就是说，本次只是验证：

> 主脚本已经能把 config 和 dataset loader 串起来

---

## 6. Implementation Requirements

请在 `scripts/run_pipeline.py` 中实现清晰的入口，建议具备以下结构：

### 6.1 main entry

至少包含：

* `main()`
* `if __name__ == "__main__": main()`

### 6.2 config loading

使用项目已有的配置读取方式。
如果仓库中已有最小配置加载器，请复用，不要重复造一套。

要求：

* 从默认配置路径读取，例如 `configs/base.yaml`
* 读取当前阶段真正需要的最少配置
* 如果缺关键配置项，报错要直接清楚

### 6.3 dataset initialization

从配置中解析数据目录，然后初始化 `ImageDatasetLoader`（或现有 loader 名称）。

要求：

* 不要把目录写死在脚本里，优先从配置读取
* 若配置中没有对应字段，可做最小安全 fallback，但要写清楚
* 路径解析应尽量稳妥

### 6.4 minimal iteration verification

脚本至少应：

* 打印总样本数
* 打印前若干个 sample record
* 尝试读取 1~3 张图像
* 输出图像 shape 或基本读取成功信息

目标是让用户运行：

```bash
python scripts/run_pipeline.py
```

时能看到“配置加载成功 + 数据读取成功 + 图像读取成功”的基本结果。

### 6.5 future pipeline placeholders

请在代码结构中为后续阶段预留清楚注释位置，例如：

* preprocess hook
* local feature extraction hook
* feature saving hook
* visualization hook

但注意：

**只保留位置和注释，不要提前实现空洞复杂类。**

---

## 7. Design Constraints

### 7.1 Keep it minimal

不要把这个脚本写成“最终调度中心”。
它只是 Milestone 1 的统一入口。

### 7.2 Keep responsibilities clean

当前脚本负责：

* 串起 config
* 串起 dataset
* 跑最小验证流程

不要把 dataset 逻辑重新写进脚本里。
不要把未来 feature extraction 逻辑强行塞进来。

### 7.3 Readable console output

输出风格保持简洁，例如：

* `Loaded config from ...`
* `Loaded 42 images from ...`
* `Previewing first 3 samples`
* `Read image: xxx shape=(H, W, C)`
* `Pipeline skeleton run completed`

不要引入复杂 logging 框架。

### 7.4 Basic error handling

至少清楚处理这些情况：

* 配置文件不存在
* 数据目录不存在
* 数据集为空
* 图像读取失败

错误应清晰直接，不要吞掉异常。

---

## 8. Suggested API Shape

最终使用方式最好接近：

```bash
python scripts/run_pipeline.py
```

脚本内部可以大致类似：

```python
def main():
    config = load_config(...)
    loader = ImageDatasetLoader(...)
    print(...)
    for sample in loader.preview(...):
        ...
```

如果你觉得需要拆出少量辅助函数，例如：

* `load_runtime_config()`
* `build_dataset_loader()`
* `run_dataset_preview()`

可以，但不要过度拆分。

---

## 9. Acceptance Criteria

完成后应满足：

1. 存在统一脚本入口 `scripts/run_pipeline.py`
2. 可通过 `python scripts/run_pipeline.py` 运行
3. 能正确加载基础配置
4. 能正确初始化 dataset loader
5. 能顺序读取并验证若干图像
6. 能输出简洁明确的基本统计和运行信息
7. 为后续 pipeline 模块预留了合理挂接位置
8. 没有提前引入复杂调度系统或无关功能

---

## 10. Expected Output From You

请直接完成代码修改，并同时提供：

1. 你新增/修改了哪些文件；
2. 运行流程说明；
3. 示例运行输出；
4. 它如何满足 acceptance criteria；
5. 是否有任何你做出的最小假设（例如 config 字段名）。

---

## 11. Important Non-Goals

请再次注意，这个 Issue **不是**：

* 完整 pipeline 实现
* feature extraction 实现
* web demo
* 复杂 CLI 系统
* 最终版运行框架

它只是 **Milestone 1 / System Skeleton** 下的主脚本入口。

宁可简单、清晰、能跑，也不要提前做重。

---

## 12. Coding Style Notes

* 尽量补全类型标注
* 代码风格统一
* 注释克制但清楚
* 不要修改无关文件
* 不要顺手重构整个项目

---

## 13. If You Encounter Ambiguity

若配置字段名与实际仓库略有不一致，请：

* 优先复用现有 config loader 和配置文件
* 做最小兼容处理
* 在结果说明里明确说明你的假设

不要因为轻微不确定性就自行扩展整个配置系统。

---

## 四、你拿到 Codex 结果后的审核重点

等它做完 Issue 1.2，你重点看这 6 件事：

1. **是否真的从 config 取路径，而不是脚本硬编码**
2. **是否只是串起 loader，而不是把 loader 逻辑重写一遍**
3. **是否能直接 `python scripts/run_pipeline.py` 运行**
4. **输出是否简洁，不是花哨日志系统**
5. **是否只做了“主入口”，没有偷跑去实现 Issue 1.3**
6. **是否给后续模块留了清楚挂点，但没有过度抽象**

---

## 五、一个小提醒

你这次贴出来的 Issue 1.2 描述本身是偏简的，所以我给你的 prompt 已经帮你补全了：

* 工程上下文
* 范围边界
* 文件约束
* 非目标
* 验收标准
