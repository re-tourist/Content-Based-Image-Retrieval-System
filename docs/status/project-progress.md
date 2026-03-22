# 项目进度状态说明（校正版）

- 状态更新时间：2026-03-22
- 作用：修正本文件早期版本与当前仓库实际状态不一致的问题
- 当前权威入口：`docs/ai/PROJECT_CONTEXT.md`
- 当前代码级接口说明：`docs/design/pipeline_skeleton.md`

## 1. 当前结论

当前主工程已经完成以下阶段：

- Milestone 1：System Skeleton
- Milestone 2：Data Preparation
- Milestone 3：Local Feature Extraction 与 Keypoint Visualization

因此，仓库当前不应再被描述为“工期1未完成”或“局部特征只在 coursework 原型中验证”。
这些判断属于更早期的状态，已经不再代表当前主工程。

当前主链路已经真实打通：

`raw image -> dataset loader -> preprocess -> local feature extraction -> feature save -> keypoint visualization`

主工程当前可以：

- 读取配置与数据目录
- 加载图像样本
- 执行最小预处理
- 使用 `SIFT` 或 `ORB` 提取局部特征
- 将特征保存为 `outputs/features/*.npz`
- 将关键点可视化结果保存为 `outputs/figures/keypoints_*.png`

## 2. 当前工程状态

### 已实现

- dataset loader
- split manifest generation and loader compatibility
- preprocess result standardization
- real local feature extraction
- feature saving
- keypoint visualization
- minimal Gradio demo bridge

### 尚未实现

- feature encoding
- TF-IDF
- inverted index
- retrieval
- rerank
- query expansion
- dense retrieval
- hybrid fusion

### 下一阶段

下一阶段应进入：

- Stage 4 / Feature Encoding

它应消费当前已经稳定落盘的 descriptor 输出，而不是重新定义局部特征阶段。

## 3. 推荐阅读顺序

如果需要继续开发或把当前状态提供给新的工程师 / ChatGPT，请优先阅读：

1. `docs/ai/PROJECT_CONTEXT.md`
2. `docs/design/pipeline_skeleton.md`
3. `configs/base.yaml`
4. `scripts/run_pipeline.py`
5. `src/features/local/local_feature_extractor.py`
6. `docs/design/dataset_structure.md`

## 4. 关于旧版状态报告

本文件的旧版内容形成于更早的开发时点，当时主工程尚未完成数据准备、真实局部特征提取和关键点可视化，因此会出现以下过期结论：

- 主工程仍停留在工期1
- 工期2未开始
- 工期3只存在旁路线原型

这些描述现在都已经不再成立。

## 5. 当前文档入口约定

为了避免后续继续出现多份上下文文档相互冲突，当前约定如下：

- `docs/ai/PROJECT_CONTEXT.md`：唯一的项目上下文与阶段交接入口
- `docs/design/pipeline_skeleton.md`：当前代码实现的主链路、接口和 `.npz` 契约说明

## 6. 后续维护建议

后续如果继续记录阶段状态：

- 新增按日期命名的状态文档
- 在文档开头明确“记录日期”和“是否仍为当前权威状态”
- 不再复制一份项目总上下文到多个文件中
- 以 `PROJECT_CONTEXT.md` 为主入口进行更新
