# Milestone 4 — Feature Encoding

---

## 1. 阶段定位（Stage Positioning）

本阶段位于系统 pipeline 的中间层，承接：

* 上游：Local Feature Extraction（SIFT / ORB）
* 下游：TF-IDF / Indexing / Retrieval

当前系统已完成：

```text
raw image
→ dataset loader
→ preprocess
→ local feature extraction
→ feature save (.npz)
→ visualization
```

本阶段负责完成：

```text
local descriptors
→ visual words（codebook）
→ image-level representation（BoW）
→ encoded feature persistence
```

---

## 2. 阶段目标（Objectives）

本阶段的核心目标：

### 2.1 构建视觉词典（Codebook）

* 基于局部特征描述子构建 visual vocabulary
* 支持多种特征类型（SIFT / ORB）
* 可复用、可持久化

---

### 2.2 实现图像级表示（BoW）

* 将局部描述子映射到 visual words
* 构建图像级 histogram 表示
* 支持空特征（empty descriptors）

---

### 2.3 建立编码结果存储体系

* 设计 encoded feature 数据格式
* 支持后续 TF-IDF / indexing 直接读取
* 保持与 feature `.npz` 解耦

---

### 2.4 提供编码接口

* 支持：

  * 从 `.npz` 加载特征
  * 从内存（LocalFeatureResult）接入
* 提供统一 encoding 输入契约

---

## 3. 非目标（Non-Goals）

本阶段明确不包含：

* ❌ TF-IDF 权重计算
* ❌ 倒排索引（inverted index）
* ❌ 相似度检索（retrieval）
* ❌ 重排序（rerank）
* ❌ 深度特征（CNN / CLIP）

---

## 4. 设计约束（Design Constraints）

### 4.1 不破坏现有系统

必须保持：

* `LocalFeatureResult` 结构不变
* `.npz` feature contract 不变
* 当前 pipeline 默认行为不变

---

### 4.2 多描述子类型支持

系统必须支持：

| 方法   | dtype   | 维度  |
| ---- | ------- | --- |
| SIFT | float32 | 128 |
| ORB  | uint8   | 32  |

约束：

* 不允许混合训练 codebook
* encoding 必须 method-aware

---

### 4.3 空描述子处理

必须显式支持：

```text
descriptors_present = 0
descriptors = None
```

要求：

* 不崩溃
* 输出合法 encoding（如全零向量）

---

### 4.4 Config 驱动

所有 encoding 行为必须由配置控制：

* codebook 参数
* sampling 策略
* encoding 开关

不得硬编码关键参数。

---

### 4.5 最小侵入原则

* 不重构现有 pipeline
* encoding 为可选阶段
* 提供独立离线处理路径

---

## 5. 系统设计（System Design）

---

### 5.1 模块划分

新增模块：

```text
src/encoding/
```

职责：

* feature loading
* descriptor sampling
* codebook training
* BoW encoding
* encoded feature storage

未来模块（本阶段不实现）：

```text
src/indexing/
```

---

### 5.2 数据流设计

```text
features (.npz)
→ descriptor sampling
→ codebook (k-means)

features (.npz)
→ codebook
→ visual word assignment
→ BoW histogram
→ encoded features (.npz)
```

---

### 5.3 数据结构（抽象层）

本阶段引入两类核心数据：

#### 1）Encoding Input（中间输入）

来源：

* LocalFeatureResult
* feature `.npz`

包含：

* descriptors
* method
* metadata（dtype / shape / presence）

---

#### 2）Encoded Feature（输出）

包含：

* image-level histogram
* visual word space信息
* descriptors统计信息
* codebook引用

---

### 5.4 存储设计

#### Codebook

```text
outputs/indices/codebooks/
```

包含：

* cluster centers
* method
* descriptor_dim
* training metadata

---

#### Encoded Features

```text
outputs/encoded/
```

特点：

* 与 feature `.npz` 解耦
* 支持批量处理
* 为 TF-IDF 直接提供输入

---

## 6. 配置设计（Configuration Design）

扩展：

```text
configs/base.yaml
```

新增：

```yaml
encoding:
  enabled: false

  input:
    feature_dir: outputs/features
    encoded_dir: outputs/encoded

  codebook:
    enabled: false
    output_dir: outputs/indices/codebooks
    n_clusters: 256
    max_descriptors: 50000
    batch_size: 1024
    random_state: 42

  bow:
    enabled: false
    normalize: false
```

设计原则：

* 默认关闭（无回归）
* 支持分阶段开启
* 支持实验控制

---

## 7. 与 Pipeline 的集成策略

### 7.1 默认行为

* encoding 默认关闭
* 原 pipeline 行为不变

---

### 7.2 可选在线编码

当启用：

```yaml
encoding.enabled = true
```

则：

* 在 feature save 后执行 encoding

---

### 7.3 离线批处理

提供独立入口：

```text
scripts/encode_features.py
```

用途：

* 构建完整数据库
* 支持大规模处理

---

## 8. 风险与设计权衡（Risks & Trade-offs）

### 8.1 Codebook 规模

* 太小 → 表达能力不足
* 太大 → 计算与存储成本高

策略：

* 初始采用固定 K（如 256）
* 后续在实验阶段调整

---

### 8.2 多方法兼容性

问题：

* SIFT / ORB 特征空间不同

决策：

* 分开 codebook（method-specific）

---

### 8.3 空样本问题

问题：

* 无 keypoints 图像

策略：

* 输出全零或合法空 encoding
* 保持 pipeline 稳定性

---

## 9. 完成定义（Definition of Done）

当满足以下条件：

* ✅ 可从 `.npz` 正确读取 descriptors
* ✅ 成功训练 codebook（SIFT / ORB）
* ✅ 可生成图像级 BoW 表示
* ✅ encoded features 成功保存
* ✅ pipeline 可选接入 encoding
* ✅ 离线批处理可运行
* ❌ 未引入 TF-IDF / retrieval

则 Milestone 4 完成。

---

## 10. 下一阶段接口（Forward Compatibility）

本阶段输出必须支持：

下一阶段（TF-IDF）：

```text
encoded features
→ DF统计
→ IDF计算
→ TF-IDF向量
```

要求：

* histogram 格式稳定
* metadata 完整
* 支持批量加载

---

# ✅ 总结（一句话）

Milestone 4 的本质：

> 把“局部特征集合”转化为“可用于检索的图像级离散表示”，并为 TF-IDF 和索引构建稳定的数据基础。
