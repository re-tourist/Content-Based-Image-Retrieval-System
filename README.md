# Hybrid Image Retrieval System

一个面向课程实践的**混合型图像检索系统**项目。  
本项目以传统图像检索主线为基础，逐步引入深度全局特征与混合检索策略，目标是实现一个**可运行、可扩展、可解释、可答辩**的图像检索系统。

---

## 1. Project Overview

图像检索（Image Retrieval）的核心问题是：

> 给定一张查询图像（query），从图像库（gallery）中找出与其最相关的图像，并按相关性排序返回。

传统图像检索方法通常依赖：

- 兴趣点检测（interest point detection）
- 局部特征描述（local feature description）
- 特征编码（visual words / Bag of Words）
- TF-IDF 表示
- 倒排索引（inverted index）
- 重排序（re-ranking）
- 查询扩展（query expansion）

而现代图像检索系统往往还会引入：

- 深度全局特征（global deep embeddings）
- 稠密向量检索（dense retrieval / ANN search）
- 多路召回融合（hybrid retrieval）

本项目希望结合两条路线的优点，构建一个**混合型图像检索系统**：

- 保留传统局部特征检索链路，满足课程主线要求；
- 引入深度全局特征检索链路，增强语义表达能力；
- 最终形成“传统检索 + 深度检索 + 重排增强”的混合系统。

---

## 2. Project Goals

本项目的目标不是只实现某一个算法，而是实现一个具有完整工程结构的图像检索系统。

### Core goals

- 实现传统图像检索主线：
  - 局部特征提取
  - 特征编码
  - TF-IDF 表示
  - 倒排索引检索
  - 重排序与查询扩展

- 实现现代增强主线：
  - 深度全局特征提取
  - 稠密向量检索
  - 与传统检索结果融合

- 构建完整工程流程：
  - 清晰的模块划分
  - 可维护的项目结构
  - 可复现的实验流程
  - 逐工期推进的开发计划

### Final target

构建一个具备以下特性的课程项目系统：

- **能运行**：支持从输入 query 到输出检索结果的完整流程
- **能扩展**：支持后续添加新特征、新索引、新重排策略
- **能解释**：可展示关键点、匹配关系、检索结果
- **能汇报**：结构清楚，便于写报告、做答辩、展示阶段成果

---

## 3. System Strategy

本项目采用**混合型图像检索策略（Hybrid Image Retrieval）**。

### Why hybrid?

单一路线通常都有明显局限：

- 仅使用局部特征与倒排索引：
  - 对近重复、裁剪、局部遮挡较强
  - 但对高层语义相似的表达能力有限

- 仅使用深度全局特征：
  - 对整体语义和外观表达较强
  - 但对精细实例级匹配与几何一致性不足

因此，本项目不走单一路线，而是采用：

> **传统局部特征路线作为基础主线，深度全局特征路线作为增强支线，最终形成混合型检索系统。**

### Design principle

系统开发遵循两条原则：

1. **宏观规划驱动**  
   先明确整体 pipeline、阶段目标和模块分工，再进行开发。

2. **阶段闭环驱动**  
   每个工期只实现当前阶段所需的最小可运行闭环，避免过早堆叠后续模块。

---

## 4. System Architecture

当前项目的整体系统架构如下：

```mermaid
flowchart LR
    subgraph A[Offline Indexing]
        A1[Raw Images]
        A2[Preprocessing]
        A3[Local Feature Extraction]
        A4[Feature Encoding]
        A5[TF-IDF Representation]
        A6[Inverted Index]

        A2 --> A3
        A3 --> A4
        A4 --> A5
        A5 --> A6
    end

    subgraph B[Online Query Pipeline]
        B1[Query Image]
        B2[Preprocessing]
        B3[Local Feature Extraction]
        B4[Encoding]
        B5[TF-IDF Query]
        B6[Sparse Retrieval]
        B7[Re-ranking]
        B8[Query Expansion]
        B9[Top-K Results]

        B1 --> B2
        B2 --> B3
        B3 --> B4
        B4 --> B5
        B5 --> B6
        B6 --> B7
        B7 --> B8
        B8 --> B9
    end

    subgraph C[Future Deep Retrieval Extension]
        C1[Global Deep Feature Extraction]
        C2[Dense Vector Index]
        C3[Dense Retrieval]
        C4[Hybrid Fusion]
    end