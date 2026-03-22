## Branch Naming Convention

本项目采用“类型 / 模块 / issue”的分支命名方式。

### Main branches
- `main`
- `dev`

### Documentation branches
- `docs/update`
- `docs/plan`
- `docs/design`
- `docs/status`

### Feature branches
- `feat/config`
- `feat/data`
- `feat/preprocess`
- `feat/features-local`
- `feat/features-global`
- `feat/encoding`
- `feat/indexing`
- `feat/retrieval`
- `feat/rerank`
- `feat/expansion`
- `feat/visualization`
- `feat/scripts`
- `feat/core`

### Chore branches
- `chore/repo`
- `chore/env`
- `chore/general`

### Recommended issue branch style
For concrete implementation tasks, create short-lived branches from `dev`:

- `feat/data/issue-1-1`
- `feat/scripts/issue-1-2`
- `feat/core/issue-1-3`
- `feat/visualization/issue-1-4`

原则：
1. 先按模块归类，再按 issue 细分；
2. 单个 issue 尽量只在一个短期分支中完成；
3. 合并后可删除 issue 分支，保留模块命名体系的一致性。

### 把当前 issue 映射到分支

这是你现在最需要的部分。

Milestone M1
Issue 1.1 — Implement Dataset Loader

推荐分支：

feat/data
Issue 1.2 — Create Main Script Entry

推荐分支：

feat/scripts
Issue 1.3 — Build Pipeline Skeleton and Define Module Interfaces

推荐分支：

feat/core

因为这个 issue 的本质不是某个具体算法模块，而是系统骨架、模块边界、主调用链。放到 feat/core 最合适。

Issue 1.4 — Build Minimal Web Demo Interface

推荐分支：

feat/visualization

或者如果你特别想强调前端展示，可以用：

feat/web-demo

但为了统一模块池，我更推荐 feat/visualization。

Milestone M2
Issue 2.1 — Dataset Structure Standardization

推荐分支：

feat/data
Issue 2.2 — Image Preprocessing Pipeline

推荐分支：

feat/preprocess
Milestone M3
Issue 3.1 — Implement Local Feature Extraction

推荐分支：

feat/features-local
Issue 3.2 — Keypoint Visualization

推荐分支：

feat/visualization