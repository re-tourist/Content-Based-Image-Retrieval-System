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