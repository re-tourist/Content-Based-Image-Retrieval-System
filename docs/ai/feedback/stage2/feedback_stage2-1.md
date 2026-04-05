已完成 Issue 2.1 的最小实现，范围控制在“统一数据结构和 split 机制”，没有扩成数据工程系统。

**变更**
- 配置补充了数据源目录和 split 文件字段：[configs/base.yaml](/d:/MyProject/Content-Based-Image-Retrieval-System/configs/base.yaml#L4)
- 配置解析现在会把 train/test 源目录和 split 文件一起解析成绝对路径：[src/utils/config.py](/d:/MyProject/Content-Based-Image-Retrieval-System/src/utils/config.py#L47)
- `ImageDatasetLoader` 增加了 split 文件加载模式，保留原来的目录扫描模式不变：[src/datasets/dataset_loader.py](/d:/MyProject/Content-Based-Image-Retrieval-System/src/datasets/dataset_loader.py#L22)
- 新增 split 生成脚本：[scripts/build_splits.py](/d:/MyProject/Content-Based-Image-Retrieval-System/scripts/build_splits.py#L1)
- 新增数据结构说明文档：[docs/design/dataset_structure.md](/d:/MyProject/Content-Based-Image-Retrieval-System/docs/design/dataset_structure.md#L1)
- 生成了标准清单：
  - [data/splits/train.txt](/d:/MyProject/Content-Based-Image-Retrieval-System/data/splits/train.txt)
  - [data/splits/query.txt](/d:/MyProject/Content-Based-Image-Retrieval-System/data/splits/query.txt)
  - [data/splits/gallery.txt](/d:/MyProject/Content-Based-Image-Retrieval-System/data/splits/gallery.txt)
- 为了让这些清单能提交进仓库，补了忽略规则例外：[.gitignore](/d:/MyProject/Content-Based-Image-Retrieval-System/.gitignore#L46)

**当前方案**
- 标准目录约定是 `data/raw`、`data/processed`、`data/splits`，文档写在 [dataset_structure.md](/d:/MyProject/Content-Based-Image-Retrieval-System/docs/design/dataset_structure.md#L10)。
- split 文件格式采用最小方案：每行一个相对 `data/` 的图片路径，例如 `train/image/...jpg`、`test/...jpg`，见 [dataset_structure.md](/d:/MyProject/Content-Based-Image-Retrieval-System/docs/design/dataset_structure.md#L57)。
- `build_splits.py` 的规则是：
  - `train.txt`：`data/train/image` 下全部图片
  - `query.txt`：`data/test` 每个类别目录排序后的第一张
  - `gallery.txt`：同目录剩余图片
- loader 现在既能继续扫目录，也能这样读 split：
  - `ImageDatasetLoader(split="query", split_file="data/splits/query.txt", data_root="data")`
  - 对接说明见 [dataset_loader.py](/d:/MyProject/Content-Based-Image-Retrieval-System/src/datasets/dataset_loader.py#L29) 和 [dataset_structure.md](/d:/MyProject/Content-Based-Image-Retrieval-System/docs/design/dataset_structure.md#L104)

**使用与验证**
- 生成清单：
  - `python scripts/build_splits.py`
- 本次实际生成结果：
  - `train.txt`: 946
  - `query.txt`: 14
  - `gallery.txt`: 28
- 已验证：
  - `python -m py_compile src/utils/config.py src/datasets/dataset_loader.py scripts/build_splits.py`
  - split 模式 loader 成功读到 `query_len=14`，并成功加载首张图，`shape=(450, 450, 3)`
  - `python scripts/run_pipeline.py` 仍可正常运行，说明没有破坏 Stage 1 主链路

最小兼容假设是：当前仓库里的真实源数据还在 `data/train/image` 和 `data/test`，这次不做强制迁移到 `data/raw`；先用 `data/splits/*.txt` 把“怎么读数据”标准化。