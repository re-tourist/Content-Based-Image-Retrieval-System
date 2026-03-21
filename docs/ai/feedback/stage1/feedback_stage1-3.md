已完成 `Issue 1.3` 的骨架搭建，并把当前阶段的模块边界固定下来了。

修改文件：

- `scripts/run_pipeline.py`
- `src/preprocess/__init__.py`
- `src/preprocess/basic_preprocess.py`
- `src/features/local/__init__.py`
- `src/features/local/local_feature_extractor.py`
- `docs/design/pipeline_skeleton.md`

当前 pipeline skeleton 调用链：

1. load config
2. resolve dataset image directory
3. build dataset loader
4. load image
5. run basic preprocess
6. run local feature extraction placeholder
7. run feature save placeholder
8. run visualization placeholder

已定义的模块接口：

- `ImageDatasetLoader` / `ImageSample`
- `preprocess_image(...) -> PreprocessResult`
- `extract_local_features(...) -> LocalFeatureResult`
- `save_feature_result(...)`
- `visualize_keypoints(...)`

已真实接通的部分：

- config 读取
- dataset loader
- image loading
- preprocess stage
- 主脚本阶段串联

当前明确为 placeholder 的部分：

- local feature extraction 真实算法
- feature save 真实落盘
- visualization 真实绘图
- encoding / indexing / retrieval / rerank / expansion / dense / hybrid

示例运行输出摘要：

```text
Loaded config from ...\configs\base.yaml
Loaded 946 images from ...\data\train\image
Running pipeline skeleton on first 3 samples
Preprocess stage completed for ...
Local feature stage placeholder executed for ...
Save stage placeholder executed for ...
Visualization stage placeholder executed for ...
Pipeline skeleton run completed
```

满足验收标准的原因：

1. 当前阶段 pipeline 结构已能清楚说明
2. 系统已可从入口调用到 dataset / preprocess / features / save / visualize
3. 输入输出边界已通过 `PreprocessResult` 和 `LocalFeatureResult` 固定
4. 后续模块挂接位置已在代码注释和设计文档中明确
5. `docs/design/pipeline_skeleton.md` 说明了当前骨架与未来扩展关系
6. `python scripts/run_pipeline.py` 已通过运行验证
7. 没有提前实现复杂算法或最终版框架
