已完成 `Issue 1.4` 的最小 Web Demo 骨架实现。

修改文件：

- `scripts/run_demo.py`
- `src/demo/__init__.py`
- `src/demo/web_demo.py`
- `docs/design/web_demo.md`
- `requirements.txt`

技术选型说明：

- 本次使用 Gradio
- 原因是它和当前 Python pipeline 集成简单，适合图片上传、预览和 Gallery 结果区
- 当前阶段不需要引入前后端分离或服务端框架

UI 主要组件：

- 标题和说明文字
- Query Image Upload
- 上传图片预览
- `Run Pipeline Skeleton` 按钮
- Pipeline status 区域
- `Top-K Results` placeholder Gallery

与当前 pipeline skeleton 的对接方式：

- Demo 不直接 import `scripts/run_pipeline.py`
- Demo 直接复用 `load_config`
- Demo 直接复用 `preprocess_image(...)`
- Demo 直接复用 `extract_local_features(...)`
- 用户上传 query image 后，会走 query image load -> preprocess -> local feature placeholder -> placeholder results return

已真实接通的部分：

- Web UI 启动入口
- query image 上传与预览组件定义
- query image 读取
- preprocess 调用
- local feature placeholder 调用
- status 文本返回
- placeholder Top-K Gallery 返回

当前明确为 placeholder 的部分：

- real retrieval
- real Top-K ranking
- keypoint visualization
- local feature visualization
- retrieval result explanation

本地运行方式：

```bash
pip install -r requirements.txt
python scripts/run_demo.py
```

本地验证情况：

- `python -m py_compile scripts/run_demo.py src/demo/__init__.py src/demo/web_demo.py` 已通过
- 运行时依赖 `gradio` 当前未在本地环境中安装
- 我尝试安装 `gradio>=5.0,<6.0` 做启动验证，但被本机 pip 临时目录权限问题阻塞，因此未能完成实际 `python scripts/run_demo.py` 启动截图级验证

满足验收标准的原因：

1. Demo 启动脚本和 Web UI 代码已建立
2. 页面已定义上传图片、图片预览、按钮和 Top-K 结果区
3. Demo 代码已和当前 pipeline skeleton 的模块边界对齐
4. 代码为未来真实 retrieval 和可视化留好了扩展位置
5. 没有引入复杂前后端架构

当前剩余阻塞不是代码结构，而是本地环境缺少可安装的 `gradio` 运行依赖。
