# Week01 局部特征图像匹配 Demo

这是一个一次性交作业用的小型 GUI demo，用 OpenCV 的局部特征方法完成图像关键点提取、匹配和最相似图像检索。目标是尽快跑通、方便截图和提交，不参与主项目正式架构。

## 功能

- 支持两种及以上局部特征算法
  - 优先使用 `SIFT + ORB`
  - 若当前环境不支持 `SIFT`，自动回退到 `ORB` 或 `AKAZE`
- 选择一张 query 图像后，在其余图像中找到最匹配的一张
- GUI 中显示
  - 待匹配图像
  - 最匹配图像
  - 匹配连线可视化图
  - 匹配时间
  - good matches 数量
  - 当前算法名称
- 自动把匹配可视化结果保存到 `coursework/week01/outputs/`
- 支持一键做两算法时间对比
- benchmark 后额外生成一张带文字图注的合并对比图，便于直接放入报告

## 依赖环境

- Python 3.10+
- OpenCV
- Tkinter
- Pillow
- NumPy

安装示例：

```bash
pip install opencv-python pillow numpy
```

说明：`Tkinter` 一般随标准 Python 安装提供；本仓库当前环境已验证可用。

## 目录说明

```text
coursework/week01/
├─ app.py
├─ core.py
├─ README.md
├─ report.md
├─ images/
└─ outputs/
```

- `app.py`：GUI 程序入口
- `core.py`：特征提取、匹配、检索、计时、结果图保存
- `images/`：可选本地图库目录
- `outputs/`：保存匹配结果图，便于截图交作业

## 图像放置方式

程序默认按以下顺序寻找图库：

1. `coursework/week01/images/`
2. 仓库已有的 `data/train/image/`
3. 仓库已有的 `data/test/`

推荐做法：

- 如果你想完全独立演示，把几张测试图复制到 `coursework/week01/images/`
- 如果不额外复制图片，程序会直接复用仓库已有图像数据

Query 图像可以在 GUI 中手动选择，也可以直接使用程序自动预加载的示例图。

## 运行方式

在仓库根目录执行：

```bash
python coursework/week01/app.py
```

启动后：

1. 点击“选择图像”选择 query 图像
2. 在下拉框中选择算法
3. 点击“开始匹配”
4. 如需对比两种算法时间，点击“两算法对比”

## 支持算法

- `SIFT`
- `ORB`
- `AKAZE`

优先推荐：

- `SIFT`
- `ORB`

若 `SIFT` 不可用，程序不会直接退出，而是自动提示并回退到可用算法。

## 输出结果

每次成功匹配后，程序会在 `coursework/week01/outputs/` 下保存一张结果图，内容为：

- query 图像
- 最佳匹配图像
- 匹配连线

执行两算法对比后，还会额外生成一张合并图，默认把两种算法的结果上下排布，并附带算法名、good matches 和耗时说明，适合直接插入实验报告。

文件名示例：

```text
sift_A0C573_20151103073308_3029240562_20260315_153000.jpg
```

## 命令行快速测试

若不想先打开 GUI，也可以直接用命令行跑一次：

```bash
python coursework/week01/core.py --benchmark --max-candidates 50
```

或指定 query：

```bash
python coursework/week01/core.py --query path/to/query.jpg --algorithm ORB
```

## 结果文档

作业报告草稿位于：

`coursework/week01/report.md`
