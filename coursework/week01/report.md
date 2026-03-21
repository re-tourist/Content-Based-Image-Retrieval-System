# 基于局部特征的图像匹配实验报告

## 1. 实验目的

本实验的目标是基于传统局部特征方法实现一个简单的图像匹配系统。系统能够对输入图像提取关键点与局部描述子，在图库中找到最匹配的一张图像，并在图形界面中显示匹配结果、匹配时间、匹配点数量等实验信息。同时，对两种关键点算法的时间性能进行比较，为后续分析不同方法的特点提供依据。

本次实验实际实现并比较了 `SIFT` 与 `ORB` 两种算法。之所以采用这两种方法，是因为它们都属于经典的基于关键点的局部特征方法，且在 OpenCV 环境中较容易实现；同时二者在匹配精度与运行速度上具有较明显差异，适合做课堂实验对比。

## 2. 算法原理简述

### 2.1 SIFT 算法原理

SIFT（Scale-Invariant Feature Transform）是一种经典的局部特征提取方法，主要特点是对尺度变化和旋转变化具有较好的鲁棒性。其基本思想如下：

1. 在不同尺度下构建图像的尺度空间，并利用高斯差分（DoG）检测尺度空间中的极值点。
2. 对候选关键点进行精确定位，去除低对比度点和不稳定的边缘点。
3. 为每个关键点分配主方向，使特征对旋转变化具有不变性。
4. 在关键点邻域统计梯度方向分布，构造 128 维浮点描述子。

由于 SIFT 描述子是浮点型特征，一般使用欧氏距离进行匹配，因此在 OpenCV 中通常采用 `BFMatcher + NORM_L2`。SIFT 的优点是匹配稳定性较好，缺点是计算量相对较大，整体运行速度较慢。

### 2.2 ORB 算法原理

ORB（Oriented FAST and Rotated BRIEF）是一种面向实时应用的轻量级局部特征方法。其核心思想是将 FAST 关键点检测与 BRIEF 描述子结合，并针对旋转问题进行改进。主要步骤如下：

1. 使用 FAST 算法检测图像中的角点。
2. 通过 Harris 响应或其他评分方式筛选较稳定的关键点。
3. 根据关键点邻域灰度分布估计主方向，使关键点具备一定的旋转不变性。
4. 使用带方向的 BRIEF 生成二值描述子。

由于 ORB 描述子是二值特征，一般使用汉明距离进行匹配，因此在 OpenCV 中通常采用 `BFMatcher + NORM_HAMMING`。ORB 的优点是速度快、实现简单，缺点是在复杂尺度变化或纹理变化较大时，匹配稳定性通常不如 SIFT。

### 2.3 本实验中的匹配策略

为了提高匹配可靠性，本实验没有直接使用全部原始匹配点，而是采用了 `knnMatch + ratio test` 的方式筛选 good matches。具体做法是：对每个特征点寻找两个最近邻匹配点，若最近邻距离明显小于次近邻距离，则认为该匹配较可靠并保留。这样可以在一定程度上减少误匹配。

在检索阶段，对 query 图像与图库中每一张候选图像分别进行特征匹配，并以 `good matches 数量` 作为主要排序依据，从而选出最佳匹配图像。若 good matches 数量相同，则进一步比较平均匹配距离，优先保留平均距离更小的结果。

## 3. 系统实现步骤

本实验系统的实现流程如下：

1. 用户在 GUI 中选择一张待匹配图像作为 query。
2. 程序遍历图库目录中的其余图像作为候选图像。
3. 对 query 图像和候选图像进行读取、灰度化和适当缩放。
4. 根据用户选择的算法提取关键点和描述子。
5. 使用 BFMatcher 对描述子进行匹配。
6. 使用 `knnMatch + ratio test` 过滤错误匹配，保留 good matches。
7. 统计每张候选图像的 good matches 数量，并选出最佳匹配图像。
8. 记录单张候选图匹配耗时和整体检索总耗时。
9. 将 query 图像、最佳匹配图像和匹配连线可视化结果显示在界面中，并保存结果图到 `outputs/` 目录。
10. 对 `SIFT` 和 `ORB` 两种算法分别运行一次，比较其运行时间和匹配效果。

<img src="C:\Users\98337\Pictures\Screenshots\屏幕截图 2026-03-15 150552.png" alt="GUI 主界面截图" style="zoom:33%;" />

​																										图 1  GUI 主界面

## 4. 关键代码说明

本实验的核心代码主要位于 `core.py` 和 `app.py` 两个文件中。

### 4.1 算法选择与回退

程序优先支持 `SIFT` 和 `ORB`。若当前 OpenCV 环境不支持某个算法，则会自动回退到其他可用算法，避免程序直接报错退出。核心逻辑如下：

```python
def resolve_algorithm(requested: str) -> tuple[str, str]:
    requested = requested.upper()
    names = available_algorithms()
    if requested in names:
        return requested, ""
    for candidate in fallback_order.get(requested, names):
        if candidate in names:
            return candidate, f"{requested} 不可用，已自动回退到 {candidate}。"
```

这一部分的作用是保证 GUI demo 在不同机器上都尽量能够运行，满足作业“能展示结果”的要求。

### 4.2 特征提取

在提取特征之前，程序会先将图像转为灰度图，并对过大的图像进行缩放，以减少计算时间。之后根据算法名称创建特征提取器，再计算关键点和描述子。核心逻辑如下：

```python
def extract_features(image_path: Path, algorithm: str):
    gray = load_image(image_path, grayscale=True)
    gray = resize_for_feature(gray)
    extractor = create_extractor(algorithm)
    keypoints, descriptors = extractor.detectAndCompute(gray, None)
    keypoints = keypoints or []
    return gray, keypoints, descriptors
```

该函数统一封装了图像读取、预处理和特征提取过程，使得 SIFT、ORB、AKAZE 等算法都可以使用相同的接口调用。

### 4.3 特征匹配与 good matches 筛选

对于描述子匹配，本实验使用 BFMatcher。SIFT 采用 `NORM_L2`，ORB 采用 `NORM_HAMMING`。随后利用 `knnMatch` 找到每个特征点的两个最近邻，再用 ratio test 筛选可靠匹配点。核心代码如下：

```python
def match_descriptors(query_descriptors, candidate_descriptors, algorithm: str):
    matcher = cv2.BFMatcher(descriptor_norm(query_descriptors, algorithm), crossCheck=False)
    raw_pairs = matcher.knnMatch(query_descriptors, candidate_descriptors, k=2)
    threshold = ratio_threshold(algorithm)
    good_matches = []

    for pair in raw_pairs:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < threshold * second.distance:
            good_matches.append(first)
```

这里保留的 `good_matches` 数量直接作为后续图像排序和结果比较的重要指标。

### 4.4 最佳匹配图像检索

系统会将 query 图像与图库中的所有候选图像逐一匹配，并记录每张候选图像的匹配数量和耗时。若当前候选图像的 good matches 更多，则更新最佳结果；若数量相同，则比较平均距离。核心逻辑如下：

```python
for candidate_path in gallery_images:
    _, candidate_keypoints, candidate_descriptors = extract_features(candidate_path, actual_algorithm)
    good_matches, raw_count = match_descriptors(query_descriptors, candidate_descriptors, actual_algorithm)
    candidate_score = CandidateScore(
        image_path=candidate_path,
        good_matches=len(good_matches),
        raw_matches=raw_count,
        average_distance=average_distance,
        elapsed_ms=elapsed_ms,
        keypoints=len(candidate_keypoints),
    )
    best_candidate = choose_better_candidate(best_candidate, candidate_score)
```

这一过程实现了“从其余图像中找出最匹配图像”的作业要求。

### 4.5 GUI 结果展示

在 `app.py` 中，程序使用 Tkinter 构建图形界面，主要提供以下功能：

- 选择 query 图像
- 选择匹配算法
- 开始匹配
- 对两种算法做时间对比
- 显示 query 图像、最佳匹配图像和匹配连线可视化图
- 显示算法名称、good matches 数量、匹配时间和结果图保存路径

其中，匹配任务使用后台线程执行，避免在计算过程中 GUI 卡死。匹配完成后，再将结果回传到界面中更新显示。

## 5. 两种算法的时间性能比较

以下结果基于当前仓库环境下的一次实际运行得到：选取 `data/test/A0C573/A0C573_20151103073308_3029240562.jpg` 作为 query，默认图库包含 987 张候选图像。不同机器、不同 query 图像以及图库规模会影响最终时间，因此本表主要用于说明相对趋势。

| 算法 | 总检索耗时 | 最佳候选单张耗时 | good matches | 结果说明 |
| --- | ---: | ---: | ---: | --- |
| SIFT | 32033.81 ms | 35.99 ms | 405 | 匹配结果较稳定，但整体耗时高于 ORB |
| ORB | 10983.28 ms | 8.62 ms | 1354 | 检索速度更快，更适合快速演示 |

从实验结果可以看出，ORB 的整体运行时间明显短于 SIFT，因此在需要快速响应的场景下更有优势；而 SIFT 作为经典浮点特征方法，通常在特征稳定性方面表现较好。对于本次课程作业，如果强调“界面演示流畅”，可以优先展示 ORB；如果强调“传统经典算法效果”，则可以展示 SIFT。

若需要重新生成实验数据，可执行：

```bash
python coursework/week01/core.py --benchmark --json
```

<img src="D:\MyProject\Content-Based-Image-Retrieval-System\coursework\week01\outputs\comparison_A0C573_20151103073308_3029240562_20260315_151235.jpg" alt="SIFT 与 ORB 对比图" style="zoom:33%;" />

​																				图 2  SIFT 与 ORB 算法匹配结果对比图

## 6. 运行结果说明

程序运行后，界面中能够展示以下实验结果：

- 待匹配图像
- 最匹配图像
- 匹配连线可视化图
- 当前算法名称
- good matches 数量
- 单张候选图匹配耗时
- 整体检索总耗时
- 结果图保存路径

实验表明，当 query 图像与图库中存在同类或相同目标图像时，系统通常可以找到正确或接近正确的最佳匹配结果。同时，SIFT 与 ORB 的对比结果也能够较直观地体现“效果与速度”的差异，满足课程作业对于算法比较与实验展示的要求。

<img src="C:\Users\98337\AppData\Roaming\Typora\typora-user-images\image-20260315151408126.png" alt="单次匹配结果截图" style="zoom:33%;" />

​																							图 3  单次匹配运行结果展示
