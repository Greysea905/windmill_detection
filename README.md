# windmill_detection

用来进行 RoboMaster 能量机关（风车）目标检测模型的训练。

产出 YOLOv11n 的 ONNX 权重，交给交付工程 [`vision_training`](../vision_training)（C++ + OpenCV）做实时识别与稳定跟踪。

---

## 1. 项目概述

任务是识别 RoboMaster「能量机关」上的两类目标：

| 类别 | 含义 |
|---|---|
| `0 = R` | 五片扇叶中心的 R 标（旋转中心） |
| `1 = target` | 扇叶末端的红色靶（有效目标，小机关 ≤1 个、大机关 ≤2 个同时亮起） |

本仓库负责：**数据集制作 → 训练 → 导出 ONNX**（Python / ultralytics）。最终的识别+跟踪在交付工程 `vision_training` 里用 C++ 完成，两者通过 `best.onnx` 衔接。

---

## 2. 项目思路

### 2.1 为什么用深度学习，而不是传统 CV（HSV + 形态学）

最初用 HSV 阈值 + 形态学 + 面积/填充率筛选，但光照、距离变化下手动调参的泛化能力很差，目标靶难以正确检出。引入 YOLOv11n 后直接学习原始 RGB 特征，鲁棒性有大幅提升。

### 2.2 为什么选 YOLOv11n

- 2.6M 参数，nano 尺寸，最终要在 CPU/边缘设备上推理。
- 不追更新版本（v12+ 有 attention 结构）：要保证导出的 ONNX 能被 OpenCV `cv::dnn` 稳定解析。
- 导出 ONNX 零额外依赖，C++ 端 `cv::dnn::readNetFromONNX` 直接加载。

### 2.3 训练不切图（全帧）

能量机关在画面里移动（旋转中心 R 也会动），固定切图区域会切偏。全帧训练不依赖机关位置、泛化更好。代价是目标相对较小，用高 `imgsz` 补偿（见下）。

### 2.4 imgsz = 1280

目标 ~80px、R 标 ~20px 属小目标，1280 只缩 89%，保留细节。

### 2.5 半自动标注（种子 → 弱模型 → 预标）

手工标 ~90 张「种子」→ 本地训一个弱模型 → 用弱模型自动预标剩余 ~800 帧 → 人工检查修正。降低了手工标注量，且最终数据质量很高（mAP50 0.99）。

### 2.6 空帧负样本

视频有大量无目标帧。按 ~16% 比例混入空标签（空 `.txt`），教模型「没靶就输出空」，降低误检。

---

## 3. 数据

- `task_3.mp4`（小机关，≤1 靶，796 帧）+ `task_4.mp4`（大机关，≤2 靶，1800 帧），均为 1440×1080 @ 30fps。
- 每 3 帧抽 1 张 → 共 **866 张**。
- 划分：按时间顺序前 80% 训练 / 后 20% 验证（避免相邻帧泄漏）。
- 最终模型：**mAP50 = 0.99，mAP50-95 = 0.934**（AutoDL GPU，100 epoch，imgsz 1280）。

---

## 4. 实现过程记录（时间线）

1. **决策和选型**：选 YOLOv11n（ONNX 可被 OpenCV 解析）。
2. **建仓库 + conda 环境**（踩坑记录）：
   - `~/.local` 包泄漏污染 conda 环境 → 用 `PYTHONNOUSERSITE=1` 隔离重建。
   - PyQt5 新版本对 `float→int` 严格化，LabelImg 老代码崩溃 → 降级 PyQt5 仍不行，最终**直接给 LabelImg 源码打补丁**（7 处 `int()` 包裹）。
   - LabelImg 在 Wayland + 4K 下缩放错误/崩溃 → 强制 X11 后端 + 缩放因子。
3. **抽帧 + 标注**：`extract_frames.py` 抽 866 帧；LabelImg 手工标 ~90 张种子（含空帧负样本）。
4. **弱模型**：本地 CPU 训 50 epoch（yolo11n, imgsz 960），mAP50 0.995。
5. **自动预标**：弱模型预标 task_3 + task_4 剩余帧，人工检查修正（几乎全对）。
6. **最终模型**：AutoDL GPU（RTX PRO 6000）训 100 epoch、imgsz 1280，导出 `best.onnx`。
7. **C++ 部署**：在 `vision_training` 用 `cv::dnn` 加载 ONNX，实现识别 + 角速度预测跟踪 + 可视化。

---

## 5. 目录结构

```
windmill_detection/
├── config/
│   ├── paths.yaml              # 视频路径、抽帧步长、模型参数、类别名
│   ├── windmill.yaml           # ultralytics 数据配置（train/val 路径、类别）
│   └── predefined_classes.txt  # LabelImg 类别名
├── resources/                  # 原始视频（gitignored，从 vision_training 复制）
├── dataset/
│   ├── raw/{task_3,task_4}/    # 抽出的原始帧 + 标注（gitignored）
│   ├── images/{train,val}/     # 切分后的图（gitignored）
│   └── labels/{train,val}/     # YOLO 标注（提交）
├── scripts/
│   ├── extract_frames.py       # ① 抽帧
│   ├── split_dataset.py        # ② 按时间切 train/val
│   ├── auto_label.py           # ③ 弱模型预标注
│   ├── train.py                # ④ 训练 + 导出 ONNX
│   └── label.sh                # LabelImg 启动脚本（避免在 ubuntu22.04 上使用时出现的显示问题）
├── runs/                       # 训练输出（gitignored）
└── export/                     # 导出的 best.onnx（gitignored）
```

---

## 6. 环境

- Python 3.10，conda 环境 `windmill`（在项目外 `~/miniconda3/envs/windmill`）。
- 创建：
  ```bash
  conda create -n windmill python=3.10 -y
  conda activate windmill
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install ultralytics labelimg onnx onnxruntime
  ```
- 注意：若 `~/.local` 有杂包会泄漏，可用 `conda env config vars set PYTHONNOUSERSITE=1 -n windmill` 隔离。

---

## 7. 使用流程（复现）

1. **抽帧**：`python scripts/extract_frames.py`
2. **标注**：`bash scripts/label.sh task_3`（LabelImg，YOLO 格式，类别 `R`/`target`）
3. **划分**：`python scripts/split_dataset.py`
4. **训练**：
   - 本地 CPU 快速验证：`python scripts/train.py --epochs 50 --imgsz 960`
   - 正式训练（AutoDL GPU）：`python scripts/train.py --device 0 --epochs 100 --imgsz 1280`
5. **导出 ONNX** → 拷入 `export/`，再交给 `vision_training` 的 C++ 端。

---

## 8. 与交付工程衔接

| 仓库 | 职责 | 技术栈 |
|---|---|---|
| `windmill_detection`（本仓库） | 数据集 + 训练 + 导出 ONNX | Python / ultralytics |
| `vision_training` | C++ 推理 + 稳定跟踪 + 提交 | C++ / OpenCV / CMake |

`best.onnx` 放在 `vision_training/model/`，由 `src/common/yolo.cpp` 加载，`src/task3_windmill/main.cpp` 完成逐帧识别与跟踪。
