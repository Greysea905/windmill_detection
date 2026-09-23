#!/usr/bin/env python3
"""用 ultralytics 训练 YOLO 检测模型并导出 ONNX。

前置：dataset/images + dataset/labels 已按 train/val 划分好（见 split_dataset.py）。

用法（任意目录运行，windmill 环境）：
    python3 scripts/train.py              # 用 config/paths.yaml 默认参数
    python3 scripts/train.py --device 0   # AutoDL 上用 GPU（本地默认 cpu）
"""
import argparse
import os

import yaml
from ultralytics import YOLO


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    with open('config/paths.yaml') as f:
        cfg = yaml.safe_load(f)
    m = cfg['model']

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=m['device'])
    parser.add_argument('--epochs', type=int, default=m['epochs'])
    parser.add_argument('--imgsz', type=int, default=m['imgsz'])
    parser.add_argument('--batch', type=int, default=m['batch'])
    args = parser.parse_args()

    model = YOLO(f'{m["name"]}.pt')  # 首次会自动下载预训练权重
    model.train(
        data='config/windmill.yaml',
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    # 导出 ONNX（供 C++ cv::dnn 调用），与训练 imgsz 保持一致
    best = 'runs/detect/train/weights/best.pt'
    YOLO(best).export(format='onnx', imgsz=args.imgsz)
    print('\n导出完成：runs/detect/train/weights/best.onnx。'
          '请拷入 export/ 再交给 C++ 端（vision_training）。')


if __name__ == '__main__':
    main()
