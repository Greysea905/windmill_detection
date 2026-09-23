#!/usr/bin/env python3
"""弱模型预标注（半自动标注后半程用）。

流程：用已训好的弱模型对 dataset/raw/ 里未标注的帧跑推理，
生成同名 YOLO .txt 预标注，再用 LabelImg 逐张核对修正。

注意：会覆盖同目录下同名 .txt，人工已标好的帧请先移走或备份。

用法（项目根目录，windmill 环境）：
    python3 scripts/auto_label.py --weights runs/detect/train/weights/best.pt \
        --raw dataset/raw/task_3 --conf 0.5
"""
import argparse
import os

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='best.pt 路径')
    parser.add_argument('--raw', required=True, help='要预标注的目录')
    parser.add_argument('--conf', type=float, default=0.5)
    args = parser.parse_args()

    model = YOLO(args.weights)
    imgs = sorted(f for f in os.listdir(args.raw)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    for img in imgs:
        path = os.path.join(args.raw, img)
        r = model.predict(path, conf=args.conf, verbose=False)[0]
        lines = []
        for b in r.boxes:
            cls = int(b.cls[0])
            x, y, w, h = b.xywhn[0].tolist()  # 归一化中心 + 宽高
            lines.append(f'{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')
        txt = os.path.join(args.raw, img.rsplit('.', 1)[0] + '.txt')
        with open(txt, 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
    print(f'完成 {len(imgs)} 张预标注，用 LabelImg 打开核对修正。')


if __name__ == '__main__':
    main()
