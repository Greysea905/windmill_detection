#!/usr/bin/env python3
"""从 resources/ 视频按固定步长抽帧到 dataset/raw/，作为标注底图。

用法（用 windmill 虚拟环境，任意目录运行均可）：
    python3 scripts/extract_frames.py            # 用 config/paths.yaml 的默认步长
    python3 scripts/extract_frames.py --step 5   # 覆盖步长
"""
import argparse
import os

import cv2
import yaml


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    with open('config/paths.yaml') as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=cfg['extract']['step'])
    args = parser.parse_args()
    step = args.step

    out_root = cfg['extract']['out_dir']  # dataset/raw
    total = 0
    for name, vpath in cfg['videos'].items():
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f'[跳过] 无法打开 {vpath}')
            continue
        out_dir = os.path.join(out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved = 0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                # 带视频前缀的 5 位帧号，跨视频命名唯一、可回溯原帧
                out_path = os.path.join(out_dir, f'{name}_{idx:05d}.jpg')
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
            idx += 1
        cap.release()
        total += saved
        print(f'{name}: 共 {n_total} 帧，步长 {step}，抽出 {saved} 张 -> {out_dir}')

    print(f'\n完成，共抽出 {total} 张。下一步：用 LabelImg 打开 dataset/raw/<video>/ 标注。')


if __name__ == '__main__':
    main()
