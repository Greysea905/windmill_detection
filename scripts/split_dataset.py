#!/usr/bin/env python3
"""把 dataset/raw/ 里已标注（有同名 .txt）的帧，按时间顺序切成 train/val。

规则：每个视频内按帧号排序，前 (1-val_ratio) 归 train，后 val_ratio 归 val。
按时间切（而非随机）避免相邻帧泄漏。只处理有标注的帧。

用法（任意目录运行）：
    python3 scripts/split_dataset.py
"""
import os
import shutil

import yaml


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    with open('config/paths.yaml') as f:
        cfg = yaml.safe_load(f)
    val_ratio = cfg['split']['val_ratio']

    img_dst = 'dataset/images'
    lbl_dst = 'dataset/labels'
    for split in ('train', 'val'):
        os.makedirs(os.path.join(img_dst, split), exist_ok=True)
        os.makedirs(os.path.join(lbl_dst, split), exist_ok=True)

    stats = {'train': 0, 'val': 0}
    for video in cfg['videos']:
        raw_dir = os.path.join('dataset/raw', video)
        if not os.path.isdir(raw_dir):
            continue
        # 只挑有同名 label 的图片，按帧号（文件名）排序
        labeled = sorted(
            f for f in os.listdir(raw_dir)
            if f.endswith('.jpg') and os.path.exists(os.path.join(raw_dir, f[:-4] + '.txt'))
        )
        n = len(labeled)
        n_val = int(round(n * val_ratio))
        for i, fname in enumerate(labeled):
            split = 'val' if i >= n - n_val else 'train'
            base = fname[:-4]
            shutil.copy(os.path.join(raw_dir, fname),
                        os.path.join(img_dst, split, fname))
            shutil.copy(os.path.join(raw_dir, base + '.txt'),
                        os.path.join(lbl_dst, split, base + '.txt'))
            stats[split] += 1
        print(f'{video}: {n} 张已标注 -> train {n - n_val} / val {n_val}')

    print(f'\n总计: train {stats["train"]} / val {stats["val"]}。'
          f'可运行: python3 scripts/train.py')


if __name__ == '__main__':
    main()
