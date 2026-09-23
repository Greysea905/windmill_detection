#!/usr/bin/env bash
# LabelImg 启动脚本（已修复 float→int 崩溃）
# 用法（先 conda activate windmill）：
#   bash scripts/label.sh task_3
#   bash scripts/label.sh task_4
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VIDEO="${1:-task_3}"

export QT_QPA_PLATFORM=xcb            # 强制 X11 后端，稳定
export QT_AUTO_SCREEN_SCALE_FACTOR=1  # 4K 屏幕自适应缩放
# 若窗口仍然太小，把上一行换成固定缩放因子：
# export QT_SCALE_FACTOR=2

exec labelImg "dataset/raw/$VIDEO" config/predefined_classes.txt
