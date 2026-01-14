#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/home/notebook/code/group/zhengxianwu/Project-embed/train_data-main/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 234640 --raw_end 240506 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 240506 --raw_end 246372 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 246372 --raw_end 252238 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 252238 --raw_end 258104 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 258104 --raw_end 263970 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 263970 --raw_end 269836 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 269836 --raw_end 275702 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 275702 --raw_end 281568 &

wait
