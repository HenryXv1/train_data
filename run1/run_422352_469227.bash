#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/home/notebook/code/group/zhengxianwu/Project-embed/train_data-main/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 422352 --raw_end 428218 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 428218 --raw_end 434084 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 434084 --raw_end 439950 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 439950 --raw_end 445816 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 445816 --raw_end 451682 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 451682 --raw_end 457548 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 457548 --raw_end 463414 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 463414 --raw_end 469227 &

wait
