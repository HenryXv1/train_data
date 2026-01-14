#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/home/notebook/code/group/zhengxianwu/Project-embed/train_data-main/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 375424 --raw_end 381290 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 381290 --raw_end 387156 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 387156 --raw_end 393022 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 393022 --raw_end 398888 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 398888 --raw_end 404754 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 404754 --raw_end 410620 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 410620 --raw_end 416486 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 416486 --raw_end 422352 &

wait
