#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/home/notebook/code/group/zhengxianwu/Project-embed/train_data-main/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 140784 --raw_end 146650 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 146650 --raw_end 152516 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 152516 --raw_end 158382 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 158382 --raw_end 164248 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 164248 --raw_end 170114 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 170114 --raw_end 175980 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 175980 --raw_end 181846 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 181846 --raw_end 187712 &

wait
