#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/home/notebook/code/group/zhengxianwu/Project-embed/train_data-main/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 187712 --raw_end 193578 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 193578 --raw_end 199444 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 199444 --raw_end 205310 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 205310 --raw_end 211176 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 211176 --raw_end 217042 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 217042 --raw_end 222908 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 222908 --raw_end 228774 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 228774 --raw_end 234640 &

wait
