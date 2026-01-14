#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/home/notebook/code/group/zhengxianwu/Project-embed/train_data-main/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 93856 --raw_end 99722 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 99722 --raw_end 105588 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 105588 --raw_end 111454 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 111454 --raw_end 117320 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 117320 --raw_end 123186 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 123186 --raw_end 129052 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 129052 --raw_end 134918 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 134918 --raw_end 140784 &

wait
