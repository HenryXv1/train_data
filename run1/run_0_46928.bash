#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/home/notebook/code/group/zhengxianwu/Project-embed/train_data-main/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 0 --raw_end 5866 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 5866 --raw_end 11732 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 11732 --raw_end 17598 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 17598 --raw_end 23464 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 23464 --raw_end 29330 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 29330 --raw_end 35196 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 35196 --raw_end 41062 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 41062 --raw_end 46928 &

wait
