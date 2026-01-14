#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/home/notebook/code/group/zhengxianwu/Project-embed/train_data-main/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 46928 --raw_end 52794 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 52794 --raw_end 58660 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 58660 --raw_end 64526 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 64526 --raw_end 70392 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 70392 --raw_end 76258 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 76258 --raw_end 82124 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 82124 --raw_end 87990 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 87990 --raw_end 93856 &

wait
