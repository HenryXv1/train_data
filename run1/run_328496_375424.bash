#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/home/notebook/code/group/zhengxianwu/Project-embed/train_data-main/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 328496 --raw_end 334362 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 334362 --raw_end 340228 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 340228 --raw_end 346094 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 346094 --raw_end 351960 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 351960 --raw_end 357826 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 357826 --raw_end 363692 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 363692 --raw_end 369558 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 369558 --raw_end 375424 &

wait
