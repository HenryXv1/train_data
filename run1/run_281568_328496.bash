#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/home/notebook/code/group/zhengxianwu/Project-embed/train_data-main/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 281568 --raw_end 287434 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 287434 --raw_end 293300 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 293300 --raw_end 299166 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 299166 --raw_end 305032 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 305032 --raw_end 310898 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 310898 --raw_end 316764 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 316764 --raw_end 322630 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 322630 --raw_end 328496 &

wait
