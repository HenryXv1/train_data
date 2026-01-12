#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/data5/llm/xhr/project/unimev2-train-data/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 725880 --raw_end 737221 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 737221 --raw_end 748562 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 748562 --raw_end 759903 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 759903 --raw_end 771244 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 771244 --raw_end 782585 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 782585 --raw_end 793926 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 793926 --raw_end 805267 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 805267 --raw_end 816615 &

wait