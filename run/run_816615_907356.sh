#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/data5/llm/xhr/project/unimev2-train-data/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 816615 --raw_end 827957 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 827957 --raw_end 839299 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 839299 --raw_end 850641 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 850641 --raw_end 861983 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 861983 --raw_end 873325 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 873325 --raw_end 884667 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 884667 --raw_end 896009 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 896009 --raw_end 907356 &

wait