#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/data5/llm/xhr/project/unimev2-train-data/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 0 --raw_end 11341 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 11341 --raw_end 22682 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 22682 --raw_end 34023 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 34023 --raw_end 45364 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 45364 --raw_end 56705 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 56705 --raw_end 68046 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 68046 --raw_end 79387 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 79387 --raw_end 90735 &

wait