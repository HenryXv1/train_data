#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/data5/llm/xhr/project/unimev2-train-data/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 272205 --raw_end 283546 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 283546 --raw_end 294887 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 294887 --raw_end 306228 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 306228 --raw_end 317569 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 317569 --raw_end 328910 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 328910 --raw_end 340251 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 340251 --raw_end 351592 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 351592 --raw_end 362940 &

wait