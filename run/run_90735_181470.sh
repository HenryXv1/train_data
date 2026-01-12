#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/data5/llm/xhr/project/unimev2-train-data/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 90735 --raw_end 102076 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 102076 --raw_end 113417 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 113417 --raw_end 124758 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 124758 --raw_end 136099 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 136099 --raw_end 147440 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 147440 --raw_end 158781 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 158781 --raw_end 170122 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 170122 --raw_end 181470 &

wait