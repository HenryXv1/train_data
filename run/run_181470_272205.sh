#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/data5/llm/xhr/project/unimev2-train-data/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 181470 --raw_end 192811 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 192811 --raw_end 204152 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 204152 --raw_end 215493 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 215493 --raw_end 226834 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 226834 --raw_end 238175 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 238175 --raw_end 249516 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 249516 --raw_end 260857 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 260857 --raw_end 272205 &

wait