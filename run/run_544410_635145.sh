#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/data5/llm/xhr/project/unimev2-train-data/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 544410 --raw_end 555751 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 555751 --raw_end 567092 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 567092 --raw_end 578433 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 578433 --raw_end 589774 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 589774 --raw_end 601115 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 601115 --raw_end 612456 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 612456 --raw_end 623797 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 623797 --raw_end 635145 &

wait