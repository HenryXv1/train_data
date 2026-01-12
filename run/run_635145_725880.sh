#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/data5/llm/xhr/project/unimev2-train-data/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 635145 --raw_end 646486 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 646486 --raw_end 657827 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 657827 --raw_end 669168 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 669168 --raw_end 680509 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 680509 --raw_end 691850 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 691850 --raw_end 703191 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 703191 --raw_end 714532 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 714532 --raw_end 725880 &

wait