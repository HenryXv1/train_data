#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/data5/llm/xhr/project/unimev2-train-data/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 362940 --raw_end 374281 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 374281 --raw_end 385622 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 385622 --raw_end 396963 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 396963 --raw_end 408304 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 408304 --raw_end 419645 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 419645 --raw_end 430986 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 430986 --raw_end 442327 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 442327 --raw_end 453675 &

wait