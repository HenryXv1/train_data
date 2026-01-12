#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=/data5/llm/xhr/project/unimev2-train-data/train_data.py

$PYTHON "$SCRIPT" --cuda 0 --raw_start 453675 --raw_end 465016 &
$PYTHON "$SCRIPT" --cuda 1 --raw_start 465016 --raw_end 476357 &
$PYTHON "$SCRIPT" --cuda 2 --raw_start 476357 --raw_end 487698 &
$PYTHON "$SCRIPT" --cuda 3 --raw_start 487698 --raw_end 499039 &
$PYTHON "$SCRIPT" --cuda 4 --raw_start 499039 --raw_end 510380 &
$PYTHON "$SCRIPT" --cuda 5 --raw_start 510380 --raw_end 521721 &
$PYTHON "$SCRIPT" --cuda 6 --raw_start 521721 --raw_end 533062 &
$PYTHON "$SCRIPT" --cuda 7 --raw_start 533062 --raw_end 544410 &

wait