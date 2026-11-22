#!/bin/bash
# The dev sets and the FLEURS-validation length table, one process after another (2 threads):
# the single-process alternative to run_devrun.sh when the CPU lane is busy.
# usage: bench/round20_ja_length/run_devrun_seq.sh PYTHON HEAD GRID TAG [SETS...]
PY=$1; HEAD=$2; GRID=$3; TAG=$4; shift 4
SETS=${@:-dev devnat devswap cvdev lentab}
LOG="$HOME/.cache/pronunciada/round20/jalength/devrun_$TAG"
mkdir -p "$LOG"
for s in $SETS; do
  env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION OMP_NUM_THREADS=2 \
    "$PY" -m bench.round20_ja_length.devrun "$s" --head "$HEAD" --grid "$GRID" --tag "$TAG" > "$LOG/$s.log" 2>&1
done
