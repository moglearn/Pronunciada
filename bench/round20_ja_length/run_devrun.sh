#!/bin/bash
# Every dev set (2 shards each) and the FLEURS-validation length table (4 shards), 4 jobs at a time
# (run inside python -m bench.heavy --lane cpu).
# usage: bench/round20_ja_length/run_devrun.sh PYTHON HEAD GRID TAG
PY=$1; HEAD=$2; GRID=$3; TAG=$4
LOG="$HOME/.cache/pronunciada/round20/jalength/devrun_$TAG"
mkdir -p "$LOG"
{
  for s in lentab; do for k in 0 1 2 3; do echo "$s $k 4"; done; done
  for s in dev devnat devswap cvdev; do for k in 0 1; do echo "$s $k 2"; done; done
} | xargs -P 4 -L 1 bash -c 'env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION OMP_NUM_THREADS=2 \
    "$0" -m bench.round20_ja_length.devrun "$1" --head "'"$HEAD"'" --grid "'"$GRID"'" --shard "$2" --of "$3" \
    --tag "'"$TAG"'" > "'"$LOG"'/$1_$2.log" 2>&1' "$PY"
