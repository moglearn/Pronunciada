#!/bin/bash
# Round 19: one sequence-head CV run with the frozen base settings (s6_smallreg) plus extra flags.
#   bench/round19/run_seq.sh NAME [extra exp_seq flags...]
# Uses $PY (the project interpreter) and logs to $PRONUNCIADA_CACHE/round19/english/seq_NAME.log.
set -e
name=$1; shift
L="${PRONUNCIADA_CACHE:-$HOME/.cache/pronunciada}/round19/english"
export OMP_NUM_THREADS=2
"$PY" -m bench.round19.exp_seq --name "$name" --d 64 --n-layers 1 --ff 128 --epochs 25 --dropout 0.2 \
  --w-corr 1.0 "$@" > "$L/seq_$name.log" 2>&1
