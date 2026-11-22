#!/bin/bash
# Training jobs one after another (run inside python -m bench.heavy).
# usage: bench/round20_ja_length/run_sweep.sh PYTHON CMD:CFG [CMD:CFG ...]
PY=$1; shift
for job in "$@"; do
  "$PY" -m bench.round20_ja_length.train "${job%%:*}" --cfg "${job##*:}" \
    > "$HOME/.cache/pronunciada/round20/jalength/${job%%:*}_${job##*:}.log" 2>&1
done
