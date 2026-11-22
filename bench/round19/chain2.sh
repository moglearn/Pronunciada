#!/bin/bash
# Round 19 second pass: remaining CV runs, one after the other (one process at a time).
L="${PRONUNCIADA_CACHE:-$HOME/.cache/pronunciada}/round19/english"
cd "$(dirname "$0")/../.."
bench/round19/run_seq.sh t_bott --bottleneck
OMP_NUM_THREADS=2 "$PY" -m bench.round19.exp_level --seq t_ccc > "$L/exp_level_t_ccc.log" 2>&1
bench/round19/run_seq.sh t_ens3 --ens 3
echo chain2 done >> "$L/chain2.log"
