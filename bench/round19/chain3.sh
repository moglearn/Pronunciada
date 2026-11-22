#!/bin/bash
# Round 19 second pass: blend weights for the 3-model sequence head (sentence, sound, word).
L="${PRONUNCIADA_CACHE:-$HOME/.cache/pronunciada}/round19/english"
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2
"$PY" -m bench.round19.exp_level --seq t_ens3 > "$L/exp_level_t_ens3.log" 2>&1
"$PY" -m bench.round19.cv_blend --seq t_ens3 > "$L/cv_blend_t_ens3.log" 2>&1
echo chain3 done >> "$L/chain3.log"
