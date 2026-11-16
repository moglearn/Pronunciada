#!/bin/sh
# usage: run_prep.sh LANG [LANG ...]   prep fleurs train/test/validation for each language, sequentially
cd "$(dirname "$0")/../.."
export PYTHONPATH=. OMP_NUM_THREADS=${PREP_THREADS:-4}
for lang in "$@"; do
  for job in "fleurs_val dev $lang" "fleurs test $lang" "fleurs train $lang"; do
    .venv/bin/python -W ignore -m bench.prosody.prep $job
  done
done
