#!/bin/sh
# Mandarin prep jobs, sequential (one ZIPA process, 4 threads)
cd "$(dirname "$0")/../.."
export PYTHONPATH=. OMP_NUM_THREADS=4
for job in "aishell3 test" "zhtw test" "fleurs_val dev zh" "fleurs test zh" "aishell3 train" "fleurs train zh"; do
  .venv/bin/python -W ignore -m bench.prosody.prep $job
done
