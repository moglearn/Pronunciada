#!/bin/sh
cd "$(dirname "$0")/../.."
export PYTHONPATH=. OMP_NUM_THREADS=4
.venv/bin/python -W ignore -m bench.prosody.prep fleurs_val dev zh
./bench/prosody/run_prep.sh ja
