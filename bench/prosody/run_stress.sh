#!/bin/sh
# shared multilingual model + per-language models, then evaluation
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
L=~/.cache/pronunciada/prosody/logs
SUF=${1:-}
.venv/bin/python -W ignore -m bench.prosody.stress_train --name stress_multi$SUF --epochs 15 > $L/stress_multi$SUF.log 2>&1
for l in en de es it pt ru; do
  .venv/bin/python -W ignore -m bench.prosody.stress_train --langs $l --no-lang-emb --name stress_$l$SUF --epochs 15 > $L/stress_$l$SUF.log 2>&1
done
OMP_NUM_THREADS=4 .venv/bin/python -W ignore -m bench.prosody.stress_eval stress_multi$SUF stress_en$SUF stress_de$SUF stress_es$SUF stress_it$SUF stress_pt$SUF stress_ru$SUF > $L/stress_eval$SUF.log 2>&1
