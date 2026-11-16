#!/bin/sh
# round-2 tone ablation: data old/new x model local/seq x transplant off/on, 8000 steps each
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
L=~/.cache/pronunciada/prosody/logs
M=~/.cache/pronunciada/prosody/models
R=~/.cache/pronunciada/prosody/results
[ -f $R/tone_ctx_s2_r2.json ] || OMP_NUM_THREADS=4 .venv/bin/python -W ignore -c "
from bench.prosody.tone2 import evaluate; import shutil
from bench.prosody.common import OUT
shutil.copy(OUT/'models'/'tone_ctx_s2.pt', OUT/'models'/'tone_ctx_s2_r2.pt'); evaluate('tone_ctx_s2_r2')" > $L/tone_ctx_s2_r2.eval.log 2>&1
for data in old new; do for model in local seq; do for tp in 0 0.4; do
  n=t2_${data}_${model}_tp${tp}
  [ -f $M/$n.pt ] || .venv/bin/python -W ignore -m bench.prosody.tone2 train --name $n --data $data --model $model --transplant $tp --steps 8000 > $L/$n.log 2>&1
  [ -f $R/$n.json ] || OMP_NUM_THREADS=4 .venv/bin/python -W ignore -m bench.prosody.tone2 eval $n > $L/$n.eval.log 2>&1
done; done; done
echo ALLDONE
