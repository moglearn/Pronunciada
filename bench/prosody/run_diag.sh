#!/bin/sh
# tone-ceiling diagnosis: 8-epoch stride-2 runs, one variable changed at a time
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
L=~/.cache/pronunciada/prosody/logs
B="--kind ctx --stride 2 --ch 64 --hid 48 --epochs 8 --workers 3"
run() { n=$1; shift; [ -f ~/.cache/pronunciada/prosody/models/$n.pt ] || .venv/bin/python -W ignore -m bench.prosody.tone_train $B --name $n "$@" > $L/$n.log 2>&1;
        [ -f ~/.cache/pronunciada/prosody/results/$n.json ] || OMP_NUM_THREADS=4 .venv/bin/python -W ignore -m bench.prosody.tone_eval $n > $L/$n.eval.log 2>&1; }
run d_ref
run d_spec --extra spec
run d_rhyme --target rhyme
run d_post_only --extra post_only
run d_post --extra post
run d_frac25 --frac 0.25
run d_frac50 --frac 0.5
run d_lowspec --extra lowspec
echo ALLDONE
