#!/bin/sh
# speechocean762 TRAIN speaker-grouped CV exactly as round 19 measured it (results/round19_english.md, second
# pass): re-extract the feature cache with the current code, the 3-seed x 5-fold sequence head "t_ens3"
# (ensemble of 3 per fold), its blend with the tree candidate (sentence scores), and the sound / word blends.
# The round-19 scripts write under $PRONUNCIADA_CACHE/round19/english; to keep the shared cache untouched they
# are run with PRONUNCIADA_CACHE pointing at a private root whose entries are symlinks to the real cache
# (all but round19 / round20 / the lock files) plus an own round19/english. The tables they write into
# results/tables are copied out and restored afterwards (git checkout), so nothing tracked changes.
#   PY=/path/to/python sh bench/round20_native_fa/so762cv.sh
set -e
PY="${PY:-python}"
REAL="${PRONUNCIADA_CACHE:-$HOME/.cache/pronunciada}"
R="$REAL/round20/nativefa/fakeroot"
mkdir -p "$R/round19/english/feats" "$R/round19/english/oof" "$REAL/round20/nativefa/so762cv/tables"
for e in "$REAL"/*; do
  n=$(basename "$e")
  case "$n" in round19|round20|heavy.lock|heavy-cpu.lock|heavy-train.lock) continue ;; esac
  [ -e "$R/$n" ] || ln -s "$e" "$R/$n"
done
export PRONUNCIADA_CACHE="$R"
"$PY" -m bench.round19.extract_en --dataset speechocean762 --split train --workers 1
"$PY" -m bench.round19.exp_seq --name t_ens3 --layers enc0+enc1+enc2+enc3+enc4+enc5+final --utt xb+deliv+f0 \
  --epochs 25 --d 64 --n-layers 1 --ff 128 --dropout 0.2 --lr 0.001 --wd 0.01 --w-corr 1.0 --w-word 0.5 \
  --w-phone 0.5 --seeds 0,1,2 --ens 3
"$PY" -m bench.round19.cv_blend --seq t_ens3
"$PY" -m bench.round19.exp_level --seq t_ens3
export PRONUNCIADA_CACHE="$REAL"
for t in seq blend level; do
  cp "results/tables/round19_english_cv_$t.json" "$REAL/round20/nativefa/so762cv/tables/"
  git checkout -- "results/tables/round19_english_cv_$t.json"
done
"$PY" -m bench.round20_native_fa.baselines so762
