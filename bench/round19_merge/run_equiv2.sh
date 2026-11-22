#!/bin/bash
# Round 19 merge checks B and C, second pass: the fully merged code (Japanese final c5eb79c and the
# sound-identity rollout 51ba321 added) against each branch and, with every switch off, against main.
# Source trees: $OUT/../src/{merged2,ja_final,id} (git archive extracts); the main, en, ko and de
# dumps of the first pass (run_equiv.sh) are reused. Resumable.
#   bash bench/round19_merge/run_equiv2.sh "<python>" <out dir>
set -u
PY="$1"
OUT="$2"
SRC="$OUT/../src"
HERE="$(cd "$(dirname "$0")" && pwd)"
OFF="PRONUNCIADA_SENTENCE_HEAD=0 PRONUNCIADA_KO_IDENTITY=0 PRONUNCIADA_DE_R19=0 PRONUNCIADA_DE_IDENTITY=0 \
PRONUNCIADA_DE_STRESS_REL=0 PRONUNCIADA_DE_LOAN=0 PRONUNCIADA_WORD_LANG=0 PRONUNCIADA_DE_TEXT=0 PRONUNCIADA_JA_ROUND19=0 \
PRONUNCIADA_IDENTITY=0"
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2

run() {  # name tree lang set precision [env...]
  local name=$1 tree=$2 lang=$3 set=$4 prec=$5; shift 5
  local f="$OUT/${name}_${lang}_${set}_${prec}.jsonl"
  [ -s "$f.done" ] && return
  echo "$(date +%H:%M:%S) $name $lang $set $prec"
  env "$@" PYTHONPATH="$SRC/$tree" "$PY" "$HERE/equiv.py" dump --lang "$lang" --set "$set" \
      --precision "$prec" --out "$f" > "$f.done.tmp" 2>"$f.err" && mv "$f.done.tmp" "$f.done"
}

for prec in fp32 int8; do
  for lang in en ko de ja zh es fr it pt ru; do
    run merged2 merged2 $lang fleurs-val $prec
    run merged2off merged2 $lang fleurs-val $prec $OFF
  done
  run branchjaf ja_final ja fleurs-val $prec
  for lang in es ru pt zh de; do
    run branchid id $lang fleurs-val $prec
  done
  run merged2 merged2 en so762-train $prec
  run merged2off merged2 en so762-train $prec $OFF
done
echo "$(date +%H:%M:%S) all dumps done"
