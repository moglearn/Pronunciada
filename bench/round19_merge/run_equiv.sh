#!/bin/bash
# Round 19 merge checks B and C: dump outputs of main, each branch and the merged code on the same
# clips, one process at a time (2 threads). Source trees are `git archive` extracts of the exact
# commits under $OUT/../src/{main,en,ko,de,ja,merged}. Resumable: finished dumps are kept.
#   bash bench/round19_merge/run_equiv.sh "<python>" <out dir>
set -u
PY="$1"
OUT="$2"
SRC="$OUT/../src"
HERE="$(cd "$(dirname "$0")" && pwd)"
OFF="PRONUNCIADA_SENTENCE_HEAD=0 PRONUNCIADA_KO_IDENTITY=0 PRONUNCIADA_DE_R19=0 PRONUNCIADA_DE_IDENTITY=0 \
PRONUNCIADA_DE_STRESS_REL=0 PRONUNCIADA_DE_LOAN=0 PRONUNCIADA_WORD_LANG=0 PRONUNCIADA_DE_TEXT=0 PRONUNCIADA_JA_ROUND19=0"
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$OUT"

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
    run merged merged $lang fleurs-val $prec
    run main main $lang fleurs-val $prec
  done
  for lang in en ko de ja; do
    run branch $lang $lang fleurs-val $prec
    run mergedoff merged $lang fleurs-val $prec $OFF
  done
  run merged merged en so762-train $prec
  run main main en so762-train $prec
  run branch en en so762-train $prec
  run mergedoff merged en so762-train $prec $OFF
done
echo "$(date +%H:%M:%S) all dumps done"
