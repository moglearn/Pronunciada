#!/bin/bash
# Round 20 release check, step 1: main (the code to be released) against the code of the round-19
# frozen run 2 (836dba6), every output, tolerance 0. Dumps the release code with
# bench/round19_merge/equiv.py on its standard clip sets (FLEURS validation 100 clips x 2 texts per
# language, plus 100 speechocean762 train clips for English), fp32 and int8. The frozen-run code's
# dumps are the round-19 merge check's (merged2_*); if a language differs, the frozen-run code is
# dumped again here (merged2re_*) to rule out a changed environment. Resumable.
#   bash bench/round20_refresh/run_equiv.sh "<python>" <release tree> <frozen tree> <round-19 dump dir> <out dir>
set -u
PY="$1"; NEW="$2"; OLD="$3"; R19="$4"; OUT="$5"
HERE="$(cd "$(dirname "$0")/.." && pwd)/round19_merge"
mkdir -p "$OUT"
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2

run() {  # name tree lang set precision
  local name=$1 tree=$2 lang=$3 set=$4 prec=$5
  local f="$OUT/${name}_${lang}_${set}_${prec}.jsonl"
  [ -s "$f.done" ] && return
  echo "$(date +%H:%M:%S) $name $lang $set $prec load $(sysctl -n vm.loadavg)"
  env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION -u AZURE_SPEECH_ENDPOINT PYTHONPATH="$tree" \
      "$PY" "$HERE/equiv.py" dump --lang "$lang" --set "$set" --precision "$prec" --out "$f" \
      > "$f.done.tmp" 2>"$f.err" && mv "$f.done.tmp" "$f.done"
}

cmp0() {  # lang set precision -> prints OK / DIFFERENT
  "$PY" "$HERE/equiv.py" compare "$OUT/main_$1_$2_$3.jsonl" "$R19/merged2_$1_$2_$3.jsonl" --tol 0 \
      > "$OUT/cmp_$1_$2_$3.json" && echo OK || echo DIFFERENT
}

for prec in fp32 int8; do
  for lang in en ko de ja es fr it pt ru zh; do
    run main "$NEW" $lang fleurs-val $prec
    r=$(cmp0 $lang fleurs-val $prec)
    echo "$(date +%H:%M:%S) compare $lang fleurs-val $prec: $r"
    if [ "$r" != OK ] && [ "$lang" != zh ]; then
      run merged2re "$OLD" $lang fleurs-val $prec
    fi
  done
  run main "$NEW" en so762-train $prec
  echo "$(date +%H:%M:%S) compare en so762-train $prec: $(cmp0 en so762-train $prec)"
done
echo "$(date +%H:%M:%S) all done"
