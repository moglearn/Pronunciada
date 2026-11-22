#!/bin/bash
# Round 20 switch check: dump every output (bench/round19_merge/equiv.py) of main (8c74789) and of this
# branch on the same 100 FLEURS-validation clips (own text + one changed word) per language, fp32 and int8.
#   bash bench/round20_mandarin/run_equiv.sh "<python>" <out dir>
# Source trees are `git archive` extracts under <out dir>/../src/{main,new}. Resumable.
#   main       main, defaults
#   new        this branch, defaults            (must equal main for every language but zh)
#   newoff     this branch, PRONUNCIADA_IDENTITY_ZH_R20=0 (must equal main for zh)
set -u
PY="$1"; OUT="$2"; SRC="$OUT/../src"
HERE="$(cd "$(dirname "$0")/../round19_merge" && pwd)"
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$OUT"
run() {  # name tree lang precision [env...]
  local name=$1 tree=$2 lang=$3 prec=$4; shift 4
  local f="$OUT/${name}_${lang}_${prec}.jsonl"
  [ -s "$f.done" ] && return
  echo "$(date +%H:%M:%S) $name $lang $prec"
  env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION "$@" PYTHONPATH="$SRC/$tree" "$PY" -W ignore "$HERE/equiv.py" dump \
      --lang "$lang" --set fleurs-val --precision "$prec" --out "$f" > "$f.done.tmp" 2>"$f.err" && mv "$f.done.tmp" "$f.done"
}
for prec in fp32 int8; do
  for lang in zh pt es ru en ko de ja fr it; do
    run main main $lang $prec
    run new new $lang $prec
  done
  run newoff new zh $prec PRONUNCIADA_IDENTITY_ZH_R20=0
done
echo "$(date +%H:%M:%S) all dumps done"
