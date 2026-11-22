#!/bin/bash
# Round 20 (latency): equivalence dumps, main 8c74789 vs this branch, every output number of
# bench/round19_merge/equiv.py. Korean and Japanese: 200 FLEURS validation clips (400 scores: own
# text + one word swapped); the other eight languages: 100 clips (200 scores); fp32 and int8.
# Source trees: $SRC/old and $SRC/new (git archive extracts). 4 dumps at a time, 2 threads each;
# run it through the lock:  python -m bench.heavy --lane cpu --need 25 --floor 12 -- bash <this> ...
#   bash bench/round20_latency/run_equiv.sh "<python>" <src dir> <out dir>
set -u
PY="$1"; SRC="$2"; OUT="$3"
EQ="$SRC/new/bench/round19_merge/equiv.py"
mkdir -p "$OUT"
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2
jobs=()
for prec in fp32 int8; do
  for lang in ko ja en de zh es fr it pt ru; do
    n=100; case $lang in ko|ja) n=200;; esac
    for tree in old new; do jobs+=("$tree $lang $prec $n"); done
  done
done
one() {
  set -- $1
  local f="$OUT/$1_$2_fleurs-val_$3.jsonl"
  [ -s "$f.done" ] && return 0
  env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION PYTHONPATH="$SRC/$1" "$PY" "$EQ" dump --lang "$2" --set fleurs-val \
      --precision "$3" --n "$4" --out "$f" > "$f.done.tmp" 2> "$f.err" && mv "$f.done.tmp" "$f.done"
  echo "$(date +%H:%M:%S) $1 $2 $3 $(test -s "$f.done" && echo ok || echo FAILED)"
}
export -f one
export PY SRC OUT EQ
printf '%s\n' "${jobs[@]}" | xargs -P 4 -I{} bash -c 'one "{}"'
echo "$(date +%H:%M:%S) all dumps done"
