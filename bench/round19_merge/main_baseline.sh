#!/bin/bash
# Baseline for check E: main c9773cc run on the benchmarks whose audio is resampled (cv-real, OMPAL,
# Lingolix), where the committed main tables predate 1.0.1 and so are not main's current outputs.
# Run from a `git archive` extract of c9773cc (pronunciada, bench, results):
#   bash bench/round19_merge/main_baseline.sh "<python>" <extract dir> <overlay cache> <log dir>
set -u
PY="$1"; TREE="$2"; OV="$3"; LOG="$4"
cd "$TREE" || exit 1
mkdir -p "$LOG"
run() {
  echo "=== $(date +%T) $*" >> "$LOG/main_baseline.log"
  env -u PRONUNCIADA_CACHE "$PY" -m bench.heavy --lane cpu --need 25 --floor 12 -- \
    env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION -u AZURE_SPEECH_ENDPOINT PRONUNCIADA_CACHE="$OV" \
    PYTHONHASHSEED=0 OMP_NUM_THREADS=2 "$PY" -m "$@" >> "$LOG/main_baseline.log" 2>&1
  echo "=== $(date +%T) rc=$?" >> "$LOG/main_baseline.log"
}
run bench.eval_cv --steps ours,report --workers 3
run bench.eval_ompal --steps ours,report --workers 3
run bench.third_party.lingolix all --workers 2
echo "=== $(date +%T) ALL DONE" >> "$LOG/main_baseline.log"
