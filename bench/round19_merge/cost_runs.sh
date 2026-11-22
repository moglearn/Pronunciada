#!/bin/bash
# Round 19 merge, check F: on-device cost (bench.ondevice) of the merged code and of 1.0.2 (main
# eaa4080 extract), alternating, then the paired latency check (bench/round19_merge/latency.py),
# old = main c9773cc extract, new = merged extract, interleaved clip by clip. One job at a time
# through the machine-wide lock. Run from the repository root:
#   bash bench/round19_merge/cost_runs.sh "<python>" <src dir> <out dir>
set -u
PY="$1"; SRC="$2"; OUT="$3"
ROOT="$(pwd)"
mkdir -p "$OUT/ondevice_main" "$OUT/latency"
LOG="$OUT/cost_runs.log"
heavy() { env -u PRONUNCIADA_CACHE "$PY" -m bench.heavy --need 30 --floor 12 -- "$@" >> "$LOG" 2>&1; }

for prec in int8 fp32; do
  for th in 1 4; do
    [ -n "${SKIP_ONDEVICE:-}" ] && continue
    echo "=== $(date +%T) ondevice round19 $prec t$th load $(sysctl -n vm.loadavg)" >> "$LOG"
    (cd "$ROOT" && heavy "$PY" -m bench.ondevice --system pronunciada-$prec --threads $th)
    echo "=== $(date +%T) ondevice 1.0.2 $prec t$th load $(sysctl -n vm.loadavg)" >> "$LOG"
    (cd "$SRC/main_eaa" && heavy "$PY" -m bench.ondevice --system pronunciada-$prec --threads $th)
  done
done
cp "$SRC"/main_eaa/results/tables/ondevice/*.json "$OUT/ondevice_main/" 2>/dev/null

for prec in fp32 int8; do
  for spec in en:so762-train en:fleurs-val ko:fleurs-val de:fleurs-val ja:fleurs-val es:fleurs-val ru:fleurs-val pt:fleurs-val zh:fleurs-val fr:fleurs-val; do
    lang=${spec%%:*}; set=${spec#*:}
    f="$OUT/latency/${lang}_${set}_${prec}.json"
    [ -s "$f" ] && continue
    echo "=== $(date +%T) latency $lang $set $prec load $(sysctl -n vm.loadavg)" >> "$LOG"
    env -u PRONUNCIADA_CACHE "$PY" -m bench.heavy --lane cpu --need 25 --floor 12 -- \
      "$PY" "$ROOT/bench/round19_merge/latency.py" --old "$SRC/main" --new "$SRC/merged2" --lang $lang --set $set \
      --precision $prec --threads 4 --n 100 --out "$f" >> "$LOG" 2>&1
  done
done
echo "=== $(date +%T) ALL DONE" >> "$LOG"
