#!/bin/bash
# Round 20 release check, step 4: on-device cost (bench.ondevice) of main and of 1.0.2 (an eaa4080
# extract, whose pronunciada/ equals c9773cc) in the same session, alternating pair by pair, int8 and
# fp32 at 1 and 4 threads, one job at a time through the machine-wide lock, as
# bench/round19_merge/cost_runs.sh did. bench.ondevice writes results/tables/ondevice/ in the tree it
# runs from; the main runs are moved to results/tables/ondevice_round20/ and the committed
# results/tables/ondevice/ is restored afterwards (git checkout). Run from the repository root:
#   bash bench/round20_refresh/cost_runs.sh "<python>" <1.0.2 tree> <log>
set -u
PY="$1"; OLD="$2"; LOG="$3"
ROOT="$(pwd)"
OUT="$ROOT/results/tables/ondevice_round20"
mkdir -p "$OUT"
heavy() { env -u PRONUNCIADA_CACHE -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION -u AZURE_SPEECH_ENDPOINT \
          "$PY" -m bench.heavy --need 30 --floor 12 -- "$@" >> "$LOG" 2>&1; }

for prec in int8 fp32; do
  for th in 1 4; do
    echo "=== $(date +%T) ondevice main $prec t$th load $(sysctl -n vm.loadavg)" >> "$LOG"
    (cd "$ROOT" && heavy "$PY" -m bench.ondevice --system pronunciada-$prec --threads $th)
    cp "$ROOT/results/tables/ondevice/pronunciada-${prec}__t${th}.json" "$OUT/"
    echo "=== $(date +%T) ondevice 1.0.2 $prec t$th load $(sysctl -n vm.loadavg)" >> "$LOG"
    (cd "$OLD" && heavy "$PY" -m bench.ondevice --system pronunciada-$prec --threads $th)
    cp "$OLD/results/tables/ondevice/pronunciada-${prec}__t${th}.json" "$OUT/pronunciada-1.0.2-same-session-${prec}__t${th}.json"
  done
done
# Mandarin sanity check only (Korean and Japanese: results/round20_latency.md): paired, interleaved
# latency of the round-19 frozen-run code (old) and main (new), FLEURS validation, 4 threads.
if [ -n "${R19_TREE:-}" ] && [ -n "${MAIN_TREE:-}" ]; then
  for prec in fp32 int8; do
    f="$OUT/../round20_release_latency_zh_${prec}.json"
    [ -s "$f" ] && continue
    echo "=== $(date +%T) latency zh $prec load $(sysctl -n vm.loadavg)" >> "$LOG"
    env -u PRONUNCIADA_CACHE PYTHONDONTWRITEBYTECODE=1 "$PY" -m bench.heavy --lane cpu --need 25 --floor 12 -- \
      "$PY" "$ROOT/bench/round19_merge/latency.py" --old "$R19_TREE" --new "$MAIN_TREE" --lang zh --set fleurs-val \
      --precision $prec --threads 4 --n 100 --out "$f" >> "$LOG" 2>&1
  done
fi
echo "=== $(date +%T) ALL DONE load $(sysctl -n vm.loadavg)" >> "$LOG"
