#!/bin/bash
# Round 19 merged code: one frozen run of the held-out benchmark suite, the steps and settings of
# the 2026-11-17 / 18 frozen runs (results/README.md) plus the Lingolix benchmark, the learner
# suite and the coaching-tips table. Run from the repository root:
#   bash bench/round19_merge/frozen_run.sh "<python>" <overlay cache> <log dir> [first step]
# * every Pronunciada output is computed fresh: PRONUNCIADA_CACHE points at an overlay
#   (bench/round19_merge/overlay.py) whose output folders start empty;
# * no Azure request can be sent: the key, region and endpoint variables are removed from every
#   job's environment (Azure numbers come from the cached responses only);
# * scoring jobs go through the machine-wide lock: python -m bench.heavy --lane cpu.
set -u
PY="$1"
OV="$2"
LOG="$3"
FIRST="${4:-1}"
mkdir -p "$LOG"
TSV="$LOG/frozen_times.tsv"

step() {  # n heavy(0/1) module args...
  local n=$1 heavy=$2; shift 2
  [ "$n" -lt "$FIRST" ] && return 0
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') START [$n] $*" >> "$LOG/frozen.log"
  local t0=$(date +%s)
  local job=(env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION -u AZURE_SPEECH_ENDPOINT
             PRONUNCIADA_CACHE="$OV" PYTHONHASHSEED=0 OMP_NUM_THREADS=2 "$PY" -m "$@")
  if [ "$heavy" = 1 ]; then
    env -u PRONUNCIADA_CACHE "$PY" -m bench.heavy --lane cpu --need 25 --floor 12 -- "${job[@]}" >> "$LOG/frozen.log" 2>&1
  else
    "${job[@]}" >> "$LOG/frozen.log" 2>&1
  fi
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') END [$n] rc=$rc $dt s" >> "$LOG/frozen.log"
  printf "%s\t%s\t%s\t%s\n" "$n" "$*" "$rc" "$dt" >> "$TSV"
  if [ $rc -ne 0 ]; then echo "step $n failed (rc=$rc); stopping" >> "$LOG/frozen.log"; exit $rc; fi
}

step 1 1 bench.eval_english --steps run,so762,calib,errdet,int8,report --workers 4
step 2 1 bench.eval_languages --steps run,eval,swap,report --workers 3
step 3 1 bench.eval_cv --steps ours,report --workers 3
step 4 1 bench.eval_ompal --steps ours,report --workers 3
step 5 1 bench.coach.dump --sets so762-test --workers 2
step 6 1 bench.coach.dump_r4 --sets so762-test,l2arctic-test,epadb-test,ompal-test --workers 3
step 7 0 bench.coach.azure_yardstick
step 8 0 bench.coach.heldout_12b
step 9 0 bench.coach.fit_r4 eval
step 10 1 bench.audit.contrast_audit run --split test --workers 2
step 11 0 bench.audit.contrast_audit report --split test
step 12 1 bench.length.heldout_runtime --workers 2
step 13 1 bench.third_party.lingolix all --workers 2
step 14 1 bench.learners.evaluate --steps ours,report --workers 3
step 15 1 bench.coach.tips_r19 --workers 2
step 16 0 bench.eval_languages --steps report
step 17 0 bench.eval_english --steps report
step 18 1 bench.eval_zipa_baseline --steps eval,report --workers 2
step 19 0 bench.scoreboard
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ALL DONE" >> "$LOG/frozen.log"
