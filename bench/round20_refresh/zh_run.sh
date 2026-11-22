#!/bin/bash
# Round 20 release check, step 2: re-run the Mandarin parts of the round-19 frozen run
# (bench/round19_merge/frozen_run.sh) with the release code, in a COPY of the frozen-run overlay
# whose Mandarin outputs were moved aside (fleurs-test-zh / fleurs-swap-zh pickles, the zh
# contrast-audit chunks, the OMPAL-test coaching dump) and whose fingerprint-keyed caches were seeded
# with the other nine languages' frozen-run outputs (bench/round20_refresh/seed.py). Every step that
# reads Mandarin outputs is run again; steps without Mandarin rows (English scoring and dumps,
# Japanese length, Lingolix) are not. Same settings as the frozen run: Azure variables removed from
# every job (cached responses only), scoring through python -m bench.heavy --lane cpu.
#   bash bench/round20_refresh/zh_run.sh "<python>" <overlay cache> <log dir> [first step]
set -u
PY="$1"
OV="$2"
LOG="$3"
FIRST="${4:-1}"
mkdir -p "$LOG"
TSV="$LOG/zh_times.tsv"

step() {  # n heavy(0/1) module args...
  local n=$1 heavy=$2; shift 2
  [ "$n" -lt "$FIRST" ] && return 0
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') START [$n] load $(sysctl -n vm.loadavg) $*" >> "$LOG/zh.log"
  local t0=$(date +%s)
  local job=(env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION -u AZURE_SPEECH_ENDPOINT
             PRONUNCIADA_CACHE="$OV" PYTHONHASHSEED=0 OMP_NUM_THREADS=2 "$PY" -m "$@")
  if [ "$heavy" = 1 ]; then
    env -u PRONUNCIADA_CACHE "$PY" -m bench.heavy --lane cpu --need 25 --floor 12 -- "${job[@]}" >> "$LOG/zh.log" 2>&1
  else
    "${job[@]}" >> "$LOG/zh.log" 2>&1
  fi
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') END [$n] rc=$rc $dt s" >> "$LOG/zh.log"
  printf "%s\t%s\t%s\t%s\n" "$n" "$*" "$rc" "$dt" >> "$TSV"
  if [ $rc -ne 0 ]; then echo "step $n failed (rc=$rc); stopping" >> "$LOG/zh.log"; exit $rc; fi
}

# frozen-run step numbers in brackets
step 1 1 bench.eval_languages --steps run,eval,swap,report --workers 3            # [2] zh natives, planted errors, swaps, tone swaps
step 2 1 bench.eval_cv --steps ours,report --workers 3                            # [3] cv-real (zh scored, others seeded)
step 3 1 bench.eval_ompal --steps ours,report --workers 3                         # [4] OMPAL
step 4 1 bench.coach.dump_r4 --sets ompal-test --workers 3                        # [6] OMPAL-test coaching dump only
step 5 0 bench.coach.azure_yardstick                                              # [7]
step 6 0 bench.coach.heldout_12b                                                  # [8]
step 7 0 bench.coach.fit_r4 eval                                                  # [9]
step 8 1 bench.audit.contrast_audit run --split test --lang zh --workers 2        # [10] zh only
step 9 0 bench.audit.contrast_audit report --split test                           # [11]
step 10 1 bench.learners.evaluate --steps ours,report --workers 3                 # [14] OMPAL part scored, others seeded
step 11 1 bench.coach.tips_r19 --workers 2                                        # [15] no cache: all languages
step 12 0 bench.eval_languages --steps report                                     # [16]
step 13 0 bench.eval_english --steps report                                       # [17]
step 14 1 bench.eval_zipa_baseline --steps eval,report --workers 2                # [18]
step 15 0 bench.scoreboard                                                        # [19]
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ALL DONE" >> "$LOG/zh.log"
