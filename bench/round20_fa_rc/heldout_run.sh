#!/bin/bash
# Round 20 false-alarm release check, step 3: the ONE held-out run of the frozen release candidate.
# The steps of the round-19 frozen run (bench/round19_merge/frozen_run.sh) that have rows for English, German,
# Spanish, French or Korean, in the overlay of bench/round20_fa_rc/overlay.py (main's outputs of those five
# languages moved aside; it / pt / ru / ja / zh outputs kept or seeded, identical by run_equiv.sh). Same settings
# as the frozen run: Azure variables and LINGOLIX_API_KEY removed from every job (cached Azure responses only;
# nothing is sent to Lingolix: Pronunciada is scored on the benchmark clips, and the comparison table is built
# with lingolix_live --report-only from cached Lingolix answers), the round-20 switches unset (the release
# defaults), scoring through python -m bench.heavy --lane cpu. Run from a clean extract of the frozen commit:
# the tables it writes land in that extract's results/.
#   bash bench/round20_fa_rc/heldout_run.sh "<python>" <overlay> <log dir> <live cache dir> [first step]
set -u
PY="$1"
OV="$2"
LOG="$3"
LIVE="$4"
FIRST="${5:-1}"
mkdir -p "$LOG"
TSV="$LOG/times.tsv"

step() {  # n heavy(0/1) module args...
  local n=$1 heavy=$2; shift 2
  [ "$n" -lt "$FIRST" ] && return 0
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') START [$n] load $(sysctl -n vm.loadavg) $*" >> "$LOG/run.log"
  local t0=$(date +%s)
  local job=(env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION -u AZURE_SPEECH_ENDPOINT -u LINGOLIX_API_KEY
             -u PRONUNCIADA_R20_NUMALT -u PRONUNCIADA_R20_DE_LEXALT -u PRONUNCIADA_R20_ES_GLIDE
             -u PRONUNCIADA_R20_KO_NINS -u PRONUNCIADA_R20_UNSAID -u PRONUNCIADA_R20_GUESS
             PRONUNCIADA_CACHE="$OV" PYTHONHASHSEED=0 OMP_NUM_THREADS=2 "$PY" -m "$@")
  if [ "$heavy" = 1 ]; then
    env -u PRONUNCIADA_CACHE "$PY" -m bench.heavy --lane cpu --need 25 --floor 12 -- "${job[@]}" >> "$LOG/run.log" 2>&1
  else
    "${job[@]}" >> "$LOG/run.log" 2>&1
  fi
  local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') END [$n] rc=$rc $dt s" >> "$LOG/run.log"
  printf "%s\t%s\t%s\t%s\n" "$n" "$*" "$rc" "$dt" >> "$TSV"
  if [ $rc -ne 0 ]; then echo "step $n failed (rc=$rc); stopping" >> "$LOG/run.log"; exit $rc; fi
}

# round-19 frozen-run step numbers in brackets
step 1 1 bench.eval_english --steps run,so762,calib,errdet,int8,report --workers 4         # [1]
step 2 1 bench.eval_languages --steps run,eval,swap,report --workers 3                     # [2] five languages scored
step 3 1 bench.eval_cv --steps ours,report --workers 3                                     # [3] five languages scored
step 4 1 bench.eval_ompal --steps ours,report --workers 3                                  # [4] all seeded (zh)
step 5 1 bench.coach.dump --sets so762-test --workers 2                                    # [5]
step 6 1 bench.coach.dump_r4 --sets so762-test,l2arctic-test,epadb-test --workers 3        # [6] ompal-test kept
step 7 0 bench.coach.azure_yardstick                                                       # [7]
step 8 0 bench.coach.heldout_12b                                                           # [8]
step 9 0 bench.coach.fit_r4 eval                                                           # [9]
step 10 1 bench.audit.contrast_audit run --split test --lang en,de,es,fr,ko --workers 2    # [10]
step 11 0 bench.audit.contrast_audit report --split test                                   # [11]
step 12 1 bench.third_party.lingolix score --only tier1_en tier1_fr tier1_es tier1_de tier1_ko tier2 tier2x --workers 2  # [13]
step 13 0 bench.third_party.lingolix report                                                # [13]
step 14 1 bench.learners.evaluate --steps ours,report --workers 3                          # [14]
step 15 1 bench.coach.tips_r19 --workers 2                                                 # [15] no cache: all languages
step 16 0 bench.eval_languages --steps report                                              # [16]
step 17 0 bench.eval_english --steps report                                                # [17]
step 18 1 bench.eval_zipa_baseline --steps eval,report --workers 2                         # [18]
step 19 0 bench.scoreboard                                                                 # [19]
step 20 0 bench.third_party.lingolix_live --report-only --pronunciada-cache "$OV" \
    --pronunciada-tier2x "$OV/third_party/lingolix/responses/pronunciada/tier2x.jsonl" --cache-dir "$LIVE"
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ALL DONE" >> "$LOG/run.log"
