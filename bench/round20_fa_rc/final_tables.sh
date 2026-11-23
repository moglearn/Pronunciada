#!/bin/bash
# Round 20 false-alarm release check, step 4: rebuild the tables for the final per-language table from per-clip
# outputs that already exist (overlay of bench/round20_fa_rc/final_overlay.py: main's outputs for the languages
# that are off, the frozen run's for Spanish). Only report / evaluation steps run: every scoring step finds its
# outputs cached and scores nothing (checked in the log). The coaching-tips table has no per-clip cache; it is
# composed from main's table and the frozen run's Spanish entry (bench/round20_fa_rc/compose_tips.py).
# Same environment rules as heldout_run.sh. Run from a clean extract of the final commit.
#   bash bench/round20_fa_rc/final_tables.sh "<python>" <overlay> <log dir> <live cache dir> <frozen-run results dir>
set -u
PY="$1"; OV="$2"; LOG="$3"; LIVE="$4"; RCRES="$5"
mkdir -p "$LOG"
step() {
  local n=$1; shift
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') START [$n] $*" >> "$LOG/final.log"
  env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION -u AZURE_SPEECH_ENDPOINT -u LINGOLIX_API_KEY \
      -u PRONUNCIADA_R20_NUMALT -u PRONUNCIADA_R20_DE_LEXALT -u PRONUNCIADA_R20_ES_GLIDE -u PRONUNCIADA_R20_KO_NINS \
      -u PRONUNCIADA_R20_UNSAID -u PRONUNCIADA_R20_GUESS \
      PRONUNCIADA_CACHE="$OV" PYTHONHASHSEED=0 OMP_NUM_THREADS=2 "$PY" -m "$@" >> "$LOG/final.log" 2>&1
  local rc=$?
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') END [$n] rc=$rc" >> "$LOG/final.log"
  [ $rc -eq 0 ] || { echo "step $n failed" >> "$LOG/final.log"; exit $rc; }
}
step 1 bench.eval_english --steps run,so762,calib,errdet,int8,report --workers 4
step 2 bench.eval_languages --steps run,eval,swap,report --workers 3
step 3 bench.eval_cv --steps ours,report --workers 3
step 4 bench.eval_ompal --steps ours,report --workers 3
step 7 bench.coach.azure_yardstick
step 8 bench.coach.heldout_12b
step 9 bench.coach.fit_r4 eval
step 11 bench.audit.contrast_audit report --split test
step 13 bench.third_party.lingolix report
step 14 bench.learners.evaluate --steps ours,report --workers 3
step 15 bench.round20_fa_rc.compose_tips "$RCRES/tables/coaching_tips_r19.json" es
step 16 bench.eval_languages --steps report
step 17 bench.eval_english --steps report
step 18 bench.eval_zipa_baseline --steps eval,report --workers 2
step 19 bench.scoreboard
step 20 bench.third_party.lingolix_live --report-only --pronunciada-cache "$OV" \
    --pronunciada-tier2x "$OV/third_party/lingolix/responses/pronunciada/tier2x.jsonl" --cache-dir "$LIVE"
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ALL DONE" >> "$LOG/final.log"
