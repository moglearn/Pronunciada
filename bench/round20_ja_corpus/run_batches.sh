#!/bin/bash
# Align the corpus in resumable batches with the trained model and the variant dictionary:
#   prep (temporary 16 kHz WAVs) -> MFA align under the cpu lane of bench.heavy -> parse -> delete WAVs.
# A batch whose aligned/<name>.jsonl exists is skipped. Run from the repository root.
set -uo pipefail
PY="${PY:-python}"
W="${PRONUNCIADA_CACHE:-$HOME/.cache/pronunciada}/round20/jacorpus"
run() {   # name where
  local name=$1 where=$2
  [ -s "$W/aligned/$name.jsonl" ] && { echo "skip $name"; return; }
  echo "[$(date +%H:%M)] prep $name"
  "$PY" -m bench.round20_ja_corpus.prep batch "$name" --where "$where" --dict variants || return
  echo "[$(date +%H:%M)] align $name"
  "$PY" -m bench.heavy --lane cpu --need 25 --floor 12 --max-restarts 3 -- bash bench/round20_ja_corpus/mfa.sh align "$name" \
    > "$W/logs/mfa_align_$name.log" 2>&1 || { echo "align failed $name"; return; }
  "$PY" -m bench.round20_ja_corpus.build parse "$name" && rm -rf "$W/align/$name/corpus" "$W/align/$name/out"
  echo "[$(date +%H:%M)] done $name"
}
run valid "r['tier']=='validated' and r['source']!='kokoro'"
for k in 0 1; do
  run kokoro_$k "r['source']=='kokoro' and r['reading_check']['agree'] and int(r['id'].rsplit('-',1)[-1])%2==$k"
done
for k in 0 1 2 3 4 5 6 7; do
  run other_$k "r['tier']=='other' and int(r['id'].rsplit('-',1)[-1])%8==$k"
done
echo ALLDONE
