#!/bin/bash
# Round 20 false-alarm release check: paired latency, main (old) against the release candidate (new),
# bench/round19_merge/latency.py (interleaved clip by clip, 4 threads, warm scorers).
#   * English: the same 200 speechocean762 TRAIN clips, fp32 and int8;
#   * en, de, es, fr, ko: the 100 practice items of bench/round20_fa_rc/practice_items.py (half of them with a
#     number, where the number-reading pass re-scores), fp32 and int8.
#   bash bench/round20_fa_rc/latency.sh "<python>" <main tree> <rc tree> <items json> <out dir>
set -u
PY="$1"; OLD="$2"; NEW="$3"; ITEMS="$4"; OUT="$5"
HERE="$(cd "$(dirname "$0")/.." && pwd)/round19_merge"
mkdir -p "$OUT"
CLEAN=(env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION -u AZURE_SPEECH_ENDPOINT -u LINGOLIX_API_KEY
       -u PRONUNCIADA_R20_NUMALT -u PRONUNCIADA_R20_DE_LEXALT -u PRONUNCIADA_R20_ES_GLIDE -u PRONUNCIADA_R20_KO_NINS
       -u PRONUNCIADA_R20_UNSAID -u PRONUNCIADA_R20_GUESS)
for prec in fp32 int8; do
  f="$OUT/latency_en_so762-train_$prec.json"
  [ -s "$f" ] || "${CLEAN[@]}" "$PY" "$HERE/latency.py" --old "$OLD" --new "$NEW" --lang en --set so762-train \
      --precision $prec --threads 4 --n 200 --out "$f"
  for lang in en de es fr ko; do
    f="$OUT/latency_${lang}_practice_$prec.json"
    [ -s "$f" ] || "${CLEAN[@]}" "$PY" "$HERE/latency.py" --old "$OLD" --new "$NEW" --lang $lang --set "items:$ITEMS" \
        --precision $prec --threads 4 --n 100 --out "$f"
  done
done
echo done
