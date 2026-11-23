#!/bin/bash
# Round 20 false-alarm release check, latency budget: added latency of the release defaults against every
# round-20 switch set to 0 (= main), same tree, on natural text: 200 original FLEURS validation clips per
# language (own text; bench/round19_merge/latency.py --set fleurs-val), paired interleaved, int8 and fp32,
# 4 threads and 1 thread. Then how often the pass re-scores (bench/round20_fa_rc/triggers.py).
# Budget rule: a language passes if its MEAN added latency is <= 2 ms at 4 threads, int8 and fp32.
#   bash bench/round20_fa_rc/latency_natural.sh "<python>" <rc tree> <out dir>
set -u
PY="$1"; RC="$2"; OUT="$3"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT"
CLEAN=(env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION -u AZURE_SPEECH_ENDPOINT -u LINGOLIX_API_KEY
       -u PRONUNCIADA_R20_NUMALT -u PRONUNCIADA_R20_DE_LEXALT -u PRONUNCIADA_R20_ES_GLIDE -u PRONUNCIADA_R20_KO_NINS
       -u PRONUNCIADA_R20_UNSAID -u PRONUNCIADA_R20_GUESS)
OFF="PRONUNCIADA_R20_NUMALT=0,PRONUNCIADA_R20_DE_LEXALT=0,PRONUNCIADA_R20_ES_GLIDE=0,PRONUNCIADA_R20_KO_NINS=0"
for th in 4 1; do
  for prec in int8 fp32; do
    for lang in en de es fr ko; do
      f="$OUT/natural_${lang}_${prec}_t${th}.json"
      [ -s "$f" ] || "${CLEAN[@]}" "$PY" "$HERE/round19_merge/latency.py" --old "$RC" --old-env "$OFF" --new "$RC" \
          --lang $lang --set fleurs-val --precision $prec --threads $th --n 200 --out "$f"
    done
  done
done
for prec in int8 fp32; do
  f="$OUT/triggers_$prec.json"
  [ -s "$f" ] || "${CLEAN[@]}" PYTHONPATH="$(dirname "$HERE")" "$PY" -m bench.round20_fa_rc.triggers "$f" --precision $prec
done
echo done
