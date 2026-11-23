#!/bin/bash
# Round 20 false-alarm release check, step 4: identity of the final per-language table, tolerance 0, every
# output of bench/round19_merge/equiv.py, fp32 and int8:
#   en, de, fr, ko, it, pt, ru, ja, zh: final defaults == main;
#   es: final defaults == the frozen release candidate's defaults (whose held-out outputs are reused).
# Reuses the dumps of run_equiv.sh (main, rcdef) and of the round-20 release check (main FLEURS validation,
# speechocean762 train).
#   bash bench/round20_fa_rc/run_equiv_final.sh "<python>" <final tree> <items json> <earlier main dump dir> <dump dir>
set -u
PY="$1"; FIN="$2"; ITEMS="$3"; OLDMAIN="$4"; OUT="$5"
EQ="$FIN/bench/round19_merge/equiv.py"
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2
CLEAN=(env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION -u AZURE_SPEECH_ENDPOINT -u LINGOLIX_API_KEY
       -u PRONUNCIADA_R20_UNSAID -u PRONUNCIADA_R20_GUESS -u PRONUNCIADA_R20_NUMALT -u PRONUNCIADA_R20_DE_LEXALT
       -u PRONUNCIADA_R20_ES_GLIDE -u PRONUNCIADA_R20_KO_NINS)
setname() { [ "$1" = practice ] && echo "items:$ITEMS" || echo "$1"; }
mainfile() { local old="$OLDMAIN/main_$1_$2_$3.jsonl"
  if [ "$2" != practice ] && [ -s "$old" ]; then echo "$old"; else echo "$OUT/main_$1_$2_$3.jsonl"; fi; }
for prec in fp32 int8; do
  for lang in en de fr ko es it pt ru ja zh; do
    sets="fleurs-val"; case $lang in en) sets="fleurs-val practice so762-train";; de|fr|ko|es) sets="fleurs-val practice";; esac
    for set in $sets; do
      f="$OUT/final_${lang}_${set}_${prec}.jsonl"
      [ -s "$f.done" ] || { "${CLEAN[@]}" PYTHONPATH="$FIN" PYTHONHASHSEED=0 "$PY" "$EQ" dump --lang $lang \
          --set "$(setname $set)" --precision $prec --out "$f" > "$f.done.tmp" 2>"$f.err" && mv "$f.done.tmp" "$f.done"; }
      if [ $lang = es ]; then ref="$OUT/rcdef_${lang}_${set}_${prec}.jsonl"; lab="rc"; else ref="$(mainfile $lang $set $prec)"; lab="main"; fi
      "$PY" "$EQ" compare "$f" "$ref" --tol 0 > "$OUT/cmpfinal_${lang}_${set}_${prec}.json" && r=OK || r=DIFFERENT
      echo "$(date +%H:%M:%S) final=$lab $lang $set $prec: $r"
    done
  done
done
echo "all done"
