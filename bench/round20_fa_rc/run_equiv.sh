#!/bin/bash
# Round 20 false-alarm release check, step 1: identity proofs, tolerance 0, every output of
# bench/round19_merge/equiv.py (to_dict, rescoring at strictness 0 / 1, details, coaching, Azure layout).
#   (a) release candidate with every round-20 switch set to 0  ==  main
#   (b) release candidate with its defaults  ==  the experimental branch with the four kept switches set
#       to 1 and the unsaid-word / guessed-word switches unset (off)
#   (c) it, pt, ru, ja, zh: release candidate with its defaults  ==  main
# Sets: FLEURS validation (100 clips x 2 texts) and 100 practice items per language
# (bench/round20_fa_rc/practice_items.py); for English also 100 speechocean762 train clips. fp32 and int8.
# main's FLEURS-validation and speechocean762-train dumps are reused from the round-20 release check when
# given (main's package code has not changed since); main is dumped here for the practice items.
#   bash bench/round20_fa_rc/run_equiv.sh "<python>" <rc tree> <experimental tree> <main tree> <items json> \
#        <earlier main dump dir> <out dir>
set -u
PY="$1"; RC="$2"; NFA="$3"; MAIN="$4"; ITEMS="$5"; OLDMAIN="$6"; OUT="$7"
EQ="$RC/bench/round19_merge/equiv.py"
mkdir -p "$OUT"
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2
CLEAN=(env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION -u AZURE_SPEECH_ENDPOINT -u LINGOLIX_API_KEY
       -u PRONUNCIADA_R20_UNSAID -u PRONUNCIADA_R20_UNSAID_TAU -u PRONUNCIADA_R20_GUESS -u PRONUNCIADA_R20_GUESS_S
       -u PRONUNCIADA_R20_NUMALT -u PRONUNCIADA_R20_DE_LEXALT -u PRONUNCIADA_R20_ES_GLIDE -u PRONUNCIADA_R20_KO_NINS)
OFF=(PRONUNCIADA_R20_NUMALT=0 PRONUNCIADA_R20_DE_LEXALT=0 PRONUNCIADA_R20_ES_GLIDE=0 PRONUNCIADA_R20_KO_NINS=0)
ON=(PRONUNCIADA_R20_NUMALT=1 PRONUNCIADA_R20_DE_LEXALT=1 PRONUNCIADA_R20_ES_GLIDE=1 PRONUNCIADA_R20_KO_NINS=1)

setname() { [ "$1" = practice ] && echo "items:$ITEMS" || echo "$1"; }

run() {  # name tree lang set precision [VAR=value ...]
  local name=$1 tree=$2 lang=$3 set=$4 prec=$5; shift 5
  local f="$OUT/${name}_${lang}_${set}_${prec}.jsonl"
  [ -s "$f.done" ] && return
  "${CLEAN[@]}" "$@" PYTHONPATH="$tree" PYTHONHASHSEED=0 \
      "$PY" "$EQ" dump --lang "$lang" --set "$(setname "$set")" --precision "$prec" --out "$f" \
      > "$f.done.tmp" 2>"$f.err" && mv "$f.done.tmp" "$f.done"
}

mainfile() {  # lang set precision
  local old="$OLDMAIN/main_$1_$2_$3.jsonl"
  if [ "$2" != practice ] && [ -s "$old" ]; then echo "$old"; else echo "$OUT/main_$1_$2_$3.jsonl"; fi
}

cmp0() {  # label a b
  "$PY" "$EQ" compare "$2" "$3" --tol 0 > "$OUT/cmp_$1.json" && echo OK || echo DIFFERENT
}

for prec in fp32 int8; do
  for lang in en de es fr ko; do
    sets="fleurs-val practice"; [ $lang = en ] && sets="$sets so762-train"
    for set in $sets; do
      echo "$(date +%H:%M:%S) $lang $set $prec load $(sysctl -n vm.loadavg)"
      run rcoff "$RC" $lang $set $prec "${OFF[@]}" &
      run rcdef "$RC" $lang $set $prec &
      run nfaon "$NFA" $lang $set $prec "${ON[@]}" &
      if [ "$(mainfile $lang $set $prec)" = "$OUT/main_${lang}_${set}_${prec}.jsonl" ]; then
        run main "$MAIN" $lang $set $prec &
      fi
      wait
      echo "$(date +%H:%M:%S) (a) off=main $lang $set $prec: $(cmp0 a_${lang}_${set}_${prec} "$OUT/rcoff_${lang}_${set}_${prec}.jsonl" "$(mainfile $lang $set $prec)")"
      echo "$(date +%H:%M:%S) (b) default=experimental $lang $set $prec: $(cmp0 b_${lang}_${set}_${prec} "$OUT/rcdef_${lang}_${set}_${prec}.jsonl" "$OUT/nfaon_${lang}_${set}_${prec}.jsonl")"
    done
  done
  for lang in it pt ru ja zh; do
    run rcdef "$RC" $lang fleurs-val $prec
    echo "$(date +%H:%M:%S) (c) default=main $lang fleurs-val $prec: $(cmp0 c_${lang}_fleurs-val_${prec} "$OUT/rcdef_${lang}_fleurs-val_${prec}.jsonl" "$(mainfile $lang fleurs-val $prec)")"
  done
done
echo "$(date +%H:%M:%S) all done"
