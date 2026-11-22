#!/bin/bash
# Round 20 (latency): paired, interleaved latency (bench/round19_merge/latency.py; 4 threads, one
# scorer running at a time) for Korean and Japanese, three pairs of source trees:
#   pre19 -> main   (round 19 as merged; c9773cc -> 8c74789)
#   main  -> new    (this branch against main)
#   pre19 -> new    (round 19 with this branch's speed-ups)
# on the ~12 s FLEURS validation clips and on short clips (ko: FLEURS validation cut at a word
# boundary near 3.5 s, bench/round20_latency/make_cut.py; ja: cv-real dev-half clips of 2.5-5 s),
# fp32 and int8, 100 clips each. Source trees: $SRC/{pre19,main,new}. Resumable. Run through the lock:
#   python -m bench.heavy --lane cpu --need 25 --floor 12 -- bash <this> "<python>" <src dir> <out dir>
set -u
PY="$1"; SRC="$2"; OUT="$3"
LAT="$SRC/new/bench/round19_merge/latency.py"
mkdir -p "$OUT"
for prec in fp32 int8; do
  for spec in ko:fleurs-val ja:fleurs-val ko:fleurs-val-cut ja:cv-dev-short ko:cv-dev-short; do
    lang=${spec%%:*}; set=${spec#*:}
    for pair in pre19:main main:new pre19:new; do
      old=${pair%%:*}; new=${pair#*:}
      f="$OUT/${lang}_${set}_${prec}_${old}_${new}.json"
      [ -s "$f" ] && continue
      echo "=== $(date +%T) $lang $set $prec $old -> $new load $(sysctl -n vm.loadavg)"
      env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION "$PY" "$LAT" --old "$SRC/$old" --new "$SRC/$new" \
          --lang $lang --set $set --precision $prec --threads 4 --n 100 --out "$f" 2>&1 | grep -v Warning | tail -1
    done
  done
done
echo "=== $(date +%T) ALL DONE"
