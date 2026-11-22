#!/bin/bash
# Round 20: contrast audit of one language under three arms, each in its own cache overlay (empty audit
# output folders, the existing edit plans copied), three processes side by side, then the per-arm report
# and the paired before / after rows (bench/round19_merge/audit_paired.py).
#   bash bench/round20_mandarin/audit_arms.sh "<python>" <lang> <split> <work dir> "<arm>=<env ...>" ...
# Run from the repository root; Azure variables are removed from every job.
set -u
PY="$1"; LANG_="$2"; SPLIT="$3"; WORK="$4"; shift 4
mkdir -p "$WORK"
pids=()
for spec in "$@"; do
  arm="${spec%%=*}"; envs="${spec#*=}"
  ov="$WORK/ov_$arm"
  [ -d "$ov" ] || "$PY" -m bench.round20_mandarin.overlay "$ov" > /dev/null
  env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION -u AZURE_SPEECH_ENDPOINT PRONUNCIADA_CACHE="$ov" $envs \
    "$PY" -W ignore -m bench.audit.contrast_audit run --split "$SPLIT" --lang "$LANG_" --workers 1 \
    > "$WORK/run_$arm.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc -ne 0 ] && { echo "an arm failed"; exit 1; }
for spec in "$@"; do
  arm="${spec%%=*}"
  env PRONUNCIADA_CACHE="$WORK/ov_$arm" "$PY" -W ignore -m bench.audit.contrast_audit report --split "$SPLIT" \
    --lang "$LANG_" --out "$WORK/report_$arm.json" > "$WORK/report_$arm.log" 2>&1
done
echo "done"
