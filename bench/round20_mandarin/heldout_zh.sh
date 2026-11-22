#!/bin/bash
# Round 20 (Mandarin): the one held-out run of the frozen candidate, three arms (off / r19 / new).
#   bash bench/round20_mandarin/heldout_zh.sh "<python>" <work dir>
# Run from the repository root, inside python -m bench.heavy --lane cpu. Azure variables removed.
set -eu
PY="$1"; WORK="$2"
mkdir -p "$WORK"
E=(env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION -u AZURE_SPEECH_ENDPOINT)
"${E[@]}" "$PY" -W ignore -m bench.round20_mandarin.heldout fleurs-collect --workers 3
"${E[@]}" "$PY" -W ignore -m bench.round20_mandarin.heldout ompal-collect --workers 3
"${E[@]}" "$PY" -W ignore -m bench.round20_mandarin.heldout cvreal-collect
bash bench/round20_mandarin/audit_arms.sh "$PY" zh test "$WORK/audit_zh" \
  "off=PRONUNCIADA_IDENTITY_ZH=0" \
  "r19=PRONUNCIADA_IDENTITY_ZH_R20=0" \
  "new=PRONUNCIADA_IDENTITY_ZH_R20=1"
echo "heldout zh collected"
