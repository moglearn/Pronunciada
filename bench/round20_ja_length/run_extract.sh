#!/bin/bash
# Run every shard of one pool, 4 at a time (inside python -m bench.heavy --lane cpu).
# usage: bench/round20_ja_length/run_extract.sh POOL NSHARDS PYTHON
POOL=$1; N=$2; PY=$3
seq 0 $((N - 1)) | xargs -P 4 -I{} env -u AZURE_SPEECH_KEY -u AZURE_SPEECH_REGION OMP_NUM_THREADS=2 \
  "$PY" -m bench.round20_ja_length.extract "$POOL" --shard {} --of "$N" --layers enc1,enc5
