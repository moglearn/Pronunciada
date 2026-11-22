#!/bin/bash
# Montreal Forced Aligner (MIT) from a private conda-forge env under the round-20 work folder.
#   mfa.sh train NAME     train an acoustic model on align/NAME (canonical dictionary), export alignments
#   mfa.sh align NAME     align align/NAME with the trained model and its (variant) dictionary
# Run through: python -m bench.heavy --lane cpu --need 25 --floor 12 -- bash bench/round20_ja_corpus/mfa.sh ...
set -euo pipefail
W="${PRONUNCIADA_CACHE:-$HOME/.cache/pronunciada}/round20/jacorpus"
MFA="$W/tools/mfa/bin/mfa"
export MFA_ROOT_DIR="$W/mfa_root" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$W/tools/mfa/bin:$PATH"
unset AZURE_SPEECH_KEY AZURE_SPEECH_REGION
cmd=$1; name=$2; model="${MODEL:-$W/align/model_ja_r20.zip}"
B="$W/align/$name"
case "$cmd" in
  train)
    "$MFA" train "$B/corpus" "$B/dict.txt" "$model" --output_directory "$B/out" --output_format json \
      --num_jobs 4 --clean --final_clean --quiet ;;
  align)
    "$MFA" align "$B/corpus" "$B/dict.txt" "$model" "$B/out" --output_format json \
      --num_jobs 4 --clean --final_clean --quiet --beam 100 --retry_beam 400 ;;
esac
