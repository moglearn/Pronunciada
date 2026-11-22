#!/bin/bash
# Fetch the LibriVox 64 kb/s mp3 chapters used by the Kokoro Speech Dataset v1.3 metadata from
# archive.org (public domain recordings; no account needed). Resumable. The list
# raw/kokoro_meta/mp3_list.tsv (book, archive identifier, file) is written by download.py kokoro-list.
W="${PRONUNCIADA_CACHE:-$HOME/.cache/pronunciada}/round20/jacorpus/raw"
while IFS=$'\t' read -r book ident f; do
  mkdir -p "$W/kokoro_mp3/$book"
  out="$W/kokoro_mp3/$book/$f"
  [ -s "$out" ] && continue
  curl -sfL --retry 3 -o "$out.part" "https://archive.org/download/$ident/$f" && mv "$out.part" "$out" || echo "FAIL $book $ident $f"
done < "$W/kokoro_meta/mp3_list.tsv"
echo DONE
