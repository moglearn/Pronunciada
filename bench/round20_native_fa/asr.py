"""Independent read-the-prompt check for every practice clip (bench/round19ko/asr_check.py, all languages).

    python -m bench.heavy --need 30 --floor 12 -- python -m bench.round20_native_fa.asr --lang ko,en,de,es,fr

The local openai/whisper-large-v3 (MIT), greedy, language fixed, no prompt: the same decoding as the
round-19 Korean filter, so the Korean CER <= 0.1 rule is reproduced. Token times come from the
model's cross-attention alignment of the SAME generated tokens (return_token_timestamps; decoding is
unchanged); tokens are grouped into words at spaces. Pronunciada plays no part.
Output (resumable): $CACHE/round20/nativefa/asr/<lang>.jsonl {clip, transcript, cer, words: [[w, start, end]]};
for Korean also asr.jsonl in the practice-set folder ({id, transcript, cer}, the round-19 format).
"""
from __future__ import annotations

import argparse
import json

from bench.round19ko.asr_check import cer
from bench.round20_native_fa.rebuild import ROOT

WLANG = {"en": "en", "de": "de", "es": "es", "fr": "fr", "ko": "ko"}


def clips(lang: str) -> list[tuple[str, str, str]]:
    from bench.round20_native_fa.collect import items

    seen = {}
    for it in items(lang):
        seen.setdefault(it["clip"], (it["audio"], it["prompt"]))
    return [(c, a, p) for c, (a, p) in sorted(seen.items())]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    a = ap.parse_args(argv)
    from pathlib import Path

    import numpy as np
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.set_num_threads(2)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = torch.float16 if dev == "mps" else torch.float32
    snaps = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--whisper-large-v3" / "snapshots"
    name = str(next(snaps.iterdir()))
    proc = WhisperProcessor.from_pretrained(name, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=dt, local_files_only=True).to(dev).eval()
    tok = proc.tokenizer
    out_dir = ROOT / "asr"
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang in a.lang.split(","):
        out_p = out_dir / f"{lang}.jsonl"
        done = set()
        if out_p.exists():
            done = {json.loads(x)["clip"] for x in out_p.read_text(encoding="utf-8").splitlines() if x.strip()}
        cl = clips(lang)
        with open(out_p, "a", encoding="utf-8") as f:
            for n, (cid, path, prompt) in enumerate(cl):
                if cid in done:
                    continue
                x, sr = sf.read(path, dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                feats = proc(x, sampling_rate=16000, return_tensors="pt").input_features.to(dev, dt)
                words = []
                with torch.no_grad():
                    try:
                        g = model.generate(feats, language=WLANG[lang], task="transcribe", num_beams=1,
                                           max_new_tokens=200, return_token_timestamps=True,
                                           return_dict_in_generate=True)
                        ids, ts = g["sequences"][0].tolist(), g["token_timestamps"][0].float().cpu().numpy()
                    except Exception:  # noqa: BLE001  (no alignment heads etc.: transcript only)
                        ids = model.generate(feats, language=WLANG[lang], task="transcribe", num_beams=1,
                                             max_new_tokens=200)[0].tolist()
                        ts = None
                hyp = tok.decode(ids, skip_special_tokens=True).strip()
                if ts is not None:
                    cur, cs, ce, prev = "", None, None, None
                    special = set(tok.all_special_ids)
                    for j, t in enumerate(ids):
                        if t in special or t >= tok.convert_tokens_to_ids("<|endoftext|>"):
                            continue
                        piece = tok.decode([t])
                        st = float(ts[j])                      # token start (HF jump times)
                        en = float(ts[j + 1]) if j + 1 < len(ts) else st
                        if piece.startswith(" ") and cur.strip():
                            words.append([cur.strip(), round(cs, 3), round(ce, 3)])
                            cur, cs = "", None
                        cur += piece
                        cs = st if cs is None else cs
                        ce = en
                    if cur.strip():
                        words.append([cur.strip(), round(cs, 3), round(ce, 3)])
                rec = {"clip": cid, "transcript": hyp, "cer": round(cer(prompt, hyp), 4), "words": words}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if n % 50 == 0:
                    print(lang, n, len(cl), flush=True)
        if lang == "ko":                         # the round-19 filter file ({id, transcript, cer}, CV clips)
            rows = [json.loads(x) for x in out_p.read_text(encoding="utf-8").splitlines() if x.strip()]
            with open(ROOT / "korean" / "asr.jsonl", "w", encoding="utf-8") as g2:
                for r in rows:
                    if r["clip"].startswith("cv-"):
                        g2.write(json.dumps({"id": r["clip"], "transcript": r["transcript"], "cer": r["cer"]},
                                            ensure_ascii=False) + "\n")
        c = np.array([json.loads(x)["cer"] for x in out_p.read_text(encoding="utf-8").splitlines() if x.strip()])
        print(lang, "clips", len(c), "cer<=0.1", int((c <= 0.1).sum()), flush=True)


if __name__ == "__main__":
    main()
