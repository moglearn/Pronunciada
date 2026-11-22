"""Round 19 (Korean): a system-independent read-the-prompt check for the Common Voice dev clips.

    python -m bench.heavy --need 30 --floor 12 -- python -m bench.round19ko.asr_check

Common Voice "other" clips are unreviewed: some speakers read another sentence or recorded noise.
The Lingolix benchmark keeps only clips whose Whisper transcript is within a 10 % character error
rate of the prompt. Same rule here, with the local openai/whisper-large-v3 (language ko, greedy),
so the dev set has the same kind of audio. Pronunciada's own scores play no part in the filter.
Output (resumable): $CACHE/round19/korean/asr.jsonl {id, transcript, cer}.
"""
from __future__ import annotations

import json
import re

import numpy as np

from bench.round19ko.cv_pool import work

_P = re.compile(r"[^\w]", re.UNICODE)


def cer(ref: str, hyp: str) -> float:
    a, b = _P.sub("", ref), _P.sub("", hyp)
    if not a:
        return 1.0
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1] / len(a)


def main():
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.set_num_threads(2)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = torch.float16 if dev == "mps" else torch.float32
    from pathlib import Path

    snaps = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--whisper-large-v3" / "snapshots"
    name = str(next(snaps.iterdir()))           # the local snapshot (no network)
    proc = WhisperProcessor.from_pretrained(name, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=dt, local_files_only=True).to(dev).eval()
    items = [json.loads(x) for x in (work() / "dev_items.jsonl").read_text(encoding="utf-8").splitlines()]
    clips = {}
    for it in items:
        if it["src"] == "cv":
            clips[it["id"]] = (it["audio"], it["prompt"])
    out_p = work() / "asr.jsonl"
    done = set()
    if out_p.exists():
        done = {json.loads(x)["id"] for x in out_p.read_text(encoding="utf-8").splitlines() if x.strip()}
    with open(out_p, "a", encoding="utf-8") as f:
        for n, (cid, (path, prompt)) in enumerate(sorted(clips.items())):
            if cid in done:
                continue
            x, sr = sf.read(path, dtype="float32")
            feats = proc(x, sampling_rate=16000, return_tensors="pt").input_features.to(dev, dt)
            with torch.no_grad():
                ids = model.generate(feats, language="ko", task="transcribe", num_beams=1, max_new_tokens=200)
            hyp = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            f.write(json.dumps({"id": cid, "transcript": hyp, "cer": round(cer(prompt, hyp), 4)}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 50 == 0:
                print(n, len(clips), flush=True)
    rows = [json.loads(x) for x in out_p.read_text(encoding="utf-8").splitlines()]
    c = np.array([r["cer"] for r in rows])
    print("clips", len(c), "cer<=0.1", int((c <= 0.1).sum()))


if __name__ == "__main__":
    main()
