"""Score a list of clips with the runtime Scorer and keep per-word / per-sound evidence.

    python -m bench.round19_ja.dump IN.jsonl OUT.jsonl [--tag NAME]

IN rows: {"id", "audio" (path, or "parquet:<path>:<row>"), "text", ...}; every other field is
copied through. OUT is appended to and resumable (ids already present are skipped). One process,
2 threads (round-19 memory rules). Environment switches (PRONUNCIADA_*) apply as usual.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _wav(audio: str):
    from pronunciada.audio import load_audio

    if audio.startswith("parquet:"):
        _, path, row = audio.rsplit(":", 2)
        from bench.pipeline.common import iter_parquet_audio

        for _, w in iter_parquet_audio(path, {int(row)}):
            return w
        raise FileNotFoundError(audio)
    return load_audio(audio)


def record(res) -> dict:
    ref = res._pronunciada["ref"]  # noqa: SLF001
    # the system's own word flag, as the Lingolix runner reads it (Azure layout ErrorType)
    from bench.third_party.lingolix import reduce_azure_shape
    from pronunciada.compat.azure import to_azure

    raw = to_azure(res, granularity="Phoneme", enable_miscue=True, phoneme_alphabet="IPA", layout="rest",
                   grading_system="HundredMark")
    flags = [w["error_type"] not in (None, "None") for w in reduce_azure_shape(raw) if w["error_type"] != "Insertion"]
    words = []
    for w, rw in zip(res.words, ref.words):
        words.append({
            "text": w.text, "span": list(rw.char_span), "prob": w.prob, "sounds": w.sounds_prob,
            "present": w.present_prob, "len": w.length_prob, "con": w.contrast_prob,
            "start": w.start, "end": w.end, "flag": flags[len(words)] if len(flags) == len(res.words) else None,
            "ph": [[p.ipa, round(p.prob, 4), p.length_expected,
                    None if p.length_prob is None else round(p.length_prob, 5), p.heard,
                    round(p.start, 3), round(p.end, 3)] for p in w.phones]})
    return {"prob": res.prob, "scores": res.scores, "words": words, "warnings": res.warnings,
            "ms": res.timing_ms.get("total")}


def run(inp: Path, out: Path, lang: str = "ja", threads: int = 2, limit: int | None = None) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    import warnings

    warnings.simplefilter("ignore")
    from pronunciada.scorer import Scorer

    rows = [json.loads(x) for x in inp.read_text(encoding="utf-8").splitlines() if x.strip()]
    done = set()
    if out.exists():
        done = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines() if x.strip()}
    todo = [r for r in rows if r["id"] not in done][:limit]
    sc = Scorer(lang, threads=threads)
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for k, r in enumerate(todo):
            try:
                res = sc.score((_wav(r["audio"]), 16000), r["text"])
                rec = {**r, **record(res)}
            except Exception as e:  # noqa: BLE001
                rec = {**r, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 50 == 49:
                f.flush()
                print(f"{k + 1}/{len(todo)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    print(f"done {len(todo)} in {time.time() - t0:.0f}s -> {out}", file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    run(a.inp, a.out, a.lang, a.threads, a.limit)


if __name__ == "__main__":
    main()
