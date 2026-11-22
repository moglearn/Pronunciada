"""Round 19 (German): score a manifest through the public API and keep per-word / per-sound detail.

    python -m bench.round19.collect --set dev [--tag base] [--limit N]
    python -m bench.round19.collect --set lingolix --tag base       # held-out: diagnosis / final only

Every item goes through Scorer("de").score(audio, target) with default settings, then
pronunciada.compat.azure.to_azure and the benchmark's Azure reduction (bench.third_party.lingolix),
exactly as the Lingolix runner does, so the "words" field is what the benchmark metric reads.
Beside it: per reference word the probability chain and the word-model inputs, per sound the
symbol, its alternatives, the chosen path, the shown probability and the evidence.

Behaviour switches are read from the environment (PRONUNCIADA_*), so a tag = one configuration.
Output: $PRONUNCIADA_CACHE/round19/german/dumps/<set>-<tag>.jsonl (resumable).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from bench.round19.devset import WORK


def manifest(name: str) -> list[dict]:
    if name == "dev":
        return [json.loads(l) for l in (WORK / "dev_manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if name == "lingolix":
        from bench.third_party import lingolix as L

        items = L.tests()["tier1_de"]
        for it in items:
            it["audio_path"] = str(L.tier1_wav(it))
        return items
    raise SystemExit(f"unknown set {name}")


def _f(x):
    x = float(x)
    return None if not math.isfinite(x) else round(x, 4)


def detail(sc, res) -> dict:
    c = res._pronunciada  # noqa: SLF001
    ref, ev, tab = c["ref"], c["ev"], c["tab"]
    W = c["W"]
    words = []
    for wi, (w, wr) in enumerate(zip(ref.words, res.words)):
        a, b = w.phones
        ph = []
        for i in range(a, b):
            p, sl = ref.phones[i], ev.slots[i]
            ph.append({"ipa": p.ipa, "alts": [[x, round(lp, 3)] for x, lp in p.alts], "alt": sl.alt_index,
                       "p": _f(res.words[wi].phones[i - a].prob), "zl": _f(c["s_zl"][i]), "zs": _f(c["s_zs"][i]),
                       "heard": sl.heard, "llr_sub": _f(sl.llr_sub), "llr_del": _f(sl.llr_del),
                       "nf": int(sl.n_frames), "peak": _f(sl.peak_post),
                       "stress": ref.syllables[p.syllable].stress, "nuc": bool(p.is_nucleus)})
        words.append({"text": w.text, "span": list(w.char_span), "prob": _f(wr.prob), "sounds": _f(wr.sounds_prob),
                      "present": _f(wr.present_prob) if wr.present_prob is not None else None,
                      "stress": _f(wr.stress_prob) if wr.stress_prob is not None else None,
                      "w_zl": _f(c["w_zl"][wi]), "w_zs": _f(c["w_zs"][wi]),
                      "W": {k: _f(W[k][wi]) for k in sc.word.num}, "phones": ph})
    return {"rwords": words, "sent_prob": _f(res.prob), "warnings": list(res.warnings)}


def run(set_name: str, tag: str, limit: int | None, threads: int) -> Path:
    from bench.third_party.lingolix import reduce_azure_shape
    from pronunciada import Scorer
    from pronunciada.compat.azure import to_azure

    out = WORK / "dumps" / f"{set_name}-{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["id"])
    items = manifest(set_name)
    if limit:
        items = items[:limit]
    ea = os.environ.get("R19_EXPLAIN_AWAY")          # bench-only: try the round-3 second pass
    sc = Scorer("de", threads=threads, explain_away=None if ea is None else ea == "1")
    t0 = time.time()
    n = 0
    with open(out, "a", encoding="utf-8") as f:
        for it in items:
            if it["id"] in done:
                continue
            wav = it.get("audio_path") or it["audio"]
            t = time.perf_counter()
            try:
                res = sc.score(wav, it["target"])
                lat = (time.perf_counter() - t) * 1000
                raw = to_azure(res, granularity="Phoneme", enable_miscue=True, phoneme_alphabet="IPA",
                               layout="rest", grading_system="HundredMark")
                vw = reduce_azure_shape(raw)
                rec = {"id": it["id"], "latency_ms": round(lat, 1), **detail(sc, res),
                       "words": [{"text": w["text"], "score": w["score"], "error_type": w["error_type"]}
                                 for w in vw]}
            except Exception as e:  # noqa: BLE001
                rec = {"id": it["id"], "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 100 == 0:
                f.flush()
                print(f"{n} items, {time.time() - t0:.0f}s", flush=True)
    print(f"{set_name}-{tag}: {n} new items in {time.time() - t0:.0f}s -> {out}", flush=True)
    return out


CORRUPT_KINDS = ("phone_sub_near", "phone_sub_far", "phone_ins", "phone_del", "word_swap")


def run_corrupt(tag: str, threads: int) -> Path:
    """Common Voice dev clips with the controlled sound errors of bench.corrupt (the protocol of the
    ten-language table: 2 errors per reference, seed = hash(item, kind)), one kind per clip (cycled),
    so the word recalibration also sees single-sound errors inside long words.
    Rows: {"id", "kind", "wrong": [per reference word], "rwords": ...}."""
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.common import stable_seed
    from bench.pipeline.evalctl import inventory
    from pronunciada import Scorer

    out = WORK / "dumps" / f"devcor-{tag}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    ea = os.environ.get("R19_EXPLAIN_AWAY")
    sc = Scorer("de", threads=threads, explain_away=None if ea is None else ea == "1")
    inv = inventory("de")
    items = [it for it in manifest("dev") if it["source"] == "cv" and not it["label"]]
    with open(out, "a", encoding="utf-8") as f:
        for j, it in enumerate(items):
            if it["id"] in done:
                continue
            ref = sc.g2p(it["target"])
            kind = CORRUPT_KINDS[j % len(CORRUPT_KINDS)]
            try:
                cor = corrupt(ref, [kind], inv, n_errors=2, seed=stable_seed(it["id"], kind, "r19"), strict=False)
            except NotApplicable:
                continue
            if not cor.errors:
                continue
            try:
                res = sc.score(it["audio"], cor.ref)
            except ValueError:
                continue
            rec = {"id": it["id"], "kind": kind, "wrong": [bool(x) for x in cor.word_wrong], **detail(sc, res)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["dev", "lingolix", "devcor"])
    ap.add_argument("--tag", default="base")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    if a.set == "devcor":
        run_corrupt(a.tag, a.threads)
    else:
        run(a.set, a.tag, a.limit, a.threads)


if __name__ == "__main__":
    sys.exit(main())
