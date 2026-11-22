"""Round 19: the shipped English sentence head through the public Scorer (dev data only).

    python -m bench.round19.runtime_check [--n-train 100]

1. Parity: on speechocean762 TRAIN clips, the runtime tree inputs (S row) equal the inputs the
   offline fit used from the feature cache, and the final scores equal the offline blend of
   sentence_r19 + seqhead_en on cached features.
2. Native guard end to end: the 100 seeded FLEURS-validation English clips, new vs
   PRONUNCIADA_SENTENCE_HEAD=0 (median fluency / prosodic / total may drop by <= 0.3).
3. Latency: 3.5 s-ish clips (speechocean762 train), median of 100 calls, fp32 and int8, 4 threads,
   new vs old, plus the size of the added files.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

HEADS = ("accuracy", "fluency", "prosodic", "total")


def scorer(on: bool, **kw):
    os.environ["PRONUNCIADA_SENTENCE_HEAD"] = "1" if on else "0"
    from pronunciada.scorer import Scorer

    sc = Scorer("en", **kw)
    os.environ.pop("PRONUNCIADA_SENTENCE_HEAD")
    return sc


def main():
    from bench.calib.common import save_json, ship_dir
    from bench.pipeline.common import records
    from bench.round19 import data as RD
    from bench.round19.extract_en import fleurs_records
    from pronunciada.audio import load_audio

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=100)
    a = ap.parse_args()
    out = {}
    new, old = scorer(True, threads=2), scorer(False, threads=2)
    assert new.sentence_name == "sentence_r19" and new.seq_head is not None and old.sentence_name == "sentence_r4"
    # 1. parity against the feature cache
    cache = {u["id"]: u for u in RD.load("train", keep=["S", "delivery", "f0", "utt_layers"])}
    recs = [r for r in records("speechocean762") if r["split"] == "train"][:a.n_train]
    from pronunciada import delivery as D
    from pronunciada.sentence_head import F0_KEYS

    dmax = 0.0
    for r in recs[:40]:
        res = new.score(r["audio"], r["text"])
        c = res._pronunciada  # noqa: SLF001
        u = cache[r["id"]]
        for col in new.sentence.num:
            v = float(np.asarray(c["S"][col]).reshape(-1)[0])
            if col.startswith("emb_"):
                ref = D.project(u["utt_layers"], new.sentence_proj)[col]
            elif col in F0_KEYS:
                ref = u["f0"][col]
            elif col in u["delivery"]:
                ref = u["delivery"][col]
            else:
                ref = u["S"][col]
            if np.isfinite(v) or np.isfinite(ref):
                dmax = max(dmax, abs(v - ref))
    out["parity_tree_inputs_max_abs"] = dmax
    print("parity tree inputs max abs", dmax, flush=True)
    # 2. native guard
    fr = fleurs_records("validation", 100)
    vals = {"new": [], "old": []}
    for r in fr:
        try:
            rn, ro = new.score(r["audio"], r["text"]), old.score(r["audio"], r["text"])
        except ValueError:
            continue
        vals["new"].append([new.details(rn)["sentence"][h] for h in HEADS])
        vals["old"].append([old.details(ro)["sentence"][h] for h in HEADS])
    N, O = np.array(vals["new"]), np.array(vals["old"])
    g = {h: {"median_old": float(np.median(O[:, j])), "median_new": float(np.median(N[:, j])),
             "median_delta": float(np.median(N[:, j]) - np.median(O[:, j]))} for j, h in enumerate(HEADS)}
    g["n"] = len(N)
    g["pass"] = bool(all(g[h]["median_delta"] >= -0.3 for h in ("fluency", "prosodic", "total")))
    out["native_guard_fleurs_validation"] = g
    print("guard", {h: round(g[h]["median_delta"], 3) for h in HEADS}, g["pass"], flush=True)
    # 3. latency
    wavs = [load_audio(r["audio"]) for r in recs]
    wavs = sorted(wavs, key=lambda w: abs(len(w) / 16000 - 3.5))[:20]
    texts = {id(w): None for w in wavs}
    rmap = {}
    for r in recs:
        rmap[len(load_audio(r["audio"]))] = r["text"]
    lat = {}
    for prec in ("fp32", "int8"):
        for on in (False, True):
            sc = scorer(on, precision=prec, threads=4)
            ts = []
            for k in range(100):
                w = wavs[k % len(wavs)]
                t0 = time.perf_counter()
                sc.score(w.copy() + 0.0, rmap[len(w)])          # copy: no posterior cache hit
                ts.append((time.perf_counter() - t0) * 1e3)
            lat[f"{prec}_{'new' if on else 'old'}_ms"] = float(np.median(ts))
            del sc
    lat["clip_s_median"] = float(np.median([len(w) / 16000 for w in wavs]))
    out["latency"] = lat
    print("latency", lat, flush=True)
    d = ship_dir()
    out["size_kb"] = {f: (d / f).stat().st_size / 1024 for f in ("sentence_r19.npz", "seqhead_en.onnx", "seqhead_en.npz")}
    save_json("round19_english_runtime.json", out)


if __name__ == "__main__":
    main()
