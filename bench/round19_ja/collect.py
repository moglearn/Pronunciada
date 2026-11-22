"""Collect length-channel inputs and word-probability pieces, so the length model and the word
factor can be refitted / re-selected offline.

    python -m bench.round19_ja.collect fit      # natives: FLEURS ja train + CV fit pool (texts as said)
    python -m bench.round19_ja.collect dev      # the round-19 dev set (set_dev.jsonl: edited + controls)
    python -m bench.round19_ja.collect devnat   # dev pool clips with the text as said (native guard)
    python -m bench.round19_ja.collect fitset   # the planted fit set (set_fit.jsonl), for word-level fits

Per scored item: the Reference, pronunciada.length.frame_inputs, the round-17 head evidence s per
sound, and per word the shipped prob and its length factor (so prob / factor is the word
probability without the length channel). The text front-end follows the current switches
(PRONUNCIADA_JA_ROUND19*), and the output folder is named after them.
Pickles in chunks of 50 under $PRONUNCIADA_CACHE/round19/japanese/collect/<tag>/<set>/.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from bench.round19_ja.cv_pool import WORK

CHUNK = 50
WCOLS = ("wl_mean", "wl_min", "ws_mean", "ws_min", "wl_min2", "ws_min2", "n_phones", "word_mean_llr_sub",
         "word_min_llr_sub", "word_min_llr_del", "word_share_del", "word_frames", "word_sum_llr_sub",
         "word_pos_first", "word_pos_last", "utt_ws_mean", "utt_wl_min")


def tag() -> str:
    from pronunciada.ja_round19 import on

    return "merge" if on("merge") else "nomerge"


def rows_for(which: str) -> list[dict]:
    from bench.round19_ja.plant import pool_rows

    if which == "fit":
        return pool_rows("fit")
    if which == "devnat":
        return pool_rows("dev")
    name = {"dev": "set_dev.jsonl", "fitset": "set_fit.jsonl"}[which]
    return [json.loads(x) for x in open(WORK / name, encoding="utf-8")]


def item(sc, res) -> dict:
    from pronunciada import contrast_heads as CH
    from pronunciada import length as L
    from pronunciada.sound_emb import phone_vectors

    c = res._pronunciada  # noqa: SLF001
    ref, ev, lpa, emb = c["ref"], c["ev"], c["lpa"], c["emb"]
    fb = sc._posteriors(c["wav"], {})[1]  # noqa: SLF001
    hs = None
    if emb and sc.length_cal and sc.length_cal.get("head"):
        hs = CH.length_evidence("ja", sc.length_cal, [p.ipa for p in ref.phones], phone_vectors(emb, ev, len(ref.phones)))
    from pronunciada.word_match import mismatch

    W = c["W"]
    wcols = {k: np.asarray(W[k], dtype=np.float64) for k in WCOLS if k in W}
    return {"ref": ref, "inp": L.frame_inputs(ref, ev, lpa, fb), "hs": hs,
            "wprob": np.array([w.prob for w in res.words]),
            "wlen": np.array([1.0 if w.length_prob is None else w.length_prob for w in res.words]),
            "sounds": np.array([w.sounds_prob for w in res.words]),
            "present": np.array([np.nan if w.present_prob is None else w.present_prob for w in res.words]),
            "con": np.array([1.0 if w.contrast_prob is None else w.contrast_prob for w in res.words]),
            "w_zl": np.asarray(c["w_zl"], dtype=np.float64), "w_zs": np.asarray(c["w_zs"], dtype=np.float64),
            "s_zl": np.asarray(c["s_zl"], dtype=np.float64), "W": wcols, "mm": mismatch(ref, ev),
            "spans": [list(w.char_span) for w in ref.words], "texts": [w.text for w in ref.words]}


def collect(which: str, threads: int = 2) -> None:
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(threads)
    import warnings

    warnings.simplefilter("ignore")
    from bench.round19_ja.dump import _wav
    from pronunciada.scorer import Scorer

    rows = rows_for(which)
    d = WORK / "collect" / tag() / which
    d.mkdir(parents=True, exist_ok=True)
    sc = None
    for k0 in range(0, len(rows), CHUNK):
        out = d / f"chunk_{k0:05d}.pkl"
        if out.exists():
            continue
        if sc is None:
            sc = Scorer("ja", threads=threads)
        items = []
        for r in rows[k0:k0 + CHUNK]:
            try:
                res = sc.score((_wav(r["audio"]), 16000), r["text"])
            except Exception as e:  # noqa: BLE001
                print("skip", r["id"], type(e).__name__, e, file=sys.stderr)
                continue
            it = item(sc, res)
            it.update({k: r.get(k) for k in ("id", "group", "source", "units", "label", "text")})
            items.append(it)
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(items))
        os.replace(tmp, out)
        print(f"{which} {min(k0 + CHUNK, len(rows))}/{len(rows)}", file=sys.stderr, flush=True)


def load(which: str, t: str | None = None) -> list[dict]:
    d = WORK / "collect" / (t or tag()) / which
    items = []
    for p in sorted(d.glob("chunk_*.pkl")):
        items += pickle.loads(p.read_bytes())
    return items


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("fit", "dev", "devnat", "fitset"))
    a = ap.parse_args(argv)
    collect(a.which)


if __name__ == "__main__":
    main()
