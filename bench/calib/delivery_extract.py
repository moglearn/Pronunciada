"""Round 5: sentence delivery features + pooled encoder activations through the PUBLIC scorer.

    python -m bench.calib.delivery_extract [--dataset speechocean762] [--split train] [--workers 6]

Every utterance goes through pronunciada.Scorer("en", delivery=True).score(audio_path, text),
the same call a user makes; the values are read back from the result's cached evidence
(res._pronunciada["delivery"], ["pooled"]), so training inputs come from the runtime code path
(pronunciada.delivery). The scorer is built with calib_dir pointing at a directory holding the
round-4 sentence model, so no projection is applied here (bench/calib/train_delivery.py fits it
in-fold from the pooled vectors).

Output: $PRONUNCIADA_CACHE/calib/delivery_<dataset>_<split>.npz with
utt [N], names [K], D [N, K] float64, emb_<layer> [N, 2C] float32.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

_SC = None


def r4_calib_dir() -> str:
    """A calib directory identical to the shipped one but with the round-4 sentence model as
    "sentence" (no delivery inputs, no projection)."""
    import shutil

    from bench.calib.common import calib_cache, ship_dir

    d = calib_cache() / "r4_calib_dir"
    d.mkdir(exist_ok=True)
    for p in ship_dir().iterdir():
        if p.is_file() and p.name not in ("sentence.npz",):
            shutil.copy2(p, d / p.name)
    if (ship_dir() / "prosody").is_dir():
        shutil.copytree(ship_dir() / "prosody", d / "prosody", dirs_exist_ok=True)
    shutil.copy2(ship_dir() / "sentence_r4.npz", d / "sentence.npz")
    cfg = d / "config.json"
    import json

    c = json.loads(cfg.read_text())
    c.pop("sentence_model", None)
    cfg.write_text(json.dumps(c, indent=1))
    return str(d)


def _init(calib_dir):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    global _SC
    from pronunciada.scorer import Scorer

    _SC = Scorer("en", threads=1, calib_dir=calib_dir, delivery=True, prosody=False)


def _one(rec):
    try:
        res = _SC.score(rec["audio"], rec["text"])
    except ValueError as e:
        return rec["id"], None, str(e)
    c = res._pronunciada  # noqa: SLF001
    return rec["id"], (c["delivery"], c["pooled"]), None


def extract(recs, workers=6, calib_dir=None):
    from concurrent.futures import ProcessPoolExecutor

    from pronunciada import delivery as D

    calib_dir = calib_dir or r4_calib_dir()
    utt, rows, emb, errs = [], [], {}, {}
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(calib_dir,)) as ex:
        for k, (uid, val, err) in enumerate(ex.map(_one, recs, chunksize=4)):
            if val is None:
                errs[uid] = err
                continue
            d, pooled = val
            utt.append(uid)
            rows.append([d.get(c, np.nan) for c in D.ALL])
            for layer, v in (pooled or {}).items():
                emb.setdefault(layer, []).append(v)
            if (k + 1) % 500 == 0:
                print(f"{k + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    out = {"utt": np.array(utt), "names": np.array(D.ALL), "D": np.array(rows, dtype=np.float64)}
    out.update({f"emb_{k}": np.stack(v).astype(np.float32) for k, v in emb.items()})
    print(f"{len(utt)} utts in {time.time() - t0:.0f}s, {len(errs)} errors", flush=True)
    return out, errs


def main():
    from bench.calib.common import calib_cache
    from bench.pipeline.common import records

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="speechocean762")
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    recs = [r for r in records(a.dataset) if r["split"] == a.split]
    out, errs = extract(recs, a.workers)
    np.savez(calib_cache() / f"delivery_{a.dataset}_{a.split}.npz", **out)
    if errs:
        print("errors:", list(errs.items())[:5])


if __name__ == "__main__":
    main()
