"""Length-model rows for discriminative fitting: natives and planted long <-> short swaps, each scored
on its own alignment (the swap is aligned as the swapped text, as at runtime).

    python -m bench.round19_ja.swapcollect fit|dev     # -> <work>/swaprows_<set>/chunk_*.pkl

fit: the fit pool natives (Common Voice ja train + FLEURS ja train); dev: the dev pool natives
(Common Voice ja dev + FLEURS validation). Swaps: bench/audit/contrasts.py generators of the
Japanese length contrasts (vowel_length, gemination), both directions, at most 2 per clip and
direction (seeds "r19-length"). Per judged sound: the feature row (round-15 features, the round-17
head evidence for "long", pronunciada.length_r19 extras), the expected length, whether the sound is a
planted swap, the source and group.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

from bench.round19_ja.cv_pool import WORK

CHUNK = 50
PER_DIR = 2


def rows_for_result(res, sc, flag=()):
    from pronunciada import contrast_heads as CH
    from pronunciada import length as L
    from pronunciada import length_r19 as R
    from pronunciada.sound_emb import phone_vectors

    c = res._pronunciada  # noqa: SLF001
    ref, ev, lpa, emb = c["ref"], c["ev"], c["lpa"], c["emb"]
    fb = sc._posteriors(c["wav"], {})[1]  # noqa: SLF001
    inp = L.frame_inputs(ref, ev, lpa, fb)
    lab = L.labels(ref)
    idx, cls, rows = L.features(ref, lab, inp)
    hs = CH.length_evidence("ja", sc.length_cal, [p.ipa for p in ref.phones], phone_vectors(emb, ev, len(ref.phones)))
    se = np.array([[s.start_frame, s.end_frame, s.alt_index] for s in ev.slots], np.int32)
    R.extra(ref, inp, idx, rows, lpa=lpa, slot_se=se, fb=fb)
    # shipped probability of the same sound (round 15 + 17), for comparison
    shp = L.score(ref, inp, L.params("ja"), head_s=hs)["prob"]
    out = []
    fl = set(flag)
    for i, cc, r in zip(idx, cls, rows):
        s = float(hs[i]) if hs is not None and np.isfinite(hs[i]) else 0.0
        r["head_s"] = s if lab[i] == "long" else -s
        r["head_rel"] = s
        r["has_head"] = float(hs is not None and np.isfinite(hs[i]))
        out.append({"i": i, "cls": cc, "lab": lab[i], "swap": i in fl, "row": r, "shipped": shp[i],
                    "word": ref.phones[i].word})
    return out


def run(which: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import warnings

    warnings.simplefilter("ignore")
    from bench.audit.contrasts import apply, get_contrast, sites
    from bench.pipeline.common import stable_seed
    from bench.round19_ja.dump import _wav
    from bench.round19_ja.plant import pool_rows
    from pronunciada.scorer import Scorer

    recs = pool_rows("fit" if which == "fit" else "dev")
    d = WORK / f"swaprows_{which}"
    d.mkdir(parents=True, exist_ok=True)
    sc = None
    for k0 in range(0, len(recs), CHUNK):
        out = d / f"chunk_{k0:05d}.pkl"
        if out.exists():
            continue
        if sc is None:
            sc = Scorer("ja", threads=2)
        items = []
        for rec in recs[k0:k0 + CHUNK]:
            try:
                wav = _wav(rec["audio"])
                res = sc.score((wav, 16000), rec["text"])
            except Exception:  # noqa: BLE001
                continue
            meta = {"id": rec["id"], "group": rec["group"], "source": rec["source"]}
            items.append({**meta, "kind": "native", "rows": rows_for_result(res, sc)})
            ref = res._pronunciada["ref"]  # noqa: SLF001
            for cn in ("vowel_length", "gemination"):
                st = sites(ref, get_contrast("ja", cn))
                for dr in ("A>B", "B>A"):
                    pool = [s for s in st if s.direction == dr]
                    rng = np.random.default_rng(stable_seed(rec["id"], cn, dr, "r19-length"))
                    for j in rng.permutation(len(pool))[:PER_DIR]:
                        try:
                            e = apply(ref, pool[j], rng)
                            r2 = sc.score((wav, 16000), e.ref)
                        except Exception:  # noqa: BLE001
                            continue
                        items.append({**meta, "kind": "swap", "contrast": cn, "direction": dr,
                                      "rows": [x for x in rows_for_result(r2, sc, e.flag) if x["swap"]]})
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(items))
        os.replace(tmp, out)
        print(f"{which} {min(k0 + CHUNK, len(recs))}/{len(recs)}", file=sys.stderr, flush=True)


def load(which: str) -> list[dict]:
    items = []
    for p in sorted((WORK / f"swaprows_{which}").glob("chunk_*.pkl")):
        items += pickle.loads(p.read_bytes())
    return items


if __name__ == "__main__":
    run(sys.argv[1])
