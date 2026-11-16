"""Round 6: outside pre-trained encoders as an extra embedding source for the sentence heads.

    python -m bench.calib.train_ext screen          # ridge-only OOF Pearson per model x layer
    python -m bench.calib.train_ext cv [--top 2]    # full round-5 pipeline + one outside source
    python -m bench.calib.train_ext harsh           # age-band split check of the finalists

Fits on speechocean762 TRAIN only. Pooled hidden states come from bench/calib/ext_encoders.py.
The round-5 configuration (SENT_NUM + disfluency + ZIPA stack-1 projection, 0.7 / 0.3
tree / ridge blend) is the reference; a candidate adds one outside source (a layer or two
layers concatenated) with its own in-fold ridge projection onto the four targets.
Adoption bar (set before looking): 3-seed CV prosodic >= round 5 + 0.015 and total not lower;
among those, the smallest model within 0.005 (prosodic) of the best.
Writes results/tables/round6_ext.json.
"""
from __future__ import annotations

import argparse

import numpy as np

from bench.calib import train_delivery as T
from bench.calib.common import calib_cache, load_json, save_json
from bench.calib.fitlib import log

MODELS = ("whisper-tiny", "whisper-base", "wav2vec2-base", "hubert-base")
SIZE_M = {"whisper-tiny": 8, "whisper-base": 20, "wav2vec2-base": 95, "hubert-base": 95}   # encoder params (M)
R5 = dict(groups=["emb", "disfluency"], layers=("enc1",), ridge_blend=T.BLEND_W)


def load_ext(data, model):
    p = calib_cache() / f"ext_{model}_train.npz"
    if not p.exists():
        return None
    z = np.load(p)
    pos = {u: i for i, u in enumerate(z["utt"])}
    ix = np.array([pos[u] for u in data["utt"]])
    return {int(k[1:]): z[k][ix].astype(np.float64) for k in z.files if k.startswith("L")}


def screen(a):
    data = T.load_data()
    Y, spk = data["Y"], data["spk"]
    out = load_json("round6_ext.json") or {}
    out["screen"] = {}
    _, P, _ = T.pick_alpha(data["emb"]["enc1"], Y, spk)
    out["screen"]["zipa-enc1"] = {h: float(np.corrcoef(P[:, j], Y[:, j])[0, 1]) for j, h in enumerate(T.HEADS)}
    log("zipa-enc1", out["screen"]["zipa-enc1"])
    for m in MODELS:
        L = load_ext(data, m)
        if L is None:
            continue
        for i, E in sorted(L.items()):
            _, P, _ = T.pick_alpha(E, Y, spk)
            r = {h: float(np.corrcoef(P[:, j], Y[:, j])[0, 1]) for j, h in enumerate(T.HEADS)}
            out["screen"][f"{m}:L{i}"] = r
            log(m, i, {h: round(v, 4) for h, v in r.items()})
    save_json("round6_ext.json", out)


def _job(args):
    name, layers, seed, folds = args
    data = T.load_data()
    ext = []
    if name:
        L = load_ext(data, name)
        ext = [np.column_stack([L[i] for i in layers])]
    return T.cv_score(data, R5["groups"], seed, layers=R5["layers"], ridge_blend=R5["ridge_blend"], ext=ext,
                      folds=folds)


def candidates(top):
    sc = load_json("round6_ext.json")["screen"]
    out = []
    for m in MODELS:
        ks = [k for k in sc if k.startswith(m + ":")]
        if not ks:
            continue
        ks.sort(key=lambda k: -(sc[k]["prosodic"] + sc[k]["total"]))
        best = [int(k.split(":L")[1]) for k in ks[:top]]
        out += [(m, (i,)) for i in best]
        if len(best) >= 2:
            out.append((m, tuple(sorted(best[:2]))))
    return out


def run_cv(a):
    from concurrent.futures import ProcessPoolExecutor

    cands = [("", ())] + candidates(a.top)
    jobs = [(m, l, s, None) for m, l in cands for s in T.SEEDS]
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(zip([(m, l, s) for m, l, s, _ in jobs], ex.map(_job, jobs)))
    out = load_json("round6_ext.json")
    base = {s: res[("", (), s)] for s in T.SEEDS}
    out["cv"] = {}
    for m, l in cands:
        r = {s: res[(m, l, s)] for s in T.SEEDS}
        c = T.compare(r, base)
        key = "round5" if not m else f"{m}:L" + "+".join(map(str, l))
        ok = bool(m) and c["prosodic"]["delta"] >= 0.015 and c["total"]["delta"] >= 0
        out["cv"][key] = {"model": m, "layers": list(l), "heads": c, "clears_bar": ok}
        log(key, {h: (round(c[h]["mean"], 4), round(c[h]["delta"], 4), round(c[h]["delta_se"], 4)) for h in T.HEADS},
            "CLEARS" if ok else "")
    clear = {k: v for k, v in out["cv"].items() if v["clears_bar"]}
    if clear:
        best = max(v["heads"]["prosodic"]["mean"] for v in clear.values())
        near = [k for k, v in clear.items() if v["heads"]["prosodic"]["mean"] >= best - 0.005]
        out["chosen"] = min(near, key=lambda k: (SIZE_M[clear[k]["model"]], -clear[k]["heads"]["prosodic"]["mean"]))
    else:
        out["chosen"] = None
    log("chosen", out["chosen"])
    save_json("round6_ext.json", out)


def age_folds(data):
    from bench.pipeline.common import records

    age = {r["id"]: r["labels"]["age"] for r in records("speechocean762") if r["split"] == "train"}
    child = np.array([age[u] < 15 for u in data["utt"]])
    return [(np.flatnonzero(~child), np.flatnonzero(child)), (np.flatnonzero(child), np.flatnonzero(~child))]


def harsh(a):
    """Train on adults -> evaluate children and vice versa (pooled OOF Pearson)."""
    from concurrent.futures import ProcessPoolExecutor

    data = T.load_data()
    folds = age_folds(data)
    out = load_json("round6_ext.json")
    keys = [k for k in out.get("cv", {}) if k == "round5" or out["cv"][k]["heads"]["prosodic"]["delta"] > 0.005]
    jobs = []
    for k in keys:
        m = out["cv"][k]["model"]
        jobs.append((m, tuple(out["cv"][k]["layers"]), 0, folds))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_job, jobs))
    out["harsh_age"] = {k: r for k, r in zip(keys, res)}
    for k, r in zip(keys, res):
        log("age split", k, {h: round(v, 4) for h, v in r.items()})
    save_json("round6_ext.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("screen", "cv", "harsh"))
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"screen": screen, "cv": run_cv, "harsh": harsh}[a.cmd](a)


if __name__ == "__main__":
    main()
