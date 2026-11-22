"""Round 19: 3-seed CV of blending the sequence head's OOF sentence scores into the tree candidate.

    python -m bench.round19.cv_blend [--seq s6_smallreg]
"""
from __future__ import annotations

import argparse

import numpy as np

from bench.round19 import data as RD
from bench.round19 import exp_utt as E

S7 = "enc0.std+enc1.std+enc2.std+enc3.std+enc4.std+enc5.std+final.std"
CAND = "deliv:all,f0,proj:" + S7


def main():
    from bench.calib.common import save_json
    from bench.data.common import cache_root

    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="s6_smallreg")
    a = ap.parse_args()
    oof = cache_root() / "round19" / "english" / "oof"
    d = E.load_train()
    Y, ch = d["Y"], d["age"] <= 12
    res = {}

    def score(P):
        r = {h: RD.pearson(P[:, j], Y[:, j]) for j, h in enumerate(RD.HEADS)}
        r["total_child"] = RD.pearson(P[ch, 3], Y[ch, 3])
        r["total_adult"] = RD.pearson(P[~ch, 3], Y[~ch, 3])
        return r

    for seed in RD.SEEDS:
        _, T = E.cv(d, CAND, seed, return_oof=True)
        _, B = E.cv(d, "base", seed, return_oof=True)
        np.savez(oof / f"tree_r19_seed{seed}.npz", T=T, B=B, ids=np.array(d["ids"]))
        z = np.load(oof / f"seq_{a.seq}_seed{seed}.npz")
        pos = {u: i for i, u in enumerate(z["ids"])}
        S = np.clip(z["S"][[pos[i] for i in d["ids"]]], 0, 10)
        res.setdefault("base", []).append(score(B))
        for w in (0.0, 0.15, 0.25, 0.35, 0.5):
            res.setdefault(f"w{w}", []).append(score(w * S + (1 - w) * T))
    out = {}
    for k, rs in res.items():
        out[k] = {"seeds": rs, "mean": {m: float(np.mean([r[m] for r in rs])) for m in rs[0]}}
        print(k, " ".join(f"{m}={v:.4f}" for m, v in out[k]["mean"].items()), flush=True)
    save_json("round19_english_cv_blend.json", out)


if __name__ == "__main__":
    main()
