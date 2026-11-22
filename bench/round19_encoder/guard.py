"""Round 19 (encoder adaptation): native guard of a candidate fitted on ALL of speechocean762 train.

    python -m bench.round19_encoder.guard --name NAME --mode {stage1,frozen,lora,full} [stage-2 args of train_cv]

Fits stage 1 (the English s6 head, seed 0) and, unless --mode stage1, stage 2 on every train clip, then
scores the 100 seeded FLEURS-validation English clips (natives; the sample of
bench/calib/native_sentence_check.py) and reports, per sentence head, median(new) - median(shipped)
on the 0-10 scale (the check of results/licence_refit.md: a drop of more than 0.3 fails), the
same for the stage-1 head alone, and the share of natives the new head puts lower than the shipped one.
Writes results/tables/round19_encoder_guard.json[NAME] and keeps the fitted head + merged stack-5 weights
under $PRONUNCIADA_CACHE/round19/encoder/final_<NAME>.pt (for export).
"""
from __future__ import annotations

import argparse
import copy
import pickle

import numpy as np

from bench.round19_encoder import en_data as RD
from bench.round19_encoder import en_exp_seq as ES
from bench.round19_encoder import en_seqhead as SH
from bench.round19_encoder import train_cv as TC


def fleurs_items(vocab):
    from bench.data.common import cache_root
    from pronunciada import features as F

    d = cache_root() / "round19" / "english" / "feats" / "fleurs_validation_fp32"
    V = []
    for p in sorted(d.glob("shard*.pkl")):
        for uid, v in pickle.loads(p.read_bytes()).items():
            if "error" in v:
                continue
            v["id"] = uid
            V.append(v)
    for v in V:
        v["labels"] = {"phone": np.full(len(v["phones"]), np.nan), "word": np.full(len(v["words"]), np.nan),
                       "sentence": {h: np.nan for h in RD.HEADS}, "age": 99}
    XV = np.array([[v["S"][c] for c in F.SENT_NUM] for v in V])
    iv, _ = ES.build(V, XV, TC.LAYERS, ["xb", "deliv", "f0"], vocab=vocab)
    cuts = TC.load_cuts("fleurs_validation")
    for it, v in zip(iv, V):
        c = cuts[v["id"]]
        assert len(c) == v["n_frames"]
        it["cut"] = c
        it["win"] = TC.windows(v["start"], v["end"], len(c))
    old = np.array([[v["sent"][h] for h in RD.HEADS] for v in V])
    return iv, old


def main():
    import os

    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.2")
    import torch

    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["stage1", "frozen", "lora", "full"], required=True)
    ap.add_argument("--e2", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr-top", type=float, default=3e-4)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--l2sp", type=float, default=0.0)
    ap.add_argument("--soup", type=int, default=1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--budget", type=int, default=6000)
    a = ap.parse_args()
    TC.Model.budget = a.budget
    from bench.calib.common import load_json, save_json

    items, vocab = TC.load_items()
    lnorm = ES.layer_norms(items, TC.LAYERS)
    iv, old = fleurs_items(vocab)
    allix = np.arange(len(items))
    cache = TC.work_dir() / "ckpt"
    cache.mkdir(exist_ok=True)
    tabn = SH.Norm(np.concatenate([it["xt_raw"] for it in items]))
    uttn = SH.Norm(np.stack([it["xu_raw"] for it in items]))
    itr = ES.apply_norms(items, allix, tabn, uttn, lnorm, TC.LAYERS)
    ite = ES.apply_norms(iv, np.arange(len(iv)), tabn, uttn, lnorm, TC.LAYERS)
    for dst, src in ((itr, items), (ite, iv)):
        for x, y in zip(dst, src):
            x["cut"], x["win"] = y["cut"], y["win"]
    p1 = cache / "stage1_all_s0.pt"
    h1 = SH.build_module(itr[0]["xt"].shape[1], len(itr[0]["xu"]), len(vocab), TC.LAYERS, **TC.S6)
    if p1.exists():
        h1.load_state_dict(torch.load(p1))
        h1.eval()
    else:
        SH.train(h1, itr, TC.LAYERS, epochs=TC.S6_TRAIN["epochs"], w_corr=TC.S6_TRAIN["w_corr"], seed=0)
        torch.save(h1.state_dict(), p1)
    S1, _, _ = SH.predict(h1, ite, TC.LAYERS)
    S1 = np.clip(S1, 0, 1) * 10
    out = {"stage1": {h: float(np.median(S1[:, j]) - np.median(old[:, j])) for j, h in enumerate(RD.HEADS)}}
    if a.mode != "stage1":
        heads, tops = [], []
        for k in range(a.soup):
            h, t = TC.stage2(copy.deepcopy(h1), itr, lnorm, a, 7 + k, a.device, print)
            heads.append(h)
            tops.append(t)
        h, t = TC.soup(heads, tops) if a.soup > 1 else (heads[0], tops[0])
        S, _, _ = TC.predict2(h, t, ite, lnorm, a.device)
        S = np.clip(S, 0, 1) * 10
        out["new"] = {h_: float(np.median(S[:, j]) - np.median(old[:, j])) for j, h_ in enumerate(RD.HEADS)}
        out["new_vs_stage1"] = {h_: float(np.median(S[:, j]) - np.median(S1[:, j])) for j, h_ in enumerate(RD.HEADS)}
        out["share_lower_than_shipped_by_0.5"] = {h_: float(np.mean(S[:, j] < old[:, j] - 0.5))
                                                  for j, h_ in enumerate(RD.HEADS)}
        torch.save({"head": h.cpu().state_dict(), "top": t.merged_initializers(), "args": vars(a),
                    "tabn": (tabn.mu, tabn.sd, tabn.clip), "uttn": (uttn.mu, uttn.sd, uttn.clip),
                    "lnorm": lnorm, "vocab": vocab}, TC.work_dir() / f"final_{a.name}.pt")
    out["worst_drop"] = float(min(min(v.values()) for k, v in out.items() if k in ("stage1", "new")))
    out["pass"] = bool(out["worst_drop"] >= -0.3)
    for k, v in out.items():
        print(k, v if not isinstance(v, dict) else {h: round(x, 3) for h, x in v.items()})
    R = load_json("round19_encoder_guard.json") or {}
    R[a.name] = {"args": vars(a), **out}
    save_json("round19_encoder_guard.json", R)


if __name__ == "__main__":
    main()
