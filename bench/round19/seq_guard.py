"""Round 19: native guard of the sequence head (trained on all of speechocean762 train) on the
100 FLEURS-validation English clips, and of its 0.25 blend with a tree candidate.

    python -m bench.round19.seq_guard [--tree-config CFG]
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np

from bench.round19 import data as RD
from bench.round19 import exp_seq as ES
from bench.round19 import seqhead as SH
from pronunciada import features as F


def main():
    import torch

    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-config", default="")
    ap.add_argument("--blend", type=float, default=0.25)
    a = ap.parse_args()
    from bench.calib.common import calib_cache, load_json, save_json
    from bench.data.common import cache_root

    layers = list(RD.LAYERS)
    keep = ["speaker", "labels", "delivery", "f0", "ph_layers", "tab_num", "phones", "phone_word", "words", "dur", "S",
            "sent", "utt_layers", "voice"]
    U = RD.load("train", keep=keep)
    z = np.load(calib_cache() / "round2_sentence_inputs_clean.npz", allow_pickle=False)
    pos = {u: i for i, u in enumerate(z["utt"])}
    U = [u for u in U if u["id"] in pos]
    items, vocab = ES.build(U, z["XB"][[pos[u["id"]] for u in U]], layers, ["xb", "deliv", "f0"])
    d = cache_root() / "round19" / "english" / "feats" / "fleurs_validation_fp32"
    V = [v for p in sorted(d.glob("shard*.pkl")) for v in pickle.loads(p.read_bytes()).values() if "error" not in v]
    for v in V:
        v["labels"] = {"phone": np.full(len(v["phones"]), np.nan), "word": np.full(len(v["words"]), np.nan),
                       "sentence": {h: np.nan for h in RD.HEADS}, "age": 99}
        v["id"] = v["text"]
    XV = np.array([[v["S"][c] for c in F.SENT_NUM] for v in V])
    iv, _ = ES.build(V, XV, layers, ["xb", "deliv", "f0"], vocab=vocab)
    lnorm = ES.layer_norms(items, layers)
    allix = np.arange(len(items))
    tabn = SH.Norm(np.concatenate([it["xt_raw"] for it in items]))
    uttn = SH.Norm(np.stack([it["xu_raw"] for it in items]))
    itr = ES.apply_norms(items, allix, tabn, uttn, lnorm, layers)
    ite = ES.apply_norms(iv, np.arange(len(iv)), tabn, uttn, lnorm, layers)
    m = SH.build_module(itr[0]["xt"].shape[1], len(itr[0]["xu"]), len(vocab), layers, d=64, n_layers=1, ff=128,
                        dropout=0.2)
    SH.train(m, itr, layers, epochs=25, w_corr=1.0, seed=0)
    S, _, _ = SH.predict(m, ite, layers)
    S = np.clip(S, 0, 1) * 10
    old = np.array([[v["sent"][h] for h in RD.HEADS] for v in V])
    out = {"seq": {h: float(np.median(S[:, j]) - np.median(old[:, j])) for j, h in enumerate(RD.HEADS)}}
    if a.tree_config:
        from bench.round19 import fit_final as FF

        M = FF.fit_all(a.tree_config)
        T = FF.predict(M, V)
        out["tree"] = {h: float(np.median(T[:, j]) - np.median(old[:, j])) for j, h in enumerate(RD.HEADS)}
        B = a.blend * S + (1 - a.blend) * T
        out[f"blend{a.blend}"] = {h: float(np.median(B[:, j]) - np.median(old[:, j])) for j, h in enumerate(RD.HEADS)}
    for k, v in out.items():
        print(k, {h: round(x, 3) for h, x in v.items()})
    R = load_json("round19_english_guard.json") or {}
    R["seqhead_s6"] = out
    save_json("round19_english_guard.json", R)


if __name__ == "__main__":
    main()
