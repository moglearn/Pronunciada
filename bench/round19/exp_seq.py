"""Round 19 (English): CV of the multi-task sequence head (bench/round19/seqhead.py), TRAIN only.

    python -m bench.round19.exp_seq --name NAME [--layers enc0+...+final] [--epochs 40] [--d 96]
        [--n-layers 2] [--utt xb+deliv+f0] [--seeds 0,1,2] [--w-corr 0.5] ...

Same folds as bench/round19/exp_utt.py. Reports OOF Pearson of the four sentence heads (all,
children <= 12, others), of the word score and of the sound score (expert labels mapped to our
sounds / words), the learned layer weights, and the OOF blend with the shipped-recipe trees.
Results: results/tables/round19_english_cv_seq.json; OOF predictions under
$PRONUNCIADA_CACHE/round19/english/oof/.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from bench.round19 import data as RD
from bench.round19 import seqhead as SH
from pronunciada import delivery as DV

HEADS = RD.HEADS


def build(utts, XB, layers, utt_parts, vocab=None):
    if vocab is None:
        vocab = {p: i + 1 for i, p in enumerate(sorted({p for u in utts for p in u["phones"]}))}
    items = []
    for u, xb in zip(utts, XB):
        parts = []
        if "xb" in utt_parts:
            parts.append(xb)
        if "deliv" in utt_parts:
            parts.append(np.array([u["delivery"].get(c, np.nan) for c in DV.ALL]))
        if "f0" in utt_parts:
            parts.append(np.array([u["f0"].get(k, np.nan) for k in ("f0_med_hz", "f0_p10_hz", "f0_p90_hz")]))
        if "dur" in utt_parts:
            parts.append(np.array([u["dur"], len(u["phones"]), len(u["words"])], float))
        lab = u["labels"]
        pw = np.asarray(u["phone_word"])
        items.append({"xl": {k: u["ph_layers"][k] for k in layers}, "xt_raw": u["tab_num"],
                      "pid": np.array([vocab.get(p, 0) for p in u["phones"]], np.int64),
                      "ppos": SH.word_pos(pw), "xu_raw": np.concatenate(parts) if parts else np.zeros(1),
                      "yp": np.asarray(lab["phone"], float) / 2.0,
                      "yw_ph": np.asarray(lab["word"], float)[pw] / 10.0, "wid": pw.astype(np.int64),
                      "ys": np.array([lab["sentence"][h] for h in HEADS], np.float32) / 10.0,
                      "yc": float(lab["sentence"].get("completeness", np.nan)) / 10.0,
                      "age": lab["age"], "spk": u["speaker"], "id": u["id"]})
    return items, vocab


def layer_norms(items, layers):
    out = {}
    for k in layers:
        X = np.concatenate([it["xl"][k] for it in items]).astype(np.float32)
        mu, sd = X.mean(0), X.std(0)
        out[k] = (mu, np.where(sd < 1e-6, 1.0, sd))
    return out


def apply_norms(items, idx, tabn, uttn, lnorm, layers):
    out = []
    for i in idx:
        it = dict(items[i])
        it["xt"] = tabn(it["xt_raw"])
        it["xu"] = uttn(it["xu_raw"][None])[0]
        it["xl"] = {k: ((it["xl"][k].astype(np.float32) - lnorm[k][0]) / lnorm[k][1]).astype(np.float16) for k in layers}
        out.append(it)
    return out


def evaluate(items, S, P, Wd):
    Y = np.stack([it["ys"] for it in items]) * 10
    S = np.clip(S, 0, 1) * 10
    age = np.array([it["age"] for it in items])
    ch = age <= 12
    r = {h: RD.pearson(S[:, j], Y[:, j]) for j, h in enumerate(HEADS)}
    r["total_child"] = RD.pearson(S[ch, 3], Y[ch, 3])
    r["total_adult"] = RD.pearson(S[~ch, 3], Y[~ch, 3])
    yp = np.concatenate([it["yp"] for it in items])
    pp = np.concatenate(P)
    m = np.isfinite(yp)
    r["phone"] = RD.pearson(pp[m], yp[m])
    yw, pw = [], []
    for it, w in zip(items, Wd):
        wid = it["wid"]
        for k in np.unique(wid):
            j = np.flatnonzero(wid == k)[0]
            if np.isfinite(it["yw_ph"][j]):
                yw.append(it["yw_ph"][j])
                pw.append(w[j])
    r["word"] = RD.pearson(pw, yw)
    return r


def main():
    import torch

    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--layers", default="enc0+enc1+enc2+enc3+enc4+enc5+final")
    ap.add_argument("--utt", default="xb+deliv+f0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--ff", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--w-corr", type=float, default=0.5)
    ap.add_argument("--w-word", type=float, default=0.5)
    ap.add_argument("--w-phone", type=float, default=0.5)
    ap.add_argument("--no-tab", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--folds", default="")
    ap.add_argument("--corr", default="pearson", choices=("pearson", "ccc"))
    ap.add_argument("--w-rank", type=float, default=0.0)
    ap.add_argument("--bottleneck", action="store_true")
    ap.add_argument("--ens", type=int, default=1, help="models per fold (different seeds), outputs averaged")
    a = ap.parse_args()
    from bench.calib.common import calib_cache, load_json, save_json
    from bench.data.common import cache_root

    layers = a.layers.split("+") if a.layers else []
    t0 = time.time()
    U = RD.load("train", keep=["speaker", "labels", "delivery", "f0", "ph_layers", "tab_num", "phones",
                               "phone_word", "words", "dur"])
    z = np.load(calib_cache() / "round2_sentence_inputs_clean.npz", allow_pickle=False)
    pos = {u: i for i, u in enumerate(z["utt"])}
    U = [u for u in U if u["id"] in pos]
    XB = z["XB"][[pos[u["id"]] for u in U]]
    items, vocab = build(U, XB, layers, a.utt.split("+"))
    del U
    if a.no_tab:
        for it in items:
            it["xt_raw"] = np.zeros((len(it["pid"]), 1))
    lnorm = layer_norms(items, layers)
    spk = np.array([it["spk"] for it in items])
    age = np.array([it["age"] for it in items])
    print(f"loaded {len(items)} utts in {time.time() - t0:.0f}s", flush=True)
    R = load_json("round19_english_cv_seq.json") or {}
    res = {"args": vars(a), "seeds": {}}
    oofdir = cache_root() / "round19" / "english" / "oof"
    oofdir.mkdir(parents=True, exist_ok=True)
    for seed in [int(s) for s in a.seeds.split(",")]:
        S = np.zeros((len(items), 4))
        P = [None] * len(items)
        Wd = [None] * len(items)
        wts = []
        fl = RD.folds(spk, age, seed)
        sel = [int(f) for f in a.folds.split(",")] if a.folds else range(len(fl))
        done = np.zeros(len(items), bool)
        for fi in sel:
            tr, te = fl[fi]
            tabn = SH.Norm(np.concatenate([items[i]["xt_raw"] for i in tr]))
            uttn = SH.Norm(np.stack([items[i]["xu_raw"] for i in tr]))
            itr = apply_norms(items, tr, tabn, uttn, lnorm, layers)
            ite = apply_norms(items, te, tabn, uttn, lnorm, layers)
            outs = []
            for e in range(a.ens):
                m = SH.build_module(itr[0]["xt"].shape[1], len(itr[0]["xu"]), len(vocab), layers, d=a.d,
                                    n_layers=a.n_layers, ff=a.ff, dropout=a.dropout, bottleneck=a.bottleneck)
                SH.train(m, itr, layers, epochs=a.epochs, lr=a.lr, wd=a.wd, w_word=a.w_word, w_phone=a.w_phone,
                         w_corr=a.w_corr, seed=seed * 10 + fi + 1000 * e, corr=a.corr, w_rank=a.w_rank)
                outs.append(SH.predict(m, ite, layers))
            s = np.mean([o[0] for o in outs], 0)
            p = [np.mean([o[1][k] for o in outs], 0) for k in range(len(te))]
            w = [np.mean([o[2][k] for o in outs], 0) for k in range(len(te))]
            S[te] = s
            for k, i in enumerate(te):
                P[i], Wd[i] = p[k], w[k]
            done[te] = True
            wts.append(torch.softmax(m.alpha, 0).detach().numpy().round(3).tolist() if layers else [])
            print(f"seed {seed} fold {fi}: {time.time() - t0:.0f}s, params {SH.n_params(m)}, layer w {wts[-1]}",
                  flush=True)
        idx = np.flatnonzero(done)
        r = evaluate([items[i] for i in idx], S[idx], [P[i] for i in idx], [Wd[i] for i in idx])
        r["layer_weights"] = wts
        res["seeds"][seed] = r
        np.savez(oofdir / f"seq_{a.name}_seed{seed}.npz", ids=np.array([it["id"] for it in items]), S=S * 10,
                 done=done)
        import pickle

        (oofdir / f"seq_{a.name}_seed{seed}_pw.pkl").write_bytes(pickle.dumps(
            {it["id"]: (P[i], Wd[i]) for i, it in enumerate(items) if done[i]}))
        print(f"seed {seed}: " + " ".join(f"{k}={v:.4f}" for k, v in r.items() if isinstance(v, float)), flush=True)
    ks = [k for k, v in next(iter(res["seeds"].values())).items() if isinstance(v, float)]
    res["mean"] = {k: float(np.mean([r[k] for r in res["seeds"].values()])) for k in ks}
    print("MEAN " + " ".join(f"{k}={v:.4f}" for k, v in res["mean"].items()), flush=True)
    R[a.name] = res
    save_json("round19_english_cv_seq.json", R)


if __name__ == "__main__":
    main()
