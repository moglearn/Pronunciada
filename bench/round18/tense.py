"""Round 18 B: Korean lax / tense obstruents (달 / 딸, 자다 / 짜다, 사다 / 싸다).

    python -m bench.round18.tense            # train the head, dev evaluation, write the model

Labels: RefPhone.tense from the Korean front-end (spelling ㄲ ㄸ ㅃ ㅆ ㅉ plus its own
post-obstruent tensification). Head: logistic regression (C = 0.05) on the pooled enc1|enc3|enc5
vector of the sound + its neighbours' mean, FLEURS TRAIN natives (bench/coach/dump_r4
fleurs-train). Evidence and cap exactly as round 17 (bench/round17/fit.py): s = logit P(expected
class), z = slope (s - s0), shown log-odds := min(., z); grid q x slope, guards native sound
acceptance -0.5 pt and planted sound AUC -0.003 (FLEURS validation, the round-17 "after"
collection of the audit clips), best lower-direction hit wins.
Swaps: the reference's tense flag is flipped at one member sound (audio tense, reference lax =
"said tense, expected lax", and the reverse). The symbol, alignment and every other input are
unchanged, so the edited sound's shown probability is the native one capped by the flipped
evidence (no re-scoring needed); 120 per direction, seed 0, at most 2 per clip and direction.
Writes pronunciada/data/coach/ko_tense.npz, the "tense" entry of contrast_heads.json and
results/tables/round18_tense_dev.json.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from bench.coach.gather import out_dir
from pronunciada import attr_heads as AH

QS = (0.002, 0.005, 0.01, 0.02, 0.05)
SLOPES = (0.5, 1.0, 2.0, 4.0)
BUDGET = {"careful": 0.05, "balanced": 0.1, "eager": 0.2}


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def train():
    from sklearn.linear_model import LogisticRegression

    from bench.pipeline.common import records
    from pronunciada.text import get_g2p

    g = get_g2p("ko")
    D = pickle.loads((out_dir() / "dump-r4-fleurs-train.pkl").read_bytes())
    texts = {r["id"]: r["text"] for r in records("fleurs", split="train", lang="ko")}
    X, y = [], []
    for rid, o in D.items():
        if o.get("lang") != "ko" or rid not in texts:
            continue
        ref = g(texts[rid])
        if [p.ipa for p in ref.phones] != list(o["ipa"]):
            continue
        V = o["vec"].astype(np.float32)
        C = AH.context_vectors(V)
        for i, p in enumerate(ref.phones):
            if p.tense is not None:
                X.append(np.concatenate([V[i], C[i]]))
                y.append(bool(p.tense))
    X, y = np.array(X), np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(C=0.05, max_iter=600).fit((X - mu) / sd, y)
    W = np.zeros((X.shape[1], 2), np.float32)
    W[:, 1] = m.coef_[0]
    b = np.array([0.0, m.intercept_[0]], np.float32)
    np.savez_compressed("pronunciada/data/coach/ko_tense.npz", mu=mu.astype(np.float16), sd=sd.astype(np.float16),
                        W=W.astype(np.float16), b=b, classes=np.array(["no", "yes"]))
    return {"n_train": int(len(y)), "n_tense": int(y.sum())}


def dev_rows(split="validation"):
    """Validation audit clips (round-17 "after" collection): natives with shown log-odds, the tense
    head's P(tense), the reference tense flags; planted probabilities."""
    from bench.round17.collect import clips as r17clips
    from bench.round17.collect import load
    from pronunciada.contrast_heads import tense_probs
    from pronunciada.text import get_g2p

    g = get_g2p("ko")
    texts = {r["id"]: r["text"] for r in r17clips("ko", split)}
    clips = load("ko", split, "after")
    nat = {"zl": [], "zs": [], "pt": [], "tense": [], "clip": []}
    planted = []
    for k, c in enumerate(clips):
        ref = g(texts[c["id"]])
        n = c["native"]
        if [p.ipa for p in ref.phones] != list(n["ipa"]):
            continue
        V = n["vec"].astype(np.float32)
        pt = tense_probs(V, [p.tense is not None for p in ref.phones])
        nat["zl"] += list(n["zl"])
        nat["zs"] += list(n["zs"])
        nat["pt"] += list(pt)
        nat["tense"] += [p.tense for p in ref.phones]
        nat["clip"] += [k] * len(V)
        for e in c["planted"]:
            planted += list(e["prob"])
    for key in ("zl", "zs", "pt", "clip"):
        nat[key] = np.asarray(nat[key], np.float64)
    return nat, np.asarray(planted, np.float64)


def swaps(nat, n=120, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for d, src in (("A>B", True), ("B>A", False)):          # A = tense: "said tense, expected lax"
        idx = [i for i, t in enumerate(nat["tense"]) if t is src]
        per = {}
        keep = []
        for i in rng.permutation(idx):
            c = int(nat["clip"][i])
            if per.get(c, 0) < 2:
                per[c] = per.get(c, 0) + 1
                keep.append(int(i))
            if len(keep) >= n:
                break
        out[d] = np.array(keep)
    return out


def shown(zl, zs, z):
    zl2, zs2 = np.minimum(zl, z), np.minimum(zs, z)
    return sig(0.5 * (zl2 + np.minimum(zs2, zl2)))


def auc(pos, neg):
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(nat, planted, sw, s0, slope):
    mem = np.array([t is not None for t in nat["tense"]])
    exp = np.array([bool(t) for t in nat["tense"]])
    pt = nat["pt"]
    s_nat = np.where(mem, logit(np.where(exp, pt, 1 - pt)), np.nan)
    z_nat = np.where(mem, slope * (s_nat - s0), np.inf) if slope else np.full(len(pt), np.inf)
    pn = shown(nat["zl"], nat["zs"], z_nat)
    t5 = np.quantile(pn, 0.05)
    res = {"native_accept": float(np.mean(pn >= 0.5)), "planted_auc": auc(1 - planted, 1 - pn), "dirs": {}}
    for d, idx in sw.items():
        flipped = ~exp[idx]                                     # the reference now says the other class
        s_sw = logit(np.where(flipped, pt[idx], 1 - pt[idx]))
        z_sw = slope * (s_sw - s0) if slope else np.full(len(idx), np.inf)
        ps = shown(nat["zl"][idx], nat["zs"][idx], z_sw)
        res["dirs"][d] = {"n": int(len(idx)), "hit5": float(np.mean(ps <= t5)), "auc": auc(1 - ps, 1 - pn)}
    res["min_hit"] = min(v["hit5"] for v in res["dirs"].values())
    return res, s_nat


def main():
    rep = {"train": train()}
    from pronunciada import contrast_heads as CH

    CH._tense_head.cache_clear()
    nat, planted = dev_rows()
    sw = swaps(nat)
    base, s_nat = evaluate(nat, planted, sw, 0.0, None)
    fin = s_nat[np.isfinite(s_nat)]
    rep["n_native_members"] = int(len(fin))
    rep["native_auc_tense_vs_lax"] = auc(nat["pt"][[t is True for t in nat["tense"]]],
                                         nat["pt"][[t is False for t in nat["tense"]]])
    grid = []
    for q in QS:
        s0 = float(np.quantile(fin, q))
        for sl in SLOPES:
            r, _ = evaluate(nat, planted, sw, s0, sl)
            ok = r["native_accept"] >= base["native_accept"] - 0.005 and r["planted_auc"] >= base["planted_auc"] - 0.003
            grid.append({"q": q, "s0": s0, "slope": sl, "ok": bool(ok), **r})
    oks = [g for g in grid if g["ok"]]
    best = max(oks, key=lambda g: (round(g["min_hit"], 3), g["native_accept"]))
    adopt = best["min_hit"] >= 0.6
    # coaching bars: <= budget native items per clip
    zs_nat = best["slope"] * (fin - best["s0"])
    n_clips = len(set(nat["clip"].astype(int)))
    coach = {}
    for s, b in BUDGET.items():
        k = int(b * n_clips)
        coach[s] = float(sig(np.sort(zs_nat)[k])) if k < len(zs_nat) else 1.0
    rep.update(base=base, best=best, adopted=bool(adopt), coach=coach, grid=grid)
    Path("results/tables/round18_tense_dev.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("train", "n_native_members", "native_auc_tense_vs_lax", "base", "adopted", "coach")},
                     indent=1, default=float))
    print("best", {k: best[k] for k in ("q", "slope", "native_accept", "planted_auc", "dirs")})
    if adopt:
        p = Path("pronunciada/data/coach/contrast_heads.json")
        m = json.loads(p.read_text())
        m["langs"]["ko"] = [e for e in m["langs"]["ko"] if e["contrast"] != "tense"] + [
            {"contrast": "tense", "head": "tense", "members": "RefPhone.tense", "s0": best["s0"], "slope": best["slope"],
             "cap": True, "coach": coach}]
        p.write_text(json.dumps(m, indent=1, ensure_ascii=False))


def test():
    """Held-out, once: FLEURS test (round-17 test clips, "after" = round-17 shipped state), tense swaps
    as on dev; before = without the tense cap, after = with the shipped parameters."""
    m = json.loads(Path("pronunciada/data/coach/contrast_heads.json").read_text())
    e = [x for x in m["langs"]["ko"] if x["contrast"] == "tense"][0]
    nat, planted = dev_rows("test")
    sw = swaps(nat)
    before, _ = evaluate(nat, planted, sw, 0.0, None)
    after, _ = evaluate(nat, planted, sw, e["s0"], e["slope"])
    rng = np.random.default_rng(0)
    clips = np.unique(nat["clip"]).astype(int)
    deltas = {d: [] for d in sw}
    for _ in range(300):
        pick = set(rng.choice(clips, len(clips)).tolist())
        keep = np.array([int(c) in pick for c in nat["clip"]])
        sub = {k: (v[keep] if isinstance(v, np.ndarray) else [x for x, kk in zip(v, keep) if kk]) for k, v in nat.items()}
        idx_map = -np.ones(len(keep), int)
        idx_map[keep] = np.arange(keep.sum())
        sw2 = {d: idx_map[i][idx_map[i] >= 0] for d, i in sw.items()}
        b, _ = evaluate(sub, planted, sw2, 0.0, None)
        a, _ = evaluate(sub, planted, sw2, e["s0"], e["slope"])
        for d in sw:
            deltas[d].append(a["dirs"][d]["hit5"] - b["dirs"][d]["hit5"])
    rep = {"before": before, "after": after,
           "delta_ci": {d: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for d, v in deltas.items()}}
    Path("results/tables/round18_tense_test.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    import sys

    test() if "test" in sys.argv[1:] else main()
