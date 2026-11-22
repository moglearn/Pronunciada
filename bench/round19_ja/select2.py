"""Round 19, second structure: choose the word factors on dev, with the dev guards.

    python -m bench.round19_ja.select2 prep --length NAME     # per-word evidence -> <work>/sel2_<NAME>.pkl
    python -m bench.round19_ja.select2 grid --length NAME     # grid + guards -> <work>/sel2_grid_<NAME>.json

Word probability = round-18 word probability with fixes 1-2 (merge, labels), its round-15 length
factor replaced by the factor of the chosen length model, times the identity factor and the match
factor (pronunciada/word_factors.py). Every factor is <= 1.

Dev metrics: the round-19 dev set (benchmark-style units: AUC, FA@80, by bucket).
Dev guards (each against round 18 = PRONUNCIADA_JA_ROUND19=0 dumps of the same clips):
* FLEURS validation natives (dev pool): words < 0.5 may rise <= 0.3 pt, words < 0.6 must not rise;
* Common Voice dev natives (dev pool): words < 0.5 and < 0.6 must not rise;
* frozen FLEURS-swap dev set (bench/calib/bias_grid.py): far and near AUC drop <= 0.003, far hit@5 % drop <= 2 pt;
* cv-real ja Test A DEV half: word mean / word min / sentence AUC drop <= 0.005.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys

import numpy as np

from bench.round19_ja.cv_pool import WORK


def _knee_at(cal, s):
    k = cal["knee"]
    s = min(max(float(s), 0.0), 1.0)
    return k[0] + (k[1] - k[0]) * 2 * s if s <= 0.5 else k[1] + (k[2] - k[1]) * (2 * s - 1)


def factor(cal, zmin, s):
    """One-sided knee factor (the round-15 / round-18 form): 1 above the knee, linear to the floor."""
    if zmin is None or not math.isfinite(zmin):
        return 1.0
    t = min(max((_knee_at(cal, s) - zmin) / cal["width"], 0.0), 1.0)
    return 1.0 - (1.0 - cal["floor"]) * t


def match_logit(cal, mm):
    n = max(mm["n"], 1)
    x = np.array([mm["ins"], mm["del"], mm["sub"], (mm["ins"] + mm["del"] + mm["sub"]) / n, math.log(n)])
    return float(cal["intercept"] + x @ np.asarray(cal["coef"]))


def _logit(p, lo=1e-6):
    p = min(max(float(p), lo), 1 - lo)
    return math.log(p / (1 - p))


def length_probs(it, name):
    from bench.round19_ja.length_fit2 import rows_of
    from pronunciada import length as L

    if name == "shipped":
        return L.score(it["ref"], it["inp"], L._load("ja"), head_s=it.get("hs"))["prob"]
    if name == "r19":
        par = {**L._load("ja"), **{k: v for k, v in L._load("ja_r19").items() if k in ("model", "word", "coach")},
               "r19": True}
        inp = dict(it["inp"], lpa=it["lpa"].astype(np.float32), fb=it["fb"].astype(np.float32),
                   slot_se=it["slot_se"].astype(np.int64))
        return L.score(it["ref"], inp, par, head_s=it.get("hs"))["prob"]
    M = json.loads((WORK / f"len2_{name}.json").read_text())
    idx, cls, rows, lab = rows_of(it)
    pr = [None] * len(it["ref"].phones)
    for c in ("v", "c"):
        sel = [k for k, x in enumerate(cls) if x == c]
        if sel:
            for k, p in zip(sel, L.p_long(M, c, [rows[k] for k in sel])):
                i = idx[k]
                pr[i] = float(p if lab[i] == "long" else 1 - p)
    return pr


def prep(name: str) -> None:
    from bench.round19_ja.collect import load
    from pronunciada.ja_identity import evidence, head

    H = head()
    out = {}
    for which in ("dev", "devnat", "devswap", "cvdev"):
        recs = []
        from pronunciada.ja_identity import classes as _classes

        for it in load(which, "v2"):
            cl = _classes(it["ref"])
            pr = length_probs(it, name)
            s_id = evidence(it["ref"], it["V"].astype(np.float32), H) if H is not None and it.get("V") is not None \
                else np.full(len(it["ref"].phones), np.nan)
            words = []
            for k, w in enumerate(it["ref"].words):
                lv = [pr[i] for i in range(*w.phones) if pr[i] is not None]
                iv = [float(x) for x in s_id[w.phones[0]:w.phones[1]] if np.isfinite(x)]
                ic = [(cl[i], "final" if i == w.phones[1] - 1 else "first" if i == w.phones[0] else "mid")
                      for i in range(*w.phones) if np.isfinite(s_id[i])]
                base = it["wprob"][k] / it["wlen"][k] if it["wlen"][k] > 0 else it["wprob"][k]
                words.append({"text": w.text, "span": it["spans"][k], "base": float(base),
                              "zlen": min(_logit(x) for x in lv) if lv else None,
                              "zid": min(iv) if iv else None, "mm": it["mm"][k],
                              "lens": [_logit(x) for x in lv], "sids": iv, "scls": ic,
                              "n_ph": w.phones[1] - w.phones[0]})
            recs.append({k: it.get(k) for k in ("id", "text", "units", "label", "group", "source", "labels", "side")}
                        | {"words": words})
        out[which] = recs
        print(which, len(recs), file=sys.stderr, flush=True)
    (WORK / f"sel2_{name}.pkl").write_bytes(pickle.dumps(out))


def fit_match() -> dict:
    """Logistic on the free-decode counts, planted fit set (collected round-19 v1, merge tag)."""
    from sklearn.linear_model import LogisticRegression

    from bench.round19_ja.collect import load
    from bench.round19_ja.wordcal import word_labels

    X, y = [], []
    for it in load("fitset", "merge"):
        for k, m in enumerate(it["mm"]):
            n = max(m["n"], 1)
            X.append([m["ins"], m["del"], m["sub"], (m["ins"] + m["del"] + m["sub"]) / n, math.log(n)])
        y += word_labels(it)
    X, y = np.array(X, float), np.array(y)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    cal = {"coef": (-lr.coef_[0]).tolist(), "intercept": float(-lr.intercept_[0])}   # logit "matches"
    (WORK / "sel2_match.json").write_text(json.dumps(cal))
    return cal


def apply(rec_list, cfg, match_cal, s=0.5):
    """Dump-like records with the factored word prob."""
    out = []
    for r in rec_list:
        ws = []
        for w in r["words"]:
            p = w["base"]
            if cfg.get("length"):
                p *= factor(cfg["length"], w["zlen"], s)
            if cfg.get("identity"):
                p *= factor(cfg["identity"], w["zid"], s)
            if cfg.get("match"):
                p *= factor(cfg["match"], match_logit(match_cal, w["mm"]), s)
            ws.append({"text": w["text"], "span": w["span"], "prob": p})
        out.append({**{k: v for k, v in r.items() if k != "words"}, "words": ws})
    return out


def knees(vals, q):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    return [float(np.quantile(v, q / 2)), float(np.quantile(v, q)), float(np.quantile(v, min(2 * q, 0.5)))]


def metrics(D, cfg, match_cal):
    from bench.eval_swap import evaluate
    from bench.metrics import roc_auc
    from bench.round19_ja.unit_eval import fa_at, roc_auc as auc2, units_of

    m = {}
    dev = apply(D["dev"], cfg, match_cal)
    for src in ("all", "cv"):
        rows = units_of([r for r in dev if src == "all" or r["source"] == src])
        y, sc = [x[2] for x in rows], [x[3] for x in rows]
        m[f"dev_auc_{src}"] = auc2(y, sc)
        m[f"dev_fa80_{src}"] = fa_at(y, sc)
        if src == "all":
            neg = [x[3] for x in rows if not x[2]]
            for b in ("length", "sub_subtle", "sub_moderate", "sub_gross", "indel"):
                P = [x[3] for x in rows if x[2] and x[4] == b]
                m[f"dev_{b}"] = auc2([1] * len(P) + [0] * len(neg), P + neg)
    nat = apply(D["devnat"], cfg, match_cal)
    for src in ("fleurs", "cv"):
        p = np.array([w["prob"] for r in nat if r["source"] == src for w in r["words"]])
        m[f"nat_{src}_lt05"], m[f"nat_{src}_lt06"] = float(np.mean(p < 0.5)), float(np.mean(p < 0.6))
    sw = apply(D["devswap"], cfg, match_cal)
    recs = []
    outs = {}
    for r in sw:
        recs.append({"id": r["id"], "lang": "ja", "text": r["text"], "labels": r["labels"]})
        outs[r["id"]] = {"status": "Success", "words": [(w["span"][0], w["span"][1], w["prob"]) for w in r["words"]]}
    ev = evaluate(recs, outs, n_boot=0)["ja"]["metrics"]
    for k in ("swap_far:auc", "swap_far:hit@5%FA", "swap_near:auc", "swap_near:hit@5%FA"):
        m[k] = ev[k]["value"]
    cv = apply(D["cvdev"], cfg, match_cal)
    y = np.array([1 if r["side"] == "rejected" else 0 for r in cv])
    m["cvdev_word_mean"] = roc_auc(y, -np.array([np.mean([w["prob"] for w in r["words"]]) for r in cv]))
    m["cvdev_word_min"] = roc_auc(y, -np.array([np.min([w["prob"] for w in r["words"]]) for r in cv]))
    return m


def guard_ok(m, b):
    return (m["nat_fleurs_lt05"] - b["nat_fleurs_lt05"] <= 0.003 and m["nat_fleurs_lt06"] <= b["nat_fleurs_lt06"] + 1e-9
            and m["nat_cv_lt05"] <= b["nat_cv_lt05"] + 1e-9 and m["nat_cv_lt06"] <= b["nat_cv_lt06"] + 1e-9
            and b["swap_far:auc"] - m["swap_far:auc"] <= 0.003 and b["swap_near:auc"] - m["swap_near:auc"] <= 0.003
            and b["swap_far:hit@5%FA"] - m["swap_far:hit@5%FA"] <= 0.02
            and b["cvdev_word_mean"] - m["cvdev_word_mean"] <= 0.005 and b["cvdev_word_min"] - m["cvdev_word_min"] <= 0.005)


def round18_baseline():
    """Guard metrics of round 18 from its own dumps (PRONUNCIADA_JA_ROUND19=0)."""
    D = {}
    for which, f in (("dev", "devF_base"), ("devnat", "devnat_base"), ("devswap", "devswap_base"), ("cvdev", "cvdev_base")):
        recs = [json.loads(x) for x in open(WORK / f"{f}.jsonl", encoding="utf-8")]
        D[which] = [{**r, "words": [{"text": w["text"], "span": w["span"], "base": w["prob"], "zlen": None, "zid": None,
                                     "mm": None} for w in r["words"]]} for r in recs if not r.get("error")]
    return metrics(D, {}, None)


def grid(name: str) -> None:
    D = pickle.loads((WORK / f"sel2_{name}.pkl").read_bytes())
    mc = json.loads((WORK / "sel2_match.json").read_text())
    b18 = round18_baseline()
    print("round18", json.dumps(b18), flush=True)
    nat_words = [w for r in D["devnat"] for w in r["words"]]
    zl = [w["zlen"] for w in nat_words]
    zi = [w["zid"] for w in nat_words]
    zm = [match_logit(mc, w["mm"]) for w in nat_words]
    fix = metrics(D, {"length": {"knee": [-3.3068, -2.0580, -0.9831], "width": 4.0, "floor": 0.4}} if name == "shipped"
                  else {}, mc)
    print("fixes1-2", json.dumps(fix), flush=True)
    res = {"round18": b18, "fixes12": fix, "grid": []}
    QS = (0.005, 0.01, 0.02, 0.05)
    for lq in QS:
        for lf in (0.4, 0.1, 0.02):
            for iq in (None, 0.005, 0.01, 0.02):
                for mq in (None, 0.005, 0.01, 0.02):
                    for width in (2.0, 4.0):
                        cfg = {"length": {"knee": knees(zl, lq), "width": width, "floor": lf}}
                        if iq:
                            cfg["identity"] = {"knee": knees(zi, iq), "width": width, "floor": 0.3}
                        if mq:
                            cfg["match"] = {"knee": knees(zm, mq), "width": width, "floor": 0.3}
                        m = metrics(D, cfg, mc)
                        ok = guard_ok(m, b18)
                        res["grid"].append({"cfg": cfg, "q": [lq, lf, iq, mq, width], "m": m, "ok": ok})
                        print(f"L q={lq} f={lf} I={iq} M={mq} w={width} ok={ok} AUC {m['dev_auc_all']:.4f} "
                              f"cv {m['dev_auc_cv']:.4f} FA80 {m['dev_fa80_all']:.3f} len {m['dev_length']:.3f} "
                              f"far {m['swap_far:auc']:.4f} natF {m['nat_fleurs_lt05']:.4f}/{m['nat_fleurs_lt06']:.4f} "
                              f"natC {m['nat_cv_lt05']:.4f}/{m['nat_cv_lt06']:.4f} cvd {m['cvdev_word_mean']:.3f}/"
                              f"{m['cvdev_word_min']:.3f}", flush=True)
    (WORK / f"sel2_grid_{name}.json").write_text(json.dumps(res, indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "grid", "match"))
    ap.add_argument("--length", default="shipped")
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        prep(a.length)
    elif a.cmd == "match":
        print(fit_match())
    else:
        grid(a.length)


if __name__ == "__main__":
    main()
