"""Round 17: head-based contrast evidence (dev = FLEURS validation, the audit's clips and swaps).

    python -m bench.round17.fit dev            # caps, coaching thresholds, ja length test -> model files
    python -m bench.round17.fit test           # held-out, once (FLEURS test chunks of bench/round17/collect)

Evidence. For a sound whose expected symbol is a member of the contrast, the attribute head's
probability of the expected class (ko / zh aspiration: head "aspirated"; ru hard / soft: head
"palatalised"; ja length: head "long"): s = logit(P). A cap on the shown sound log-odds
    z_cap = slope * (s - s0),   s0 = the q-quantile of s over natives' member sounds,
so natives sit above 0 except their lowest q. Shown zl, zs := min(zl, z_cap), min(zs, z_cap)
(composed with round 16 by the minimum).
Rule (fixed before looking at test): grid q x slope; keep configs within the dev guards (native
sound acceptance drop <= 0.5 pt over all native sounds of the audit clips, planted-substitution
sound AUC drop <= 0.003); pick the config with the best lower-direction hit@5 % (native
false-alarm threshold re-derived after the cap), ties -> higher native acceptance. Adopted when
the lower-direction hit improves by >= 10 points.
Coaching. A member sound with sigmoid(z_cap) below a per-sensitivity bar gets the head's feature
claim (and an item if not already flagged); bar = the value giving at most 0.05 / 0.1 / 0.2
native items per clip (careful / balanced / eager). Mandarin: checked on OMPAL dev initials.
Japanese length: the head's logit as an extra input to the length channel's decision, adopted
only if the dev hit@5 % (native length false alarms) rises by >= 3 points (5-fold by clip).
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.round17.collect import CONTRASTS, load
from pronunciada import attr_heads as AH

HEAD = {("ko", "aspiration"): "aspirated", ("zh", "aspiration"): "aspirated", ("ru", "hard_soft"): "palatalised",
        ("ja", "vowel_length"): "long", ("ja", "gemination"): "long"}
QS = (0.002, 0.005, 0.01, 0.02, 0.05)
SLOPES = (0.5, 1.0, 2.0, 4.0)
COACH_BUDGET = {"careful": 0.05, "balanced": 0.1, "eager": 0.2}


def members(lang, cname):
    from bench.audit.contrasts import get_contrast

    c = get_contrast(lang, cname)
    return sorted({x for p in c.pairs for x in p})


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def prepare(lang, split, tag="base"):
    """Flat arrays: natives (all sounds) and edits / planted (flagged sounds), with head evidence."""
    heads = AH.shipped()
    C = load(lang, split, tag)
    nat = defaultdict(list)
    ed = defaultdict(list)
    pl = defaultdict(list)
    for k, clip in enumerate(C):
        n = clip["native"]
        V = n["vec"].astype(np.float32)
        ctx = AH.context_vectors(V)
        P = heads.predict(V, range(len(V)), ctx)
        for i in range(len(V)):
            nat["clip"].append(k)
            for f in ("ipa", "zl", "zs", "prob", "length"):
                nat[f].append(n[f][i])
            nat["P"].append(P[i])
        for e in clip["edits"]:
            Ve = e["vec"].astype(np.float32)
            Pe = heads.predict(Ve, [0], ctx[[min(e["index"], len(V) - 1)]])[0]
            ed["clip"].append(k)
            for f in ("contrast", "direction", "src", "dst", "alt", "word_prob", "nat_word_prob"):
                ed[f].append(e[f])
            ed["ipa"].append(e["ipa"][0])
            ed["zl"].append(e["zl"][0])
            ed["zs"].append(e["zs"][0])
            ed["prob"].append(e["prob"][0])
            ed["length"].append(e["length"][0])
            ed["P"].append(Pe)
        for e in clip["planted"]:
            Vp = e["vec"].astype(np.float32)
            for q in range(len(e["ipa"])):
                Pp = heads.predict(Vp, [q], Vp)[0]       # context: own vector (neighbours not stored)
                pl["clip"].append(k)
                for f in ("ipa", "zl", "zs", "prob"):
                    pl[f].append(e[f][q])
                pl["P"].append(Pp)
    out = {}
    for name, d in (("nat", nat), ("ed", ed), ("pl", pl)):
        out[name] = {f: (np.asarray(v) if f in ("zl", "zs", "prob", "clip", "word_prob", "nat_word_prob") else v)
                     for f, v in d.items()}
    out["n_clips"] = len(C)
    return out


def evidence(D, h, mem):
    """s = logit P(expected class) where the sound is a member, NaN elsewhere."""
    s = np.full(len(D["ipa"]), np.nan)
    for i, (ipa, P) in enumerate(zip(D["ipa"], D["P"])):
        if ipa in mem and h in P:
            lb = AH.labels(ipa).get(h)
            if lb is not None:
                s[i] = logit(P[h].get(lb, 0.0))
    return s


def shown(D, s, s0, slope):
    zl, zs = D["zl"].astype(np.float64), D["zs"].astype(np.float64)
    if slope is None:
        return sig(0.5 * (zl + np.minimum(zs, zl)))
    zc = np.where(np.isfinite(s), slope * (s - s0), np.inf)
    zl2, zs2 = np.minimum(zl, zc), np.minimum(zs, zc)
    return sig(0.5 * (zl2 + np.minimum(zs2, zl2)))


def auc(pos, neg):
    from scipy.stats import rankdata

    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metrics(D, cname, s_nat, s_ed, s_pl, s0, slope):
    pn = shown(D["nat"], s_nat, s0, slope)
    pe = shown(D["ed"], s_ed, s0, slope)
    pp = shown(D["pl"], s_pl, s0, slope)
    t5 = float(np.quantile(pn, 0.05))
    sel = np.array([c == cname for c in D["ed"]["contrast"]])
    dirs = {}
    for d in sorted(set(np.array(D["ed"]["direction"])[sel])):
        m = sel & (np.array(D["ed"]["direction"]) == d)
        dirs[d] = {"n": int(m.sum()), "hit5": float(np.mean(pe[m] <= t5)), "auc": auc(1 - pe[m], 1 - pn)}
    return {"native_accept": float(np.mean(pn >= 0.5)), "planted_auc": auc(1 - pp, 1 - pn), "dirs": dirs,
            "min_hit": min(v["hit5"] for v in dirs.values()) if dirs else None}


def fit_cap(D, lang, cname):
    h = HEAD[(lang, cname)]
    mem = set(members(lang, cname))
    s_nat, s_ed, s_pl = evidence(D["nat"], h, mem), evidence(D["ed"], h, mem), evidence(D["pl"], h, mem)
    base = metrics(D, cname, s_nat, s_ed, s_pl, 0.0, None)
    grid = []
    fin = s_nat[np.isfinite(s_nat)]
    for q in QS:
        s0 = float(np.quantile(fin, q))
        for sl in SLOPES:
            m = metrics(D, cname, s_nat, s_ed, s_pl, s0, sl)
            ok = (m["native_accept"] >= base["native_accept"] - 0.005 and m["planted_auc"] >= base["planted_auc"] - 0.003)
            grid.append({"q": q, "s0": s0, "slope": sl, "ok": bool(ok), **m})
    oks = [g for g in grid if g["ok"]]
    best = max(oks, key=lambda g: (round(g["min_hit"], 3), g["native_accept"])) if oks else None
    adopt = bool(best and best["min_hit"] >= base["min_hit"] + 0.10)
    return {"head": h, "members": sorted(mem), "base": base, "best": best, "adopted": adopt, "grid": grid,
            "n_native_members": int(np.isfinite(s_nat).sum())}


def coach_bars(D, lang, cname, cap):
    """Per sensitivity: the highest bar on sigmoid(z_cap) with <= budget native items per clip."""
    h = HEAD[(lang, cname)]
    mem = set(members(lang, cname))
    s_nat, s_ed = evidence(D["nat"], h, mem), evidence(D["ed"], h, mem)
    pn = sig(cap["slope"] * (s_nat[np.isfinite(s_nat)] - cap["s0"]))
    sel = np.array([c == cname for c in D["ed"]["contrast"]]) & np.isfinite(s_ed)
    pe = sig(cap["slope"] * (s_ed[sel] - cap["s0"]))
    dirs = np.array(D["ed"]["direction"])[sel]
    out = {}
    for sens, b in COACH_BUDGET.items():
        k = int(b * D["n_clips"])
        bar = float(np.sort(pn)[k]) if k < len(pn) else 1.0
        out[sens] = {"bar": bar, "native_per_clip": float(np.sum(pn < bar) / D["n_clips"]),
                     "hit": {d: float(np.mean(pe[dirs == d] < bar)) for d in sorted(set(dirs))}}
    return out


def ja_length(D):
    """Does the head's long-evidence add to the length channel? 5-fold by clip, logistic on
    [logit length_prob, s]; hit@5 % native length false alarms, before / after."""
    from sklearn.linear_model import LogisticRegression

    h = "long"
    res = {}
    for cname in ("vowel_length", "gemination"):
        mem = set(members("ja", cname))
        nat_ok = [i for i, (ipa, lp) in enumerate(zip(D["nat"]["ipa"], D["nat"]["length"])) if ipa in mem and lp is not None]
        ed_ok = [i for i, (c, ipa, lp) in enumerate(zip(D["ed"]["contrast"], D["ed"]["ipa"], D["ed"]["length"]))
                 if c == cname and ipa in mem and lp is not None]
        sn, se = evidence(D["nat"], h, mem), evidence(D["ed"], h, mem)
        Xn = np.column_stack([logit([D["nat"]["length"][i] for i in nat_ok]), sn[nat_ok]])
        Xe = np.column_stack([logit([D["ed"]["length"][i] for i in ed_ok]), se[ed_ok]])
        cn, ce = D["nat"]["clip"][nat_ok], D["ed"]["clip"][ed_ok]
        X = np.vstack([Xn, Xe])
        y = np.r_[np.zeros(len(Xn)), np.ones(len(Xe))]
        cl = np.r_[cn, ce]
        score = np.zeros(len(y))
        for f in range(5):
            te = cl % 5 == f
            m = LogisticRegression().fit(X[~te], y[~te])
            score[te] = m.decision_function(X[te])
        def hit(sc):
            t = np.quantile(sc[y == 0], 0.95)
            return float(np.mean(sc[y == 1] > t))
        res[cname] = {"n_edits": int(len(Xe)), "n_native": int(len(Xn)), "hit_length_only": hit(-X[:, 0]),
                      "hit_combined": hit(score), "auc_length_only": auc(-Xe[:, 0], -Xn[:, 0]),
                      "auc_combined": auc(score[y == 1], score[y == 0])}
    return res


def step_dev():
    rep = {}
    model = {"version": 1, "fitted_on": "FLEURS validation natives + contrast-audit swaps (bench/round17/fit.py)",
             "langs": {}}
    for lang in ("ko", "ru", "zh", "ja"):
        D = prepare(lang, "validation")
        rep[lang] = {}
        for cname in CONTRASTS[lang]:
            r = fit_cap(D, lang, cname)
            if r["best"] is not None:
                r["coach"] = coach_bars(D, lang, cname, r["best"])
            rep[lang][cname] = {k: v for k, v in r.items() if k != "grid"}
            rep[lang][cname]["grid"] = [{k: v for k, v in g.items() if k != "dirs"} | {"dirs": g["dirs"]} for g in r["grid"]]
            b = r["best"]
            print(lang, cname, "base", r["base"]["dirs"], r["base"]["native_accept"], r["base"]["planted_auc"], flush=True)
            print("   best", b and {k: b[k] for k in ("q", "slope", "native_accept", "planted_auc", "dirs")}, "adopt", r["adopted"], flush=True)
            if b is not None and (lang, cname) in (("ko", "aspiration"), ("ru", "hard_soft"), ("zh", "aspiration")):
                model["langs"].setdefault(lang, []).append(
                    {"contrast": cname, "head": r["head"], "members": r["members"], "s0": b["s0"], "slope": b["slope"],
                     "cap": r["adopted"], "coach": {s: v["bar"] for s, v in r["coach"].items()}})
        if lang == "ja":
            rep["ja"]["length_test"] = ja_length(D)
            print("ja length", rep["ja"]["length_test"], flush=True)
    Path("results/tables/round17_dev.json").write_text(json.dumps(rep, indent=1, default=float))
    out = Path(os.environ.get("R17_MODEL_OUT", "pronunciada/data/coach/contrast_heads.json"))
    out.write_text(json.dumps(model, indent=1))




def step_ja():
    """Fit the Japanese length-channel head combination on all dev items and write it into
    pronunciada/data/length/ja.json ("head"); quantile map onto the timing model's natives."""
    from sklearn.linear_model import LogisticRegression

    D = prepare("ja", "validation")
    par_p = Path("pronunciada/data/length/ja.json")
    par = json.loads(par_p.read_text())
    mem_all = set(members("ja", "vowel_length")) | set(members("ja", "gemination"))
    head = {"members": sorted(mem_all)}
    qs = np.r_[0.0005, 0.001, 0.002, 0.005, np.linspace(0.01, 0.99, 99), 0.995, 0.998, 0.999, 0.9995]
    rep = {}
    for cname, cls in (("vowel_length", "v"), ("gemination", "c")):
        mem = set(members("ja", cname))
        sn, se = evidence(D["nat"], "long", mem), evidence(D["ed"], "long", mem)
        ln = np.array([np.nan if x is None else x for x in D["nat"]["length"]], float)
        le = np.array([np.nan if x is None else x for x in D["ed"]["length"]], float)
        mn = np.isfinite(sn) & np.isfinite(ln)
        me = np.isfinite(se) & np.isfinite(le) & np.array([c == cname for c in D["ed"]["contrast"]])
        Xn = np.column_stack([logit(ln[mn]), sn[mn]])
        Xe = np.column_stack([logit(le[me]), se[me]])
        m = LogisticRegression().fit(np.vstack([Xn, Xe]), np.r_[np.zeros(len(Xn)), np.ones(len(Xe))])
        # orient so that higher = "realised as expected" (like the timing model's logit)
        a, b = -m.coef_[0]
        c0 = -float(m.intercept_[0])
        zn = a * Xn[:, 0] + b * Xn[:, 1] + c0
        head[cls] = {"coef": [float(a), float(b)], "b": c0, "q_new": np.quantile(zn, qs).tolist(),
                     "q_orig": np.quantile(Xn[:, 0], qs).tolist(), "contrast": cname}
        rep[cname] = {"coef": [float(a), float(b)], "n_native": int(len(Xn)), "n_edits": int(len(Xe))}
    par["head"] = head
    par_p.write_text(json.dumps(par, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    {"dev": step_dev, "ja": step_ja}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
