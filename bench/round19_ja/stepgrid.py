"""Round 19: the capped two-sided step chosen by grid on dev unit AUC (2-fold by speaker), then the
dev guards for the best settings.

    python -m bench.round19_ja.stepgrid --length r19

step(word) = min(cap, c0 + a_id * min(s_min - k_id, 0) + a_len * min(z_len - k_len, 0) - a_mm * extra_sounds)
on the round-18 word log-odds (fixes 1-2, without the round-15 length factor); s_min = the word's
lowest identity margin, z_len = its lowest length logit (length model given by --length).
Selection: highest mean of the two folds' dev AUC (each fold's setting chosen on the other fold is
also reported); then the guards of select2 for the top settings; the frozen setting is the best one
that passes every guard.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle

import numpy as np

from bench.round19_ja.cv_pool import WORK
from bench.round19_ja.step import fold_of


def unit_index(recs):
    """For every unit of every dev record: (word indices into the flat word list, label, fold, src)."""
    from bench.round19_ja.unit_eval import _PUNCT

    flat, units = [], []
    for r in recs:
        base = len(flat)
        flat += r["words"]
        text = r["text"]
        spans, cur = [], 0
        for k, w in enumerate(r["words"]):
            t = _PUNCT.sub("", w["text"])
            j = text.find(t, cur) if t else -1
            if j >= 0:
                spans.append((j, j + len(t), base + k))
                cur = j + len(t)
        lab = r.get("label") or {}
        pos = lab.get("word_index") if lab else None
        f = fold_of(r.get("group") or r["id"])
        for k, u in enumerate(r["units"]):
            ws = [i for s, e, i in spans if s < u["end"] and e > u["start"]]
            units.append((ws, int(k == pos), f, r.get("source"), lab.get("bucket") if k == pos else "control"))
    return flat, units


OFFSETS: dict = {}


def offsets(natives, q):
    """Per (class, position) native q-quantile of the identity margin (dev-pool natives)."""
    from collections import defaultdict

    v = defaultdict(list)
    for w in natives:
        for x, c in zip(w["sids"], w["scls"]):
            v[tuple(c)].append(x)
    allv = [x for xs in v.values() for x in xs]
    g = float(np.quantile(allv, q))
    return {k: (float(np.quantile(xs, q)) if len(xs) >= 200 else g) for k, xs in v.items()}, g


def smin(w):
    if not w["sids"]:
        return 99.0
    off, g = OFFSETS["off"], OFFSETS["g"]
    return min(x - off.get(tuple(c), g) for x, c in zip(w["sids"], w["scls"]))


def arrays(flat):
    z = np.array([math.log(max(w["base"], 1e-6) / max(1 - w["base"], 1e-6)) for w in flat])
    s = np.array([smin(w) for w in flat])
    L = np.array([min(w["lens"]) if w["lens"] else 99.0 for w in flat])
    ins = np.array([w["mm"]["ins"] for w in flat], float)
    return z, s, L, ins


def step_of(cfg, s, L, ins):
    c0, a_id, k_id, a_len, k_len, a_mm, cap = cfg
    u = c0 + a_id * np.minimum(s - k_id, 0) + a_len * np.minimum(L - k_len, 0) - a_mm * ins
    return np.minimum(u, cap)


def unit_auc(zw, units, fold=None):
    from bench.round19_ja.unit_eval import roc_auc

    y, sc = [], []
    for ws, lab, f, _, _ in units:
        if fold is not None and f != fold:
            continue
        y.append(lab)
        sc.append(1.0 if not ws else 1.0 - min(1 / (1 + math.exp(-zw[i])) for i in ws))
    return roc_auc(y, sc)


GRID = {"c0": (2.0, 3.0), "a_id": (0.0, 0.5, 1.0, 1.5, 2.0), "k_id": (0.0, 1.0, 2.0, 3.0, 4.0),
        "a_len": (0.0, 0.5, 1.0, 2.0), "k_len": (-2.0, -1.0, 0.0), "a_mm": (0.0, 0.5), "cap": (3.0,)}


def main(argv=None):
    from bench.round19_ja.select2 import guard_ok, metrics, round18_baseline

    ap = argparse.ArgumentParser()
    ap.add_argument("--length", default="r19")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--offq", type=float, default=0.0, help="per class / position native quantile offset (0 = none)")
    a = ap.parse_args(argv)
    D = pickle.loads((WORK / f"sel2_{a.length}.pkl").read_bytes())
    if a.offq:
        OFFSETS["off"], OFFSETS["g"] = offsets([w for r in D["devnat"] for w in r["words"]], a.offq)
    else:
        OFFSETS["off"], OFFSETS["g"] = {}, 0.0
    flat, units = unit_index(D["dev"])
    z, s, L, ins = arrays(flat)
    b18 = round18_baseline()
    nflat = [w for r in D["devnat"] for w in r["words"]]
    nsrc = np.array([r["source"] for r in D["devnat"] for _ in r["words"]])
    nz, ns, nL, nins = arrays(nflat)

    def natives_ok(cfg):
        p = 1 / (1 + np.exp(-(nz + step_of(cfg, ns, nL, nins))))
        r = {f"nat_{src}_lt0{t}": float(np.mean(p[nsrc == src] < t / 10)) for src in ("fleurs", "cv") for t in (5, 6)}
        return (r["nat_fleurs_lt05"] - b18["nat_fleurs_lt05"] <= 0.003 and r["nat_fleurs_lt06"] <= b18["nat_fleurs_lt06"]
                and r["nat_cv_lt05"] <= b18["nat_cv_lt05"] and r["nat_cv_lt06"] <= b18["nat_cv_lt06"])

    res = []
    for cfg in itertools.product(*GRID.values()):
        if (cfg[1] == 0 and cfg[2] != 0.0) or (cfg[3] == 0 and cfg[4] != -1.0):
            continue
        if not natives_ok(cfg):          # native guard first (cheap); the other guards below
            continue
        zw = z + step_of(cfg, s, L, ins)
        res.append((cfg, unit_auc(zw, units, 0), unit_auc(zw, units, 1)))
    print(len(res), "settings pass the native guard", flush=True)
    best0 = max(res, key=lambda x: x[1])
    best1 = max(res, key=lambda x: x[2])
    cross = {"fold1_with_fold0_choice": best0[2], "fold0_with_fold1_choice": best1[1],
             "choices": [best0[0], best1[0]]}
    print("cross-fitted dev AUC:", json.dumps(cross))
    res.sort(key=lambda x: -(x[1] + x[2]))
    out = {"cross": cross, "round18": b18, "top": []}
    for cfg, a0, a1 in res[:a.top]:
        E = {}
        for k in ("dev", "devnat", "devswap", "cvdev"):
            rr = []
            for r in D[k]:
                ws = []
                for w in r["words"]:
                    zz = math.log(max(w["base"], 1e-6) / max(1 - w["base"], 1e-6))
                    st = float(step_of(cfg, np.array([smin(w)]),
                                       np.array([min(w["lens"]) if w["lens"] else 99.0]),
                                       np.array([w["mm"]["ins"]], float))[0])
                    ws.append({"text": w["text"], "span": w["span"], "base": 1 / (1 + math.exp(-(zz + st))),
                               "zlen": None, "zid": None, "mm": None})
                rr.append({**{x: v for x, v in r.items() if x != "words"}, "words": ws})
            E[k] = rr
        m = metrics(E, {}, None)
        ok = guard_ok(m, b18)
        out["top"].append({"cfg": dict(zip(GRID, cfg)), "auc_folds": [a0, a1], "m": m, "ok": ok})
        print(f"{dict(zip(GRID, cfg))} folds {a0:.4f}/{a1:.4f} ok={ok} AUC {m['dev_auc_all']:.4f} FA80 {m['dev_fa80_all']:.3f} "
              f"len {m['dev_length']:.3f} sub {m['dev_sub_subtle']:.3f}/{m['dev_sub_moderate']:.3f}/{m['dev_sub_gross']:.3f} "
              f"indel {m['dev_indel']:.3f} far {m['swap_far:auc']:.4f} near {m['swap_near:auc']:.4f} "
              f"natF {m['nat_fleurs_lt05']:.4f}/{m['nat_fleurs_lt06']:.4f} natC {m['nat_cv_lt05']:.4f}/"
              f"{m['nat_cv_lt06']:.4f} cvd {m['cvdev_word_mean']:.3f}/{m['cvdev_word_min']:.3f}", flush=True)
    out["offsets"] = {"q": a.offq, "by_class": {"|".join(k): v for k, v in OFFSETS["off"].items()}, "global": OFFSETS["g"]}
    (WORK / f"stepgrid_{a.length}_q{a.offq}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
