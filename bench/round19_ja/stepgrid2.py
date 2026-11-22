"""Round 19, part 3: extend candidate 2's step with the new evidence, by coordinate search on dev under
the dev guards (the log-loss fit of bench/round19_ja/stepfit.py ranked worse than candidate 2).

    python -m bench.round19_ja.stepgrid2

step = min(cap, c0 + a_id*min(s - k_id, 0) + a_len*min(z_len - k_len, 0) - a_mm*extra      (candidate 2)
             + a_long*min(z_long - k_long, 0)          extra weight on expected-LONG vowels said short
             + a_ins*min(ins - k_ins, 0)                extra-sound evidence (llr_ins, inside / before the word)
             + a_del*min(del - k_del, 0)                missing-sound evidence (lowest llr_del)
             - a_rat*max(|word_ratio| - k_rat, 0))      word duration against its mora count
Search: start at candidate 2 with the new weights 0; for each new term in turn, the grid value with
the highest dev unit AUC among settings that pass every dev guard (natives first). Dev length items
are also split by direction (text long / audio short and the reverse).
"""
from __future__ import annotations

import json
import math
import pickle

import numpy as np

from bench.round19_ja.cv_pool import WORK


def _lg(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def load(which):
    return pickle.loads((WORK / "stepfeat" / f"{which}.pkl").read_bytes())


def word_ev(w, P):
    """Evidence of one word for the step (candidate-2 inputs + new ones)."""
    off, g = P["offsets"], P["offset_global"]
    f = w["f"]
    ss = [s["s"] - off.get(f"{s['cls']}|{s['pos']}", g) for s in f["sounds"] if s["s"] is not None]
    lz = [_lg(s["lp"]) for s in f["sounds"] if "lp" in s]
    lzl = [_lg(s["lp"]) for s in f["sounds"] if "lp" in s and s.get("lcls") == "v" and s["lexp"] == "long"]
    ins = [x for x in (f["ins_gap"], f["ins_in"]) if x is not None]
    return {"s": min(ss) if ss else None, "zlen": min(lz) if lz else None, "zlong": min(lzl) if lzl else None,
            "extra": f["extra"], "ins": min(ins) if ins else 0.0, "del": f["del_min"], "ratio": f["word_ratio"]}


def step(cfg, e):
    u = cfg["c0"] - cfg["a_mm"] * e["extra"]
    if e["s"] is not None:
        u += cfg["a_id"] * min(e["s"] - cfg["k_id"], 0.0)
    if e["zlen"] is not None:
        u += cfg["a_len"] * min(e["zlen"] - cfg["k_len"], 0.0)
    if e["zlong"] is not None:
        u += cfg.get("a_long", 0) * min(e["zlong"] - cfg.get("k_long", 0), 0.0)
    u += cfg.get("a_ins", 0) * min(e["ins"] - cfg.get("k_ins", 0), 0.0)
    u += cfg.get("a_del", 0) * min(e["del"] - cfg.get("k_del", 0), 0.0)
    u -= cfg.get("a_rat", 0) * max(abs(e["ratio"]) - cfg.get("k_rat", 0.3), 0.0)
    return min(u, cfg["cap"])


def apply(recs, cfg, P, EV):
    out = []
    for r in recs:
        ws = []
        for w in r["words"]:
            e = EV[id(w)]
            ws.append({"text": w["text"], "span": w["span"], "base": 1 / (1 + math.exp(-(_lg(w["base"]) + step(cfg, e)))),
                       "zlen": None, "zid": None, "mm": None})
        out.append({**{k: v for k, v in r.items() if k != "words"}, "words": ws})
    return out


def length_dirs(recs):
    """Dev length units by direction: AUC of planted length units against control units."""
    from bench.round19_ja.unit_eval import roc_auc, units_of

    rows = units_of(recs, prob_fn=lambda w: w["base"])
    neg = [x[3] for x in rows if not x[2]]
    out = {}
    for d in ("text_long", "text_short"):
        pos = []
        for x in rows:
            if x[2] and x[4] == "length":
                r = next(rr for rr in recs if rr["id"] == x[0])
                lab = r["label"]
                o, n = lab["orig_phones"], lab["new_phones"]
                longer_text = (n.count("ː") + sum(len(t) > 1 and t[0] == t[1] for t in n.split())) > \
                              (o.count("ː") + sum(len(t) > 1 and t[0] == t[1] for t in o.split()))
                if (d == "text_long") == longer_text:
                    pos.append(x[3])
        out[d] = (len(pos), roc_auc([1] * len(pos) + [0] * len(neg), pos + neg))
    return out


def main():
    from bench.round19_ja.select2 import guard_ok, metrics, round18_baseline

    P = json.loads(open(Path_ws()).read())
    D = {k: load(k) for k in ("dev", "devnat", "devswap", "cvdev")}
    EV = {id(w): word_ev(w, P) for k in D for r in D[k] for w in r["words"]}
    b18 = round18_baseline()
    cfg = dict(P["cfg"])
    base_m = metrics({k: apply(v, cfg, P, EV) for k, v in D.items()}, {}, None)
    print("candidate-2 step reproduced:", round(base_m["dev_auc_all"], 4), "guards", guard_ok(base_m, b18), flush=True)
    print("  length by direction", length_dirs(apply(D["dev"], cfg, P, EV)), flush=True)
    log = [{"cfg": dict(cfg), "m": base_m}]
    grids = [("a_long", (0.0, 0.5, 1.0, 2.0), "k_long", (-1.0, 0.0, 1.0)),
             ("a_ins", (0.0, 0.1, 0.25, 0.5), "k_ins", (-4.0, -2.0, 0.0)),
             ("a_del", (0.0, 0.1, 0.25, 0.5), "k_del", (-4.0, -2.0, 0.0)),
             ("a_rat", (0.0, 1.0, 2.0, 4.0), "k_rat", (0.2, 0.4, 0.6))]
    best = base_m
    for a_name, a_vals, k_name, k_vals in grids:
        cands = []
        for av in a_vals:
            for kv in (k_vals if av else (k_vals[0],)):
                c = dict(cfg, **{a_name: av, k_name: kv})
                m = metrics({k: apply(v, c, P, EV) for k, v in D.items()}, {}, None)
                ok = guard_ok(m, b18)
                cands.append((ok, m["dev_auc_all"], -m["dev_fa80_all"], c, m))
                print(f"{a_name}={av} {k_name}={kv} ok={ok} AUC {m['dev_auc_all']:.4f} FA80 {m['dev_fa80_all']:.3f} "
                      f"len {m['dev_length']:.3f} indel {m['dev_indel']:.3f} natF {m['nat_fleurs_lt06']:.4f} "
                      f"natC {m['nat_cv_lt06']:.4f} far {m['swap_far:auc']:.4f} cvd {m['cvdev_word_mean']:.3f}/{m['cvdev_word_min']:.3f}",
                      flush=True)
        okc = [x for x in cands if x[0]]
        pick = max(okc, key=lambda x: (x[1], x[2]))
        cfg, best = pick[3], pick[4]
        log.append({"cfg": dict(cfg), "m": best})
    print("final", cfg, round(best["dev_auc_all"], 4), round(best["dev_fa80_all"], 4))
    print("  length by direction", length_dirs(apply(D["dev"], cfg, P, EV)))
    (WORK / "stepgrid2.json").write_text(json.dumps({"log": log, "final": cfg, "round18": b18}, indent=1, default=float))


def compare():
    """Paired dev comparison of candidate 2 and the extended step, overall and per speaker fold."""
    from bench.round19_ja.step import fold_of
    from bench.round19_ja.unit_eval import paired, roc_auc, units_of

    P = json.loads(open(Path_ws()).read())
    D = load("dev")
    EV = {id(w): word_ev(w, P) for r in D for w in r["words"]}
    fin = json.loads((WORK / "stepgrid2.json").read_text())["final"]
    A = units_of(apply(D, dict(P["cfg"]), P, EV), prob_fn=lambda w: w["base"])
    B = units_of(apply(D, fin, P, EV), prob_fn=lambda w: w["base"])
    print("paired (extended - candidate 2):", json.dumps(paired(A, B, 2000)))
    for f in (0, 1):
        ia = [x for x in A if fold_of(x[1]) == f]
        ib = [x for x in B if fold_of(x[1]) == f]
        print(f"fold {f}: AUC {roc_auc([x[2] for x in ia], [x[3] for x in ia]):.4f} -> "
              f"{roc_auc([x[2] for x in ib], [x[3] for x in ib]):.4f}")


def Path_ws():
    from pathlib import Path

    return Path(__file__).resolve().parents[2] / "pronunciada" / "data" / "calib" / "word_step" / "ja.json"


if __name__ == "__main__":
    main()
