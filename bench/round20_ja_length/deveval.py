"""Dev metrics and guards of the devrun outputs, per grid setting, against current main.

    python -m bench.round20_ja_length.deveval TAG [--boot 1000] [--out results/tables/round20_ja_length_dev.json]

Metrics (bench/round19_ja/select2.metrics, the round-19 dev set and its guards): dev AUC / FA@80
(all, Common Voice part), buckets, natives below 0.5 / 0.6 (FLEURS validation, CV dev), FLEURS-swap
dev far / near, cv-real dev half word mean / min. Plus: the dev length units by direction (text long
/ audio short and text short / audio long), the own error flag on planted length units and on
control units, paired speaker-bootstrap differences of AUC and FA@80 against main, and the length
table on FLEURS validation (sounds and words caught at 5 % native false alarms, by contrast and
direction).
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.round20_ja_length.common import OUT


def load_tag(tag: str, setting: str) -> dict:
    D = {}
    for which in ("dev", "devnat", "devswap", "cvdev"):
        recs = []
        for f in sorted(glob.glob(str(OUT / "devrun" / tag / f"{which}_{setting}.s*.jsonl"))):
            recs += [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()]
        recs = [r for r in recs if not r.get("error")]
        recs.sort(key=lambda r: r["id"])
        D[which] = [{**r, "words": [{"text": w["text"], "span": w["span"], "base": w["prob"], "flag": w.get("flag"),
                                     "zlen": None, "zid": None, "mm": None} for w in r["words"]]} for r in recs]
    return D


def settings(tag: str) -> list[str]:
    s = {Path(f).name.split(".s")[0][len("dev_"):] for f in glob.glob(str(OUT / "devrun" / tag / "dev_*.s*.jsonl"))}
    s.discard("zh")
    return ["main"] + sorted(s - {"main"})


def length_dirs(recs) -> dict:
    """Dev planted length units by direction: AUC against control units, and own-flag catches."""
    from bench.round19_ja.unit_eval import roc_auc, units_of

    rows = units_of(recs, prob_fn=lambda w: w["base"])
    byid = {r["id"]: r for r in recs}
    neg = [x[3] for x in rows if not x[2]]
    out = {}
    for d in ("text_long", "text_short"):
        pos, fl = [], []
        for x in rows:
            if x[2] and x[4] == "length":
                lab = byid[x[0]]["label"]
                o, n = lab["orig_phones"], lab["new_phones"]
                longer = (n.count("ː") + sum(len(t) > 1 and t[0] == t[1] for t in n.split())) > \
                         (o.count("ː") + sum(len(t) > 1 and t[0] == t[1] for t in o.split()))
                if (d == "text_long") == longer:
                    pos.append(x[3])
                    fl.append((not x[6]) or any(w.get("flag") for w in x[6]))
        out[d] = {"n": len(pos), "auc": roc_auc([1] * len(pos) + [0] * len(neg), pos + neg),
                  "flag_caught": int(sum(fl))}
    return out


def flags(recs) -> dict:
    from bench.round19_ja.unit_eval import units_of

    rows = units_of(recs, prob_fn=lambda w: w["base"])
    fl = [(not x[6]) or any(w.get("flag") for w in x[6]) for x in rows]
    return {"length_caught": int(sum(f for f, x in zip(fl, rows) if x[2] and x[4] == "length")),
            "length_n": int(sum(1 for x in rows if x[2] and x[4] == "length")),
            "recall": float(np.mean([f for f, x in zip(fl, rows) if x[2]])),
            "control_flag_rate": float(np.mean([f for f, x in zip(fl, rows) if not x[2]]))}


def lentab(tag: str) -> dict:
    rows = []
    for f in sorted(glob.glob(str(OUT / "devrun" / tag / "lentab.s*.jsonl"))):
        rows += [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()]
    if not rows:
        return {}
    tags = list(rows[0]["nat_len"])
    out = {"n_clips": len(rows)}
    for t in tags:
        nat = np.array([x for r in rows for x in r["nat_len"][t]], float)
        natw = np.array([x for r in rows for x in r["nat_word"][t]], float)
        t5, tw5 = float(np.quantile(nat, 0.05)), float(np.quantile(natw, 0.05))
        E = [e for r in rows for e in r["edits"]]
        o = {"sound_threshold_5": t5, "word_threshold_5": tw5, "native_word_accept_0.5": float(np.mean(natw >= 0.5)),
             "native_word_below_0.6": float(np.mean(natw < 0.6)), "sound": {}, "word": {}}
        for c, d in sorted({(e["contrast"], e["direction"]) for e in E}) + [("all", "")]:
            sel = [e for e in E if c == "all" or (e["contrast"] == c and e["direction"] == d)]
            lv = np.array([e["len"][t] for e in sel if e["len"][t] is not None], float)
            wv = np.array([e["word"][t] for e in sel], float)
            k = c if c == "all" else f"{c}:{d}"
            o["sound"][k] = {"n": int(len(lv)), "hit5": float(np.mean(lv <= t5)) if len(lv) else None}
            o["word"][k] = {"n": int(len(wv)), "hit5": float(np.mean(wv <= tw5)) if len(wv) else None}
        out[t] = o
    return out


def main(argv=None) -> None:
    from bench.round19_ja.select2 import metrics
    from bench.round19_ja.unit_eval import paired, units_of

    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--only", default="")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    S = settings(a.tag)
    if a.only:
        S = ["main"] + [s for s in a.only.split(",") if s != "main"]
    res = {"settings": {}}
    base = load_tag(a.tag, "main")
    ref = metrics(base, {}, None)
    A = units_of(base["dev"], prob_fn=lambda w: w["base"])
    for s in S:
        D = base if s == "main" else load_tag(a.tag, s)
        m = metrics(D, {}, None)
        g = {k: m[k] - ref[k] for k in m}
        o = {"m": m, "length_dirs": length_dirs(D["dev"]), "flags": flags(D["dev"])}
        if s != "main":
            B = units_of(D["dev"], prob_fn=lambda w: w["base"])
            o["paired_vs_main"] = paired(A, B, a.boot) if a.boot else None
            o["guards_vs_main"] = {
                "nat_fleurs_lt05": g["nat_fleurs_lt05"] <= 0.003, "nat_fleurs_lt06": g["nat_fleurs_lt06"] <= 1e-9,
                "nat_cv_lt05": g["nat_cv_lt05"] <= 1e-9, "nat_cv_lt06": g["nat_cv_lt06"] <= 1e-9,
                "swap_far_auc": g["swap_far:auc"] >= -0.003, "swap_near_auc": g["swap_near:auc"] >= -0.003,
                "swap_far_hit": g["swap_far:hit@5%FA"] >= -0.02,
                "cvdev_word_mean": g["cvdev_word_mean"] >= -0.005, "cvdev_word_min": g["cvdev_word_min"] >= -0.005}
            o["guards_ok"] = all(o["guards_vs_main"].values())
        res["settings"][s] = o
        pv = o.get("paired_vs_main") or {}
        print(f"{s:16s} AUC {m['dev_auc_all']:.4f} FA80 {m['dev_fa80_all']:.4f} len {m['dev_length']:.4f} "
              f"dirs L {o['length_dirs']['text_long']['auc']:.3f}/S {o['length_dirs']['text_short']['auc']:.3f} "
              f"flagL {o['flags']['length_caught']}/{o['flags']['length_n']} ctl {o['flags']['control_flag_rate']:.3f} "
              f"natF {m['nat_fleurs_lt05']:.4f}/{m['nat_fleurs_lt06']:.4f} natC {m['nat_cv_lt05']:.4f}/{m['nat_cv_lt06']:.4f} "
              f"far {m['swap_far:auc']:.4f} near {m['swap_near:auc']:.4f} cvd {m['cvdev_word_mean']:.3f}/{m['cvdev_word_min']:.3f} "
              f"ok {o.get('guards_ok', '-')} p {pv.get('auc', {}).get('p', '-')}", flush=True)
    res["lentab"] = lentab(a.tag)
    for t, o in res["lentab"].items():
        if isinstance(o, dict):
            print("lentab", t, {k: round(v["hit5"], 3) for k, v in o["word"].items()},
                  "sounds", round(o["sound"]["all"]["hit5"], 3), "accept", round(o["native_word_accept_0.5"], 4))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
