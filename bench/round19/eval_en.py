"""Round 19 (English): held-out speechocean762 TEST through the public Scorer, by age group.

    python -m bench.round19.eval_en run --tag base [--precision fp32] [--env PRONUNCIADA_SENTENCE_HEAD=0]
    python -m bench.round19.eval_en compare --a base --b r19 [--precision fp32]

`run` scores every test clip with pronunciada.Scorer (bench.calib.api_run, outputs kept apart from
the shared cache under $PRONUNCIADA_CACHE/round19/english/api) and computes the same phone / word /
sentence correlations as bench.eval_english (speaker-cluster bootstrap 95 % CIs, paired difference
to Azure), plus the sentence scores for Everyone / Adults (13+) / Adults 18+ / Children <= 12.
`compare` gives paired before / after differences (speaker bootstrap, two-sided p).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

N_BOOT = 2000
GROUPS = {"everyone": lambda a: True, "adults_13plus": lambda a: a >= 13,
          "adults_18plus": lambda a: a >= 18, "children_le12": lambda a: a <= 12}
SENT = ("accuracy", "fluency", "prosodic", "total")


def root() -> Path:
    from bench.data.common import cache_root

    return cache_root() / "round19" / "english"


def api_for(tag, precision, workers=4):
    from bench.calib.api_run import run
    from bench.data import load

    os.environ["PRONUNCIADA_API_DIR"] = str(root() / "api")
    recs = load("speechocean762", split="test")
    return run(recs, "en", precision=precision, workers=workers, threads=1,
               tag=f"so762-test-{tag}-{precision}")


def ages():
    from bench.data import load

    return {r["id"]: (r["labels"]["age"], r["speaker"]) for r in load("speechocean762", split="test")}


def sentence_rows(api):
    from bench.data import load

    rows = []
    for rec in load("speechocean762", split="test"):
        o = api.get(rec["id"])
        if o is None or "error" in o:
            continue
        rows.append({"id": rec["id"], "spk": rec["speaker"], "age": rec["labels"]["age"],
                     **{f"y_{k}": float(rec["labels"]["sentence"][k]) for k in SENT},
                     **{f"o_{k}": float(o["sentence"][k]) for k in SENT}})
    return rows


def group_table(api):
    from bench.metrics import bootstrap_ci, pearson

    rows = sentence_rows(api)
    out = {}
    for g, f in GROUPS.items():
        R = [r for r in rows if f(r["age"])]
        spk = np.array([r["spk"] for r in R])
        out[g] = {"n_utts": len(R), "n_speakers": int(len(set(spk)))}
        for k in SENT:
            y = np.array([r[f"y_{k}"] for r in R])
            s = np.array([r[f"o_{k}"] for r in R])
            ci = bootstrap_ci(pearson, s, y, clusters=spk, n_boot=N_BOOT)
            out[g][k] = {"pearson": ci["value"], "lo": ci["lo"], "hi": ci["hi"]}
    return out


def cmd_run(a):
    from bench.eval_english import eval_so762

    for kv in a.env or []:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    api = api_for(a.tag, a.precision, a.workers)
    so, _ = eval_so762(api)
    res = {"tag": a.tag, "precision": a.precision, "env": a.env or [], "so762": so, "groups": group_table(api)}
    p = root() / f"heldout_{a.tag}_{a.precision}.json"
    p.write_text(json.dumps(res, indent=1, default=float))
    print("phone", round(so["phone"]["ours_B"]["pearson"], 4), "word", round(so["word"]["ours"]["pearson"], 4))
    for k in SENT:
        print(k, round(so["sentence"][k]["ours"]["pearson"], 4))
    for g, r in res["groups"].items():
        print(g, r["n_utts"], {k: round(r[k]["pearson"], 4) for k in SENT})
    print("wrote", p)


def cmd_compare(a):
    from bench.metrics import paired_bootstrap, pearson

    os.environ["PRONUNCIADA_API_DIR"] = str(root() / "api")
    import pickle

    A = pickle.loads((root() / "api" / f"so762-test-{a.a}-{a.precision}.pkl").read_bytes())
    B = pickle.loads((root() / "api" / f"so762-test-{a.b}-{a.precision}.pkl").read_bytes())
    ra = {r["id"]: r for r in sentence_rows(A)}
    rb = {r["id"]: r for r in sentence_rows(B)}
    ids = [i for i in ra if i in rb]
    out = {}
    for g, f in GROUPS.items():
        I = [i for i in ids if f(ra[i]["age"])]
        spk = np.array([ra[i]["spk"] for i in I])
        out[g] = {}
        for k in SENT:
            y = np.array([ra[i][f"y_{k}"] for i in I])
            sa = np.array([ra[i][f"o_{k}"] for i in I])
            sb = np.array([rb[i][f"o_{k}"] for i in I])
            pb = paired_bootstrap(lambda yy, ss: pearson(ss, yy), y, sb, sa, clusters=spk, n_boot=N_BOOT)
            p2 = min(1.0, 2 * min(pb["p_value"], 1 - pb["p_value"] + 1 / (N_BOOT + 1)))
            out[g][k] = {"a": pearson(sa, y), "b": pearson(sb, y), "delta": pb["delta"], "lo": pb["lo"],
                         "hi": pb["hi"], "p_two_sided": p2}
            print(g, k, f"{pearson(sa, y):.4f} -> {pearson(sb, y):.4f}  d={pb['delta']:+.4f} "
                        f"[{pb['lo']:+.4f}, {pb['hi']:+.4f}] p={p2:.3f}")
    p = root() / f"compare_{a.a}_vs_{a.b}_{a.precision}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--precision", default="fp32")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--env", action="append")
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--precision", default="fp32")
    a = ap.parse_args()
    (root() / "api").mkdir(parents=True, exist_ok=True)
    cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    main()
