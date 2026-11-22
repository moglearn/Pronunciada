"""OpenPronounce 1.1.0 against Pronunciada on speechocean762 test (all 2,500 recordings, sentence known).

    python -m bench.rivals.eval_so762 [--modes default,fast,student]

Items: every test utterance; a record OpenPronounce could not score gets its floor (0) and is counted.
Groups by speaker age, OpenPronounce's rule (benchmarks/groups.py): children 12 and under, teenagers
13-17, adults 18 and over; "everyone" = all 2,500.

Pronunciada = the cached per-utterance outputs of the shipped scorer (fp32) that bench/eval_english.py
reads ($PRONUNCIADA_CACHE/calib/api/speechocean762-test-fp32.pkl); no Pronunciada re-scoring here.
OpenPronounce = the raw outputs of bench.rivals.openpronounce, attached to Pronunciada's reference
words by bench.rivals.convert.

Tests (95 % intervals: 1,000 speaker resamples, seed 0; paired difference Pronunciada - OpenPronounce on
the same resamples, two-sided p from bench.metrics.paired_bootstrap):
  sentence  Pearson of each system's total / accuracy / fluency / prosodic with the same expert score.
  word      Pearson of the word score with expert word accuracy; ROC AUC for "expert word accuracy <= 6"
            (and < 5, OpenPronounce's own label) against the rest, error signal = minus the word score.
            Expert words are matched to reference words exactly as bench/eval_zipa_baseline.eval_so762 does.
  sound     Pearson of the sound score with the expert phone accuracy (0-2), same matching.
Writes results/tables/rivals_so762.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tables" / "rivals_so762.json"
N_BOOT = 1000
DIMS = ("total", "accuracy", "fluency", "prosodic")
GROUPS = (("everyone", 0, 200), ("adults (18+)", 18, 200), ("children (12 and under)", 0, 12),
          ("teenagers (13-17)", 13, 17))


def _two(p1: float) -> float:
    return min(1.0, 2 * min(p1, 1 - p1 + 1 / (N_BOOT + 1)))


def collect(system: str) -> dict:
    from bench.calib.common import calib_cache
    from bench.data import load
    from bench.pipeline.common import build_ref
    from bench.pipeline.labels import align_reference, arpa_to_ours
    from bench.rivals.common import ext_dir

    api = pickle.loads((calib_cache() / "api" / "speechocean762-test-fp32.pkl").read_bytes())
    d = ext_dir(system, "speechocean762")
    S = {k: [] for k in ("id", "spk", "age", "op_ok")}
    for k in DIMS:
        S[f"y_{k}"], S[f"pa_{k}"], S[f"op_{k}"] = [], [], []
    W = {k: [] for k in ("y", "spk", "age", "pa", "op", "op_flag")}
    P = {k: [] for k in ("y", "spk", "age", "pa", "op")}
    cnt = Counter()
    for rec in load("speechocean762", split="test"):
        o = api[rec["id"]]
        p = d / f"{rec['id']}.json"
        b = json.loads(p.read_text()) if p.exists() else {"error": "missing"}
        ok = "error" not in b
        cnt["op_scored" if ok else "op_failed"] += 1
        ref = build_ref("en", rec["text"])
        assert [x.ipa for x in ref.phones] == o["phones"], rec["id"]
        if ok:
            assert b["phones"] == o["phones"], rec["id"]
        L = rec["labels"]
        spk, age = rec["speaker"], int(L["age"])
        S["id"].append(rec["id"]), S["spk"].append(spk), S["age"].append(age), S["op_ok"].append(ok)
        for k in DIMS:
            S[f"y_{k}"].append(float(L["sentence"][k]))
            S[f"pa_{k}"].append(float(o["sentence"][k]))
            S[f"op_{k}"].append(float(b["extra"]["scores"][k]) if ok else 0.0)
        nw, nph = len(o["words"]), len(o["phones"])
        pa_w, pa_p = np.asarray(o["word_reg"], float), np.asarray(o["sound"]["B"]["reg"], float)
        op_w = np.asarray(b["gop_w"], float) if ok else np.zeros(nw)
        op_f = np.asarray(b["match_w"], float) if ok else np.zeros(nw)
        op_p = np.asarray(b["gop"], float) if ok else np.zeros(nph)
        Wl = L["words"]
        tp = [[arpa_to_ours(x) for x in w["phones"]] for w in Wl]
        o2t, wpairs = align_reference(ref, [w["text"] for w in Wl], tp)
        t2o = {v: k for k, v in o2t.items()}
        tw = {}
        for i, (j, _) in o2t.items():
            tw.setdefault(j, set()).add(ref.phones[i].word)
        for i, j in wpairs:
            tw.setdefault(j, set()).add(i)
        for j, w in enumerate(Wl):
            ow = sorted(tw.get(j, set()))
            fb = [k for x in ow for k in range(*ref.words[x].phones)]
            for k, s in enumerate(w["phones_accuracy"]):
                i = t2o.get((j, k), -1)
                for name, arr in (("pa", pa_p), ("op", op_p)):
                    P[name].append(float(arr[i] if i >= 0 else (np.mean(arr[fb]) if fb else np.mean(arr))))
                P["y"].append(float(s)), P["spk"].append(spk), P["age"].append(age)
            W["y"].append(float(w["accuracy"])), W["spk"].append(spk), W["age"].append(age)
            for name, arr in (("pa", pa_w), ("op", op_w), ("op_flag", op_f)):
                W[name].append(float(np.mean(arr[ow]) if ow else np.mean(arr)))
            cnt["words"] += 1
            cnt["words_fallback"] += not ow
    return {"S": {k: np.asarray(v) for k, v in S.items()}, "W": {k: np.asarray(v) for k, v in W.items()},
            "P": {k: np.asarray(v) for k, v in P.items()}, "counts": dict(cnt)}


def compare(metric, y, a, b, spk) -> dict:
    """metric(y, s): Pronunciada (a) and OpenPronounce (b), speaker-bootstrap intervals, paired difference."""
    from bench.metrics import bootstrap_ci, paired_bootstrap

    out = {"n": int(len(y)), "speakers": int(len(set(spk.tolist())))}
    for name, s in (("pronunciada", a), ("openpronounce", b)):
        ci = bootstrap_ci(lambda yy, ss: metric(yy, ss), y, s, clusters=spk, n_boot=N_BOOT)
        out[name] = {"value": ci["value"], "ci": [ci["lo"], ci["hi"]]}
    pb = paired_bootstrap(metric, y, a, b, clusters=spk, n_boot=N_BOOT)
    out["delta"] = {"value": pb["delta"], "ci": [pb["lo"], pb["hi"]], "p": _two(pb["p_value"])}
    return out


def evaluate(system: str, with_sound: bool = True) -> dict:
    from bench.metrics import pearson, roc_auc, spearman

    C = collect(system)
    S, W, P = C["S"], C["W"], C["P"]
    pr = lambda y, s: pearson(s, y)                     # noqa: E731
    res = {"system": system, "counts": C["counts"], "groups": {}}
    for gname, lo, hi in GROUPS:
        g = {}
        m = (S["age"] >= lo) & (S["age"] <= hi)
        if len(set(S["spk"][m].tolist())) < 3:
            continue
        for k in DIMS:
            g[f"sentence {k}"] = compare(pr, S[f"y_{k}"][m], S[f"pa_{k}"][m], S[f"op_{k}"][m], S["spk"][m])
        g["sentence total, Spearman"] = compare(lambda y, s: spearman(s, y), S["y_total"][m], S["pa_total"][m],
                                                S["op_total"][m], S["spk"][m])
        # sensitivity: the same Pearson with each speaker left out in turn (range of the difference)
        spk_m = S["spk"][m]
        loo = []
        for s_out in sorted(set(spk_m.tolist())):
            k_ = spk_m != s_out
            loo.append((pearson(S["pa_total"][m][k_], S["y_total"][m][k_])
                        - pearson(S["op_total"][m][k_], S["y_total"][m][k_]), s_out))
        g["sentence total"]["leave_one_speaker_out_delta"] = {"min": min(loo)[0], "min_speaker": min(loo)[1],
                                                              "max": max(loo)[0], "max_speaker": max(loo)[1]}
        mw = (W["age"] >= lo) & (W["age"] <= hi)
        g["word Pearson"] = compare(pr, W["y"][mw], W["pa"][mw], W["op"][mw], W["spk"][mw])
        for tag, bad in (("word AUC, accuracy <= 6", W["y"] <= 6), ("word AUC, accuracy < 5", W["y"] < 5)):
            yb = bad[mw].astype(float)
            g[tag] = compare(lambda y, s: roc_auc(y, -s), yb, W["pa"][mw], W["op"][mw], W["spk"][mw])
            g[tag]["n_positive"] = int(yb.sum())
            # the app's own report (word flagged or not), AUC of a binary signal
            from bench.metrics import bootstrap_ci

            ci = bootstrap_ci(lambda y, s: roc_auc(y, -s), yb, W["op_flag"][mw], clusters=W["spk"][mw], n_boot=N_BOOT)
            g[tag]["openpronounce_flags_only"] = {"value": ci["value"], "ci": [ci["lo"], ci["hi"]]}
            fl = W["op_flag"][mw] < 50
            g[tag]["openpronounce_flags"] = {"flagged_share": float(fl.mean()),
                                             "precision": float(yb[fl].mean()) if fl.any() else math.nan,
                                             "recall": float(fl[yb == 1].mean()) if yb.any() else math.nan}
        if with_sound:
            mp = (P["age"] >= lo) & (P["age"] <= hi)
            g["sound Pearson"] = compare(pr, P["y"][mp], P["pa"][mp], P["op"][mp], P["spk"][mp])
        res["groups"][gname] = g
        print(gname, {k: (round(v["pronunciada"]["value"], 3), round(v["openpronounce"]["value"], 3),
                          round(v["delta"]["value"], 3), round(v["delta"]["p"], 3)) for k, v in g.items()}, flush=True)
    ok = S["op_ok"]
    res["openpronounce_scored_only_total_everyone"] = float(pearson(S["op_total"][ok], S["y_total"][ok]))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default="default")
    a = ap.parse_args(argv)
    from bench.rivals.convert import SYSTEM

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out["protocol"] = __doc__.split("\n\n", 1)[1]
    for mode in a.modes.split(","):
        out[mode] = evaluate(SYSTEM[mode])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
