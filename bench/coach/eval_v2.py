"""Coaching round 2, HELD-OUT evaluation (run once, after bench/coach/fit_v2.py froze the model).

    python -m bench.coach.eval_v2

speechocean762 TEST  flags vs expert phone scores (error = score < 2; also < 1.5), named heard
                     sound vs the experts' pronounced phone, "missing" vs <DEL>
OMPAL TEST           error-type precision / recall per character, learners only
L2-ARCTIC / EpaDB    report only (non-commercial licences), same protocol as eval_learners
Every system is evaluated from the dumps (bench/coach/dump.py) through pronunciada.coach.decide,
the function coaching() itself uses. "v1" = the round-1 rule (sound prob < 0.5 at the default
strictness, heard = the same-type candidate, German heard posterior >= 0.25).
Writes results/tables/coaching_v2_heldout.json and two example outputs.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.coach.dump import load_dump
from bench.coach.fit_v2 import TARGETS, add_initial, fold, so762_rows, within1
from pronunciada import coach as Co


def v1_decide(X, lang):
    p = 1 / (1 + np.exp(-0.5 * (np.asarray(X["zl"], float) + np.minimum(X["zs"], X["zl"]))))
    flag = p < 0.5
    heard = [None] * len(p)
    for i in np.flatnonzero(flag):
        c = X["cand"][i]
        if c is not None and not (lang == "de" and np.exp(X["lh"][i]) < 0.25):
            heard[i] = c
    return {"flag": flag, "heard": heard, "kind": ["substitution" if f else None for f in flag]}


def so762_eval(mdl):
    R = so762_rows("so762-test", "test")
    m = np.isfinite(R["acc"])
    utts = np.asarray(R["utt"])
    out = {}
    systems = [("v1", None)] + [(s, s) for s in TARGETS]
    for name, sens in systems:
        flag = np.zeros(len(m), bool)
        heard = [None] * len(m)
        kind = [None] * len(m)
        for u in np.unique(utts):
            ix = np.flatnonzero(utts == u)
            X = {k: (v[ix] if isinstance(v, np.ndarray) else [v[i] for i in ix]) for k, v in R.items()}
            d = v1_decide(X, "en") if sens is None else Co.decide(X, "en", sens, mdl)
            flag[ix] = d["flag"]
            for q, i in enumerate(ix):
                heard[i], kind[i] = d["heard"][q], d["kind"][q]
        f = flag & m
        y = R["acc"] < 2
        y2 = R["acc"] < 1.5
        sub = [i for i in np.flatnonzero(f) if R["pron"][i] not in (None, "<DEL>", "<unk>") and heard[i]]
        allsub = [i for i in np.flatnonzero(m) if R["pron"][i] not in (None, "<DEL>", "<unk>")]
        named = [i for i in np.flatnonzero(f) if heard[i]]
        miss = [i for i in np.flatnonzero(f) if kind[i] == "missing"]
        by_cls = defaultdict(list)
        for i in sub:
            by_cls[Co.sound_class(R["expected"][i])].append(within1(R["pron"][i], fold(heard[i])))
        out[name] = {
            "flagged": int(f.sum()), "precision": float(y[f].mean()), "recall": float(y[f].sum() / y[m].sum()),
            "precision_acc<1.5": float(y2[f].mean()), "recall_acc<1.5": float(y2[f].sum() / y2[m].sum()),
            "named": len(named), "named_on_substitutions": len(sub),
            "heard_exact": float(np.mean([R["pron"][i] == fold(heard[i]) for i in sub])) if sub else None,
            "heard_within1": float(np.mean([within1(R["pron"][i], fold(heard[i])) for i in sub])) if sub else None,
            "heard_coverage_of_substitutions": len(sub) / max(len(allsub), 1),
            "named_on_correct": sum(1 for i in named if R["acc"][i] >= 2),
            "named_on_deleted": sum(1 for i in named if R["pron"][i] == "<DEL>"),
            "missing_said": len(miss),
            "missing_precision": float(np.mean([R["pron"][i] == "<DEL>" for i in miss])) if miss else None,
            "heard_by_class": {c: {"n": len(v), "within1": float(np.mean(v))} for c, v in sorted(by_cls.items())},
            "n_sounds": int(m.sum()), "n_substitutions": len(allsub)}
    return out, R


def ompal_eval(mdl):
    from bench.coach.fit_v2 import ompal_units

    rows = ompal_units("ompal-test", "test", mdl)
    out = {"units": len(rows)}
    for sens in TARGETS:
        res = {}
        for key, col, lab, thr, sign in (
                ("tone", "tone", "tone_wrong", mdl["tone"]["thresholds"][sens], -1),
                ("consonant", "ini", "cons_wrong", mdl["by_lang"]["zh"]["initial"][sens], 1),
                ("vowel", "fin", "vowel_wrong", mdl["by_lang"]["zh"]["final"][sens], 1)):
            v = np.array([r[col] for r in rows], float)
            y = np.array([r[lab] for r in rows], bool)
            f = np.isfinite(v) & ((v <= thr) if sign < 0 else (v >= thr))
            res[key] = {"flagged": int(f.sum()), "precision": float(y[f].mean()) if f.any() else None,
                        "recall": float(y[f].sum() / max(y.sum(), 1)), "threshold": thr}
        out[sens] = res
    return out


def learners_eval(mdl):
    from bench.coach import eval_learners as EL

    out = {}
    for ds, name in (("l2arctic", "l2arctic-test"), ("epadb", "epadb-test")):
        D = load_dump(name)
        for sysname, sens in [("v2_" + s, s) for s in TARGETS]:
            O = {}
            for rid, o in D.items():
                if "error" in o:
                    O[rid] = o
                    continue
                X = add_initial(dict(o["sounds"]))
                d = Co.decide(X, "en", sens, mdl)
                O[rid] = {sysname: [{"gi": int(i), "heard": d["heard"][i], "kind": d["kind"][i]}
                                    for i in np.flatnonzero(d["flag"])]}
            EL._load = lambda _ds, O=O: O  # noqa: SLF001  (feed the offline decisions)
            out.setdefault(ds, {})[sysname] = EL.english(ds, sysname)
    return out


def examples(R, mdl, n=2):
    """Two so762 test utterances with a named substitution that the experts also marked."""
    from bench.data import load

    recs = {r["id"]: r for r in load("speechocean762", split="test")}
    utts = np.asarray(R["utt"])
    out = []
    for u in dict.fromkeys(R["utt"]):
        ix = np.flatnonzero(utts == u)
        X = {k: (v[ix] if isinstance(v, np.ndarray) else [v[i] for i in ix]) for k, v in R.items()}
        d = Co.decide(X, "en", "balanced", mdl)
        hits = [q for q in np.flatnonzero(d["flag"]) if d["heard"][q] and X["pron"][q] not in (None, "<DEL>", "<unk>")]
        if not hits or len(np.flatnonzero(d["flag"])) > 4:
            continue
        rec = recs[u]
        ours = [{"word": rec["text"].split()[int(X["word"][q])] if int(X["word"][q]) < len(rec["text"].split()) else None,
                 "expected": X["expected"][q], "heard": d["heard"][q], "kind": d["kind"][q],
                 "error_prob": round(float(d["error_prob"][q]), 3)} for q in np.flatnonzero(d["flag"])]
        experts = [{"word": w["text"], "phones": w["phones"], "scores": w["phones_accuracy"],
                    "mispronunciations": w["mispronunciations"]} for w in rec["labels"]["words"]
                   if min(w["phones_accuracy"]) < 2 or w["mispronunciations"]]
        out.append({"id": u, "text": rec["text"], "ours": ours, "experts": experts})
        if len(out) >= n:
            break
    return out


def main():
    mdl = Co.model()
    so, R = so762_eval(mdl)
    rep = {"so762_test": so, "ompal_test": ompal_eval(mdl), "learners": learners_eval(mdl),
           "examples": examples(R, mdl)}
    Path("results/tables/coaching_v2_heldout.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    for k, v in so.items():
        print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if x != "heard_by_class"})
        print("   by class", v["heard_by_class"])
    print(json.dumps(rep["ompal_test"], indent=0))
    for ds, v in rep["learners"].items():
        for s, x in v.items():
            print(ds, s, {k: x[k] for k in ("flagged_sub", "coverage_sub", "precision_exact", "precision_within1",
                                            "sub_exact", "sub_within1", "diagnosed_labels")})
    print(json.dumps(rep["examples"], ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
