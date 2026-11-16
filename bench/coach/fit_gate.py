"""Fit the coaching reliability gate on FLEURS VALIDATION planted errors (bench/coach/gather.py).

    python -m bench.coach.fit_gate

Rule (fixed before looking at the numbers): among grid settings whose POOLED "within one
articulatory feature" precision of heard_ipa on flagged planted substitutions is >= 0.85,
take the one with the highest pooled coverage (diagnosed / all planted substitutions).
Tones / stress: the loosest posterior bar whose exact precision is >= 0.85 (else the bar with
the highest precision that still diagnoses >= 20 % of the flagged items, reported as such).
Writes pronunciada/data/coach/gate.json and results/tables/coaching_gate.json.
"""
from __future__ import annotations

import itertools
import json
import math
import pickle
from pathlib import Path

import numpy as np

from bench.coach.gather import out_dir
from pronunciada import articulatory as A
from pronunciada.coach import DEFAULT_GATE, heard_candidate

LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")
TARGET = 0.85


def within1(t, h):
    try:
        d = A.diff(t, h)
        return len(d) <= 1 and "type" not in d
    except KeyError:
        return False


def load():
    rows, pros = [], []
    for l in LANGS:
        p = out_dir() / f"gather-{l}.pkl"
        if not p.exists():
            continue
        d = pickle.loads(p.read_bytes())
        for r in d["phones"]:
            r = {k: v for k, v in r.items() if k != "mx"}
            h, lh, le = heard_candidate(r.pop("m").astype(np.float64), r["expected"], r["alts"], l)
            r.update(lang=l, cand=h, lh=lh, lr=lh - le)
            rows.append(r)
        pros += [dict(x, lang=l) for x in d["pros"]]
    return rows, pros


def is_missing(r, g):
    return (r["n_frames"] <= g["missing_max_frames"] and not r["slot_heard"]
            and r["llr_del"] <= g["missing_min_llr_del"])


def passes(r, g):
    g = {**g, **g.get("by_lang", {}).get(r["lang"], {})}
    return (r["cand"] is not None and r["lr"] >= g["min_log_ratio"]
            and math.exp(r["lh"]) >= g["min_heard_post"] and r["n_frames"] >= g["min_frames"])


def phone_stats(rows, g, langs):
    out = {}
    for l in list(langs) + ["pooled"]:
        R = [r for r in rows if l == "pooled" or r["lang"] == l]
        sub = [r for r in R if r["kind"] in ("phone_sub_near", "phone_sub_far") and r["truth"]]
        fl = [r for r in sub if r["prob"] < g["flag_prob"]]
        dg = [r for r in fl if not is_missing(r, g) and passes(r, g)]
        clean = [r for r in R if r["kind"] == "clean"]
        cdg = [r for r in clean if not is_missing(r, g) and passes(r, g)]
        ins = [r for r in R if r["kind"] == "phone_ins"]
        ins_fl = [r for r in ins if r["prob"] < g["flag_prob"]]
        miss_pred = [r for r in fl + ins_fl + clean if is_missing(r, g)]
        out[l] = {"planted": len(sub), "flagged": len(fl), "diagnosed": len(dg),
                  "coverage": len(dg) / max(len(sub), 1), "coverage_of_flagged": len(dg) / max(len(fl), 1),
                  "precision_exact": float(np.mean([r["cand"] == r["truth"] for r in dg])) if dg else None,
                  "precision_within1": float(np.mean([within1(r["truth"], r["cand"]) for r in dg])) if dg else None,
                  "near": _sub(dg, sub, "phone_sub_near"), "far": _sub(dg, sub, "phone_sub_far"),
                  "clean_flagged": len(clean), "clean_diagnosed": len(cdg),
                  "missing_planted": len(ins), "missing_flagged": len(ins_fl),
                  "missing_said": sum(r["kind"] == "phone_ins" for r in miss_pred),
                  "missing_precision": (float(np.mean([r["kind"] == "phone_ins" for r in miss_pred]))
                                        if miss_pred else None),
                  "missing_coverage": sum(r["kind"] == "phone_ins" for r in miss_pred) / max(len(ins), 1)}
    return out


def _sub(dg, sub, kind):
    d = [r for r in dg if r["kind"] == kind]
    n = sum(r["kind"] == kind for r in sub)
    return {"coverage": len(d) / max(n, 1),
            "precision_within1": float(np.mean([within1(r["truth"], r["cand"]) for r in d])) if d else None}


def class_stats(rows, g):
    """Precision by sound class of the truth (vowel / consonant manner) on diagnosed rows."""
    out = {}
    dg = [r for r in rows if r["kind"] in ("phone_sub_near", "phone_sub_far") and r["truth"]
          and r["prob"] < g["flag_prob"] and not is_missing(r, g) and passes(r, g)]
    for r in dg:
        f = A.features(r["truth"])
        c = "vowel" if f["type"] == "vowel" else f["manner"]
        out.setdefault(c, []).append(within1(r["truth"], r["cand"]))
    return {c: {"n": len(v), "precision_within1": float(np.mean(v))} for c, v in sorted(out.items())}


def fit_phone(rows):
    g0 = dict(DEFAULT_GATE)
    # missing: zero frames, nothing heard, llr_del bar from a small grid (precision >= 0.85)
    best_m = None
    for x in (2.0, 1.0, 0.0, -1.0, -2.0, -4.0):
        g = dict(g0, missing_max_frames=0, missing_min_llr_del=x)
        s = phone_stats(rows, g, [])["pooled"]
        if s["missing_precision"] is not None and s["missing_precision"] >= TARGET:
            if best_m is None or s["missing_coverage"] > best_m[1]:
                best_m = (x, s["missing_coverage"])
    g0["missing_min_llr_del"] = best_m[0] if best_m else -1e9   # -1e9: never say "missing"
    grid = []
    for lr, hp, nf in itertools.product((-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0),
                                        (0.0, 0.02, 0.05, 0.1, 0.2, 0.3), (0, 1, 2)):
        g = dict(g0, min_log_ratio=lr, min_heard_post=hp, min_frames=nf)
        s = phone_stats(rows, g, [])["pooled"]
        grid.append({"min_log_ratio": lr, "min_heard_post": hp, "min_frames": nf,
                     "coverage": s["coverage"], "precision_within1": s["precision_within1"],
                     "precision_exact": s["precision_exact"]})
    ok = [x for x in grid if x["precision_within1"] is not None and x["precision_within1"] >= TARGET]
    pick = max(ok, key=lambda x: (x["coverage"], x["precision_within1"]))
    g = dict(g0, **{k: pick[k] for k in ("min_log_ratio", "min_heard_post", "min_frames")})
    # confidence table: precision within one feature by log-ratio quartile of diagnosed rows
    dg = [r for r in rows if r["kind"] in ("phone_sub_near", "phone_sub_far") and r["truth"]
          and r["prob"] < g["flag_prob"] and not is_missing(r, g) and passes(r, g)]
    lr = np.array([r["lr"] for r in dg])
    ok1 = np.array([within1(r["truth"], r["cand"]) for r in dg], float)
    edges = [float(round(e, 3)) for e in np.quantile(lr, [0.25, 0.5, 0.75])]
    k = np.searchsorted(edges, lr, side="right")
    vals = [float(ok1[k == j].mean()) if (k == j).any() else TARGET for j in range(4)]
    vals = list(np.maximum.accumulate(vals))           # monotone in the log-ratio
    g["confidence"] = {"edges": edges, "values": [round(v, 3) for v in vals]}
    # per language: where the pooled setting misses the target in that language, the loosest
    # heard-posterior bar that reaches it (with >= 20 diagnoses), else no diagnosis there
    g["by_lang"] = {}
    for l in LANGS:
        s = phone_stats(rows, g, [l])[l]
        if s["precision_within1"] is None or s["precision_within1"] >= TARGET:
            continue
        for hp in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
            gl = dict(g, min_heard_post=max(hp, g["min_heard_post"]))
            sl = phone_stats(rows, gl, [l])[l]
            if sl["diagnosed"] >= 20 and (sl["precision_within1"] or 0) >= TARGET:
                g["by_lang"][l] = {"min_heard_post": gl["min_heard_post"]}
                break
        else:
            g["by_lang"][l] = {"min_heard_post": 1.01}
    return g, grid


def _pros_cal(lang):
    """The word-level prosody channel calibrator the Scorer loads (pronunciada.scorer)."""
    from pronunciada import calibrate as C
    from pronunciada.prosody import default_model, selected_model

    d = C.data_dir() / "prosody"
    pp = d / f"{lang}.json"
    if selected_model(lang) != default_model(lang):
        pm = d / f"{lang}.{selected_model(lang)}.json"
        pp = pm if pm.exists() else pp
    return json.loads(pp.read_text()) if pp.exists() else None


def add_word_channel(pros):
    """x["word_prob"] = WordResult.tone_prob / stress_prob (the channel probability at the
    default strictness). Tone words: the planted syllable's probability plus the CLEAN run's
    probabilities of the word's other syllables (the planted run kept only the planted ones)."""
    from bench.pipeline.common import build_ref, fleurs_validation_records
    from pronunciada.scorer import _pros_prob

    cals = {}
    texts = {}
    for x in pros:
        l = x["lang"]
        if l not in cals:
            cals[l] = _pros_cal(l)
        cal = cals[l]
        if cal is None:
            x["word_prob"] = None
            continue
        if x["kind"] == "stress":
            x["word_prob"] = _pros_prob(cal, [x["stress_prob"]], 0.5)
    clean = {}
    for x in pros:
        if x["kind"] == "tone" and not x["planted"]:
            clean[(x["utt"], x["syl"])] = x["tone_prob"]
    refs = {}
    for x in pros:
        if x["kind"] != "tone":
            continue
        if x["lang"] not in texts:
            texts[x["lang"]] = {r["id"]: r["text"] for r in fleurs_validation_records(x["lang"])}
        if x["utt"] not in refs:
            refs[x["utt"]] = build_ref(x["lang"], texts[x["lang"]][x["utt"]])
        ref = refs[x["utt"]]
        w = ref.words[ref.syllables[x["syl"]].word]
        ps = [x["tone_prob"]] + [clean[(x["utt"], j)] for j in range(*w.syllables)
                                 if j != x["syl"] and (x["utt"], j) in clean]
        x["word_prob"] = _pros_prob(cals[x["lang"]], ps, 0.5)


def pros_curve(pros, kind, bars):
    P = [x for x in pros if x["kind"] == kind and x["planted"]]
    C = [x for x in pros if x["kind"] == kind and not x["planted"]]
    curve = []
    for b in bars:
        res = {"bar": b}
        for nm, S in (("planted", P), ("clean", C)):
            if kind == "tone":
                fl = [x for x in S if x["tone_prob"] is not None and x["tone_prob"] < 0.5
                      and x["word_prob"] is not None and x["word_prob"] < 0.5]
                hd = [(x, int(np.argmax(x["tone_probs"])) + 1) for x in fl]
                dg = [(x, h) for x, h in hd if h not in {x["expected"], *x["alts"]}
                      and max(x["tone_probs"]) >= b]
            else:
                fl = [x for x in S if x["stress_prob"] is not None and x["stress_prob"] < 0.5
                      and x["word_prob"] is not None and x["word_prob"] < 0.5]
                hd = [(x, int(np.argmax(x["dist"]))) for x in fl]
                dg = [(x, h) for x, h in hd if h != x["expected"] and max(x["dist"]) >= b]
            res[nm] = {"n": len(S), "flagged": len(fl), "diagnosed": len(dg),
                       "precision": float(np.mean([h == x["truth"] for x, h in dg])) if dg else None,
                       "coverage": len(dg) / max(len(S), 1)}
        curve.append(res)
    ok = [c for c in curve if c["planted"]["precision"] is not None and c["planted"]["precision"] >= TARGET]
    if ok:
        pick, met = min(ok, key=lambda c: c["bar"]), True
    else:
        cand = [c for c in curve if c["planted"]["diagnosed"] >= 0.2 * max(c["planted"]["flagged"], 1)]
        pick, met = max(cand or curve, key=lambda c: c["planted"]["precision"] or 0), False
    return pick, curve, met


def per_lang_pros(pros, kind, bar):
    out = {}
    for l in LANGS:
        sub = [x for x in pros if x["lang"] == l]
        if any(x["kind"] == kind for x in sub):
            out[l] = pros_curve(sub, kind, [bar])[1][0]
    return out


def main():
    rows, pros = load()
    add_word_channel(pros)
    g, grid = fit_phone(rows)
    tone_pick, tone_curve, tone_met = pros_curve(pros, "tone", (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    st_pick, st_curve, st_met = pros_curve(pros, "stress", (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    g["tone_min_post"], g["stress_min_post"] = tone_pick["bar"], st_pick["bar"]
    g["tone_confidence"] = round(tone_pick["planted"]["precision"] or 0, 3)
    g["stress_confidence"] = round(st_pick["planted"]["precision"] or 0, 3)
    ps = phone_stats(rows, g, LANGS)
    g["missing_confidence"] = round(ps["pooled"]["missing_precision"] or 0, 3)
    curves = {}                                         # precision / coverage vs log-ratio bar
    for l in list(LANGS) + ["pooled"]:
        curves[l] = []
        for lr in (-1.0, 0.0, 1.0, 2.0, 3.0):
            gl = dict(g, min_log_ratio=lr)
            s = phone_stats(rows, gl, [l] if l != "pooled" else [])[l]
            curves[l].append({"min_log_ratio": lr, "coverage": s["coverage"],
                              "precision_within1": s["precision_within1"], "precision_exact": s["precision_exact"]})
    keep = {k: g[k] for k in DEFAULT_GATE}
    keep["by_lang"] = g["by_lang"]
    Path("pronunciada/data/coach/gate.json").write_text(json.dumps(
        {"schema_version": 1, "fitted_on": "FLEURS validation, planted substitutions (bench/coach/fit_gate.py)",
         "thresholds": keep}, indent=1))
    rep = {"thresholds": keep, "phones": ps, "classes": class_stats(rows, g), "grid": grid,
           "curves": curves, "tone": {"curve": tone_curve, "target_met": tone_met,
                                      "by_lang": per_lang_pros(pros, "tone", g["tone_min_post"])},
           "stress": {"curve": st_curve, "target_met": st_met,
                      "by_lang": per_lang_pros(pros, "stress", g["stress_min_post"])}}
    Path("results/tables/coaching_gate.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(keep, indent=0))
    for l, s in ps.items():
        print(f"{l:6s} planted {s['planted']:5d} flagged {s['flagged']:4d} diag {s['diagnosed']:4d} "
              f"cov {s['coverage']:.3f} ex {s['precision_exact'] or 0:.3f} w1 {s['precision_within1'] or 0:.3f} "
              f"clean {s['clean_diagnosed']}/{s['clean_flagged']} miss P {s['missing_precision']} cov {s['missing_coverage']:.3f}")
    print("tone", tone_pick, tone_met)
    print("stress", st_pick, st_met)
    print(json.dumps(rep["classes"]))


if __name__ == "__main__":
    main()
