"""Round 18 A: fit and check the word-level contrast factor (pronunciada/contrast_word.py).

    python -m bench.round18.fit dev        # dev table + adopted parameter files
    python -m bench.round18.fit test       # held-out, once (adopted languages; before / after paired)

Factor: f = 1 while the word's lowest contrast log-odds z is above a knee, then linearly down to
`floor` over `width` (4) log-odds units. Knee at strictness 0.5 = the q-quantile of z over native
words with a judged sound (FLEURS validation); strictness 0 / 1 use q / 2 and 2 q.
Rule (fixed in advance, as for the tone / stress / length channels): grid q x floor; among the
settings whose native word acceptance at strictness 0.5 drops <= 0.5 pt, the strictest (largest
q, then lowest floor). Adopted per language when (dev): contrast-swap word hit@5 % (pooled over
the language's contrasts) +15 pt or more; near-swap word AUC and planted word-error AUC do not
drop; learners (speechocean762 train for en, OMPAL dev for zh): no drop beyond 0.005.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.round18.collect import CONTRASTS, load_set
from pronunciada.contrast_word import word_factor

LANGS = ("zh", "en", "ru", "ja", "de", "ko")
QS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
FLOORS = (0.2, 0.4, 0.6, 0.8)
TOL = 1e-4                  # "does not drop": tolerance for ties at the 4th decimal
WIDTH = 4.0


def auc(pos, neg):
    from scipy.stats import rankdata

    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def factor(cal, z, s=0.5):
    if cal is None or not np.isfinite(z):
        return 1.0
    return word_factor(cal, [z], s)


def apply(cal, wp, z, s=0.5):
    return np.array([p * factor(cal, zz, s) for p, zz in zip(wp, z)])


def tense_swaps(D, cal_tense, n=120, seed=0):
    """Korean lax / tense swaps at word level (offline: only the reference flag changes, so the
    word probability is the native one; the word's z takes the flipped tense evidence)."""
    rng = np.random.default_rng(seed)
    items = {"A>B": [], "B>A": []}
    keys = sorted(D)
    for d, src in (("A>B", True), ("B>A", False)):
        cand = []
        for k in keys:
            o = D[k]
            if o is None or o.get("tense_z") is None:
                continue
            for i, t in enumerate(o["tense"]):
                if t is src and np.isfinite(o["tense_z"][i]):
                    cand.append((k, i))
        per = defaultdict(int)
        for j in rng.permutation(len(cand)):
            k, i = cand[j]
            if per[k] >= 2:
                continue
            per[k] += 1
            o = D[k]
            w = o["word_of"][i]
            zi = o["tense_z"][i]
            z_flip = -zi - 2 * cal_tense["slope"] * cal_tense["s0"]
            others = [o["contrast_z"][q] for q in range(len(o["word_of"])) if o["word_of"][q] == w and q != i
                      and np.isfinite(o["contrast_z"][q])]
            items[d].append((o["clean"]["wp"][0.5][w], min([z_flip] + others)))
            if len(items[d]) >= n:
                break
    return items


def dev_parts(lang):
    D = load_set("fleurs-val", lang)
    nat_wp, nat_z = [], []
    edits = defaultdict(list)                       # contrast/direction -> (wp, z)
    near = {"swap_near": {"pos": [], "neg": []}, "swap_far": {"pos": [], "neg": []}, "orig": []}
    cor_pos, cor_neg = [], []
    for k, o in D.items():
        if o is None:
            continue
        c = o["clean"]
        nat_wp += c["wp"][0.5]
        nat_z += c["z"]
        for e in o["edits"]:
            w = e["word"]
            edits[f"{e['contrast']}/{e['direction']}"].append((e["wp"][0.5][w], e["z"][w]))
        for kind, r in o["cor"].items():
            for p, z, wr in zip(r["wp"][0.5], r["z"], r["word_wrong"]):
                (cor_pos if wr else cor_neg).append((p, z))
        for sid, r in o["swap"].items():
            v = r["labels"]["variant"]
            if v == "orig":
                near["orig"] += list(zip(r["wp"][0.5], r["z"]))
            elif v in near and r["labels"].get("char_span_in_new_text"):
                a, b = r["labels"]["char_span_in_new_text"]
                hit = [j for j, (x, y) in enumerate(r["span"]) if x < b and y > a]
                if not hit:
                    continue
                pz = [(r["wp"][0.5][j], r["z"][j]) for j in hit]
                near[v]["pos"].append(pz)
                near[v]["neg"] += [(r["wp"][0.5][j], r["z"][j]) for j in range(len(r["span"])) if j not in hit]
    if lang == "ko":
        tz = json.loads(Path("pronunciada/data/coach/contrast_heads.json").read_text())["langs"]["ko"]
        tz = [e for e in tz if e["contrast"] == "tense"]
        if tz:
            for d, v in tense_swaps(D, tz[0]).items():
                edits[f"tense/{d}"] = v
    return {"nat_wp": np.array(nat_wp), "nat_z": np.array(nat_z), "edits": dict(edits), "near": near,
            "cor_pos": cor_pos, "cor_neg": cor_neg}


def metrics(P, cal):
    nw = apply(cal, P["nat_wp"], P["nat_z"])
    t5 = np.quantile(nw, 0.05)
    out = {"native_word_accept": float(np.mean(nw >= 0.5)), "contrast": {}}
    allp = []
    for k, v in sorted(P["edits"].items()):
        pe = np.array([p * factor(cal, z) for p, z in v])
        out["contrast"][k] = {"n": len(v), "word_hit5": float(np.mean(pe <= t5))}
        allp += list(pe <= t5)
    out["contrast_pooled_hit5"] = float(np.mean(allp)) if allp else None
    for v in ("swap_near", "swap_far"):
        pos = [min(p * factor(cal, z) for p, z in grp) for grp in P["near"][v]["pos"]]
        neg = [p * factor(cal, z) for p, z in P["near"][v]["neg"]]
        orig = np.array([p * factor(cal, z) for p, z in P["near"]["orig"]])
        out[v] = {"auc": auc(1 - np.array(pos), 1 - np.array(neg)),
                  "hit5": float(np.mean(np.array(pos) <= np.quantile(orig, 0.05))) if len(orig) else None,
                  "n": len(pos)}
    cp = [p * factor(cal, z) for p, z in P["cor_pos"]]
    cn = [p * factor(cal, z) for p, z in P["cor_neg"]]
    out["planted_word_auc"] = auc(1 - np.array(cp), 1 - np.array(cn))
    return out


def near_misses_of_contrast(P, lang, base):
    """Dev near-swap words missed at 5 % native false alarms (before), and how many of them have a
    judged contrast sound (finite z)."""
    orig = np.array([p for p, _ in P["near"]["orig"]])
    t = np.quantile(orig, 0.05)
    miss = [grp for grp in P["near"]["swap_near"]["pos"] if min(p for p, _ in grp) > t]
    with_z = [grp for grp in miss if any(np.isfinite(z) for _, z in grp)]
    return {"missed": len(miss), "with_contrast_sound": len(with_z),
            "of_which_below_native_z_1pct": None}


def learners(lang, cal, split=None):
    """en: speechocean762 word P vs expert word accuracy (Pearson, AUC for accuracy < 6.5);
    zh: OMPAL word AUCs (any / tone-only / segment wrong; all speakers)."""
    from bench.data import load

    split = split or {"en": "train", "zh": "dev"}.get(lang)
    if lang == "en":
        D = load_set(f"so762-{split}", "en")
        ys, pb, pa = [], [], []
        for r in load("speechocean762", split=split):
            o = D.get(r["id"])
            W = r["labels"]["words"]
            if o is None or "error" in o or len(o["wp"][0.5]) != len(W):
                continue
            for w, p, z in zip(W, o["wp"][0.5], o["z"]):
                ys.append(w["accuracy"])
                pb.append(p)
                pa.append(p * factor(cal, z))
        ys, pb, pa = np.array(ys, float), np.array(pb), np.array(pa)
        bad = ys < 6.5
        return {"n_words": int(len(ys)), "pearson": [float(np.corrcoef(ys, pb)[0, 1]), float(np.corrcoef(ys, pa)[0, 1])],
                "auc_acc_below_6.5": [auc(1 - pb[bad], 1 - pb[~bad]), auc(1 - pa[bad], 1 - pa[~bad])]}
    if lang == "zh":
        from bench.eval_ompal import overlap, word_char_spans

        D = load_set(f"ompal-{split}", "zh")
        rows = []
        for r in load("ompal", split=split):
            lab = r["labels"]
            if lab["word_level_dropped"]:
                continue
            o = D.get(r["id"])
            if o is None or "error" in o:
                continue
            spans = word_char_spans(r["text"], o["text"])
            for u in lab["units"]:
                if u["span"] is None:
                    continue
                ws = [j for j, sp in enumerate(spans) if sp is not None and overlap(*u["span"], *sp)]
                b = min([o["wp"][0.5][j] for j in ws] or [0.0])
                a = min([o["wp"][0.5][j] * factor(cal, o["z"][j]) for j in ws] or [0.0])
                rows.append((u["any_wrong"], u["tone_only"], u["seg_wrong"], u["correct"], b, a))
        R = np.array(rows, float)
        out = {}
        for name, col in (("any", 0), ("tone_only", 1), ("seg", 2)):
            pos = R[:, col] == 1
            neg = R[:, 3] == 1
            out[name] = [auc(1 - R[pos, 4], 1 - R[neg, 4]), auc(1 - R[pos, 5], 1 - R[neg, 5])]
        return out
    return None


def make_cal(nat_z, q, floor):
    z = nat_z[np.isfinite(nat_z)]
    return {"knee": [float(np.quantile(z, q / 2)), float(np.quantile(z, q)), float(np.quantile(z, min(2 * q, 0.5)))],
            "width": WIDTH, "floor": floor, "native_quantile": q}


def step_dev():
    rep = {}
    for lang in LANGS:
        P = dev_parts(lang)
        base = metrics(P, None)
        grid = []
        for q in QS:
            for fl in FLOORS:
                cal = make_cal(P["nat_z"], q, fl)
                m = metrics(P, cal)
                ok = m["native_word_accept"] >= base["native_word_accept"] - 0.005
                grid.append({"q": q, "floor": fl, "ok": bool(ok), "cal": cal, **m})
        L0 = learners(lang, None)
        for g in grid:                       # every dev guard, per setting
            g["near_ok"] = all((g[v]["auc"] or 0) >= (base[v]["auc"] or 0) - TOL for v in ("swap_near", "swap_far"))
            g["planted_ok"] = g["planted_word_auc"] >= base["planted_word_auc"] - TOL
        oks = [g for g in grid if g["ok"] and g["near_ok"] and g["planted_ok"]]
        oks.sort(key=lambda g: (g["q"], -g["floor"]), reverse=True)        # strictest first
        pick, L = None, None
        for g in oks:
            L = learners(lang, g["cal"])
            ok_l = True
            if L:
                for k, v in L.items():
                    if isinstance(v, list) and v[0] is not None and v[1] < v[0] - 0.005:
                        ok_l = False
            if ok_l:
                pick = g
                break
        res = {"base": base, "pick": pick, "grid": grid, "near_misses": near_misses_of_contrast(P, lang, base),
               "learners_base": L0}
        if pick is not None:
            res["learners"] = L
            gain = (pick["contrast_pooled_hit5"] or 0) - (base["contrast_pooled_hit5"] or 0)
            res["checks"] = {"contrast_gain": gain}
            res["adopted"] = bool(gain >= 0.15)
        else:
            res["adopted"] = False
        rep[lang] = res
        b, p = base, pick
        print(lang, "base", {k: b[k] for k in ("native_word_accept", "contrast_pooled_hit5", "planted_word_auc")},
              {v: b[v]["auc"] for v in ("swap_near", "swap_far")}, flush=True)
        if p:
            print("   pick q", p["q"], "floor", p["floor"], {k: p[k] for k in ("native_word_accept", "contrast_pooled_hit5", "planted_word_auc")},
                  {v: p[v]["auc"] for v in ("swap_near", "swap_far")}, flush=True)
            print("   per contrast", {k: (b["contrast"][k]["word_hit5"], v["word_hit5"]) for k, v in p["contrast"].items()})
            print("   checks", res["checks"], "learners", res.get("learners"), "adopted", res["adopted"], flush=True)
            print("   near misses", res["near_misses"], flush=True)
    Path("results/tables/round18_dev.json").write_text(json.dumps(rep, indent=1, default=float))
    out = Path("pronunciada/data/calib/contrast_word")
    out.mkdir(exist_ok=True)
    for lang, r in rep.items():
        f = out / f"{lang}.json"
        if r.get("adopted"):
            f.write_text(json.dumps({"lang": lang, **r["pick"]["cal"],
                                     "fit_on": "FLEURS validation natives; contrast swaps of the audit (bench/round18/fit.py)"}, indent=1))
        elif f.exists():
            f.unlink()


def _cal(lang):
    p = Path(f"pronunciada/data/calib/contrast_word/{lang}.json")
    return json.loads(p.read_text()) if p.exists() else None


def test_audit(lang, cal):
    """FLEURS test audit swaps (+ ko tense swaps) at word level, before / after."""
    D = load_set("fleurs-test", lang)
    nat_wp, nat_z = [], []
    edits = defaultdict(list)
    for o in D.values():
        if o is None:
            continue
        nat_wp += o["clean"]["wp"][0.5]
        nat_z += o["clean"]["z"]
        for e in o["edits"]:
            edits[f"{e['contrast']}/{e['direction']}"].append((e["wp"][0.5][e["word"]], e["z"][e["word"]]))
    if lang == "ko":
        tz = [e for e in json.loads(Path("pronunciada/data/coach/contrast_heads.json").read_text())["langs"]["ko"]
              if e["contrast"] == "tense"]
        if tz:
            for d, v in tense_swaps(D, tz[0]).items():
                edits[f"tense/{d}"] = v
    P = {"nat_wp": np.array(nat_wp), "nat_z": np.array(nat_z), "edits": dict(edits),
         "near": {"swap_near": {"pos": [], "neg": []}, "swap_far": {"pos": [], "neg": []}, "orig": []},
         "cor_pos": [(1.0, np.nan)], "cor_neg": [(1.0, np.nan)]}
    b, a = metrics(P, None), metrics(P, cal)
    return {"native_word_accept": [b["native_word_accept"], a["native_word_accept"]],
            "contrast": {k: [b["contrast"][k]["word_hit5"], a["contrast"][k]["word_hit5"], v["n"]] for k, v in a["contrast"].items()},
            "pooled": [b["contrast_pooled_hit5"], a["contrast_pooled_hit5"]]}


def test_swapbench(lang, cal, n_boot=1000):
    from bench.data import load
    from bench.eval_swap import _metrics, azure_outputs, collect, evaluate

    O = load_set("swapbench", lang)
    recs = [r for r in load("fleurs-swap", split="test") if r["lang"] == lang]
    before = {k: v["on"] for k, v in O.items()}
    after = {}
    for k, v in O.items():
        on = dict(v["on"])
        z = v.get("z") or []
        if on.get("words") and len(z) == len(on["words"]):
            on["words"] = [(a_, b_, sc * factor(cal, zz)) for (a_, b_, sc), zz in zip(on["words"], z)]
        after[k] = on
    eb, ea = evaluate(recs, before)[lang], evaluate(recs, after)[lang]
    az = evaluate(recs, azure_outputs(recs)).get(lang)
    cb = collect(recs, before)[lang]["clusters"]
    ca = collect(recs, after)[lang]["clusters"]
    keys = sorted(cb)
    variants = ("swap_near", "swap_far")
    rng = np.random.default_rng(0)
    diffs = defaultdict(list)
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        mb = _metrics([cb[keys[i]] for i in idx], [v for v in variants])
        ma = _metrics([ca[keys[i]] for i in idx], [v for v in variants])
        for m in ("swap_near:auc", "swap_far:auc", "swap_near:hit@5%FA", "swap_near:right_word"):
            if m in mb and m in ma:
                diffs[m].append(ma[m] - mb[m])
    p = {m: float(np.mean(np.array(v) <= 0)) for m, v in diffs.items()}
    pick = ("swap_near:auc", "swap_near:hit@5%FA", "swap_near:right_word", "swap_far:auc", "swap_far:hit@5%FA",
            "swap_far:right_word", "native:fa<60")
    g = lambda e: {m: e["metrics"].get(m, {}).get("value") for m in pick}  # noqa: E731
    return {"before": g(eb), "after": g(ea), "azure": g(az) if az else None,
            "p_after_not_better": p, "ci_after_minus_before": {m: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
                                                              for m, v in diffs.items()}}


def test_lang10(lang, cal):
    from bench.eval_languages import S_LIST, eval_lang
    from pronunciada.calibrate import config

    D = load_set("lang10", lang)
    for o in D.values():                      # latency is not part of this comparison
        if "error" not in o:
            o.setdefault("latency_ms", 0.0)
            o.setdefault("timing", {})
    cfg = config()
    b0 = 0.0 if cfg["sound_model"].get(lang, cfg["universal_sound_model"]) != cfg["universal_sound_model"] \
        else float(cfg["bias"].get(lang, 0.0))
    import copy

    A = copy.deepcopy(D)
    for o in A.values():
        if "error" in o:
            continue
        for r in [o["clean"]] + list(o["cor"].values()):
            z = r.get("z") or []
            for var in r["V"]:
                for s in S_LIST:
                    wp = r["V"][var]["wp"][s]
                    if len(z) == len(wp):
                        r["V"][var]["wp"][s] = np.array([p * factor(cal, zz, s) for p, zz in zip(wp, z)], np.float32)
    eb, ea = eval_lang(lang, D, b0), eval_lang(lang, A, b0)
    g = lambda e: {"native_words": e["native"]["0.5"]["words"], "word_auc": e["word"]["auc"],  # noqa: E731
                   "word_hit5": e["word"]["hit_at_5pct"], "lowest": e["word"]["lowest_is_corrupted"],
                   "native_sounds": e["native"]["0.5"]["sounds"], "sound_auc": e["sound"]["auc"]}
    return {"before": g(eb), "after": g(ea), "full_after": ea}


def step_test():
    rep = {}
    for lang in LANGS:
        cal = _cal(lang)
        if cal is None:
            continue
        r = {"audit": test_audit(lang, cal), "swapbench": test_swapbench(lang, cal), "lang10": test_lang10(lang, cal)}
        if lang in ("en", "zh"):
            r["learners_test"] = learners(lang, cal, "test")
        sb = r["swapbench"]
        regress = ((sb["after"]["swap_near:auc"] or 0) < (sb["before"]["swap_near:auc"] or 0)
                   and sb["p_after_not_better"].get("swap_near:auc", 0) > 0.9
                   and r["audit"]["pooled"][1] <= r["audit"]["pooled"][0])
        r["revert"] = bool(regress)
        rep[lang] = {k: v for k, v in r.items() if k != "lang10"} | {"lang10": {k: r["lang10"][k] for k in ("before", "after")}}
        print(lang, json.dumps({k: v for k, v in rep[lang].items()}, default=float)[:1800], flush=True)
    Path("results/tables/round18_test.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    {"dev": step_dev, "test": step_test}[sys.argv[1] if len(sys.argv) > 1 else "dev"]()
