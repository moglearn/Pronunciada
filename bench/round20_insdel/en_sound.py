"""Round 20 (insdel), English sounds: does "dropped in the greedy decode" help the sound error signal?

    python -m bench.round20_insdel.en_sound          # speechocean762 TRAIN only (collect.py --lang en --set so762-train)

Signal = 1 - P(sound ok) at strictness 0.5 from the uncapped sound_B outputs (the signal the
L2-ARCTIC / EpaDB marked-sound test reads). Candidate: z' = z - a * d, d = the sound's
greedy-decode drop indicator (plain: 1; weighted: its native deletion cost), lower only.
Labels: "marked" = expert 0-2 score < 1.5; "deleted" = the experts' <DEL>; correlation = Pearson
of P with the 0-2 score. Speaker-grouped 5-fold CV: a (and the variant) chosen on four folds by
marked-sound AUC, applied to the fifth; pooled out-of-fold outputs; paired speaker bootstrap.
Writes results/tables/round20_insdel_en_so762.json.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from bench.round20_insdel import collect as CO
from bench.round20_insdel import core as K
from bench.round20_insdel.ablate import ROOT, fold_of

A_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)
CFG = ("plain", "weighted", "weighted_top2", "weighted_noalt")


def _auc(y, s):
    from bench.round19_ja.unit_eval import roc_auc

    return roc_auc(y, s)


def table(recs, cfgname, costs):
    rows = defaultdict(list)
    for it, r in recs.items():
        lab = r["labels"]
        f = K.rec_features(r, K.CFGS[cfgname], costs)
        dropped = f["dropped"].astype(float)
        if costs is not None:
            from bench.round20_insdel.greedy_match import base, units

            w = np.array([costs.d(base(units(p)[0])) if units(p) else 1.0 for p in r["phones"]["ipa"]])
            dropped = dropped * w
        zl, zs = np.asarray(r["coach_zl"], np.float64), np.asarray(r["coach_zs"], np.float64)
        z = 0.5 * zl + 0.5 * np.minimum(zs, zl)
        ok = np.isfinite(lab["phone_score"])
        n = int(ok.sum())
        rows["item"] += [it] * n
        rows["spk"] += [r["meta"]["speaker"]] * n
        rows["z"] += list(z[ok])
        rows["d"] += list(dropped[ok])
        rows["score"] += list(lab["phone_score"][ok])
        rows["del"] += list(lab["phone_del"][ok])
    return {k: np.asarray(v) for k, v in rows.items()}


def metrics(T, z):
    p = K.sig(z)
    return {"auc_marked": _auc(T["score"] < 1.5, 1 - p), "auc_deleted": _auc(T["del"], 1 - p),
            "auc_lt2": _auc(T["score"] < 2, 1 - p), "pearson": float(np.corrcoef(p, T["score"])[0, 1])}


def main():
    recs = {k: r for k, r in CO.load("en", "so762-train").items() if "error" not in r}
    spk = {it: r["meta"]["speaker"] for it, r in recs.items()}
    folds = {it: fold_of(s, 5) for it, s in spk.items()}
    out = {"n_clips": len(recs)}
    # drop rates (diagnosis)
    T0 = table(recs, "plain", None)
    out["drop_rate"] = {"expert_correct(2.0)": float(np.mean(T0["d"][T0["score"] >= 2] > 0)),
                        "marked(<1.5)": float(np.mean(T0["d"][T0["score"] < 1.5] > 0)),
                        "expert_deleted": float(np.mean(T0["d"][T0["del"]] > 0)),
                        "n_sounds": int(len(T0["d"])), "n_deleted": int(T0["del"].sum()),
                        "n_marked": int((T0["score"] < 1.5).sum())}
    out["drop_alone_auc"] = {"marked": _auc(T0["score"] < 1.5, T0["d"]), "deleted": _auc(T0["del"], T0["d"])}
    print(json.dumps(out), flush=True)
    # weighted variants: insertion / deletion costs from the fit folds' clips (learner speech; no
    # native English pool is collected here)
    oof = {}
    picks = []
    base_T = T0
    zcv = base_T["z"].copy()
    item_idx = defaultdict(list)
    for i, it in enumerate(base_T["item"]):
        item_idx[it].append(i)
    for f in range(5):
        fit = {it: r for it, r in recs.items() if folds[it] != f}
        best = (-1.0, None)
        tabs = {}
        for c in CFG:
            costs = None
            if K.CFGS[c]["weighted"]:
                costs = K.native_table(list(fit.values()), K.CFGS[c])
            tabs[c] = table(recs, c, costs)
            Tc = tabs[c]
            m = np.array([folds[it] != f for it in Tc["item"]])
            for a in A_GRID:
                auc = _auc(Tc["score"][m] < 1.5, 1 - K.sig(Tc["z"][m] - a * Tc["d"][m]))
                if auc > best[0] + 1e-9:
                    best = (auc, (c, a))
        c, a = best[1]
        picks.append({"fold": f, "fit_auc": best[0], "cfg": c, "a": a})
        print("fold", f, best, flush=True)
        Tc = tabs[c]
        m = np.array([folds[it] == f for it in Tc["item"]])
        zcv[m] = Tc["z"][m] - a * Tc["d"][m]
    out["folds"] = picks
    m0, m1 = metrics(base_T, base_T["z"]), metrics(base_T, zcv)
    out["shipped"], out["insdel_oof"] = m0, m1
    # paired speaker bootstrap
    rng = np.random.default_rng(0)
    sp = base_T["spk"]
    keys = sorted(set(sp))
    idx = defaultdict(list)
    for i, s in enumerate(sp):
        idx[s].append(i)
    D = defaultdict(list)
    for _ in range(1000):
        pick = np.concatenate([idx[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        Tb = {k: v[pick] for k, v in base_T.items()}
        a0, a1 = metrics(Tb, base_T["z"][pick]), metrics(Tb, zcv[pick])
        for k in a0:
            D[k].append(a1[k] - a0[k])
    out["paired"] = {k: {"diff": m1[k] - m0[k], "ci": [float(np.quantile(v, .025)), float(np.quantile(v, .975))],
                         "p": float(min(1, 2 * min(np.mean(np.asarray(v) <= 0), np.mean(np.asarray(v) >= 0))))}
                     for k, v in D.items()}
    print(json.dumps({k: out[k] for k in ("shipped", "insdel_oof", "paired")}, indent=1, default=float))
    (ROOT / "results" / "tables" / "round20_insdel_en_so762.json").write_text(json.dumps(out, indent=1, default=float))


def valcor():
    """FLEURS validation (en) natives with the bench.eval_languages controlled errors: sound-level
    AUC of the planted sounds of each kind against every sound of the clean clips; clean sounds
    below P 0.5. Speaker groups = FLEURS sentence ids; 5-fold CV as in main()."""
    recs = {k: r for k, r in CO.load("en", "val-cor").items() if "error" not in r}
    grp = {it: r["meta"]["speaker"] for it, r in recs.items()}
    folds = {it: fold_of(g, 5) for it, g in grp.items()}

    def tab(c, costs):
        rows = defaultdict(list)
        for it, r in recs.items():
            kind = r["meta"]["cor_kind"]
            if kind != "clean" and not r.get("phone_wrong"):
                continue
            f = K.rec_features(r, K.CFGS[c], costs)
            d = f["dropped"].astype(float)
            if costs is not None:
                from bench.round20_insdel.greedy_match import base, units

                d = d * np.array([costs.d(base(units(p)[0])) if units(p) else 1.0 for p in r["phones"]["ipa"]])
            zl, zs = np.asarray(r["coach_zl"], np.float64), np.asarray(r["coach_zs"], np.float64)
            z = 0.5 * zl + 0.5 * np.minimum(zs, zl)
            if kind == "clean":
                sel = np.ones(len(z), bool)
                lab = np.zeros(len(z), bool)
            else:
                lab = np.asarray(r["phone_wrong"], bool)
                sel = lab
            n = int(sel.sum())
            rows["item"] += [it] * n
            rows["grp"] += [grp[it]] * n
            rows["kind"] += [kind] * n
            rows["y"] += list(lab[sel])
            rows["z"] += list(z[sel])
            rows["d"] += list(d[sel])
        return {k: np.asarray(v) for k, v in rows.items()}

    def met(T, z, m=None):
        m = np.ones(len(z), bool) if m is None else m
        y, s, kd = T["y"][m], 1 - K.sig(z[m]), T["kind"][m]
        o = {"auc_all": _auc(y, s), "clean_lt05": float(np.mean(K.sig(z[m][kd == "clean"]) < 0.5))}
        for k in sorted(set(kd) - {"clean"}):
            q = (kd == k) | (kd == "clean")
            o[f"auc_{k}"] = _auc(y[q], s[q])
        return o

    T0 = tab("plain", None)
    out = {"drop_rate_clean": float(np.mean(T0["d"][T0["kind"] == "clean"] > 0)),
           "drop_rate_by_kind": {k: float(np.mean(T0["d"][T0["kind"] == k] > 0)) for k in sorted(set(T0["kind"]))}}
    zcv = T0["z"].copy()
    picks = []
    for f in range(5):
        fit = [r for it, r in recs.items() if folds[it] != f and r["meta"]["cor_kind"] == "clean"]
        best = (-1.0, None)
        tabs = {}
        for c in CFG:
            costs = K.native_table(fit, K.CFGS[c]) if K.CFGS[c]["weighted"] else None
            Tc = tabs[c] = tab(c, costs)
            m = np.array([folds[it] != f for it in Tc["item"]])
            n0 = np.sum(K.sig(Tc["z"][m & (Tc["kind"] == "clean")]) < 0.5)
            for a in A_GRID:
                z = Tc["z"] - a * Tc["d"]
                if np.sum(K.sig(z[m & (Tc["kind"] == "clean")]) < 0.5) > n0:
                    continue
                auc = _auc(Tc["y"][m], 1 - K.sig(z[m]))
                if auc > best[0] + 1e-9:
                    best = (auc, (c, a))
        c, a = best[1]
        picks.append({"fold": f, "fit_auc": best[0], "cfg": c, "a": a})
        print("valcor fold", f, best, flush=True)
        Tc = tabs[c]
        m = np.array([folds[it] == f for it in Tc["item"]])
        zcv[m] = Tc["z"][m] - a * Tc["d"][m]
    out["folds"] = picks
    out["shipped"], out["insdel_oof"] = met(T0, T0["z"]), met(T0, zcv)
    rng = np.random.default_rng(0)
    keys = sorted(set(T0["grp"]))
    idx = defaultdict(list)
    for i, g in enumerate(T0["grp"]):
        idx[g].append(i)
    D = defaultdict(list)
    for _ in range(1000):
        pick = np.concatenate([idx[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        Tb = {k: v[pick] for k, v in T0.items()}
        a0, a1 = met(Tb, T0["z"][pick]), met(Tb, zcv[pick])
        for k in a0:
            if np.isfinite(a0[k]) and np.isfinite(a1[k]):
                D[k].append(a1[k] - a0[k])
    out["paired"] = {k: {"ci": [float(np.quantile(v, .025)), float(np.quantile(v, .975))],
                         "p": float(min(1, 2 * min(np.mean(np.asarray(v) <= 0), np.mean(np.asarray(v) >= 0))))}
                     for k, v in D.items()}
    print(json.dumps(out, indent=1, default=float))
    (ROOT / "results" / "tables" / "round20_insdel_en_valcor.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    import sys

    valcor() if sys.argv[1:] == ["valcor"] else main()
