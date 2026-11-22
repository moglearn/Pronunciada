"""Round 20 (insdel): offline ablation on the collected dev sets (bench/round20_insdel/collect.py).

    python -m bench.round20_insdel.ablate --lang ja         # writes results/tables/round20_insdel_<lang>.json

1. Feature alone: unit-level AUC (indel bucket vs all correct units, and every planted unit vs all
   correct units) of each feature x variant, the unit signal being the largest feature over the
   words that overlap the unit. Weighted cost tables come from the control clips of the OTHER
   speaker fold (2 folds by speaker), so no feature value of a clip used its own clip.
2. As an extra lower-only term on the shipped word log-odds, z - a * max(0, f - k), speaker-grouped
   4-fold CV: per held-out fold the cost table, the feature, a and k are chosen on the other three
   folds (highest dev AUC among the settings whose fit-fold native control words below 0.6 do not
   rise); out-of-fold probabilities are pooled and compared with the shipped ones (paired speaker
   bootstrap, 1000 draws).
3. Frozen candidate = the same rule on the whole dev set; its dev guards (native pools below
   0.5 / 0.6, far swaps, whole-wrong-word, controlled errors, cv-real dev) against the shipped
   outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.round20_insdel import collect as CO
from bench.round20_insdel import core as K

ROOT = Path(__file__).resolve().parents[2]
A_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
K_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
TERM_FEATS = ("sum", "rate", "ins", "del", "nedit", "edge_sum", "mass")


def fold_of(group: str, n: int) -> int:
    return int(hashlib.md5(str(group).encode()).hexdigest(), 16) % n


# ----------------------------------------------------------------------------- language adapters

def _auc(y, s):
    from bench.round19_ja.unit_eval import roc_auc

    return roc_auc(y, s)


def _fa(y, s):
    from bench.round19_ja.unit_eval import fa_at

    return fa_at(y, s)


class Lang:
    """Dev records and unit rows for one language."""

    def __init__(self, lang: str):
        self.lang = lang
        self.sets = {}
        names = {"ja": ("dev", "devnat", "devswap", "cvdev"), "ko": ("dev", "val-cor", "zeroth-test")}.get(
            lang, ("dev", "valnat"))
        for s in names:
            d = CO.load(lang, s)
            self.sets[s] = {k: r for k, r in d.items() if "error" not in r}
        if lang == "ko":     # the round-19 Korean dev rule: CV clips only if Whisper heard the prompt (CER <= 0.1)
            from bench.round19ko.offline import keep_items
            from bench.round19ko.cv_pool import work as ko_work

            self.sets["dev"] = keep_items(self.sets["dev"], ko_work() / "asr.jsonl")
        self.dev = self.sets["dev"]
        self.items = sorted(self.dev)
        self._units = None

    def group(self, it: str) -> str:
        m = self.dev[it]["meta"]
        return m.get("group") or m.get("speaker") or it

    def label(self, it: str):
        return self.dev[it]["meta"].get("label")

    def is_control(self, it: str) -> bool:
        return not self.label(it)

    # unit rows: list of (item, group, label, bucket, [word indices overlapping])
    def units(self):
        if self._units is not None:
            return self._units
        rows = []
        if self.lang == "ja":
            from bench.round19_ja.unit_eval import _PUNCT

            for it in self.items:
                r = self.dev[it]
                m = r["meta"]
                lab = m.get("label") or {}
                if lab and not lab.get("word_level_scorable", True):
                    continue
                text, cur, spans = m["text"], 0, []
                for j, wt in enumerate(r["words"]["text"]):
                    t = _PUNCT.sub("", wt)
                    q = text.find(t, cur) if t else -1
                    if q >= 0:
                        spans.append((q, q + len(t), j))
                        cur = q + len(t)
                pos = lab.get("word_index") if lab else None
                b = lab.get("bucket") if lab else "control"
                for k, u in enumerate(m["units"]):
                    ws = [j for s, e, j in spans if s < u["end"] and e > u["start"]]
                    rows.append((it, self.group(it), int(k == pos), b if k == pos else "control", ws))
        else:
            from bench import textswap as TS
            from pronunciada.text import get_g2p

            g = get_g2p(self.lang)
            cache = {}
            for it in self.items:
                r = self.dev[it]
                m = r["meta"]
                t = m.get("target") or m.get("text")
                if t not in cache:
                    cache[t] = [tuple(u.span) for u in TS.units(g(t), self.lang)]
                lab = m.get("label") or {}
                pos = lab.get("word_index") if lab else None
                for k, (a, b) in enumerate(cache[t]):
                    ws = [j for j, (c, d) in enumerate(r["words"]["span"]) if c < b and d > a]
                    rows.append((it, self.group(it), int(k == pos), (lab.get("bucket") if k == pos else "control"), ws))
        self._units = rows
        return rows

    def unit_signal(self, P: dict) -> np.ndarray:
        """P: item -> word probabilities; signal = 1 - lowest probability over the unit's words."""
        return np.array([1.0 - min(P[it][j] for j in ws) if ws else 1.0 for it, _, _, _, ws in self.units()])

    # native word pools for the guard: name -> list of (set, item)
    def native_pools(self):
        if self.lang == "ja":
            nat = self.sets["devnat"]
            return {"fleurs": [("devnat", k) for k, r in nat.items() if r["meta"].get("source") == "fleurs"],
                    "cv": [("devnat", k) for k, r in nat.items() if r["meta"].get("source") == "cv"]}
        if self.lang == "ko":
            vc = self.sets["val-cor"]
            return {"fleurs-val": [("val-cor", k) for k, r in vc.items() if r["meta"].get("cor_kind") == "clean"],
                    "cv-ctl": [("dev", k) for k, r in self.dev.items() if r["meta"]["src"] == "cv" and not r["meta"]["label"]],
                    "zeroth": [("zeroth-test", k) for k in self.sets["zeroth-test"]]}
        vn = self.sets["valnat"]
        return {"fleurs-val": [("valnat", k) for k in vn],
                "cv-ctl": [("dev", k) for k, r in self.dev.items() if r["meta"].get("src") == "cv" and not r["meta"].get("label")]}


# ----------------------------------------------------------------------------- metrics

def core_metrics(y, s, b, sel=None):
    y, s, b = np.asarray(y), np.asarray(s), np.asarray(b)
    if sel is not None:
        y, s, b = y[sel], s[sel], b[sel]
    out = {"auc": _auc(y, s), "fa80": _fa(y, s)}
    neg = s[y == 0]
    for bk in sorted(set(b[y == 1])):
        p = s[(y == 1) & (b == bk)]
        out[f"b_{bk}"] = _auc(np.r_[np.ones(len(p)), np.zeros(len(neg))], np.r_[p, neg])
    return out


def paired_boot(groups, y, s0, s1, b, n=1000, seed=0):
    groups = np.asarray(groups)
    keys = sorted(set(groups))
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    rng = np.random.default_rng(seed)
    d = defaultdict(list)
    lv = defaultdict(lambda: ([], []))
    for _ in range(n):
        pick = np.concatenate([idx[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        m0 = core_metrics(y, s0, b, pick)
        m1 = core_metrics(y, s1, b, pick)
        for k in m0:
            if k in m1 and np.isfinite(m0[k]) and np.isfinite(m1[k]):
                d[k].append(m1[k] - m0[k])
                lv[k][0].append(m0[k])
                lv[k][1].append(m1[k])
    out = {}
    for k, v in d.items():
        v = np.asarray(v)
        better = v > 0 if k != "fa80" else v < 0
        out[k] = {"diff_ci": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
                  "p": float(min(1.0, 2 * min(np.mean(v <= 0), np.mean(v >= 0)))),
                  "p_not_better": float(1 - np.mean(better)),
                  "ci0": [float(np.quantile(lv[k][0], 0.025)), float(np.quantile(lv[k][0], 0.975))],
                  "ci1": [float(np.quantile(lv[k][1], 0.025)), float(np.quantile(lv[k][1], 0.975))]}
    return out


# ----------------------------------------------------------------------------- features

def feats_for(L: Lang, recs: dict, cfgname: str, costs) -> dict:
    cfg = K.CFGS[cfgname]
    return {k: K.rec_features(r, cfg, costs) for k, r in recs.items()}


def control_recs(L: Lang, items) -> list:
    return [L.dev[i] for i in items if L.is_control(i)]


def alone(L: Lang, out: dict):
    U = L.units()
    y = np.array([u[2] for u in U])
    b = np.array([u[3] for u in U])
    folds = {it: fold_of(L.group(it), 2) for it in L.items}
    base = {it: np.asarray(L.dev[it]["words"]["prob"]) for it in L.items}
    s0 = L.unit_signal(base)
    res = {"shipped": core_metrics(y, s0, b)}
    print("shipped", json.dumps({k: round(v, 4) for k, v in res["shipped"].items()}), flush=True)
    for cfgname, cfg in K.CFGS.items():
        F = {}
        for f in (0, 1):
            costs = K.native_table(control_recs(L, [i for i in L.items if folds[i] != f]), cfg) if cfg["weighted"] else None
            F.update(feats_for(L, {i: L.dev[i] for i in L.items if folds[i] == f}, cfgname, costs))
        for fe in K.FEATS:
            P = {it: np.exp(-F[it][fe]) for it in L.items}
            m = core_metrics(y, L.unit_signal(P), b)
            res[f"{cfgname}:{fe}"] = m
            print(cfgname, fe, json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)
    out["alone"] = res


def native_words(L: Lang, P_by_set: dict, pools: dict) -> dict:
    """Share of native words below 0.5 / 0.6 per pool."""
    out = {}
    for name, lst in pools.items():
        p = np.concatenate([np.asarray(P_by_set[s][k]) for s, k in lst]) if lst else np.zeros(0)
        out[name] = {"n": int(len(p)), "lt05": int(np.sum(p < 0.5)), "lt06": int(np.sum(p < 0.6))}
    return out


def choose(L: Lang, fit_items, F_by_cfg, y, b, U, base):
    """Best (cfg, feature, a, k) on fit_items by AUC with fit-fold control words below 0.6 not rising."""
    fit = set(fit_items)
    sel = np.array([u[0] in fit for u in U])
    ctl = [i for i in fit_items if L.is_control(i)]
    p0 = np.concatenate([base[i] for i in ctl])
    n0 = int(np.sum(p0 < 0.6))
    s0 = L.unit_signal(base)
    best = (core_metrics(y, s0, b, sel)["auc"], None)
    for cfgname, F in F_by_cfg.items():
        for fe in TERM_FEATS:
            for a in A_GRID:
                for k in K_GRID:
                    P = {i: K.apply_term(base[i], F[i][fe], a, k) for i in fit_items}
                    pc = np.concatenate([P[i] for i in ctl])
                    if int(np.sum(pc < 0.6)) > n0:
                        continue
                    Pf = dict(base)
                    Pf.update(P)
                    auc = core_metrics(y, L.unit_signal(Pf), b, sel)["auc"]
                    if auc > best[0] + 1e-9:
                        best = (auc, (cfgname, fe, a, k))
    return best


def term_cv(L: Lang, out: dict, cfgs=("plain", "weighted", "weighted_noalt", "weighted_top2"), nf: int = 4):
    U = L.units()
    y = np.array([u[2] for u in U])
    b = np.array([u[3] for u in U])
    g = np.array([u[1] for u in U])
    base = {it: np.asarray(L.dev[it]["words"]["prob"], np.float64) for it in L.items}
    folds = {it: fold_of(L.group(it), nf) for it in L.items}
    oof = dict(base)
    picks = []
    for f in range(nf):
        fit_items = [i for i in L.items if folds[i] != f]
        hold = [i for i in L.items if folds[i] == f]
        F_by = {}
        for c in cfgs:
            costs = K.native_table(control_recs(L, fit_items), K.CFGS[c]) if K.CFGS[c]["weighted"] else None
            F_by[c] = feats_for(L, {i: L.dev[i] for i in L.items}, c, costs)
        auc, pick = choose(L, fit_items, F_by, y, b, U, base)
        picks.append({"fold": f, "fit_auc": auc, "pick": pick})
        print("fold", f, auc, pick, flush=True)
        if pick is not None:
            c, fe, a, k = pick
            for i in hold:
                oof[i] = K.apply_term(base[i], F_by[c][i][fe], a, k)
    s0, s1 = L.unit_signal(base), L.unit_signal(oof)
    m0, m1 = core_metrics(y, s0, b), core_metrics(y, s1, b)
    pb = paired_boot(g, y, s0, s1, b)
    print("OOF shipped", json.dumps({k: round(v, 4) for k, v in m0.items()}))
    print("OOF insdel ", json.dumps({k: round(v, 4) for k, v in m1.items()}))
    print("paired", json.dumps({k: [round(x, 4) for x in v["diff_ci"]] + [round(v["p"], 4)] for k, v in pb.items()}))
    ctl = [i for i in L.items if L.is_control(i)]
    nat0 = int(np.sum(np.concatenate([base[i] for i in ctl]) < 0.6))
    nat1 = int(np.sum(np.concatenate([oof[i] for i in ctl]) < 0.6))
    out["cv"] = {"folds": picks, "shipped": m0, "insdel_oof": m1, "paired": pb,
                 "dev_control_words_lt06": [nat0, nat1]}
    return oof


def frozen(L: Lang, out: dict, cfgs=("plain", "weighted", "weighted_noalt", "weighted_top2")):
    U = L.units()
    y = np.array([u[2] for u in U])
    b = np.array([u[3] for u in U])
    base = {it: np.asarray(L.dev[it]["words"]["prob"], np.float64) for it in L.items}
    F_by, tables = {}, {}
    for c in cfgs:
        tables[c] = K.native_table(control_recs(L, L.items), K.CFGS[c]) if K.CFGS[c]["weighted"] else None
        F_by[c] = feats_for(L, L.dev, c, tables[c])
    auc, pick = choose(L, L.items, F_by, y, b, U, base)
    out["frozen"] = {"pick": pick, "dev_auc_in_sample": auc}
    print("frozen pick", pick, auc, flush=True)
    if pick is None:
        return None
    c, fe, a, k = pick
    T = tables[c]
    out["frozen"]["costs"] = None if T is None else {"ins": T.ins, "del": T.dele}
    # guards on the other sets with the frozen candidate
    P0 = {s: {k2: np.asarray(r["words"]["prob"], np.float64) for k2, r in d.items()} for s, d in L.sets.items()}
    P1 = {}
    for s, d in L.sets.items():
        Fs = F_by[c] if s == "dev" else feats_for(L, d, c, T)
        P1[s] = {k2: K.apply_term(P0[s][k2], Fs[k2][fe], a, k) for k2 in d}
    pools = L.native_pools()
    out["frozen"]["native_before"] = native_words(L, P0, pools)
    out["frozen"]["native_after"] = native_words(L, P1, pools)
    print("natives before", out["frozen"]["native_before"])
    print("natives after ", out["frozen"]["native_after"])
    out["frozen"]["guards"] = lang_guards(L, P0, P1)
    print("guards", json.dumps(out["frozen"]["guards"]), flush=True)
    return P0, P1


def lang_guards(L: Lang, P0, P1) -> dict:
    g = {}
    if L.lang == "ja":
        from bench.round19_ja.select2 import metrics

        def D(P):
            o = {}
            for s in ("dev", "devnat", "devswap", "cvdev"):
                o[s] = [{**L.sets[s][k]["meta"], "words": [{"text": t, "span": list(sp), "base": float(p), "zlen": None,
                                                            "zid": None, "mm": None}
                                                           for t, sp, p in zip(L.sets[s][k]["words"]["text"],
                                                                               L.sets[s][k]["words"]["span"], P[s][k])]}
                        for k in sorted(L.sets[s])]
            return o
        m0, m1 = metrics(D(P0), {}, None), metrics(D(P1), {}, None)
        g = {"before": m0, "after": m1}
    elif L.lang == "ko":
        vc = L.sets["val-cor"]

        def planted(P, kind=None):
            pos, neg = [], []
            for k, r in vc.items():
                ck = r["meta"].get("cor_kind")
                if ck == "clean":
                    neg += list(P["val-cor"][k])
                elif r.get("word_wrong") and (kind is None or ck == kind):
                    pos += [P["val-cor"][k][j] for j, w in enumerate(r["word_wrong"]) if w]
            yy = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            return _auc(yy, 1 - np.r_[pos, neg])
        from bench.eval_languages import KINDS

        g = {"planted_auc": [planted(P0), planted(P1)]}
        for kd in KINDS:
            g[f"planted_auc_{kd}"] = [planted(P0, kd), planted(P1, kd)]
    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--skip-alone", action="store_true")
    a = ap.parse_args(argv)
    L = Lang(a.lang)
    out = {"lang": a.lang, "n_items": len(L.items)}
    if not a.skip_alone:
        alone(L, out)
    term_cv(L, out)
    frozen(L, out)
    p = ROOT / "results" / "tables" / f"round20_insdel_{a.lang}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", p)


if __name__ == "__main__":
    main()
