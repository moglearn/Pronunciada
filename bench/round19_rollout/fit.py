"""Round 19 rollout: place a language's sound-identity word rule and decide whether it ships.

    python -m bench.round19_rollout.fit --lang fr evidence     # head evidence for pool / dev / devcor / val
    python -m bench.round19_rollout.fit --lang fr select [--save]

Both forms of round 19 are tried on the same evidence (s = identity margin minus a class offset):
  knee  (German): factor 1 while the word's lowest s is above a knee (a native quantile of word minima),
        then linearly down to floor over width. Grid: offset none / class 5 % native quantile x knee
        quantile 0.5 / 1 / 2 / 5 % x floor 0.3 / 0.6 x width 2 / 4.
  step  (Korean): word log-odds += c . [min(s,0), min(s+4,0), mean min(s,4), soft count s<0, 1], capped at
        +3, no raise when the word's lowest s < no_raise_below; c by logistic regression with the old
        log-odds as offset on the dev FIT fold (changed word = 1, every other word = 0) plus the
        controlled errors of the fit fold (L2 10). Grid: positive weight 1 / 2 x no_raise_below -2 / 0 /
        never raise.
Guards (fixed before looking; all on data outside every benchmark and held-out set):
  G1 native pool (Common Voice pool speakers + dev fit-fold controls): words with P < 0.5 and P < 0.6
     rise by <= 0.3 pt;
  G2 FLEURS validation natives (swap-set originals and protocol clean clips) P < 0.6 rise <= 0.3 pt, and
     Common Voice dev natives (all dev CV controls) P < 0.6 rise <= 0.3 pt;
  G3 controlled sound errors (dev fit fold) word AUC not lower (1e-4);
  G4 dev fit fold: far swaps (sub_gross) and whole-wrong-word items (word_omission) AUC not lower (1e-4);
     FLEURS validation far-swap AUC not lower (1e-4);
  G5 stress languages: FLEURS validation planted stress-error word AUC falls by <= 0.005.
Pick: among settings passing every guard, the lowest dev FIT-fold FA@80 (ties: higher AUC), each form;
the form with the lower fit-fold FA@80 wins. Ships only if on the CHECK fold AUC is higher and FA@80
not higher than without the head (else nothing ships for the language).
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.round19_rollout.collect import iter_rows
from bench.round19_rollout.sources import work
from bench.round19_rollout.train import SPEC, Ref
from pronunciada import identity as ID

SETS = ("cvpool", "dev", "devcor", "val")
STRESS_LANGS = {"en", "es", "it", "pt", "ru", "de"}


def load_head(lang):
    z = dict(np.load(work(lang) / "head.npz", allow_pickle=False))
    H = {k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in z.items()}
    H["index"] = {str(c): j for j, c in enumerate(H["classes"].tolist())}
    return H


def cmd_evidence(lang):
    H, spec = load_head(lang), SPEC[lang]
    for name in SETS:
        out = []
        for r in iter_rows(lang, name):
            s, cls = ID.raw_evidence(Ref(r["ph"]), r["V"].astype(np.float32), spec, H)
            r = {k: v for k, v in r.items() if k != "V"}
            r.update(s=s, cls=cls)
            out.append(r)
        (work(lang) / f"ev_{name}.pkl").write_bytes(pickle.dumps(out))
        print(name, len(out), flush=True)


def load_ev(lang, name):
    return pickle.loads((work(lang) / f"ev_{name}.pkl").read_bytes())


# ----------------------------------------------------------------------------- word values

def word_vals(r, off):
    """Per reference word: the array of its judged s values (offset applied)."""
    nw = len(r["prob"])
    out = [[] for _ in range(nw)]
    for x, c, w in zip(r["s"], r["cls"], r["word"]):
        if c is not None and np.isfinite(x):
            out[int(w)].append(x - off.get(c, 0.0))
    return [np.array(v) for v in out]


def pre_prob(r):
    """Word P before the presence factor (the identity rule is applied there, as in the scorer)."""
    pres = np.where(np.isfinite(r["present"]), r["present"], 1.0)
    return np.clip(r["prob"].astype(np.float64) / np.maximum(pres, 1e-9), 0.0, 1.0), pres


def new_probs(r, rule, off):
    pp, pres = pre_prob(r)
    vals = word_vals(r, off)
    out = np.array([ID.apply_word(rule, v, float(p), 0.5) if len(v) else float(p) for v, p in zip(vals, pp)])
    return out * pres


# ----------------------------------------------------------------------------- dev metric (Lingolix rules)

def dev_items(lang):
    from bench.round19_rollout.devset import manifest

    return {it["id"]: it for it in manifest(lang)}


def bench_rec(r, probs=None):
    """The Lingolix-shaped word list of a dev row, with reference words' scores replaced by probs."""
    if probs is None:
        return {"id": r["id"], "words": r["words"]}
    ws, k = [], 0
    nonins = [w for w in r["words"] if w.get("error_type") != "Insertion"]
    if len(nonins) != len(probs):
        return {"id": r["id"], "words": r["words"]}
    for w in r["words"]:
        if w.get("error_type") == "Insertion":
            ws.append(w)
            continue
        ws.append({**w, "score": round(min(1.0, max(0.0, float(probs[k]))) * 100, 1) / 100})
        k += 1
    return {"id": r["id"], "words": ws}


def dev_metrics(items, rows, probs_of=None, with_units=False):
    from bench.round19.evaluate import _metrics
    from bench.third_party import lingolix as L

    resp = {r["id"]: bench_rec(r, probs_of(r) if probs_of else None) for r in rows}
    its = [items[r["id"]] for r in rows]
    us = L.collect(its, resp)["units"]
    m = _metrics(us)
    return (m, us) if with_units else m


# ----------------------------------------------------------------------------- guards

def auc(pos, neg):
    from bench.metrics import roc_auc

    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(roc_auc(np.r_[np.ones(len(pos)), np.zeros(len(neg))], -np.r_[pos, neg]))


def val_metrics(val, probs_of):
    """FLEURS validation guard rows (the German guards_dev measurements)."""
    from bench.round19.guards_dev import _metrics_of

    d = {"swap": [], "pros": []}
    for r in val:
        p = probs_of(r)
        m = r["meta"]
        if m["kind"] == "swap":
            d["swap"].append({"id": r["id"], "variant": m["variant"], "k": m["k"], "spans": r["spans"],
                              "words": m["units"], "prob": list(p)})
        else:
            d["pros"].append({"utt": m["utt"], "ref": m["ref"], "prob": list(p),
                              "wrong": r["wrong"] if r["wrong"] is not None else [False] * len(p)})
    return _metrics_of(d)


class Ctx:
    def __init__(self, lang):
        self.lang = lang
        self.items = dev_items(lang)
        self.pool = load_ev(lang, "cvpool")
        self.dev = load_ev(lang, "dev")
        self.cor = load_ev(lang, "devcor")
        self.val = load_ev(lang, "val")
        self.dev_fit = [r for r in self.dev if r["meta"]["fold"] == "fit"]
        self.dev_chk = [r for r in self.dev if r["meta"]["fold"] == "check"]
        self.nat = self.pool + [r for r in self.dev_fit if not r["meta"].get("label")]
        self.cv_nat = [r for r in self.dev if not r["meta"].get("label") and r["meta"]["src"] == "cv"]
        self.cor_fit = [r for r in self.cor if r["meta"]["fold"] == "fit"]
        self.cor_nat = [r for r in self.dev_fit if not r["meta"].get("label") and r["meta"]["src"] == "cv"]
        by = defaultdict(list)
        for r in self.nat:
            for x, c in zip(r["s"], r["cls"]):
                if c is not None and np.isfinite(x):
                    by[c].append(x)
        self.offsets = {"none": {}, "q05": {c: float(np.quantile(v, 0.05)) for c, v in by.items() if len(v) >= 50}}

    def measure(self, probs_of):
        P = lambda rows: np.concatenate([probs_of(r) for r in rows]) if rows else np.zeros(0)  # noqa: E731
        m = {}
        pn = P(self.nat)
        m["pool_below_05"], m["pool_below_06"] = float(np.mean(pn < 0.5)), float(np.mean(pn < 0.6))
        pc = P(self.cv_nat)
        m["cvdev_below_06"] = float(np.mean(pc < 0.6))
        err = np.concatenate([probs_of(r)[np.array(r["wrong"])] for r in self.cor_fit])
        m["controlled_auc"] = auc(1 - err, 1 - P(self.cor_nat))
        fm = dev_metrics(self.items, self.dev_fit, probs_of)
        m["fit_auc"], m["fit_fa80"], m["fit_buckets"] = fm["auc"], fm["fa80"], fm["bucket_auc"]
        vm = val_metrics(self.val, probs_of)
        m["val"] = vm
        return m


def guards_ok(m, b, lang):
    why = []
    if m["pool_below_05"] - b["pool_below_05"] > 0.003 or m["pool_below_06"] - b["pool_below_06"] > 0.003:
        why.append("G1 pool")
    if m["val"]["native_below_0.6"] - b["val"]["native_below_0.6"] > 0.003 \
            or m["val"]["swap_native_below_0.6"] - b["val"]["swap_native_below_0.6"] > 0.003 \
            or m["cvdev_below_06"] - b["cvdev_below_06"] > 0.003:
        why.append("G2 natives")
    if m["controlled_auc"] < b["controlled_auc"] - 1e-4:
        why.append("G3 controlled")
    for k in ("sub_gross", "word_omission"):
        if k in b["fit_buckets"] and m["fit_buckets"][k] < b["fit_buckets"][k] - 1e-4:
            why.append(f"G4 {k}")
    if m["val"]["swap_far_auc"] < b["val"]["swap_far_auc"] - 1e-4:
        why.append("G4 val far")
    if lang in STRESS_LANGS and m["val"].get("n_stress") and m["val"]["stress_auc"] < b["val"]["stress_auc"] - 0.005:
        why.append("G5 stress")
    return why


# ----------------------------------------------------------------------------- forms

def knee_grid(ctx):
    for on, q, floor, width in itertools.product(("none", "q05"), (0.005, 0.01, 0.02, 0.05), (0.3, 0.6), (2.0, 4.0)):
        off = ctx.offsets[on]
        mins = np.array([v.min() for r in ctx.nat for v in word_vals(r, off) if len(v)])
        knee = float(np.quantile(mins, q))
        knees = [float(np.quantile(mins, q / 2)), knee, float(np.quantile(mins, min(1.0, 2 * q)))]
        yield {"form": "knee", "offset_kind": on, "q": q, "knee": knees, "floor": floor, "width": width}, off


def step_design(rows, off, label_of):
    """(features, offset logit, label, weight) of every judged word of rows."""
    F, z, y = [], [], []
    for r in rows:
        pp, _ = pre_prob(r)
        lab = label_of(r)
        for wi, v in enumerate(word_vals(r, off)):
            if not len(v):
                continue
            p = min(max(float(pp[wi]), 1e-6), 1 - 1e-6)
            F.append(ID.step_features(v))
            z.append(np.log(p / (1 - p)))
            y.append(lab[wi])
    return np.array(F), np.array(z), np.array(y, float)


def fit_offset_logistic(F, z, y, w, l2=10.0, iters=200):
    c = np.zeros(F.shape[1])
    for _ in range(iters):
        eta = z + F @ c
        p = 1 / (1 + np.exp(-eta))
        g = F.T @ (w * (p - y)) + l2 * np.r_[c[:-1], 0.0]
        Hm = (F * (w * p * (1 - p))[:, None]).T @ F + l2 * np.diag(np.r_[np.ones(F.shape[1] - 1), 0.0]) + 1e-6 * np.eye(F.shape[1])
        c -= np.linalg.solve(Hm, g)
    return c


def step_grid(ctx):
    off = ctx.offsets["q05"]

    def dev_lab(r):
        lab = (r["meta"].get("label") or {})
        n = len(r["prob"])
        out = [0] * n
        if lab:
            k = _ref_word_of_unit(r, lab.get("word_index"))
            if k is not None:
                out[k] = 1
        return out

    F1, z1, y1 = step_design(ctx.dev_fit, off, dev_lab)
    F2, z2, y2 = step_design(ctx.cor_fit, off, lambda r: [int(x) for x in r["wrong"]])
    F, z, y = np.vstack([F1, F2]), np.r_[z1, z2], np.r_[y1, y2]
    for wpos in (1.0, 2.0):
        w = np.where(y > 0, wpos, 1.0)
        c = fit_offset_logistic(F, z, y, w)
        for nrb in (-2.0, 0.0, -99.0):
            rule = {"form": "step", "coef": [float(x) for x in c], "cap": 3.0, "wpos": wpos,
                    "no_raise_below": (None if nrb == -99.0 else nrb)}
            if nrb == -99.0:
                rule["cap"] = 0.0          # never raise: a one-sided step
            yield rule, off


def _ref_word_of_unit(r, k):
    """Reference word index of the k-th whitespace unit of the text (the changed word of a dev item)."""
    import re

    if k is None:
        return None
    toks = [m.span() for m in re.finditer(r"\S+", r["text"])]
    if k >= len(toks):
        return None
    a, b = toks[k]
    hits = [i for i, (s, e) in enumerate(r["spans"]) if s < b and e > a]
    return hits[0] if hits else None


def cmd_select(lang, save):
    ctx = Ctx(lang)
    base_of = lambda r: r["prob"].astype(np.float64)  # noqa: E731
    b = ctx.measure(base_of)
    print(f"BASE fit AUC {b['fit_auc']:.4f} FA {100 * b['fit_fa80']:.2f} | pool<.5 {100 * b['pool_below_05']:.2f} "
          f"<.6 {100 * b['pool_below_06']:.2f} | val nat<.6 {100 * b['val']['native_below_0.6']:.2f} cv<.6 "
          f"{100 * b['cvdev_below_06']:.2f} | ctrl {b['controlled_auc']:.4f} | far {b['fit_buckets'].get('sub_gross', 0):.4f} "
          f"om {b['fit_buckets'].get('word_omission', 0):.4f} valfar {b['val']['swap_far_auc']:.4f} near "
          f"{b['val']['swap_near_auc']:.4f} stress {b['val'].get('stress_auc', float('nan')):.4f}", flush=True)
    res = []
    for rule, off in itertools.chain(knee_grid(ctx), step_grid(ctx)):
        f = lambda r, rule=rule, off=off: new_probs(r, rule, off)  # noqa: E731
        m = ctx.measure(f)
        why = guards_ok(m, b, lang)
        res.append({"rule": rule, "m": m, "fail": why})
        tag = " ".join(f"{k}={v}" for k, v in rule.items() if k not in ("knee", "coef", "form"))
        print(f"{rule['form']:4s} {tag:52s} fit AUC {m['fit_auc']:.4f} FA {100 * m['fit_fa80']:.2f} subtle "
              f"{m['fit_buckets'].get('sub_subtle', 0):.3f} | pool<.6 {100 * (m['pool_below_06'] - b['pool_below_06']):+.2f} "
              f"val<.6 {100 * (m['val']['native_below_0.6'] - b['val']['native_below_0.6']):+.2f} cv<.6 "
              f"{100 * (m['cvdev_below_06'] - b['cvdev_below_06']):+.2f} near {m['val']['swap_near_auc']:.4f} "
              f"{'OK' if not why else 'x ' + ','.join(why)}", flush=True)
    picks = {}
    for form in ("knee", "step"):
        ok = [x for x in res if x["rule"]["form"] == form and not x["fail"]]
        if ok:
            picks[form] = min(ok, key=lambda x: (round(x["m"]["fit_fa80"], 5), -x["m"]["fit_auc"]))
    out = {"lang": lang, "base": b, "grid": res, "picks": {k: v["rule"] for k, v in picks.items()}}
    win = min(picks.values(), key=lambda x: (round(x["m"]["fit_fa80"], 5), -x["m"]["fit_auc"])) if picks else None
    if win:
        rule = win["rule"]
        off = ctx.offsets[rule.get("offset_kind", "q05")]
        f = lambda r: new_probs(r, rule, off)  # noqa: E731
        c0 = dev_metrics(ctx.items, ctx.dev_chk)
        c1 = dev_metrics(ctx.items, ctx.dev_chk, f)
        ships = c1["auc"] > c0["auc"] and c1["fa80"] <= c0["fa80"] + 1e-9
        print(f"PICK {rule['form']} check AUC {c0['auc']:.4f} -> {c1['auc']:.4f} FA@80 {100 * c0['fa80']:.2f} -> "
              f"{100 * c1['fa80']:.2f}  ships={ships}", flush=True)
        for form, x in picks.items():
            r2 = x["rule"]
            o2 = ctx.offsets[r2.get("offset_kind", "q05")]
            cc = dev_metrics(ctx.items, ctx.dev_chk, lambda r: new_probs(r, r2, o2))
            print(f"   {form}: fit FA {100 * x['m']['fit_fa80']:.2f}  check AUC {cc['auc']:.4f} FA {100 * cc['fa80']:.2f}")
        cal = {**SPEC[lang], **{k: v for k, v in rule.items() if k not in ("offset_kind", "q", "wpos")},
               "offset": off, "lang": lang,
               "fit_on": "round-19 rollout dev fit fold + Common Voice pool speakers (bench/round19_rollout/fit.py)"}
        out.update(win=win["rule"], check={"before": c0, "after": c1}, ships=ships, cal=cal)
        if save and ships:
            p = Path("pronunciada/data/calib/identity") / f"{lang}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(cal, indent=1, ensure_ascii=False) + "\n")
    (work(lang) / "fit.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    if save:
        Path(f"results/tables/round19_rollout_{lang}_fit.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["evidence", "select"])
    ap.add_argument("--lang", required=True)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "evidence":
        cmd_evidence(a.lang)
    else:
        cmd_select(a.lang, a.save)


if __name__ == "__main__":
    main()
