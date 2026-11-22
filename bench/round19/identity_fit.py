"""Round 19b (German): place the sound-identity word factor (pronunciada/de_identity.py).

    python -m bench.round19.identity_fit [--save]

Native pool (for every quantile): the Common Voice TRAIN speakers held out of the head's training
(bench.round19.identity_train.pool_speaker) + the dev FIT-fold control items (CV dev + FLEURS val).
Grid: class offset (none / the class's 5 % native quantile) x knee quantile of native word minima
(0.5 / 1 / 2 / 5 %) x floor (0.3 / 0.6) x width (2 / 4). Knee at strictness 0 / 1 = half / double
the quantile.
Guards, fixed in advance, all on data outside every benchmark:
  * native pool: share of words with P < 0.5 rises by <= 0.3 pt, and P < 0.6 by <= 0.3 pt;
  * controlled sound errors (bench.corrupt kinds on CV dev, fit fold) word AUC not lower (1e-4);
  * dev fit fold: sub_gross (far swaps) and word_omission (wholly wrong / extra words) bucket AUC not
    lower (1e-4).
Pick: the guarded setting with the lowest dev FIT-fold FA@80 (ties: higher AUC); then the CHECK
fold is reported for it.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from bench.round19.devset import WORK
from bench.round19.identity_collect import load
from bench.round19.identity_train import pool_speaker
from pronunciada import de_identity as DI


class _P:
    def __init__(self, ipa, word, nuc, syl):
        self.ipa, self.word, self.is_nucleus, self.syllable = ipa, word, nuc, syl


class _W:
    def __init__(self, a, b):
        self.phones = (a, b)


class _S:
    stress = 0


class _Ref:
    """Minimal reference built from stored symbols (corrupted references are not stored): every
    nucleus counts as unstressed, so a word-final e is never judged."""
    def __init__(self, ipa, word):
        from pronunciada.text._eu_common import is_nucleus_ipa

        self.phones = [_P(x, int(w), is_nucleus_ipa(x), 0) for x, w in zip(ipa, word)]
        nw = int(max(word)) + 1 if len(word) else 0
        self.words = [_W(int(np.flatnonzero(word == k)[0]), int(np.flatnonzero(word == k)[-1]) + 1) if (word == k).any()
                      else _W(0, 0) for k in range(nw)]
        self.syllables = [_S()]


def raw_rows(rows, H):
    from pronunciada.text import get_g2p

    g = get_g2p("de")
    out = []
    for r in rows:
        if r.get("wrong") is not None:           # controlled errors: the corrupted reference's symbols
            ref = _Ref(list(r["ipa"]), np.asarray(r["word"]))
        else:
            ref = g(r["text"])
        if [p.ipa for p in ref.phones] != list(r["ipa"]):
            continue
        s = DI.raw_evidence(ref, r["V"].astype(np.float32), H)
        out.append({**{k: r[k] for k in ("id", "meta", "wrong", "word_prob", "word")}, "s": s, "cls": DI.classes(ref)})
    return out


def word_min(r, off):
    s = np.array([x - off.get(c, 0.0) if c is not None and np.isfinite(x) else np.nan for x, c in zip(r["s"], r["cls"])])
    nw = len(r["word_prob"])
    m = np.full(nw, np.nan)
    for i, w in enumerate(r["word"]):
        if np.isfinite(s[i]) and (not np.isfinite(m[w]) or s[i] < m[w]):
            m[w] = s[i]
    return m


def factor(m, knee, floor, width):
    t = np.clip((knee - m) / width, 0.0, 1.0)
    return np.where(np.isfinite(m), 1.0 - (1.0 - floor) * t, 1.0)


def main(argv=None):
    from bench.metrics import roc_auc
    from bench.round19.evaluate import _metrics, load_dump, units
    from bench.round19.collect import manifest
    from bench.round19.whatif import rebuild

    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--dump", default="dev-r19c")
    a = ap.parse_args(argv)
    H = DI._head()
    pool = raw_rows([r for r in load("cv-train") if pool_speaker(r)], H)
    dev = raw_rows(load("dev"), H)
    cor = raw_rows(load("devcor"), H)
    nat = pool + [r for r in dev if not r["meta"].get("label") and r["meta"].get("fold") == "fit"]
    # class offsets: 5 % native quantile per class
    by = {}
    for r in nat:
        for x, c in zip(r["s"], r["cls"]):
            if c is not None and np.isfinite(x):
                by.setdefault(c, []).append(x)
    offsets = {"none": {}, "q05": {c: float(np.quantile(v, 0.05)) for c, v in by.items() if len(v) >= 50}}
    D = load_dump(a.dump)
    items = manifest("dev")
    fit_items = [it for it in items if it["fold"] == "fit"]
    chk_items = [it for it in items if it["fold"] == "check"]
    devmap = {r["id"]: r for r in dev}

    def dev_metrics(fac_of, its):
        def fn(w, c):
            f = fac_of.get(c["rec"]["id"])
            return w["prob"] * (f[c["wi"]] if f is not None and c["wi"] < len(f) else 1.0)
        d = {k: (rebuild(v, fn) if not v.get("error") and k in fac_of else v) for k, v in D.items()}
        return _metrics(units(its, d))

    nat_p = np.concatenate([r["word_prob"] for r in nat])
    cor_fit = [r for r in cor if r["meta"].get("fold") == "fit"]
    cor_err = lambda F: np.concatenate([r["word_prob"][np.array(r["wrong"])] * F[r["id"]][np.array(r["wrong"])] for r in cor_fit])  # noqa: E731
    cor_nat = [r for r in dev if not r["meta"].get("label") and r["meta"].get("src") == "cv" and r["meta"].get("fold") == "fit"]

    base_fit = dev_metrics({}, fit_items)
    ones = {r["id"]: np.ones(len(r["word_prob"])) for r in cor_fit + cor_nat}
    e0 = cor_err(ones)
    n0 = np.concatenate([r["word_prob"] for r in cor_nat])
    base_cor = roc_auc(np.r_[np.ones(len(e0)), np.zeros(len(n0))], -np.r_[e0, n0])
    print(f"base fit: AUC {base_fit['auc']:.4f} FA@80 {100 * base_fit['fa80']:.2f} | far {base_fit['bucket_auc'].get('sub_gross'):.4f} "
          f"omission {base_fit['bucket_auc'].get('word_omission'):.4f} | controlled AUC {base_cor:.4f} | "
          f"pool natives <0.5 {100 * np.mean(nat_p < 0.5):.2f} % <0.6 {100 * np.mean(nat_p < 0.6):.2f} %", flush=True)
    res = []
    for on, q, floor, width in itertools.product(("none", "q05"), (0.005, 0.01, 0.02, 0.05), (0.3, 0.6), (2.0, 4.0)):
        off = offsets[on]
        mins = {r["id"]: word_min(r, off) for r in nat + dev + cor}
        allmin = np.concatenate([mins[r["id"]] for r in nat])
        allmin = allmin[np.isfinite(allmin)]
        knee = float(np.quantile(allmin, q))
        F = {k: factor(m, knee, floor, width) for k, m in mins.items()}
        pn = np.concatenate([r["word_prob"] * F[r["id"]] for r in nat])
        d05, d06 = np.mean(pn < 0.5) - np.mean(nat_p < 0.5), np.mean(pn < 0.6) - np.mean(nat_p < 0.6)
        e1 = cor_err(F)
        n1 = np.concatenate([r["word_prob"] * F[r["id"]] for r in cor_nat])
        cauc = roc_auc(np.r_[np.ones(len(e1)), np.zeros(len(n1))], -np.r_[e1, n1])
        m = dev_metrics({k: F[k] for k in devmap}, fit_items)
        ok = (d05 <= 0.003 and d06 <= 0.003 and cauc >= base_cor - 1e-4
              and m["bucket_auc"]["sub_gross"] >= base_fit["bucket_auc"]["sub_gross"] - 1e-4
              and m["bucket_auc"]["word_omission"] >= base_fit["bucket_auc"]["word_omission"] - 1e-4)
        row = {"offset": on, "q": q, "floor": floor, "width": width, "knee": knee, "d_native_05": d05, "d_native_06": d06,
               "controlled_auc": cauc, "fit_auc": m["auc"], "fit_fa80": m["fa80"], "fit_buckets": m["bucket_auc"], "ok": ok}
        res.append(row)
        print(f"{on:4s} q {q:.3f} floor {floor} width {width}: natives +{100 * d05:.2f}/+{100 * d06:.2f} pt  ctrl {cauc:.4f}  "
              f"fit AUC {m['auc']:.4f} FA {100 * m['fa80']:.2f} subtle {m['bucket_auc']['sub_subtle']:.3f} far "
              f"{m['bucket_auc']['sub_gross']:.4f} om {m['bucket_auc']['word_omission']:.4f} {'OK' if ok else ''}", flush=True)
    okr = [r for r in res if r["ok"]]
    pick = min(okr, key=lambda r: (round(r["fit_fa80"], 5), -r["fit_auc"])) if okr else None
    out = {"base_fit": base_fit, "base_controlled": base_cor, "grid": res, "pick": pick}
    if pick:
        off = offsets[pick["offset"]]
        mins = {r["id"]: word_min(r, off) for r in dev}
        F = {k: factor(m, pick["knee"], pick["floor"], pick["width"]) for k, m in mins.items()}
        mc0, mc1 = dev_metrics({}, chk_items), dev_metrics(F, chk_items)
        print(f"PICK {pick['offset']} q {pick['q']} floor {pick['floor']} width {pick['width']} knee {pick['knee']:.3f}: "
              f"check AUC {mc0['auc']:.4f} -> {mc1['auc']:.4f}  FA@80 {100 * mc0['fa80']:.2f} -> {100 * mc1['fa80']:.2f}  "
              f"subtle {mc0['bucket_auc']['sub_subtle']:.3f} -> {mc1['bucket_auc']['sub_subtle']:.3f}")
        out["check"] = {"before": mc0, "after": mc1}
        allmin = np.concatenate([word_min(r, off) for r in nat])
        allmin = allmin[np.isfinite(allmin)]
        knees = [float(np.quantile(allmin, pick["q"] / 2)), pick["knee"], float(np.quantile(allmin, min(1.0, 2 * pick["q"])))]
        cal = {"form": "knee", "offset": off, "knee": knees, "floor": pick["floor"], "width": pick["width"],
               "native_quantile": pick["q"],
               "fit_on": "round-19 German dev fit fold + held-out Common Voice train native pool (bench/round19/identity_fit.py)"}
        out["cal"] = cal
        if a.save:
            Path("pronunciada/data/calib/de_identity_word.json").write_text(json.dumps(cal, indent=1, ensure_ascii=False) + "\n")
    (WORK / "identity_fit.json").write_text(json.dumps(out, indent=1, default=float))
    if a.save:
        Path("results/tables/round19_de_identity_fit.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
