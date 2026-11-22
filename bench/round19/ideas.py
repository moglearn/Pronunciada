"""Round 19 (German): the project owner's extra hypotheses, measured offline on a dev dump.

    python -m bench.round19.ideas dev-r19a

(2) per-sound native floor: for every sound symbol, the 2nd percentile of its shown lenient / strict
    log-odds on FIT-fold control words; each sound's log-odds minus its floor (plus the mean floor,
    so the overall level is unchanged) replaces it in the word inputs (min / second min / mean /
    share below 0), the word model is re-run and the German recalibration refitted on the fit fold;
    compared on the CHECK fold with the same refit on unmodified inputs.
(3) function words: the shown word log-odds of words in the front-end's function-word list
    (pronunciada.text.de.FUNC) raised by delta; per delta: dev metric (fit / check) and the share of
    changed function words (target or original a function word) still caught at the FA@80 threshold
    of the unmodified system.
"""
from __future__ import annotations

import argparse
import copy
import math
import re
from collections import defaultdict

import numpy as np

from bench.round19 import fit_rerank as R
from bench.round19.collect import manifest
from bench.round19.evaluate import _metrics, load_dump, units
from bench.round19.whatif import rebuild
from bench.round19.wordmodel import sounds as wm_sounds


def floors(D, M, q=0.02):
    vals = defaultdict(lambda: ([], []))
    for k, r in D.items():
        m = M[k]
        if r.get("error") or m["label"] or m["fold"] != "fit":
            continue
        for w in r["rwords"]:
            for p in w["phones"]:
                if p["zl"] is not None:
                    vals[p["ipa"]][0].append(p["zl"])
                    vals[p["ipa"]][1].append(p["zs"])
    fl = {s: (float(np.quantile(a, q)), float(np.quantile(b, q))) for s, (a, b) in vals.items() if len(a) >= 30}
    ml = float(np.mean([v[0] for v in fl.values()]))
    ms = float(np.mean([v[1] for v in fl.values()]))
    return fl, ml, ms


def adjust(D, fl, ml, ms):
    out = {}
    for k, r in D.items():
        if r.get("error"):
            out[k] = r
            continue
        r = copy.deepcopy(r)
        for w in r["rwords"]:
            zl = np.array([(p["zl"] if p["zl"] is not None else 0.0) - fl.get(p["ipa"], (ml, ms))[0] + ml for p in w["phones"]])
            zs = np.array([(p["zs"] if p["zs"] is not None else 0.0) - fl.get(p["ipa"], (ml, ms))[1] + ms for p in w["phones"]])
            W = w["W"]
            sl, ss = np.sort(zl), np.sort(zs)
            W["wl_min"], W["ws_min"] = float(sl[0]), float(ss[0])
            W["wl_min2"] = float(sl[1] if len(sl) > 1 else sl[0])
            W["ws_min2"] = float(ss[1] if len(ss) > 1 else ss[0])
            W["wl_mean"], W["ws_mean"] = float(zl.mean()), float(zs.mean())
            W["wl_share_below"], W["ws_share_below"] = float((zl < 0).mean()), float((zs < 0).mean())
        allw = [w for w in r["rwords"]]
        if allw:
            s = wm_sounds([w["W"] for w in allw])
            for w, v in zip(allw, s):
                w["prob"] = w["prob"] / max(w["sounds"], 1e-6) * float(v)
                w["sounds"] = float(v)
        out[k] = r
    return out


def evaluate_dump(D, items, par):
    fn = R.apply_fn(par)
    d = {k: (rebuild(v, fn) if not v.get("error") else v) for k, v in D.items()}
    return _metrics(units(items, d))


def idea2(name):
    M = {m["id"]: m for m in manifest("dev")}
    D0 = load_dump(name)
    # the dump's word probabilities already include the shipped recalibration; undo it to get the
    # word-model output, as fit_rerank expects
    fl, ml, ms = floors(D0, M)
    spread = sorted(fl.items(), key=lambda x: x[1][0])
    print(f"(2) floors over {len(fl)} sounds: lowest {[(s, round(v[0], 2)) for s, v in spread[:6]]}, "
          f"highest {[(s, round(v[0], 2)) for s, v in spread[-4:]]}, mean {ml:.2f}")
    D1 = adjust(D0, fl, ml, ms)
    check = [it for it in manifest("dev") if it["fold"] == "check"]
    for tag, D in (("as is", D0), ("floor", D1)):
        R.load_dump = lambda n, D=D: D       # fit_rerank.table reads through load_dump
        par = R.fit(name, 0.1)
        m = evaluate_dump(D, check, par)
        print(f"    {tag:6s} refit -> check AUC {m['auc']:.4f} FA@80 {100 * m['fa80']:.2f}")


def idea3(name):
    from pronunciada.text.de import FUNC

    items = manifest("dev")
    D = load_dump(name)
    base = {f: _metrics(units([it for it in items if it["fold"] == f], D)) for f in ("fit", "check")}

    def is_func(t):
        return re.sub(r"[^\w]", "", t).lower() in FUNC

    # threshold of the unmodified system (all dev)
    us = units(items, D)
    pos = sorted((s for u in us for s, l in zip(u[1], u[0]) if l), reverse=True)
    thr = pos[math.ceil(0.8 * len(pos)) - 1]
    fw = [it for it in items if it.get("label") and (is_func(it["label"].get("original") or "")
                                                   or is_func(it["label"].get("replacement") or it["label"].get("added") or ""))]
    print(f"(3) changed items involving a function word: {len(fw)}; base fit FA {100 * base['fit']['fa80']:.2f} "
          f"check FA {100 * base['check']['fa80']:.2f}")
    for delta in (0.0, 0.5, 1.0, 2.0, 3.0):
        def fn(w, c, delta=delta):
            p = w["prob"]
            if is_func(w["text"]) and delta:
                z = math.log(max(p, 1e-6) / max(1 - p, 1e-6)) + delta
                return 1 / (1 + math.exp(-z))
            return p
        d = {k: (rebuild(v, fn) if not v.get("error") else v) for k, v in D.items()}
        mf = _metrics(units([it for it in items if it["fold"] == "fit"], d))
        mc = _metrics(units([it for it in items if it["fold"] == "check"], d))
        uf = units(fw, d)
        caught = sum(1 for u in uf for s, l in zip(u[1], u[0]) if l and s >= thr)
        n = sum(sum(u[0]) for u in uf)
        print(f"    delta {delta:.1f}: fit AUC {mf['auc']:.4f} FA {100 * mf['fa80']:.2f}  check AUC {mc['auc']:.4f} "
              f"FA {100 * mc['fa80']:.2f}  function-word changes caught {caught}/{n}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    a = ap.parse_args(argv)
    idea3(a.name)
    idea2(a.name)


if __name__ == "__main__":
    main()
