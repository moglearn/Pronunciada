"""Round 11: paired held-out comparison, old stress default vs stress_v3, same clips.

    .venv/bin/python -m bench.prosody.r11_compare <old api dir> <new api dir> <old cv fingerprint> <new cv fingerprint>

* swap benchmark (fleurs-swap test): per language and variant (swap_near, swap_far) AUC /
  hit@5%FA / right word, paired cluster bootstrap (clusters = source utterance, as
  bench.eval_swap.evaluate), new - old, two-sided p.
* ten-language controlled errors (fleurs test, bench.eval_languages caches): word-error AUC,
  stress-shift (prosody) word AUC of WordResult.prob at s = 0.5, native word acceptance; paired
  utterance bootstrap.
* cv-real test: primary AUC (SentenceResult.prob), paired speaker bootstrap
  (bench.eval_cv.speaker_bootstrap).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

LANGS = ("en", "de", "es", "it", "pt", "ru")
LOCALES = {"en": "en", "de": "de", "es": "es", "it": "it", "pt": "pt", "ru": "ru"}
N_BOOT = 1000


def p2(d):
    d = np.asarray(d)
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def swap(old_dir, new_dir):
    from bench.data import load
    from bench.eval_swap import _metrics, collect

    recs = load("fleurs-swap", split="test")
    out = {}
    for lang in LANGS:
        rs = [r for r in recs if r["lang"] == lang]
        O = {k: v["on"] for k, v in pickle.loads((old_dir / f"fleurs-swap-{lang}.pkl").read_bytes()).items()}
        N = {k: v["on"] for k, v in pickle.loads((new_dir / f"fleurs-swap-{lang}.pkl").read_bytes()).items()}
        co, cn = collect(rs, O)[lang]["clusters"], collect(rs, N)[lang]["clusters"]
        keys = sorted(set(co) & set(cn))
        lo, ln = [co[k] for k in keys], [cn[k] for k in keys]
        var = ["swap_near", "swap_far"]
        po, pn = _metrics(lo, var), _metrics(ln, var)
        rng = np.random.default_rng(0)
        d = {m: [] for m in po if isinstance(po[m], float)}
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(keys), len(keys))
            mo, mn = _metrics([lo[i] for i in idx], var), _metrics([ln[i] for i in idx], var)
            for m in d:
                if np.isfinite(mo.get(m, np.nan)) and np.isfinite(mn.get(m, np.nan)):
                    d[m].append(mn[m] - mo[m])
        out[lang] = {m: {"old": po[m], "new": pn[m], "diff": pn[m] - po[m], "p": p2(d[m]),
                         "ci": [float(np.quantile(d[m], .025)), float(np.quantile(d[m], .975))]}
                     for m in d if m.split(":")[1] in ("auc", "hit@5%FA", "right_word") and d[m]}
        print("swap", lang, {m: (round(v["old"], 3), round(v["new"], 3), round(v["p"], 3)) for m, v in out[lang].items()})
    return out


def ten(old_dir, new_dir):
    from bench.metrics import roc_auc

    out = {}
    for lang in LANGS:
        O = pickle.loads((old_dir / f"fleurs-test-{lang}.pkl").read_bytes())
        N = pickle.loads((new_dir / f"fleurs-test-{lang}.pkl").read_bytes())
        uids = sorted(k for k in O if k in N and "clean" in O[k] and "clean" in N[k])

        def rows(D, kind):
            sc, y, g = [], [], []
            for gi, u in enumerate(uids):
                c = D[u]
                if kind == "native":
                    w = c["clean"]["V"]["on"]["wp"][0.5]
                    sc += list(w)
                    y += [0] * len(w)
                    g += [gi] * len(w)
                    continue
                for k, d in c.get("cor", {}).items():
                    if (kind == "prosody") != (k == "prosody"):
                        continue
                    w = d["V"]["on"]["wp"][0.5]
                    ww = np.asarray(d["word_wrong"], bool)
                    sc += list(w)
                    y += list(ww.astype(int))
                    g += [gi] * len(w)
            return np.array(sc), np.array(y), np.array(g)

        res = {}
        for kind in ("word", "prosody", "native"):
            so, yo, go = rows(O, kind)
            sn, yn, gn = rows(N, kind)
            if len(so) != len(sn) or not (yo == yn).all():
                res[kind] = {"note": "unpaired (different references)"}
                continue
            f = (lambda s, y: float(np.mean(s >= 0.5))) if kind == "native" else (lambda s, y: roc_auc(y, 1 - s))
            if kind != "native" and (yo.sum() == 0 or yo.sum() == len(yo)):
                continue
            rng = np.random.default_rng(0)
            byg = [np.flatnonzero(go == i) for i in range(len(uids))]
            d = []
            for _ in range(N_BOOT):
                I = np.concatenate([byg[i] for i in rng.integers(0, len(uids), len(uids))])
                d.append(f(sn[I], yn[I]) - f(so[I], yo[I]))
            res[kind] = {"old": f(so, yo), "new": f(sn, yn), "diff": f(sn, yn) - f(so, yo), "p": p2(d),
                         "ci": [float(np.quantile(d, .025)), float(np.quantile(d, .975))]}
        out[lang] = res
        print("ten", lang, {k: (round(v.get("old", np.nan), 4), round(v.get("new", np.nan), 4), round(v.get("p", np.nan), 3))
                            for k, v in res.items() if "old" in v})
    return out


def cv(fp_old, fp_new):
    from bench import eval_cv as EC

    recs = EC.records(list(LOCALES.values()), half="test")
    root = EC.ours_root()

    def load(fp):
        out = {}
        for r in EC.unique_clips(recs):
            p = root / fp / f"{r['id']}.json"
            if p.exists():
                out[r["id"]] = json.loads(p.read_text())
        return out

    O, N = load(fp_old), load(fp_new)
    out = {}
    for loc in LOCALES.values():
        for test in ("A", "B", "Aweak"):
            rs = [r for r in recs if r["locale"] == loc and r["test"] == test and r["id"] in O and r["id"] in N]
            if not rs or not 0 < sum(r["label"] for r in rs) < len(rs):
                continue
            y = np.array([r["label"] for r in rs])
            a = np.array([N[r["id"]].get(EC.OURS_PRIMARY, 0.0) if N[r["id"]]["status"] == "Success" else 0.0 for r in rs])
            b = np.array([O[r["id"]].get(EC.OURS_PRIMARY, 0.0) if O[r["id"]]["status"] == "Success" else 0.0 for r in rs])
            bs = EC.speaker_bootstrap(y, a, b, [r["client_id"] for r in rs])
            out[f"{loc}/{test}"] = bs
            print("cv", loc, test, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in bs.items()
                                    if not isinstance(v, (list, dict))})
    return out


if __name__ == "__main__":
    from bench.prosody.common import save_json

    od, nd = Path(sys.argv[1]), Path(sys.argv[2])
    res = {"swap": swap(od, nd), "ten": ten(od, nd), "cv": cv(sys.argv[3], sys.argv[4])}
    save_json(res, "r11_compare.json")
