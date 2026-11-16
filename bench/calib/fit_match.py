"""Round 9: the prompt-match channel P_match (pronunciada/match.py), pooled over ten languages.

    python -m bench.calib.fit_match collect [--n 150] [--workers 3]   # FLEURS validation rows
    python -m bench.calib.fit_match dev [--workers 3]                 # cv-real DEV halves (selection)
    python -m bench.calib.fit_match fit [--neg-weight W] [--write]    # fit, LOLO CV, dev table

FLEURS VALIDATION only for fitting (never test). Per clip, the public Scorer (match channel off)
on:
  own    the clip with its own sentence                                    match (label 1)
  slip   its own sentence with ONE word replaced by a word of another clip  match (label 1)
  heavy  its own sentence with 30-60 % of the words replaced / deleted / with words inserted
  other  a different validation sentence of similar length
  noise  low-level white noise of the clip's duration, own sentence
  trunc  the first third of the clip, own sentence
(heavy / other / noise / trunc = no match, label 0). "slip" is a positive on purpose: one changed
word is the word level's job, P_match must stay ~1 there. Inputs: pronunciada.match.INPUTS.
Logistic regression, the negatives down-weighted to NEG_WEIGHT of the positives' total weight,
monotone (every input may only lower P_match; inputs with the wrong sign are dropped and the
model refitted). Evaluation: leave-one-language-out. Selection of NEG_WEIGHT: cv-real DEV halves
(bench.eval_cv --half dev manifests), ours only.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import defaultdict

import numpy as np

LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")
KINDS = ("own", "slip", "heavy", "other", "noise", "trunc")
POSITIVE = ("own", "slip")
NEG_WEIGHT = 0.25


def work_dir():
    from bench.calib.common import calib_cache

    d = calib_cache() / "round9_match"
    d.mkdir(exist_ok=True)
    return d


def _row(res, **kw) -> dict:
    c = res._pronunciada  # noqa: SLF001
    return {**kw, "base": float(res.prob), "e": dict(c["match_e"]),
            "present": [None if w.present_prob is None else float(w.present_prob) for w in res.words],
            "warnings": list(res.warnings)}


def _edit(text: str, spans, rng, frac: float, pool: list[str], sep: str) -> str:
    """Replace / delete / insert-before ~frac of the words (right to left keeps offsets valid)."""
    k = max(1, int(round(frac * len(spans))))
    pick = sorted(rng.choice(len(spans), min(k, len(spans)), replace=False), reverse=True)
    for i in pick:
        a, b = spans[i]
        op = rng.integers(3)
        w = pool[int(rng.integers(len(pool)))]
        if op == 0:
            text = text[:a] + w + text[b:]
        elif op == 1:
            text = text[:a] + text[b:]
        else:
            text = text[:a] + w + sep + text[a:]
    return text


def collect(lang: str, n: int) -> list[dict]:
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    from bench.pipeline.common import iter_parquet_audio, records
    from pronunciada.scorer import Scorer

    cache = work_dir() / f"rows_{lang}_{n}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    sc = Scorer(lang, threads=1, prosody=False, explain_away=False, match=False)
    allrecs = records("fleurs-validation", lang=lang)
    rng = np.random.default_rng(0)
    recs = [allrecs[i] for i in sorted(rng.choice(len(allrecs), min(n, len(allrecs)), replace=False))]
    by = {int(r["audio"].split(":")[2]): r for r in recs}
    path = recs[0]["audio"].split(":")[1]
    sep = "" if lang in ("ja", "zh") else " "
    texts = [r["text"] for r in allrecs]
    rows, clips = [], []
    for i, wav in iter_parquet_audio(path, set(by)):
        rec = by[i]
        try:
            res = sc.score((wav, 16000), rec["text"])
        except ValueError:
            continue
        rows.append(_row(res, lang=lang, utt=rec["id"], kind="own", label=1))
        clips.append((rec, wav, res))
    pool = [w.text for _, _, res in clips for w in res.words]
    for rec, wav, res in clips:
        ref = res._pronunciada["ref"]  # noqa: SLF001
        spans = [tuple(w.char_span) for w in ref.words]
        rid = rec["id"]
        L = len(rec["text"])
        cands = [t for t in texts if t != rec["text"]]
        other = min(cands, key=lambda t: (abs(len(t) - L), t))
        frac = float(rng.uniform(0.3, 0.6))
        variants = [("slip", _edit(rec["text"], spans, rng, 1.0 / max(len(spans), 1) * 0.999, pool, sep), wav),
                    ("heavy", _edit(rec["text"], spans, rng, frac, pool, sep), wav),
                    ("other", other, wav),
                    ("noise", rec["text"], (rng.standard_normal(len(wav)) * 10 ** (-50 / 20)).astype(np.float32)),
                    ("trunc", rec["text"], wav[: len(wav) // 3])]
        for kind, text, audio in variants:
            if kind == "slip" and len(spans) < 2:
                continue
            try:
                r2 = sc.score((audio, 16000), text)
            except ValueError:
                continue
            rows.append(_row(r2, lang=lang, utt=rid, kind=kind, label=int(kind in POSITIVE),
                             **({"frac": frac} if kind == "heavy" else {})))
    cache.write_bytes(pickle.dumps(rows))
    return rows


def _collect_job(args):
    return collect(*args)


def load_rows(n: int) -> list[dict]:
    return [r for lg in LANGS for r in pickle.loads((work_dir() / f"rows_{lg}_{n}.pkl").read_bytes())]


# ----------------------------------------------------------------------------- cv-real dev

_SC = {}


def _dev_init():
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def _dev_one(rec):
    from pronunciada.scorer import Scorer

    if rec["lang"] not in _SC:
        _SC.clear()
        _SC[rec["lang"]] = Scorer(rec["lang"], threads=1, match=False)
    try:
        res = _SC[rec["lang"]].score(rec["audio"], rec["sentence"])
    except ValueError:
        return {"id": rec["id"], "status": "error"}
    return _row(res, id=rec["id"], status="Success")


def dev_rows(workers: int = 3) -> list[dict]:
    """Ours (match channel off) on every cv-real DEV clip, all locales; cached per fingerprint
    of the code that produces the rows' inputs (everything but the match coefficients)."""
    from concurrent.futures import ProcessPoolExecutor

    from bench import cv_build as CB

    recs = CB.load_manifest(CB.LOCALES, half="dev")
    assert recs and all(r["half"] == "dev" for r in recs)
    p = work_dir() / "dev_rows.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    clips = list({r["id"]: r for r in recs}.values())
    clips.sort(key=lambda r: r["lang"])
    t0 = time.time()
    with ProcessPoolExecutor(min(workers, 3), initializer=_dev_init) as ex:
        out = list(ex.map(_dev_one, clips, chunksize=8))
    byid = {o["id"]: o for o in out}
    rows = [dict(byid[r["id"]], locale=r["locale"], label=r["label"], client_id=r["client_id"],
                 test=r["test"]) for r in recs]
    p.write_bytes(pickle.dumps(rows))
    print(f"dev rows: {len(rows)} ({time.time() - t0:.0f} s)")
    return rows


def _so_one(rec):
    from pronunciada.scorer import Scorer

    if "en" not in _SC:
        _SC["en"] = Scorer("en", threads=1, match=False)
    try:
        res = _SC["en"].score(rec["audio"], rec["text"])
    except ValueError:
        return {"id": rec["id"], "status": "error"}
    return _row(res, id=rec["id"], status="Success", y=float(rec["labels"]["sentence"]["accuracy"]),
                speaker=rec["speaker"])


def so762_rows(workers: int = 3) -> list[dict]:
    """Guard data: ours (match off) on speechocean762 TRAIN (never test)."""
    from concurrent.futures import ProcessPoolExecutor

    from bench.data import load

    p = work_dir() / "so762_train_rows.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    recs = load("speechocean762", split="train")
    with ProcessPoolExecutor(min(workers, 3), initializer=_dev_init) as ex:
        rows = list(ex.map(_so_one, recs, chunksize=8))
    p.write_bytes(pickle.dumps(rows))
    return rows


def so762_guard(rows, coef) -> dict:
    """Sentence probability vs 'accuracy >= 7' (bench.eval_english midpoint) before / after."""
    from bench.calib.common import ece
    from bench.metrics import roc_auc

    rows = [r for r in rows if "base" in r]
    y = np.array([r["y"] >= 7.0 for r in rows]).astype(int)
    a = np.array([r["base"] for r in rows])
    b = a * predict(coef, X_of(rows))
    return {"n": len(rows), "pos_rate": float(y.mean()),
            "before": {"ece": ece(y, a), "auc": float(roc_auc(y, a)), "mean_p": float(a.mean())},
            "after": {"ece": ece(y, b), "auc": float(roc_auc(y, b)), "mean_p": float(b.mean())}}


# ----------------------------------------------------------------------------- fit

def X_of(rows) -> np.ndarray:
    from pronunciada import match as M

    return np.array([M.inputs(r["e"], r["present"]) for r in rows])


def fit(rows, neg_weight: float = NEG_WEIGHT, X=None) -> list[float]:
    from sklearn.linear_model import LogisticRegression

    X = X_of(rows) if X is None else X
    y = np.array([r["label"] for r in rows])
    w = np.ones(len(y))
    w[y == 0] = neg_weight * (y == 1).sum() / max((y == 0).sum(), 1)
    keep = list(range(X.shape[1]))
    while True:
        m = LogisticRegression(C=1.0, max_iter=5000).fit(X[:, keep], y, sample_weight=w)
        bad = [k for k, c in zip(keep, m.coef_[0]) if c > 0]
        if not bad:
            break
        keep = [k for k in keep if k not in bad]
    coef = np.zeros(X.shape[1] + 1)
    coef[0] = m.intercept_[0]
    for k, c in zip(keep, m.coef_[0]):
        coef[1 + k] = c
    return coef.tolist()


def predict(coef, X) -> np.ndarray:
    b = np.asarray(coef)
    return 1.0 / (1.0 + np.exp(-(b[0] + X @ b[1:])))


def fleurs_report(rows, p) -> dict:
    from bench.metrics import roc_auc

    y = np.array([r["label"] for r in rows])
    kind = np.array([r["kind"] for r in rows])
    base = np.array([r["base"] for r in rows])
    out = {"auc_match_vs_not": float(roc_auc(1 - y, 1 - p)) if 0 < y.sum() < len(y) else None}
    for k in KINDS:
        m = kind == k
        if m.any():
            out[k] = {"n": int(m.sum()), "mean_p_match": float(p[m].mean()), "share_p_lt_0.5": float(np.mean(p[m] < 0.5)),
                      "accept_before": float(np.mean(base[m] >= 0.5)), "accept_after": float(np.mean(base[m] * p[m] >= 0.5))}
    return out


def lolo(rows, neg_weight) -> dict:
    X = X_of(rows)
    lang = np.array([r["lang"] for r in rows])
    out = {}
    for lg in LANGS:
        te = lang == lg
        c = fit([r for r, t in zip(rows, te) if not t], neg_weight, X[~te])
        out[lg] = fleurs_report([r for r, t in zip(rows, te) if t], predict(c, X[te]))
    return out


def dev_table(rows, coef) -> dict:
    from bench.eval_cv import speaker_bootstrap

    ok = [r for r in rows if r.get("status") == "Success" or "base" in r]
    for r in rows:
        if "base" not in r:          # failure = lowest score, as in eval_cv
            r.update(base=0.0, e=None)
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["locale"], r["test"])].append(r)
    for (loc, test), rs in sorted(groups.items()):
        y = np.array([r["label"] for r in rs])
        if not 0 < y.sum() < len(y):
            continue
        a = np.array([r["base"] for r in rs])
        X = np.array([np.zeros(len(coef) - 1) if r["e"] is None else X_of([r])[0] for r in rs])
        pm = predict(coef, X)
        b = a * pm
        bs = speaker_bootstrap(y, b, a, [r["client_id"] for r in rs], n_boot=1000)
        out[f"{loc}/{test}"] = {"n": len(rs), "before": bs["azure"], "after": bs["ours"], "delta": bs["delta"],
                                "delta_ci": bs["delta_ci"], "p": bs["p_two_sided"],
                                "p_match_lt_0.5": {"rejected": float(np.mean(pm[y == 1] < 0.5)),
                                                   "accepted": float(np.mean(pm[y == 0] < 0.5))}}
    del ok
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("collect", "dev", "so762", "fit"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--neg-weight", type=float, default=NEG_WEIGHT)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.step == "collect":
        with ProcessPoolExecutor(min(a.workers, 3)) as ex:
            for lg, rs in zip(LANGS, ex.map(_collect_job, [(lg, a.n) for lg in LANGS])):
                print(lg, len(rs), flush=True)
        return
    if a.step == "dev":
        dev_rows(a.workers)
        return
    if a.step == "so762":
        so762_rows(a.workers)
        return
    rows = load_rows(a.n)
    coef = fit(rows, a.neg_weight)
    from pronunciada import match as M

    lo = lolo(rows, a.neg_weight)
    dev = dev_table(dev_rows(a.workers), coef) if (work_dir() / "dev_rows.pkl").exists() else {}
    out = {"coef": coef, "inputs": list(M.INPUTS), "clips": {k: list(v) for k, v in M.CLIPS.items()},
           "against_llr": M.AGAINST_LLR, "neg_weight": a.neg_weight,
           "fit_on": f"FLEURS validation, {a.n} clips x {len(LANGS)} languages, pooled; kinds {KINDS}, "
                     f"positives {POSITIVE}",
           "n_rows": len(rows), "in_sample": fleurs_report(rows, predict(coef, X_of(rows))),
           "leave_one_language_out": lo, "cv_real_dev": dev,
           "so762_train": so762_guard(so762_rows(), coef) if (work_dir() / "so762_train_rows.pkl").exists() else None}
    if out["so762_train"]:
        print("so762 train", json.dumps({k: out["so762_train"][k] for k in ("before", "after")}))
    print("coef", dict(zip(["b0", *M.INPUTS], np.round(coef, 3).tolist())))
    for lg, r in lo.items():
        print(f"LOLO {lg}: auc {r['auc_match_vs_not']:.3f} " + " ".join(
            f"{k} p={v['mean_p_match']:.3f}/<.5:{v['share_p_lt_0.5']:.2f}" for k, v in r.items() if k in KINDS)
            + f"  native accept {r['own']['accept_before']:.3f}->{r['own']['accept_after']:.3f}")
    if dev:
        d = [v["delta"] for v in dev.values()]
        for k, v in dev.items():
            print(f"dev {k}: {v['before']:.3f} -> {v['after']:.3f} ({v['delta']:+.3f}, p={v['p']:.2f})")
        print(f"dev mean delta {np.mean(d):+.4f}, min {np.min(d):+.4f}")
    tag = f"w{a.neg_weight}"
    (work_dir() / f"fit_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    if a.write:
        from bench.calib.common import save_json, ship_dir

        ship = {k: out[k] for k in ("coef", "inputs", "clips", "against_llr", "neg_weight", "fit_on", "n_rows")}
        (ship_dir() / "match.json").write_text(json.dumps(ship, indent=1, default=float))
        save_json("match.json", out)
        print("wrote pronunciada/data/calib/match.json")


if __name__ == "__main__":
    main()
