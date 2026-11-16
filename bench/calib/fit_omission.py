"""Round 4: the omission channel — P(word was said at all), replacing the hard "omitted" flag.

    python -m bench.calib.fit_omission [--n 120] [--workers 5] [--lang all]

Why (error analysis of the Azure text-swap benchmark, done after the held-out evaluation and
disclosed as such): the round-1 rule "a word is omitted when every sound has llr_del < -2" fired
on 3-6 % of native one-sound words (pt "é", fr "à", ja "を" ...), the benchmark adapter scores an
omitted word 0, and that single false zero was the lowest word of the sentence in ~3/4 of the
cases where we blamed the wrong word. At the same time a truly skipped one-sound word kept a
word probability around 0.93.

FLEURS VALIDATION only (never test). For n random clips per language: the public Scorer on the
correct text (every word = said) and on the text with ONE extra word inserted at a random word
boundary (that word = not said); three insertions per clip, one per word-length class (1, 2,
>= 3 sounds), the inserted word drawn from the other validation sentences of the language.
Calibrator, pooled over the ten languages: logistic regression on
    [max llr_del over the word's sounds, mean llr_del, one-sound word, two-sound word]
with sample weights so that omissions are PRIOR of the weight within each word-length class. P_present multiplies the word
probability; WordResult.omitted = P_present < 0.5. Evaluation: 5-fold utterance-grouped CV and
leave-one-language-out.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")
PRIOR = 0.03          # share of reference words a learner skips (weighting only)
CLIP = 20.0
FEATURES = "[1, max llr_del, mean llr_del, one-sound word, two-sound word]"


def word_feats(llr_del) -> list[float]:
    """Features of one word from the llr_del of its sounds (shared with pronunciada.scorer)."""
    d = np.clip(np.nan_to_num(np.asarray(llr_del, dtype=np.float64), nan=0.0), -CLIP, CLIP)
    return [float(d.max()), float(d.mean()), float(len(d) == 1), float(len(d) == 2)]


def collect(lang: str, n: int) -> list[dict]:
    import os

    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    from bench.calib.common import calib_cache
    from bench.pipeline.common import iter_parquet_audio, records
    from pronunciada.scorer import Scorer

    cache = calib_cache() / f"omission_rows_{lang}_{n}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    sc = Scorer(lang, threads=1, prosody=False, explain_away=False)
    recs = records("fleurs-validation", lang=lang)
    rng = np.random.default_rng(0)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), min(n, len(recs)), replace=False))]
    by = {int(r["audio"].split(":")[2]): r for r in recs}
    path = recs[0]["audio"].split(":")[1]
    sep = "" if lang in ("ja", "zh") else " "
    clean, rows = [], []
    pools: dict[int, list[str]] = {1: [], 2: [], 3: []}

    def row(res, wi, utt, said):
        tab = res._pronunciada["tab"]  # noqa: SLF001
        a, b = res._pronunciada["ref"].words[wi].phones  # noqa: SLF001
        w = res.words[wi]
        return {"utt": utt, "lang": lang, "said": said, "f": word_feats(tab["llr_del"][a:b]),
                "nph": b - a, "old_flag": bool(res._pronunciada["W"]["omitted_f"][wi]),  # noqa: SLF001
                "sounds": float(w.sounds_prob), "text": w.text}

    for i, wav in iter_parquet_audio(path, set(by)):
        rec = by[i]
        try:
            res = sc.score((wav, 16000), rec["text"])
        except ValueError:
            continue
        clean.append((rec, wav, res))
        for wi, w in enumerate(res.words):
            rows.append(row(res, wi, rec["id"], True))
            pools[min(len(w.phones), 3)].append(w.text)
    for rec, wav, res in clean:
        ref = res._pronunciada["ref"]  # noqa: SLF001
        if len(ref.words) < 4:
            continue
        for cls in (1, 2, 3):
            if not pools[cls]:
                continue
            k = int(rng.integers(1, len(ref.words) - 1))
            c = ref.words[k].char_span[0]
            ins = None
            for _ in range(20):
                cand = pools[cls][int(rng.integers(len(pools[cls])))]
                if cand.lower() not in (ref.words[k].text.lower(), ref.words[k - 1].text.lower()):
                    ins = cand
                    break
            if ins is None:
                continue
            text = rec["text"][:c] + ins + sep + rec["text"][c:]
            try:
                r2 = sc.score((wav, 16000), text)
            except ValueError:
                continue
            hit = [j for j, w in enumerate(r2._pronunciada["ref"].words)  # noqa: SLF001
                   if tuple(w.char_span) == (c, c + len(ins))]
            if len(hit) != 1:
                continue            # the front-end merged / split the inserted text
            rows.append(row(r2, hit[0], rec["id"], False))
    cache.write_bytes(pickle.dumps(rows))
    return rows


def fit(rows) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    X = np.array([r["f"] for r in rows])
    y = np.array([int(r["said"]) for r in rows])
    # the prior holds WITHIN each word-length class (the insertions are balanced over the classes,
    # native words are not: without this a one-sound word would look 15x more likely to be skipped)
    cls = np.minimum(np.array([r["nph"] for r in rows]), 3)
    w = np.ones(len(rows))
    for c in (1, 2, 3):
        n_om, n_ok = ((y == 0) & (cls == c)).sum(), ((y == 1) & (cls == c)).sum()
        w[(y == 0) & (cls == c)] = PRIOR / (1 - PRIOR) * n_ok / max(n_om, 1)
    m = LogisticRegression(C=1.0, max_iter=5000).fit(X, y, sample_weight=w)
    coef = np.concatenate([m.intercept_, m.coef_[0]])
    if coef[1] < 0 or coef[2] < 0:      # must be monotone in the deletion evidence
        m = LogisticRegression(C=1.0, max_iter=5000).fit(X[:, [0, 2, 3]], y, sample_weight=w)
        coef = np.array([m.intercept_[0], m.coef_[0][0], 0.0, m.coef_[0][1], m.coef_[0][2]])
        if coef[1] < 0:
            raise RuntimeError("non-monotone omission calibrator")
    return coef


def prob(coef, rows) -> np.ndarray:
    X = np.array([r["f"] for r in rows])
    return 1.0 / (1.0 + np.exp(-(coef[0] + X @ np.asarray(coef[1:]))))


def _report(rows, p) -> dict:
    from bench.metrics import roc_auc

    said = np.array([r["said"] for r in rows])
    nph = np.minimum(np.array([r["nph"] for r in rows]), 3)
    old = np.array([r["old_flag"] for r in rows])
    out = {"auc": float(roc_auc((~said).astype(int), 1 - p))}
    for c in (1, 2, 3):
        m = nph == c
        out[f"sounds{'>=' if c == 3 else '='}{c}"] = {
            "n_said": int((m & said).sum()), "n_omitted": int((m & ~said).sum()),
            "false_omitted_new": float(np.mean(p[m & said] < 0.5)), "false_omitted_old": float(np.mean(old[m & said])),
            "caught_new": float(np.mean(p[m & ~said] < 0.5)) if (m & ~said).any() else None,
            "caught_old": float(np.mean(old[m & ~said])) if (m & ~said).any() else None}
    return out


def evaluate(rows) -> dict:
    from sklearn.model_selection import GroupKFold

    utt = np.array([r["utt"] for r in rows])
    lang = np.array([r["lang"] for r in rows])
    p = np.ones(len(rows))
    for tr, te in GroupKFold(5).split(rows, None, utt):
        p[te] = prob(fit([rows[i] for i in tr]), [rows[i] for i in te])
    out = {"cv": _report(rows, p), "leave_one_language_out": {}}
    for lg in sorted(set(lang)):
        te = np.flatnonzero(lang == lg)
        c = fit([rows[i] for i in np.flatnonzero(lang != lg)])
        sub = [rows[i] for i in te]
        out["leave_one_language_out"][lg] = _report(sub, prob(c, sub))
    return out


def _job(args):
    return collect(*args)


def main():
    from concurrent.futures import ProcessPoolExecutor

    from bench.calib.common import save_json, ship_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="all")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    langs = list(LANGS) if a.lang == "all" else a.lang.split(",")
    with ProcessPoolExecutor(min(a.workers, len(langs))) as ex:
        rows = [r for rs in ex.map(_job, [(lg, a.n) for lg in langs]) for r in rs]
    coef = fit(rows)
    ev = evaluate(rows)
    out = {"coef": coef.tolist(), "features": FEATURES, "clip": CLIP, "prior": PRIOR,
           "fit_on": f"FLEURS validation, {a.n} clips x {len(langs)} languages, pooled",
           "n_words_said": int(sum(r["said"] for r in rows)),
           "n_words_omitted": int(sum(not r["said"] for r in rows)), "dev": ev}
    (ship_dir() / "omission.json").write_text(json.dumps(out, indent=1, default=float))
    save_json("omission.json", out)
    print("coef", np.round(coef, 3).tolist())
    print("cv", json.dumps(ev["cv"], indent=1, default=float))
    for lg, r in ev["leave_one_language_out"].items():
        print(lg, f"auc {r['auc']:.3f}", " ".join(
            f"{k}: FA {v['false_omitted_old']:.3f}->{v['false_omitted_new']:.3f} "
            f"hit {v['caught_old'] if v['caught_old'] is None else round(v['caught_old'], 2)}->"
            f"{v['caught_new'] if v['caught_new'] is None else round(v['caught_new'], 2)}"
            for k, v in r.items() if k != "auc"))


if __name__ == "__main__":
    main()
