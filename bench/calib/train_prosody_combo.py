"""Round 2: learned combination of the sounds-based word probability with tone / stress / pitch.

    python -m bench.calib.train_prosody_combo [--n 150] [--workers 5]

FLEURS VALIDATION only. Per language with a prosody model (zh tone, ja pitch accent, en de es it
pt ru stress), every sampled clip goes through the public Scorer (prosody on, combination off)
against its correct reference and against controlled-error references: the five sound kinds
(phone_sub_near, phone_sub_far, phone_ins, phone_del, word_swap) and the language's prosodic kind
(bench.corrupt `tone` / `stress` / `pitch`), 2 errors per utterance, fixed seeds.
Per word: z = sounds-based word logit at s = 0.5, q = logit of the word's lowest tone / stress /
pitch probability, n = syllables. Label: words of the correct reference and untouched words of the
corrupted references -> ok; a word with a controlled sound or prosodic error -> not ok.
Model: logistic regression on [z, q, n] (words that have q; others keep the sounds probability).
Evaluation: 5-fold utterance-grouped CV on the same validation words. Adopted only if native word
acceptance at s = 0.5 stays >= 0.97, the prosodic-error word AUC improves, and the sound-error word
AUC drops by <= 0.005. Writes results/tables/prosody_combo.json and the "prosody_combo" entry of
pronunciada/data/calib/config.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from bench.calib.common import save_json, ship_dir

PROS_KIND = {"zh": "tone", "ja": "pitch", "en": "stress", "de": "stress", "es": "stress",
             "it": "stress", "pt": "stress", "ru": "stress"}
SOUND_KINDS = ("phone_sub_near", "phone_sub_far", "phone_ins", "phone_del", "word_swap")


def _word_rows(res, kind_tag, wrong, field):
    c = res._pronunciada  # noqa: SLF001
    from pronunciada.calibrate import blend_logit

    zw = blend_logit(c["w_zl"], np.minimum(c["w_zs"], c["w_zl"]), 0.5)
    rows = []
    for wi, w in enumerate(res.words):
        vals = [getattr(s, field) for s in w.syllables if getattr(s, field) is not None]
        rows.append((float(zw[wi]), min(vals) if vals else None, len(w.syllables), kind_tag, bool(wrong[wi])))
    return rows


def run_lang(args):
    lang, n = args
    import os

    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.common import iter_parquet_audio, records, stable_seed
    from bench.pipeline.evalctl import inventory
    from pronunciada.scorer import Scorer

    sc = Scorer(lang, threads=1)
    sc.combo = None
    field = {"tone": "tone_prob", "stress": "stress_prob", "pitch": "pitch_prob"}[PROS_KIND[lang]]
    if sc.prosody is None:
        return lang, {"error": f"no prosody model: {sc.prosody_error}"}
    recs = records("fleurs-validation", lang=lang)
    rng = np.random.default_rng(0)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), min(n, len(recs)), replace=False))]
    by = {int(r["audio"].split(":")[2]): r for r in recs}
    path = recs[0]["audio"].split(":")[1]
    inv = inventory(lang)
    out = []
    for i, wav in iter_parquet_audio(path, set(by)):
        rec = by[i]
        try:
            res = sc.score((wav, 16000), rec["text"])
        except ValueError:
            continue
        ref = res._pronunciada["ref"]  # noqa: SLF001
        out += [(rec["id"],) + r for r in _word_rows(res, "clean", np.zeros(len(ref.words), bool), field)]
        for kind in SOUND_KINDS + (PROS_KIND[lang],):
            try:
                cor = corrupt(ref, [kind], inv, n_errors=2, seed=stable_seed(rec["id"], kind, "combo-v1"), strict=False)
            except NotApplicable:
                continue
            if not cor.errors:
                continue
            r2 = sc.score((wav, 16000), cor.ref)
            tag = "prosody" if kind == PROS_KIND[lang] else "sound"
            out += [(rec["id"],) + r for r in _word_rows(r2, tag, np.asarray(cor.word_wrong), field)]
    return lang, {"rows": out, "prosody_error": sc.prosody_error}


def evaluate(rows):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    from bench.metrics import roc_auc

    utt = np.array([r[0] for r in rows])
    z = np.array([r[1] for r in rows])
    q = np.array([np.nan if r[2] is None else r[2] for r in rows])
    ns = np.array([r[3] for r in rows], float)
    tag = np.array([r[4] for r in rows])
    wrong = np.array([r[5] for r in rows])
    has = np.isfinite(q)
    qq = np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
    X = np.column_stack([z, np.where(has, qq, 0.0), ns])
    y = (~wrong).astype(int)
    zc = z.copy()
    idx = np.flatnonzero(has)
    for tr, te in GroupKFold(5).split(idx, None, utt[idx]):
        m = LogisticRegression(C=1.0, max_iter=1000).fit(X[idx[tr]], y[idx[tr]])
        zc[idx[te]] = m.decision_function(X[idx[te]])
    final = LogisticRegression(C=1.0, max_iter=1000).fit(X[idx], y[idx]) if len(idx) else None
    p0, p1 = 1 / (1 + np.exp(-z)), 1 / (1 + np.exp(-zc))
    clean = tag == "clean"

    def auc(p, sel):
        m = sel | clean
        yy = (sel & wrong)[m].astype(int)
        return roc_auc(yy, 1 - p[m]) if 0 < yy.sum() < len(yy) else float("nan")

    res = {"n_words": int(len(rows)), "share_with_prosody": float(has.mean()),
           "native_accept": {"sounds": float(np.mean(p0[clean] >= 0.5)), "combined": float(np.mean(p1[clean] >= 0.5))},
           "prosodic_auc": {"sounds": auc(p0, tag == "prosody"), "combined": auc(p1, tag == "prosody")},
           "sound_auc": {"sounds": auc(p0, tag == "sound"), "combined": auc(p1, tag == "sound")},
           "n_prosodic_errors": int((wrong & (tag == "prosody")).sum()),
           "n_sound_errors": int((wrong & (tag == "sound")).sum())}
    res["coef"] = ([float(final.intercept_[0])] + [float(c) for c in final.coef_[0]]) if final is not None else None
    res["adopted"] = bool(final is not None and res["native_accept"]["combined"] >= 0.97
                          and res["prosodic_auc"]["combined"] > res["prosodic_auc"]["sounds"]
                          and res["sound_auc"]["combined"] >= res["sound_auc"]["sounds"] - 0.005)
    return res


def main():
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    out = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for lang, d in ex.map(run_lang, [(l, a.n) for l in PROS_KIND]):
            if "error" in d:
                out[lang] = {"kind": PROS_KIND[lang], "adopted": False, "error": d["error"]}
            else:
                out[lang] = {"kind": PROS_KIND[lang], **evaluate(d["rows"])}
            print(lang, json.dumps({k: v for k, v in out[lang].items() if k != "coef"}), flush=True)
    save_json("prosody_combo.json", out)
    p = ship_dir() / "config.json"
    cfg = json.loads(p.read_text())
    cfg["prosody_combo"] = {l: {"kind": v["kind"], "adopted": v["adopted"], "coef": v.get("coef"),
                                "features": "[1, sounds word logit at s, logit(min prosody prob), n syllables]"}
                            for l, v in out.items()}
    p.write_text(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
