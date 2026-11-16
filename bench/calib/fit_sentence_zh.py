"""Round 7, optional step: Mandarin sentence scores recalibrated on real learners (OMPAL DEV).

    python -m bench.calib.fit_sentence_zh [--write]

Per teacher score (accuracy, fluency, prosody), a ridge regression (alpha 1) on strictness-free
inputs the Scorer already has: the four round-4 sentence heads (scores accuracy / fluency /
prosodic / total, 0..1) and three tone summaries of the utterance (mean syllable tone_prob, mean
log tone_prob, share of syllables with tone_prob < 0.5). Target = mean teacher score (1..5);
output score = clip((prediction - 1) / 4, 0, 1), i.e. the teacher scale mapped to 0..1.
Selection: 5-fold speaker-grouped CV on OMPAL DEV (learners); adopt a score only if its CV Pearson
gains >= 0.03 over today's score (accuracy must pass for anything to be adopted). Guard: on
FLEURS-zh VALIDATION natives (200 clips, clean prompts) the median of an adopted score may drop by
at most 0.03 (0.3 points on the 0-10 scale). Never OMPAL TEST or FLEURS test.
Writes pronunciada/data/calib/sentence_zh_recal.json (read by Scorer for zh only).
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

SCORES = (("accuracy", "accuracy"), ("fluency", "fluency"), ("prosodic", "prosody"))
HEADS = ("accuracy", "fluency", "prosodic", "total")


def features(scores: dict, tone_probs) -> list[float]:
    """Shared with the runtime (pronunciada.scorer._sentence_recal)."""
    t = np.asarray([x for x in tone_probs if x is not None] or [1.0], np.float64)
    return [float(scores[k]) for k in HEADS] + [float(t.mean()), float(np.log(np.clip(t, 1e-3, 1)).mean()),
                                                float((t < 0.5).mean())]


def ompal_dev():
    import bench.eval_ompal as E

    recs = E.records("dev")
    out, _ = E.ours_outputs(recs)
    X, Y, G = [], [], []
    for r in recs:
        o, s = out.get(r["id"]), r["labels"]["sentence"]
        if not o or o.get("status") != "Success" or not s or r["labels"]["native"]:
            continue
        X.append(features(o["sentence"], [t for *_, t in o["syllables"]]))
        Y.append([s[k] for _, k in SCORES])
        G.append(r["speaker"])
    return np.array(X), np.array(Y, float), np.array(G)


def fleurs_val(n=200):
    from bench.calib.common import calib_cache

    p = calib_cache() / f"sentence_zh_fleurs_val_{n}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    from bench.prosody.prep import fleurs_validation
    from pronunciada import prosody
    from pronunciada.scorer import Scorer

    sc = Scorer("zh", threads=2)
    recs = list(fleurs_validation("zh"))
    rng = np.random.default_rng(0)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), min(n, len(recs)), replace=False))]
    X = []
    for r in recs:
        try:
            res = sc.score((r["wav"], 16000), r["text"])
        except ValueError:
            continue
        X.append(features(res.scores, [s.tone_prob for w in res.words for s in w.syllables]))
    out = (np.array(X), prosody.selected_tone_model())
    p.write_bytes(pickle.dumps(out))
    return out


def main():
    from sklearn.linear_model import Ridge

    from bench.metrics import pearson
    from pronunciada import prosody

    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    X, Y, G = ompal_dev()
    XF, tm = fleurs_val()
    assert tm == prosody.selected_tone_model()
    sp = sorted(set(G))
    fo = np.array([sp.index(g) % 5 for g in G])
    res, adopt = {}, {}
    for j, (ours, teach) in enumerate(SCORES):
        pr = np.zeros(len(Y))
        for i in range(5):
            m = Ridge(1.0).fit(X[fo != i], Y[fo != i, j])
            pr[fo == i] = m.predict(X[fo == i])
        base = pearson(X[:, HEADS.index(ours)], Y[:, j])
        cvp = pearson(pr, Y[:, j])
        m = Ridge(1.0).fit(X, Y[:, j])
        newF = np.clip((m.predict(XF) - 1) / 4, 0, 1)
        med_old, med_new = float(np.median(XF[:, HEADS.index(ours)])), float(np.median(newF))
        ok = cvp - base >= 0.03 and med_old - med_new <= 0.03
        res[ours] = {"teacher": teach, "pearson_today": base, "pearson_cv": cvp, "fleurs_native_median_today": med_old,
                     "fleurs_native_median_new": med_new, "adopt": bool(ok),
                     "coef": [float(m.intercept_)] + [float(c) for c in m.coef_]}
        print(f"{ours:9s} today {base:.3f}  CV {cvp:.3f}  FLEURS native median {med_old:.3f} -> {med_new:.3f}  "
              f"{'ADOPT' if ok else 'keep'}")
        adopt[ours] = ok
    if not adopt["accuracy"]:
        print("accuracy does not pass: nothing adopted")
        for k in res:
            res[k]["adopt"] = False
    from bench.calib.common import save_json

    save_json("sentence_zh_recal_cv.json", {"n_utts": int(len(Y)), "n_speakers": len(sp), "tone_model": tm, **res})
    if a.write and any(r["adopt"] for r in res.values()):
        from pronunciada.calibrate import data_dir

        out = {"lang": "zh", "tone_model": tm, "inputs": list(HEADS) + ["mean_tone_prob", "mean_log_tone_prob",
                                                                       "share_tone_prob_below_0.5"],
               "output": "clip((ridge - 1) / 4, 0, 1)", "fit_on": "OMPAL DEV learners (28 speakers)",
               "scores": {k: {"coef": r["coef"]} for k, r in res.items() if r["adopt"]}, "cv": res}
        (data_dir() / "sentence_zh_recal.json").write_text(json.dumps(out, indent=1))
        print("wrote sentence_zh_recal.json", [k for k, r in res.items() if r["adopt"]])


if __name__ == "__main__":
    main()
