"""Round 19 (German): a knee on the stress factor of the word probability (offline sweep).

    python -m bench.round19.stress_knee

Shipped: WordResult.prob = sounds x P_pros x present, P_pros the round-3 / 4 word-level stress
calibrator. Candidate: P_pros' = min(1, P_pros / k), i.e. no penalty while the stress evidence is
above the knee k, the same shape as the length and contrast factors (natives are rarely below a
low knee). Two data sets, both outside the Lingolix benchmark:
  * dev (bench.round19.devset): segmental planted edits on CV / FLEURS validation natives, the
    Lingolix metric (AUC, FA@80), fit fold;
  * the stress guard: the prosody-channel rows of bench.calib.fit_prosody (FLEURS validation,
    200 clips: correct references, 2 planted stress errors, 5 sound-error kinds), with the
    shipped calibrator: prosodic-error word AUC / hit@5 % (product), sound-error word AUC, native
    word acceptance at s = 0.5.
"""
from __future__ import annotations

import json

import numpy as np

from bench.round19.devset import WORK

GRID = (0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0)


def guard_rows():
    """Collected afresh with the current package (the bench cache predates the 2026-11-17 refit);
    the stress channel calibrator is switched off during collection (fit_prosody.collect)."""
    import pickle

    from bench.calib import fit_prosody as FP

    p = WORK / "prosody_rows_de_200.pkl"
    if not p.exists():
        rows, meta = FP.collect("de", 200)
        p.write_bytes(pickle.dumps(rows))
    return pickle.loads(p.read_bytes())


def guard_metrics(rows, k):
    from bench.calib import fit_prosody as FP
    from bench.metrics import roc_auc

    par = json.loads((FP.__file__ and __import__("pathlib").Path(FP.__file__).resolve().parents[2]
                      / "pronunciada" / "data" / "calib" / "prosody" / "de.json").read_text())
    F = FP._feats(rows)
    pp = FP._pp(par, F, 0.5)
    if k > 0:
        pp = np.minimum(1.0, pp / k)
    snd = np.array([r["snd"][1] for r in rows])
    prod = snd * pp
    ref = np.array([r["ref"] for r in rows])
    wrong = np.array([r["wrong"] for r in rows])
    clean = ref == "clean"

    def det(score, tag):
        m = (ref == tag) & wrong | clean
        y = ((ref == tag) & wrong)[m].astype(int)
        thr = np.quantile(score[clean], 0.05)
        return roc_auc(y, 1 - score[m]), float(np.mean(score[(ref == tag) & wrong] < thr))

    pa, ph = det(prod, "prosody")
    sa, sh = det(prod, "sound")
    return {"stress_auc": pa, "stress_hit5": ph, "sound_auc": sa, "sound_hit5": sh,
            "native_accept": float(np.mean(prod[clean] >= 0.5))}


def dev_metrics(k, fold="fit", source=None):
    from bench.round19.whatif import metrics

    def fn(w, c):
        st = w.get("stress")
        if not st or k <= 0:
            return w["prob"]
        return w["prob"] / st * min(1.0, st / k)

    return metrics("dev-base", fn, fold=fold, source=source)


def main():
    rows = guard_rows()
    out = []
    for k in GRID:
        g = guard_metrics(rows, k)
        d = dev_metrics(k)
        dc = dev_metrics(k, source="cv")
        r = {"knee": k, **g, "dev_auc": d["auc"], "dev_fa80": d["fa80"], "dev_cv_auc": dc["auc"], "dev_cv_fa80": dc["fa80"]}
        out.append(r)
        print(" ".join(f"{a} {b:.4f}" for a, b in r.items()), flush=True)
    (WORK / "stress_knee_sweep.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
