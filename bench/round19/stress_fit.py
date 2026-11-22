"""Round 19b (German): when should the stress channel lower a word? Trade-off curve and choice.

    python -m bench.round19.stress_fit [--save]

Rows: bench.round19.stress_collect (dev clips: native words, and 2 planted stress errors per clip).
Shipped: factor = P_pros (word-level stress calibrator, data/calib/prosody/de.json).
Candidates, all fitted on the FIT fold and read on the CHECK fold:
  knee        f = min(1, P_pros / k)
  rel         a reliability-aware recalibration of the stress evidence: logistic regression of
              "stress error" vs "native" on [min stress logit, second min, n scored syllables] and
              the min logit's interactions with the reliability of the pitch / energy evidence
              (voiced share of the word, mean voicing probability, word energy, log n syllables,
              clip SNR, speaking rate); then f = min(1, P_rel / k).
For each k: natives lowered by stress (share of native words with sounds x presence >= tau but
word P < tau; tau = 0.8, the Lingolix operating point, and 0.5) against planted stress-error word
AUC / hit @ 5 % native flags (word P), pooled and by source.
Rule (fixed before reading the check fold): among settings whose check-fold native-by-stress flags
at tau = 0.8 fall by >= 50 % and whose check-fold stress-error AUC is >= shipped - 0.01, take the one
with the highest stress-error AUC (ties: larger fall). Azure has no per-word stress, so the "no worse
than Azure" condition has no figure to compare with (results/SCOREBOARD.md).
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np

from bench.round19.devset import WORK

CAL = Path(__file__).resolve().parents[2] / "pronunciada" / "data" / "calib" / "prosody" / "de.json"
REL = ("voiced_share", "voicing", "energy_db", "log_nsyl", "snr_db", "rate")
KNEES = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
SHIFTS = (0.5, 1.0, 1.5, 2.0, 3.0)


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def table(rows):
    q = [np.sort(logit(r["vals"])) for r in rows]
    F = np.array([[x[0], x[1] if len(x) > 1 else x[0], len(x)] for x in q], np.float64)
    R = np.array([[r["voiced_share"], r["voicing"], r["energy_db"], math.log(max(1, r["n_syl"])),
                   r["snr_db"] if r["snr_db"] is not None else np.nan, r["rate"]] for r in rows], np.float64)
    med = np.nanmedian(R, 0)
    R = np.where(np.isfinite(R), R, med)
    return F, R


def shipped_pp(F, s=0.5):
    cal = json.loads(CAL.read_text())
    b = cal["coef"]
    z = b[0] + b[1] * F[:, 0] + b[2] * F[:, 1] + b[3] * F[:, 2]
    zl, zs = z + cal["c_len"], np.minimum(z + cal["c_str"], z + cal["c_len"])
    return 1 / (1 + np.exp(-((1 - s) * zl + s * zs)))


def design(F, R, mu, sd):
    Rz = (R - mu) / sd
    return np.column_stack([F, F[:, :1] * Rz, Rz])


def fit_rel(F, R, y, ctl):
    from sklearn.linear_model import LogisticRegression

    mu, sd = R.mean(0), R.std(0) + 1e-9
    X = design(F, R, mu, sd)
    xm, xs = X.mean(0), X.std(0) + 1e-9
    w = np.where(y == 1, 0.1 / 0.9 * (y == 0).sum() / max(1, (y == 1).sum()), 1.0)
    m = LogisticRegression(C=0.1, max_iter=3000).fit((X - xm) / xs, 1 - y, sample_weight=w)
    coef = m.coef_[0] / xs
    b0 = float(m.intercept_[0] - (m.coef_[0] * xm / xs).sum())
    # intercept: same share of native words below P 0.5 as the shipped calibrator
    target = float((shipped_pp(F[ctl]) < 0.5).mean())
    z = X[ctl] @ coef + b0
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if ((z + mid) < 0).mean() > target:
            lo = mid
        else:
            hi = mid
    return {"rel_mu": mu.tolist(), "rel_sd": sd.tolist(), "coef": coef.tolist(), "b0": b0 + hi, "rel": list(REL)}


def rel_pp(par, F, R):
    X = design(F, R, np.array(par["rel_mu"]), np.array(par["rel_sd"]))
    return 1 / (1 + np.exp(-(X @ np.array(par["coef"]) + par["b0"])))


def metrics(rows, pp_fac):
    from bench.metrics import roc_auc

    snd = np.array([r["sounds"] * (r["present"] if r["present"] is not None else 1.0) for r in rows])
    prob = snd * pp_fac
    clean = np.array([r["ref"] == "clean" for r in rows])
    err = np.array([r["ref"] == "stress" and r["wrong"] for r in rows])
    out = {}
    for tau in (0.8, 0.5):
        out[f"native_by_stress_{tau}"] = float(np.mean((snd[clean] >= tau) & (prob[clean] < tau)))
    out["native_below_0.5"] = float(np.mean(prob[clean] < 0.5))
    yy = np.r_[np.ones(err.sum()), np.zeros(clean.sum())]
    out["stress_auc"] = roc_auc(yy, -np.r_[prob[err], prob[clean]])
    out["stress_hit5"] = float(np.mean(prob[err] < np.quantile(prob[clean], 0.05)))
    out["n_err"], out["n_native"] = int(err.sum()), int(clean.sum())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args(argv)
    rows = pickle.loads((WORK / "stress_rows.pkl").read_bytes())
    rows = [r for r in rows if r["ref"] == "clean" or r["wrong"]]
    fit = [r for r in rows if r["fold"] == "fit"]
    chk = [r for r in rows if r["fold"] == "check"]
    Ff, Rf = table(fit)
    yf = np.array([int(r["wrong"]) for r in fit])
    ctl = np.array([r["ref"] == "clean" for r in fit])
    par = fit_rel(Ff, Rf, yf, ctl)
    Fc, Rc = table(chk)
    curves = []
    base = {}
    grid = [(f, k) for f in ("knee", "rel") for k in KNEES] + [("relshift", d) for d in SHIFTS] \
        + [("shift", d) for d in SHIFTS]
    for fam, k in grid:
        if True:
            res = {"family": fam, "knee": k}
            for tag, rs, F, R in (("fit", fit, Ff, Rf), ("check", chk, Fc, Rc)):
                if fam in ("relshift", "shift"):     # k = log-odds shift towards acceptance
                    z = logit(rel_pp(par, F, R) if fam == "relshift" else shipped_pp(F)) + k
                    fac = 1 / (1 + np.exp(-z))
                else:
                    pp = shipped_pp(F) if fam == "knee" else rel_pp(par, F, R)
                    fac = np.minimum(1.0, pp / k) if k > 0 else pp
                res[tag] = metrics(rs, fac)
                for src in ("cv", "fleurs"):
                    sel = [i for i, r in enumerate(rs) if r["src"] == src]
                    res[f"{tag}_{src}"] = metrics([rs[i] for i in sel], fac[sel])
            if fam == "knee" and k == 0.0:
                base = res
            curves.append(res)
    b = base["check"]
    print(f"shipped (check): natives lowered by stress @0.8 {100 * b['native_by_stress_0.8']:.2f} % @0.5 "
          f"{100 * b['native_by_stress_0.5']:.2f} %  stress AUC {b['stress_auc']:.4f} hit5 {b['stress_hit5']:.3f} "
          f"(n err {b['n_err']}, natives {b['n_native']})")
    ok = []
    for c in curves:
        m = c["check"]
        red = 1 - m["native_by_stress_0.8"] / max(b["native_by_stress_0.8"], 1e-9)
        c["reduction_0.8"] = red
        c["d_auc"] = m["stress_auc"] - b["stress_auc"]
        print(f"{c['family']:5s} k {c['knee']:.2f}: by-stress @0.8 {100 * m['native_by_stress_0.8']:.2f} % "
              f"(-{100 * red:.0f} %) @0.5 {100 * m['native_by_stress_0.5']:.2f} %  AUC {m['stress_auc']:.4f} "
              f"({c['d_auc']:+.4f}) hit5 {m['stress_hit5']:.3f} | CV AUC {c['check_cv']['stress_auc']:.4f} "
              f"FLEURS AUC {c['check_fleurs']['stress_auc']:.4f}")
        if red >= 0.5 and c["d_auc"] >= -0.01:
            ok.append(c)
    pick = max(ok, key=lambda c: (round(c["check"]["stress_auc"], 4), c["reduction_0.8"])) if ok else None
    print("pick:", None if pick is None else (pick["family"], pick["knee"]))
    out = {"rel": par, "curves": curves, "pick": None if pick is None else {"family": pick["family"], "knee": pick["knee"]}}
    (WORK / "stress_fit.json").write_text(json.dumps(out, indent=1))
    if a.save and pick:
        Path("results/tables/round19_de_stress_curve.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
