"""Round 20 (German, time-boxed): can a narrow gate give back the stress detection the round-19
reliability factor cost, without lowering natives?

    python -m bench.round20_mandarin.de_stress [--save]

Rows: bench.round19.stress_collect (German dev clips only: Common Voice dev + FLEURS validation; native
words and 2 planted stress errors per clip). Word P before the stress factor = sounds x presence.
Arms: "full" = the shipped stress channel P_pros (round-19 before the reliability factor), "rel" = the
shipped reliability factor (pronunciada.stress_rel with data/calib/stress_rel/de.json), and gates:
the full factor where the pitch evidence is reliable (voiced share >= v and mean voicing >= u) AND the
word's sounds are confidently right (sounds x presence >= s0), the reliability factor elsewhere.
Rule (fixed before the grid): native words below 0.6 and below 0.8 (whole dev, and check fold) must not
rise against "rel"; among those, the highest planted stress-error AUC; it is worth a held-out look only
if it recovers at least half of the dev AUC lost between "full" and "rel".
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np

from bench.round19.devset import WORK
from bench.round19.stress_fit import shipped_pp, table
from bench.round20_mandarin.collect import OUT


def rel_factor(rows, R):
    """The shipped reliability factor (runtime function), per row."""
    from pronunciada import stress_rel as SR

    cal = json.loads(Path("pronunciada/data/calib/stress_rel/de.json").read_text())
    pros = json.loads(Path("pronunciada/data/calib/prosody/de.json").read_text())
    return np.array([SR.factor(cal, pros, list(r["vals"]), list(x), 0.5) for r, x in zip(rows, R)])


def metrics(rows, fac):
    from bench.metrics import roc_auc

    snd = np.array([r["sounds"] * (r["present"] if r["present"] is not None else 1.0) for r in rows])
    prob = snd * fac
    clean = np.array([r["ref"] == "clean" for r in rows])
    err = np.array([r["ref"] == "stress" and r["wrong"] for r in rows])
    yy = np.r_[np.ones(err.sum()), np.zeros(clean.sum())]
    return {"native_below_0.6": float(np.mean(prob[clean] < 0.6)), "native_below_0.8": float(np.mean(prob[clean] < 0.8)),
            "stress_auc": roc_auc(yy, -np.r_[prob[err], prob[clean]]),
            "stress_hit5": float(np.mean(prob[err] < np.quantile(prob[clean], 0.05))),
            "n_err": int(err.sum()), "n_native": int(clean.sum())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args(argv)
    rows = pickle.loads((WORK / "stress_rows.pkl").read_bytes())
    rows = [r for r in rows if r["ref"] == "clean" or r["wrong"]]
    F, R = table(rows)
    full = shipped_pp(F)
    rel = rel_factor(rows, R)
    snd = np.array([r["sounds"] * (r["present"] if r["present"] is not None else 1.0) for r in rows])
    chk = np.array([r["fold"] == "check" for r in rows])
    sets = {"all": np.ones(len(rows), bool), "check": chk}

    def ev(fac):
        return {k: metrics([r for r, m in zip(rows, sel) if m], fac[sel]) for k, sel in sets.items()}

    res = {"full": ev(full), "rel": ev(rel), "grid": []}
    for k in ("full", "rel"):
        m = res[k]["all"]
        print(f"{k:5s} nat<.6 {100 * m['native_below_0.6']:.2f} nat<.8 {100 * m['native_below_0.8']:.2f} "
              f"AUC {m['stress_auc']:.4f} hit5 {m['stress_hit5']:.3f} (check AUC {res[k]['check']['stress_auc']:.4f})")
    b = res["rel"]
    for v, u, s0 in itertools.product((0.6, 0.7, 0.8, 0.9), (0.0, 0.97, 0.99), (0.8, 0.9, 0.95, 0.98)):
        gate = (R[:, 0] >= v) & (R[:, 1] >= u) & (snd >= s0)
        fac = np.where(gate, full, rel)
        m = ev(fac)
        ok = all(m[k][q] <= b[k][q] + 1e-12 for k in sets for q in ("native_below_0.6", "native_below_0.8"))
        res["grid"].append({"v": v, "u": u, "s0": s0, "share_gated": float(gate.mean()), "m": m, "ok": ok})
        x = m["all"]
        print(f"v {v} u {u} s0 {s0} gated {100 * gate.mean():.1f}% | nat<.6 {100 * x['native_below_0.6']:.2f} "
              f"<.8 {100 * x['native_below_0.8']:.2f} AUC {x['stress_auc']:.4f} hit5 {x['stress_hit5']:.3f} "
              f"| check AUC {m['check']['stress_auc']:.4f} {'OK' if ok else ''}")
    ok = [g for g in res["grid"] if g["ok"]]
    pick = max(ok, key=lambda g: g["m"]["all"]["stress_auc"]) if ok else None
    need = b["all"]["stress_auc"] + 0.5 * (res["full"]["all"]["stress_auc"] - b["all"]["stress_auc"])
    res["pick"] = pick and {k: pick[k] for k in ("v", "u", "s0")}
    res["worth_heldout"] = bool(pick and pick["m"]["all"]["stress_auc"] >= need)
    print("pick", res["pick"], "AUC needed", round(need, 4), "worth a held-out look:", res["worth_heldout"])
    (OUT / "de_stress.json").write_text(json.dumps(res, indent=1, default=float))
    if a.save:
        Path("results/tables/round20_mandarin_de_stress_dev.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
