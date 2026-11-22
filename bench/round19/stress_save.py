"""Round 19 (German): write data/calib/stress_rel/de.json from the dev fit (bench/round19/stress_fit.py).

    python -m bench.round19.stress_save [--shift 1.0]

The reliability-weighted recalibration fitted on the dev FIT fold, with the log-odds shift of the
chosen trade-off point (1.0 = natives lowered by stress -48 %, stress-error AUC -0.018 on the dev check
fold), and the fit-fold medians that replace a missing reliability input.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from bench.round19.devset import WORK
from bench.round19.stress_fit import table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=float, default=1.0)
    a = ap.parse_args()
    fit = json.loads((WORK / "stress_fit.json").read_text())["rel"]
    rows = pickle.loads((WORK / "stress_rows.pkl").read_bytes())
    rows = [r for r in rows if (r["ref"] == "clean" or r["wrong"]) and r["fold"] == "fit"]
    R = np.array([[r["voiced_share"], r["voicing"], r["energy_db"], np.log(max(1, r["n_syl"])),
                   r["snr_db"] if r["snr_db"] is not None else np.nan, r["rate"]] for r in rows], np.float64)
    med = np.nanmedian(R, 0)
    _, R2 = table(rows)
    assert np.allclose(R2.mean(0), np.array(fit["rel_mu"]))
    out = {"lang": "de", **fit, "rel_median": med.tolist(), "shift": a.shift,
           "features": "[min stress logit, second min, n scored syllables] + min logit x standardised reliability + reliability",
           "fit_on": "round-19 German dev fit fold: native words and 2 planted stress errors per clip (Common Voice dev, FLEURS validation)"}
    p = Path("pronunciada/data/calib/stress_rel/de.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    print("written", p)


if __name__ == "__main__":
    main()
