"""Round 19: does a log-mel VTLN warp for high-F0 voices make the acoustic evidence agree better
with the experts? speechocean762 TRAIN, no fitting.

    python -m bench.round19.vtln_screen --warped DIR

Compares the plain feature cache with one extracted under PRONUNCIADA_VTLN (only clips with median
F0 above the threshold differ). Reports, per group (children <= 12 / adults, warped clips only):
the Pearson of the expert phone score with the sound evidence (llr_sub, z_llr_sub, peak_post) and
of the sentence total with utterance means of the same, plain vs warped.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from bench.round19 import data as RD

COLS = ("llr_sub", "z_llr_sub", "peak_post", "llr_del")


def load(d):
    out = {}
    for p in sorted(Path(d).glob("shard*.pkl")):
        for k, v in pickle.loads(p.read_bytes()).items():
            if "error" not in v:
                out[k] = {"tab": v["tab_num"], "names": v["tab_names"], "lab": v["labels"], "f0": v["f0"],
                          "sent": v["sent"]}
    return out


def main():
    from bench.calib.common import load_json, save_json

    ap = argparse.ArgumentParser()
    ap.add_argument("--warped", required=True)
    ap.add_argument("--tag", default="vtln091_220")
    a = ap.parse_args()
    A, B = load(RD.feat_dir("train")), load(a.warped)
    ids = [k for k in A if k in B]
    changed = [k for k in ids if not np.array_equal(np.nan_to_num(A[k]["tab"]), np.nan_to_num(B[k]["tab"]))]
    print(f"{len(ids)} utts, {len(changed)} warped", flush=True)
    res = {"n_warped": len(changed)}
    for grp, f in (("children", lambda x: x <= 12), ("adults", lambda x: x > 12)):
        I = [k for k in changed if f(A[k]["lab"]["age"])]
        r = {"n_utts": len(I)}
        for c in COLS:
            for nm, S in (("plain", A), ("warped", B)):
                xs, ys, us, ut = [], [], [], []
                for k in I:
                    j = list(S[k]["names"]).index(c)
                    v = S[k]["tab"][:, j].astype(float)
                    y = np.asarray(S[k]["lab"]["phone"], float)
                    m = np.isfinite(v) & np.isfinite(y)
                    xs += list(v[m])
                    ys += list(y[m])
                    us.append(np.nanmean(v))
                    ut.append(S[k]["lab"]["sentence"]["total"])
                r[f"{c}_phone_{nm}"] = RD.pearson(xs, ys)
                r[f"{c}_utt_total_{nm}"] = RD.pearson(us, ut)
        r["r4_total_plain"] = RD.pearson([A[k]["sent"]["total"] for k in I], [A[k]["lab"]["sentence"]["total"] for k in I])
        r["r4_total_warped"] = RD.pearson([B[k]["sent"]["total"] for k in I], [B[k]["lab"]["sentence"]["total"] for k in I])
        res[grp] = r
        print(grp, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    R = load_json("round19_english_vtln.json") or {}
    R[a.tag] = res
    save_json("round19_english_vtln.json", R)


if __name__ == "__main__":
    main()
