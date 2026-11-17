"""Round 16 guard: the English contrast cap (dental fricatives) on speechocean762 TRAIN, using the
sound calibrator's out-of-fold logits (sound_oof.parquet, B_z_len / B_z_str) and the training
feature table: phone Pearson of the shown probability, detection AUC (expert score < 2), and the
number of sounds the cap touches.

    python -m bench.calib.caps_so762 path/to/contrast_caps.json
"""
from __future__ import annotations

import json
import sys

import numpy as np


def main(path):
    import pandas as pd

    from bench.calib.common import calib_cache, load_table, table_columns
    from bench.metrics import pearson, roc_auc
    from pronunciada import calibrate as C

    df = load_table("speechocean762", "train")
    t = table_columns(df)
    od = pd.read_parquet(calib_cache() / "sound_oof.parquet", columns=["utt", "phone_index", "B_z_len", "B_z_str"])
    key = {(u, int(i)): j for j, (u, i) in enumerate(zip(od.utt.values, od.phone_index.values))}
    j = np.array([key.get((u, int(i)), -1) for u, i in zip(df.utt.values, df.phone_index.values)])
    ok = j >= 0
    zl = np.where(ok, od.B_z_len.values[np.maximum(j, 0)], np.nan)
    zs = np.where(ok, od.B_z_str.values[np.maximum(j, 0)], np.nan)
    raw = json.loads(open(path).read())
    c = raw["classes"]["dental_fricative"]
    caps = [{"class": "dental_fricative", "members": set(c["members"]["en"]), "gate": c["gate"],
             "coef": np.asarray(c["coef"]), "b": float(c["b"]), "offset": float(raw["langs"]["en"]["dental_fricative"])}]
    tt = {k: np.asarray(t[k]) for k in C.CAP_FEATS}
    tt["phone"], tt["comp_tok"] = list(df.phone.values), list(df.comp_tok.values)
    zl2, zs2 = C.apply_contrast_caps(caps, tt, zl, zs)
    y = np.asarray(t["phone_score"], np.float64)
    m = np.isfinite(y) & ok
    out = {}
    for name, a, b in (("before", zl, zs), ("after", zl2, zs2)):
        p = 1 / (1 + np.exp(-C.blend_logit(a, np.minimum(b, a), 0.5)))
        out[name] = {"pearson_prob": float(pearson(p[m], y[m])), "auc_lt2": float(roc_auc((y[m] < 2).astype(int), -p[m]))}
    out["n_changed"] = int(np.sum(np.abs(zl2 - zl) > 1e-9))
    out["n_sounds"] = int(m.sum())
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
