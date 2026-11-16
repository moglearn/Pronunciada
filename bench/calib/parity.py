"""Parity: per-phone features recomputed through the RUNTIME path vs the stored feature tables.

    python -m bench.calib.parity [--n 200]

Runtime path = pronunciada.text G2P -> pronunciada.adapter.apply -> pronunciada.align.score_reference ->
pronunciada.features.phone_rows / derive (cached ZIPA posteriors as input, the same as the tables).
For FLEURS validation the corrupted references are rebuilt with bench.pipeline.features.fleurs_utt
(which itself calls pronunciada.features). Reports max |difference| per numeric column and the number
of mismatching string values; writes results/tables/feature_parity.json.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np


def runtime_rows(dataset: str, rec: dict, lang: str = "en"):
    from bench.pipeline.common import build_ref, load_lp
    from pronunciada.adapter import apply
    from pronunciada.align import score_reference
    from pronunciada.features import phone_rows
    from pronunciada.variants import NativeStats

    ref = build_ref(lang, rec["text"])
    lp = load_lp(dataset, rec)
    if ref is None or lp is None:
        return None
    lpa = apply(lp, lang)
    ev = score_reference(lpa, ref)
    return phone_rows(lpa, ref, ev, NativeStats.load(lang), float(rec["dur"]))


def compare(old, new_rows, cols_num, cols_str, acc):
    for c in cols_num:
        a = old[c].to_numpy(dtype=float)
        b = np.array([np.nan if r.get(c) is None else float(r[c]) for r in new_rows])
        both_nan = np.isnan(a) & np.isnan(b)
        nan_mismatch = int((np.isnan(a) ^ np.isnan(b)).sum())
        d = np.where(both_nan, 0.0, np.abs(a - b))
        d = d[np.isfinite(d)]
        e = acc.setdefault(c, {"max_abs": 0.0, "nan_mismatch": 0, "n": 0, "n_diff_gt_1e-9": 0})
        e["max_abs"] = max(e["max_abs"], float(d.max()) if len(d) else 0.0)
        e["n_diff_gt_1e-9"] += int((d > 1e-9).sum())
        e["nan_mismatch"] += nan_mismatch
        e["n"] += len(a)
    for c in cols_str:
        a = [None if (x is None or (isinstance(x, float) and math.isnan(x))) else str(x) for x in old[c]]
        b = [None if r.get(c) is None else str(r[c]) for r in new_rows]
        e = acc.setdefault(c, {"mismatch": 0, "n": 0})
        e["mismatch"] += sum(x != y for x, y in zip(a, b))
        e["n"] += len(a)


def main():
    import pandas as pd

    from bench.pipeline.analyze import save
    from bench.pipeline.common import feat_dir, records

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    fd = feat_dir()
    out = {}
    NUM = None
    for ds, n in (("speechocean762", a.n), ("epadb", a.n // 2), ("l2arctic", a.n // 2)):
        df = pd.read_parquet(fd / f"{ds}.parquet")
        recs = {r["id"]: r for r in records(ds)}
        utts = list(dict.fromkeys(df.utt))
        rng = np.random.default_rng(0)
        pick = [utts[i] for i in sorted(rng.choice(len(utts), min(n, len(utts)), replace=False))]
        acc = {}
        n_rows = n_len_mismatch = 0
        if NUM is None:
            from pronunciada.features import STR_COLS

            sample = runtime_rows(ds, recs[pick[0]])
            NUM = [c for c in sample[0] if c not in STR_COLS and c in df.columns]
            STR = [c for c in sample[0] if c in STR_COLS and c in df.columns]
        for u in pick:
            old = df[df.utt == u].sort_values("phone_index")
            new = runtime_rows(ds, recs[u])
            if new is None or len(new) != len(old):
                n_len_mismatch += 1
                continue
            compare(old, new, NUM, STR, acc)
            n_rows += len(old)
        out[ds] = {"utts": len(pick), "rows": n_rows, "utts_len_mismatch": n_len_mismatch,
                   "columns": acc}
        print(ds, len(pick), "utts", n_rows, "rows; worst:",
              sorted(((k, v.get("max_abs", v.get("mismatch"))) for k, v in acc.items()),
                     key=lambda x: -x[1])[:4], flush=True)
    # FLEURS validation incl. corrupted references (through the pipeline, which calls pronunciada.features)
    from bench.pipeline.common import LANGS
    from bench.pipeline.features import config, fleurs_utt

    df = pd.read_parquet(fd / "fleurs-validation.parquet")
    acc = {}
    n_rows = n_len_mismatch = n_utts = 0
    for lang in LANGS:
        recs = records("fleurs-validation", lang=lang)
        sub = df[df.lang == lang]
        utts = list(dict.fromkeys(sub.utt))
        rng = np.random.default_rng(1)
        pick = set(utts[i] for i in rng.choice(len(utts), min(a.n // 10, len(utts)), replace=False))
        cfg = config(lang)
        for r in recs:
            if r["id"] not in pick:
                continue
            r["_dataset"] = "fleurs-validation"
            new = fleurs_utt((r, lang, cfg))
            old = sub[sub.utt == r["id"]]
            n_utts += 1
            for kind, o in old.groupby("ref_kind", sort=False):
                nn = [x for x in new if x["ref_kind"] == kind]
                o = o.sort_values("phone_index")
                if len(nn) != len(o):
                    n_len_mismatch += 1
                    continue
                compare(o, nn, NUM, STR, acc)
                n_rows += len(o)
    out["fleurs-validation"] = {"utts": n_utts, "rows": n_rows, "refs_len_mismatch": n_len_mismatch,
                                "columns": acc}
    print("fleurs-validation", n_utts, "utts", n_rows, "rows; worst:",
          sorted(((k, v.get("max_abs", v.get("mismatch"))) for k, v in acc.items()), key=lambda x: -x[1])[:4])
    save("feature_parity.json", out)


if __name__ == "__main__":
    main()
