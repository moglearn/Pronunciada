"""Paired before / after of contrast-audit rows (FLEURS test), clip bootstrap.

    python bench/round19_merge/audit_paired.py <before cache dir> <after cache dir> lang:contrast:direction ...

Each cache dir is an audit/contrast_v1_test folder (chunks per language). Both runs must use the same
edit plan. Word hit@5 % as in bench/audit/contrast_audit.analyse (threshold = 5th percentile of the
run's own native word probabilities); the bootstrap resamples clips (natives and edits together),
2,000 replicates, seed 0; two-sided p = 2 min(P(d <= 0), P(d >= 0)).
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys

import numpy as np


def load(d, lang):
    clips, edits = {}, []
    for f in sorted(glob.glob(os.path.join(d, lang, "chunk_250_*.pkl"))):
        r = pickle.load(open(f, "rb"))
        for c in r["clips"]:
            clips[c["id"]] = np.asarray(c["word_prob"], np.float64)
        edits += [e for e in r["edits"] if "error" not in e]
    return clips, edits


def key(e):
    return (e["id"], e["contrast"], e.get("direction"), e.get("index", e.get("src")), e.get("src"), e.get("dst"))


def main():
    before, after = sys.argv[1], sys.argv[2]
    out = []
    for spec in sys.argv[3:]:
        lang, contrast, direction = spec.split(":")
        cb, eb = load(before, lang)
        ca, ea = load(after, lang)
        sel = lambda E: {key(e): e["word_prob"] for e in E if e["contrast"] == contrast and e.get("direction") == direction}
        B, A = sel(eb), sel(ea)
        common = sorted(set(B) & set(A))
        ids = sorted(set(cb) & set(ca))
        by_clip = {}
        for k in common:
            by_clip.setdefault(k[0], []).append(k)
        rng = np.random.default_rng(0)

        def stat(clip_ids):
            nb = np.concatenate([cb[i] for i in clip_ids])
            na = np.concatenate([ca[i] for i in clip_ids])
            ks = [k for i in clip_ids for k in by_clip.get(i, [])]
            if not ks:
                return np.nan, np.nan
            tb, ta = np.percentile(nb, 5), np.percentile(na, 5)
            return (np.mean([B[k] <= tb for k in ks]), np.mean([A[k] <= ta for k in ks]))

        hb, ha = stat(ids)
        d = []
        for _ in range(2000):
            s = rng.choice(ids, len(ids))
            x, y = stat(list(s))
            if np.isfinite(x):
                d.append(y - x)
        d = np.asarray(d)
        p = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))
        out.append({"row": spec, "n_edits_paired": len(common), "n_before": len(B), "n_after": len(A),
                    "word_hit5_before": hb, "word_hit5_after": ha, "delta": ha - hb,
                    "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p": p})
        print(json.dumps(out[-1]))
    if os.environ.get("OUT"):
        json.dump(out, open(os.environ["OUT"], "w"), indent=1)


if __name__ == "__main__":
    main()
