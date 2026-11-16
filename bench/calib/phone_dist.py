"""Precompute the articulatory distance between every pair of ZIPA speech tokens for the runtime.

    python -m bench.calib.phone_dist

pronunciada/features.py needs `comp_dist` = panphon distance between the canonical token and its
best competitor (both single tokens). panphon is a bench-only dependency, so the full
token x token table (bench.corrupt.PhoneDistance, NaN where panphon cannot parse a token, e.g.
modifiers) is written to pronunciada/data/calib/phone_dist.npz and read at runtime with numpy.
"""
from __future__ import annotations

import math

import numpy as np


def main():
    from bench.corrupt import PhoneDistance, phone_distance
    from bench.pipeline.common import repo_root
    from pronunciada.text.ipa import load_tokens

    toks = load_tokens()
    V = len(toks)
    D = np.full((V, V), np.nan, dtype=np.float32)
    pd_ = phone_distance()
    for i in range(4, V):
        for j in range(4, V):
            if j < i:
                D[i, j] = D[j, i]
                continue
            try:
                d = pd_(toks[i], toks[j])
            except Exception:  # noqa: BLE001
                d = PhoneDistance.UNKNOWN
            D[i, j] = math.nan if d >= PhoneDistance.UNKNOWN else float(d)
    pd_.save()
    out = repo_root() / "pronunciada" / "data" / "calib" / "phone_dist.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, tokens=np.array(toks), dist=D)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB), finite {np.isfinite(D).sum()} of {V * V}")


if __name__ == "__main__":
    main()
