"""Three-way look at the resampled-audio tables: committed main table (11-17 package), main c9773cc
re-run (bench/round19_merge/main_baseline.sh) and the round 19 frozen run. Prints every numeric leaf
that differs, with its path, for cv_real.json, ompal.json and third_party_lingolix.json.

    python bench/round19_merge/drift.py <committed dir> <main re-run dir> <round 19 dir> [--keys substr,...]
"""
from __future__ import annotations

import json
import sys


def flat(x, p="", o=None):
    o = {} if o is None else o
    if isinstance(x, dict):
        for k, v in x.items():
            flat(v, f"{p}.{k}" if p else str(k), o)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            flat(v, f"{p}[{i}]", o)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        o[p] = float(x)
    return o


def main():
    a, b, c = sys.argv[1:4]
    keys = sys.argv[sys.argv.index("--keys") + 1].split(",") if "--keys" in sys.argv else []
    for name in ("cv_real.json", "ompal.json", "third_party_lingolix.json"):
        A, B, C = (flat(json.load(open(f"{d}/{name}"))) for d in (a, b, c))
        n_ab = sum(1 for k in A if k in B and abs(A[k] - B[k]) > 1e-9)
        n_bc = sum(1 for k in B if k in C and abs(B[k] - C[k]) > 1e-9)
        print(f"== {name}: values {len(B)}; committed vs main re-run differ {n_ab}; main re-run vs round 19 differ {n_bc}")
        for k in sorted(B):
            if keys and not any(s in k for s in keys):
                continue
            if k in A and k in C and (abs(A[k] - B[k]) > 1e-9 or abs(B[k] - C[k]) > 1e-9):
                print(f"  {k}: committed {A[k]:.4f}  main {B[k]:.4f}  round19 {C[k]:.4f}")


if __name__ == "__main__":
    main()
