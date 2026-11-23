"""Adapter candidates: the packaged per-language adapters stored smaller.

    python -m bench.opt.build_adapters [fp16 int8 lowrank32]

For each variant, two folders under $PRONUNCIADA_CACHE/opt/candidates/adapters-<variant>/:
  eval/<lang>.npz     the weights as the variant would reproduce them, stored as float32, so the
                      current loader (pronunciada.adapter, which casts to float32) runs them as-is
                      through PRONUNCIADA_ADAPTER_DIR;
  compact/<lang>.npz  the compact encoding itself (np.savez_compressed), whose size is the saving.
Variants: fp16 (every weight array as float16); int8 (symmetric per-output-column int8 with a
float32 scale per column); lowrank<r> (every square or 2-D-reshaped weight matrix W [.., n, m]
replaced by its rank-r SVD truncation, factors stored float32; biases and small arrays as they are).
Only arrays with at least 256 values are changed. build.json lists sizes per language.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .common import opt_root

PKG = Path(__file__).resolve().parents[2] / "pronunciada" / "data" / "adapters"
META = ("kind", "floor", "scale", "meta")


def variant(a: np.ndarray, kind: str):
    """(reconstructed float32 array, dict of compact arrays)."""
    if kind == "fp16":
        h = a.astype(np.float16)
        return h.astype(np.float32), {"h": h}
    if kind == "int8":
        m = a.reshape(-1, a.shape[-1])
        s = np.abs(m).max(0) / 127.0
        s[s == 0] = 1.0
        q = np.clip(np.round(m / s), -127, 127).astype(np.int8)
        return (q.astype(np.float32) * s).reshape(a.shape).astype(np.float32), {"q": q, "s": s.astype(np.float32)}
    if kind.startswith("lowrank"):
        r = int(kind[len("lowrank"):])
        m = a.reshape(-1, a.shape[-1]).astype(np.float64)
        if min(m.shape) <= r:
            return a.astype(np.float32), {"w": a.astype(np.float32)}
        u, sv, vt = np.linalg.svd(m, full_matrices=False)
        U, V = (u[:, :r] * sv[:r]).astype(np.float32), vt[:r].astype(np.float32)
        return (U @ V).reshape(a.shape).astype(np.float32), {"u": U, "v": V}
    raise SystemExit(f"unknown variant {kind}")


def build(kind: str) -> dict:
    root = opt_root() / "candidates" / f"adapters-{kind}"
    (root / "eval").mkdir(parents=True, exist_ok=True)
    (root / "compact").mkdir(parents=True, exist_ok=True)
    info = {"variant": kind, "langs": {}}
    for p in sorted(PKG.glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            P = {k: z[k] for k in z.files}
        ev, cp, maxerr = {}, {}, 0.0
        for k, v in P.items():
            if k in META or v.dtype.kind not in "f" or v.size < 256:
                ev[k] = v
                cp[k] = v
                continue
            rec, comp = variant(v.astype(np.float32), kind)
            ev[k] = rec
            maxerr = max(maxerr, float(np.abs(rec - v).max()))
            for ck, cv in comp.items():
                cp[f"{k}__{ck}"] = cv
        np.savez(root / "eval" / p.name, **ev)
        np.savez_compressed(root / "compact" / p.name, **cp)
        info["langs"][p.stem] = {"packaged_bytes": p.stat().st_size,
                                 "compact_bytes": (root / "compact" / p.name).stat().st_size,
                                 "max_abs_weight_change": maxerr, "kind": str(P["kind"])}
    L = info["langs"].values()
    info["packaged_total"] = sum(x["packaged_bytes"] for x in L)
    info["compact_total"] = sum(x["compact_bytes"] for x in L)
    (root / "build.json").write_text(json.dumps(info, indent=1))
    print(kind, info["packaged_total"], "->", info["compact_total"], flush=True)
    return info


if __name__ == "__main__":
    for k in sys.argv[1:] or ["fp16", "int8", "lowrank32"]:
        build(k)
