"""Collect the round-20 Japanese length tables into results/tables/round20_ja_length_*.json.

    python -m bench.round20_ja_length.tables

Reads the cache folder ($PRONUNCIADA_CACHE/round20/jalength): probe.json, cv_*.json, dev_e1_w8.json,
grid_g*.json (bench.round20_ja_length.deveval --out), latency_*.json, r18_baseline.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from bench.round20_ja_length.common import OUT

T = Path(__file__).resolve().parents[2] / "results" / "tables"

# Measured on 20 Common Voice ja fit clips (bench/round20_ja_length, frame-rate check): mean relative
# step |x[t+1] - x[t]| / |x[t]| by t mod 8. Small steps inside blocks of 2^k frames and large ones
# at block edges show the stack's effective rate.
FRAME_STEPS = {
    "enc0": [0.902, 0.91, 0.905, 0.888, 0.895, 0.9, 0.909, 0.913],
    "enc1": [0.388, 0.906, 0.395, 0.905, 0.379, 0.911, 0.392, 0.912],
    "enc2": [0.378, 0.501, 0.383, 0.947, 0.368, 0.498, 0.384, 0.941],
    "enc3": [0.314, 0.403, 0.318, 0.801, 0.31, 0.404, 0.321, 1.002],
    "enc4": [0.379, 0.464, 0.388, 0.86, 0.38, 0.471, 0.395, 0.944],
    "enc5": [0.353, 0.755, 0.353, 0.881, 0.352, 0.746, 0.363, 0.911],
    "final": [0.345, 0.733, 0.345, 0.879, 0.344, 0.726, 0.355, 0.948],
}
EFFECTIVE_HZ = {"enc0": 50, "enc1": 25, "enc2": 12.5, "enc3": 6.25, "enc4": 12.5, "enc5": 25, "final": 25}


def main() -> None:
    probe = json.loads((OUT / "probe.json").read_text())
    (T / "round20_ja_length_probe.json").write_text(json.dumps(
        {"frame_steps_by_t_mod_8": FRAME_STEPS, "effective_rate_hz": EFFECTIVE_HZ,
         "output_grid_ms": 20, "linear_probe": probe}, indent=1))
    cv = {}
    for f in sorted(OUT.glob("cv_*.json")):
        d = json.loads(f.read_text())
        cv[f.stem[3:]] = {"cfg": d["cfg"], "oof": d["oof"]}
    dev_pool = json.loads((OUT / "dev_e1_w8.json").read_text())
    knees = json.loads((OUT / "knees_e1_w8.json").read_text())
    (T / "round20_ja_length_cv.json").write_text(json.dumps(
        {"cv_oof_fit_pool": cv, "dev_pool_e1_w8": dev_pool, "native_knees_e1_w8": knees}, indent=1))
    grids = {}
    for f in sorted(OUT.glob("grid_g*.json")):
        d = json.loads(f.read_text())
        grids[f.stem[5:]] = {s: {k: o[k] for k in ("m", "length_dirs", "flags", "guards_vs_main", "guards_ok",
                                                   "paired_vs_main") if k in o}
                             for s, o in d["settings"].items()} | {"lentab": d.get("lentab")}
    (T / "round20_ja_length_dev.json").write_text(json.dumps(grids, indent=1, default=float))
    extra = {}
    for f in sorted(OUT.glob("latency_*.json")) + sorted(OUT.glob("r18_*.json")):
        d = json.loads(f.read_text())
        extra[f.stem] = {k: v for k, v in d.items() if k not in ("old_tree", "new_tree")}   # no local paths
    (T / "round20_ja_length_extra.json").write_text(json.dumps(extra, indent=1, default=float))
    print("written", [p.name for p in T.glob("round20_ja_length_*.json")])


if __name__ == "__main__":
    main()
