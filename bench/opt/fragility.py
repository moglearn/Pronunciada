"""Numerical fragility of a backbone: how far the log-posteriors move when the fbank features move a
little.

    python -m bench.opt.fragility [--langs en,de,ja,zh] [--n 5] [--eps 1e-4] [--backbones int8,fp32,<file>]

Clips: the committed practice clips (FLEURS train). Per clip, features x (Scorer path: decode,
fbank), then the backbone on x and on x + d for two perturbations of size eps: Gaussian noise
(sd eps, seed 0) and a constant shift (+eps). Reported per backbone and perturbation: max and
mean |change| of the log-posteriors (all finite entries), the 99.9th percentile, and the share of frames whose argmax token changes.
Single process, 2 threads.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .common import default_backbone, practice_clips


def run(backbones: dict, langs, n: int, eps: float, threads: int = 2) -> dict:
    from pronunciada.acoustic import AcousticModel
    from pronunciada.audio import load_audio

    clips = [c for l in langs for c in practice_clips([l], n)]
    feats = []
    for c in clips:
        feats.append(load_audio(c["audio"]))
    out = {"kind": "opt_fragility", "date": time.strftime("%Y-%m-%d %H:%M"), "eps": eps, "n_clips": len(clips),
           "langs": list(langs), "results": {}}
    for name, path in backbones.items():
        import os

        os.environ["PRONUNCIADA_BACKBONE"] = str(path)
        am = AcousticModel(precision="fp32", threads=threads)
        rng = np.random.default_rng(0)
        R = {}
        for kind in ("gaussian", "shift"):
            mx, tot, cnt, flips, frames, allv = 0.0, 0.0, 0, 0, 0, []
            for wav in feats:
                x = am._features(wav)
                d = rng.normal(0, eps, x.shape).astype(np.float32) if kind == "gaussian" else np.float32(eps)
                a = am.posteriors_from_features(x)
                b = am.posteriors_from_features((x + d).astype(np.float32))
                fin = np.isfinite(a) & np.isfinite(b)
                diff = np.abs(a - b)[fin]
                mx = max(mx, float(diff.max()))
                tot += float(diff.sum())
                cnt += diff.size
                allv.append(diff)
                flips += int((a.argmax(1) != b.argmax(1)).sum())
                frames += len(a)
            allv = np.concatenate(allv)
            R[kind] = {"max_abs_dlogp": mx, "mean_abs_dlogp": tot / cnt, "p999_abs_dlogp": float(np.quantile(allv, 0.999)),
                       "argmax_changed_share": flips / frames, "frames": frames}
            print(name, kind, json.dumps(R[kind]), flush=True)
        out["results"][name] = R
        del am
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="en,de,ja,zh")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--backbones", default="fp32,int8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    bb = {}
    for b in a.backbones.split(","):
        bb[b] = default_backbone(b) if b in ("fp32", "int8") else Path(b)
    r = run(bb, a.langs.split(","), a.n, a.eps)
    Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
