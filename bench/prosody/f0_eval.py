"""F0 tracker vs Praat (parselmouth, offline cross-check only).

    .venv/bin/python -m bench.prosody.f0_eval dev      # 50 FLEURS-validation clips (tuning)
    .venv/bin/python -m bench.prosody.f0_eval report   # 50 other clips (aishell3/so762/fleurs train)

Metrics on the 10 ms grid: GPE = share of frames voiced in both with |f - f_ref| > 20 % f_ref;
VDE = share of all frames whose voiced/unvoiced decision differs; FPE = std (cents) of the
deviation on frames without gross error; octave = share of both-voiced frames off by ~2x or ~0.5x.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")

from pronunciada.audio import load_audio  # noqa: E402
from pronunciada.prosody import f0 as F  # noqa: E402

LANGS = ["en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko"]


def praat(wav: np.ndarray, T: int):
    import parselmouth

    snd = parselmouth.Sound(wav.astype(np.float64), 16000)
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=50.0, pitch_ceiling=600.0)
    f = p.selected_array["frequency"]
    ts = p.xs()
    grid = np.arange(T) * 0.01 + 0.005
    idx = np.clip(np.round((grid - ts[0]) / 0.01).astype(int), 0, len(f) - 1)
    out = f[idx].copy()
    out[(grid < ts[0] - 0.006) | (grid > ts[-1] + 0.006)] = 0.0
    return out


def clips(which: str):
    if which == "dev":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        from g2p_eval import fleurs_validation

        for lang in LANGS:
            for k, (uid, _, wav) in enumerate(fleurs_validation(lang)):
                if k >= 5:
                    break
                yield f"{lang}-val-{uid}", wav
    else:
        from bench.data import load

        for r in load("aishell3", split="train")[::600][:10]:
            yield r["id"], load_audio(r["audio"])
        for r in load("speechocean762", split="train")[::250][:10]:
            yield r["id"], load_audio(r["audio"])
        for lang in LANGS:
            for r in load("fleurs", split="train", lang=lang)[::300][:3]:
                yield r["id"], load_audio(r["audio"])


def evaluate(which: str, verbose: bool = True) -> dict:
    tot = dict(both=0, gross=0, frames=0, vde=0, octave=0, ref_v=0, our_v=0)
    cents = []
    per = []
    t_ours = 0.0
    t_praat = 0.0
    secs = 0.0
    for cid, wav in clips(which):
        t = time.perf_counter()
        tr = F.track(wav)
        t_ours += time.perf_counter() - t
        t = time.perf_counter()
        ref = praat(wav, len(tr.f0))
        t_praat += time.perf_counter() - t
        secs += len(wav) / 16000
        rv = ref > 0
        ov = tr.voiced
        both = rv & ov
        ratio = tr.f0[both] / ref[both]
        gross = np.abs(ratio - 1) > 0.2
        octv = (np.abs(ratio - 2) < 0.3) | (np.abs(ratio - 0.5) < 0.08)
        tot["both"] += int(both.sum())
        tot["gross"] += int(gross.sum())
        tot["octave"] += int(octv.sum())
        tot["frames"] += len(ref)
        tot["vde"] += int((rv != ov).sum())
        tot["ref_v"] += int(rv.sum())
        tot["our_v"] += int(ov.sum())
        cents.append(1200 * np.log2(ratio[~gross]))
        per.append((cid, gross.mean() if len(gross) else 0, (rv != ov).mean()))
    c = np.concatenate(cents)
    res = dict(n=len(per), seconds=secs, GPE=tot["gross"] / max(tot["both"], 1),
               VDE=tot["vde"] / tot["frames"], octave=tot["octave"] / max(tot["both"], 1),
               FPE_cents=float(np.std(c)), ref_voiced=tot["ref_v"] / tot["frames"],
               our_voiced=tot["our_v"] / tot["frames"], ms_per_10s_ours=1e4 * t_ours / secs,
               ms_per_10s_praat=1e4 * t_praat / secs)
    if verbose:
        for cid, g, v in sorted(per, key=lambda x: -x[1])[:5]:
            print(f"  worst {cid}: GPE {g:.3f} VDE {v:.3f}")
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
    return res


if __name__ == "__main__":
    evaluate(sys.argv[1] if len(sys.argv) > 1 else "dev")
