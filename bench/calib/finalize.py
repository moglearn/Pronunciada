"""Per-language bias of the universal sound model + the shipped calibrator config.

    python -m bench.calib.finalize [--per-lang 400] [--workers 5]

Bias: one logit offset per language, added to both heads of the universal sound model, such that
the median native s=0.5 logit on FLEURS TRAIN (correct references, cached posteriors, the same
runtime feature code) equals the English median (English offset = 0 by definition). Reported:
native acceptance (share of sounds with P >= 0.5 at strictness 0 / 0.5 / 1) before and after, on
the same FLEURS-train utterances (the fitting data; the held-out view is results/languages.md).

Config (pronunciada/data/calib/config.json): universal_sound_model, per-language sound model
(English: sound_B if it beat A on speaker-grouped CV, see calib_sound.json), bias per language,
and the size of every shipped calibrator file.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from bench.calib.common import load_json, save_json, ship_dir


def _utt(args):
    rec, lang = args
    from bench.pipeline.common import build_ref, load_lp, near_silent
    from pronunciada.adapter import apply
    from pronunciada.align import score_reference
    from pronunciada.calibrate import load_level
    from pronunciada.features import phone_table
    from pronunciada.variants import NativeStats

    ref = build_ref(lang, rec["text"])
    lp = load_lp("fleurs", rec)
    if ref is None or lp is None or near_silent(lp, rec["dur"]):
        return None
    lpa = apply(lp, lang)
    t = phone_table(lpa, ref, score_reference(lpa, ref), NativeStats.load(lang), float(rec["dur"]))
    o = load_level("sound_U").logits(t)
    return o["z_len"], o["z_str"]


def main():
    from bench.pipeline.common import LANGS, pmap, records
    from pronunciada.calibrate import blend_logit

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=400)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    out = {"per_lang": a.per_lang, "by_lang": {}}
    Z = {}
    for lang in LANGS:
        recs = records("fleurs", split="train", lang=lang)
        rng = np.random.default_rng(0)
        recs = [recs[i] for i in sorted(rng.choice(len(recs), min(a.per_lang, len(recs)), replace=False))]
        res = [r for r in pmap(_utt, [(r, lang) for r in recs], workers=a.workers, chunksize=4) if r is not None]
        zl = np.concatenate([r[0] for r in res])
        zs = np.concatenate([r[1] for r in res])
        Z[lang] = (zl, zs, len(res))
        print(lang, len(res), "utts", len(zl), "sounds", flush=True)
    med = {lang: float(np.median(blend_logit(zl, zs, 0.5))) for lang, (zl, zs, _) in Z.items()}
    bias = {lang: med["en"] - m for lang, m in med.items()}
    for lang, (zl, zs, n) in Z.items():
        b = bias[lang]
        acc = {}
        for tag, off in (("before", 0.0), ("after", b)):
            acc[tag] = {str(s): float(np.mean(blend_logit(zl + off, zs + off, s) >= 0)) for s in (0.0, 0.5, 1.0)}
        out["by_lang"][lang] = {"utts": n, "sounds": int(len(zl)), "median_logit_s05": med[lang],
                                "bias": b, "native_acceptance": acc}
        print(lang, f"bias {b:+.3f}", acc, flush=True)
    sound = load_json("calib_sound.json")
    wide = load_json("calib_wide.json")
    en_model = "sound_B" if sound.get("english_choice") == "B" else "sound_U"
    cfg = {"universal_sound_model": "sound_U", "sound_model": {"en": en_model},
           "bias": {k: round(v, 6) for k, v in bias.items()},
           "universal_trained_on": wide.get("chosen") if wide else None,
           "files_kb": {}}
    d = ship_dir()
    for p in sorted(d.glob("*.npz")):
        cfg["files_kb"][p.name] = round(p.stat().st_size / 1024, 1)
    shipped = {"sound_U.npz", "word.npz", "sentence.npz", "phone_dist.npz", f"{en_model}.npz"}
    cfg["shipped_total_kb"] = round(sum(v for k, v in cfg["files_kb"].items() if k in shipped), 1)
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    out["config"] = cfg
    save_json("calib_bias.json", out)
    print(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
