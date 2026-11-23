"""Write the golden test vectors of the dependency-free reference (pronunciada/reference).

    python scripts/make_reference_golden.py [--out tests/data/reference_golden] [--per-lang 2]

Inputs come from the committed practice list (bench/opt/clips.json, FLEURS train; the shortest
`--per-lang` clips of each language): the fp32 backbone, the language's adapter and text
front-end give the adapted log-posteriors and the target sounds. The posteriors are stored
clipped below at -24 nat and quantised to steps of 1/256 nat (int16, so every stored value is
exactly representable in float32 and float64); all
expected outputs are computed on the stored values by the reference;
tests/test_reference_module.py checks that production (pronunciada.align, pronunciada.calibrate)
gives the same numbers on them (bounds stated there). The calibration input rows are built by
production (pronunciada.features) from the stored posteriors. Needs the fp32 backbone in the
model cache and the FLEURS manifest (bench.data).

File format: one gzip-compressed JSON file per language, <lang>.json.gz:
  {"format": 1, "lang", "tokens": [127 symbols], "clips": [clip, ...]}
clip = {"id", "text", "T", "lp_q": base64 of T x 127 little-endian int16 (row-major),
        "lp_scale": 256, "phones": [{"ipa", "alts": [[ipa, log_prior], ...]}],
        "word_phones": [[start, end], ...],
        "cases": [{"name", "frames": [t0, t1], "phones": [p0, p1], "insertion", "wild_slots",
                   "exact", "expected": evidence}],
        "sound_model": {"file", "rows": X, "expected": [{head: value}]},
        "word_model": {"file", "rows": X, "expected": [{head: value}]},
        "omission": {"llr_del": [[...] per word], "expected": [P per word]}}
evidence = output of pronunciada.reference.align.score_phones. Floats that are not finite are
written as the strings "NaN", "Infinity", "-Infinity".
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCALE = 256
FLOOR = -24.0      # stored log-posteriors are clipped below at -24 nat (posterior 3.8e-11): halves the files
LANGS = ("en", "de", "es", "fr", "it", "pt", "ru", "ja", "ko", "zh")


def enc(x):
    """JSON-safe value: non-finite floats as strings, tuples as lists."""
    if isinstance(x, float):
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "Infinity" if x > 0 else "-Infinity"
        return x
    if isinstance(x, (list, tuple)):
        return [enc(v) for v in x]
    if isinstance(x, dict):
        return {k: enc(v) for k, v in x.items()}
    return x


def quantise(lp: np.ndarray) -> np.ndarray:
    q = np.round(np.clip(np.asarray(lp, np.float64), FLOOR, 32767 / SCALE) * SCALE)
    return q.astype("<i2")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "tests" / "data" / "reference_golden"))
    ap.add_argument("--per-lang", type=int, default=2)
    ap.add_argument("--langs", default=",".join(LANGS))
    a = ap.parse_args(argv)

    from bench.opt.common import practice_clips
    from pronunciada import calibrate as C
    from pronunciada import features as F
    from pronunciada.acoustic import AcousticModel
    from pronunciada.adapter import Adapter
    from pronunciada.audio import load_audio
    from pronunciada.reference import align as RA
    from pronunciada.reference import calibrate as RC
    from pronunciada.reference.tokens import load_tokens
    from pronunciada.text import get_g2p
    from pronunciada.variants import NativeStats

    os.makedirs(a.out, exist_ok=True)
    toks = load_tokens()
    am = AcousticModel(precision="fp32", threads=2)
    cfg = C.config()
    uname = cfg["universal_sound_model"]
    d = C.data_dir()
    ru, rw = RC.LevelModel(str(d / f"{uname}.npz")), RC.LevelModel(str(d / "word.npz"))
    uni, word = C.load_level(uname), C.load_level("word")
    om = json.loads((d / "omission.json").read_text())
    for lang in a.langs.split(","):
        g2p, ad, native = get_g2p(lang), Adapter.load(lang), NativeStats.load(lang)
        clips = sorted(practice_clips([lang]), key=lambda c: (c["dur"], c["id"]))[:a.per_lang]
        out = []
        for c in clips:
            wav = load_audio(c["audio"])
            lp = am.posteriors(wav)
            lpa = ad(lp) if ad is not None else lp
            q = quantise(lpa)
            lpq = q.astype(np.float64) / SCALE
            ref = g2p(c["text"])
            phones = [{"ipa": p.ipa, "alts": [[x, float(w)] for x, w in p.alts]} for p in ref.phones]
            T, n = len(lpq), len(phones)
            lpl = lpq.tolist()
            w0 = list(range(*ref.words[0].phones))
            cases = [dict(name="plain", frames=[0, T], phones=[0, n], insertion=False, wild_slots=[], exact=False),
                     dict(name="insertion", frames=[0, T], phones=[0, n], insertion=True, wild_slots=[], exact=False),
                     dict(name="wild_word0", frames=[0, T], phones=[0, n], insertion=False, wild_slots=w0, exact=False),
                     dict(name="exact_crop", frames=[0, min(T, 80)], phones=[0, min(n, 5)], insertion=False,
                          wild_slots=[], exact=True),
                     dict(name="fast_crop_vs_exact", frames=[0, min(T, 80)], phones=[0, min(n, 5)],
                          insertion=True, wild_slots=[], exact=False),
                     dict(name="too_short", frames=[0, 6], phones=[0, min(n, 12)], insertion=True,
                          wild_slots=[], exact=False)]
            for cs in cases:
                t0, t1 = cs["frames"]
                p0, p1 = cs["phones"]
                cs["expected"] = RA.score_phones(lpl[t0:t1], phones[p0:p1], toks, wild_slots=cs["wild_slots"],
                                                 insertion=cs["insertion"], exact=cs["exact"])
            from pronunciada.align import score_reference

            ev = score_reference(lpq.astype(np.float32), ref)
            tab = F.phone_table(lpq.astype(np.float32), ref, ev, native, len(wav) / 16000)
            X = uni.matrix(tab)
            rows = [[float(v) for v in r] for r in X]
            pl = uni.logits(X=X)
            W = F.word_table(tab, pl["z_len"], pl["z_str"], pl["reg"])
            XW = np.column_stack([np.asarray(W[k], dtype=np.float64) for k in word.num])
            wrows = [[float(v) for v in r] for r in XW]
            llr = [[float(v) for v in tab["llr_del"][w.phones[0]:w.phones[1]]] for w in ref.words]
            out.append({
                "id": c["id"], "text": c["text"], "T": T, "lp_scale": SCALE,
                "lp_q": base64.b64encode(q.tobytes()).decode("ascii"),
                "phones": phones, "word_phones": [list(w.phones) for w in ref.words], "cases": cases,
                "sound_model": {"file": f"pronunciada/data/calib/{uname}.npz", "rows": rows,
                                "expected": [ru.predict(r) | ru.logits(r) for r in rows]},
                "word_model": {"file": "pronunciada/data/calib/word.npz", "rows": wrows,
                               "expected": [rw.logits(r) for r in wrows]},
                "omission": {"file": "pronunciada/data/calib/omission.json", "llr_del": llr,
                             "expected": [RC.present_prob(om, x) for x in llr]},
            })
        doc = {"format": 1, "lang": lang, "tokens": toks, "clips": out}
        p = Path(a.out) / f"{lang}.json.gz"
        with gzip.GzipFile(p, "wb", mtime=0) as f:
            f.write(json.dumps(enc(doc), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        print(lang, p, p.stat().st_size, "bytes", flush=True)


if __name__ == "__main__":
    main()
