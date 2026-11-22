"""Round 20: the per-word factors the round-19 rollout rows did not keep, for offline what-ifs.

    python -m bench.round20_mandarin.collect --lang zh --set dev [--part 0/4] [--workers 4]

Scores the same items as bench.round19_rollout.collect (sets cvpool / dev / devcor / val, never a
held-out set) with every identity switch off, and keeps per reference word: the word P, the sounds,
tone / stress, contrast, length and presence factors, and the per-syllable tone / stress
probabilities. Rows: $PRONUNCIADA_CACHE/round20/mandarin/<lang>/extra/<set>/<chunk>.pkl, keyed by the
round-19 item id, so they join the stored identity evidence (ev_<set>.pkl) one to one.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np

from bench.round19_rollout.sources import CACHE

CHUNK = 40
OUT = CACHE / "round20" / "mandarin"


def out_dir(lang: str, name: str):
    return OUT / lang / "extra" / name


def _nan(x):
    return np.nan if x is None else float(x)


def force_insertion() -> None:
    """Make the aligner compute the extra-sound evidence (llr_ins) for every language. It adds an output
    only; nothing else in the evidence changes."""
    import pronunciada.align as AL

    if getattr(AL.score_reference, "_r20", False):
        return
    orig = AL.score_reference

    def sr(*a, **k):
        k["insertion"] = True
        return orig(*a, **k)

    sr._r20 = True
    AL.score_reference = sr


def extra_row(res) -> dict:
    from pronunciada.identity import word_ins

    ws = res.words
    c = res._pronunciada  # noqa: SLF001
    li = getattr(c["ev"], "llr_ins", None)
    return {
        "ins": np.array([np.nan if (x := word_ins(li, *w.phones)) is None else x for w in c["ref"].words], np.float64),
        "prob": np.array([w.prob for w in ws], np.float64),
        "sounds": np.array([w.sounds_prob for w in ws], np.float64),
        "present": np.array([_nan(w.present_prob) for w in ws], np.float64),
        "tone": np.array([_nan(w.tone_prob) for w in ws], np.float64),
        "stress": np.array([_nan(w.stress_prob) for w in ws], np.float64),
        "contrast": np.array([_nan(w.contrast_prob) for w in ws], np.float64),
        "length": np.array([_nan(w.length_prob) for w in ws], np.float64),
        "syl_tone": [[_nan(s.tone_prob) for s in w.syllables] for w in ws],
        "syl_stress": [[_nan(s.stress_prob) for s in w.syllables] for w in ws],
    }


def run(lang: str, name: str, part: str | None) -> None:
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    os.environ["PRONUNCIADA_IDENTITY"] = "0"
    force_insertion()
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.common import stable_seed
    from bench.pipeline.evalctl import inventory
    from bench.round19_rollout.collect import jobs
    from pronunciada import Scorer

    k_part, n_part = (int(x) for x in part.split("/")) if part else (0, 1)
    out = out_dir(lang, name)
    out.mkdir(parents=True, exist_ok=True)
    js = jobs(lang, name)
    sc = Scorer(lang, threads=2)
    inv = inventory(lang)
    t0 = time.time()
    for k in range(0, len(js), CHUNK):
        if (k // CHUNK) % n_part != k_part:
            continue
        p = out / f"{k // CHUNK:05d}.pkl"
        if p.exists():
            continue
        rows = {}
        for iid, audio, text, _meta in js[k:k + CHUNK]:
            try:
                if isinstance(text, tuple):
                    _, tgt, sid, kind, tag = text
                    ref = sc.g2p(tgt)
                    try:
                        cor = corrupt(ref, [kind], inv, n_errors=2, seed=stable_seed(sid, kind, tag), strict=False)
                    except NotApplicable:
                        continue
                    if not cor.errors:
                        continue
                    text = cor.ref
                rows[iid] = extra_row(sc.score(audio, text))
            except Exception as e:  # noqa: BLE001
                print("skip", iid, type(e).__name__, e, flush=True)
        p.write_bytes(pickle.dumps(rows))
        print(lang, name, f"part {k_part}/{n_part}", k + CHUNK, len(js), f"{time.time() - t0:.0f}s", flush=True)


def load(lang: str, name: str) -> dict:
    out = {}
    for p in sorted(out_dir(lang, name).glob("*.pkl")):
        out.update(pickle.loads(p.read_bytes()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--part")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    if a.workers > 1:
        import subprocess
        import sys

        for s in a.set.split(","):
            ps = [subprocess.Popen([sys.executable, "-m", "bench.round20_mandarin.collect", "--lang", a.lang, "--set", s,
                                    "--part", f"{k}/{a.workers}"]) for k in range(a.workers)]
            if any(p.wait() for p in ps):
                raise SystemExit("worker failed")
        return
    for s in a.set.split(","):
        run(a.lang, s, a.part)


if __name__ == "__main__":
    main()
