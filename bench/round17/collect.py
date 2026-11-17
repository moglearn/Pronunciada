"""Round 17 inputs: head-based contrast evidence for ko / ru / zh / ja (FLEURS validation = dev,
FLEURS test = held-out), through the public Scorer.

    python -m bench.round17.collect --split validation,test [--lang ko,ru,zh,ja] [--workers 3]

Per language, the contrast audit's clip choice and edit plan (bench/audit/contrast_audit:
choose / make_plan; FLEURS test uses the same rules on 250 test clips) plus two planted
substitutions per clip (bench.corrupt phone_sub_near / phone_sub_far, the planted-error guard).
Stored per native sound and per edited sound: symbol, alternatives, the SHOWN sound log-odds
(s_zl / s_zs, after the round-16 caps), the length channel's probability, the pooled ZIPA vector
(float16); per word the word probability. Chunks of 25 clips are cached under
$PRONUNCIADA_CACHE/bench/round17/<split>/<lang>/. Set PRONUNCIADA_HEAD_CAPS=0 for "before" runs
once the round-17 caps exist (tag "base").
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import zlib
from pathlib import Path

import numpy as np

CONTRASTS = {"ko": ("aspiration",), "ru": ("hard_soft",), "zh": ("aspiration",), "ja": ("vowel_length", "gemination")}
CHUNK = 25


def root(split, lang, tag):
    base = os.environ.get("PRONUNCIADA_CACHE")
    r = Path(base).expanduser() if base else Path.home() / ".cache" / "pronunciada"
    d = r / "bench" / "round17" / tag / split / lang
    d.mkdir(parents=True, exist_ok=True)
    return d


def clips(lang, split):
    from bench.audit.contrast_audit import choose
    from bench.pipeline.common import records

    if split == "validation":
        return choose(lang, 250)
    recs = records("fleurs", split="test", lang=lang)
    rng = np.random.default_rng(0)
    return [recs[i] for i in sorted(rng.choice(len(recs), min(250, len(recs)), replace=False))]


def plan_for(lang, split, recs, g2p):
    from bench.audit.contrast_audit import cache_dir, make_plan

    p = cache_dir(lang) / "plan_250.json" if split == "validation" else root(split, lang, "plan") / "plan.json"
    if p.exists():
        return json.loads(p.read_text())["plan"]
    plan = make_plan(lang, recs, g2p)
    p.write_text(json.dumps(plan, ensure_ascii=False))
    return plan["plan"]


def _rows(res, idx):
    from pronunciada.sound_emb import phone_vectors

    c = res._pronunciada  # noqa: SLF001
    ref = c["ref"]
    V = phone_vectors(c["emb"], c["ev"], len(ref.phones))
    lp = (c.get("length") or {}).get("prob") or [None] * len(ref.phones)
    probs = [p.prob for w in res.words for p in w.phones]
    return {"ipa": [ref.phones[i].ipa for i in idx], "alts": [[a for a, _ in ref.phones[i].alts] for i in idx],
            "zl": np.array([c["s_zl"][i] for i in idx], np.float32), "zs": np.array([c["s_zs"][i] for i in idx], np.float32),
            "prob": np.array([probs[i] for i in idx], np.float32),
            "length": [lp[i] for i in idx], "word": [ref.phones[i].word for i in idx],
            "vec": V[list(idx)].astype(np.float16)}


def run_chunk(args):
    lang, split, k0, tag = args
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    out_p = root(split, lang, tag) / f"chunk_{k0:04d}.pkl"
    if out_p.exists():
        return str(out_p)
    import warnings

    warnings.simplefilter("ignore")
    from bench.audit.contrasts import Site, apply
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.common import iter_parquet_audio, stable_seed
    from bench.pipeline.evalctl import inventory
    from pronunciada.audio import load_audio
    from pronunciada.scorer import Scorer
    from pronunciada.text import get_g2p

    g2p = get_g2p(lang)
    recs = clips(lang, split)
    plan = plan_for(lang, split, recs, g2p)
    chunk = recs[k0:k0 + CHUNK]
    sc = Scorer(lang, threads=1)
    inv = inventory(lang)
    if str(chunk[0]["audio"]).startswith("parquet:"):
        by = {int(r["audio"].split(":")[2]): r for r in chunk}
        path = chunk[0]["audio"].split(":")[1]
        it = ((by[i], wav) for i, wav in iter_parquet_audio(path, set(by)))
    else:
        it = ((r, load_audio(r["audio"])) for r in chunk)
    rows = {"clips": []}
    for rec, wav in it:
        try:
            ref = g2p(rec["text"])
            res = sc.score((wav, 16000), ref)
        except Exception:  # noqa: BLE001
            continue
        n = len(ref.phones)
        clip = {"id": rec["id"], "native": _rows(res, range(n)),
                "word_prob": [w.prob for w in res.words], "edits": [], "planted": []}
        for sd in plan.get(rec["id"], []):
            if sd["contrast"] not in CONTRASTS[lang]:
                continue
            site = Site(**sd)
            rng = np.random.default_rng(zlib.crc32(f"{rec['id']}/{sd}".encode()))
            try:
                e = apply(ref, site, rng)
                r2 = sc.score((wav, 16000), e.ref)
            except Exception:  # noqa: BLE001
                continue
            d = _rows(r2, e.flag)
            d.update(contrast=site.contrast, direction=site.direction, src=site.src, dst=site.dst,
                     index=site.index, alt=bool(e.partner_in_alts), word_prob=float(r2.words[e.word].prob),
                     nat_word_prob=float(res.words[e.word].prob))
            clip["edits"].append(d)
        for kind in ("phone_sub_near", "phone_sub_far"):
            try:
                cor = corrupt(ref, [kind], inv, n_errors=1, seed=stable_seed(rec["id"], kind, "round17"), strict=False)
            except NotApplicable:
                continue
            if not cor.errors:
                continue
            r2 = sc.score((wav, 16000), cor.ref)
            idx = [i for i, w in enumerate(cor.phone_wrong) if w]
            d = _rows(r2, idx)
            d["kind"] = kind
            clip["planted"].append(d)
        rows["clips"].append(clip)
    tmp = out_p.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(rows))
    os.replace(tmp, out_p)
    return str(out_p)


def load(lang, split, tag="base"):
    d = root(split, lang, tag)
    out = []
    for p in sorted(d.glob("chunk_*.pkl")):
        out += pickle.loads(p.read_bytes())["clips"]
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--lang", default="ko,ru,zh,ja")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()
    from pronunciada.text import get_g2p

    jobs = []
    for split in a.split.split(","):
        for lang in a.lang.split(","):
            recs = clips(lang, split)
            plan_for(lang, split, recs, get_g2p(lang))          # plans first (no race between workers)
            n = len(recs)
            jobs += [(lang, split, k0, a.tag) for k0 in range(0, n, CHUNK)]
    with ProcessPoolExecutor(min(a.workers, 3)) as ex:
        for k, p in enumerate(ex.map(run_chunk, jobs)):
            print(k + 1, len(jobs), p, flush=True)


if __name__ == "__main__":
    main()
