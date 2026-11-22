"""Round 19 rollout: score item sets through the public Scorer and keep what the sound-identity head
and its word rule need, so every later choice is an offline what-if on the stored rows.

    python -m bench.round19_rollout.collect --lang fr --set dev [--part 0/4]

Sets
  fltrain   FLEURS TRAIN natives (head training)
  cvtrain   Common Voice "train" speakers of bench.round19_rollout.sources (head training)
  cvpool    Common Voice "pool" speakers (held out of the head; operating points only)
  dev       the planted-change dev set (bench.round19_rollout.devset), Lingolix-style word list kept
  devcor    Common Voice dev controls with bench.corrupt controlled errors (one kind per clip)
  val       FLEURS VALIDATION guards: the bench.calib.bias_grid swap set (orig / swap_near / swap_far)
            and the fit_prosody protocol on 200 clips (clean, 2 planted stress errors when the language
            has stress, 5 sound-error kinds)
  test      HELD OUT (final run only): fleurs-swap TEST and the same protocol on all FLEURS TEST clips;
            every item is scored twice in the same process, identity off and on (shared posteriors)
Per scored item: sounds (symbol, front-end alternatives), word index of every sound, pooled
enc1|enc3|enc5 vectors (float16, not for "test"), word P (and P before the presence factor), labels.
Rows: $PRONUNCIADA_CACHE/round19/rollout/<lang>/rows/<set>/<part>_<k>.pkl (resumable chunks).
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np

from bench.round19_rollout.sources import cv_rows, work

CHUNK = 40
STRESS_LANGS = {"en", "es", "it", "pt", "ru", "de"}


def jobs(lang: str, name: str) -> list[tuple]:
    """[(id, audio, text | ("cor", text, seed_id, kind, seed_tag), meta)]."""
    if name == "fltrain":
        from bench.data import load

        return [(r["id"], r["audio"], r["text"], {"src": "fleurs-train"})
                for r in sorted(load("fleurs", "train", lang), key=lambda r: r["id"])]
    if name in ("cvtrain", "cvpool"):
        return [(f"cv-{c['stem']}", c["audio"], c["sentence"], {"src": name, "speaker": c["spk"]})
                for c in cv_rows(lang, name[2:])]
    if name == "dev":
        from bench.round19_rollout.devset import manifest

        return [(it["id"], it["audio"], it["target"], {"src": it["source"], "fold": it["fold"], "label": it["label"],
                                                       "speaker": it["speaker"]}) for it in manifest(lang)]
    if name == "devcor":
        from bench.round19.collect import CORRUPT_KINDS
        from bench.round19_rollout.devset import manifest

        items = [it for it in manifest(lang) if it["source"] == "cv" and not it["label"]]
        return [(it["id"] + "__cor", it["audio"], ("cor", it["target"], it["id"], CORRUPT_KINDS[j % len(CORRUPT_KINDS)], "r19"),
                 {"src": "cv", "fold": it["fold"], "speaker": it["speaker"]}) for j, it in enumerate(items)]
    if name == "val":
        return _guard_jobs(lang, test=False)
    if name == "test":
        return _guard_jobs(lang, test=True)
    raise SystemExit(name)


def _guard_jobs(lang: str, test: bool) -> list[tuple]:
    from bench.calib.fit_prosody import SOUND_KINDS

    kinds = (("stress",) if lang in STRESS_LANGS else ()) + tuple(SOUND_KINDS)
    out = []
    if test:
        from bench.data import load

        for r in load("fleurs-swap", "test", lang):
            lab = r["labels"]
            out.append((r["id"], r["audio"], r["text"], {"kind": "swap", "variant": lab["variant"],
                                                         "k": lab.get("word_index"), "units": lab["words"],
                                                         "utt": r["id"].split("__")[0]}))
        recs = load("fleurs", "test", lang)
    else:
        from bench.calib.bias_grid import dev_swap
        from bench.round19_rollout.devset import materialise
        from bench.round19_rollout.sources import fleurs_val

        allv = fleurs_val(lang)
        rng = np.random.default_rng(0)
        pros = {allv[i]["id"] for i in sorted(rng.choice(len(allv), min(200, len(allv)), replace=False))}
        sw = dev_swap(lang)
        need = pros | {r["src"] for r in sw}
        recs = [r for r in allv if r["id"] in need]
        materialise(lang, recs)
        audio = {r["id"]: r["audio"] for r in recs}
        for r in sw:
            if r["src"] not in audio:
                continue
            lab = r["labels"]
            out.append((r["id"], audio[r["src"]], r["text"], {"kind": "swap", "variant": lab["variant"],
                                                              "k": lab.get("word_index"), "units": lab["words"],
                                                              "utt": r["src"]}))
        recs = [r for r in recs if r["id"] in pros]
    for rec in recs:
        out.append((rec["id"] + "__clean", rec["audio"], rec["text"], {"kind": "pros", "ref": "clean", "utt": rec["id"]}))
        for k in kinds:
            out.append((rec["id"] + "__" + k, rec["audio"], ("cor", rec["text"], rec["id"], k, "prosody-v1"),
                        {"kind": "pros", "ref": "stress" if k == "stress" else "sound", "utt": rec["id"]}))
    return out


def _row(res, keep_v: bool, bench_words: bool):
    from pronunciada.sound_emb import phone_vectors

    c = res._pronunciada  # noqa: SLF001
    ref = c["ref"]
    row = {"text": res.text, "ph": [(p.ipa, [(a, round(float(lp), 3)) for a, lp in p.alts]) for p in ref.phones],
           "word": np.array([p.word for p in ref.phones], np.int32),
           "spans": [tuple(w.char_span) for w in ref.words],
           "prob": np.array([w.prob for w in res.words], np.float32),
           "present": np.array([np.nan if w.present_prob is None else w.present_prob for w in res.words], np.float32),
           "sounds": np.array([w.sounds_prob for w in res.words], np.float32)}
    if keep_v:
        if not c.get("emb"):
            raise RuntimeError("no activations: the scorer must run with embeddings")
        row["V"] = phone_vectors(c["emb"], c["ev"], len(ref.phones)).astype(np.float16)
    if bench_words:
        from bench.third_party.lingolix import reduce_azure_shape
        from pronunciada.compat.azure import to_azure

        raw = to_azure(res, granularity="Phoneme", enable_miscue=True, phoneme_alphabet="IPA",
                       layout="rest", grading_system="HundredMark")
        row["words"] = [{"text": w["text"], "score": w["score"], "error_type": w["error_type"]}
                        for w in reduce_azure_shape(raw)]
    return row


def run(lang: str, name: str, part: str | None) -> None:
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.common import stable_seed
    from bench.pipeline.evalctl import inventory
    from pronunciada import Scorer

    k_part, n_part = (int(x) for x in part.split("/")) if part else (0, 1)
    out = work(lang) / "rows" / name
    out.mkdir(parents=True, exist_ok=True)
    js = jobs(lang, name)
    sc = Scorer(lang, threads=2)
    inv = inventory(lang)
    test = name == "test"
    t0, n = time.time(), 0
    for k in range(0, len(js), CHUNK):
        if (k // CHUNK) % n_part != k_part:
            continue
        p = out / f"{k // CHUNK:05d}.pkl"
        if p.exists():
            continue
        rows = []
        for iid, audio, text, meta in js[k:k + CHUNK]:
            wrong = None
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
                    text, wrong = cor.ref, [bool(x) for x in cor.word_wrong]
                    meta = dict(meta, ckind=kind)
                if test:
                    os.environ[f"PRONUNCIADA_IDENTITY_{lang.upper()}"] = "0"
                    r0 = _row(sc.score(audio, text), False, False)
                    os.environ[f"PRONUNCIADA_IDENTITY_{lang.upper()}"] = "1"
                    row = _row(sc.score(audio, text), False, False)
                    row["prob_off"] = r0["prob"]
                else:
                    row = _row(sc.score(audio, text), True, name == "dev")
            except Exception as e:  # noqa: BLE001  (unreadable clip, empty text, ...)
                if "no activations" in str(e):
                    raise
                print("skip", iid, type(e).__name__, e, flush=True)
                continue
            row.update(id=iid, meta=meta, wrong=wrong)
            rows.append(row)
        p.write_bytes(pickle.dumps(rows))
        n += len(rows)
        print(lang, name, f"part {k_part}/{n_part}", k + CHUNK, len(js), f"{time.time() - t0:.0f}s", flush=True)


def load(lang: str, name: str, drop_v: bool = False):
    """Rows of a set (iterator over chunks to keep memory low when drop_v)."""
    out = []
    for p in sorted((work(lang) / "rows" / name).glob("*.pkl")):
        rows = pickle.loads(p.read_bytes())
        if drop_v:
            for r in rows:
                r.pop("V", None)
        out += rows
    return out


def iter_rows(lang: str, name: str):
    for p in sorted((work(lang) / "rows" / name).glob("*.pkl")):
        yield from pickle.loads(p.read_bytes())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--part")
    ap.add_argument("--workers", type=int, default=1, help="run this many --part processes (lock-helper lane)")
    a = ap.parse_args(argv)
    if a.workers > 1:
        import subprocess
        import sys

        for s in a.set.split(","):
            jobs(a.lang, s)                      # materialises shared audio once, before the workers start
            ps = [subprocess.Popen([sys.executable, "-m", "bench.round19_rollout.collect", "--lang", a.lang, "--set", s,
                                    "--part", f"{k}/{a.workers}"]) for k in range(a.workers)]
            codes = [p.wait() for p in ps]
            if any(codes):
                raise SystemExit(f"worker failed: {codes}")
        return
    for s in a.set.split(","):
        run(a.lang, s, a.part)


if __name__ == "__main__":
    main()
