"""Round 19b (German): pooled per-sound activation vectors for the sound-identity head.

    python -m bench.round19.identity_collect --set fleurs-train|cv-train|dev|devcor|devfar

Through the public Scorer("de") (default settings, so alignment and vectors are what the runtime
sees): per item the reference sounds (symbols, word index), the pooled enc1|enc3|enc5 vector of each
sound (pronunciada.sound_emb.phone_vectors, float16) and the word probabilities.
  fleurs-train   FLEURS de TRAIN natives (1000 clips)
  cv-train       Common Voice de train shard-0 natives (bench.round19.cv_train)
  dev            the round-19 dev items (controls and planted real-word edits)
  devcor         Common Voice dev controls with bench.corrupt controlled errors (one kind per clip)
Chunks: $CACHE/round19/german/ident/<set>/<k>.pkl (resumable).
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np

from bench.round19.devset import WORK

CHUNK = 50


def jobs(name):
    """[(id, audio, text or Reference factory, meta)]."""
    if name == "fleurs-train":
        from bench.data import load

        return [(r["id"], r["audio"], r["text"], {"src": "fleurs-train"}) for r in sorted(load("fleurs", "train", "de"), key=lambda r: r["id"])]
    if name == "cv-train":
        import json

        from bench.round19.cv_train import OUT_AUDIO

        sel = json.loads((WORK / "cv_train_select.json").read_text())
        import hashlib

        return [(f"cvtr-{c['stem']}", str(OUT_AUDIO / f"{c['stem']}.flac"), c["sentence"],
                 {"src": "cv-train", "speaker": hashlib.sha1(c["speaker"].encode()).hexdigest()[:12]})
                for c in sel if (OUT_AUDIO / f"{c['stem']}.flac").exists()]
    if name == "dev":
        from bench.round19.collect import manifest

        return [(it["id"], it["audio"], it["target"], {"src": it["source"], "fold": it["fold"], "label": it["label"],
                                                       "speaker": it["speaker"]}) for it in manifest("dev")]
    if name == "devcor":
        from bench.round19.collect import CORRUPT_KINDS, manifest

        items = [it for it in manifest("dev") if it["source"] == "cv" and not it["label"]]
        return [(it["id"] + "__cor", it["audio"], ("cor", it["target"], it["id"], CORRUPT_KINDS[j % len(CORRUPT_KINDS)]),
                 {"src": "cv", "fold": it["fold"], "speaker": it["speaker"]}) for j, it in enumerate(items)]
    raise SystemExit(name)


def run(name):
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.common import stable_seed
    from bench.pipeline.evalctl import inventory
    from pronunciada import Scorer
    from pronunciada.sound_emb import phone_vectors

    out = WORK / "ident" / name
    out.mkdir(parents=True, exist_ok=True)
    js = jobs(name)
    sc = Scorer("de", threads=2)
    inv = inventory("de")
    t0 = time.time()
    for k in range(0, len(js), CHUNK):
        p = out / f"{k // CHUNK:04d}.pkl"
        if p.exists():
            continue
        rows = []
        for iid, audio, text, meta in js[k:k + CHUNK]:
            wrong = None
            try:
                if isinstance(text, tuple):
                    _, tgt, src_id, kind = text
                    ref = sc.g2p(tgt)
                    try:
                        cor = corrupt(ref, [kind], inv, n_errors=2, seed=stable_seed(src_id, kind, "r19"), strict=False)
                    except NotApplicable:
                        continue
                    if not cor.errors:
                        continue
                    res = sc.score(audio, cor.ref)
                    wrong = [bool(x) for x in cor.word_wrong]
                    meta = dict(meta, kind=kind)
                else:
                    res = sc.score(audio, text)
            except ValueError:
                continue
            c = res._pronunciada  # noqa: SLF001
            if not c.get("emb"):
                raise RuntimeError("no activations: the German scorer must run with embeddings")
            ref = c["ref"]
            V = phone_vectors(c["emb"], c["ev"], len(ref.phones)).astype(np.float16)
            rows.append({"id": iid, "meta": meta, "wrong": wrong, "text": res.text,
                         "ipa": [ph.ipa for ph in ref.phones], "word": np.array([ph.word for ph in ref.phones], np.int32),
                         "V": V, "word_prob": np.array([w.prob for w in res.words], np.float32),
                         "word_sounds": np.array([w.sounds_prob for w in res.words], np.float32),
                         "word_text": [w.text for w in res.words],
                         "word_span": [list(w.char_span) for w in ref.words]})
        p.write_bytes(pickle.dumps(rows))
        print(name, k + CHUNK, len(js), f"{time.time() - t0:.0f}s", flush=True)


def load(name):
    out = []
    for p in sorted((WORK / "ident" / name).glob("*.pkl")):
        out += pickle.loads(p.read_bytes())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    run(ap.parse_args().set)
