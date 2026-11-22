"""Round 19 (English): per-utterance feature cache for the sentence / word / sound heads.

    python -m bench.round19.extract_en --dataset speechocean762 --split train [--precision fp32]
        [--workers 4] [--out DIR]

Every utterance goes through the PUBLIC scorer (pronunciada.Scorer("en", delivery=True)), with
the acoustic model's graph extended to expose the output of EVERY encoder stack (enc0..enc5)
plus the final encoder output (pronunciada.acoustic.ALL_LAYER_OUTPUTS). Nothing is fitted here.

Per utterance (pickled in resumable shards under DIR/<dataset>_<split>_<precision>/):
  ids, speaker, age, labels (sentence scores; per OUR phone the expert phone score or NaN; per OUR
  word the mean expert word accuracy or NaN, via bench.pipeline.labels.align_reference),
  phone table (sound_B inputs), shipped outputs (sound_E 0-2, sound_U logits, word logits,
  word regressions, sentence_r4 inputs and outputs), delivery features (pronunciada.delivery,
  pitch included) + F0 summary, per-phone spans, per-phone mean of every layer (float16,
  the window of pronunciada.sound_emb.phone_vectors), per-word mean of every layer and
  per-utterance mean ++ std of every layer over speech frames.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np

_SC = None
SHARD = 100


def out_root() -> Path:
    from bench.data.common import cache_root

    return cache_root() / "round19" / "english" / "feats"


def _init(precision):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    os.environ.setdefault("PRONUNCIADA_ALL_LAYERS", "1")
    global _SC
    from pronunciada.scorer import Scorer

    _SC = Scorer("en", precision=precision, threads=2, delivery=True)
    assert _SC.am.embeddings and "enc0" in _SC.am.embed_outputs, "all-layer model not loaded"


def layer_phone_means(emb: dict, ev, n: int) -> dict:
    out = {}
    T = len(next(iter(emb.values())))
    win = []
    for sl in ev.slots[:n]:
        a, b = int(sl.start_frame), int(sl.end_frame)
        if b <= a:
            a, b = a - 1, a + 1
        a, b = max(a - 1, 0), min(b + 1, T)
        if b <= a:
            a = max(min(a, T - 1), 0)
            b = a + 1
        win.append((a, b))
    for k, v in emb.items():
        v = np.asarray(v, np.float32)
        out[k] = np.stack([v[a:b].mean(0) for a, b in win]).astype(np.float16)
    return out


def labels_for(rec, ref):
    from bench.pipeline.labels import align_reference, arpa_to_ours

    W = rec["labels"]["words"]
    tp = [[arpa_to_ours(p) for p in w["phones"]] for w in W]
    o2t, wpairs = align_reference(ref, [w["text"] for w in W], tp)
    ph = np.full(len(ref.phones), np.nan)
    for i, (j, k) in o2t.items():
        ph[i] = float(W[j]["phones_accuracy"][k])
    tw = {}
    for i, (j, _) in o2t.items():
        tw.setdefault(ref.phones[i].word, set()).add(j)
    for i, j in wpairs:
        tw.setdefault(i, set()).add(j)
    wd = np.full(len(ref.words), np.nan)
    for i, js in tw.items():
        wd[i] = float(np.mean([W[j]["accuracy"] for j in js]))
    lab = rec["labels"]
    return {"sentence": {k: float(lab["sentence"][k]) for k in ("accuracy", "fluency", "prosodic", "total", "completeness")},
            "phone": ph, "word": wd, "age": lab.get("age"), "gender": lab.get("gender")}


def _one(rec):
    from pronunciada import delivery as D
    from pronunciada.calibrate import logit

    sc = _SC
    try:
        res = sc.score(rec["audio"], rec["text"])
    except ValueError as e:
        return rec["id"], {"error": str(e)}
    c = res._pronunciada  # noqa: SLF001
    ref, ev, tab, emb = c["ref"], c["ev"], c["tab"], c["emb"]
    n = len(ref.phones)
    d = sc.details(res)
    num = sc.sound.num
    X = np.column_stack([np.asarray(tab[k], np.float64) for k in num]).astype(np.float32)
    u = sc.sound_outputs(res, sc.universal)
    W = c["W"]
    wcols = sorted(k for k, v in W.items() if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == len(ref.words)
                   and v.dtype.kind in "fib")
    S = c["S"]
    scols = sorted(k for k, v in S.items() if isinstance(v, np.ndarray) and v.size == 1 and v.dtype.kind in "fib")
    trk = c["f0"]
    f0 = {}
    if trk is not None and len(trk.f0):
        v = trk.f0[trk.voiced & (trk.f0 > 0)]
        f0 = {"f0_med_hz": float(np.median(v)) if len(v) else np.nan,
              "f0_p10_hz": float(np.percentile(v, 10)) if len(v) else np.nan,
              "f0_p90_hz": float(np.percentile(v, 90)) if len(v) else np.nan,
              "n_voiced10": int(len(v))}
    spans = c.get("word_spans") or D.word_frame_spans(ref, ev, len(c["lpa"]))
    wmean = {k: np.stack([np.asarray(v[a:b], np.float32).mean(0) for a, b in spans]).astype(np.float16)
             for k, v in emb.items()}
    pooled = D.pool_embeddings(emb, ev)
    out = {
        "text": res.text, "n_frames": int(ev.n_frames), "dur": float(c["dur"]),
        "phones": [p.ipa for p in ref.phones], "phone_word": np.array([p.word for p in ref.phones]),
        "phone_cls": [str(x) for x in tab.get("cls", [""] * n)],
        "phone_pos": [str(x) for x in tab.get("position", [""] * n)],
        "words": [w.text for w in ref.words], "word_phones": [tuple(w.phones) for w in ref.words],
        "start": np.array([s.start_frame for s in ev.slots[:n]], np.int32),
        "end": np.array([s.end_frame for s in ev.slots[:n]], np.int32),
        "word_spans": np.array(spans, np.int32),
        "tab_num": X, "tab_names": list(num),
        "sound_E_reg": np.asarray(d["phone_score"], np.float32),
        "sound_B": (np.asarray(c["s_zl"], np.float32), np.asarray(c["s_zs"], np.float32)),
        "sound_U": (np.asarray(logit(u["len"]), np.float32), np.asarray(logit(u["str"]), np.float32),
                    np.asarray(u["reg"], np.float32)),
        "W": {k: np.asarray(W[k], np.float32) for k in wcols},
        "w_z": (np.asarray(c["w_zl"], np.float32), np.asarray(c["w_zs"], np.float32)),
        "w_reg": np.asarray(c["w_reg"], np.float32), "w_reg_r2": np.asarray(c["w_reg_r2"], np.float32),
        "S": {k: float(np.asarray(S[k]).reshape(-1)[0]) for k in scols},
        "sent": {k: float(c["sent"][f"reg_{k}"][0]) for k in ("accuracy", "fluency", "prosodic", "total")},
        "sent_z": (float(c["sent_zl"]), float(c["sent_zs"])),
        "delivery": {k: float(v) for k, v in (c["delivery"] or {}).items()},
        "f0": f0,
        "voice": __import__("pronunciada.sentence_head", fromlist=["voice_cues"]).voice_cues(c["wav"]),
        "ph_layers": layer_phone_means(emb, ev, n),
        "w_layers": wmean,
        "utt_layers": {k: v.astype(np.float16) for k, v in pooled.items()},
        "word_prob": np.array([w.prob for w in res.words], np.float32),
        "timing": res.timing_ms,
    }
    out["labels"] = labels_for(rec, ref) if "sentence" in (rec.get("labels") or {}) else None
    out["speaker"] = rec.get("speaker")
    return rec["id"], out


def fleurs_records(split: str, n: int, seed: int = 0) -> list[dict]:
    """n seeded FLEURS English clips (the sample of bench/calib/native_sentence_check.py for
    split "validation", seed 0), with the waveform loaded."""
    from bench.pipeline.common import iter_parquet_audio, records

    recs = records(f"fleurs-{split}", lang="en") if split == "validation" else \
        [r for r in records("fleurs", lang="en") if r.get("split") == split]
    rng = np.random.default_rng(seed)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), min(n, len(recs)), replace=False))]
    by = {int(r["audio"].split(":")[2]): r for r in recs}
    path = recs[0]["audio"].split(":")[1]
    out = []
    for i, wav in iter_parquet_audio(path, set(by)):
        out.append({**by[i], "audio": (wav, 16000)})
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor

    from bench.pipeline.common import records

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="speechocean762")
    ap.add_argument("--split", default="train")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=100, help="fleurs: clips to sample")
    a = ap.parse_args()
    if a.dataset == "fleurs":
        recs = fleurs_records(a.split, a.n)
    else:
        recs = [r for r in records(a.dataset) if r["split"] == a.split]
    if a.limit:
        recs = recs[:a.limit]
    d = Path(a.out) if a.out else out_root() / f"{a.dataset}_{a.split}_{a.precision}"
    d.mkdir(parents=True, exist_ok=True)
    shards = [recs[i:i + SHARD] for i in range(0, len(recs), SHARD)]
    todo = [(k, s) for k, s in enumerate(shards) if not (d / f"shard{k:03d}.pkl").exists()]
    print(f"{len(recs)} records, {len(shards)} shards, {len(todo)} to do -> {d}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.precision,)) as ex:
        for k, s in todo:
            out = dict(ex.map(_one, s, chunksize=2))
            tmp = d / f"shard{k:03d}.pkl.part"
            tmp.write_bytes(pickle.dumps(out, protocol=5))
            os.replace(tmp, d / f"shard{k:03d}.pkl")
            print(f"shard {k + 1}/{len(shards)} ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
