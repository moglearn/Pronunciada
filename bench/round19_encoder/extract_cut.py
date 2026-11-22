"""Round 19 (encoder adaptation): cache the cut tensors (bench/round19_encoder/cut.py) per clip.

    python -m bench.round19_encoder.extract_cut --dataset speechocean762 --split train
    python -m bench.round19_encoder.extract_cut --dataset fleurs --split validation --n 100

Same waveform and fbank as the public scorer (pronunciada.audio.load_audio -> AcousticModel._features),
the fp32 graph, batch 1. Per clip: "cut" = float16 [T, 512] = stack 4 channels 0:384 ++ stack 3
channels 384:512 (= in0 ++ in1 ++ in2 of the top graph), plus the clip's frame count. The frozen
stack-5 output and final output are NOT stored: the top part recomputes them exactly.
Resumable shards of 100 clips under $PRONUNCIADA_CACHE/round19/encoder/cut/<dataset>_<split>/.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np


def out_dir(dataset, split):
    from bench.data.common import cache_root

    d = cache_root() / "round19" / "encoder" / "cut" / f"{dataset}_{split}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fleurs_records(split: str, n: int, seed: int = 0) -> list[dict]:
    """The n seeded FLEURS English clips of bench/round19/extract_en.py (english engineer's cache),
    waveform loaded. Same sample as bench/calib/native_sentence_check.py for validation, seed 0."""
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
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    from bench.pipeline.common import records
    from bench.round19_encoder import cut as C
    from pronunciada.acoustic import AcousticModel
    from pronunciada.audio import load_audio

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="speechocean762")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    if a.dataset == "fleurs":
        recs = fleurs_records(a.split, a.n)
    else:
        recs = [r for r in records(a.dataset) if r["split"] == a.split]
    d = out_dir(a.dataset, a.split)
    am = AcousticModel(precision="fp32", threads=1)
    sess = C.session(C.model_dir() / "cut.onnx", a.threads)
    shards = [recs[i:i + 100] for i in range(0, len(recs), 100)]
    t0 = time.time()
    for k, s in enumerate(shards):
        p = d / f"shard{k:03d}.pkl"
        if p.exists():
            continue
        out = {}
        for r in s:
            x = am._features(load_audio(r["audio"]))  # noqa: SLF001
            c = C.run_cut(sess, x)
            cut = np.concatenate([c["in0"][:, 0], c["in1"][:, 0], c["in2"][:, 0]], 1).astype(np.float16)
            out[r["id"]] = {"cut": cut, "n_frames": int(len(cut)), "text": r.get("text")}
        tmp = p.with_suffix(".part")
        tmp.write_bytes(pickle.dumps(out, protocol=5))
        os.replace(tmp, p)
        print(f"shard {k + 1}/{len(shards)} ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
