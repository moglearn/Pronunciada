"""Round 6 guard rail for the reported 0-10 word score: round-6 vs round-2 word regression on
NATIVE speech (100 FLEURS VALIDATION clips per language, the same sample as
bench/calib/native_sentence_check.py).

    python -m bench.calib.native_word_check [--write-config]

Both scores come from the same public Scorer call (round 6 = WordResult details "word_accuracy"
with word_score_model forced for every language, round 2 = the cached "w_reg_r2"). A language
whose native median word score drops by more than 0.3 keeps the round-2 score; --write-config
sets config.json "word_score_model" = {lang: "word_reg_r6"} for the others.
Writes results/tables/round6_native_word.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

MAX_DROP = 0.3


def calib_dir_all() -> str:
    import shutil

    from bench.calib.common import calib_cache, ship_dir
    from bench.pipeline.common import LANGS

    d = calib_cache() / "r6_word_all_calib_dir"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(ship_dir(), d, ignore=shutil.ignore_patterns("__pycache__"))
    c = json.loads((d / "config.json").read_text())
    c["word_score_model"] = {lg: "word_reg_r6" for lg in LANGS}
    (d / "config.json").write_text(json.dumps(c, indent=1))
    return str(d)


def run_lang(args):
    lang, n, cdir = args
    import os

    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    from bench.pipeline.common import iter_parquet_audio, records
    from pronunciada.scorer import Scorer

    sc = Scorer(lang, threads=1, calib_dir=cdir, prosody=False)
    recs = records("fleurs-validation", lang=lang)
    rng = np.random.default_rng(0)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), min(n, len(recs)), replace=False))]
    by = {int(r["audio"].split(":")[2]): r for r in recs}
    path = recs[0]["audio"].split(":")[1]
    new, old = [], []
    for i, wav in iter_parquet_audio(path, set(by)):
        try:
            res = sc.score((wav, 16000), by[i]["text"])
        except ValueError:
            continue
        c = res._pronunciada  # noqa: SLF001
        new += list(np.asarray(c["w_reg"], dtype=float))
        old += list(np.asarray(c["w_reg_r2"], dtype=float))
    return lang, np.array(new), np.array(old)


def main():
    from concurrent.futures import ProcessPoolExecutor

    from bench.calib.common import save_json, ship_dir
    from bench.pipeline.common import LANGS

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--write-config", action="store_true")
    a = ap.parse_args()
    cdir = calib_dir_all()
    out = {"n_clips_per_lang": a.n, "max_median_drop": MAX_DROP, "langs": {}}
    with ProcessPoolExecutor(a.workers) as ex:
        for lang, new, old in ex.map(run_lang, [(lg, a.n, cdir) for lg in LANGS]):
            r = {"n_words": int(len(new)), "median_r2": float(np.median(old)), "median_r6": float(np.median(new)),
                 "mean_r2": float(old.mean()), "mean_r6": float(new.mean()),
                 "share_below_5_r2": float((old < 5).mean()), "share_below_5_r6": float((new < 5).mean())}
            r["median_delta"] = r["median_r6"] - r["median_r2"]
            r["mean_delta"] = r["mean_r6"] - r["mean_r2"]
            r["adopt"] = bool(r["median_delta"] >= -MAX_DROP)
            out["langs"][lang] = r
            print(lang, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    save_json("round6_native_word.json", out)
    if a.write_config:
        p = ship_dir() / "config.json"
        cfg = json.loads(p.read_text())
        cfg["word_score_model"] = {lg: "word_reg_r6" for lg, r in out["langs"].items() if r["adopt"]}
        cfg["word_score_model_note"] = (
            "round 6: reported 0-10 word score with word-pooled ZIPA stack-1 activations "
            "(bench/calib/train_word_emb.py) for the listed languages; the others keep word_reg.npz "
            f"(native FLEURS-validation median dropped by more than {MAX_DROP}, bench/calib/native_word_check.py)")
        p.write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
        print("config word_score_model:", cfg["word_score_model"])


if __name__ == "__main__":
    main()
