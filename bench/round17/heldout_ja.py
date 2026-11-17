"""Round 17, Japanese held-out (run once): the length channel's head input changes Japanese word
probabilities, so the word-level benchmarks are re-run paired, before (PRONUNCIADA_HEAD_CAPS=0)
and after, in the same tree. Azure: cached answers only (no requests).

    python -m bench.round17.heldout_ja [--workers 3]

* ten-language benchmark rows for ja (bench.eval_languages._utt / eval_lang on FLEURS test);
* the swap benchmark's ja records (bench.eval_languages._swap_one, bench.eval_swap.evaluate);
* cv-real ja, test half (bench.eval_cv summarize_result / evaluate with the cached Azure scores).
Pickles under $PRONUNCIADA_CACHE/bench/round17/heldout_ja/. Writes results/tables/round17_ja_heldout.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from bench.round17.collect import root


def _init(flag):
    os.environ["PRONUNCIADA_HEAD_CAPS"] = flag
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    from bench.calib import api_run

    api_run._init("ja", "fp32", 1)


def _cv_one(rec):
    from bench.calib import api_run
    from bench.eval_cv import summarize_result

    try:
        return rec["id"], summarize_result(api_run._SC.score(rec["audio"], rec["sentence"]))
    except ValueError as e:
        return rec["id"], {"status": "error", "error": str(e)[:200]}


def run(flag, workers):
    from concurrent.futures import ProcessPoolExecutor

    from bench import eval_cv as CV
    from bench.data import load
    from bench.eval_languages import _swap_one, _utt

    d = root("test", "ja", "heldout_ja")
    out = {}
    for name, fn, recs in (("languages", _utt, load("fleurs", split="test", lang="ja")),
                           ("swap", _swap_one, [r for r in load("fleurs-swap", split="test") if r["lang"] == "ja"]),
                           ("cv", _cv_one, [r for r in CV.unique_clips(CV.records(["ja"])) if r.get("half") == "test"])):
        p = d / f"{name}_{flag}.pkl"
        if p.exists():
            out[name] = pickle.loads(p.read_bytes())
            continue
        with ProcessPoolExecutor(min(workers, 3), initializer=_init, initargs=(flag,)) as ex:
            o = dict(ex.map(fn, recs, chunksize=4))
        p.write_bytes(pickle.dumps(o))
        out[name] = o
        print(flag, name, len(o), flush=True)
    return out


def main():
    from bench import eval_cv as CV
    from bench.eval_languages import eval_lang
    from bench.eval_swap import evaluate as swap_eval
    from pronunciada.calibrate import config

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    cfg = config()
    b = 0.0 if cfg["sound_model"].get("ja", cfg["universal_sound_model"]) != cfg["universal_sound_model"] \
        else float(cfg["bias"].get("ja", 0.0))
    rep = {}
    from bench.data import load

    srecs = [r for r in load("fleurs-swap", split="test") if r["lang"] == "ja"]
    cvrecs = [r for r in CV.records(["ja"]) if r.get("half") == "test"]
    az = CV.azure_scores(cvrecs)
    for flag, name in (("0", "before"), ("1", "after")):
        o = run(flag, a.workers)
        lang = eval_lang("ja", o["languages"], b)
        sw = swap_eval(srecs, {k: v["on"] for k, v in o["swap"].items()})
        cv = CV.evaluate(cvrecs, o["cv"], az)
        rep[name] = {"languages": lang, "swap": sw.get("ja", sw), "cv": cv}
    Path("results/tables/round17_ja_heldout.json").write_text(json.dumps(rep, indent=1, default=float))

    def pick(r):
        L = r["languages"]
        return {"native_word_0.5": L["native"]["0.5"], "sound_auc": L["sound"]["auc"], "word_auc": L["word"]["auc"]}
    print(json.dumps({k: pick(v) for k, v in rep.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
