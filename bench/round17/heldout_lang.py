"""Round 17 held-out (run once): ten-language benchmark rows (bench.eval_languages, FLEURS test)
for ko / ru / zh, paired before (PRONUNCIADA_HEAD_CAPS=0) / after in the same tree. Word,
sentence, swap-benchmark and cv-real inputs do not read the shown sound probability, so only the
sound rows can move; they are re-run here to show it.

    python -m bench.round17.heldout_lang [--langs ko,ru,zh] [--workers 3]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

from bench.round17.collect import root


def _init(lang, flag):
    os.environ["PRONUNCIADA_HEAD_CAPS"] = flag
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    from bench.calib import api_run

    api_run._init(lang, "fp32", 1)


def main():
    from concurrent.futures import ProcessPoolExecutor

    from bench.data import load
    from bench.eval_languages import _utt, eval_lang
    from pronunciada.calibrate import config

    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="ko,ru,zh")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    cfg = config()
    out = {}
    for lang in a.langs.split(","):
        b = 0.0 if cfg["sound_model"].get(lang, cfg["universal_sound_model"]) != cfg["universal_sound_model"] \
            else float(cfg["bias"].get(lang, 0.0))
        out[lang] = {}
        for flag, name in (("0", "before"), ("1", "after")):
            p = root("test", lang, "heldout_lang") / f"languages_{flag}.pkl"
            if p.exists():
                o = pickle.loads(p.read_bytes())
            else:
                with ProcessPoolExecutor(min(a.workers, 3), initializer=_init, initargs=(lang, flag)) as ex:
                    o = dict(ex.map(_utt, load("fleurs", split="test", lang=lang), chunksize=4))
                p.write_bytes(pickle.dumps(o))
            r = eval_lang(lang, o, b)
            out[lang][name] = r
            print(lang, name, "native", r["native"]["0.5"], "sound auc", round(r["sound"]["auc"], 4),
                  "word auc", round(r["word"]["auc"], 4), flush=True)
    Path("results/tables/round17_languages_heldout.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
