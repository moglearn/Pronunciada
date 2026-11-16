"""Round 5 guard rail: round-5 vs round-4 sentence heads on NATIVE speech of every language.

    python -m bench.calib.native_sentence_check [--n 100] [--write-config]

~100 FLEURS VALIDATION clips per language (seeded sample) go through the public Scorer with the
round-5 sentence model forced for every language; the round-4 heads (sentence_r4.npz) are
applied to the same sentence inputs (the SENT_NUM columns of the same call). For fluency,
prosodic and total (0-10) the native medians are compared; a language whose median drops by
more than 0.3 on any of them keeps the round-4 heads: --write-config sets
config.json "sentence_model" = {lang: "sentence_r4"} for those languages.
Writes results/tables/round5_native.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

KEYS = ("accuracy", "fluency", "prosodic", "total")
CHECK = ("fluency", "prosodic", "total")
MAX_DROP = 0.3


def calib_dir_all_r5() -> str:
    """Copy of the shipped calib directory with no per-language sentence override."""
    import shutil

    from bench.calib.common import calib_cache, ship_dir

    d = calib_cache() / "r5_all_calib_dir"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(ship_dir(), d, ignore=shutil.ignore_patterns("__pycache__"))
    c = json.loads((d / "config.json").read_text())
    c.pop("sentence_model", None)
    (d / "config.json").write_text(json.dumps(c, indent=1))
    return str(d)


def run_lang(args):
    lang, n, cdir = args
    import os

    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    from bench.pipeline.common import iter_parquet_audio, records
    from pronunciada import features as F
    from pronunciada.calibrate import LevelModel, data_dir
    from pronunciada.scorer import Scorer

    sc = Scorer(lang, threads=1, calib_dir=cdir, prosody=False)
    r4 = LevelModel(data_dir() / "sentence_r4.npz")
    recs = records("fleurs-validation", lang=lang)
    rng = np.random.default_rng(0)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), min(n, len(recs)), replace=False))]
    by = {int(r["audio"].split(":")[2]): r for r in recs}
    path = recs[0]["audio"].split(":")[1]
    new, old = {k: [] for k in KEYS}, {k: [] for k in KEYS}
    rows = []
    for i, wav in iter_parquet_audio(path, set(by)):
        try:
            res = sc.score((wav, 16000), by[i]["text"])
        except ValueError:
            continue
        c = res._pronunciada  # noqa: SLF001
        X = np.column_stack([np.asarray(c["S"][k], dtype=np.float64) for k in F.SENT_NUM])
        o = r4.predict(X=X)
        for k in KEYS:
            new[k].append(float(c["sent"][f"reg_{k}"][0]))
            old[k].append(float(o[f"reg_{k}"][0]))
        rows.append({"utt": by[i]["id"], "S": {k: float(np.asarray(v)[0]) for k, v in c["S"].items()},
                     "delivery": c["delivery"], "pooled": c["pooled"]})
    import pickle

    from bench.calib.common import calib_cache

    (calib_cache() / f"r5_native_rows_{lang}_{n}.pkl").write_bytes(pickle.dumps(rows))
    return lang, new, old, sc.delivery


def main():
    from concurrent.futures import ProcessPoolExecutor

    from bench.calib.common import save_json, ship_dir
    from bench.pipeline.common import LANGS

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--write-config", action="store_true")
    a = ap.parse_args()
    cdir = calib_dir_all_r5()
    out = {"n_per_lang": a.n, "max_median_drop": MAX_DROP, "split": "fleurs-validation", "langs": {}}
    with ProcessPoolExecutor(a.workers) as ex:
        for lang, new, old, dlv in ex.map(run_lang, [(lg, a.n, cdir) for lg in LANGS]):
            r = {"n": len(new["total"]), "delivery_active": dlv}
            for k in KEYS:
                nn, oo = np.array(new[k]), np.array(old[k])
                r[k] = {"median_r4": float(np.median(oo)), "median_r5": float(np.median(nn)),
                        "median_delta": float(np.median(nn) - np.median(oo)),
                        "mean_r4": float(oo.mean()), "mean_r5": float(nn.mean()),
                        "corr_r4_r5": float(np.corrcoef(nn, oo)[0, 1]) if len(nn) > 2 else None}
            r["keep_r4"] = bool(any(r[k]["median_delta"] < -MAX_DROP for k in CHECK))
            out["langs"][lang] = r
            print(lang, r["n"], {k: (round(r[k]["median_r4"], 2), round(r[k]["median_r5"], 2)) for k in KEYS},
                  "KEEP R4" if r["keep_r4"] else "", flush=True)
    keep = sorted(lg for lg, r in out["langs"].items() if r["keep_r4"])
    out["keep_r4"] = keep
    save_json("round5_native.json", out)
    if a.write_config:
        p = ship_dir() / "config.json"
        cfg = json.loads(p.read_text())
        if keep:
            cfg["sentence_model"] = {lg: "sentence_r4" for lg in keep}
        else:
            cfg.pop("sentence_model", None)
        cfg["sentence_model_note"] = (
            "round 5: languages listed in sentence_model keep the round-4 sentence heads because "
            f"the round-5 heads lowered their native FLEURS-validation median fluency / prosodic / "
            f"total by more than {MAX_DROP} (bench/calib/native_sentence_check.py)")
        p.write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
        print("config sentence_model:", cfg.get("sentence_model"))


if __name__ == "__main__":
    main()
