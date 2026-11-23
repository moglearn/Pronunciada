"""Baselines a later before / after gate will read, measured with the current code (no held-out data).

    python -m bench.round20_native_fa.baselines practice      # FA@80 / AUC per language (core.py)
    python -m bench.round20_native_fa.baselines errdet        # marked-sound detection, L2-ARCTIC / EpaDB TRAIN
    python -m bench.round20_native_fa.baselines ondevice      # bench.ondevice, output kept out of results/

errdet: every TRAIN clip of L2-ARCTIC and EpaDB through the public Scorer (bench.calib.api_run, outputs in
this job's cache), annotation phones matched to ours exactly as bench.eval_english.err_items does (no
Azure columns); error score = 1 - P(sound) at strictness 0.5 of the shipped sound model; ROC AUC with a
speaker-cluster bootstrap (2,000 draws), all errors and substitutions / deletions separately.
These corpora are non-commercial: measured only, nothing is fitted or selected on them.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from bench.round20_native_fa.rebuild import ROOT


def practice() -> dict:
    from bench.round20_native_fa import core as K

    out = {}
    for lang in K.LANGS:
        rows = K.word_rows(lang)
        auc, fa = K.metrics(rows)
        ci = K.boot(rows, K.metrics)
        neg = [r for r in rows if not r["label"]]
        out[lang] = {"auc": auc, "auc_ci": ci[0], "fa80": fa, "fa80_ci": ci[1], "n_words": len(rows),
                     "n_changed": sum(r["label"] for r in rows), "n_items": len({r["item"] for r in rows}),
                     "n_speakers": len({r["speaker"] for r in rows}),
                     "low06_pct": 100 * sum(r["score"] < K.APP_THRESHOLD for r in neg) / len(neg)}
        print(lang, json.dumps(out[lang]))
    K.save_json("baseline_practice.json", out)
    return out


def errdet(workers: int = 2) -> dict:
    os.environ["PRONUNCIADA_API_DIR"] = str(ROOT / "api")
    from bench.calib.api_run import run
    from bench.data import load
    from bench.eval_english import N_BOOT, _their_entries, blend
    from bench.metrics import bootstrap_ci, roc_auc
    from bench.pipeline.common import build_ref
    from bench.pipeline.labels import align_reference

    res = {}
    for ds in ("l2arctic", "epadb"):
        recs = load(ds, split="train")
        api = run(recs, "en", precision="fp32", workers=workers, tag=f"{ds}-train-fp32")
        rows = []
        for rec in recs:
            o = api.get(rec["id"])
            if o is None or "error" in o:
                continue
            ref = build_ref("en", rec["text"])
            ents = _their_entries(rec, ds)
            if ds == "l2arctic":
                words = [w["text"] for w in rec["labels"]["words"]]
                tp = [[e["ph"] for e in ents if e["word"] == j] for j in range(len(words))]
                pos = {}
                for k, e in enumerate(ents):
                    pos.setdefault(e["word"], []).append(k)
                o2t, _ = align_reference(ref, words, tp)
                t2o = {pos[j][q]: i for i, (j, q) in o2t.items() if j in pos and q < len(pos[j])}
            else:
                o2t, _ = align_reference(ref, None, [[e["ph"] for e in ents]])
                t2o = {q: i for i, (_, q) in o2t.items()}
            matched = [t2o.get(k, -1) for k in range(len(ents))]
            name = next(iter(o["sound"]))          # the shipped sound model of the API run
            d = o["sound"][name]
            pr = blend(d["z_len"], d["z_str"], 0.5)
            for k, e in enumerate(ents):
                if e["label"] == "unk":
                    continue
                i = matched[k]
                if i < 0:
                    nb = [matched[q] for q in (k - 1, k + 1) if 0 <= q < len(ents) and matched[q] >= 0]
                    s = float(1 - np.mean(pr[nb])) if nb else float(1 - np.mean(pr))
                else:
                    s = float(1 - pr[i])
                rows.append((e["label"], rec["speaker"], s, name))
        y = np.array([r[0] != "correct" for r in rows], int)
        s = np.array([r[2] for r in rows], float)
        spk = np.array([r[1] for r in rows])
        out = {"model": rows[0][3], "n": int(len(y)), "n_err": int(y.sum()), "auc": roc_auc(y, s)}
        ci = bootstrap_ci(roc_auc, y, s, clusters=spk, n_boot=N_BOOT)
        out["auc_ci"] = [ci["lo"], ci["hi"]]
        for et in ("sub", "del"):
            m = np.array([r[0] in ("correct", et) for r in rows])
            out[f"auc_{et}"] = roc_auc(y[m], s[m])
        res[ds] = out
        print(ds, json.dumps(out), flush=True)
    from bench.round20_native_fa import core as K

    K.save_json("baseline_errdet_train.json", res)
    return res


def so762() -> dict:
    """Summary of the speechocean762 TRAIN speaker-grouped CV re-run (so762cv.sh: the round-19 scripts
    bench/round19/extract_en, exp_seq --name t_ens3, cv_blend, exp_level, on features re-extracted with main)."""
    from bench.round20_native_fa import core as K

    d = ROOT / "so762cv" / "tables"
    seq = json.loads((d / "round19_english_cv_seq.json").read_text())["t_ens3"]["mean"]
    blend = json.loads((d / "round19_english_cv_blend.json").read_text())
    lev = json.loads((d / "round19_english_cv_level.json").read_text())["t_ens3"]["mean"]
    out = {"note": "5-fold speaker-grouped CV on speechocean762 TRAIN, 3 seeds, children and others dealt separately"
                   " (bench/round19/data.folds); the shipped blends: sentence 0.35 x sequence head + 0.65 x trees,"
                   " sound 0.3, word 0.4",
           "sentence_shipped_blend_w0.35": blend["w0.35"]["mean"], "sentence_trees_only": blend["w0.0"]["mean"],
           "sentence_round2_base": blend["base"]["mean"], "sequence_head_alone": seq,
           "sound_blend_w0.3": lev["sound"]["w0.3"], "word_blend_w0.4": lev["word"]["w0.4"]}
    K.save_json("baseline_so762_train_cv.json", out)
    print(json.dumps(out, indent=1))
    return out


def ondevice_paired(rounds: int = 3) -> None:
    """Round-20 latency cost: bench.ondevice (pronunciada-fp32 and -int8, 4 threads, the same 200 speechocean762
    TRAIN clips) with the round-20 switches off and on (B+C+E, and B alone), interleaved `rounds` times so that
    machine load hits both alike. Medians of the per-run p50 / mean per-clip times."""
    import statistics

    import bench.ondevice as OD
    from bench.data import load

    OD.clips = lambda n: [{"id": r["id"], "audio": r["audio"], "text": r["text"], "dur": r["dur"]}
                          for r in load("speechocean762", split="train")[:n]]
    cfgs = {"off": {}, "B": {"PRONUNCIADA_R20_UNSAID": "1"},
            "B+C+E": {"PRONUNCIADA_R20_UNSAID": "1", "PRONUNCIADA_R20_NUMALT": "1", "PRONUNCIADA_R20_ES_GLIDE": "1",
                      "PRONUNCIADA_R20_DE_LEXALT": "1", "PRONUNCIADA_R20_KO_NINS": "1"}}
    res = {}
    for _ in range(rounds):
        for sysname in ("pronunciada-fp32", "pronunciada-int8"):
            for name, env in cfgs.items():
                for k in [k for k in os.environ if k.startswith("PRONUNCIADA_R20")]:
                    del os.environ[k]
                os.environ.update(env)
                r = OD.measure(sysname, 4, 200)
                t = r.get("times") or []
                res.setdefault(f"{sysname}|{name}", []).append(
                    {"p50": r["p50_s"], "mean": (sum(t) / len(t)) if t else r["rtf"] * r["audio_seconds"] / r["n_clips"],
                     "load": r.get("load_avg_1min_before")})
                print(sysname, name, round(1000 * r["p50_s"], 2), flush=True)
    for k in [k for k in os.environ if k.startswith("PRONUNCIADA_R20")]:
        del os.environ[k]
    out = {k: {"p50_ms": 1000 * statistics.median(x["p50"] for x in v),
               "mean_ms": 1000 * statistics.median(x["mean"] for x in v), "runs": v} for k, v in res.items()}
    from bench.round20_native_fa import core as K

    K.save_json("r20_latency.json", out)
    print(json.dumps({k: (round(v["p50_ms"], 2), round(v["mean_ms"], 2)) for k, v in out.items()}))


def ondevice() -> None:
    """bench.ondevice for pronunciada-fp32 / int8 at 4 threads, written to this job's cache.
    One change to the protocol: the 200 clips are the first 200 speechocean762 TRAIN recordings instead of
    TEST (no held-out audio is run in this job). Also: size of the package data folder."""
    import bench.ondevice as OD
    from bench.data import load

    OD.OUT = ROOT / "ondevice"
    OD.OUT.mkdir(parents=True, exist_ok=True)
    OD.clips = lambda n: [{"id": r["id"], "audio": r["audio"], "text": r["text"], "dur": r["dur"]}
                          for r in load("speechocean762", split="train")[:n]]
    OD.main(["--system", "pronunciada-fp32,pronunciada-int8", "--threads", "4"])
    from pathlib import Path

    import pronunciada

    d = Path(pronunciada.__file__).parent / "data"
    total = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    (OD.OUT / "package_data.json").write_text(json.dumps({"pronunciada_data_bytes": total,
                                                          "pronunciada_data_mb": total / 2**20}))
    print("package data MB", round(total / 2**20, 2))


if __name__ == "__main__":
    {"practice": practice, "errdet": errdet, "ondevice": ondevice, "so762": so762,
     "ondevice_paired": ondevice_paired}[sys.argv[1]]()
