"""Round 19 (German): the guards on VALIDATION data, before / after (PRONUNCIADA_DE_R19=0 / 1).

    PRONUNCIADA_DE_R19=0 python -m bench.round19.guards_dev collect --tag off
    python -m bench.round19.guards_dev collect --tag on
    python -m bench.round19.guards_dev report off on

Through the public Scorer("de") with default settings:
  * swap: bench.calib.bias_grid dev swap set (FLEURS validation: orig / swap_near / swap_far texts):
    word AUC of the swapped word vs native words, hit @ 5 % native flags, native words < 0.6;
  * stress / sound: the fit_prosody protocol on 200 FLEURS validation clips (same clip choice and
    seeds): clean reference, 2 planted stress errors, 5 sound-error kinds (bench.corrupt);
    stress-error word AUC / hit @ 5 % (WordResult.prob), sound-error word AUC / hit @ 5 %,
    native word acceptance (P >= 0.5).
Rows: $PRONUNCIADA_CACHE/round19/german/guards_<tag>.pkl.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from bench.round19.devset import WORK


def collect(tag):
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    from bench.calib.bias_grid import dev_swap
    from bench.calib.fit_prosody import SOUND_KINDS
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.common import iter_parquet_audio, records, stable_seed
    from bench.pipeline.evalctl import inventory
    from pronunciada.scorer import Scorer

    sc = Scorer("de", threads=2)
    out = {"swap": [], "pros": []}
    # swap set
    sw = dev_swap("de")
    by_src = {}
    for r in sw:
        by_src.setdefault(r["src"], []).append(r)
    recs = {r["id"]: r for r in records("fleurs-validation", lang="de")}
    # stress / sound set: fit_prosody's choice of 200 clips
    allv = records("fleurs-validation", lang="de")
    rng = np.random.default_rng(0)
    pros = {allv[i]["id"] for i in sorted(rng.choice(len(allv), min(200, len(allv)), replace=False))}
    want = set(by_src) | pros
    rows = {int(recs[i]["audio"].split(":")[2]): recs[i] for i in want if i in recs}
    path = allv[0]["audio"].split(":")[1]
    inv = inventory("de")
    n = 0
    for i, wav in iter_parquet_audio(path, set(rows)):
        rec = rows[i]
        for r in by_src.get(rec["id"], []):
            try:
                res = sc.score((wav, 16000), r["text"])
            except ValueError:
                continue
            lab = r["labels"]
            out["swap"].append({"id": r["id"], "variant": lab["variant"], "k": lab.get("word_index"),
                                "spans": [w.char_span for w in res._pronunciada["ref"].words],
                                "words": lab["words"], "prob": [w.prob for w in res.words]})
        if rec["id"] in pros:
            try:
                res = sc.score((wav, 16000), rec["text"])
            except ValueError:
                continue
            ref = res._pronunciada["ref"]
            out["pros"].append({"utt": rec["id"], "ref": "clean", "prob": [w.prob for w in res.words],
                                "wrong": [False] * len(res.words)})
            for k in ("stress",) + SOUND_KINDS:
                try:
                    cor = corrupt(ref, [k], inv, n_errors=2, seed=stable_seed(rec["id"], k, "prosody-v1"), strict=False)
                except NotApplicable:
                    continue
                if not cor.errors:
                    continue
                r2 = sc.score((wav, 16000), cor.ref)
                out["pros"].append({"utt": rec["id"], "ref": "stress" if k == "stress" else "sound",
                                    "prob": [w.prob for w in r2.words], "wrong": list(map(bool, cor.word_wrong))})
        n += 1
        if n % 50 == 0:
            print(n, flush=True)
    (WORK / f"guards_{tag}.pkl").write_bytes(pickle.dumps(out))


def collect_test(tag):
    """The same measurements on FLEURS TEST (held-out; final run only): the fleurs-swap test set of
    bench.eval_swap (orig / swap_near / swap_far) and the stress / sound-error protocol on all 400
    FLEURS test clips."""
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    from bench.calib.fit_prosody import SOUND_KINDS
    from bench.corrupt import NotApplicable, corrupt
    from bench.data import load
    from bench.pipeline.common import stable_seed
    from bench.pipeline.evalctl import inventory
    from pronunciada.scorer import Scorer

    sc = Scorer("de", threads=2)
    inv = inventory("de")
    out = {"swap": [], "pros": []}
    for r in load("fleurs-swap", "test", "de"):
        try:
            res = sc.score(r["audio"], r["text"])
        except ValueError:
            continue
        lab = r["labels"]
        out["swap"].append({"id": r["id"], "variant": lab["variant"], "k": lab.get("word_index"),
                            "spans": [w.char_span for w in res._pronunciada["ref"].words],
                            "words": lab["words"], "prob": [w.prob for w in res.words]})
    for rec in load("fleurs", "test", "de"):
        try:
            res = sc.score(rec["audio"], rec["text"])
        except ValueError:
            continue
        ref = res._pronunciada["ref"]
        out["pros"].append({"utt": rec["id"], "ref": "clean", "prob": [w.prob for w in res.words],
                            "wrong": [False] * len(res.words)})
        for k in ("stress",) + SOUND_KINDS:
            try:
                cor = corrupt(ref, [k], inv, n_errors=2, seed=stable_seed(rec["id"], k, "prosody-v1"), strict=False)
            except NotApplicable:
                continue
            if not cor.errors:
                continue
            r2 = sc.score(rec["audio"], cor.ref)
            out["pros"].append({"utt": rec["id"], "ref": "stress" if k == "stress" else "sound",
                                "prob": [w.prob for w in r2.words], "wrong": list(map(bool, cor.word_wrong))})
    (WORK / f"guards_{tag}.pkl").write_bytes(pickle.dumps(out))


def boot_delta(tag0, tag1, n_boot=1000):
    """Paired bootstrap over clips (source utterances) of every metric: after - before."""
    from bench.metrics import roc_auc  # noqa: F401  (metrics() imports it)

    d0 = pickle.loads((WORK / f"guards_{tag0}.pkl").read_bytes())
    d1 = pickle.loads((WORK / f"guards_{tag1}.pkl").read_bytes())
    src = lambda r: r["id"].split("__")[0]  # noqa: E731
    utts = sorted({src(r) for r in d0["swap"]} | {r["utt"] for r in d0["pros"]})
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(utts), len(utts))
        cnt = np.bincount(pick, minlength=len(utts))
        w = dict(zip(utts, cnt))
        ms = []
        for d in (d0, d1):
            dd = {"swap": [r for r in d["swap"] for _ in range(w.get(src(r), 0))],
                  "pros": [r for r in d["pros"] for _ in range(w.get(r["utt"], 0))]}
            ms.append(_metrics_of(dd))
        out.append({k: ms[1][k] - ms[0][k] for k in ms[0]})
    return out


def _unit_probs(row):
    """Per evaluation unit (whitespace token) of a swap row: the lowest word P inside it."""
    out = []
    for u in row["words"]:
        a, b = u["span"]
        ps = [p for (s, e), p in zip(row["spans"], row["prob"]) if s < b and e > a]
        out.append(min(ps) if ps else np.nan)
    return out


def metrics(tag):
    return _metrics_of(pickle.loads((WORK / f"guards_{tag}.pkl").read_bytes()))


def _metrics_of(d):
    from bench.metrics import roc_auc

    m = {}
    nat = np.array([p for r in d["swap"] if r["variant"] == "orig" for p in _unit_probs(r)])
    nat = nat[np.isfinite(nat)]
    thr = np.quantile(nat, 0.05)
    m["swap_native_below_0.6"] = float((nat < 0.6).mean())
    for v in ("swap_near", "swap_far"):
        err = np.array([_unit_probs(r)[r["k"]] for r in d["swap"] if r["variant"] == v])
        err = err[np.isfinite(err)]
        y = np.r_[np.ones(len(err)), np.zeros(len(nat))]
        m[f"{v}_auc"] = roc_auc(y, 1 - np.r_[err, nat])
        m[f"{v}_hit5"] = float((err < thr).mean())
    P = d["pros"]
    clean = np.array([p for r in P if r["ref"] == "clean" for p in r["prob"]])
    m["native_accept"] = float((clean >= 0.5).mean())
    m["native_below_0.6"] = float((clean < 0.6).mean())
    t5 = np.quantile(clean, 0.05)
    for tag2 in ("stress", "sound"):
        err = np.array([p for r in P if r["ref"] == tag2 for p, w in zip(r["prob"], r["wrong"]) if w])
        y = np.r_[np.ones(len(err)), np.zeros(len(clean))]
        m[f"{tag2}_auc"] = roc_auc(y, 1 - np.r_[err, clean])
        m[f"{tag2}_hit5"] = float((err < t5).mean())
        m[f"n_{tag2}"] = int(len(err))
    m["n_native"] = int(len(clean))
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "collect-test", "report"])
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--tag")
    ap.add_argument("--boot", type=int, default=0, help="paired bootstrap of the last two tags")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(a.tag)
    elif a.cmd == "collect-test":
        collect_test(a.tag)
    else:
        ms = {t: metrics(t) for t in a.tags}
        bd = boot_delta(a.tags[-2], a.tags[-1], a.boot) if a.boot else None
        res = {}
        for k in ms[a.tags[0]]:
            line = f"{k:24s} " + "  ".join(f"{ms[t][k]:.4f}" for t in a.tags)
            res[k] = {t: ms[t][k] for t in a.tags}
            if bd and not k.startswith("n_"):
                v = np.array([b[k] for b in bd])
                d = ms[a.tags[-1]][k] - ms[a.tags[-2]][k]
                p = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
                line += f"   delta {d:+.4f} [{np.quantile(v, .025):+.4f}, {np.quantile(v, .975):+.4f}] p={p:.3f}"
                res[k]["delta_ci"] = [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
                res[k]["p"] = float(p)
            print(line)
        if a.json:
            from pathlib import Path

            Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
