"""Round 20 (Mandarin): held-out rows, three arms, each row run ONCE after the candidate is frozen.

Arms (environment switches, all in one process, shared scorer):
  off  PRONUNCIADA_IDENTITY_ZH=0                                   (no identity rule)
  r19  PRONUNCIADA_IDENTITY_ZH=1 PRONUNCIADA_IDENTITY_ZH_R20=0     (current main)
  new  PRONUNCIADA_IDENTITY_ZH=1 PRONUNCIADA_IDENTITY_ZH_R20=1     (round-20 rule)

    python -m bench.round20_mandarin.heldout fleurs-collect [--part k/n]   # fleurs-swap TEST + FLEURS TEST protocol
    python -m bench.round20_mandarin.heldout fleurs                        # tables (Azure from cached responses)
    python -m bench.round20_mandarin.heldout ompal-collect [--part k/n]    # OMPAL TEST
    python -m bench.round20_mandarin.heldout ompal
    python -m bench.round20_mandarin.heldout cvreal-collect                # cv-real TEST half
    python -m bench.round20_mandarin.heldout cvreal

FLEURS: the scoreboard's tone-swap rows (bench.eval_swap: AUC, hit@5%FA, right_word; paired against
Azure's cached responses, utterance-clustered bootstrap) and the rollout's FLEURS-test rows
(bench.round19_rollout.heldout: natives below 0.6, near / far / tone, sound errors), with paired
arm-vs-arm differences. OMPAL: word AUC any / tone-only / consonant-vowel (bench.eval_ompal unit
definitions), paired speaker bootstrap (2,000 draws). cv-real: as the rollout (speaker bootstrap).
No Azure request is ever sent.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.round20_mandarin.collect import OUT

LANG = "zh"
ARMS = {"off": {"PRONUNCIADA_IDENTITY_ZH": "0", "PRONUNCIADA_IDENTITY_ZH_R20": "1"},
        "r19": {"PRONUNCIADA_IDENTITY_ZH": "1", "PRONUNCIADA_IDENTITY_ZH_R20": "0"},
        "new": {"PRONUNCIADA_IDENTITY_ZH": "1", "PRONUNCIADA_IDENTITY_ZH_R20": "1"}}
W = OUT / LANG / "heldout"
TABLES = Path("results/tables")


def _arm(name):
    os.environ.update(ARMS[name])


def _threads():
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    for v in ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION", "AZURE_SPEECH_ENDPOINT"):
        os.environ.pop(v, None)


def _parts(part):
    return (int(x) for x in part.split("/")) if part else (0, 1)


# ----------------------------------------------------------------------------- FLEURS test

def cmd_fleurs_collect(part):
    _threads()
    from bench.corrupt import NotApplicable, corrupt
    from bench.eval_swap import sentence_result_output
    from bench.pipeline.common import stable_seed
    from bench.pipeline.evalctl import inventory
    from bench.round19_rollout.collect import jobs
    from pronunciada import Scorer

    k_part, n_part = _parts(part)
    d = W / "fleurs"
    d.mkdir(parents=True, exist_ok=True)
    js = jobs(LANG, "test")
    sc = Scorer(LANG, threads=2)
    inv = inventory(LANG)
    for k in range(0, len(js), 40):
        if (k // 40) % n_part != k_part or (d / f"{k // 40:05d}.pkl").exists():
            continue
        rows = []
        for iid, audio, text, meta in js[k:k + 40]:
            wrong = None
            if isinstance(text, tuple):
                _, tgt, sid, kind, tag = text
                try:
                    cor = corrupt(sc.g2p(tgt), [kind], inv, n_errors=2, seed=stable_seed(sid, kind, tag), strict=False)
                except NotApplicable:
                    continue
                if not cor.errors:
                    continue
                text, wrong = cor.ref, [bool(x) for x in cor.word_wrong]
            row = {"id": iid, "meta": meta, "wrong": wrong}
            for a in ARMS:
                _arm(a)
                try:
                    res = sc.score(audio, text)
                except ValueError as e:
                    row[f"out_{a}"] = {"status": f"error: {e}", "words": []}
                    continue
                spans = [tuple(w.char_span) for w in res._pronunciada["ref"].words]  # noqa: SLF001
                row["spans"] = spans
                row[f"prob_{a}"] = np.array([w.prob for w in res.words], np.float64)
                row[f"out_{a}"] = sentence_result_output(res, spans)
            rows.append(row)
        (d / f"{k // 40:05d}.pkl").write_bytes(pickle.dumps(rows))
        print("fleurs", k + 40, len(js), flush=True)


def _load(sub):
    out = []
    for p in sorted((W / sub).glob("*.pkl")):
        out += pickle.loads(p.read_bytes())
    return out


def _rollout_rows(rows, a, b):
    """Rows in the shape of bench.round19_rollout.heldout (prob_off = arm a, prob = arm b)."""
    return [{"id": r["id"], "meta": r["meta"], "wrong": r["wrong"], "spans": r["spans"],
             "prob_off": r[f"prob_{a}"], "prob": r[f"prob_{b}"]} for r in rows if f"prob_{a}" in r and f"prob_{b}" in r]


def cmd_fleurs():
    import bench.round19_rollout.collect as CO
    from bench.data import load
    from bench.eval_swap import azure_outputs, evaluate, paired
    from bench.round19_rollout.heldout import cmd_fleurs as rollout_fleurs

    rows = _load("fleurs")
    recs = [r for r in load("fleurs-swap", split="test") if r["lang"] == LANG]
    ids = {r["id"] for r in recs}
    outs = {a: {r["id"]: r[f"out_{a}"] for r in rows if r["id"] in ids} for a in ARMS}
    az = azure_outputs(recs)
    res = {"n_rows": len(rows), "swap": {}, "paired_vs_azure": {}, "paired_arms": {}, "rollout": {}}
    keys = ("tone:auc", "tone:hit@5%FA", "tone:right_word", "swap_near:auc", "swap_near:hit@5%FA",
            "swap_far:auc", "swap_far:right_word", "native:fa<60")
    for a in ARMS:
        m = evaluate(recs, outs[a])[LANG]["metrics"]
        res["swap"][a] = {k: m[k] for k in keys if k in m}
        res["paired_vs_azure"][a] = paired(recs, outs[a], az)[LANG]
    res["swap"]["azure"] = {k: v for k, v in evaluate(recs, az)[LANG]["metrics"].items() if k in keys}
    for a, b in (("off", "new"), ("r19", "new"), ("off", "r19")):
        res["paired_arms"][f"{b}-{a}"] = paired(recs, outs[b], outs[a])[LANG]
    keep = CO.load
    try:
        for a, b in (("off", "new"), ("r19", "new"), ("off", "r19")):
            rr = _rollout_rows(rows, a, b)
            CO.load = lambda lang_, split, rr=rr: rr  # noqa: E731
            tmp = W / f"rollout_{a}_{b}.json"
            print(f"--- rollout rows {a} -> {b}")
            rollout_fleurs(LANG, "test", str(tmp))
            res["rollout"][f"{b}-{a}"] = json.loads(tmp.read_text())
    finally:
        CO.load = keep
    for a in ARMS:
        s = res["swap"][a]
        pv = res["paired_vs_azure"][a]
        print(f"{a:4s} tone AUC {s['tone:auc']['value']:.4f} hit5 {s['tone:hit@5%FA']['value']:.3f} right "
              f"{s['tone:right_word']['value']:.3f} | vs Azure right_word d {pv['tone:right_word']['delta']:+.3f} "
              f"[{pv['tone:right_word']['lo']:+.3f}, {pv['tone:right_word']['hi']:+.3f}] p {pv['tone:right_word']['p_two_sided']:.3f}"
              f" | native fa<60 {s['native:fa<60']['value']:.4f} near {s['swap_near:auc']['value']:.4f} far "
              f"{s['swap_far:auc']['value']:.4f}")
    for k, v in res["paired_arms"].items():
        print(k, {m: (round(x["delta"], 4), round(x["p_two_sided"], 3)) for m, x in v.items() if isinstance(x, dict)})
    (TABLES / "round20_mandarin_heldout_fleurs.json").write_text(json.dumps(res, indent=1, default=float))


# ----------------------------------------------------------------------------- OMPAL test

def cmd_ompal_collect(part):
    _threads()
    from bench import eval_ompal as EO
    from pronunciada import Scorer

    k_part, n_part = _parts(part)
    d = W / "ompal"
    d.mkdir(parents=True, exist_ok=True)
    recs = EO.records("test")
    sc = Scorer(LANG, threads=2)
    for k in range(0, len(recs), 40):
        if (k // 40) % n_part != k_part or (d / f"{k // 40:05d}.pkl").exists():
            continue
        rows = []
        for r in recs[k:k + 40]:
            row = {"id": r["id"]}
            for a in ARMS:
                _arm(a)
                try:
                    row[a] = EO.summarize(sc.score(r["audio"], r["text"]), r["text"])
                except ValueError as e:
                    row[a] = {"status": "error", "error": str(e)[:200]}
            rows.append(row)
        (d / f"{k // 40:05d}.pkl").write_bytes(pickle.dumps(rows))
        print("ompal", k + 40, len(recs), flush=True)


def cmd_ompal(n_boot=2000):
    from bench import eval_ompal as EO
    from bench.metrics import roc_auc

    recs = EO.records("test")
    rows = {r["id"]: r for r in _load("ompal")}
    T = {a: EO.unit_table(recs, {i: r[a] for i, r in rows.items()}, {})["cols"] for a in ARMS}
    c = T["off"]
    spk = c["spk"]
    cls = {"any": c["any_wrong"].astype(bool), "tone_only": c["tone_only"].astype(bool),
           "cv": c["seg_wrong"].astype(bool)}
    corr = c["correct"].astype(bool)

    def auc(score, pos):
        m = pos | corr
        return roc_auc(pos[m].astype(int), -score[m])

    boots = EO.boot_indices(spk, n_boot, 0)
    res = {"n_units": int(len(spk)), "n_speakers": int(len(set(spk.tolist())))}
    for name, pos in cls.items():
        row = {a: auc(T[a]["ours_word"], pos) for a in ARMS}
        for a, b in (("off", "new"), ("r19", "new"), ("off", "r19")):
            d = []
            for ix in boots:
                p, cc = pos[ix], corr[ix]
                m = p | cc
                if p[m].any() and (~p[m]).any():
                    d.append(roc_auc(p[m].astype(int), -T[b]["ours_word"][ix][m])
                             - roc_auc(p[m].astype(int), -T[a]["ours_word"][ix][m]))
            d = np.array(d)
            row[f"{b}-{a}"] = {"delta": row[b] - row[a], "ci": list(np.quantile(d, [.025, .975])),
                               "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
        res[name] = row
        print(f"OMPAL word AUC {name:9s} off {row['off']:.4f} r19 {row['r19']:.4f} new {row['new']:.4f} | "
              + " ".join(f"{k} {v['delta']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] p={v['p']:.3f}"
                         for k, v in row.items() if isinstance(v, dict)))
    (TABLES / "round20_mandarin_heldout_ompal.json").write_text(json.dumps(res, indent=1, default=float))


# ----------------------------------------------------------------------------- cv-real test half

def cmd_cvreal_collect():
    _threads()
    from bench.round19_rollout.heldout import cvreal_items
    from pronunciada import Scorer

    sc = Scorer(LANG, threads=2)
    rows = {a: {} for a in ARMS}
    for r in cvreal_items(LANG, "test"):
        for a in ARMS:
            _arm(a)
            try:
                res = sc.score(r["wav"], r["sentence"])
                wp = [w.prob for w in res.words]
                rows[a][r["id"]] = (res.prob, float(np.mean(wp)), float(np.min(wp)))
            except ValueError:
                rows[a][r["id"]] = (0.0, 0.0, 0.0)
    W.mkdir(parents=True, exist_ok=True)
    (W / "cvreal_test.pkl").write_bytes(pickle.dumps(rows))


def cmd_cvreal(n_boot=1000):
    from bench.metrics import roc_auc
    from bench.round19_rollout.heldout import cvreal_items

    its = cvreal_items(LANG, "test")
    R = pickle.loads((W / "cvreal_test.pkl").read_bytes())
    res = {}
    for test in sorted({r["test"] for r in its}):
        sel = [r for r in its if r["test"] == test and r["id"] in R["off"]]
        y = np.array([r["label"] for r in sel])
        if len(set(y.tolist())) < 2:
            continue
        spk = sorted({r["speaker"] for r in sel})
        by = {s: [i for i, r in enumerate(sel) if r["speaker"] == s] for s in spk}
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by[spk[j]] for j in rng.integers(0, len(spk), len(spk))]) for _ in range(n_boot)]
        for k, name in enumerate(("sentence prob", "word mean", "word min")):
            S = {a: np.array([1 - R[a][r["id"]][k] for r in sel]) for a in ARMS}
            row = {a: roc_auc(y, S[a]) for a in ARMS}
            for a, b in (("off", "new"), ("r19", "new"), ("off", "r19")):
                d = np.array([roc_auc(y[ix], S[b][ix]) - roc_auc(y[ix], S[a][ix]) for ix in draws
                              if len(set(y[ix].tolist())) == 2])
                row[f"{b}-{a}"] = {"delta": row[b] - row[a], "ci": list(np.quantile(d, [.025, .975])),
                                   "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
            row["n"] = len(sel)
            res[f"{test} {name}"] = row
            print(f"cv-real {test} {name:13s} off {row['off']:.4f} r19 {row['r19']:.4f} new {row['new']:.4f} | "
                  + " ".join(f"{kk} {v['delta']:+.4f} p={v['p']:.3f}" for kk, v in row.items() if isinstance(v, dict)))
    (TABLES / "round20_mandarin_heldout_cvreal.json").write_text(json.dumps(res, indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fleurs-collect", "fleurs", "ompal-collect", "ompal", "cvreal-collect", "cvreal"])
    ap.add_argument("--part")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    if a.workers > 1:
        import subprocess
        import sys

        ps = [subprocess.Popen([sys.executable, "-m", "bench.round20_mandarin.heldout", a.cmd, "--part",
                                f"{k}/{a.workers}"]) for k in range(a.workers)]
        if any(p.wait() for p in ps):
            raise SystemExit("worker failed")
        return
    {"fleurs-collect": lambda: cmd_fleurs_collect(a.part), "fleurs": cmd_fleurs,
     "ompal-collect": lambda: cmd_ompal_collect(a.part), "ompal": cmd_ompal,
     "cvreal-collect": cmd_cvreal_collect, "cvreal": cmd_cvreal}[a.cmd]()


if __name__ == "__main__":
    main()
