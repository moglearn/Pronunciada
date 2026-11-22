"""Round 19 rollout: held-out rows, run once per language after its candidate is frozen.

    python -m bench.round19_rollout.heldout priv --tag fr_off          # private cache root (symlinks)
    PRONUNCIADA_CACHE=<root> PRONUNCIADA_IDENTITY_FR=0 python -m bench.third_party.lingolix score --only tier1_fr
    python -m bench.round19_rollout.heldout lingolix --lang fr          # paired off / on, both rule sets
    python -m bench.round19_rollout.heldout fleurs --lang fr            # FLEURS test rows (collect set "test")
    python -m bench.round19_rollout.heldout cvreal-collect --lang fr    # cv-real TEST half, off and on
    python -m bench.round19_rollout.heldout cvreal --lang fr

Private cache roots: $PRONUNCIADA_CACHE/round19/rollout/priv/<tag>/ hold symlinks to every shared
folder except third_party/lingolix/responses and audit, so nothing shared is overwritten.
Intervals: 1000 bootstrap draws over clips (Lingolix, as report.py; FLEURS test, by source clip) or
speakers (cv-real); paired differences with two-sided p.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from bench.round19_rollout.sources import CACHE, ROOT, work

PRIV = ROOT / "priv"


def cmd_priv(tag: str, audit_from: str | None = None) -> Path:
    root = PRIV / tag
    root.mkdir(parents=True, exist_ok=True)
    for p in CACHE.iterdir():
        if p.name in ("third_party", "audit", "round19") or p.name.endswith(".lock"):
            continue
        q = root / p.name
        if not q.exists():
            q.symlink_to(p)
    tp = root / "third_party"
    tp.mkdir(exist_ok=True)
    for p in (CACHE / "third_party").iterdir():
        if p.name != "lingolix" and not (tp / p.name).exists():
            (tp / p.name).symlink_to(p)
    lx = tp / "lingolix"
    lx.mkdir(exist_ok=True)
    for p in (CACHE / "third_party" / "lingolix").iterdir():
        if p.name.startswith("responses") or (lx / p.name).exists():
            continue
        (lx / p.name).symlink_to(p)
    (root / "audit").mkdir(exist_ok=True)
    print(root)
    return root


def _responses(tag: str, test: str) -> dict:
    p = PRIV / tag / "third_party" / "lingolix" / "responses" / "pronunciada" / f"{test}.jsonl"
    out = {}
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            out[r["id"]] = r
    return out


def paired(units0, units1, fns, n=1000, seed=0):
    from bench.third_party import lingolix as L  # noqa: F401

    rng = np.random.default_rng(seed)
    N = len(units0)
    out = {k: [] for k in fns}
    for _ in range(n):
        ix = rng.integers(0, N, N)
        u0, u1 = [units0[i] for i in ix], [units1[i] for i in ix]
        for k, f in fns.items():
            out[k].append((f(u0), f(u1)))
    return {k: np.array(v) for k, v in out.items()}


def _lab_sc(us):
    return [l for u in us for l in u[0]], [s for u in us for s in u[1]]


def cmd_lingolix(lang: str, off: str, on: str, out_json: str | None):
    from bench.third_party import lingolix as L

    T = L.tests()[f"tier1_{lang}"]
    main_unv = {it["id"]: it.get("unverified", []) for it in L.load_manifest("repo_main_f8750aa", f"tier1/manifest_{lang}.jsonl")}
    R0, R1 = _responses(off, f"tier1_{lang}"), _responses(on, f"tier1_{lang}")
    res = {}
    for rules, kw in (("v2026.10", {}), ("main", {"exclude_unverified": main_unv})):
        m0, m1 = L.metrics(T, R0, **kw), L.metrics(T, R1, **kw)
        c0, c1 = L.collect(T, R0, **kw)["units"], L.collect(T, R1, **kw)["units"]
        key = lambda u: u[2]["id"]  # noqa: E731
        c0.sort(key=key)
        c1.sort(key=key)
        assert [key(u) for u in c0] == [key(u) for u in c1]
        fns = {"auc": lambda us: L.roc_auc(*_lab_sc(us)), "fa80": lambda us: L.fpr_at_recall(*_lab_sc(us), 0.8)}
        B = paired(c0, c1, fns)
        row = {}
        for k, mk in (("auc", "roc_auc"), ("fa80", "fpr_at_recall_80")):
            d = B[k][:, 1] - B[k][:, 0]
            better = 1 if k == "auc" else -1
            p = min(1.0, 2 * min(np.mean(d * better <= 0), np.mean(d * better >= 0)))
            row[k] = {"off": m0[mk], "on": m1[mk], "off_ci": list(np.quantile(B[k][:, 0], [.025, .975])),
                      "on_ci": list(np.quantile(B[k][:, 1], [.025, .975])), "delta": m1[mk] - m0[mk],
                      "delta_ci": list(np.quantile(d, [.025, .975])), "p": float(p)}
        row["buckets"] = {"off": m0["by_bucket"], "on": m1["by_bucket"]}
        row["verdicts"] = {"off": m0["verdicts"], "on": m1["verdicts"]}
        row["n_failed"] = [m0["n_failed"], m1["n_failed"]]
        res[rules] = row
        print(f"{lang} {rules}: AUC {row['auc']['off']:.4f} -> {row['auc']['on']:.4f} [{row['auc']['on_ci'][0]:.3f}, "
              f"{row['auc']['on_ci'][1]:.3f}]  d {row['auc']['delta']:+.4f} [{row['auc']['delta_ci'][0]:+.4f}, "
              f"{row['auc']['delta_ci'][1]:+.4f}] p={row['auc']['p']:.3f} | FA@80 {100 * row['fa80']['off']:.2f} -> "
              f"{100 * row['fa80']['on']:.2f} % [{100 * row['fa80']['on_ci'][0]:.2f}, {100 * row['fa80']['on_ci'][1]:.2f}]  d "
              f"{100 * row['fa80']['delta']:+.2f} [{100 * row['fa80']['delta_ci'][0]:+.2f}, {100 * row['fa80']['delta_ci'][1]:+.2f}] "
              f"p={row['fa80']['p']:.3f}")
        print("   buckets off", {k: round(v, 3) for k, v in m0["by_bucket"].items()})
        print("   buckets on ", {k: round(v, 3) for k, v in m1["by_bucket"].items()})
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1, default=float))


# ----------------------------------------------------------------------------- FLEURS test

def _guard_dict(rows, which):
    d = {"swap": [], "pros": []}
    for r in rows:
        m = r["meta"]
        p = list(r["prob_off"] if which == "off" else r["prob"])
        if m["kind"] == "swap":
            d["swap"].append({"id": r["id"], "variant": m["variant"], "k": m["k"], "spans": r["spans"],
                              "words": m["units"], "prob": p, "utt": m["utt"]})
        else:
            d["pros"].append({"utt": m["utt"], "ref": m["ref"], "prob": p,
                              "wrong": r["wrong"] if r["wrong"] is not None else [False] * len(p)})
    return d


def cmd_fleurs(lang: str, split: str, out_json: str | None, n_boot=1000):
    from bench.round19.guards_dev import _metrics_of as _m
    from bench.round19.guards_dev import _unit_probs
    from bench.round19_rollout.collect import load

    def _metrics_of(d):
        """The German guard rows, plus planted tone swaps (Mandarin fleurs-swap "tone" variant) when present."""
        m = _m(d)
        tone = np.array([_unit_probs(r)[r["k"]] for r in d["swap"] if r["variant"] == "tone"])
        if len(tone):
            from bench.metrics import roc_auc

            nat = np.array([p for r in d["swap"] if r["variant"] == "orig" for p in _unit_probs(r)])
            nat, tone = nat[np.isfinite(nat)], tone[np.isfinite(tone)]
            m["swap_tone_auc"] = roc_auc(np.r_[np.ones(len(tone)), np.zeros(len(nat))], 1 - np.r_[tone, nat])
            m["swap_tone_hit5"] = float((tone < np.quantile(nat, 0.05)).mean())
        return m

    rows = load(lang, split)
    d0, d1 = _guard_dict(rows, "off"), _guard_dict(rows, "on")
    m0, m1 = _metrics_of(d0), _metrics_of(d1)
    utts = sorted({r["utt"] for r in d0["swap"]} | {r["utt"] for r in d0["pros"]})
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(n_boot):
        cnt = dict(zip(utts, np.bincount(rng.integers(0, len(utts), len(utts)), minlength=len(utts))))
        ms = []
        for d in (d0, d1):
            dd = {"swap": [r for r in d["swap"] for _ in range(cnt.get(r["utt"], 0))],
                  "pros": [r for r in d["pros"] for _ in range(cnt.get(r["utt"], 0))]}
            ms.append(_metrics_of(dd))
        draws.append({k: ms[1][k] - ms[0][k] for k in ms[0]})
    res = {}
    for k in m0:
        res[k] = {"off": m0[k], "on": m1[k]}
        if not k.startswith("n_"):
            v = np.array([x[k] for x in draws])
            res[k].update(delta=m1[k] - m0[k], delta_ci=list(np.quantile(v, [.025, .975])),
                          p=float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))))
            print(f"{k:24s} {m0[k]:.4f} -> {m1[k]:.4f}  d {m1[k] - m0[k]:+.4f} [{res[k]['delta_ci'][0]:+.4f}, "
                  f"{res[k]['delta_ci'][1]:+.4f}] p={res[k]['p']:.3f}")
        else:
            print(f"{k:24s} {m0[k]} {m1[k]}")
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1, default=float))


# ----------------------------------------------------------------------------- cv-real test half

def cvreal_items(lang, half):
    from bench.cv_build import OUT_DIR, audio_path

    from bench.round19_rollout.sources import locale

    out = []
    for l in (OUT_DIR / f"{locale(lang)}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["half"] == half:
            r["wav"] = str(audio_path(locale(lang), r["path"].rsplit(".", 1)[0]))
            out.append(r)
    return out


def cmd_cvreal_collect(lang, half):
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    from pronunciada import Scorer

    sc = Scorer(lang, threads=2)
    env = f"PRONUNCIADA_IDENTITY_{lang.upper()}"
    rows = {"off": {}, "on": {}}
    for r in cvreal_items(lang, half):
        for tag, val in (("off", "0"), ("on", "1")):
            os.environ[env] = val
            try:
                res = sc.score(r["wav"], r["sentence"])
                wp = [w.prob for w in res.words]
                rows[tag][r["id"]] = (res.prob, float(np.mean(wp)), float(np.min(wp)))
            except ValueError:
                rows[tag][r["id"]] = (0.0, 0.0, 0.0)
    (work(lang) / f"cvreal_{half}.pkl").write_bytes(pickle.dumps(rows))


def cmd_cvreal(lang, half, out_json, n_boot=1000):
    from bench.metrics import roc_auc

    its = cvreal_items(lang, half)
    R = pickle.loads((work(lang) / f"cvreal_{half}.pkl").read_bytes())
    res = {}
    for test in sorted({r["test"] for r in its}):
        sel = [r for r in its if r["test"] == test and r["id"] in R["off"]]
        spk = sorted({r["speaker"] for r in sel})
        by = {s: [i for i, r in enumerate(sel) if r["speaker"] == s] for s in spk}
        y = np.array([r["label"] for r in sel])
        if len(set(y.tolist())) < 2:
            continue
        rng = np.random.default_rng(0)
        draws = [np.concatenate([by[spk[j]] for j in rng.integers(0, len(spk), len(spk))]) for _ in range(n_boot)]
        for k, name in enumerate(("sentence prob", "word mean", "word min")):
            S = [np.array([1 - R[t][r["id"]][k] for r in sel]) for t in ("off", "on")]
            a = [roc_auc(y, s) for s in S]
            d = np.array([roc_auc(y[ix], S[1][ix]) - roc_auc(y[ix], S[0][ix]) for ix in draws if len(set(y[ix].tolist())) == 2])
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            res[f"{test} {name}"] = {"off": a[0], "on": a[1], "delta": a[1] - a[0],
                                     "delta_ci": list(np.quantile(d, [.025, .975])), "p": float(p), "n": len(sel)}
            print(f"Test {test} {name:13s} {a[0]:.4f} -> {a[1]:.4f}  d {a[1] - a[0]:+.4f} "
                  f"[{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}] p={p:.3f}")
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["priv", "lingolix", "fleurs", "cvreal-collect", "cvreal"])
    ap.add_argument("--lang")
    ap.add_argument("--tag")
    ap.add_argument("--off")
    ap.add_argument("--on")
    ap.add_argument("--split", default="test")
    ap.add_argument("--half", default="test")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "priv":
        cmd_priv(a.tag)
    elif a.cmd == "lingolix":
        cmd_lingolix(a.lang, a.off or f"{a.lang}_off", a.on or f"{a.lang}_on", a.json)
    elif a.cmd == "fleurs":
        cmd_fleurs(a.lang, a.split, a.json)
    elif a.cmd == "cvreal-collect":
        cmd_cvreal_collect(a.lang, a.half)
    else:
        cmd_cvreal(a.lang, a.half, a.json)


if __name__ == "__main__":
    main()
