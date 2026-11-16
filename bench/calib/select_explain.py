"""Round 3: select the explain-away second pass (threshold, max passes) on FLEURS VALIDATION.

    python -m bench.calib.select_explain [--n 60] [--workers 5]

Per language n random validation clips; references: correct, word_swap, phone_sub_near,
phone_sub_far (2 errors, fixed seeds). First pass = Scorer with explain-away off; every grid point
re-runs only Scorer.explain_away on the cached first pass (same posteriors). Metrics on the
sounds-based word probability at s = 0.5: word AUC (corrupted vs untouched words of the corrupted
references), right-word rate (lowest word of a corrupted reference is a corrupted one), native
word acceptance (P >= 0.5 on the correct references), hit @ 5 % native flags, added latency.
Rule (fixed before running): maximise the right-word rate subject to native acceptance and word
AUC each not lower than without the pass by more than 0.002; ties -> fewer passes, higher
threshold. Writes results/tables/explain_away_dev.json and the "explain_away" entry of
pronunciada/data/calib/config.json. The calibrators are not refitted on second-pass features.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

KINDS = ("word_swap", "phone_sub_near", "phone_sub_far")
GRID = [(t, m) for t in (0.1, 0.2, 0.3, 0.5) for m in (1, 2, 3)]
LANGS = ("en", "zh", "es", "it", "ja", "ko", "fr", "de", "pt", "ru")


def _p(c):
    from pronunciada.calibrate import blend_logit, sigmoid

    return sigmoid(blend_logit(c["w_zl"], np.minimum(c["w_zs"], c["w_zl"]), 0.5))


def run_lang(args):
    lang, n = args
    import os

    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.common import iter_parquet_audio, records, stable_seed
    from bench.pipeline.evalctl import inventory
    from pronunciada.scorer import Scorer

    sc = Scorer(lang, threads=1, explain_away=False, prosody=False)
    recs = records("fleurs-validation", lang=lang)
    rng = np.random.default_rng(7)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), min(n, len(recs)), replace=False))]
    by = {int(r["audio"].split(":")[2]): r for r in recs}
    inv = inventory(lang)
    rows = []          # (utt, ref kind, wrong[], {config: p[]}, {config: ms})
    for i, wav in iter_parquet_audio(recs[0]["audio"].split(":")[1], set(by)):
        rec = by[i]
        try:
            res = sc.score((wav, 16000), rec["text"])
        except ValueError:
            continue
        ref = res._pronunciada["ref"]  # noqa: SLF001
        refs = [("clean", ref, np.zeros(len(ref.words), bool))]
        for k in KINDS:
            try:
                cor = corrupt(ref, [k], inv, n_errors=2, seed=stable_seed(rec["id"], k, "explain-v1"), strict=False)
            except NotApplicable:
                continue
            if cor.errors:
                refs.append((k, cor.ref, np.asarray(cor.word_wrong)))
        from pronunciada.audio import load_audio

        lpa = sc._posteriors(load_audio((wav, 16000)), {})  # noqa: SLF001  (cached by score())
        for kind, r, wrong in refs:
            c = sc.score((wav, 16000), r)._pronunciada  # noqa: SLF001
            ps, ms = {"off": _p(c)}, {"off": 0.0}
            for thr, mp in GRID:
                t = time.perf_counter()
                c2 = sc.explain_away(lpa, c, c["dur"], thr, mp)
                ms[f"{thr}/{mp}"] = (time.perf_counter() - t) * 1e3
                ps[f"{thr}/{mp}"] = _p(c2)
            rows.append((rec["id"], kind, wrong, ps, ms))
    return lang, rows


def metrics(rows, cfg):
    from bench.metrics import roc_auc

    nat = np.concatenate([p[cfg] for _, k, _, p, _ in rows if k == "clean"])
    cor = [(w, p[cfg]) for _, k, w, p, _ in rows if k != "clean"]
    y = np.concatenate([w for w, _ in cor]).astype(int)
    s = np.concatenate([p for _, p in cor])
    thr = np.quantile(nat, 0.05)
    right = [bool(w[int(np.argmin(p))]) for w, p in cor if w.any() and (~w).any()]
    ms = [m[cfg] for *_, m in rows]
    return {"native_accept": float(np.mean(nat >= 0.5)), "word_auc": roc_auc(y, 1 - s),
            "right_word": float(np.mean(right)), "hit_at_5pct": float(np.mean(s[y == 1] < thr)),
            "added_ms_median": float(np.median(ms)), "added_ms_p95": float(np.percentile(ms, 95))}


def main():
    from concurrent.futures import ProcessPoolExecutor

    from bench.calib.common import save_json, ship_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        data = dict(ex.map(run_lang, [(l, a.n) for l in LANGS]))
    allrows = [r for v in data.values() for r in v]
    cfgs = ["off"] + [f"{t}/{m}" for t, m in GRID]
    out = {"pooled": {c: metrics(allrows, c) for c in cfgs},
           "by_lang": {l: {c: metrics(v, c) for c in cfgs} for l, v in data.items()}}
    base = out["pooled"]["off"]
    ok = [c for c in cfgs[1:] if out["pooled"][c]["native_accept"] >= base["native_accept"] - 0.002
          and out["pooled"][c]["word_auc"] >= base["word_auc"] - 0.002]
    best = max(ok, key=lambda c: (round(out["pooled"][c]["right_word"], 4), -int(c.split("/")[1]),
                                  float(c.split("/")[0]))) if ok else None
    if best and out["pooled"][best]["right_word"] <= base["right_word"]:
        best = None
    out["chosen"] = best
    for c in cfgs:
        print(c, {k: round(v, 4) for k, v in out["pooled"][c].items()}, flush=True)
    print("chosen", best)
    save_json("explain_away_dev.json", out)
    p = ship_dir() / "config.json"
    cfg = json.loads(p.read_text())
    cfg["explain_away"] = ({"enabled": True, "threshold": float(best.split("/")[0]), "max_passes": int(best.split("/")[1])}
                           if best else {"enabled": False})
    p.write_text(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
