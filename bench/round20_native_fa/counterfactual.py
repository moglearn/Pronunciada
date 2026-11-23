"""Counterfactual re-scoring: each word against each of its alternative pronunciations (variants.py).

    python -m bench.round20_native_fa.counterfactual --lang en [--max-score 0.9]

Words re-scored: every changed word, and every correct word whose benchmark score is below
--max-score (best-of-N can only raise a score, so words above that bound cannot cross the FA@80
threshold, which is far below it in every language). Only prompt tokens that are exactly one
front-end word. Per word and variant: the word's probability when the variant is its canonical
form (same audio, same cached posteriors, Scorer.score on the spliced Reference).
Check: the canonical form put through the same splice must give the original probability
(max |diff| reported; it must be 0).
Output: $CACHE/round20/nativefa/cf/<lang>.jsonl (resumable by item).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

from bench.round20_native_fa import core as K
from bench.round20_native_fa import variants as VA
from bench.round20_native_fa.rebuild import ROOT


def run(lang: str, max_score: float, n_check: int = 300) -> None:
    from pronunciada import Scorer

    rows = K.word_rows(lang)
    its = {it["id"]: it for it in K.practice_items(lang)}
    want = defaultdict(list)
    for r in rows:
        if len(r["iw"]) == 1 and (r["label"] or r["score"] < max_score):
            want[r["item"]].append(r)
    out_p = ROOT / "cf" / f"{lang}.jsonl"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_p.exists():
        done = {json.loads(l)["item"] for l in out_p.read_text(encoding="utf-8").splitlines() if l.strip()}
    sc = Scorer(lang, threads=2)
    g2p = sc.g2p
    t0, n, checks, maxdiff = time.time(), 0, 0, 0.0
    order = sorted(want, key=lambda k: its[k]["clip"])
    with open(out_p, "a", encoding="utf-8") as f:
        for iid in order:
            if iid in done:
                continue
            it = its[iid]
            res = sc.score(it["audio"], it["target"])
            c = res._pronunciada  # noqa: SLF001
            ref, ev = c["ref"], c["ev"]
            recs = []
            for r in want[iid]:
                wi = r["iw"][0]
                base = float(res.words[wi].prob)
                rec = {"item": iid, "k": r["k"], "wi": wi, "label": r["label"], "base": round(base, 5),
                       "canon": " ".join(p.ipa for p in ref.phones[ref.words[wi].phones[0]:ref.words[wi].phones[1]]),
                       "var": []}
                if checks < n_check:
                    nr = VA.splice(lang, g2p, ref, wi, [(v["ipa"], v["stress"] if v["nuc"] else None)
                                                         for v in VA.word_view(ref, wi)])
                    if nr is not None:
                        same = float(sc.score(it["audio"], nr).words[wi].prob)
                        maxdiff = max(maxdiff, abs(same - base))
                        checks += 1
                for kind, toks, swaps in VA.all_variants(lang, g2p, ref, wi, ev):
                    try:
                        nr = VA.splice(lang, g2p, ref, wi, toks, swaps)
                        if nr is None:
                            continue
                        p = float(sc.score(it["audio"], nr).words[wi].prob)
                    except Exception as e:  # noqa: BLE001
                        rec.setdefault("errors", []).append(f"{kind}: {type(e).__name__}: {e}")
                        continue
                    rec["var"].append({"kind": kind, "ph": " ".join(t[0] for t in toks), "p": round(p, 5)})
                recs.append(rec)
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                print(lang, n, len(order), f"{time.time() - t0:.0f}s", "checks", checks, "max|diff|", maxdiff, flush=True)
    print(lang, "done", n, "items", f"{time.time() - t0:.0f}s", "canonical splice checks", checks, "max|diff|", maxdiff,
          flush=True)
    (ROOT / "cf" / f"{lang}.check.json").write_text(json.dumps({"checks": checks, "max_abs_diff": maxdiff}))


def load(lang: str) -> dict[tuple[str, int], dict]:
    p = ROOT / "cf" / f"{lang}.jsonl"
    out = {}
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            out[(r["item"], r["k"])] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--max-score", type=float, default=0.9)
    a = ap.parse_args(argv)
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    for l in a.lang.split(","):
        run(l, a.max_score)


if __name__ == "__main__":
    main()
