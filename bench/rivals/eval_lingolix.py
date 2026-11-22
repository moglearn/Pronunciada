"""OpenPronounce 1.1.0 against Pronunciada on the Lingolix pronunciation benchmark (v2026.10).

    python -m bench.rivals.eval_lingolix [--mode default]

Same metric code as Pronunciada's own run (bench/third_party/lingolix.py, the benchmark's report.py
re-implemented): test 1 (planted errors, 200 Common Voice clips per language, en fr es de) word ROC AUC
and FA@80 (share of correct words flagged at the threshold that catches 80 % of the planted errors);
test 2 (500 speechocean762 test clips) word and utterance Pearson with the experts.

OpenPronounce's output is put in the shape the benchmark's adapters read: one entry per whitespace
token of the prompt, score = 1 - p of the word model (min over the OpenPronounce words inside the token,
so "elle-même" is one entry; "not said" words p = max(p, missing probability)), error_type set when the
app reports the word; utterance score = its `score`. Test 2 reuses the speechocean762 run (same clips,
same prompt; bench.rivals.openpronounce used the corpus prompt in sentence case).
Paired differences: Pronunciada - OpenPronounce on the same 1,000 resamples of utterances (test 1, as
report.py resamples) or speakers (test 2), two-sided p.
Writes results/tables/rivals_lingolix.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tables" / "rivals_lingolix.json"
WORD_RE = re.compile(r"\b[\w']+\b")


def op_response(item_id: str, target: str, r: dict) -> dict:
    o = r["out"]
    if "error" in o or o.get("score") is None:
        return {"id": item_id, "vendor": "openpronounce", "error": o.get("error", "no score")}
    from bench.rivals.convert import word_probs

    wp = word_probs(o)
    rep = {int(e["position"]) for e in o.get("errors", [])}
    spans = [(m.start(), m.end()) for m in WORD_RE.finditer(r["text"])]
    words, cur = [], 0
    for tok in target.split():
        a = target.find(tok, cur)
        b = a + len(tok)
        cur = b
        hit = [i for i, (s, e) in enumerate(spans) if s < b and e > a and i in wp]
        if not hit:
            words.append({"text": tok, "score": 0.0, "scored": False, "error_type": "Omission"})
            continue
        words.append({"text": tok, "score": round(1 - max(wp[i] for i in hit), 4), "scored": True,
                      "error_type": "Mispronunciation" if any(i in rep for i in hit) else "None"})
    return {"id": item_id, "vendor": "openpronounce", "words": words,
            "raw": {"NBest": [{"PronunciationAssessment": {"PronScore": float(o["score"])}}]},
            "latency_ms": 1000 * o.get("sec", math.nan)}


def responses(mode: str) -> dict[str, dict]:
    from bench.rivals.openpronounce import read_raw
    from bench.third_party import lingolix as L

    T = L.tests()
    out = {}
    for lang in ("en", "fr", "es", "de"):
        R = read_raw(mode, f"lingolix-t1-{lang}")
        out[f"tier1_{lang}"] = {it["id"]: op_response(it["id"], it["target"], R[it["id"]])
                                for it in T[f"tier1_{lang}"] if it["id"] in R}
    R = read_raw(mode, "speechocean762")
    out["tier2"] = {it["id"]: op_response(it["id"], it["target"], R[f"so762-{it['id']}"])
                    for it in T["tier2"] if f"so762-{it['id']}" in R}
    return out


def paired_t1(manifest, ra, rb, fn) -> dict:
    """fn(labels, scores) on the same utterance resamples for both systems (report.py's resampling)."""
    from bench.third_party import lingolix as L

    ua = {u[2]["id"]: u for u in L.collect(manifest, ra)["units"]}
    ub = {u[2]["id"]: u for u in L.collect(manifest, rb)["units"]}
    ids = [i for i in ua if i in ub]
    rng = random.Random(0)
    d = []
    for _ in range(L.BOOTSTRAP):
        pick = [ids[rng.randrange(len(ids))] for _ in ids]
        va = fn([x for i in pick for x in ua[i][0]], [x for i in pick for x in ua[i][1]])
        vb = fn([x for i in pick for x in ub[i][0]], [x for i in pick for x in ub[i][1]])
        if not (math.isnan(va) or math.isnan(vb)):
            d.append(va - vb)
    d = np.asarray(d)
    p = float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))))
    return {"n_common": len(ids), "delta_ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))], "p": p}


def paired_t2(manifest, ra, rb, which: str) -> dict:
    from scipy.stats import pearsonr

    from bench.third_party import lingolix as L

    per = {}
    for it in manifest:
        if it["id"] in ra and it["id"] in rb and not ra[it["id"]].get("error") and not rb[it["id"]].get("error"):
            pa = L.collect([it], {it["id"]: ra[it["id"]]})[f"{which}_pairs"]
            pb = L.collect([it], {it["id"]: rb[it["id"]]})[f"{which}_pairs"]
            if len(pa) == len(pb):
                per[it["id"]] = (pa, pb)
    spk = {}
    for i in per:
        spk.setdefault(i[:4], []).append(i)
    names = sorted(spk)
    rng = np.random.default_rng(0)
    d = []
    for _ in range(1000):
        pick = rng.integers(0, len(names), len(names))
        A = [p for s in pick for i in spk[names[s]] for p in per[i][0]]
        B = [p for s in pick for i in spk[names[s]] for p in per[i][1]]
        d.append(pearsonr([x[0] for x in A], [x[1] for x in A]).statistic
                 - pearsonr([x[0] for x in B], [x[1] for x in B]).statistic)
    d = np.asarray(d)
    p = float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))))
    return {"n_common": len(per), "delta_ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))], "p": p}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="default")
    a = ap.parse_args(argv)
    from bench.third_party import lingolix as L

    T = L.tests()
    OP = responses(a.mode)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    res = {"tier1": {}, "tier2": {}}
    for lang in ("en", "fr", "es", "de"):
        man = T[f"tier1_{lang}"]
        PA = L.read_responses(f"tier1_{lang}")
        mp, mo = L.metrics(man, PA), L.metrics(man, OP[f"tier1_{lang}"])
        row = {"n_items": len(man), "openpronounce_failed": mo["n_failed"]}
        for k in ("roc_auc", "fpr_at_recall_80"):
            fn = L.roc_auc if k == "roc_auc" else (lambda l, s: L.fpr_at_recall(l, s, 0.80))
            ci_key = "roc_auc_ci" if k == "roc_auc" else "fpr_at_recall_80_ci"
            row[k] = {"pronunciada": {"value": mp[k], "ci": mp[ci_key]},
                      "openpronounce": {"value": mo[k], "ci": mo[ci_key]},
                      "delta": mp[k] - mo[k], **paired_t1(man, PA, OP[f"tier1_{lang}"], fn)}
        row["verdicts"] = {"pronunciada": mp["verdicts"], "openpronounce": mo["verdicts"]}
        row["localisation_top1"] = {"pronunciada": mp["localisation_top1"], "openpronounce": mo["localisation_top1"]}
        res["tier1"][lang] = row
        print(lang, "AUC", round(mp["roc_auc"], 3), round(mo["roc_auc"], 3), "FA@80",
              round(mp["fpr_at_recall_80"], 3), round(mo["fpr_at_recall_80"], 3), flush=True)
    man = T["tier2"]
    PA = L.read_responses("tier2")
    mp, mo = L.metrics(man, PA), L.metrics(man, OP["tier2"])
    for which, key in (("word", "pcc_word"), ("utt", "pcc_utterance")):
        res["tier2"][key] = {"pronunciada": {"value": mp.get(key), "ci": L._pcc_ci(man, PA, which)},
                             "openpronounce": {"value": mo.get(key), "ci": L._pcc_ci(man, OP["tier2"], which)},
                             "delta": mp.get(key) - mo.get(key), **paired_t2(man, PA, OP["tier2"], which)}
    for k in ("roc_auc", "fpr_at_recall_80", "word_coverage", "n_failed"):
        res["tier2"][k] = {"pronunciada": mp.get(k), "openpronounce": mo.get(k)}
    print("tier2", {k: (v["pronunciada"], v["openpronounce"]) if isinstance(v, dict) and "delta" not in v
                    else (round(v["pronunciada"]["value"], 3), round(v["openpronounce"]["value"], 3), round(v["p"], 3))
                    for k, v in res["tier2"].items()})
    out[a.mode] = L._clean(res)
    out["protocol"] = __doc__.split("\n\n", 1)[1]
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
