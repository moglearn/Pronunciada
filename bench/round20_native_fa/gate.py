"""Practice gate of the round-20 fixes: native false alarms, changed-word AUC by kind, paired tests.

    python -m bench.round20_native_fa.gate fit                 # choose B's threshold and D2's factor on the fit half
    python -m bench.round20_native_fa.gate table --tags B,C,...   # check half (and fresh sets when scored)

Measuring stick for every number here: the round-20 practice metric (core.word_rows with fix_labels and
measure: label index fixed, punctuation-only tokens not words, no-break-space numbers one token).
Native false alarms: FA@80 (% of correct words at or above the error-signal threshold that catches 80 % of the
changed words) and the % of correct words scoring below 0.6. Paired tests: 1,000 speaker-cluster bootstrap
draws of the difference (tag minus main) on the same items; two-sided p.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict

from bench.round20_native_fa import core as K

KINDS = ("sub_subtle", "sub_moderate", "sub_gross", "indel", "word_omission")


def rows(lang, tag=None, fold=None):
    rr = K.word_rows(lang, tag, fix_labels=True, measure=True)
    return [r for r in rr if fold in (None, "all") or r["fold"] == fold]


def stats(rr) -> dict:
    labels = [r["label"] for r in rr]
    err = [r["err"] for r in rr]
    from bench.third_party import lingolix as L

    neg = [r for r in rr if not r["label"]]
    out = {"auc": L.roc_auc(labels, err), "fa80": L.fpr_at_recall(labels, err, 0.8),
           "low06": sum(r["score"] < K.APP_THRESHOLD for r in neg) / max(1, len(neg)), "n_neg": len(neg),
           "n_pos": sum(labels)}
    for k in KINDS:
        sub = [r for r in rr if not r["label"] or r["kind"] == k]
        out[f"auc_{k}"] = L.roc_auc([r["label"] for r in sub], [r["err"] for r in sub])
    return out


def paired(base, new, n=1000, seed=0):
    """Speaker-cluster bootstrap of new - base for fa80, low06, auc (rows matched by item and token)."""
    from bench.third_party import lingolix as L

    key = lambda r: (r["item"], r["k"])  # noqa: E731
    nb = {key(r): r for r in new}
    pairs = [(r, nb[key(r)]) for r in base if key(r) in nb]
    g = defaultdict(list)
    for a, b in pairs:
        g[a["speaker"]].append((a, b))
    groups = list(g.values())
    rng = random.Random(seed)

    def m(rs):
        lab = [r["label"] for r in rs]
        e = [r["err"] for r in rs]
        neg = [r for r in rs if not r["label"]]
        return (L.fpr_at_recall(lab, e, 0.8), sum(r["score"] < K.APP_THRESHOLD for r in neg) / max(1, len(neg)),
                L.roc_auc(lab, e))

    d0 = [y - x for x, y in zip(m([a for a, _ in pairs]), m([b for _, b in pairs]))]
    draws = []
    for _ in range(n):
        pick = [p for gg in (groups[rng.randrange(len(groups))] for _ in range(len(groups))) for p in gg]
        x, y = m([a for a, _ in pick]), m([b for _, b in pick])
        draws.append([yy - xx for xx, yy in zip(x, y)])
    out = {}
    for j, nm in enumerate(("fa80", "low06", "auc")):
        v = sorted(d[j] for d in draws if not math.isnan(d[j]))
        p = 2 * min(sum(x <= 0 for x in v), sum(x >= 0 for x in v)) / len(v)
        out[nm] = {"d": d0[j], "ci": [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]], "p": min(1.0, p)}
    return out


# ----------------------------------------------------------------------------- D2 offline (fitting only)

def _norm(t):
    import unicodedata

    return re.sub(r"^\W+|\W+$", "", unicodedata.normalize("NFC", t).lower().replace("’", "'"))


def guessed_rows(lang, rr):
    """Rows whose front-end words are all guessed (pronunciada.r20.guessed_words, re-implemented on the dump)."""
    dump = K.dump_rec(lang)
    its = {it["id"]: it for it in K.practice_items(lang)}
    out = set()
    for r in rr:
        rec, it = dump[r["item"]], its[r["item"]]
        oov = {w.split(":", 1)[1] for w in rec.get("warnings", []) if w.startswith("oov:")}
        ok = bool(r["iw"])
        for j in r["iw"]:
            w = rec["rwords"][j]
            parts = [x for x in re.split(r"[-']", _norm(w["text"])) if x]
            g = _norm(w["text"]) in oov or any(x in oov for x in parts)
            if not g and lang in ("en", "es", "fr") and j > 0 and w["text"][:1].isupper():
                before = it["target"][:w["span"][0]].rstrip()
                g = bool(before) and before[-1] not in ".!?:"
            ok = ok and g
        if ok:
            out.add((r["item"], r["k"]))
    return out


def d2_transform(rr, gset, s, gross=0.05):
    out = []
    for r in rr:
        if (r["item"], r["k"]) in gset and gross <= r["score"] < 1.0:
            sc = round((1 - s * (1 - r["score"])) * 100, 1) / 100
            r = dict(r, score=sc, err=1 - sc)
        out.append(r)
    return out


def fit():
    res = {"B": {}, "D2": {}}
    from bench.round20_native_fa.runs import TAUS

    base = {l: rows(l, None, "fit") for l in K.LANGS}
    b0 = {l: stats(base[l])["fa80"] for l in K.LANGS}
    for tau in TAUS:
        rel = []
        per = {}
        for l in K.LANGS:
            st = stats(rows(l, f"B{tau}", "fit"))
            per[l] = st["fa80"]
            rel.append((st["fa80"] - b0[l]) / b0[l])
        res["B"][str(tau)] = {"mean_rel_fa80": sum(rel) / len(rel), "fa80": per}
        print("B", tau, round(100 * sum(rel) / len(rel), 1), {l: round(100 * v, 2) for l, v in per.items()}, flush=True)
    gs = {l: guessed_rows(l, base[l]) for l in ("en", "de", "es", "fr")}
    for s in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2):
        rel, per = [], {}
        for l in ("en", "de", "es", "fr"):
            st = stats(d2_transform(base[l], gs[l], s))
            per[l] = st["fa80"]
            rel.append((st["fa80"] - b0[l]) / b0[l])
        res["D2"][str(s)] = {"mean_rel_fa80": sum(rel) / len(rel), "fa80": per}
        print("D2", s, round(100 * sum(rel) / len(rel), 1), {l: round(100 * v, 2) for l, v in per.items()}, flush=True)
    res["B_choice"] = min(res["B"], key=lambda k: res["B"][k]["mean_rel_fa80"])
    res["D2_choice"] = min(res["D2"], key=lambda k: res["D2"][k]["mean_rel_fa80"])
    res["base_fit_fa80"] = b0
    print("choice", res["B_choice"], res["D2_choice"])
    K.save_json("r20_fit.json", res)
    return res


def table(tags, fold="check", langs=K.LANGS):
    out = {}
    for l in langs:
        b = rows(l, None, fold)
        out[l] = {"main": stats(b)}
        for t in tags:
            try:
                n = rows(l, t, fold)
            except Exception as e:  # noqa: BLE001
                print("skip", l, t, e)
                continue
            out[l][t] = {**stats(n), "paired": paired(b, n)}
        print(l, json.dumps({k: {m: round(v[m], 4) for m in ("auc", "fa80", "low06")} for k, v in out[l].items()}),
              flush=True)
    return out


def baserate(n=2000):
    """Share of words the front-end has to guess (pronunciada.r20.guessed_in_ref; oov and names separately) and of
    number tokens, on FLEURS TRAIN transcripts and Common Voice dev sentences (text only)."""
    from bench.data import load
    from bench.round19_rollout.sources import cv_meta, read_tsv
    from pronunciada.r20 import guessed_in_ref
    from pronunciada.text import get_g2p

    out = {}
    for lang in ("en", "de", "es", "fr", "ko"):
        g = get_g2p(lang)
        for src in ("fleurs_train", "cv_dev"):
            if src == "fleurs_train":
                texts = [r["text"] for r in load("fleurs", "train", lang)][:n]
            else:
                texts = [r["sentence"] for r in read_tsv(cv_meta(lang) / "dev.tsv")][:n]
            words = guessed = oov = num = 0
            for t in texts:
                try:
                    ref = g(t)
                except Exception:  # noqa: BLE001
                    continue
                gs = guessed_in_ref(ref)
                o = {w.split(":", 1)[1] for w in ref.warnings if w.startswith("oov:")}
                words += len(ref.words)
                guessed += len(gs)
                oov += sum(1 for w in ref.words if _norm(w.text) in o)
                num += sum(1 for w in ref.words if re.search(r"\d", w.text))
            out[f"{lang}/{src}"] = {"sentences": len(texts), "words": words, "guessed_pct": 100 * guessed / max(1, words),
                                    "oov_pct": 100 * oov / max(1, words), "number_pct": 100 * num / max(1, words)}
            print(lang, src, {k: round(v, 2) if isinstance(v, float) else v for k, v in out[f"{lang}/{src}"].items()},
                  flush=True)
    K.save_json("r20_baserate.json", out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "table", "baserate"])
    ap.add_argument("--tags", default="")
    ap.add_argument("--fold", default="check")
    ap.add_argument("--langs", default=",".join(K.LANGS))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        fit()
    elif a.cmd == "baserate":
        baserate()
    else:
        res = table([t for t in a.tags.split(",") if t], a.fold, a.langs.split(","))
        K.save_json(a.out or f"gate_{a.fold}.json", res)


if __name__ == "__main__":
    main()
