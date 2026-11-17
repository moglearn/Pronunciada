"""Round 17: before ("base", no head evidence) vs after, from bench/round17/collect.py chunks.

    python -m bench.round17.compare --split validation|test

Per language and contrast direction: sound hit@5 % (threshold = 5 % quantile of all native sounds'
shown probability in that run) and AUC; native sound acceptance (P >= 0.5), planted-substitution
sound AUC; Japanese: the length channel's hit@5 % (natives judged by the channel) and word-level
numbers (native word acceptance, hit@5 % of the swapped word). Paired clip bootstrap (500) for
the hit deltas, thresholds re-derived per resample. Writes results/tables/round17_<split>.json.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.round17.collect import CONTRASTS, load
from bench.round17.fit import auc

N_BOOT = 500


def flat(clips):
    nat = defaultdict(list)
    ed = defaultdict(list)
    pl = defaultdict(list)
    wn = defaultdict(list)
    for k, c in enumerate(clips):
        n = c["native"]
        nat["clip"] += [k] * len(n["prob"])
        nat["prob"] += list(n["prob"])
        nat["length"] += [np.nan if x is None else x for x in n["length"]]
        wn["clip"] += [k] * len(c["word_prob"])
        wn["prob"] += list(c["word_prob"])
        for e in c["edits"]:
            ed["clip"].append(k)
            for f in ("contrast", "direction", "word_prob"):
                ed[f].append(e[f])
            ed["prob"].append(float(e["prob"][0]))
            ed["length"].append(np.nan if e["length"][0] is None else e["length"][0])
        for e in c["planted"]:
            pl["clip"] += [k] * len(e["prob"])
            pl["prob"] += list(e["prob"])
    return ({k: np.asarray(v) for k, v in nat.items()}, {k: np.asarray(v) for k, v in ed.items()},
            {k: np.asarray(v) for k, v in pl.items()}, {k: np.asarray(v) for k, v in wn.items()})


def hits(nat, ed, sel, key="prob", nat_key=None):
    nk = nat[nat_key or key]
    nk = nk[np.isfinite(nk)]
    t = np.quantile(nk, 0.05)
    v = ed[key][sel]
    v = v[np.isfinite(v)]
    return float(np.mean(v <= t)) if len(v) else None


def summary(clips, lang):
    nat, ed, pl, wn = flat(clips)
    out = {"native_sound_accept": float(np.mean(nat["prob"] >= 0.5)), "native_word_accept": float(np.mean(wn["prob"] >= 0.5)),
           "planted_sound_auc": auc(1 - pl["prob"], 1 - nat["prob"]), "contrasts": {}}
    for c in CONTRASTS[lang]:
        out["contrasts"][c] = {}
        for d in sorted(set(ed["direction"][ed["contrast"] == c])):
            sel = (ed["contrast"] == c) & (ed["direction"] == d)
            r = {"n": int(sel.sum()), "sound_hit5": hits(nat, ed, sel), "sound_auc": auc(1 - ed["prob"][sel], 1 - nat["prob"]),
                 "word_hit5": float(np.mean(ed["word_prob"][sel] <= np.quantile(wn["prob"], 0.05)))}
            if lang == "ja":
                r["length_hit5"] = hits(nat, ed, sel, "length")
            out["contrasts"][c][d] = r
    return out


def boot_delta(cb, ca, lang, key, rng):
    """Paired clip bootstrap of after - before for every (contrast, direction) value of `key`."""
    ids = np.arange(len(cb))
    res = defaultdict(list)
    for _ in range(N_BOOT):
        b = rng.choice(ids, len(ids))
        sb, sa = summary([cb[i] for i in b], lang), summary([ca[i] for i in b], lang)
        for c, dd in sa["contrasts"].items():
            for d, r in dd.items():
                x, y = sb["contrasts"][c].get(d, {}).get(key), r.get(key)
                if x is not None and y is not None:
                    res[f"{c}/{d}"].append(y - x)
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--langs", default="ko,ru,zh,ja")
    ap.add_argument("--boot", type=int, default=1)
    a = ap.parse_args()
    rep = {}
    rng = np.random.default_rng(0)
    for lang in a.langs.split(","):
        cb, ca = load(lang, a.split, "base"), load(lang, a.split, "after")
        idb = {c["id"]: c for c in cb}
        ca = [c for c in ca if c["id"] in idb]
        cb = [idb[c["id"]] for c in ca]
        rep[lang] = {"clips": len(ca), "before": summary(cb, lang), "after": summary(ca, lang)}
        if a.boot:
            key = "length_hit5" if lang == "ja" else "sound_hit5"
            rep[lang]["delta_ci_" + key] = boot_delta(cb, ca, lang, key, rng)
        b, f = rep[lang]["before"], rep[lang]["after"]
        print(lang, "native sound", round(b["native_sound_accept"], 4), "->", round(f["native_sound_accept"], 4),
              "| native word", round(b["native_word_accept"], 4), "->", round(f["native_word_accept"], 4),
              "| planted AUC", round(b["planted_sound_auc"], 4), "->", round(f["planted_sound_auc"], 4), flush=True)
        for c in f["contrasts"]:
            for d in f["contrasts"][c]:
                print("   ", c, d, {k: (round(b["contrasts"][c][d][k], 3), round(v, 3)) for k, v in f["contrasts"][c][d].items()
                                   if isinstance(v, float)}, flush=True)
        if a.boot:
            print("    CI", rep[lang].get("delta_ci_length_hit5") or rep[lang].get("delta_ci_sound_hit5"))
    Path(f"results/tables/round17_{a.split}.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
