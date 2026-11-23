"""How often fix B (unsaid words) fires, and what it costs.

    python -m bench.round20_native_fa.btrigger

Practice sets (from the B run dumps): share of control items and of changed items (by kind) in which the trigger
fired. speechocean762 TRAIN (learners, every clip, through Scorer with B on): share of clips where it fired, and the
time of the round-20 pass (Scorer timing "r20") averaged over all clips and over the clips where it fired.
"""
from __future__ import annotations

import json
import os
from collections import Counter

from bench.round20_native_fa import core as K
from bench.round20_native_fa.collect import load


def practice(tag):
    out = {}
    for lang in K.LANGS:
        d = load(lang, tag)
        its = {it["id"]: it for it in K.practice_items(lang)}
        cnt, fired = Counter(), Counter()
        for iid, it in its.items():
            kind = (it.get("label") or {}).get("bucket") or "control"
            cnt[kind] += 1
            r = d.get(iid)
            if r and (r.get("r20") or {}).get("unsaid"):
                fired[kind] += 1
        out[lang] = {k: {"n": cnt[k], "fired_pct": 100 * fired[k] / cnt[k]} for k in cnt}
        print(lang, {k: round(v["fired_pct"], 1) for k, v in out[lang].items()}, flush=True)
    return out


def so762():
    from bench.data import load as dload
    from pronunciada import Scorer

    os.environ["PRONUNCIADA_R20_UNSAID"] = "1"
    sc = Scorer("en", threads=2)
    n = fired = 0
    t_all, t_fired = [], []
    for rec in dload("speechocean762", split="train"):
        try:
            r = sc.score(rec["audio"], rec["text"])
        except ValueError:
            continue
        n += 1
        info = r._pronunciada.get("r20") or {}
        t = r.timing_ms.get("r20", 0.0)
        t_all.append(t)
        if info.get("unsaid"):
            fired += 1
            t_fired.append(t)
    out = {"clips": n, "fired": fired, "fired_pct": 100 * fired / max(1, n),
           "r20_ms_mean_all": sum(t_all) / max(1, len(t_all)),
           "r20_ms_mean_when_fired": sum(t_fired) / max(1, len(t_fired))}
    print("so762 train", out, flush=True)
    return out


def main():
    res = {"practice": practice("B0.02"), "so762_train": so762()}
    K.save_json("r20_btrigger.json", res)


if __name__ == "__main__":
    main()
