"""Round 20 (insdel): features on collected records, native cost tables, the lower-only word term.

Variants (cfg): weighted (feature-distance substitutions + native insertion / deletion costs) or
plain (unit costs); alts (native-variant allowance: alternatives and free drops) or canonical
only; tol ("half" = half the gap to the neighbouring words, or a number of frames); top2 (a drop
whose sound is among the top-2 tokens of some frame of the word's region counts half).
"""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache

import numpy as np

from bench.round20_insdel import greedy_match as ID

FEATS = ("ins", "del", "sum", "rate", "mass", "nedit", "edge_sum")
CFGS = {
    "plain": {"weighted": False, "alts": True, "tol": "half", "top2": False},
    "plain_noalt": {"weighted": False, "alts": False, "tol": "half", "top2": False},
    "weighted": {"weighted": True, "alts": True, "tol": "half", "top2": False},
    "weighted_noalt": {"weighted": True, "alts": False, "tol": "half", "top2": False},
    "weighted_top2": {"weighted": True, "alts": True, "tol": "half", "top2": True},
    "weighted_tol2": {"weighted": True, "alts": True, "tol": 2, "top2": False},
}


@lru_cache(maxsize=None)
def tok_id() -> dict:
    from pronunciada.acoustic import load_tokens

    return {t: i for i, t in enumerate(load_tokens())}


def rec_features(r: dict, cfg: dict, costs: ID.Costs | None) -> dict:
    ph = r["phones"]
    C = costs if (cfg["weighted"] and costs is not None) else ID.Costs(weighted=cfg["weighted"])
    f = ID.features(ph["ipa"], ph["alts"], ph["word"], ph["start"], ph["end"], len(r["words"]["prob"]), r["T"],
                    r["free"], r["topv"][:, 0].astype(np.float32), r["top"], costs=C, allow_alts=cfg["alts"],
                    tol=cfg["tol"], soft_top2=cfg["top2"], tok_id=tok_id())
    f["sum"] = f["ins"] + f["del"]
    f["nedit"] = f["n_ins"] + f["n_del"]
    f["edge_sum"] = f["sum"] * (1 + f["edge"])
    return f


def native_table(recs: list[dict], cfg: dict, m: float = 20.0) -> ID.Costs:
    """Insertion / deletion costs per unit base from native records (plain alignment counts):
    cost = clip(1 + ln(mean rate / smoothed rate) / 3, 0.3, 1.5)."""
    ins, heard, dele, exp_ = Counter(), Counter(), Counter(), Counter()
    C0 = ID.Costs(weighted=False)
    for r in recs:
        ph = r["phones"]
        hu = ID.heard_units(r["free"])
        mids = np.array([(a + b) / 2 for _, a, b, _ in hu]) if hu else np.zeros(0)
        regs = ID.word_regions(ph["word"], ph["start"], ph["end"], len(r["words"]["prob"]), r["T"], cfg["tol"])
        for k, reg in enumerate(regs):
            idx = np.flatnonzero(ph["word"] == k)
            if reg is None or not len(idx):
                continue
            variants, fd = [], []
            for i in idx:
                vs = {ID.units(ph["ipa"][i])}
                f = False
                if cfg["alts"]:
                    for a, _ in ph["alts"][i]:
                        if a == "":
                            f = True
                        else:
                            vs.add(ID.units(a))
                vs.discard(())
                variants.append(sorted(vs, key=lambda t: (len(t), t)) or [ID.units(ph["ipa"][i])])
                fd.append(f)
            sel = [q for q in range(len(hu)) if reg[0] <= mids[q] < reg[1]]
            hh = tuple(hu[q][0] for q in sel)
            res = ID.align_word(variants, fd, hh, C0)
            for u in hh:
                heard[ID.base(u)] += 1
            for q in res[5]:
                ins[ID.base(hh[q])] += 1
            for v in variants:
                if v and v[0]:
                    exp_[ID.base(v[0][0])] += 1
            for d in res[6]:
                if variants[d] and variants[d][0]:
                    dele[ID.base(variants[d][0][0])] += 1

    def table(num, den):
        tot_n, tot_d = sum(num.values()), max(sum(den.values()), 1)
        rbar = max(tot_n / tot_d, 1e-6)
        out = {}
        for u, d in den.items():
            r = (num[u] + m * rbar) / (d + m)
            out[u] = float(np.clip(1 + math.log(rbar / r) / 3, 0.3, 1.5))
        return out

    return ID.Costs(ins=table(ins, heard), dele=table(dele, exp_), weighted=True)


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def apply_term(p, f, a: float, k: float, two_sided: float = 0.0):
    """Lower-only: z - a * max(0, f - k). two_sided > 0 additionally raises words with f == 0 by
    that many log-odds (only an alternative; the default is lower-only)."""
    z = logit(p) - a * np.maximum(0.0, np.asarray(f, np.float64) - k)
    if two_sided:
        z = z + two_sided * (np.asarray(f) <= 0)
    return sig(z)
