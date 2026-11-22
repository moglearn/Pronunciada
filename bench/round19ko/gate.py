"""Round 19 (Korean, part 2): selective identity evidence (offline helpers shared by fit2.py).

A sound "fires" only when all hold:
  * the identity head's top class is a different class than the expected one (rival = top);
  * the aligner agrees that the reference sound is not the best explanation: llr_sub <= -llr_min
    (the "anything was said here" path beats the reference sound by at least llr_min nats);
  * alignment is confident: the sound got >= 1 frame (nfr_min) of the canonical path;
and its margin s is below the class x position threshold thr[(cls, pos)] = the q_c native quantile
of s for that class and position (native pool), shifted by `shift`.
Word evidence e = min over fired sounds of (s - thr) (<= shift); NaN when nothing fires.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def class_pos_thresholds(rows: dict, qc: float, min_n: int = 40) -> dict:
    by, byc = defaultdict(list), defaultdict(list)
    for r in rows.values():
        st = r["st"]
        for c, p, s in zip(st["cls"], st["pos"], st["s"]):
            if c is not None and np.isfinite(s):
                by[(c, p)].append(s)
                byc[c].append(s)
    thr = {}
    for (c, p), v in by.items():
        src = v if len(v) >= min_n else byc[c]
        thr[f"{c}|{p}"] = float(np.quantile(src, qc))
    for c, v in byc.items():
        thr[f"{c}|*"] = float(np.quantile(v, qc))
    return thr


def sound_e(st, thr: dict, llr_min: float, nfr_min: int, need_rival: bool = True):
    n = len(st["cls"])
    e = np.full(n, np.nan)
    for i in range(n):
        c = st["cls"][i]
        if c is None or not np.isfinite(st["s"][i]):
            continue
        if need_rival and (st["top"][i] is None or st["top"][i] == c):
            continue
        if st["llr"][i] > -llr_min or st["nfr"][i] < nfr_min:
            continue
        t = thr.get(f"{c}|{st['pos'][i]}", thr.get(f"{c}|*"))
        if t is None:
            continue
        e[i] = st["s"][i] - t
    return e


def word_e(r, e):
    nw = len(r["prob"])
    w = r["st"]["word"]
    out = np.full(nw, np.nan)
    for j in range(nw):
        v = e[w == j]
        v = v[np.isfinite(v)]
        if len(v):
            out[j] = v.min()
    return out
