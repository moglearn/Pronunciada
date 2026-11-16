"""Compact learned letter-to-sound fallback for out-of-vocabulary words.

Training (offline, from the shipped lexicon): hard-EM alignment letter -> {nothing, one phone,
two phones}, then a back-off table over letter context windows
(0,0) -> (0,1) -> (1,1) -> (1,2) -> (2,2) -> (2,3) -> (3,3) [left, right letters].
Only contexts whose majority output differs from their back-off parent are kept, so the table
is small (a few hundred KB gzipped). Prediction = most specific stored context.

Outputs are phone strings joined with "_" ("k_s"), "" for a silent letter; vowels may carry
a stress digit like the lexicon ("a1").
"""
from __future__ import annotations

import collections
import gzip
import math

LADDER = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]


def _key(padded: str, i: int, k: int, m: int) -> str:
    return f"{padded[max(0, i - k):i]}|{padded[i]}|{padded[i + 1:i + 1 + m]}"


class LTS:
    def __init__(self, table: dict[str, str]):
        self.table = table

    @classmethod
    def load(cls, fileobj) -> "LTS":
        t: dict[str, str] = {}
        with gzip.open(fileobj, "rt", encoding="utf-8") as f:
            for line in f:
                k, _, v = line.rstrip("\n").partition("\t")
                t[k] = v
        return cls(t)

    def save(self, path: str) -> None:
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
            for k in sorted(self.table):
                f.write(f"{k}\t{self.table[k]}\n")

    def predict(self, word: str) -> list[str]:
        padded = "^" + word + "$"
        out: list[str] = []
        t = self.table
        for i in range(1, len(padded) - 1):
            res = None
            for k, m in reversed(LADDER):
                v = t.get(_key(padded, i, k, m))
                if v is not None:
                    res = v
                    break
            if res:
                out.extend(res.split("_"))
        return out


def _base(ph: str) -> str:
    return ph[:-1] if ph and ph[-1] in "012" else ph


def align(pairs: list[tuple[str, list[str]]], iters: int = 4):
    """Hard-EM letter->phone(s) alignment. Returns list of (word, outputs per letter)."""
    # init with co-occurrence
    c: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for w, ph in pairs:
        bp = [_base(p) for p in ph]
        for ch in w:
            c[ch][""] += 0.5
            for p in bp:
                c[ch][p] += 1.0 / len(bp)
    logp = _normalise(c, pair_floor=1e-4)
    aligned = []
    for it in range(iters):
        c = collections.defaultdict(collections.Counter)
        aligned = []
        for w, ph in pairs:
            a = _viterbi(w, ph, logp)
            if a is None:
                continue
            aligned.append((w, a))
            for ch, o in zip(w, a):
                c[ch]["_".join(_base(x) for x in o.split("_")) if o else ""] += 1
        logp = _normalise(c, pair_floor=1e-6)
    return aligned


def _normalise(c, pair_floor):
    logp = {}
    for ch, cnt in c.items():
        tot = sum(cnt.values()) + 1e-9
        d = {o: math.log(v / tot) for o, v in cnt.items()}
        d["__pair__"] = math.log(pair_floor)
        d["__single__"] = math.log(1e-5)
        logp[ch] = d
    return logp


def _viterbi(w: str, ph: list[str], logp):
    n, m = len(w), len(ph)
    if m > 2 * n:
        return None
    NEG = -1e18
    bp = [_base(p) for p in ph]
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        lp = logp.get(w[i - 1], {"__pair__": -20.0, "__single__": -12.0})
        eps = lp.get("", -12.0)
        for j in range(0, m + 1):
            best, arg = dp[i - 1][j] + eps, 0
            if j >= 1:
                v = dp[i - 1][j - 1] + lp.get(bp[j - 1], lp["__single__"])
                if v > best:
                    best, arg = v, 1
            if j >= 2:
                v = dp[i - 1][j - 2] + lp.get(bp[j - 2] + "_" + bp[j - 1], lp["__pair__"])
                if v > best:
                    best, arg = v, 2
            dp[i][j], back[i][j] = best, arg
    if dp[n][m] < -1e17:
        return None
    out, j = [], m
    for i in range(n, 0, -1):
        a = back[i][j]
        out.append("_".join(ph[j - a:j]))
        j -= a
    return out[::-1]


def train(pairs: list[tuple[str, list[str]]], iters: int = 4, min_count: int = 2) -> LTS:
    aligned = align(pairs, iters)
    counts = [collections.defaultdict(collections.Counter) for _ in LADDER]
    for w, a in aligned:
        padded = "^" + w + "$"
        for i in range(1, len(padded) - 1):
            o = a[i - 1]
            for li, (k, m) in enumerate(LADDER):
                counts[li][_key(padded, i, k, m)][o] += 1
    table: dict[str, str] = {}
    model = LTS(table)
    for li, (k, m) in enumerate(LADDER):
        new = {}
        for key, cnt in counts[li].items():
            o, n = cnt.most_common(1)[0]
            if li == 0:
                new[key] = o
                continue
            if n < min_count:
                continue
            # parent prediction with the table built so far (levels < li)
            left, ch, right = key.split("|")
            pk, pm = LADDER[li - 1]
            par = None
            for kk, mm in reversed(LADDER[:li]):
                kk2 = f"{left[max(0, len(left) - kk):]}|{ch}|{right[:mm]}"
                if kk2 in table:
                    par = table[kk2]
                    break
            if par != o:
                new[key] = o
        table.update(new)
    return model


class StressModel:
    """Suffix table predicting the stressed nucleus counted from the word end (0 = last).
    Trained on words whose stress is known; prediction = longest stored suffix, default `dflt`."""

    def __init__(self, table: dict[str, int], dflt: int = 1):
        self.table, self.dflt = table, dflt

    def predict(self, word: str, n_nuclei: int) -> int:
        """Index (from the start) of the stressed nucleus."""
        w = "^" + word
        r = self.dflt
        for k in range(min(len(w), 8), 0, -1):
            v = self.table.get(w[-k:])
            if v is not None:
                r = v
                break
        r = min(r, n_nuclei - 1)
        return n_nuclei - 1 - r

    @classmethod
    def train(cls, data: list[tuple[str, int]], dflt: int = 1, min_count: int = 3) -> "StressModel":
        cnt: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for w, r in data:
            w = "^" + w
            for k in range(1, min(len(w), 8) + 1):
                cnt[w[-k:]][r] += 1
        table: dict[str, int] = {}
        for k in range(1, 9):
            for suf, c in cnt.items():
                if len(suf) != k:
                    continue
                v, n = c.most_common(1)[0]
                if sum(c.values()) < min_count or n / sum(c.values()) < 0.6:
                    continue
                par = dflt
                for kk in range(k - 1, 0, -1):
                    if suf[-kk:] in table:
                        par = table[suf[-kk:]]
                        break
                if v != par:
                    table[suf] = v
        return cls(table, dflt)

    def save(self, path: str) -> None:
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
            f.write(f"#default\t{self.dflt}\n")
            for k in sorted(self.table):
                f.write(f"{k}\t{self.table[k]}\n")

    @classmethod
    def load(cls, fileobj) -> "StressModel":
        t, d = {}, 1
        with gzip.open(fileobj, "rt", encoding="utf-8") as f:
            for line in f:
                k, _, v = line.rstrip("\n").partition("\t")
                if k == "#default":
                    d = int(v)
                else:
                    t[k] = int(v)
        return cls(t, d)
