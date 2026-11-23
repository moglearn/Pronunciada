"""Calibrators and the strictness rule (port of pronunciada/calibrate.py and of three small
formulas of pronunciada/scorer.py).

Level model file (<name>.npz, see pronunciada.reference.npz): member "meta" is UTF-8 JSON (bytes)
with "num" (numeric input columns), "cat" (categorical columns), "vocab", "heads"
({head: {"kind": "prob" | "reg", "calib": {"kind": "platt" | "isotonic" | "none",
"params": [a, b]}, "clip": [lo, hi]}}). Per head <h>, the tree arrays <h>/feature (int16,
-1 = leaf), <h>/threshold (float64), <h>/value (float32, read as double), <h>/left and
<h>/right (int16, relative to the tree's root), <h>/missing_left (uint8), <h>/cat (int16, index
into <h>/bitsets, -1 = numeric split), <h>/bitsets (uint32, 8 words = 256 bits per categorical
split), <h>/roots (int32, first node of each tree), <h>/baseline (float64), <h>/depth (int32),
<h>/known_cols (int16) and <h>/known (uint32 x 8 per column: the category codes seen in
training), and for isotonic maps <h>/calib_x, <h>/calib_y (float32).

The input matrix X (one row per sound, word or sentence; columns = meta "num" then "cat", NaN =
missing) is built by pronunciada.features.matrix; it is an input here.
"""
from __future__ import annotations

import json

from .npz import load_npz
from .numeric import NAN, exp, fma, is_finite, log, log1p, minimum, sum_pairwise

P_CLIP = 1e-4                 # probabilities are kept in [1e-4, 1 - 1e-4]
S_MIN, S_MAX = -0.5, 1.5      # strictness range
BIG = 1.7976931348623157e308  # largest double


def clip(x: float, lo: float, hi: float) -> float:
    """np.clip: NaN stays NaN."""
    if x != x:
        return x
    return lo if x < lo else (hi if x > hi else x)


def logit(p: float) -> float:
    p = clip(float(p), P_CLIP, 1 - P_CLIP)
    return log(p) - log1p(-p)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + exp(-z))


def clip_strictness(s: float) -> float:
    return min(max(float(s), S_MIN), S_MAX)


def blend_logit(z_len: float, z_str: float, s: float) -> float:
    """(1 - s) * z_len + s * z_str (inputs must satisfy z_str <= z_len)."""
    s = clip_strictness(s)
    return (1.0 - s) * z_len + s * z_str


def blend(p_len: float, p_str: float, s: float) -> float:
    zl = logit(p_len)
    zs = minimum(logit(p_str), zl)
    return sigmoid(blend_logit(zl, zs, s))


def interp(x: float, xs, ys) -> float:
    """np.interp(x, xs, ys) for increasing xs: ys[0] left of xs[0], ys[-1] right of xs[-1];
    j = largest index with xs[j] <= x; x == xs[j] -> ys[j]; else the line through
    (xs[j], ys[j]), (xs[j+1], ys[j+1]): fma(slope, x - xs[j], ys[j])."""
    n = len(xs)
    if x != x:
        return x
    if x < xs[0]:
        return ys[0]
    if x > xs[n - 1]:
        return ys[n - 1]
    lo, hi = 0, n
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if x >= xs[mid]:
            lo = mid + 1
        else:
            hi = mid
    j = lo - 1
    if j == n - 1 or xs[j] == x:
        return ys[j]
    slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j])
    # production's NumPy build evaluates slope * d + y as one fused multiply-add (one rounding);
    # a NumPy build without FMA contraction rounds twice and can differ in the last bit
    r = fma(slope, x - xs[j], ys[j])
    if r != r:
        r = fma(slope, x - xs[j + 1], ys[j + 1])
        if r != r and ys[j] == ys[j + 1]:
            r = ys[j]
    return r


def _bit(words, code: int) -> bool:
    """Bit `code` (0..255) of a 256-bit set stored as 8 uint32 words, little-endian bit order."""
    return ((words[code // 32] >> (code % 32)) & 1) == 1


class TreeEnsemble:
    """raw(x) = baseline + sum over trees of the leaf reached by x (scikit-learn
    HistGradientBoosting semantics). At an inner node with input value v = x[feature]:
      v is NaN                -> left iff missing_left;
      categorical split       -> code = int(v) (truncation); right iff code is outside 0..255 or
                                 its bit is not set in the node's bitset;
      numeric split           -> right iff v > threshold; when NaN goes right (missing_left
                                 false) a threshold of +inf is read as the largest double, so
                                 v = +inf also goes right.
    Before the walk, a categorical column listed in known_cols whose value is not a code seen
    in training (not finite, outside 0..255 or bit not set in `known`) is set to NaN.
    The leaf values (float32 widened to double) are added in tree order with pairwise summation,
    then added to the baseline."""

    def __init__(self, g: dict):
        self.feature = g["feature"].data
        self.threshold = g["threshold"].data
        self.value = g["value"].data
        self.missing_left = [bool(x) for x in g["missing_left"].data]
        self.cat = g["cat"].data
        bs = g["bitsets"].data
        self.bitsets = [bs[8 * k:8 * k + 8] for k in range(len(bs) // 8)]
        self.roots = g["roots"].data
        self.baseline = float(g["baseline"].scalar())
        self.depth = int(g["depth"].scalar())
        self.known_cols = g["known_cols"].data if "known_cols" in g else []
        kn = g["known"].data if "known" in g else []
        self.known = [kn[8 * k:8 * k + 8] for k in range(len(kn) // 8)]
        N = len(self.feature)
        ends = list(self.roots[1:]) + [N]
        self.left, self.right = [0] * N, [0] * N
        for t, r in enumerate(self.roots):
            for i in range(r, ends[t]):
                self.left[i] = g["left"].data[i] + r
                self.right[i] = g["right"].data[i] + r

    def mask_unknown(self, x: list) -> list:
        x = list(x)
        for k, j in enumerate(self.known_cols):
            v = x[j]
            ok = is_finite(v) and 0 <= v < 256
            if not (ok and _bit(self.known[k], int(v))):
                x[j] = NAN
        return x

    def leaf(self, x: list, root: int) -> float:
        node = root
        while self.feature[node] >= 0:
            v = x[self.feature[node]]
            if v != v:
                right = not self.missing_left[node]
            elif self.cat[node] >= 0:
                code = int(v)
                right = not (0 <= code < 256 and _bit(self.bitsets[self.cat[node]], code))
            else:
                th = self.threshold[node]
                if not self.missing_left[node] and th == float("inf"):
                    th = BIG
                right = v > th
            node = self.right[node] if right else self.left[node]
        return self.value[node]

    def raw(self, x: list) -> float:
        x = self.mask_unknown([float(v) for v in x])
        if not self.roots:
            return self.baseline
        return self.baseline + sum_pairwise([self.leaf(x, r) for r in self.roots])


class Head:
    """kind "reg": raw clipped to [lo, hi] (when given); kind "prob": calibration map of raw,
    clipped to [1e-4, 1 - 1e-4]: platt = sigmoid(a * raw + b), isotonic = interp(raw, calib_x,
    calib_y), none = sigmoid(raw)."""

    def __init__(self, name, trees, kind, calib_kind=None, params=None, xs=None, clip_=None):
        self.name, self.trees, self.kind = name, trees, kind
        self.calib_kind, self.params, self.xs, self.clip = calib_kind, params, xs, clip_

    def predict(self, x: list) -> float:
        r = self.trees.raw(x)
        if self.kind == "reg":
            return r if self.clip is None else clip(r, self.clip[0], self.clip[1])
        if self.calib_kind == "platt":
            p = sigmoid(self.params[0] * r + self.params[1])
        elif self.calib_kind == "isotonic":
            p = interp(r, self.xs, self.params)
        else:
            p = sigmoid(r)
        return clip(p, P_CLIP, 1 - P_CLIP)


class LevelModel:
    def __init__(self, path: str):
        arr = load_npz(path)
        self.meta = json.loads(bytes(arr.pop("meta").data).decode("utf-8"))
        self.num = self.meta["num"]
        self.cat = self.meta["cat"]
        self.vocab = self.meta.get("vocab", {})
        self.encoding = self.meta.get("encoding", "codes")
        self.heads = {}
        for name, h in self.meta["heads"].items():
            g = {k[len(name) + 1:]: v for k, v in arr.items() if k.startswith(name + "/")}
            trees = TreeEnsemble(g)
            ck = params = xs = None
            if h["kind"] == "prob":
                ck = h["calib"]["kind"]
                params = [float(v) for v in g["calib_y"].data] if "calib_y" in g \
                    else [float(v) for v in h["calib"].get("params", [])]
                xs = [float(v) for v in g["calib_x"].data] if "calib_x" in g else None
            self.heads[name] = Head(name, trees, h["kind"], ck, params, xs,
                                    tuple(h["clip"]) if h.get("clip") else None)

    def predict(self, x: list, heads=None) -> dict:
        """{head: output} for one input row; P(strict) is capped at P(lenient)."""
        out = {k: h.predict(x) for k, h in self.heads.items() if heads is None or k in heads}
        if "len" in out and "str" in out:
            out["str"] = minimum(out["str"], out["len"])
        return out

    def logits(self, x: list, heads=None) -> dict:
        """predict(), with the probability heads as log-odds: z_len = logit(P_len),
        z_str = min(logit(P_str), z_len)."""
        out = self.predict(x, heads)
        res = {k: v for k, v in out.items() if k not in ("len", "str")}
        if "len" in out:
            res["z_len"] = logit(out["len"])
        if "str" in out:
            res["z_str"] = minimum(logit(out["str"]), res.get("z_len", float("inf")))
        return res


# ----------------------------------------------------------------------------- scorer formulas

def present_prob(cal: dict, llr_del: list) -> float:
    """P(word was said) (omission channel, data/calib/omission.json): llr_del of the word's
    sounds, NaN -> 0, clipped to [-clip, clip];
    z = b0 + b1 * max + b2 * mean + b3 * [one sound] + b4 * [two sounds]; sigmoid(z).
    mean = pairwise sum / count."""
    c = cal["clip"]
    d = []
    for v in llr_del:
        v = float(v)
        if v != v:
            v = 0.0
        elif v == float("inf"):
            v = BIG
        elif v == float("-inf"):
            v = -BIG
        d.append(clip(v, -c, c))
    b = cal["coef"]
    mx = d[0]
    for v in d:
        if v > mx:
            mx = v
    mean = sum_pairwise(d) / len(d)
    z = b[0] + b[1] * mx + b[2] * mean + b[3] * (1.0 if len(d) == 1 else 0.0) + b[4] * (1.0 if len(d) == 2 else 0.0)
    return sigmoid(z)


def prosody_prob(cal: dict, probs: list, s: float) -> float:
    """P_pros(word) (tone / stress channel, data/calib/prosody/<lang>.json): q = sorted logits of
    the word's syllable probabilities; z = b0 + b1 * q[0] + b2 * q[1] (q[0] if one) + b3 * count;
    lenient = z + c_len, strict = z + c_str; sigmoid(blend_logit(lenient, min(strict, lenient), s))."""
    q = sorted(logit(p) for p in probs)
    f = [q[0], q[1] if len(q) > 1 else q[0], float(len(q))]
    b = cal["coef"]
    z = b[0] + b[1] * f[0] + b[2] * f[1] + b[3] * f[2]
    zl, zs = z + cal["c_len"], z + cal["c_str"]
    return sigmoid(blend_logit(zl, min(zs, zl), s))
