"""Calibrators and the strictness rule (DESIGN section 4), numpy only.

Models are gradient-boosted trees trained offline (bench/calib, scikit-learn
HistGradientBoosting) and exported to compact .npz files; here they are evaluated by a small
vectorised tree walker. One file = one level (sound / word / sentence) = several heads sharing the
same input columns:

* ``reg*``  regression heads (sound: human 0-2 phone score; word: 0-10 accuracy; sentence: 0-10
  accuracy / fluency / prosodic / total)
* ``len``   P(lenient ok), ``str``  P(strict ok): boosted log-odds followed by a 1-D calibration
  map (Platt or isotonic, chosen on out-of-fold predictions by ECE).

Strictness s in [0, 1] (values outside extrapolate, clipped to [-0.5, 1.5]):

    logit P_s = (1 - s) * logit P_lenient + s * logit P_strict

P_strict <= P_lenient is enforced before blending, so P_s is non-increasing in s.
A per-language logit offset (``bias``, fitted on native FLEURS-train speech) is added to both
logits of the universal sound model.
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

P_CLIP = 1e-4
S_MIN, S_MAX = -0.5, 1.5


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), P_CLIP, 1 - P_CLIP)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def clip_strictness(s: float) -> float:
    return float(min(max(float(s), S_MIN), S_MAX))


def blend_logit(z_len, z_str, s: float):
    """Strictness rule in log-odds (inputs must already satisfy z_str <= z_len)."""
    s = clip_strictness(s)
    return (1.0 - s) * np.asarray(z_len, dtype=np.float64) + s * np.asarray(z_str, dtype=np.float64)


def blend(p_len, p_str, s: float):
    """P_s from P(lenient ok) and P(strict ok); enforces P_strict <= P_lenient first."""
    zl = logit(p_len)
    zs = np.minimum(logit(p_str), zl)
    return sigmoid(blend_logit(zl, zs, s))


# ----------------------------------------------------------------------------- trees

class TreeEnsemble:
    """Boosted trees: raw = baseline + sum_t leaf_t(x).

    Node arrays are concatenated over trees (children relative to the tree root): feature
    (-1 = leaf), threshold, left, right, value, missing_left, cat (index into bitsets, -1 =
    numeric split); `known` = per categorical column the codes seen in training (others are
    treated as missing). Split rule (as scikit-learn
    HistGradientBoosting): NaN -> missing_left; categorical -> left iff the integer code is in
    the node's bitset; numeric -> left iff x <= threshold."""

    def __init__(self, d: dict):
        self.feature = d["feature"].astype(np.int64)
        self.threshold = d["threshold"].astype(np.float64)
        self.value = d["value"].astype(np.float64)
        self.missing_left = d["missing_left"].astype(bool)
        self.cat = d["cat"].astype(np.int64)
        self.bitsets = d["bitsets"].astype(np.uint32).reshape(-1, 8)
        self.roots = d["roots"].astype(np.int64)
        self.baseline = float(d["baseline"])
        self.depth = int(d["depth"])
        # categorical columns: codes not seen in training are treated as missing
        self.known_cols = d.get("known_cols", np.zeros(0)).astype(np.int64)
        self.known = d.get("known", np.zeros((0, 8))).astype(np.uint32).reshape(-1, 8)
        # children are stored relative to their tree's root (int16) to keep the files small
        sizes = np.diff(np.r_[self.roots, len(self.feature)])
        base = np.repeat(self.roots, sizes)
        self.left = d["left"].astype(np.int64) + base
        self.right = d["right"].astype(np.int64) + base
        # leaves point to themselves so the walk can run a fixed number of steps
        leaf = self.feature < 0
        idx = np.arange(len(self.feature))
        self.left = np.where(leaf, idx, self.left)
        self.right = np.where(leaf, idx, self.right)
        self._f = np.maximum(self.feature, 0)
        self._compile()

    @property
    def n_trees(self) -> int:
        return len(self.roots)

    def _mask_unknown(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        for k, j in enumerate(self.known_cols):
            x = X[:, j]
            ok = np.isfinite(x) & (x >= 0) & (x < 256)
            xi = np.where(ok, x, 0).astype(np.int64)
            seen = ok & (((self.known[k, xi // 32] >> (xi % 32).astype(np.uint32)) & 1) == 1)
            X[~seen, j] = np.nan
        return X

    def raw_reference(self, X: np.ndarray) -> np.ndarray:
        """Straightforward node-by-node walk (reference for raw())."""
        X = self._mask_unknown(X)
        n = len(X)
        if n == 0 or self.n_trees == 0:
            return np.full(n, self.baseline)
        node = np.broadcast_to(self.roots, (n, self.n_trees)).copy()
        rows = np.arange(n)[:, None]
        for _ in range(self.depth):
            f = self._f[node]
            x = X[rows, f]
            nan = np.isnan(x)
            go_left = np.where(nan, self.missing_left[node], x <= self.threshold[node])
            c = self.cat[node]
            iscat = (c >= 0) & ~nan
            if iscat.any():
                xc = np.where(iscat, x, 0).astype(np.int64)
                word = self.bitsets[np.maximum(c, 0), np.clip(xc // 32, 0, 7)]
                inset = ((word >> (xc % 32).astype(np.uint32)) & 1).astype(bool) & (xc >= 0) & (xc < 256)
                go_left = np.where(iscat, inset, go_left)
            node = np.where(go_left, self.left[node], self.right[node])
        return self.baseline + self.value[node].sum(axis=1)

    # ------------------------------------------------------------------ compiled fast path
    def _compile(self):
        """Complete-binary-tree layout of depth D per tree, so one walk step is
        ``g = 2g + off + go_right`` on global slot indices. Missing values: every numeric split
        reads its own copy of the input column, column j (NaN -> -inf: NaN goes left) or
        j + n_features (NaN -> +inf: NaN goes right). Categorical splits read a code column
        (NaN -> 256) and look the decision up in a per-split 257-entry table. Shallow leaves get
        always-left dummy splits (threshold +inf on the -inf copy)."""
        D = max(self.depth, 1)
        n_int, n_leaf = 2 ** D - 1, 2 ** D
        T = self.n_trees
        nfeat = int(self.feature.max()) + 1 if (self.feature >= 0).any() else 1
        cat_feats = sorted({int(self.feature[g]) for g in np.flatnonzero(self.cat >= 0)})
        self._nfeat, self._cat_feats = nfeat, np.array(cat_feats, dtype=np.int64)
        code_col = np.zeros(max(nfeat, 1), dtype=np.int64)
        for k, f in enumerate(cat_feats):
            code_col[f] = 2 * nfeat + k
        feat, thr, val = self.feature, self.threshold, self.value
        # (tree, level, position) of every node, level by level from the roots (vectorised)
        N = len(feat)
        tree = np.repeat(np.arange(T), np.diff(np.r_[self.roots, N]))
        lev = np.full(N, -1, dtype=np.int64)
        pos = np.zeros(N, dtype=np.int64)
        cur = self.roots.copy()
        lev[cur], pos[cur] = 0, 0
        for d in range(D):
            cur = cur[feat[cur] >= 0]
            if not len(cur):
                break
            for child, add in ((self.left[cur], 0), (self.right[cur], 1)):
                lev[child] = d + 1
                pos[child] = 2 * pos[cur] + add
            cur = np.r_[self.left[cur], self.right[cur]]
        assert (lev >= 0).all()
        F = np.zeros((T, n_int), dtype=np.int64)
        TH = np.full((T, n_int), np.inf)
        CB = np.full((T, n_int), -1, dtype=np.int64)        # base offset into LUT, -1 = numeric
        LV = np.zeros((T, n_leaf))
        leaf = feat < 0
        LV[tree[leaf], pos[leaf] << (D - lev[leaf])] = val[leaf]
        inner = np.flatnonzero(~leaf)
        slot = (1 << lev[inner]) - 1 + pos[inner]
        ti = tree[inner]
        isc = self.cat[inner] >= 0
        miss = self.missing_left[inner]
        big = np.finfo(np.float64).max
        fnum = np.where(miss, feat[inner], feat[inner] + nfeat)
        tnum = np.where(miss | (thr[inner] != np.inf), thr[inner], big)
        F[ti, slot] = np.where(isc, code_col[np.maximum(feat[inner], 0)], fnum)
        TH[ti, slot] = np.where(isc, np.inf, tnum)
        cn = inner[isc]
        if len(cn):
            codes = np.arange(256)
            words = self.bitsets[self.cat[cn]]                                   # [n_cat, 8]
            inset = ((words[:, codes // 32] >> (codes % 32).astype(np.uint32)) & 1).astype(bool)
            lut = np.concatenate([~inset, ~self.missing_left[cn][:, None]], axis=1).astype(np.int64)
            CB[ti[isc], slot[isc]] = 257 * np.arange(len(cn))
            luts = [lut.ravel()]
        else:
            luts = []
        off = np.arange(T) * n_int
        self._D, self._F, self._TH, self._CB = D, F.ravel(), TH.ravel(), CB.ravel()
        self._LUT = luts[0] if luts else np.zeros(1, dtype=np.int64)
        self._has_cat = bool(luts)
        self._LV = LV.ravel()
        self._root = off[None, :]
        self._step_off = (1 - off)[None, :]               # g' = 2g + (1 - off) + right
        self._leaf_off = (np.arange(T) * n_leaf - n_int - off)[None, :]
        self._ncol = 2 * nfeat + len(cat_feats)

    def _inputs(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        Xm = X[:, :self._nfeat]
        nan = np.isnan(Xm)
        X2 = np.empty((n, self._ncol))
        X2[:, :self._nfeat] = np.where(nan, -np.inf, Xm)
        X2[:, self._nfeat:2 * self._nfeat] = np.where(nan, np.inf, Xm)
        if len(self._cat_feats):
            c = X[:, self._cat_feats]
            X2[:, 2 * self._nfeat:] = np.where(np.isnan(c), 256, c)
        return X2

    def raw(self, X: np.ndarray) -> np.ndarray:
        """baseline + sum of the leaf values (sklearn HistGradientBoosting semantics)."""
        X = self._mask_unknown(X)
        n = len(X)
        if n == 0 or self.n_trees == 0:
            return np.full(n, self.baseline)
        X2 = self._inputs(X).ravel()
        base = (np.arange(n) * self._ncol)[:, None]
        g = np.broadcast_to(self._root, (n, self.n_trees))
        F, TH, CB, LUT = self._F, self._TH, self._CB, self._LUT
        for _ in range(self._D):
            x = X2[base + F[g]]
            right = x > TH[g]
            if self._has_cat:
                cb = CB[g]
                isc = cb >= 0
                if isc.any():
                    right = np.where(isc, LUT[np.where(isc, cb + np.where(isc, x, 0).astype(np.int64), 0)] == 1, right)
            g = 2 * g + self._step_off + right
        return self.baseline + self._LV[g + self._leaf_off].sum(axis=1)


class CalibMap:
    """1-D map from boosted log-odds to a calibrated probability."""

    def __init__(self, kind: str, params: np.ndarray, xs: np.ndarray | None = None):
        self.kind = kind
        self.params = np.asarray(params, dtype=np.float64)
        self.xs = None if xs is None else np.asarray(xs, dtype=np.float64)

    def __call__(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        if self.kind == "platt":
            a, b = self.params
            p = sigmoid(a * z + b)
        elif self.kind == "isotonic":
            p = np.interp(z, self.xs, self.params)
        else:                                   # "none"
            p = sigmoid(z)
        return np.clip(p, P_CLIP, 1 - P_CLIP)


class Head:
    def __init__(self, name: str, trees: TreeEnsemble, kind: str, calib: CalibMap | None,
                 clip: tuple[float, float] | None = None):
        self.name, self.trees, self.kind, self.calib, self.clip = name, trees, kind, calib, clip

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Regression heads: the prediction (clipped to the label range); probability heads: the
        calibrated probability."""
        r = self.trees.raw(X)
        if self.kind == "reg":
            return r if self.clip is None else np.clip(r, *self.clip)
        return self.calib(r)


class LevelModel:
    """All heads of one level + the input specification (numeric columns, categorical columns
    with their vocabularies)."""

    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as z:
            arr = {k: z[k] for k in z.files}
        self.meta = json.loads(bytes(arr.pop("meta")).decode("utf-8"))
        self.path = str(path)
        self.num: list[str] = self.meta["num"]
        self.cat: list[str] = self.meta["cat"]
        self.vocab: dict[str, list[str]] = self.meta.get("vocab", {})
        self.encoding: str = self.meta.get("encoding", "codes")
        self.heads: dict[str, Head] = {}
        for name, h in self.meta["heads"].items():
            g = {k[len(name) + 1:]: v for k, v in arr.items() if k.startswith(name + "/")}
            trees = TreeEnsemble(g)
            cal = None
            if h["kind"] == "prob":
                c = h["calib"]
                cal = CalibMap(c["kind"], g.get("calib_y", np.array(c.get("params", []))),
                               g.get("calib_x"))
            self.heads[name] = Head(name, trees, h["kind"], cal, tuple(h["clip"]) if h.get("clip") else None)

    def matrix(self, t: dict) -> np.ndarray:
        from .features import matrix

        return matrix(t, self.num, self.cat, self.vocab, self.encoding)

    def predict(self, t: dict | None = None, X: np.ndarray | None = None,
                heads=None) -> dict[str, np.ndarray]:
        """{head: output} (all heads, or only `heads`); P(strict) is capped at P(lenient) when
        both are computed."""
        X = self.matrix(t) if X is None else X
        out = {k: h.predict(X) for k, h in self.heads.items() if heads is None or k in heads}
        if "len" in out and "str" in out:
            out["str"] = np.minimum(out["str"], out["len"])
        return out

    def logits(self, t: dict | None = None, X: np.ndarray | None = None,
               heads=None) -> dict[str, np.ndarray]:
        out = self.predict(t, X, heads)
        res = {k: v for k, v in out.items() if k not in ("len", "str")}
        if "len" in out:
            res["z_len"] = logit(out["len"])
        if "str" in out:
            res["z_str"] = np.minimum(logit(out["str"]), res.get("z_len", np.inf))
        return res


# ----------------------------------------------------------------------------- packaged models

def data_dir() -> Path:
    return Path(str(resources.files("pronunciada").joinpath("data/calib")))


@lru_cache(maxsize=None)
def load_level(name: str, directory: str | None = None) -> LevelModel:
    p = Path(directory) if directory else data_dir()
    return LevelModel(p / f"{name}.npz")


def load_projection(name: str, directory: str | None = None) -> dict | None:
    """Linear projection of pooled encoder activations shipped inside a level file (round-5
    sentence model): arrays proj/mu, proj/sd, proj/W, proj/b and meta["projection"] =
    {"names": [...], "layers": [...]}. None when the file has none."""
    p = (Path(directory) if directory else data_dir()) / f"{name}.npz"
    with np.load(p, allow_pickle=False) as z:
        if "proj/W" not in z.files:
            return None
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))["projection"]
        return {"names": list(meta["names"]), "layers": list(meta["layers"]),
                **{k: z[f"proj/{k}"].astype(np.float64) for k in ("mu", "sd", "W", "b")}}


def load_linear(name: str, directory: str | None = None) -> dict | None:
    """Ridge heads blended with the trees (round-5 sentence model): arrays lin/med, lin/lo,
    lin/hi, lin/mu, lin/sd, lin/W, lin/b and meta["linear_blend"] = {"heads", "tree_weight": w}.
    a = clip(x with NaN -> med, lo, hi);  output = clip(w * trees + (1 - w) * ((a - mu) / sd) @ W + b, 0, 10)."""
    p = (Path(directory) if directory else data_dir()) / f"{name}.npz"
    with np.load(p, allow_pickle=False) as z:
        if "lin/W" not in z.files:
            return None
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))["linear_blend"]
        return {"heads": list(meta["heads"]), "w": float(meta["tree_weight"]),
                **{k: z[f"lin/{k}"].astype(np.float64) for k in ("med", "lo", "hi", "mu", "sd", "W", "b")}}


def apply_linear_blend(lin: dict | None, X: np.ndarray, out: dict) -> dict:
    """Blend the regression outputs `out` (already clipped trees) with the ridge heads."""
    if lin is None:
        return out
    A = np.clip(np.where(np.isnan(X), lin["med"], X), lin["lo"], lin["hi"])
    z = ((A - lin["mu"]) / lin["sd"]) @ lin["W"] + lin["b"]
    out = dict(out)
    for j, h in enumerate(lin["heads"]):
        out[h] = np.clip(lin["w"] * np.asarray(out[h], dtype=np.float64) + (1 - lin["w"]) * z[:, j], 0.0, 10.0)
    return out


@lru_cache(maxsize=None)
def config(directory: str | None = None) -> dict:
    """pronunciada/data/calib/config.json: which sound model each language uses and its bias."""
    p = (Path(directory) if directory else data_dir()) / "config.json"
    return json.loads(p.read_text(encoding="utf-8"))


def sound_model_for(lang: str, directory: str | None = None) -> tuple[str, float]:
    """(sound model name, logit bias) shipped for `lang`."""
    c = config(directory)
    m = c.get("sound_model", {}).get(lang, c["universal_sound_model"])
    b = 0.0 if m != c["universal_sound_model"] else float(c.get("bias", {}).get(lang, 0.0))
    return m, b


# ----------------------------------------------------------------------------- export helper

def save_level(path: str | Path, num: list[str], cat: list[str], vocab: dict, heads: dict,
               extra_meta: dict | None = None, encoding: str = "mixed") -> None:
    """heads: {name: {"kind": "reg"|"prob", "trees": {array dict}, "clip": (lo, hi) | None,
    "calib": {"kind": "platt"|"isotonic"|"none", "params": [...], "x": [...]}}}"""
    arrays = {}
    meta = {"num": list(num), "cat": list(cat), "vocab": vocab, "encoding": encoding, "heads": {},
            **(extra_meta or {})}
    for name, h in heads.items():
        for k, v in h["trees"].items():
            arrays[f"{name}/{k}"] = np.asarray(v)
        m = {"kind": h["kind"], "n_trees": int(len(h["trees"]["roots"])),
             "depth": int(h["trees"]["depth"])}
        if h.get("clip") is not None:
            m["clip"] = [float(h["clip"][0]), float(h["clip"][1])]
        if h["kind"] == "prob":
            c = h["calib"]
            m["calib"] = {"kind": c["kind"]}
            if c["kind"] == "platt":
                m["calib"]["params"] = [float(x) for x in c["params"]]
            elif c["kind"] == "isotonic":
                arrays[f"{name}/calib_x"] = np.asarray(c["x"], dtype=np.float32)
                arrays[f"{name}/calib_y"] = np.asarray(c["params"], dtype=np.float32)
        meta["heads"][name] = m
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **arrays)


# ----------------------------------------------------------------------------- contrast caps (round 16)

CAP_FEATS = ("llr_sub", "log_ratio_comp", "z_llr_sub", "peak_post", "comp_mean", "can_mean")


@lru_cache(maxsize=None)
def _caps_file(directory: str | None) -> dict:
    p = (Path(directory) if directory else data_dir()) / "contrast_caps.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def contrast_caps(lang: str, directory: str | None = None) -> list[dict]:
    """[{members, gate, coef, b, offset}] for the contrast classes adopted in `lang` (round 16,
    bench/calib/contrast_fix.py): a small logistic on the raw evidence of one sound, applied as a
    cap on its shown log-odds only when the expected sound is a member of the class and the top
    competing token is its contrast partner (e.g. expected p, competitor ʰ)."""
    import os

    if os.environ.get("PRONUNCIADA_CONTRAST_CAPS", "1") == "0":     # before / after comparisons
        return []
    c = _caps_file(directory)
    out = []
    for cls, off in c.get("langs", {}).get(lang, {}).items():
        m = c["classes"][cls]
        out.append({"class": cls, "members": set(m["members"][lang]), "gate": m["gate"],
                    "coef": np.asarray(m["coef"], np.float64), "b": float(m["b"]), "offset": float(off)})
    return out


def contrast_cap_z(caps: list[dict], tab: dict) -> np.ndarray:
    """Per sound, the lowest round-16 cap log-odds among the caps that fire (NaN where none)."""
    n = len(tab["phone"])
    out = np.full(n, np.nan)
    if not caps or not n:
        return out
    X = np.column_stack([np.clip(np.nan_to_num(np.asarray(tab[k], np.float64), nan=0.0, posinf=30.0, neginf=-30.0),
                                 -30.0, 30.0) for k in CAP_FEATS])
    phone, comp = list(tab["phone"]), list(tab["comp_tok"])
    for c in caps:
        g = c["gate"]
        fire = np.array([phone[i] in c["members"] and comp[i] in g.get(phone[i], g.get("*", ()))
                         for i in range(n)], bool)
        if fire.any():
            z2 = X[fire] @ c["coef"] + c["b"] + c["offset"]
            cur = out[fire]
            out[fire] = np.where(np.isfinite(cur), np.minimum(cur, z2), z2)
    return out


def apply_contrast_caps(caps: list[dict], tab: dict, zl: np.ndarray, zs: np.ndarray):
    """Shown sound log-odds with the contrast caps applied (zl, zs; unchanged where no cap fires)."""
    if not caps:
        return zl, zs
    n = len(zl)
    X = np.column_stack([np.clip(np.nan_to_num(np.asarray(tab[k], np.float64), nan=0.0, posinf=30.0, neginf=-30.0),
                                 -30.0, 30.0) for k in CAP_FEATS])
    phone, comp = list(tab["phone"]), list(tab["comp_tok"])
    zl, zs = np.array(zl, np.float64), np.array(zs, np.float64)
    for c in caps:
        g = c["gate"]
        fire = np.array([phone[i] in c["members"] and comp[i] in g.get(phone[i], g.get("*", ()))
                         for i in range(n)], bool)
        if fire.any():
            z2 = X[fire] @ c["coef"] + c["b"] + c["offset"]
            zl[fire] = np.minimum(zl[fire], z2)
            zs[fire] = np.minimum(zs[fire], z2)
    return zl, zs
