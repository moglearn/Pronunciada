"""Shared training helpers for the calibrators (bench only: sklearn / pandas allowed)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "5")

from pronunciada import features as F  # noqa: E402

N_BINS = 15                  # ECE: equal-mass, 15 bins (as in the benchmark)


def calib_cache() -> Path:
    from bench.data.common import cache_root

    p = cache_root() / "calib"
    p.mkdir(parents=True, exist_ok=True)
    return p


def tables_dir() -> Path:
    from bench.pipeline.common import repo_root

    p = repo_root() / "results" / "tables"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(name: str, obj) -> None:
    (tables_dir() / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))


def load_json(name: str):
    p = tables_dir() / name
    return json.loads(p.read_text()) if p.exists() else None


def ship_dir() -> Path:
    from bench.pipeline.common import repo_root

    p = repo_root() / "pronunciada" / "data" / "calib"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------- tables

def table_columns(df) -> dict:
    """DataFrame (feature-table rows, groups contiguous and in phone order) -> column dict with
    the derived columns of pronunciada.features.derive."""
    import pandas as pd

    t = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            t[c] = s.to_numpy().astype(float)
        elif c in F.STR_COLS or not pd.api.types.is_numeric_dtype(s):
            t[c] = [None if (x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA) else x
                    for x in s.tolist()]
        else:
            t[c] = s.to_numpy(dtype=np.float64, na_value=np.nan)
    group = (df["utt"].astype(str) + "|" + df["ref_kind"].astype(str)).to_numpy()
    F.derive(t, group)
    t["_group"] = group
    return t


def load_table(dataset: str, split: str | None = None, query=None):
    import pandas as pd

    from bench.pipeline.common import feat_dir

    df = pd.read_parquet(feat_dir() / f"{dataset}.parquet")
    if split is not None:
        df = df[df.split == split]
    if query is not None:
        df = df[query(df)]
    df = df.sort_values(["utt", "ref_kind", "phone_index"], kind="stable").reset_index(drop=True)
    return df


def vocab_for(t: dict, cats: list[str]) -> dict[str, list[str]]:
    return {c: sorted({"none" if v is None else str(v) for v in t[c]}) for c in cats}


def sub(t: dict, idx) -> dict:
    idx = np.asarray(idx)
    out = {}
    for k, v in t.items():
        if isinstance(v, np.ndarray):
            out[k] = v[idx]
        else:
            out[k] = [v[i] for i in idx]
    return out


def concat(ts: list[dict], keys: list[str]) -> dict:
    out = {}
    for k in keys:
        vals = [t[k] for t in ts]
        if all(isinstance(v, np.ndarray) for v in vals):
            out[k] = np.concatenate(vals)
        else:
            out[k] = [x for v in vals for x in (v.tolist() if isinstance(v, np.ndarray) else v)]
    return out


# ----------------------------------------------------------------------------- models

HGB_GRID = {
    "g1": dict(learning_rate=0.1, max_iter=200, max_leaf_nodes=31, min_samples_leaf=40),
    "g2": dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40),
    "g3": dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=63, min_samples_leaf=100),
}
SMALL_GRID = {   # word / sentence level (fewer rows)
    "s1": dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=40),
    "s2": dict(learning_rate=0.03, max_iter=300, max_leaf_nodes=7, min_samples_leaf=30),
    "s3": dict(learning_rate=0.05, max_iter=150, max_leaf_nodes=31, min_samples_leaf=80),
}


def hgb(kind: str, params: dict, cat_mask):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    kw = dict(max_depth=6, l2_regularization=1.0, early_stopping=False, random_state=0,
              categorical_features=np.asarray(cat_mask, bool) if np.any(cat_mask) else None,
              **params)
    return HistGradientBoostingRegressor(**kw) if kind == "reg" else HistGradientBoostingClassifier(**kw)


def fit(kind, params, X, y, cat_mask, w=None):
    m = hgb(kind, params, cat_mask)
    m.fit(X, y if kind == "reg" else y.astype(int), sample_weight=w)
    return m


def raw(m, X) -> np.ndarray:
    """Regression prediction or classifier log-odds."""
    if hasattr(m, "predict_proba"):
        return m._raw_predict(X)[:, 0]
    return m.predict(X)


def _bits_to_set(b) -> set[int]:
    return {k for k in range(256) if (int(b[k // 32]) >> (k % 32)) & 1}


def _set_to_bits(vals) -> np.ndarray:
    out = np.zeros(8, dtype=np.uint32)
    for v in vals:
        v = int(v)
        if 0 <= v < 256:
            out[v // 32] |= np.uint32(1 << (v % 32))
    return out


def export_trees(m, n_features: int | None = None) -> dict:
    """sklearn HistGradientBoosting -> arrays for pronunciada.calibrate.TreeEnsemble.

    With categorical features sklearn ordinal-encodes them internally (categories seen in fit,
    unknown -> NaN) and moves them in front of the numeric columns. The export undoes both:
    feature indices refer to the ORIGINAL columns, split bitsets are over the raw integer codes,
    and `known` holds, per original column, the bitset of codes seen in fit (others = missing)."""
    pre = getattr(m, "_preprocessor", None)
    nf = int(m.n_features_in_) if n_features is None else n_features
    known = np.full((nf, 8), 0xFFFFFFFF, dtype=np.uint32)
    is_cat = np.zeros(nf, bool)
    if pre is not None:
        cat_mask = np.asarray(m.is_categorical_, bool)
        cat_idx = np.flatnonzero(cat_mask)
        num_idx = np.flatnonzero(~cat_mask)
        order = np.r_[cat_idx, num_idx]                       # transformed column -> original
        enc = pre.named_transformers_["encoder"]
        cats = {int(cat_idx[k]): np.asarray(c, dtype=np.float64) for k, c in enumerate(enc.categories_)}
        for j, c in cats.items():
            c = c[np.isfinite(c)]
            if np.any((c != np.round(c)) | (c < 0) | (c >= 256)):
                raise ValueError("categorical codes must be integers in [0, 256)")
            known[j] = _set_to_bits(c)
            is_cat[j] = True
    else:
        order = np.arange(nf)
        cats = {}
    feats, thr, left, right, val, miss, cat, roots, bits = [], [], [], [], [], [], [], [], []
    nb = 0
    depth = 0
    n_nodes = 0
    for it in m._predictors:
        (p,) = it
        nd = p.nodes
        roots.append(n_nodes)
        n_nodes += len(nd)
        isleaf = nd["is_leaf"].astype(bool)
        orig = order[np.where(isleaf, 0, nd["feature_idx"]).astype(np.int64)]
        feats.append(np.where(isleaf, -1, orig).astype(np.int16))
        thr.append(np.where(isleaf, 0.0, nd["num_threshold"]).astype(np.float64))  # float32 flips splits
        left.append(np.where(isleaf, 0, nd["left"]).astype(np.int16))
        right.append(np.where(isleaf, 0, nd["right"]).astype(np.int16))
        val.append(np.where(isleaf, nd["value"], 0.0))
        miss.append(nd["missing_go_to_left"].astype(np.uint8))
        iscat = nd["is_categorical"].astype(bool) & ~isleaf
        cat.append(np.where(iscat, nd["bitset_idx"].astype(np.int64) + nb, -1).astype(np.int16))
        b = np.asarray(p.raw_left_cat_bitsets, dtype=np.uint32).reshape(-1, 8).copy()
        # raw_left_cat_bitsets are over sklearn's ENCODED categories: map back to raw codes
        for k in range(len(b)):
            nodes_k = np.flatnonzero(iscat & (nd["bitset_idx"] == k))
            if not len(nodes_k):
                continue
            j = int(orig[nodes_k[0]])
            if j in cats:
                b[k] = _set_to_bits([cats[j][e] for e in _bits_to_set(b[k]) if e < len(cats[j])])
        bits.append(b)
        nb += len(b)
        depth = max(depth, int(p.get_max_depth()))
        assert len(nd) < 32767
    base = np.asarray(m._baseline_prediction).ravel()[0]
    return {"feature": np.concatenate(feats), "threshold": np.concatenate(thr),
            "left": np.concatenate(left), "right": np.concatenate(right),
            "value": np.concatenate(val).astype(np.float32), "missing_left": np.concatenate(miss),
            "cat": np.concatenate(cat),
            "bitsets": (np.concatenate(bits) if nb else np.zeros((0, 8), np.uint32)),
            "known": known[is_cat] if is_cat.any() else np.zeros((0, 8), np.uint32),
            "known_cols": np.flatnonzero(is_cat).astype(np.int16),
            "roots": np.asarray(roots, dtype=np.int32), "baseline": np.float64(base),
            "depth": np.int32(depth)}


def walker_parity(m, X) -> float:
    """max |numpy walker - sklearn| of the raw predictions (float32 leaf values)."""
    from pronunciada.calibrate import TreeEnsemble

    te = TreeEnsemble(export_trees(m))
    return float(np.max(np.abs(te.raw(X) - raw(m, X)))) if len(X) else 0.0


# ----------------------------------------------------------------------------- calibration

def ece(y, p, n_bins=N_BINS) -> float:
    from bench.metrics import ece as _ece

    return _ece(np.asarray(y).astype(int), np.asarray(p, float), n_bins=n_bins, strategy="mass")


def fit_calib(z, y, kind):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    z = np.asarray(z, float)
    y = np.asarray(y).astype(int)
    if kind == "platt":
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
        return {"kind": "platt", "params": [float(lr.coef_[0, 0]), float(lr.intercept_[0])]}
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(z, y)
    xs, ys = ir.X_thresholds_, ir.y_thresholds_
    return {"kind": "isotonic", "params": ys.tolist(), "x": xs.tolist()}


def apply_calib(c, z):
    from pronunciada.calibrate import CalibMap

    return CalibMap(c["kind"], np.asarray(c["params"]), None if c.get("x") is None else np.asarray(c["x"]))(z)


def choose_calib(z, y, groups, w=None):
    """Platt vs isotonic on out-of-fold log-odds: 5-fold grouped CV of the 1-D map, lower ECE
    wins. Returns (chosen map fitted on all, {kind: cv ece})."""
    from sklearn.model_selection import GroupKFold

    z = np.asarray(z, float)
    y = np.asarray(y).astype(int)
    res = {}
    for kind in ("platt", "isotonic"):
        p = np.zeros(len(z))
        for tr, te in GroupKFold(5).split(z, y, groups):
            p[te] = apply_calib(fit_calib(z[tr], y[tr], kind), z[te])
        res[kind] = ece(y, p)
    best = min(res, key=res.get)
    return fit_calib(z, y, best), res


# ----------------------------------------------------------------------------- folds

def speaker_folds(groups, k=5):
    from sklearn.model_selection import GroupKFold

    return list(GroupKFold(k).split(np.zeros(len(groups)), None, groups))
