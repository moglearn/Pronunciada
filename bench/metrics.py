"""Evaluation metrics (numpy / scipy only).

Conventions: `y` are binary labels (1 = positive class), `s` are scores where larger means
"more positive". For error detection use y = 1 for "mispronounced" and s = 1 - P(ok) (or -logit).
All functions ignore nothing silently: NaNs raise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats


def _arr(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if np.isnan(a).any():
        raise ValueError("NaN in input")
    return a


def _bin(y) -> np.ndarray:
    a = np.asarray(y).ravel()
    u = np.unique(a)
    if not set(u.tolist()) <= {0, 1, True, False}:
        raise ValueError(f"labels must be binary 0/1, got {u[:5]}")
    return a.astype(int)


# ----------------------------------------------------------------------------- correlation

def pearson(x, y) -> float:
    x, y = _arr(x), _arr(y)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    x, y = _arr(x), _arr(y)
    return pearson(stats.rankdata(x), stats.rankdata(y))


# ----------------------------------------------------------------------------- ranking

def roc_auc(y, s) -> float:
    """Area under the ROC curve (Mann-Whitney U with tie correction)."""
    y, s = _bin(y), _arr(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_curve(y, s) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds) with decision rule s >= threshold; starts at (0, 0)."""
    y, s = _bin(y), _arr(s)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    distinct = np.r_[np.where(np.diff(s_sorted))[0], len(s) - 1]
    tps = np.cumsum(y_sorted)[distinct]
    fps = (distinct + 1) - tps
    P, N = max(y.sum(), 1), max(len(y) - y.sum(), 1)
    return (np.r_[0.0, fps / N], np.r_[0.0, tps / P], np.r_[np.inf, s_sorted[distinct]])


def _pr_points(y, s):
    y, s = _bin(y), _arr(s)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    distinct = np.r_[np.where(np.diff(s_sorted))[0], len(s) - 1]
    tps = np.cumsum(y_sorted)[distinct].astype(float)
    pred_pos = (distinct + 1).astype(float)
    precision = tps / pred_pos
    recall = tps / max(y.sum(), 1)
    return precision, recall, s_sorted[distinct]


def pr_auc(y, s) -> float:
    """Average precision (step-wise PR AUC, as sklearn.metrics.average_precision_score)."""
    y = _bin(y)
    if y.sum() == 0:
        return float("nan")
    precision, recall, _ = _pr_points(y, s)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


@dataclass
class PRF:
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def prf_at(y, s, threshold: float) -> PRF:
    """Precision / recall / F1 for the rule `s >= threshold` -> positive."""
    y, s = _bin(y), _arr(s)
    pred = s >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return PRF(float(threshold), p, r, f, tp, fp, fn)


def best_f1(y, s) -> PRF:
    """Threshold (one of the observed scores) maximising F1."""
    y = _bin(y)
    precision, recall, thr = _pr_points(y, s)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    i = int(np.argmax(f1))
    return prf_at(y, s, float(thr[i]))


def eer(y, s) -> tuple[float, float]:
    """Equal error rate and its threshold (linear interpolation where FPR crosses FNR)."""
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    d = fpr - fnr                      # increasing from -1 to +1 (roughly)
    i = int(np.argmax(d >= 0))
    if i == 0:
        return float(fpr[0]), float(thr[0])
    d0, d1 = d[i - 1], d[i]
    w = 0.0 if d1 == d0 else -d0 / (d1 - d0)
    e = fpr[i - 1] + w * (fpr[i] - fpr[i - 1])
    t = thr[i] if not np.isfinite(thr[i - 1]) else thr[i - 1] + w * (thr[i] - thr[i - 1])
    return float(e), float(t)


# ----------------------------------------------------------------------------- calibration

def brier(y, p) -> float:
    y, p = _bin(y), _arr(p)
    return float(np.mean((p - y) ** 2))


def reliability_table(y, p, n_bins: int = 10, strategy: str = "width") -> list[dict]:
    """Bins of predicted probability. strategy "width" = equal-width on [0,1], "mass" = equal-count.
    Returns rows {lo, hi, n, mean_p, frac_pos}; empty bins are omitted."""
    y, p = _bin(y), _arr(p)
    if strategy == "width":
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    elif strategy == "mass":
        order = np.argsort(p, kind="mergesort")
        idx = np.empty(len(p), dtype=int)
        idx[order] = np.minimum((np.arange(len(p)) * n_bins) // max(len(p), 1), n_bins - 1)
    else:
        raise ValueError(strategy)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin": b, "lo": float(p[m].min()) if strategy == "mass" else float(edges[b]),
                     "hi": float(p[m].max()) if strategy == "mass" else float(edges[b + 1]),
                     "n": int(m.sum()), "mean_p": float(p[m].mean()), "frac_pos": float(y[m].mean())})
    return rows


def ece(y, p, n_bins: int = 10, strategy: str = "width") -> float:
    """Expected calibration error: sum_b n_b/N * |mean_p_b - frac_pos_b|."""
    rows = reliability_table(y, p, n_bins, strategy)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] / n * abs(r["mean_p"] - r["frac_pos"]) for r in rows))


# ----------------------------------------------------------------------------- bootstrap

def _cluster_index(clusters, n: int):
    if clusters is None:
        return None
    c = np.asarray(clusters)
    if len(c) != n:
        raise ValueError("clusters must have one entry per item")
    uniq, inv = np.unique(c, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    return members


def _resample(rng, n: int, members):
    if members is None:
        return rng.integers(0, n, n)
    pick = rng.integers(0, len(members), len(members))
    return np.concatenate([members[k] for k in pick])


def bootstrap_ci(metric: Callable[..., float], *arrays, clusters: Sequence | None = None,
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap CI of metric(*arrays). Items are resampled with replacement; when
    `clusters` is given (e.g. utterance or speaker id per item) whole clusters are resampled.
    Returns {value, lo, hi, n_boot_valid}."""
    arrs = [np.asarray(a) for a in arrays]
    n = len(arrs[0])
    if any(len(a) != n for a in arrs):
        raise ValueError("arrays must have equal length")
    members = _cluster_index(clusters, n)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = _resample(rng, n, members)
        v = metric(*[a[idx] for a in arrs])
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    lo, hi = (np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if len(vals) else (np.nan, np.nan))
    return {"value": float(metric(*arrs)), "lo": float(lo), "hi": float(hi),
            "n_boot_valid": int(len(vals))}


def paired_bootstrap(metric: Callable[..., float], y, s_a, s_b, clusters: Sequence | None = None,
                     n_boot: int = 2000, higher_is_better: bool = True, seed: int = 0) -> dict:
    """Paired bootstrap test of "system A is better than system B" on the same items.
    metric(y, s) -> float. Returns {a, b, delta, lo, hi, p_value} where delta = a - b (sign-flipped
    when lower is better, so delta > 0 always means A better) and p_value is the one-sided
    bootstrap probability that A is not better (share of resamples with delta <= 0)."""
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    n = len(y)
    if not (len(s_a) == len(s_b) == n):
        raise ValueError("arrays must have equal length")
    sign = 1.0 if higher_is_better else -1.0
    members = _cluster_index(clusters, n)
    rng = np.random.default_rng(seed)
    a, b = metric(y, s_a), metric(y, s_b)
    deltas = []
    for _ in range(n_boot):
        idx = _resample(rng, n, members)
        d = sign * (metric(y[idx], s_a[idx]) - metric(y[idx], s_b[idx]))
        if np.isfinite(d):
            deltas.append(d)
    deltas = np.asarray(deltas)
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    p = (np.sum(deltas <= 0) + 1) / (len(deltas) + 1)
    return {"a": float(a), "b": float(b), "delta": float(sign * (a - b)), "lo": float(lo),
            "hi": float(hi), "p_value": float(p), "n_boot_valid": int(len(deltas))}
