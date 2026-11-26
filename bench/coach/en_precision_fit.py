"""Refit the English balanced phoneme-class thresholds on SpeechOcean762 train.

From the repository root, install the bench extra (`pip install -e ".[bench]"`)
and run `python -m bench.coach.en_precision_fit --prepare` to fetch train if
needed, generate train inputs and print the fitted balanced setting as JSON.
Omit `--prepare` on subsequent runs to reuse the input caches. Use a fresh
PRONUNCIADA_CACHE for preparation to keep other benchmark caches intact.

The historical scalar balanced threshold is the baseline. Speaker-grouped
nested folds refit the sound detail and coaching heads; the held-out scores
from those folds fit the final class thresholds. All folds are on official
train. The output goes to stdout; progress goes to stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

from bench.calib import common, train_sound_emb
from bench.calib.train_sound import KINDS
from bench.coach.fit_v2 import fit_logistic
from bench.data import load
from bench.pipeline import features as pipeline
from pronunciada import calibrate, coach, features

ROOT = Path(__file__).resolve().parents[2]
DATASET = "speechocean762"
BASELINE_THRESHOLD = 0.48919697137923074
EPS = 1e-6


def safe_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def select_threshold(scores, y, base_tp, base_fp):
    if not len(scores):
        return None
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    tp = np.cumsum(y[order])[ends]
    fp = ends + 1 - tp
    valid = np.flatnonzero((tp >= base_tp) & (fp < base_fp))
    if not valid.size:
        return None
    i = max(valid, key=lambda j: (tp[j] / (tp[j] + fp[j]), tp[j], s[ends[j]]))
    return float(s[ends[i]])


def shifted_threshold(p, delta, threshold):
    """Export the inclusive boundary, correcting floating-point round-trip at train ties."""
    boundary = threshold - delta
    if threshold <= safe_logit(0) + delta:
        return 0.0
    if threshold > safe_logit(1) + delta:
        return float(np.nextafter(1.0, np.inf))
    result = float(expit(boundary))
    included = safe_logit(p) + delta >= threshold
    if np.any(included):
        result = min(result, float(p[included].min()))
    if np.any(~included):
        result = max(result, float(np.nextafter(p[~included].max(), np.inf)))
    if not np.array_equal(p >= result, included):
        raise ValueError("Exported threshold differs from the inclusive logit rule")
    return result


def masks(p, classes, eligible, rule):
    cuts = np.array([rule["thresholds"].get(c, rule["fallback"]) for c in classes])
    return eligible & (p >= cuts)


def tune(p, classes, y, eligible, base):
    z = safe_logit(p)
    offsets = {}
    corrected = z.copy()
    for cls in np.unique(classes[eligible]):
        idx = eligible & (classes == cls)
        zi, yi = z[idx], y[idx]
        delta = brentq(lambda d: float((expit(zi + d) - yi).sum() + d), -50, 50)
        offsets[cls] = delta
        corrected[classes == cls] += delta
    threshold = select_threshold(corrected[eligible], y[eligible],
                                 int((base & y).sum()), int((base & ~y).sum()))
    if threshold is None:
        return None
    return {"logit_threshold": threshold, "offsets": offsets, "fallback": float(expit(threshold)),
            "thresholds": {cls: shifted_threshold(p[classes == cls], d, threshold)
                           for cls, d in offsets.items()}}


def probabilities(F, y, tr, te):
    return coach._logistic(fit_logistic(F[tr], y[tr]), F[te])


def confusion(y, flagged):
    tp = int((y & flagged).sum())
    fp = int((~y & flagged).sum())
    fn = int(y.sum()) - tp
    tn = len(y) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.}


def prepare_table(train):
    import pandas as pd

    cfg = pipeline.config("en")
    rows = []
    for i, rec in enumerate(train, 1):
        phones, _, _ = pipeline.so762_utt((dict(rec, _dataset=DATASET), "en", cfg))
        if not phones:
            raise ValueError(f"No feature rows for {rec['id']}")
        rows.extend(phones)
        if i % 500 == 0:
            print(f"Train table: {i}", flush=True)
    pipeline._write(pd.DataFrame(rows), pipeline.feat_dir() / f"{DATASET}.parquet")


def prepare_prototypes():
    from bench.calib.sound_emb import proto_path

    path = ROOT / "pronunciada/data/calib/sound_E_emb.npz"
    with np.load(path) as shipped:
        np.savez(proto_path("en"), **{k: shipped["proto_" + k] for k in ("mu", "sd", "P", "phones")})


def checked_folds(groups):
    folds = common.speaker_folds(groups)
    visits = np.zeros(len(groups), dtype=int)
    for tr, te in folds:
        if np.intersect1d(groups[tr], groups[te]).size:
            raise ValueError("A speaker appears in both training and validation")
        visits[te] += 1
    if not np.all(visits == 1):
        raise ValueError("Fold validation does not cover each row once")
    return folds


def detail_oof(train):
    df, table, labels, groups, _, vectors = train_sound_emb.load()
    speakers = {r["id"]: r["speaker"] for r in train}
    if (set(df.utt) != set(speakers) or not df.utt.map(speakers).eq(df.speaker).all()
            or df.duplicated(["utt", "phone_index"]).any() or not np.isfinite(vectors).all()):
        raise ValueError("Incomplete or incompatible train features/vectors")

    model_name = calibrate.config()["sound_detail_model"]["en"]
    model = calibrate.load_level(model_name)
    spec = features.SOUND_SETS[train_sound_emb.SPEC]
    if model_name != "sound_E" or model.num != list(spec["num"]) + train_sound_emb.EXTRA:
        raise ValueError("Shipped English detail model has different inputs")
    if model.cat != list(spec["cat"]) or model.encoding != "mixed":
        raise ValueError("Unexpected English detail encoding")
    cat_mask = features.categorical_mask(model.num, model.cat, model.vocab, model.encoding)
    params = common.load_json("calib_sound.json")["params"]
    splits = checked_folds(groups)
    fold_ids = np.full(len(df), -1)
    for fold, (_, te) in enumerate(splits):
        fold_ids[te] = fold
    ok_vectors = np.isfinite(vectors[:, 0])

    @lru_cache(maxsize=None)
    def held_logits(excluded):
        held = np.isin(fold_ids, excluded)
        tr, te = np.flatnonzero(~held), np.flatnonzero(held)
        projection = train_sound_emb.projection_for(vectors, labels["reg"], groups, tr, te, ok_vectors)
        x = model.matrix(dict(table, emb_proj=projection))
        logits = np.full((len(df), 2), np.nan)
        for j, head in enumerate(("len", "str")):
            y = labels[head]
            labelled = tr[np.isfinite(y[tr])]
            fitted = common.fit(KINDS[head], params[head], x[labelled], y[labelled], cat_mask)
            logits[te, j] = common.raw(fitted, x[te])
        print(f"Sound_E raw logits fitted with folds {excluded} held out", flush=True)
        return logits

    def calibrated(excluded, target):
        available = sorted(set(range(len(splits))) - set(excluded))
        # Reuse logits from the next level of exclusion to fit each calibration map.
        # For nested exclusions use one calibration fold to limit the number of fits.
        if len(excluded) > 1:
            available = available[:1]
        cal_parts = [splits[k][1] for k in available]
        cal_logits = np.concatenate([
            held_logits(tuple(sorted((*excluded, k))))[rows]
            for k, rows in zip(available, cal_parts, strict=True)
        ])
        cal_rows = np.concatenate(cal_parts)
        raw = held_logits(excluded)[target]
        logits = np.empty_like(raw)
        for j, head in enumerate(("len", "str")):
            y = labels[head][cal_rows]
            valid = np.isfinite(y)
            cal = common.fit_calib(cal_logits[valid, j], y[valid], model.heads[head].calib.kind)
            p = common.apply_calib(cal, raw[:, j])
            p = np.clip(p, 1e-6, 1 - 1e-6)
            logits[:, j] = np.log(p / (1 - p))
        logits[:, 1] = np.minimum(logits[:, 0], logits[:, 1])
        return logits

    def add_logits(frame, prefix, logits):
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite detail OOF logits")
        frame[prefix + "_z_len"], frame[prefix + "_z_str"] = logits.T

    logits = np.empty((len(df), 2))
    out = df[["utt", "phone_index", "speaker", "phone", "phone_score"]].copy()
    out["fold"] = fold_ids
    for fold, (_, te) in enumerate(splits):
        logits[te] = calibrated((fold,), te)
    for outer, (_, te) in enumerate(splits):
        nested = np.empty_like(logits)
        nested[te] = logits[te]
        for inner, (_, val) in enumerate(splits):
            if inner != outer:
                nested[val] = calibrated(tuple(sorted((outer, inner))), val)
        add_logits(out, f"E_outer{outer}", nested)
        for inner in sorted(set(range(len(splits))) - {outer}):
            inner_features = nested.copy()
            for k, (_, train_rows) in enumerate(splits):
                if k not in (outer, inner):
                    inner_features[train_rows] = calibrated(tuple(sorted((outer, inner, k))), train_rows)
            add_logits(out, f"E_outer{outer}_inner{inner}", inner_features)
    return out


def assemble_inputs(detail, dump, train):
    detail = detail.set_index(["utt", "phone_index"])
    if not detail.index.is_unique:
        raise ValueError("Duplicate train phone keys")
    records = {r["id"]: r["speaker"] for r in train}
    if set(dump) != set(records) or set(detail.index.get_level_values("utt")) != set(records):
        raise ValueError("Feature and coaching inputs must cover the train recordings")
    if detail.groupby("speaker").fold.nunique().ne(1).any() or set(detail.fold) != set(range(5)):
        raise ValueError("Invalid detail fold assignments")

    prefixes = {(outer, None): f"E_outer{outer}" for outer in range(5)}
    prefixes.update({(outer, inner): f"E_outer{outer}_inner{inner}"
                     for outer in range(5) for inner in range(5) if inner != outer})
    matrices = {pair: [] for pair in prefixes}
    labels, classes, speakers, folds, eligible = [], [], [], [], []
    missing_word_prob = coach.gate_params()["missing_word_prob"]
    for uid, speaker in sorted(records.items()):
        row = dump[uid]
        if "error" in row:
            raise ValueError(f"Could not score {uid}: {row['error']}")
        X = row["sounds"]
        d = detail.loc[uid]
        expected = X["expected"]
        if (not np.array_equal(d.index, np.arange(len(expected)))
                or not d.speaker.eq(speaker).all() or d.phone.tolist() != list(expected)):
            raise ValueError(f"Incompatible phone alignment for {uid}")
        omitted = np.array([w["present_prob"] is not None and w["present_prob"] < missing_word_prob
                            for w in row["words"]], dtype=bool)
        scores = d.phone_score.to_numpy()
        valid = np.isfinite(scores)
        for pair, prefix in prefixes.items():
            zl, zs = d[prefix + "_z_len"].to_numpy(), d[prefix + "_z_str"].to_numpy()
            if not np.isfinite(zl).all() or not np.isfinite(zs).all():
                raise ValueError(f"Nonfinite detail logits for {uid}, folds {pair}")
            matrices[pair].append(coach.feature_matrix(dict(X, zl=zl, zs=zs))[valid])
        labels.extend(scores[valid] < 2)
        classes.extend(coach.sound_class(phone) for phone, keep in zip(expected, valid, strict=True) if keep)
        speakers.extend(d.speaker.to_numpy()[valid])
        folds.extend(d.fold.to_numpy()[valid])
        eligible.extend((~omitted[X["word"]])[valid])
    return ({pair: np.concatenate(rows) for pair, rows in matrices.items()},
            np.asarray(labels), np.asarray(classes), np.asarray(speakers),
            np.asarray(folds), np.asarray(eligible))


def evaluate(matrices, y, classes, groups, fold_ids, eligible, threshold):
    base = np.zeros(len(y), bool)
    candidate = base.copy()
    all_scores = np.empty(len(y))
    folds = []
    for fold in np.unique(fold_ids):
        held = fold_ids == fold
        tr, te = np.flatnonzero(~held), np.flatnonzero(held)
        if not len(te) or np.intersect1d(groups[tr], groups[te]).size:
            raise ValueError("Invalid outer split by speaker")
        p = np.full(len(tr), np.nan)
        for inner in np.unique(fold_ids[tr]):
            val = fold_ids[tr] == inner
            p[val] = probabilities(matrices[fold, inner], y, tr[~val], tr[val])
        rule = tune(p, classes[tr], y[tr], eligible[tr], eligible[tr] & (p >= threshold))
        all_scores[te] = probabilities(matrices[fold, None], y, tr, te)
        base[te] = eligible[te] & (all_scores[te] >= threshold)
        candidate[te] = base[te] if rule is None else masks(all_scores[te], classes[te], eligible[te], rule)
        folds.append({"fold": int(fold), "held_speakers": sorted(set(groups[te])), "has_rule": rule is not None,
                      "baseline": confusion(y[te], base[te]), "candidate": confusion(y[te], candidate[te])})
    return all_scores, base, candidate, folds


def fit_thresholds(matrices, y, classes, speakers, folds, eligible):
    scores, base, candidate, results = evaluate(
        matrices, y, classes, speakers, folds, eligible, BASELINE_THRESHOLD,
    )
    a, b = confusion(y, base), confusion(y, candidate)
    print(f"Train OOF baseline: {a}; candidate: {b}", file=sys.stderr)
    if not (b["fp"] < a["fp"] and b["tp"] >= a["tp"] and all(f["has_rule"] for f in results)):
        raise ValueError("The candidate did not improve the train out-of-fold operating point")
    rule = tune(scores, classes, y, eligible, base)
    if rule is None:
        raise ValueError("No feasible final train threshold")
    return {"fallback": rule["fallback"], "thresholds": rule["thresholds"]}


def main():
    from bench.calib import sound_emb
    from bench.coach import dump
    from bench.data import speechocean762
    from bench.data.common import data_dir, merge_parts

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepare", action="store_true", help="regenerate the SpeechOcean762 train inputs")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        root = data_dir(DATASET)
        if args.prepare and not (root / "manifest.jsonl").exists():
            speechocean762.fetch_split("train")
            merge_parts(root, ("train",))
        train = load(DATASET, "train")
        if len(train) != 2500 or len({r["id"] for r in train}) != 2500 or len({r["speaker"] for r in train}) != 125:
            raise ValueError("Expected 2500 distinct train recordings from 125 speakers")
        if args.prepare:
            prepare_table(train)
            prepare_prototypes()
            sound_emb.so762(workers=1)
            dump.run("so762-train", workers=1)
        detail = detail_oof(train)
        inputs = assemble_inputs(detail, dump.load_dump("so762-train"), train)
        setting = fit_thresholds(*inputs)
    print(json.dumps(setting, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
