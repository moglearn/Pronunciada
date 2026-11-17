"""Round 15: the length channel — collect, fit, evaluate (one language at a time).

    python -m bench.length.fit_length collect --lang ja --split train        # natives (fit data)
    python -m bench.length.fit_length collect --lang ja --split validation   # dev: natives + swaps
    python -m bench.length.fit_length fit --lang ja [--dry-run]              # -> pronunciada/data/length/ja.json
    python -m bench.length.heldout_runtime --lang ja                         # FLEURS test, once (shipped runtime)

Collection runs the public Scorer WITHOUT the length channel (Scorer(..., length=False)) and
stores, per scored reference, the compact timing inputs of pronunciada.length.frame_inputs, the
reference, the word probabilities at strictness 0 / 0.5 / 1 and the sound probabilities, so the
channel can be refitted offline. Items per clip:
  clean     the correct reference (natives)
  length    one planted length swap (bench/audit/contrasts.py generators of the language's length
            contrasts, both directions, at most PER_DIR per clip and direction, fixed seeds)
  sound     bench.corrupt sound errors (the five kinds, 2 errors, seeds "prosody-v1" as in
            bench/calib/fit_prosody.py)
  swap:*    the frozen near / far word-swap dev set (bench/calib/bias_grid.py build-dev),
            validation only
Fit: the sound-level model (logistic per vowel / consonant class, balanced classes) on TRAIN
natives with the text's own long / short labels; the word factor exactly as
bench/calib/fit_prosody.py fits tones / stress (its fit / evaluate, on dev rows: clean + length
swaps), with the native-quantile pair chosen by the strictest-pair rule at MAX_NATIVE_DROP.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import zlib
from pathlib import Path

import numpy as np

CONTRASTS = {"ja": ("vowel_length", "gemination"), "it": ("gemination",), "de": ("vowel_length",)}
SOUND_KINDS = ("phone_sub_near", "phone_sub_far", "phone_ins", "phone_del", "word_swap")
PER_DIR = 2
CHUNK = 25
MAX_NATIVE_DROP = 0.005
VERSION = "len_v1"


def cache_dir(lang: str, split: str) -> Path:
    from bench.data.common import cache_root

    p = cache_root() / "length" / VERSION / lang / split
    p.mkdir(parents=True, exist_ok=True)
    return p


def _records(lang: str, split: str, n: int | None):
    from bench.pipeline.common import records

    recs = records("fleurs-validation", lang=lang) if split == "validation" else \
        records("fleurs", split=split, lang=lang)
    recs = sorted(recs, key=lambda r: r["id"])
    if n and n < len(recs):
        rng = np.random.default_rng(0)
        recs = [recs[i] for i in sorted(rng.choice(len(recs), n, replace=False))]
    return recs


def _audio_iter(recs):
    from pronunciada.audio import load_audio

    pq = [r for r in recs if r["audio"].startswith("parquet:")]
    if pq:
        from bench.pipeline.common import iter_parquet_audio

        by = {int(r["audio"].split(":")[2]): r for r in pq}
        for i, wav in iter_parquet_audio(pq[0]["audio"].split(":")[1], set(by)):
            yield by[i], wav
    for r in recs:
        if not r["audio"].startswith("parquet:"):
            yield r, load_audio(r["audio"])


def _item(sc, res, tag, utt, wrong_words=None, wrong_phones=None, extra=None):
    from pronunciada import length as L

    c = res._pronunciada  # noqa: SLF001
    ref = c["ref"]
    wp = [[w.prob for w in r.words] for r in (sc.rescore(res, 0.0), res, sc.rescore(res, 1.0))]
    it = {"utt": utt, "tag": tag, "ref": ref, "inp": L.frame_inputs(ref, c["ev"], c["lpa"], _fb(sc, c)),
          "wp": np.asarray(wp, dtype=np.float64).T, "sp": np.array([p.prob for w in res.words for p in w.phones]),
          "wrong_words": np.zeros(len(ref.words), bool) if wrong_words is None else np.asarray(wrong_words, bool),
          "wrong_phones": [] if wrong_phones is None else list(wrong_phones),
          "spans": [tuple(w.char_span) for w in ref.words]}
    it.update(extra or {})
    return it


def _fb(sc, c):
    wav = c["wav"]
    import hashlib

    key = hashlib.sha1(wav.tobytes()).hexdigest()
    return sc._lp_cache[key][1] if key in sc._lp_cache else sc.am._features(wav)  # noqa: SLF001


def collect(lang: str, split: str, n: int | None, threads: int = 2) -> None:
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(threads)
    import warnings

    warnings.simplefilter("ignore")
    from bench.audit.contrasts import apply, get_contrast, sites
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.common import stable_seed
    from bench.pipeline.evalctl import inventory
    from pronunciada.scorer import Scorer

    d = cache_dir(lang, split)
    recs = _records(lang, split, n)
    natives_only = split == "train"
    dev = {}
    if split == "validation":
        from bench.calib.bias_grid import dev_swap

        for r in dev_swap(lang):
            dev.setdefault(r["src"], []).append(r)
    sc = inv = None
    for k0 in range(0, len(recs), CHUNK):
        out = d / f"chunk_{len(recs)}_{k0:05d}.pkl"
        if out.exists():
            continue
        if sc is None:
            sc = Scorer(lang, threads=threads, length=False)
            inv = None if natives_only else inventory(lang)
        items = []
        for rec, wav in _audio_iter(recs[k0:k0 + CHUNK]):
            try:
                res = sc.score((wav, 16000), rec["text"])
            except ValueError:
                continue
            ref = res._pronunciada["ref"]  # noqa: SLF001
            items.append(_item(sc, res, "clean", rec["id"]))
            if natives_only:
                continue
            for cn in CONTRASTS[lang]:
                c = get_contrast(lang, cn)
                st = sites(ref, c)
                for dr in ("A>B", "B>A"):
                    pool = [s for s in st if s.direction == dr]
                    rng = np.random.default_rng(stable_seed(rec["id"], cn, dr, "length-v1"))
                    for j in rng.permutation(len(pool))[:PER_DIR]:
                        e = apply(ref, pool[j], rng)
                        r2 = sc.score((wav, 16000), e.ref)
                        ww = np.zeros(len(e.ref.words), bool)
                        ww[e.word] = True
                        items.append(_item(sc, r2, "length", rec["id"], ww, e.flag,
                                           {"contrast": cn, "direction": dr, "src": pool[j].src,
                                            "dst": pool[j].dst}))
            for kind in SOUND_KINDS:
                try:
                    cor = corrupt(ref, [kind], inv, n_errors=2, seed=stable_seed(rec["id"], kind, "prosody-v1"),
                                  strict=False)
                except NotApplicable:
                    continue
                if not cor.errors:
                    continue
                r2 = sc.score((wav, 16000), cor.ref)
                items.append(_item(sc, r2, "sound", rec["id"], cor.word_wrong,
                                   [i for i, x in enumerate(cor.phone_wrong) if x], {"kind": kind}))
            for dv in dev.get(rec["id"], []):
                try:
                    r2 = sc.score((wav, 16000), dv["text"])
                except ValueError:
                    continue
                items.append(_item(sc, r2, "swap:" + dv["labels"]["variant"], rec["id"], extra={"dsid": dv["id"]}))
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(items))
        os.replace(tmp, out)
        print(f"{lang} {split} {k0 + CHUNK}/{len(recs)} items {len(items)}", flush=True)


def load(lang: str, split: str) -> list[dict]:
    items = []
    for p in sorted(cache_dir(lang, split).glob("chunk_*.pkl")):
        items += pickle.loads(p.read_bytes())
    return items


# ----------------------------------------------------------------------------- fit


def table(items, model=None):
    """Per labelled sound: (item index, phone index, class, label, feature row, P(long) or None)."""
    from pronunciada import length as L

    out = []
    for k, it in enumerate(items):
        lab = L.labels(it["ref"])
        idx, cls, rows = L.features(it["ref"], lab, it["inp"])
        for i, c, r in zip(idx, cls, rows):
            pl = float(L.p_long(model, c, [r])[0]) if model is not None and c in model else None
            out.append((k, i, c, lab[i], r, pl))
    return out


N_TREES, DEPTH, LR = 120, 3, 0.1


def _complete(tree, depth=DEPTH):
    """sklearn regression tree -> complete BFS arrays (feat [2^d - 1], thr, leaf [2^d]); a leaf
    reached early is extended with always-left splits (thr = +inf) carrying its value."""
    t = tree.tree_
    n_int = 2 ** depth - 1
    feat = np.zeros(n_int, dtype=np.int64)
    thr = np.full(n_int, np.inf)
    leaf = np.zeros(2 ** depth)

    def walk(node, pos, d):
        if d == depth:
            leaf[pos - n_int] = t.value[node][0][0]
            return
        if t.children_left[node] == -1:          # early leaf: always go left
            feat[pos], thr[pos] = 0, np.inf
            walk(node, 2 * pos + 1, d + 1)
            walk(node, 2 * pos + 2, d + 1)
            return
        feat[pos], thr[pos] = t.feature[node], t.threshold[node]
        walk(t.children_left[node], 2 * pos + 1, d + 1)
        walk(t.children_right[node], 2 * pos + 2, d + 1)

    walk(0, 0, 0)
    return feat, thr, leaf


def fit_sound(items) -> dict:
    """Boosted depth-3 trees per class on train natives (balanced classes), exported complete."""
    from sklearn.ensemble import GradientBoostingClassifier

    from pronunciada import length as L

    tab = table(items)
    model = {}
    for c, names in (("v", L.V_FEATURES), ("c", L.C_FEATURES)):
        rows = [t for t in tab if t[2] == c]
        y = np.array([t[3] == "long" for t in rows], dtype=int)
        if len(rows) < 50 or y.sum() < 20:
            continue
        X = np.array([[t[4][f] for f in names] for t in rows], dtype=np.float64)
        w = np.where(y == 1, (len(y) - y.sum()) / y.sum(), 1.0)
        m = GradientBoostingClassifier(n_estimators=N_TREES, max_depth=DEPTH, learning_rate=LR, subsample=0.8,
                                       random_state=0).fit(X, y, sample_weight=w)
        F, T, V = zip(*(_complete(e[0]) for e in m.estimators_))
        init = float(m.decision_function(X[:1])[0] - LR * sum(e[0].predict(X[:1])[0] for e in m.estimators_))
        M = {"features": list(names), "init": round(init, 6), "learning_rate": LR,
             "feat": np.array(F).tolist(), "thr": [[float(f"{v:.7g}") for v in r] for r in np.array(T)],
             "leaf": np.round(np.array(V), 6).tolist(), "n_long": int(y.sum()), "n_short": int(len(y) - y.sum())}
        chk = L.p_long({c: dict(M)}, c, [dict(zip(names, x)) for x in X[:200]])
        ref = 1 / (1 + np.exp(-m.decision_function(X[:200])))
        assert np.max(np.abs(chk - ref)) < 1e-3, np.max(np.abs(chk - ref))
        model[c] = M
    return model


def length_probs(it, model) -> list:
    from pronunciada import length as L

    key = id(model)
    if it.get("_lp_key") != key:
        it["_lp"] = L.score(it["ref"], it["inp"], {"model": model})["prob"]
        it["_lp_key"] = key
    return it["_lp"]


def sound_eval(items, model) -> dict:
    """Sound level: threshold at 5 % / 10 % of native labelled sounds; catch of swapped sounds."""
    from bench.metrics import roc_auc

    nat, pos = [], {}
    for it in items:
        pr = length_probs(it, model)
        if it["tag"] == "clean":
            nat += [p for p in pr if p is not None]
        elif it["tag"] == "length":
            v = [pr[i] for i in it["wrong_phones"] if pr[i] is not None]
            if v:
                pos.setdefault((it["contrast"], it["direction"]), []).append(min(v))
    nat = np.array(nat)
    out = {"n_native_sounds": int(len(nat))}
    n_all = sum(len(it["ref"].phones) for it in items if it["tag"] == "clean")
    for q in (5, 10):
        t = float(np.quantile(nat, q / 100))
        out[f"threshold_{q}"] = t
        out[f"native_flag_share_of_all_sounds_{q}"] = float((nat <= t).sum() / n_all)
    allp = []
    for key, v in sorted(pos.items()):
        v = np.array(v)
        allp.append(v)
        out[f"{key[0]}:{key[1]}"] = {"n": int(len(v)), "hit5": float(np.mean(v <= out["threshold_5"])),
                                     "hit10": float(np.mean(v <= out["threshold_10"])),
                                     "auc": float(roc_auc(np.r_[np.ones(len(v)), np.zeros(len(nat))], -np.r_[v, nat]))}
    if allp:
        v = np.concatenate(allp)
        out["all"] = {"n": int(len(v)), "hit5": float(np.mean(v <= out["threshold_5"])),
                      "hit10": float(np.mean(v <= out["threshold_10"])),
                      "auc": float(roc_auc(np.r_[np.ones(len(v)), np.zeros(len(nat))], -np.r_[v, nat]))}
    return out


GRID_Q = (0.002, 0.005, 0.01, 0.02)
GRID_WIDTH = (1.0, 2.0, 4.0)
GRID_FLOOR = (0.2, 0.4)
MAX_AUC_DROP = 0.003


def word_factor(par, vals, s):
    from pronunciada import length as L

    return L.word_factor(par, vals, s) if vals else 1.0


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def word_eval(items, model, cal) -> dict:
    """Word level at s = 0 / 0.5 / 1: native acceptance before / after, length-swapped words caught
    at the 5 % native word threshold, planted sound-error word AUC before / after (natives +
    planted references, s = 0.5)."""
    from bench.metrics import roc_auc

    out = {}
    tags, wrong, snd, fac = [], [], [], []
    for it in items:
        if it["tag"] not in ("clean", "length", "sound"):
            continue
        pr = length_probs(it, model)
        for wi, w in enumerate(it["ref"].words):
            v = [pr[i] for i in range(*w.phones) if pr[i] is not None]
            tags.append(it["tag"])
            wrong.append(bool(it["wrong_words"][wi]))
            snd.append(it["wp"][wi])
            fac.append([word_factor(cal, v, s) for s in (0.0, 0.5, 1.0)])
    tags, wrong, snd, fac = np.array(tags), np.array(wrong), np.array(snd), np.array(fac)
    prod = snd * fac
    nat = tags == "clean"
    for k, s in enumerate((0.0, 0.5, 1.0)):
        out[f"native_accept_{s}"] = {"before": float(np.mean(snd[nat, k] >= 0.5)),
                                     "after": float(np.mean(prod[nat, k] >= 0.5))}
    Lw = (tags == "length") & wrong
    for name, sc in (("before", snd[:, 1]), ("after", prod[:, 1])):
        t5 = np.quantile(sc[nat], 0.05)
        out[f"length_word_hit5_{name}"] = float(np.mean(sc[Lw] < t5))
        y = np.r_[np.ones(Lw.sum()), np.zeros(nat.sum())]
        out[f"length_word_auc_{name}"] = float(roc_auc(y, -np.r_[sc[Lw], sc[nat]]))
    m = nat | (tags == "sound")
    y = ((tags == "sound") & wrong)[m].astype(int)
    out["sound_error_word_auc_before"] = float(roc_auc(y, -snd[m, 1]))
    out["sound_error_word_auc_after"] = float(roc_auc(y, -prod[m, 1]))
    out["n_length_words"] = int(Lw.sum())
    return out


def native_logits(items, model):
    z = []
    for it in items:
        if it["tag"] == "clean":
            z += [p for p in length_probs(it, model) if p is not None]
    return _logit(z)


def make_cal(nz, q, width, floor):
    return {"word": {"knee": [float(np.quantile(nz, q / 2)), float(np.quantile(nz, q)),
                              float(np.quantile(nz, min(2 * q, 0.5)))],
                     "width": width, "floor": floor, "native_quantile": q}}


def word_fit(items, model):
    """Fixed rule on dev: among GRID_Q x GRID_WIDTH x GRID_FLOOR, the configurations whose native
    word acceptance at s = 0.5 drops by <= MAX_NATIVE_DROP and whose planted sound-error word AUC
    and near-swap AUC drop by <= MAX_AUC_DROP; pick the highest length-swap word hit@5 %."""
    nz = native_logits(items, model)
    cands = []
    for q in GRID_Q:
        for wd in GRID_WIDTH:
            for fl in GRID_FLOOR:
                cal = make_cal(nz, q, wd, fl)
                e = word_eval(items, model, cal)
                na = e["native_accept_0.5"]
                sw = swap_eval(items, model, cal)
                near = sw["before"].get("swap_near:auc", np.nan) - sw["after"].get("swap_near:auc", np.nan)
                ok = (na["before"] - na["after"] <= MAX_NATIVE_DROP
                      and e["sound_error_word_auc_before"] - e["sound_error_word_auc_after"] <= MAX_AUC_DROP
                      and near <= MAX_AUC_DROP)
                cands.append({"q": q, "width": wd, "floor": fl, "ok": ok, "near_auc_drop": near,
                              "native_drop": na["before"] - na["after"], **e})
                print(f"q={q} w={wd} floor={fl} ok={ok} native_drop={na['before'] - na['after']:.4f} "
                      f"hit5={e['length_word_hit5_after']:.3f} sound_auc_drop="
                      f"{e['sound_error_word_auc_before'] - e['sound_error_word_auc_after']:.4f} near_drop={near:.4f}",
                      flush=True)
    feas = [c for c in cands if c["ok"]] or cands
    best = max(feas, key=lambda c: (c["length_word_hit5_after"], -c["native_drop"]))
    return make_cal(nz, best["q"], best["width"], best["floor"]), best, cands


COACH_PRECISION = {"careful": 0.8, "balanced": 0.7, "eager": 0.6}   # the targets of the other kinds
COACH_Q = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)


def coach_thresholds(items, model, prior=0.05):
    """Coaching operating points on length_prob, like the other kinds: per sensitivity the
    loosest native-quantile threshold (COACH_Q) whose estimated precision reaches
    COACH_PRECISION, with precision = prior x catch / (prior x catch + (1 - prior) x native
    flag rate) for a PRIOR share of wrong-length sounds (planted swaps stand in for learner
    errors; no learner data with length labels exists). None reaches it -> threshold 0 (off)."""
    nat = []
    pos = []
    for it in items:
        pr = length_probs(it, model)
        if it["tag"] == "clean":
            nat += [p for p in pr if p is not None]
        elif it["tag"] == "length":
            v = [pr[i] for i in it["wrong_phones"] if pr[i] is not None]
            pos += [min(v)] if v else []
    nat, pos = np.array(nat), np.array(pos)
    out = {"thresholds": {}, "precision": {}, "catch": {}, "native_flag": {}, "prior": prior,
           "target_precision": COACH_PRECISION}
    for name, target in COACH_PRECISION.items():
        best = (0.0, None, 0.0, 0.0)
        for q in COACH_Q:
            t = float(np.quantile(nat, q))
            hit, fa = float(np.mean(pos <= t)), float(np.mean(nat <= t))
            prec = prior * hit / max(prior * hit + (1 - prior) * fa, 1e-9)
            if prec >= target and hit > best[2]:
                best = (t, prec, hit, fa)
        out["thresholds"][name] = best[0]
        out["precision"][name] = None if best[1] is None else round(best[1], 3)
        out["catch"][name] = best[2]
        out["native_flag"][name] = best[3]
    return out


def swap_eval(items, model, par) -> dict:
    """Near / far word-swap dev AUC (bench.eval_swap.evaluate) before / after the length factor."""
    from bench.calib.bias_grid import dev_swap
    from bench.eval_swap import evaluate

    ids = {it.get("dsid") for it in items if it["tag"].startswith("swap:")}
    recs = [r for r in dev_swap(items[0]["ref"].lang) if r["id"] in ids]
    out = {}
    for name, use in (("before", False), ("after", True)):
        outs = {}
        for it in items:
            if not it["tag"].startswith("swap:"):
                continue
            pr = length_probs(it, model) if use else None
            ws = []
            for wi, w in enumerate(it["ref"].words):
                p = float(it["wp"][wi][1])
                if use:
                    p *= word_factor(par, [pr[i] for i in range(*w.phones) if pr[i] is not None], 0.5)
                ws.append((it["spans"][wi][0], it["spans"][wi][1], p))
            outs[it["dsid"]] = {"status": "Success", "words": ws}
        m = evaluate(recs, outs, n_boot=0)
        lg = next(iter(m))
        out[name] = {k: v["value"] for k, v in m[lg]["metrics"].items() if k.startswith(("swap_near", "swap_far"))}
    return out


def fit_cmd(lang: str, dry: bool, out_json: str | None):
    from bench.pipeline.common import repo_root

    tr = load(lang, "train")
    dv = load(lang, "validation")
    model = fit_sound([it for it in tr if it["tag"] == "clean"])
    se = sound_eval(dv, model)
    print(json.dumps(se, indent=1, default=float), flush=True)
    cal, best, cands = word_fit(dv, model)
    sw = swap_eval(dv, model, cal)
    co = coach_thresholds(dv, model)
    clean_model = {c: {k: v for k, v in M.items() if not k.startswith("_")} for c, M in model.items()}
    res = {"lang": lang, "kind": "length", "model": clean_model, **cal, "coach": co,
           "fit_on": {"sound_model": f"FLEURS train natives, {sum(it['tag'] == 'clean' for it in tr)} clips",
                      "word_factor_and_coach": f"FLEURS validation, {sum(it['tag'] == 'clean' for it in dv)} clips"},
           "dev": {"sound": se, "word": best, "grid": cands, "swap": sw}}
    print("chosen", {k: best[k] for k in ("q", "width", "floor", "ok", "native_drop", "near_auc_drop",
                                          "length_word_hit5_before", "length_word_hit5_after",
                                          "sound_error_word_auc_before", "sound_error_word_auc_after")})
    print("coach", co)
    if out_json:
        Path(out_json).write_text(json.dumps(res, indent=1, default=float))
    if not dry:
        d = repo_root() / "pronunciada" / "data" / "length"
        d.mkdir(exist_ok=True)
        ship = {k: res[k] for k in ("lang", "kind", "model", "word", "coach", "fit_on")}
        ship["dev_summary"] = {"sound_hit5": se.get("all", {}).get("hit5"),
                               "word_hit5": best["length_word_hit5_after"], "native_drop": best["native_drop"]}
        (d / f"{lang}.json").write_text(json.dumps(ship, separators=(",", ":")))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "fit"])
    ap.add_argument("--lang", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.lang, a.split, a.n, a.threads)
    else:
        fit_cmd(a.lang, a.dry_run, a.out)


if __name__ == "__main__":
    main()
