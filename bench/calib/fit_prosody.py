"""Round 3: the prosody channel — one idempotent command per language.

    python -m bench.calib.fit_prosody --lang zh            # or ja, en, de, es, it, pt, ru, all
                                      [--n 200] [--workers 5]

Re-run it whenever pronunciada.prosody.ProsodyScorer's model for the language changes (e.g. when
tone_zh_v2.onnx lands); it reads whatever ProsodyScorer loads and overwrites
pronunciada/data/calib/prosody/<lang>.json (and results/tables/prosody_<lang>.json).

FLEURS VALIDATION only. For n random clips: the public Scorer (prosody on, prosody channel and
explain-away off) on the correct reference, on a controlled-PROSODIC-error reference (bench.corrupt
kind tone (zh) / pitch (ja) / stress (en de es it pt ru), 2 errors, fixed seed) and, for the
before/after report only, on the five controlled sound-error kinds.
Word-level prosody calibrator P_pros(word): logistic regression on [lowest prosody logit over the
word's scored syllables, second lowest (= lowest for one syllable), number of scored syllables],
fitted on words of the correct and the prosodic-error references (native and untouched words -> ok,
the word carrying the tone / stress / pitch error -> not ok), sample weights so that errors are
10 % of the weight. Monotone: the coefficients of both logits must be >= 0 (else the second
minimum is dropped). Operating points: lenient = logit shifted so 1 - Q_LEN of native dev words
have P_pros >= 0.5, strict = shifted so 1 - Q_STR do (pair chosen per language by the rule next
to Q_GRID below); strictness s interpolates in log-odds.
Evaluation: 5-fold utterance-grouped CV of the whole fit (regression + shifts).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json

import numpy as np

KIND = {"zh": "tone", "ja": "pitch", "en": "stress", "de": "stress", "es": "stress",
        "it": "stress", "pt": "stress", "ru": "stress"}
FIELD = {"tone": "tone_prob", "stress": "stress_prob", "pitch": "pitch_prob"}
SOUND_KINDS = ("phone_sub_near", "phone_sub_far", "phone_ins", "phone_del", "word_swap")
ERR_PRIOR = 0.10
# Share of native dev words the channel alone may put below 0.5 at the lenient / strict end.
# Round 4: chosen per language by a fixed rule on dev CV — the strictest pair of Q_GRID whose
# native word acceptance at the default strictness (0.5) drops by at most MAX_NATIVE_DROP when
# the channel is multiplied in (round 3 used (0.01, 0.05) everywhere: 1.4-2.8 points).
Q_GRID = ((0.01, 0.05), (0.005, 0.02), (0.0025, 0.01), (0.001, 0.005))
MAX_NATIVE_DROP = 0.010
Q_LEN, Q_STR = Q_GRID[0]
EXPERIMENTAL = {"ja": "pitch-accent labels agree with native speech on ~71 % of morae"}


def _words(res, field, wrong):
    from pronunciada.calibrate import blend_logit, sigmoid

    c = res._pronunciada  # noqa: SLF001
    zl, zs = c["w_zl"], np.minimum(c["w_zs"], c["w_zl"])
    snd = {s: sigmoid(blend_logit(zl, zs, s)) for s in (0.0, 0.5, 1.0)}
    out = []
    ref, pros = c.get("ref"), c.get("prosody")
    for wi, w in enumerate(res.words):
        vals = [getattr(x, field) for x in w.syllables if getattr(x, field) is not None]
        out.append({"vals": [float(v) for v in vals], "snd": [float(snd[s][wi]) for s in (0.0, 0.5, 1.0)],
                    "wrong": bool(wrong[wi])})
        if field == "tone_prob" and pros is not None:
            # per scored syllable, for offline re-scoring under other tone-leniency rules
            s0, s1 = ref.words[wi].syllables
            out[-1]["syl"] = [{"i": i, "probs": pros[i]["tone_probs"], "tone": ref.syllables[i].tone,
                               "alts": list(ref.syllables[i].tone_alts)}
                              for i in range(s0, s1) if pros[i].get("tone_prob") is not None]
            out[-1]["ref_obj"] = ref
    return out


def collect(lang, n):
    import os

    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.common import iter_parquet_audio, records, stable_seed
    from bench.pipeline.evalctl import inventory
    from pronunciada.scorer import Scorer

    sc = Scorer(lang, threads=1, explain_away=False)
    sc.pros_cal = None
    if sc.prosody is None:
        raise RuntimeError(f"no prosody model for {lang}: {sc.prosody_error}")
    kind, field = KIND[lang], FIELD[KIND[lang]]
    recs = records("fleurs-validation", lang=lang)
    rng = np.random.default_rng(0)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), min(n, len(recs)), replace=False))]
    by = {int(r["audio"].split(":")[2]): r for r in recs}
    path = recs[0]["audio"].split(":")[1]
    inv = inventory(lang)
    rows = []
    for i, wav in iter_parquet_audio(path, set(by)):
        rec = by[i]
        try:
            res = sc.score((wav, 16000), rec["text"])
        except ValueError:
            continue
        ref = res._pronunciada["ref"]  # noqa: SLF001
        for w in _words(res, field, np.zeros(len(ref.words), bool)):
            rows.append({"utt": rec["id"], "ref": "clean", **w})
        for k in (kind,) + SOUND_KINDS:
            try:
                cor = corrupt(ref, [k], inv, n_errors=2, seed=stable_seed(rec["id"], k, "prosody-v1"), strict=False)
            except NotApplicable:
                continue
            if not cor.errors:
                continue
            r2 = sc.score((wav, 16000), cor.ref)
            tag = "prosody" if k == kind else "sound"
            for w in _words(r2, field, np.asarray(cor.word_wrong)):
                rows.append({"utt": rec["id"], "ref": tag, **w})
    meta = sc.prosody.meta if sc.prosody is not None else {}
    return rows, meta


def cached_collect(lang, n):
    """collect(), cached per (language, n, prosody model files)."""
    import pickle

    from bench.calib.common import calib_cache
    from bench.pipeline.common import repo_root

    import os

    files = sorted((repo_root() / "pronunciada" / "data" / "prosody").glob("*.onnx"))
    h = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()[:10]
    # the model choice / tone leniency can be switched by environment variables: part of the key
    env = "".join(f"_{os.environ[v]}" for v in ("PRONUNCIADA_TONE_MODEL", "PRONUNCIADA_TONE_LENIENCY")
                  if os.environ.get(v))
    p = calib_cache() / f"prosody_rows_{lang}_{n}_{h}{env}_r5.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    out = collect(lang, n)
    p.write_bytes(pickle.dumps(out))
    return out


LENIENCY = None      # --leniency: re-score the collected tone rows offline under these rules


def apply_leniency(rows, meta, rules):
    """Recompute each word's tone values with pronunciada.prosody.tone_leniency `rules` added to
    the acceptable set (same formula as ProsodyScorer._tone), from the stored tone_probs."""
    from pronunciada.prosody import _platt
    from pronunciada.prosody.tone_leniency import extra_tones

    out = []
    for r in rows:
        if "syl" not in r:
            out.append(r)
            continue
        vals = []
        for q in r["syl"]:
            acc = {q["tone"], *q["alts"]} | extra_tones(r["ref_obj"], q["i"], rules)
            vals.append(float(_platt(sum(q["probs"][t - 1] for t in acc if 1 <= t <= 5), meta["platt"])))
        out.append(dict(r, vals=vals))
    return out


def get_rows(lang, n):
    rows, meta = cached_collect(lang, n)
    if LENIENCY is not None:
        rows = apply_leniency(rows, meta, LENIENCY)
    return rows, meta


def sweep(lang, n):
    """Dev-CV trade-off of the native quantiles (report only; the shipped pair is Q_LEN, Q_STR)."""
    global Q_LEN, Q_STR
    rows, _ = get_rows(lang, n)
    keep, out = (Q_LEN, Q_STR), []
    for q in Q_GRID:
        Q_LEN, Q_STR = q
        e = evaluate(rows)
        na = e["native_accept"]["0.5"]
        out.append((q, na["before"] - na["after"], e["prosodic_error"]["product"]["auc"],
                    e["prosodic_error"]["product"]["hit_at_5pct"], e["prosodic_error"]["p_pros"]["auc"],
                    e["sound_error"]["before"]["auc"] - e["sound_error"]["after"]["auc"]))
    Q_LEN, Q_STR = keep
    return lang, out


def _sweep(args):
    return sweep(*args)


def _feats(rows):
    from pronunciada.calibrate import logit

    F = np.full((len(rows), 3), np.nan)
    for k, r in enumerate(rows):
        if r["vals"]:
            q = np.sort(logit(np.asarray(r["vals"])))
            F[k] = [q[0], q[1] if len(q) > 1 else q[0], len(q)]
    return F


def fit(rows):
    """LR + operating points on the given rows (clean + prosody refs). Returns params dict."""
    from sklearn.linear_model import LogisticRegression

    sel = [r for r in rows if r["ref"] in ("clean", "prosody") and r["vals"]]
    X = _feats(sel)
    y = np.array([0 if r["wrong"] else 1 for r in sel])
    n_err, n_ok = (y == 0).sum(), (y == 1).sum()
    w = np.where(y == 0, ERR_PRIOR / (1 - ERR_PRIOR) * n_ok / max(n_err, 1), 1.0)
    cols = [0, 1, 2]
    m = LogisticRegression(C=1.0, max_iter=2000).fit(X[:, cols], y, sample_weight=w)
    if m.coef_[0][0] < 0 or m.coef_[0][1] < 0:
        cols = [0, 2]
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X[:, cols], y, sample_weight=w)
    coef = np.zeros(4)
    coef[0] = m.intercept_[0]
    for j, c in enumerate(cols):
        coef[1 + c] = m.coef_[0][j]
    if coef[1] < 0:
        raise RuntimeError("non-monotone prosody calibrator (negative coefficient on the lowest logit)")
    nat = [r for r in rows if r["ref"] == "clean" and r["vals"]]
    zn = _z(coef, _feats(nat))
    return {"coef": coef.tolist(), "c_len": float(-np.quantile(zn, Q_LEN)), "c_str": float(-np.quantile(zn, Q_STR)),
            "n_fit_words": int(len(sel)), "n_fit_errors": int(n_err)}


def _z(coef, F):
    return coef[0] + coef[1] * F[:, 0] + coef[2] * F[:, 1] + coef[3] * F[:, 2]


def _pp(par, F, s):
    from pronunciada.calibrate import blend_logit, sigmoid

    z = _z(np.asarray(par["coef"]), F)
    zl, zs = z + par["c_len"], z + par["c_str"]
    return np.where(np.isfinite(F[:, 0]), sigmoid(blend_logit(zl, np.minimum(zs, zl), s)), 1.0)


def evaluate(rows):
    """5-fold utterance-grouped CV of the complete fit; metrics before (sounds only) / after."""
    from sklearn.model_selection import GroupKFold

    from bench.metrics import roc_auc

    utt = np.array([r["utt"] for r in rows])
    F = _feats(rows)
    ref = np.array([r["ref"] for r in rows])
    wrong = np.array([r["wrong"] for r in rows])
    snd = np.array([r["snd"] for r in rows])                 # [n, 3] at s = 0, 0.5, 1
    pp = np.ones((len(rows), 3))
    for tr, te in GroupKFold(5).split(rows, None, utt):
        par = fit([rows[i] for i in tr])
        for k, s in enumerate((0.0, 0.5, 1.0)):
            pp[te, k] = _pp(par, F[te], s)
    prod = snd * pp
    clean = ref == "clean"
    out = {"n_words": int(len(rows)), "share_native_words_scored": float(np.mean(np.isfinite(F[clean, 0])))}
    out["native_accept"] = {str(s): {"before": float(np.mean(snd[clean, k] >= 0.5)), "after": float(np.mean(prod[clean, k] >= 0.5)),
                                     "p_pros_only": float(np.mean(pp[clean, k] >= 0.5))}
                            for k, s in enumerate((0.0, 0.5, 1.0))}

    def det(score, tag):
        m = (ref == tag) & wrong | clean
        y = ((ref == tag) & wrong)[m].astype(int)
        nat = score[clean]
        thr = np.quantile(nat, 0.05)
        return {"auc": roc_auc(y, 1 - score[m]), "hit_at_5pct": float(np.mean(score[(ref == tag) & wrong] < thr)),
                "n_err": int(y.sum())}

    syl = [q for r in rows if r["ref"] == "clean" for q in r.get("syl", ())]
    if syl:                                # zh: heard tone (argmax) is the tone or an alt
        lr = LENIENCY or ()
        from pronunciada.prosody.tone_leniency import extra_tones

        agree = collections.defaultdict(list)
        for r in rows:
            if r["ref"] != "clean":
                continue
            for q in r.get("syl", ()):
                acc = {q["tone"], *q["alts"]} | extra_tones(r["ref_obj"], q["i"], lr) if "ref_obj" in r else {q["tone"], *q["alts"]}
                ok = int(np.argmax(q["probs"])) + 1 in acc
                agree["all"].append(ok)
                agree[f"tone{q['tone']}"].append(ok)
        out["native_syllable_agreement"] = {k: float(np.mean(v)) for k, v in sorted(agree.items())}
    out["prosodic_error"] = {"sounds": det(snd[:, 1], "prosody"), "p_pros": det(pp[:, 1], "prosody"),
                             "product": det(prod[:, 1], "prosody")}
    out["sound_error"] = {"before": det(snd[:, 1], "sound"), "after": det(prod[:, 1], "sound")}
    return out


DRY_RUN = False


def run(lang, n):
    from bench.calib.common import save_json, ship_dir
    from bench.pipeline.common import repo_root

    global Q_LEN, Q_STR
    rows, meta = get_rows(lang, n)
    for q in Q_GRID:
        Q_LEN, Q_STR = q
        ev = evaluate(rows)
        na = ev["native_accept"]["0.5"]
        if na["before"] - na["after"] <= MAX_NATIVE_DROP:
            break
    par = fit(rows)
    files = sorted((repo_root() / "pronunciada" / "data" / "prosody").glob("*.onnx"))
    out = {"lang": lang, "kind": KIND[lang], **par,
           "features": "[1, min prosody logit, second-min prosody logit, n scored syllables]",
           "err_prior": ERR_PRIOR, "native_quantiles": [Q_LEN, Q_STR], "fit_on": f"FLEURS validation, {n} clips",
           "prosody_model_meta": meta,
           "prosody_model_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in files},
           "experimental": EXPERIMENTAL.get(lang), "dev_cv": ev}
    if DRY_RUN:
        return lang, out
    d = ship_dir() / "prosody"
    d.mkdir(exist_ok=True)
    (d / f"{lang}.json").write_text(json.dumps(out, indent=1, default=float))
    save_json(f"prosody_{lang}.json", out)
    return lang, out


def _run(args):
    return run(*args)


def main():
    from concurrent.futures import ProcessPoolExecutor

    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, help="one of zh ja en de es it pt ru, or all")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--sweep", action="store_true", help="print the native-quantile trade-off and exit")
    ap.add_argument("--dry-run", action="store_true", help="evaluate only; do not overwrite the shipped json")
    ap.add_argument("--leniency", default=None,
                    help="zh, with --dry-run: re-score offline with these tone-leniency rules (comma list or none)")
    a = ap.parse_args()
    global DRY_RUN, LENIENCY
    DRY_RUN = a.dry_run
    if a.leniency is not None:
        if not a.dry_run:
            raise SystemExit("--leniency only with --dry-run (ship rules through the model json / runtime)")
        LENIENCY = tuple(x for x in a.leniency.split(",") if x and x != "none")
    langs = list(KIND) if a.lang == "all" else a.lang.split(",")
    if a.sweep:
        with ProcessPoolExecutor(min(a.workers, len(langs))) as ex:
            for lang, rows in ex.map(_sweep, [(l, a.n) for l in langs]):
                for q, dn, auc, hit, pauc, ds in rows:
                    print(f"{lang} q={q}: native accept drop {dn:.4f}  prosodic AUC {auc:.3f} hit@5 {hit:.3f} "
                          f"(channel alone {pauc:.3f})  sound AUC drop {ds:.4f}", flush=True)
        return
    if len(langs) == 1:
        res = [run(langs[0], a.n)]
    else:
        with ProcessPoolExecutor(min(a.workers, len(langs))) as ex:
            res = list(ex.map(_run, [(l, a.n) for l in langs]))
    for lang, o in res:
        e = o["dev_cv"]
        print(lang, "coef", [round(x, 3) for x in o["coef"]], "c_len %.2f c_str %.2f" % (o["c_len"], o["c_str"]),
              "native s=.5 before/after %.3f/%.3f" % (e["native_accept"]["0.5"]["before"], e["native_accept"]["0.5"]["after"]),
              "pros AUC sounds/p_pros/product %.3f/%.3f/%.3f" % (e["prosodic_error"]["sounds"]["auc"], e["prosodic_error"]["p_pros"]["auc"], e["prosodic_error"]["product"]["auc"]),
              "hit %.3f" % e["prosodic_error"]["product"]["hit_at_5pct"],
              "native syllable agreement %s" % {k: round(v, 3) for k, v in e.get("native_syllable_agreement", {}).items()},
              "sound AUC %.4f -> %.4f" % (e["sound_error"]["before"]["auc"], e["sound_error"]["after"]["auc"]), flush=True)


if __name__ == "__main__":
    main()
