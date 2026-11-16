"""Round 12: the sound-calibrator blind spot, all ten languages, FLEURS VALIDATION only.

    python -m bench.calib.sound2 run --lang xx        # collect per-phone evidence (dev)
    python -m bench.calib.sound2 diagnose             # miss taxonomy, all languages
    python -m bench.calib.sound2 fit                  # second-opinion model, leave-one-language-out
    python -m bench.calib.sound2 evaluate             # dev before / after per language

run: per validation clip, the public Scorer (shipped, explain-away off) on the correct prompt, the
controlled-error references (bench.eval_languages kinds + prosody kind, seeds "features-v1") and
the dev swap texts (bench/calib/bias_grid.py devswap_<lang>.jsonl). Stored per reference: per
phone the raw evidence and the feature columns the word calibrator reads, the universal sound
logits (with the language bias) and the shown sound logits; per word the raw prosody values and
P(present); labels. Offline, the word calibrator is re-run (features.word_table +
the shipped word model) on modified sound logits, so candidates are compared without re-scoring.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict

import numpy as np

LANGS = ("en", "zh", "es", "it", "ja", "ko", "fr", "de", "pt", "ru")
SOUND_KINDS = ("phone_sub_near", "phone_sub_far", "phone_del", "phone_ins", "word_swap")
PHONE_KINDS = ("phone_sub_near", "phone_sub_far", "phone_ins", "word_swap")
TAB_COLS = ("word_index", "llr_sub", "llr_del", "word_mean_llr_sub", "word_min_llr_sub", "word_mean_llr_del",
            "word_min_llr_del", "word_share_del", "word_frames", "z_llr_sub", "z_llr_del", "peak_post",
            "n_frames", "llr_sub_per_frame", "log_ratio_comp", "is_nucleus_f", "llr_sub_rel_word",
            "word_n_phones", "alt_nonzero", "utt_median_llr_sub")
POS = {"initial": 0, "medial": 1, "final": 2, "only": 3}


def work_dir():
    from bench.calib.common import calib_cache

    d = calib_cache() / "sound2"
    d.mkdir(exist_ok=True)
    return d


_SC = None
_PROTO = None


def _init(lang):
    from bench.calib import api_run

    api_run._init(lang, "fp32", 1)
    global _SC, _PROTO
    _SC = api_run._SC
    _SC.ea = None
    from bench.calib.sound_emb import Protos, proto_path

    _PROTO = Protos(lang) if (proto_path(lang).exists() and _SC.am.embeddings) else None


def _item(res, phone_wrong=None, word_wrong=None, extra=None):
    from pronunciada.scorer import _present_prob
    from pronunciada.variants import phone_position

    sc = _SC
    c = res._pronunciada  # noqa: SLF001
    ref, ev, tab = c["ref"], c["ev"], c["tab"]
    u = sc.universal.logits(X=sc.universal.matrix(tab))
    P = {k: np.asarray(tab[k], np.float32) for k in TAB_COLS if k in tab}
    P["uzl"] = (u["z_len"] + sc.universal_bias).astype(np.float32)
    P["uzs"] = (u["z_str"] + sc.universal_bias).astype(np.float32)
    P["ureg"] = np.asarray(u["reg"], np.float32)
    P["szl"], P["szs"] = np.asarray(c["s_zl"], np.float32), np.asarray(c["s_zs"], np.float32)
    P["pos"] = np.array([POS[phone_position(ref, i)] for i in range(len(ref.phones))], np.int8)
    field = {"zh": "tone_prob", "ja": None}.get(sc.lang, "stress_prob" if sc.lang in
                                                ("en", "de", "es", "it", "pt", "ru") else None)
    raw = [[float(getattr(s, field)) for s in w.syllables if getattr(s, field) is not None] if field else []
           for w in res.words]
    present = np.array([_present_prob(sc.omit_cal, tab["llr_del"][w.phones[0]:w.phones[1]]) if sc.omit_cal else 1.0
                        for w in ref.words], np.float32)
    bad = not np.isfinite(ev.L_can) or ev.L_can <= -1e5
    if _PROTO is not None:                      # candidate d: prototype features (sound_emb.py)
        from bench.calib.sound_emb import last_emb, phone_vectors

        F3 = _PROTO.features(phone_vectors(last_emb(sc), ev, len(ref.phones)), ref)
        P["proto_cos"], P["proto_other"], P["proto_margin"] = F3[:, 0], F3[:, 1], F3[:, 2]
    out = {"P": P, "ipa": [p.ipa for p in ref.phones], "raw": raw, "present": present, "bad": bad,
           "w_zl": np.asarray(c["w_zl"], np.float32), "w_zs": np.asarray(c["w_zs"], np.float32)}
    if phone_wrong is not None:
        out["phone_wrong"] = np.asarray(phone_wrong, bool)
        out["word_wrong"] = np.asarray(word_wrong, bool)
    if extra:
        out.update(extra)
    return out


def _swap_target(lang, ref_new, ref_orig, k):
    """(changed phone indices in ref_new, word indices of unit k) for a one-unit swap."""
    from bench import textswap as TS

    un, uo = TS.units(ref_new, lang), TS.units(ref_orig, lang)
    if k >= len(un) or k >= len(uo):
        return [], []
    a, b = un[k].span
    words = [i for i, w in enumerate(ref_new.words) if w.char_span[0] < b and w.char_span[1] > a]
    idx = [i for wi in words for i in range(*ref_new.words[wi].phones)]
    pn, po = un[k].phones, uo[k].phones
    ch = [idx[j] for j in range(min(len(pn), len(po), len(idx))) if pn[j] != po[j]] if len(pn) == len(po) else []
    return ch, words


def _clip(args):
    rec, wav, swaps = args
    from bench.calib.fit_prosody import KIND
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.common import stable_seed
    from bench.pipeline.evalctl import inventory

    sc = _SC
    audio = (wav, 16000)
    try:
        res = sc.score(audio, rec["text"])
    except ValueError:
        return rec["id"], None
    ref = res._pronunciada["ref"]  # noqa: SLF001
    items = {"clean": _item(res)}
    inv = inventory(sc.lang)
    pk = KIND.get(sc.lang)
    for k in SOUND_KINDS + ((pk,) if pk else ()):
        try:
            cor = corrupt(ref, [k], inv, n_errors=2, seed=stable_seed(rec["id"], k, "features-v1"), strict=False)
        except NotApplicable:
            continue
        if not cor.errors:
            continue
        r2 = sc.score(audio, cor.ref)
        items[f"cor:{k}"] = _item(r2, cor.phone_wrong, cor.word_wrong)
    sw = {}
    for s in swaps:
        try:
            r3 = sc.score(audio, s["text"])
        except ValueError:
            continue
        lab = s["labels"]
        extra = {"spans": [tuple(w.char_span) for w in r3._pronunciada["ref"].words], "variant": lab["variant"]}  # noqa: SLF001
        if lab["variant"] != "orig" and lab.get("word_index") is not None:
            ch, ws = _swap_target(sc.lang, r3._pronunciada["ref"], ref, lab["word_index"])  # noqa: SLF001
            extra.update(changed=ch, unit_words=ws)
        sw[s["id"]] = _item(r3, extra=extra)
    return rec["id"], {"items": items, "swap": sw}


def run(lang, workers):
    from concurrent.futures import ProcessPoolExecutor

    from bench.calib.bias_grid import dev_swap
    from bench.pipeline.common import iter_parquet_audio, records

    recs = records("fleurs-validation", lang=lang)
    by_row = {int(r["audio"].split(":")[2]): r for r in recs}
    path = recs[0]["audio"].split(":")[1]
    swaps = defaultdict(list)
    for s in dev_swap(lang):
        swaps[s["src"]].append(s)
    t0 = time.time()
    data = {}
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(lang,)) as ex:
        futs = [ex.submit(_clip, (by_row[i], wav, swaps.get(by_row[i]["id"], [])))
                for i, wav in iter_parquet_audio(path, set(by_row))]
        for f in futs:
            uid, out = f.result()
            if out is not None:
                data[uid] = out
    (work_dir() / f"run_{lang}.pkl").write_bytes(pickle.dumps(data))
    print(f"{lang}: {len(data)} clips ({time.time() - t0:.0f}s)", flush=True)


def load(lang):
    return pickle.loads((work_dir() / f"run_{lang}.pkl").read_bytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run"])
    ap.add_argument("--lang", default="all")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    for lang in (LANGS if a.lang == "all" else a.lang.split(",")):
        run(lang, a.workers)



# ============================================================================ offline

_WORD = None


def word_model():
    global _WORD
    if _WORD is None:
        from pronunciada.calibrate import load_level

        _WORD = load_level("word")
    return _WORD


def word_logits(it, uzl, uzs):
    """Shipped word calibrator on (possibly modified) universal sound logits of one reference."""
    from pronunciada import calibrate as C
    from pronunciada import features as F

    P = it["P"]
    if it["bad"]:
        lo = float(C.logit(C.P_CLIP))
        n = len(it["w_zl"])
        return np.full(n, lo), np.full(n, lo)
    t = {k: np.asarray(v, np.float64) for k, v in P.items()}
    t["word_index"] = np.asarray(P["word_index"], np.int64)
    W = F.word_table(t, np.asarray(uzl, np.float64), np.asarray(uzs, np.float64), np.asarray(P["ureg"], np.float64))
    m = word_model()
    X = np.column_stack([np.asarray(W[c], dtype=np.float64) for c in m.num])
    w = m.logits(X=X)
    return w["z_len"], w["z_str"]


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def _blend(zl, zs, s=0.5):
    from pronunciada.calibrate import blend_logit

    zl, zs = np.asarray(zl, np.float64), np.asarray(zs, np.float64)
    return blend_logit(zl, np.minimum(zs, zl), s)


def prosody_cal(lang):
    from bench.calib.common import ship_dir

    p = ship_dir() / "prosody" / f"{lang}.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    return None if c.get("experimental") else c


def outputs(it, transform, cal):
    """(shown sound P, word P, sounds-only word P) at s = 0.5 for one reference under transform:
    transform(it) -> (uzl, uzs, szl, szs) or None for the shipped outputs."""
    from pronunciada.scorer import _pros_prob

    P = it["P"]
    if transform is None:
        uzl, uzs, szl, szs = P["uzl"], P["uzs"], P["szl"], P["szs"]
        wzl, wzs = it["w_zl"], it["w_zs"]
    else:
        uzl, uzs, szl, szs = transform(it)
        wzl, wzs = word_logits(it, uzl, uzs)
    ps = _sig(_blend(szl, szs))
    snd = _sig(_blend(wzl, wzs))
    pp = np.array([_pros_prob(cal, r, 0.5) if (cal and r) else 1.0 for r in it["raw"]])
    return ps, snd * pp * it["present"], snd


def metrics(lang, data, transform=None, n_boot=0, cache=None):
    from bench.calib.bias_grid import dev_swap
    from bench.eval_swap import evaluate
    from bench.metrics import roc_auc

    cal = prosody_cal(lang)
    nat_s, nat_w, ys, ss, yw, sw, pe = [], [], [], [], [], [], []
    for uid, d in data.items():
        ps, wp, _ = outputs(d["items"]["clean"], transform, cal)
        nat_s.append(ps)
        nat_w.append(wp)
        for key, it in d["items"].items():
            if not key.startswith("cor:"):
                continue
            k = key[4:]
            ps, wp, _ = outputs(it, transform, cal)
            if k in SOUND_KINDS:
                if k in PHONE_KINDS:
                    ys.append(it["phone_wrong"])
                    ss.append(ps)
                yw.append(it["word_wrong"])
                sw.append(wp)
            else:
                pe.append(wp[it["word_wrong"]])
    nat_s, nat_w = np.concatenate(nat_s), np.concatenate(nat_w)
    ys, ss = np.concatenate(ys).astype(int), np.concatenate(ss)
    yw, sw = np.concatenate(yw).astype(int), np.concatenate(sw)
    out = {"native_words": float(np.mean(nat_w >= 0.5)), "native_sounds": float(np.mean(nat_s >= 0.5)),
           "sound_auc": float(roc_auc(ys, 1 - ss)), "word_auc": float(roc_auc(yw, 1 - sw))}
    if pe:
        pe = np.concatenate(pe)
        y = np.r_[np.ones(len(pe)), np.zeros(len(nat_w))].astype(int)
        out["pros_auc"] = float(roc_auc(y, 1 - np.r_[pe, nat_w]))
    outs, pos = {}, []
    recs = [r for r in dev_swap(lang) if r["src"] in data]
    lab = {r["id"]: r["labels"] for r in recs}
    for d in data.values():
        for sid, it in d["swap"].items():
            _, wp, _ = outputs(it, transform, cal)
            outs[sid] = {"status": "Success", "words": [(a, c, float(p)) for (a, c), p in zip(it["spans"], wp)]}
            L = lab.get(sid)
            if L and L["variant"] == "swap_near":
                a, c = L["char_span_in_new_text"]
                s_ = [w[2] for w in outs[sid]["words"] if w[0] < c and a < w[1]]
                if s_:
                    pos.append(min(s_))
    m = evaluate(recs, outs, n_boot=n_boot)[lang]["metrics"]
    for v in ("swap_near", "swap_far"):
        for k in ("auc", "hit@5%FA", "right_word"):
            x = m.get(f"{v}:{k}")
            if x:
                out[f"{v}:{k}"] = [x["value"], x.get("lo"), x.get("hi")]
    pos = np.array(pos)
    out["near_gt09"] = float(np.mean(pos > 0.9))
    out["near_gt08"] = float(np.mean(pos > 0.8))
    return out


# ----------------------------------------------------------------------------- diagnosis

EV_STRONG, EV_WEAK = -2.0, -0.5


def _cat(llr, ps, wp):
    if llr > EV_WEAK:
        return "no evidence"
    if llr > EV_STRONG:
        return "weak evidence"
    if ps >= 0.5:
        return "evidence, discounted by sound calibrator"
    return "evidence, sound flagged, lost in word model"


def diagnose(lang, data):
    """Near-swap misses (replaced word P > 0.8) and planted single-sound substitutions whose word
    stays > 0.8: where the changed sound is, and at which stage the evidence was lost."""
    from bench.calib.bias_grid import dev_swap

    cal = prosody_cal(lang)
    lab = {r["id"]: r["labels"] for r in dev_swap(lang)}
    rows = []
    for d in data.values():
        for sid, it in d["swap"].items():
            L = lab.get(sid)
            if not L or L["variant"] != "swap_near" or not it.get("changed"):
                continue
            ps, wp, snd = outputs(it, None, cal)
            wpu = min(wp[w] for w in it["unit_words"]) if it["unit_words"] else 1.0
            for i in it["changed"][:1]:
                P = it["P"]
                rows.append({"src": "swap", "miss": wpu > 0.8, "pos": int(P["pos"][i]),
                             "vowel": bool(P["is_nucleus_f"][i] > 0.5), "wlen": int(P["word_n_phones"][i]),
                             "llr_sub": float(P["llr_sub"][i]), "llr_del": float(P["llr_del"][i]),
                             "ps": float(ps[i]), "wp": float(wpu), "ipa": it["ipa"][i],
                             "cat": _cat(float(P["llr_sub"][i]), float(ps[i]), wpu)})
        for key in ("cor:phone_sub_near", "cor:phone_sub_far"):
            it = d["items"].get(key)
            if it is None:
                continue
            ps, wp, snd = outputs(it, None, cal)
            P = it["P"]
            wi = np.asarray(P["word_index"], int)
            for i in np.flatnonzero(it["phone_wrong"]):
                rows.append({"src": key[4:], "miss": wp[wi[i]] > 0.8, "pos": int(P["pos"][i]),
                             "vowel": bool(P["is_nucleus_f"][i] > 0.5), "wlen": int(P["word_n_phones"][i]),
                             "llr_sub": float(P["llr_sub"][i]), "llr_del": float(P["llr_del"][i]),
                             "ps": float(ps[i]), "wp": float(wp[wi[i]]), "ipa": it["ipa"][i],
                             "cat": _cat(float(P["llr_sub"][i]), float(ps[i]), float(wp[wi[i]]))})
    nat = np.concatenate([d["items"]["clean"]["P"]["llr_sub"] for d in data.values()])
    return rows, {"native_llr_le_-2": float(np.mean(nat <= EV_STRONG)), "native_llr_le_-0.5": float(np.mean(nat <= EV_WEAK)),
                  "n_native": int(len(nat))}



# ----------------------------------------------------------------------------- second opinion

FEATS = ("llr_sub", "llr_del_c", "z_llr_sub", "llr_sub_per_frame", "peak_post", "log_nf", "final", "only",
         "nucleus", "short", "llr_x_final", "llr_x_only", "llr_x_nucleus", "llr_x_short", "llr_sub_rel_word",
         "utt_median_llr_sub")
MONO = ("llr_sub", "llr_x_final", "llr_x_only", "llr_x_nucleus", "llr_x_short")   # see check_monotone


def feature_matrix(P):
    llr = np.asarray(P["llr_sub"], np.float64)
    llr = np.clip(np.nan_to_num(llr, nan=0.0), -30, 0)
    pos = np.asarray(P["pos"])
    fin, only = (pos == 2).astype(float), (pos == 3).astype(float)
    nuc = (np.asarray(P["is_nucleus_f"]) > 0.5).astype(float)
    short = (np.asarray(P["word_n_phones"]) <= 3).astype(float)
    cols = {"llr_sub": llr, "llr_del_c": np.clip(np.nan_to_num(np.asarray(P["llr_del"], np.float64)), -20, 20),
            "z_llr_sub": np.clip(np.nan_to_num(np.asarray(P["z_llr_sub"], np.float64)), -20, 5),
            "llr_sub_per_frame": np.clip(np.nan_to_num(np.asarray(P["llr_sub_per_frame"], np.float64)), -10, 0),
            "peak_post": np.nan_to_num(np.asarray(P["peak_post"], np.float64)),
            "log_nf": np.log1p(np.nan_to_num(np.asarray(P["n_frames"], np.float64))),
            "final": fin, "only": only, "nucleus": nuc, "short": short,
            "llr_x_final": llr * fin, "llr_x_only": llr * only, "llr_x_nucleus": llr * nuc, "llr_x_short": llr * short,
            "llr_sub_rel_word": np.clip(np.nan_to_num(np.asarray(P["llr_sub_rel_word"], np.float64)), -30, 30),
            "utt_median_llr_sub": np.nan_to_num(np.asarray(P["utt_median_llr_sub"], np.float64))}
    return np.column_stack([cols[k] for k in FEATS])


def training_rows(lang, data):
    """Natives' own sounds (clean references) -> 1, planted near / far substitutions -> 0."""
    X, y = [], []
    for d in data.values():
        it = d["items"]["clean"]
        if not it["bad"]:
            X.append(feature_matrix(it["P"]))
            y.append(np.ones(len(it["ipa"])))
        for key in ("cor:phone_sub_near", "cor:phone_sub_far"):
            it = d["items"].get(key)
            if it is None or it["bad"]:
                continue
            m = it["phone_wrong"]
            X.append(feature_matrix(it["P"])[m])
            y.append(np.zeros(int(m.sum())))
    return np.concatenate(X), np.concatenate(y)


def fit_lr(parts):
    """Pooled logistic regression; every language and class gets equal total weight."""
    from sklearn.linear_model import LogisticRegression

    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    w = np.concatenate([np.where(p[1] == 1, 0.5 / max((p[1] == 1).sum(), 1), 0.5 / max((p[1] == 0).sum(), 1))
                        for p in parts]) * len(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(C=1.0, max_iter=3000).fit((X - mu) / sd, y, sample_weight=w)
    coef = m.coef_[0] / sd
    b = float(m.intercept_[0] - (m.coef_[0] * mu / sd).sum())
    return {"feats": list(FEATS), "coef": coef.tolist(), "b": b}


def z2(model, P):
    return feature_matrix(P) @ np.asarray(model["coef"]) + model["b"]


def check_monotone(model):
    """d z2 / d llr_sub >= 0 for every flag combination (less evidence against -> more acceptable)."""
    c = dict(zip(model["feats"], model["coef"]))
    base = c["llr_sub"]
    worst = min(base + f * c["llr_x_final"] + o * c["llr_x_only"] + n * c["llr_x_nucleus"] + s * c["llr_x_short"]
                for f in (0, 1) for o in (0, 1) for n in (0, 1) for s in (0, 1) if not (f and o))
    return worst >= 0, worst


def set_offset(model, native_z, q):
    """Shift so that a share q of the training natives' sounds get z2 < 0."""
    out = dict(model)
    out["offset"] = float(-np.quantile(native_z, q))
    out["q"] = q
    return out


def make_transform(model, mode="min"):
    def tr(it):
        P = it["P"]
        z = z2(model, P) + model["offset"]
        if mode == "min":
            return (np.minimum(P["uzl"], z), np.minimum(P["uzs"], z),
                    np.minimum(P["szl"], z), np.minimum(P["szs"], z))
        raise ValueError(mode)
    return tr


if __name__ == "__main__":
    main()
