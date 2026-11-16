"""Round 8 part 2: per-language sound bias chosen by a detection-aware rule (FLEURS VALIDATION).

    python -m bench.calib.bias_grid build-dev --lang xx          # frozen dev swap set
    python -m bench.calib.bias_grid run --lang xx --tag T [--adapter P] [--measured] [--vowel-min .15]
    python -m bench.calib.bias_grid select --lang xx --tag T --base TAG0

run: per validation clip, the public Scorer (explain-away off, as shipped) on the correct prompt,
the controlled-error references of bench.eval_languages (5 sound kinds + the prosody kind, seeds
"features-v1"), the fit_prosody references (seeds "prosody-v1", on fit_prosody's 200-clip subset)
and the dev swap texts. The alignment / features are computed once; the calibrators are re-run
for every bias of the grid (Scorer._calibrate with universal_bias = sound_bias = b). Stored: sound
and word log-odds per bias, raw syllable prosody probabilities, P(present), labels.

select (offline): for every bias, the prosody channel is re-fitted exactly as
bench.calib.fit_prosody.run does (same rows, same leniency rule, same fit) and the word
probability is rebuilt as in Scorer._assemble (sounds x P_pros x P_present). Metrics: dev
near-swap AUC (bench.eval_swap.evaluate), native word / sound acceptance at s = 0.5, planted
sound-error AUC (sub_near, sub_far, ins, word_swap) and word-error AUC (all 5 kinds), prosodic
(stress / tone) shift AUC. RULE (fixed before looking): grid 0 .. current bias in steps of 0.1
(plus the current value); feasible = versus the round-7 baseline run (shipped bias, shipped
prosody calibrator) native words and native sounds drop <= 0.5 points, sound and word AUC drop
<= 0.005, prosodic AUC drop <= 0.010; choose the highest near AUC, ties -> the larger |bias|.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict

import numpy as np

MAX_DUR = 20.0
SOUND_KINDS = ("phone_sub_near", "phone_sub_far", "phone_del", "phone_ins", "word_swap")
PHONE_KINDS = ("phone_sub_near", "phone_sub_far", "phone_ins", "word_swap")
RULE = {"native_words": 0.005, "native_sounds": 0.005, "sound_auc": 0.005, "word_auc": 0.005,
        "pros_auc": 0.010}


def work_dir():
    from bench.calib.common import calib_cache

    d = calib_cache() / "bias_grid"
    d.mkdir(exist_ok=True)
    return d


def grid_for(b0: float) -> list[float]:
    n = int(np.floor(abs(b0) / 0.1 + 1e-9))
    g = [round(np.sign(b0) * 0.1 * k, 6) for k in range(n + 1)]
    if abs(abs(b0) - 0.1 * n) > 1e-6:
        g.append(round(b0, 6))
    return g


# ----------------------------------------------------------------------------- dev swap set

def build_dev(lang: str, n: int = 400):
    from bench import textswap as TS
    from bench.data import load
    from bench.data.fleurs_swap import lexicon_words
    from bench.pipeline.common import records
    from pronunciada.text import get_g2p

    g2p = get_g2p(lang)
    val = sorted(records("fleurs-validation", lang=lang), key=lambda r: r["id"])
    texts = [r["text"] for r in load("fleurs", "train", lang)] + [r["text"] for r in val]
    vocab = TS.Vocab(lang, texts, g2p, lexicon_words(lang) if lang not in ("zh", "ja", "ko") else None)
    srcs = [r for r in val if r["dur"] <= MAX_DUR and TS.source_ok(lang, r["text"], g2p(r["text"]))][:n]
    out, cov = [], defaultdict(int)
    for r in srcs:
        vs = TS.make_variants(r["id"], lang, r["text"], g2p, vocab)
        for v in ("orig", "swap_far", "swap_near", "tone"):
            if v in vs:
                text, lab = vs[v]
                cov[v] += 1
                out.append({"id": f"{r['id']}__{v}", "src": r["id"], "audio": r["audio"], "lang": lang,
                            "text": text, "split": "dev", "dur": r["dur"], "labels": lab})
    p = work_dir() / f"devswap_{lang}.jsonl"
    p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    print(lang, "sources", len(srcs), dict(cov), flush=True)


def dev_swap(lang):
    p = work_dir() / f"devswap_{lang}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ----------------------------------------------------------------------------- run

_SC = None
_CFG = {}


def _init(lang, cfg):
    from bench.calib import api_run

    api_run._init(lang, "fp32", 1)
    global _SC, _CFG
    _SC, _CFG = api_run._SC, cfg
    _SC.ea = None
    if cfg.get("adapter"):
        from pronunciada.adapter import Adapter

        _SC.adapter = Adapter.from_file(cfg["adapter"], lang)
    if lang == "ru":
        _SC.g2p.measured_targets = bool(cfg.get("measured"))
        if cfg.get("vowel_min") is not None:
            _SC.g2p.vowel_min = float(cfg["vowel_min"])


def _field(sc):
    from bench.calib.fit_prosody import FIELD, KIND

    k = KIND.get(sc.lang)
    return k, (FIELD[k] if k else None)


def _item(res, biases, phone_wrong=None, word_wrong=None, spans=None):
    from pronunciada import calibrate as C
    from pronunciada.scorer import _present_prob

    sc = _SC
    c = res._pronunciada  # noqa: SLF001
    ref, ev, tab = c["ref"], c["ev"], c["tab"]
    kind, field = _field(sc)
    raw = []
    for w in res.words:
        raw.append([float(getattr(s, field)) for s in w.syllables if field and getattr(s, field) is not None]
                   if field else [])
    present = np.array([_present_prob(sc.omit_cal, tab["llr_del"][w.phones[0]:w.phones[1]]) if sc.omit_cal else 1.0
                        for w in ref.words])
    bad = not np.isfinite(ev.L_can) or ev.L_can <= -1e5
    Z = {}
    keep = (sc.universal_bias, sc.sound_bias)
    for b in biases:
        sc.universal_bias = sc.sound_bias = b
        c2 = sc._calibrate(tab, ref, c["dur"], c.get("delivery"), c.get("word_pooled"))
        if bad:
            lo = float(C.logit(C.P_CLIP))
            c2 = {k: np.full_like(np.asarray(c2[k], dtype=np.float64), lo) for k in ("s_zl", "s_zs", "w_zl", "w_zs")}
        Z[b] = tuple(np.asarray(c2[k], np.float32) for k in ("s_zl", "s_zs", "w_zl", "w_zs"))
    sc.universal_bias, sc.sound_bias = keep
    out = {"Z": Z, "raw": raw, "present": present.astype(np.float32)}
    if phone_wrong is not None:
        out["phone_wrong"] = np.asarray(phone_wrong, bool)
        out["word_wrong"] = np.asarray(word_wrong, bool)
    if spans is not None:
        out["spans"] = spans
    return out


def _clip(args):
    rec, wav, swaps, in_fp, biases = args
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
    nw = len(ref.words)
    items = {"clean": _item(res, biases, None, None)}
    inv = inventory(sc.lang)
    pk = KIND.get(sc.lang)
    for tag, seed, kinds in (("cor", "features-v1", SOUND_KINDS + ((pk,) if pk else ())),
                             ("fp", "prosody-v1", ((pk,) + SOUND_KINDS) if (pk and in_fp) else ())):
        for k in kinds:
            try:
                cor = corrupt(ref, [k], inv, n_errors=2, seed=stable_seed(rec["id"], k, seed), strict=False)
            except NotApplicable:
                continue
            if not cor.errors:
                continue
            r2 = sc.score(audio, cor.ref)
            items[f"{tag}:{k}"] = _item(r2, biases, cor.phone_wrong, cor.word_wrong)
    sw = {}
    for s in swaps:
        try:
            r3 = sc.score(audio, s["text"])
        except ValueError:
            continue
        c = r3._pronunciada  # noqa: SLF001
        sw[s["id"]] = _item(r3, biases, spans=[tuple(w.char_span) for w in c["ref"].words])
    return rec["id"], {"items": items, "swap": sw, "fp": in_fp, "nw": nw}


def run(lang, tag, cfg, workers):
    from concurrent.futures import ProcessPoolExecutor

    from bench.pipeline.common import iter_parquet_audio, records
    from pronunciada.calibrate import config

    b0 = float(config()["bias"].get(lang, 0.0)) if cfg.get("b0") is None else float(cfg["b0"])
    biases = sorted(set(grid_for(b0) + [round(b0, 6)] + [round(x, 6) for x in cfg.get("extra", [])]))
    recs = records("fleurs-validation", lang=lang)
    rng = np.random.default_rng(0)            # fit_prosody's subset (collect(): seed 0, n = 200)
    fp_ids = {recs[i]["id"] for i in sorted(rng.choice(len(recs), min(200, len(recs)), replace=False))}
    by_row = {int(r["audio"].split(":")[2]): r for r in recs}
    path = recs[0]["audio"].split(":")[1]
    swaps = defaultdict(list)
    for s in dev_swap(lang):
        swaps[s["src"]].append(s)
    t0 = time.time()
    data = {}
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(lang, cfg)) as ex:
        futs = [ex.submit(_clip, (by_row[i], wav, swaps.get(by_row[i]["id"], []), by_row[i]["id"] in fp_ids, biases))
                for i, wav in iter_parquet_audio(path, set(by_row))]
        for k, f in enumerate(futs):
            uid, out = f.result()
            if out is not None:
                data[uid] = out
            if (k + 1) % 100 == 0:
                print(f"{lang}/{tag}: {k + 1}/{len(futs)} ({time.time() - t0:.0f}s)", flush=True)
    meta = {"lang": lang, "tag": tag, "cfg": cfg, "b0": b0, "biases": biases}
    (work_dir() / f"run_{lang}_{tag}.pkl").write_bytes(pickle.dumps({"meta": meta, "data": data}))
    print(f"{lang}/{tag}: {len(data)} clips, biases {biases} ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- offline metrics

def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def _blend(zl, zs, s):
    from pronunciada.calibrate import blend_logit

    return blend_logit(np.asarray(zl, np.float64), np.minimum(np.asarray(zs, np.float64), np.asarray(zl, np.float64)), s)


def word_probs(it, b, cal, s=0.5):
    from pronunciada.scorer import _pros_prob

    _, _, wzl, wzs = it["Z"][b]
    snd = _sig(_blend(wzl, wzs, s))
    pp = np.array([_pros_prob(cal, r, s) if (cal and r) else 1.0 for r in it["raw"]])
    return snd * pp * it["present"], snd


def refit_prosody(lang, data, b):
    """bench.calib.fit_prosody.run on rows rebuilt at bias b (None if the language has no channel)."""
    from bench.calib import fit_prosody as FP

    if lang not in FP.KIND or lang == "ja":
        return None
    rows = []
    for uid, d in data.items():
        if not d["fp"]:
            continue
        for key, it in d["items"].items():
            if key == "clean":
                tag = "clean"
            elif key.startswith("fp:"):
                tag = "prosody" if key[3:] == FP.KIND[lang] else "sound"
            else:
                continue
            _, _, wzl, wzs = it["Z"][b]
            snd = [_sig(_blend(wzl, wzs, s)) for s in (0.0, 0.5, 1.0)]
            ww = it.get("word_wrong", np.zeros(len(it["raw"]), bool))
            for wi, r in enumerate(it["raw"]):
                rows.append({"utt": uid, "ref": tag, "vals": list(r), "snd": [float(x[wi]) for x in snd],
                             "wrong": bool(ww[wi])})
    for q in FP.Q_GRID:
        FP.Q_LEN, FP.Q_STR = q
        ev = FP.evaluate(rows)
        na = ev["native_accept"]["0.5"]
        if na["before"] - na["after"] <= FP.MAX_NATIVE_DROP:
            break
    par = FP.fit(rows)
    return {"lang": lang, "kind": FP.KIND[lang], **par, "native_quantiles": [FP.Q_LEN, FP.Q_STR],
            "features": "[1, min prosody logit, second-min prosody logit, n scored syllables]",
            "err_prior": FP.ERR_PRIOR, "fit_on": f"FLEURS validation, 200 clips (round 8 bias grid, bias {b})",
            "dev_cv": ev}


def metrics(lang, data, b, cal, n_boot=0):
    from bench.eval_swap import evaluate
    from bench.metrics import roc_auc

    nat_s, nat_w = [], []
    ys, ss, yw, sw, pe = [], [], [], [], []
    for uid, d in data.items():
        it = d["items"]["clean"]
        szl, szs, _, _ = it["Z"][b]
        nat_s.append(_sig(_blend(szl, szs, 0.5)))
        nat_w.append(word_probs(it, b, cal)[0])
        for key, it in d["items"].items():
            if not key.startswith("cor:"):
                continue
            k = key[4:]
            wp = word_probs(it, b, cal)[0]
            if k in SOUND_KINDS:
                if k in PHONE_KINDS:
                    szl, szs, _, _ = it["Z"][b]
                    ys.append(it["phone_wrong"])
                    ss.append(_sig(_blend(szl, szs, 0.5)))
                yw.append(it["word_wrong"])
                sw.append(wp)
            else:
                pe.append(wp[it["word_wrong"]])
    nat_s, nat_w = np.concatenate(nat_s), np.concatenate(nat_w)
    ys, ss = np.concatenate(ys).astype(int), np.concatenate(ss)
    yw, sw = np.concatenate(yw).astype(int), np.concatenate(sw)
    out = {"bias": b, "native_words": float(np.mean(nat_w >= 0.5)), "native_sounds": float(np.mean(nat_s >= 0.5)),
           "sound_auc": float(roc_auc(ys, 1 - ss)), "word_auc": float(roc_auc(yw, 1 - sw))}
    if pe:
        pe = np.concatenate(pe)
        y = np.r_[np.ones(len(pe)), np.zeros(len(nat_w))].astype(int)
        out["pros_auc"] = float(roc_auc(y, 1 - np.r_[pe, nat_w]))
    outs = {}
    for d in data.values():
        for sid, it in d["swap"].items():
            wp = word_probs(it, b, cal)[0]
            outs[sid] = {"status": "Success", "words": [(a, c, float(p)) for (a, c), p in zip(it["spans"], wp)]}
    recs = [r for r in dev_swap(lang) if r["src"] in data]
    m = evaluate(recs, outs, n_boot=n_boot)[lang]["metrics"]
    for v in ("swap_near", "swap_far", "tone"):
        for k in ("auc", "hit@5%FA", "right_word"):
            x = m.get(f"{v}:{k}")
            if x:
                out[f"{v}:{k}"] = [x["value"], x.get("lo"), x.get("hi")]
    pos = []
    for r in recs:
        if r["labels"]["variant"] == "swap_near" and r["id"] in outs:
            a, c = r["labels"]["char_span_in_new_text"]
            s_ = [w[2] for w in outs[r["id"]]["words"] if w[0] < c and a < w[1]]
            if s_:
                pos.append(min(s_))
    out["near_gt09"] = float(np.mean(np.array(pos) > 0.9)) if pos else None
    out["native_fa60"] = m.get("native:fa<60", {}).get("value")
    return out


def load_run(lang, tag):
    return pickle.loads((work_dir() / f"run_{lang}_{tag}.pkl").read_bytes())


def select(lang, tag, base_tag, write_dir=None):
    """Grid table + the rule's choice. Baseline = base_tag run at its shipped bias with the
    shipped prosody calibrator."""
    from pronunciada.calibrate import config
    from pronunciada.scorer import Scorer  # noqa: F401  (path of the shipped calibrator)
    from bench.calib.common import ship_dir

    R0 = load_run(lang, base_tag)
    b_ship = round(float(config()["bias"].get(lang, 0.0)), 6)
    pp = ship_dir() / "prosody" / f"{lang}.json"
    cal0 = json.loads(pp.read_text()) if pp.exists() else None
    if cal0 and cal0.get("experimental"):
        cal0 = None
    base = metrics(lang, R0["data"], b_ship, cal0)
    R = load_run(lang, tag)
    rows, cals = [], {}
    for b in R["meta"]["biases"]:
        cal = refit_prosody(lang, R["data"], b)
        cals[b] = cal
        m = metrics(lang, R["data"], b, cal)
        feas = {"native_words": base["native_words"] - m["native_words"] <= RULE["native_words"],
                "native_sounds": base["native_sounds"] - m["native_sounds"] <= RULE["native_sounds"],
                "sound_auc": base["sound_auc"] - m["sound_auc"] <= RULE["sound_auc"],
                "word_auc": base["word_auc"] - m["word_auc"] <= RULE["word_auc"]}
        if "pros_auc" in base and "pros_auc" in m:
            feas["pros_auc"] = base["pros_auc"] - m["pros_auc"] <= RULE["pros_auc"]
        m["feasible"] = all(feas.values())
        m["fails"] = [k for k, v in feas.items() if not v]
        rows.append(m)
    ok = [r for r in rows if r["feasible"]]
    choice = max(ok, key=lambda r: (round(r["swap_near:auc"][0], 6), abs(r["bias"]))) if ok else None
    res = {"lang": lang, "tag": tag, "base_tag": base_tag, "baseline": base, "grid": rows,
           "choice": choice["bias"] if choice else None}
    if choice:
        res["choice_ci"] = metrics(lang, R["data"], choice["bias"], cals[choice["bias"]], n_boot=1000)
        res["choice_prosody"] = cals[choice["bias"]]
    res["baseline_ci"] = metrics(lang, R0["data"], b_ship, cal0, n_boot=1000)
    (work_dir() / f"select_{lang}_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"{lang} baseline (bias {b_ship}):", _fmt(base))
    for r in rows:
        print(f"  b={r['bias']:+.2f} {'OK ' if r['feasible'] else 'x  '}{_fmt(r)} {r['fails']}")
    print(f"{lang} choice: {res['choice']}", flush=True)
    return res


def _fmt(m):
    g = lambda k: (m.get(k) or [float("nan")])[0] if isinstance(m.get(k), list) else m.get(k, float("nan"))  # noqa: E731
    return (f"near {g('swap_near:auc'):.4f} >0.9 {m.get('near_gt09') or float('nan'):.3f} hit5 {g('swap_near:hit@5%FA'):.3f} "
            f"rw {g('swap_near:right_word'):.3f} far {g('swap_far:auc'):.4f} natW {m['native_words']:.4f} "
            f"natS {m['native_sounds']:.4f} snd {m['sound_auc']:.4f} word {m['word_auc']:.4f} "
            f"pros {m.get('pros_auc', float('nan')):.4f}" + (f" tone {g('tone:auc'):.4f}" if 'tone:auc' in m else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build-dev", "run", "select"])
    ap.add_argument("--lang", required=True)
    ap.add_argument("--tag", default="shipped")
    ap.add_argument("--base", default="shipped")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--measured", action="store_true")
    ap.add_argument("--vowel-min", type=float, default=None)
    ap.add_argument("--b0", type=float, default=None, help="grid end (default: the shipped bias)")
    ap.add_argument("--extra", default="", help="extra bias values, comma list")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    if a.step == "build-dev":
        build_dev(a.lang)
    elif a.step == "run":
        run(a.lang, a.tag, {"adapter": a.adapter, "measured": a.measured, "vowel_min": a.vowel_min, "b0": a.b0,
                                 "extra": [float(x) for x in a.extra.split(",") if x]},
            a.workers)
    else:
        select(a.lang, a.tag, a.base)


if __name__ == "__main__":
    main()
