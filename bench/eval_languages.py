"""All 10 languages on FLEURS TEST through the public API: the "verified support" table.

    python -m bench.eval_languages [--steps run,eval,swap,report] [--workers 5] [--lang all]

Per clip: Scorer(lang).score(audio_path, text) on the correct prompt (native acceptance), then the
same audio against controlled-error references (bench.corrupt kinds phone_sub_near,
phone_sub_far, phone_del, phone_ins, word_swap; 2 errors / utterance; seed = hash(utt id, kind),
the protocol of the feature tables) passed as Reference objects to the same Scorer.score call
(the posteriors of the clip are reused from the scorer's cache, nothing else is). Writes
results/languages.md and results/tables/languages.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time

import numpy as np

from bench.calib.common import calib_cache, load_json, save_json

KINDS = ("phone_sub_near", "phone_sub_far", "phone_del", "phone_ins", "word_swap")
PHONE_KINDS = ("phone_sub_near", "phone_sub_far", "phone_ins", "word_swap")
S_LIST = (0.0, 0.5, 1.0)


PROS_KIND = {"zh": "tone", "ja": "pitch", "en": "stress", "de": "stress", "es": "stress",
             "it": "stress", "pt": "stress", "ru": "stress"}


def _pack(sc, res):
    """Per reference: for variant "on" (shipped, with explain-away) and "off" (first pass): phone
    log-odds, word sounds-based P and WordResult.prob (sounds x prosody channel) at S_LIST,
    sentence log-odds."""
    from pronunciada.calibrate import blend_logit, sigmoid

    c = res._pronunciada  # noqa: SLF001
    fac = {}
    for s_ in S_LIST:
        r = sc.rescore(res, s_)
        fac[s_] = np.array([w.prob / w.sounds_prob if w.sounds_prob else 1.0 for w in r.words], np.float64)
    out = {"szl": float(c["sent_zl"]), "szs": float(c["sent_zs"]), "V": {},
           "phone_word": np.array([p.word for p in c["ref"].phones], np.int32),
           "explained": list(c.get("explained", []))}
    for var, cc in (("on", c), ("off", c.get("first_pass", c))):
        wl, ws_ = np.asarray(cc["w_zl"]), np.minimum(np.asarray(cc["w_zs"]), np.asarray(cc["w_zl"]))
        snd = {s_: sigmoid(blend_logit(wl, ws_, s_)) for s_ in S_LIST}
        out["V"][var] = {"zl": np.asarray(cc["s_zl"], np.float32), "zs": np.asarray(cc["s_zs"], np.float32),
                         "ws": {s_: snd[s_].astype(np.float32) for s_ in S_LIST},
                         "wp": {s_: (snd[s_] * fac[s_]).astype(np.float32) for s_ in S_LIST}}
    return out


def _utt(rec):
    from bench.calib import api_run
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.common import stable_seed
    from bench.pipeline.evalctl import inventory

    sc = api_run._SC
    try:
        t = time.perf_counter()
        res = sc.score(rec["audio"], rec["text"])
        lat = (time.perf_counter() - t) * 1e3
    except ValueError as e:
        return rec["id"], {"error": str(e)}
    out = {"clean": _pack(sc, res), "latency_ms": lat, "timing": res.timing_ms, "dur": rec["dur"],
           "warnings": res.warnings, "cor": {}}
    ref = res._pronunciada["ref"]  # noqa: SLF001
    inv = inventory(sc.lang)
    kinds = KINDS + ((PROS_KIND[sc.lang],) if sc.lang in PROS_KIND else ())
    for kind in kinds:
        try:
            cor = corrupt(ref, [kind], inv, n_errors=2, seed=stable_seed(rec["id"], kind, "features-v1"),
                          strict=False)
        except NotApplicable:
            continue
        if not cor.errors:
            continue
        r2 = sc.score(rec["audio"], cor.ref)
        d = _pack(sc, r2)
        d["phone_wrong"] = np.asarray(cor.phone_wrong, bool)
        d["word_wrong"] = np.asarray(cor.word_wrong, bool)
        out["cor"]["prosody" if kind == PROS_KIND.get(sc.lang) else kind] = d
    return rec["id"], out


def run_lang(lang, workers, force=False):
    from concurrent.futures import ProcessPoolExecutor

    from bench.calib.api_run import _init
    from bench.data import load

    d = calib_cache() / "api"
    d.mkdir(exist_ok=True)
    path = d / f"fleurs-test-{lang}.pkl"
    recs = load("fleurs", split="test", lang=lang)
    if path.exists() and not force:
        out = pickle.loads(path.read_bytes())
        if all(r["id"] in out for r in recs):
            return out
    t0 = time.time()
    out = {}
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(lang, "fp32", 1)) as ex:
        for k, (uid, s) in enumerate(ex.map(_utt, recs, chunksize=2)):
            out[uid] = s
            if (k + 1) % 100 == 0:
                print(f"fleurs/{lang}: {k + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    path.write_bytes(pickle.dumps(out))
    print(f"fleurs/{lang}: {len(out)} utts in {time.time() - t0:.0f}s", flush=True)
    return out


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def pblend(zl, zs, s):
    from pronunciada.calibrate import blend_logit

    return sig(blend_logit(zl, np.minimum(zs, zl), s))


def eval_lang(lang, data, bias, var="on"):
    """var: "on" = shipped (with the explain-away pass), "off" = first pass only."""
    from bench.calib.common import ece
    from bench.metrics import bootstrap_ci, roc_auc

    ok = {k: v for k, v in data.items() if "error" not in v}
    V = lambda d: d["V"][var]  # noqa: E731
    r = {"utts": len(ok), "errors": len(data) - len(ok), "native": {}, "native_before_bias": {},
         "native_sounds_only_words": {}}
    for s in S_LIST:
        ps = np.concatenate([pblend(V(v["clean"])["zl"], V(v["clean"])["zs"], s) for v in ok.values()])
        pw = np.concatenate([V(v["clean"])["wp"][s] for v in ok.values()])
        pws = np.concatenate([V(v["clean"])["ws"][s] for v in ok.values()])
        pz = np.array([pblend(v["clean"]["szl"], v["clean"]["szs"], s) for v in ok.values()])
        r["native"][str(s)] = {"sounds": float(np.mean(ps >= 0.5)), "words": float(np.mean(pw >= 0.5)),
                               "sentences": float(np.mean(pz >= 0.5))}
        r["native_sounds_only_words"][str(s)] = float(np.mean(pws >= 0.5))
        ps0 = np.concatenate([pblend(V(v["clean"])["zl"] - bias, V(v["clean"])["zs"] - bias, s) for v in ok.values()])
        r["native_before_bias"][str(s)] = {"sounds": float(np.mean(ps0 >= 0.5))}
    nat_s = np.concatenate([pblend(V(v["clean"])["zl"], V(v["clean"])["zs"], 0.5) for v in ok.values()])
    nat_w = np.concatenate([V(v["clean"])["wp"][0.5] for v in ok.values()])
    nat_ws = np.concatenate([V(v["clean"])["ws"][0.5] for v in ok.values()])
    thr_s, thr_w = np.quantile(nat_s, 0.05), np.quantile(nat_w, 0.05)
    r["thr_5pct"] = {"sound": float(thr_s), "word": float(thr_w)}
    ys, ss, gs, ks = [], [], [], []
    yw, sw, gw, sws = [], [], [], []
    lowest, n_ref = 0, 0
    pros_err, pros_err_s = [], []
    snd_err, snd_err_s = [], []
    for uid, v in ok.items():
        for kind, d in v["cor"].items():
            p = pblend(V(d)["zl"], V(d)["zs"], 0.5)
            pwd, pwd_s = V(d)["wp"][0.5], V(d)["ws"][0.5]
            if kind == "prosody":
                pros_err.append(pwd[d["word_wrong"]])
                pros_err_s.append(pwd_s[d["word_wrong"]])
                continue
            snd_err.append(pwd[d["word_wrong"]])
            snd_err_s.append(pwd_s[d["word_wrong"]])
            if kind in PHONE_KINDS:
                ys.append(d["phone_wrong"])
                ss.append(p)
                gs += [uid] * len(p)
                ks += [kind] * len(p)
            yw.append(d["word_wrong"])
            sw.append(pwd)
            sws.append(pwd_s)
            gw += [uid] * len(pwd)
            if d["word_wrong"].any() and (~d["word_wrong"]).any():
                n_ref += 1
                lowest += bool(d["word_wrong"][int(np.argmin(pwd))])
    ys, ss, gs, ks = np.concatenate(ys).astype(int), np.concatenate(ss), np.array(gs), np.array(ks)
    yw, sw, gw, sws = np.concatenate(yw).astype(int), np.concatenate(sw), np.array(gw), np.concatenate(sws)
    ci = bootstrap_ci(roc_auc, ys, 1 - ss, clusters=gs, n_boot=300)
    r["sound"] = {"auc": roc_auc(ys, 1 - ss), "auc_ci": [ci["lo"], ci["hi"]], "n_wrong": int(ys.sum()),
                  "n_ok": int(len(ys) - ys.sum()), "hit_at_5pct": float(np.mean(ss[ys == 1] < thr_s)),
                  "ece_vs_controlled": ece(1 - ys, ss), "by_kind": {}}
    for kind in PHONE_KINDS:
        m = ks == kind
        if m.any() and 0 < ys[m].sum() < m.sum():
            r["sound"]["by_kind"][kind] = {"auc": roc_auc(ys[m], 1 - ss[m]),
                                           "hit_at_5pct": float(np.mean(ss[m][ys[m] == 1] < thr_s))}
    ci = bootstrap_ci(roc_auc, yw, 1 - sw, clusters=gw, n_boot=300)
    r["word"] = {"auc": roc_auc(yw, 1 - sw), "auc_ci": [ci["lo"], ci["hi"]], "auc_sounds_only": roc_auc(yw, 1 - sws),
                 "n_wrong": int(yw.sum()), "n_ok": int(len(yw) - yw.sum()), "hit_at_5pct": float(np.mean(sw[yw == 1] < thr_w)),
                 "lowest_is_corrupted": lowest / max(n_ref, 1), "n_refs": n_ref}

    def vs_native(err, nat):
        err = np.concatenate(err) if err else np.zeros(0)
        if not len(err):
            return {"auc": float("nan"), "hit_at_5pct": float("nan"), "n_err": 0}
        y = np.r_[np.ones(len(err)), np.zeros(len(nat))].astype(int)
        return {"auc": roc_auc(y, 1 - np.r_[err, nat]), "hit_at_5pct": float(np.mean(err < np.quantile(nat, 0.05))),
                "n_err": int(len(err))}

    # prosody channel report: error words vs native words (s = 0.5)
    r["prosody"] = {"kind": PROS_KIND.get(lang),
                    "prosodic_error": {"product": vs_native(pros_err, nat_w), "sounds_only": vs_native(pros_err_s, nat_ws)},
                    "sound_error": {"product": vs_native(snd_err, nat_w), "sounds_only": vs_native(snd_err_s, nat_ws)}}
    lat = np.array([v["latency_ms"] for v in ok.values()])
    dur = np.array([v["dur"] for v in ok.values()])
    r["latency"] = {"median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)),
                    "median_audio_s": float(np.median(dur)), "rtf": float(lat.sum() / 1e3 / dur.sum()),
                    "explain_away_median_ms": float(np.median([v["timing"].get("explain_away", 0.0) for v in ok.values()])),
                    "share_utts_with_pass": float(np.mean([bool(v["clean"]["explained"]) for v in ok.values()]))}
    r["warnings"] = int(sum(bool(v["warnings"]) for v in ok.values()))
    return r


# ----------------------------------------------------------------------------- Azure text-swap

def swap_eval(workers):
    """Our scorer on bench/eval_swap.py's fleurs-swap records (documented interface:
    {"status", "words": [(char_start, char_end, score)], "sentence": {...}})."""
    from bench.pipeline.common import repo_root

    root = repo_root()
    if not (root / "bench" / "eval_swap.py").exists() or not (root / "results" / "azure_swap.md").exists():
        return None
    from bench.data import load

    try:
        recs = load("fleurs-swap", split="test")
    except FileNotFoundError:
        return None
    outs = {}
    by_lang = {}
    for r in recs:
        by_lang.setdefault(r["lang"], []).append(r)
    from concurrent.futures import ProcessPoolExecutor

    from bench.calib.api_run import _init

    for lang, rs in sorted(by_lang.items()):
        p = calib_cache() / "api" / f"fleurs-swap-{lang}.pkl"
        if p.exists():
            o = pickle.loads(p.read_bytes())
        else:
            with ProcessPoolExecutor(workers, initializer=_init, initargs=(lang, "fp32", 1)) as ex:
                o = dict(ex.map(_swap_one, rs, chunksize=6))   # orig + variants of a clip together
            p.write_bytes(pickle.dumps(o))
        outs.update(o)
        print("swap", lang, len(o), flush=True)
    return outs


def _swap_one(rec):
    from bench.calib import api_run
    from bench.eval_swap import sentence_result_output
    from pronunciada.calibrate import blend_logit, sigmoid

    sc = api_run._SC
    try:
        res = sc.score(rec["audio"], rec["text"])
    except ValueError as e:
        o = {"status": f"error: {e}", "words": []}
        return rec["id"], {"on": o, "off": o}
    c = res._pronunciada  # noqa: SLF001
    spans = [tuple(w.char_span) for w in c["ref"].words]
    on = sentence_result_output(res, spans)
    f = c.get("first_pass")
    if f is None:
        return rec["id"], {"on": on, "off": on}
    snd = sigmoid(blend_logit(f["w_zl"], np.minimum(f["w_zs"], f["w_zl"]), res.strictness))
    words = [(a, b, float(snd[k] * (w.prob / w.sounds_prob if w.sounds_prob else 1.0)))
             for k, (w, (a, b)) in enumerate(zip(res.words, spans))]
    return rec["id"], {"on": on, "off": {**on, "words": words}}


def swap_tables(outs) -> tuple[str, dict]:
    """bench.eval_swap.evaluate for ours and for Azure (its cached responses) on the same records;
    side-by-side markdown + both result dicts."""
    from bench.data import load
    from bench.eval_swap import VARIANTS, azure_outputs, evaluate

    recs = load("fleurs-swap", split="test")
    ours = evaluate(recs, {k: v["on"] for k, v in outs.items()})
    ours_off = evaluate(recs, {k: v["off"] for k, v in outs.items()})
    az = evaluate(recs, azure_outputs(recs))

    def g(r, k):
        x = r["metrics"].get(k)
        if not x or x.get("value") is None or (isinstance(x["value"], float) and not np.isfinite(x["value"])):
            return "–"
        v = x["value"]
        return f"{v:.3f}" + (f" [{x['lo']:.3f}, {x['hi']:.3f}]" if "lo" in x else "")

    L = ["Same records, same evaluation code (`bench.eval_swap.evaluate`, utterance-clustered 95 % CIs); "
         "ours = `Scorer(lang).score(audio, text)` word probabilities via `bench.eval_swap.sentence_result_output` "
         "(omitted words score 0), sentence score = SentenceResult.prob; Azure = its cached responses "
         "(word AccuracyScore / 100, sentence AccuracyScore). AUC = replaced word vs the untouched words of the "
         "swapped records; hit@5%FA = share of replaced words flagged at the threshold flagging 5 % of the words "
         "of the original records; right word = replaced word has the lowest score; sent AUC = swapped vs "
         "original record by sentence score.\n"]
    for v in VARIANTS:
        rows = [lang for lang in ours if f"{v}:auc" in ours[lang]["metrics"]]
        if not rows:
            continue
        L.append(f"**{v}**\n")
        L.append("| lang | n | AUC ours | AUC Azure | hit@5%FA ours | hit@5%FA Azure | right word ours | right word Azure | sent AUC ours | sent AUC Azure |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for lang in rows:
            o, a_ = ours[lang], az.get(lang, {"metrics": {}})
            L.append(f"| {lang} | {o['metrics'][f'{v}:n']['value']} | {g(o, f'{v}:auc')} | {g(a_, f'{v}:auc')} | "
                     f"{g(o, f'{v}:hit@5%FA')} | {g(a_, f'{v}:hit@5%FA')} | {g(o, f'{v}:right_word')} | "
                     f"{g(a_, f'{v}:right_word')} | {g(o, f'{v}:sent_auc_prob')} | {g(a_, f'{v}:sent_auc_AccuracyScore')} |")
        L.append("")
    L.append("Native (orig) words: share with score < 0.6 / < 0.8 (ours / Azure):\n")
    L.append("| lang | FA<0.6 ours | FA<0.6 Azure | FA<0.8 ours | FA<0.8 Azure |\n|---|---|---|---|---|")
    for lang in ours:
        o, a_ = ours[lang], az.get(lang, {"metrics": {}})
        L.append(f"| {lang} | {g(o, 'native:fa<60')} | {g(a_, 'native:fa<60')} | {g(o, 'native:fa<80')} | {g(a_, 'native:fa<80')} |")
    return "\n".join(L) + "\n", {"ours": ours, "ours_no_explain": ours_off, "azure": az}


# ----------------------------------------------------------------------------- report

def f3(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def report(R):
    from bench.pipeline.common import LANGS, repo_root

    cfg = json.loads((__import__("pronunciada.calibrate", fromlist=["data_dir"]).data_dir() / "config.json").read_text())
    L = ["# All 10 languages on FLEURS TEST (public API)\n",
         "`pronunciada.Scorer(lang).score(audio_path, text)` on native FLEURS test speech (correct prompts) and on "
         "controlled-error references of the same audio (bench.corrupt: phone_sub_near, phone_sub_far, phone_del, "
         "phone_ins, word_swap; 2 errors per utterance; fixed seeds). Controlled errors are a PROXY for learner "
         "errors (the audio is native; the reference is wrong), so these AUCs are much higher than on real learner "
         "data (see english_vs_azure.md §3). Sound model: English = shipped B, other languages = universal U + the "
         "per-language logit bias fitted on FLEURS train. Generated by `python -m bench.eval_languages`.\n",
         "**Change made after looking at this test set (stated as required):** the first run multiplied the "
         "prosody model's tone (zh) / pitch-accent (ja) probabilities into WordResult.prob. It gave native word "
         "acceptance 0.374 (zh) and 0.718 (ja) at s = 0.5 and word AUC 0.649 / 0.676. The per-word tone / pitch "
         "factor on native speech has median 0.36 (zh) / 0.63 (ja) on this test set and 0.37 / 0.64 on 40 FLEURS-"
         "validation clips per language, so the product rejects correctly spoken words. WordResult.prob became the "
         "sounds-based probability (tone_prob / stress_prob in their own fields); zh and ja were re-run. In round 2 "
         "(§5) a LEARNED combination replaced it where it passed the validation rule (zh, ja, ru). "
         "No other language was affected (no stress model was available for en / de / es / it / pt / ru).\n",
         "## 1. Verified-support table (strictness 0.5 unless stated)\n",
         "Native acceptance = share of native sounds / words / sentences with P ≥ 0.5 (word P = WordResult.prob: round-2 word model, plus the learned tone / pitch / stress combination for zh / ja / ru). Error AUC = ROC AUC of "
         "1 − P for corrupted vs untouched units of the corrupted references (sound: sub_near, sub_far, ins, "
         "word_swap; word: all 5 kinds). Hit @ 5 % = share of corrupted units flagged at the threshold that flags "
         "5 % of native units. Lowest word = share of corrupted references whose lowest-scoring word is a "
         "corrupted one. ECE = 15 equal-mass bins, sound P vs the controlled labels. Latency = median end-to-end "
         "ms per clip (clean prompt, 1 ONNX thread, 5 processes in parallel).\n",
         "| lang | utts | native sounds | native words | native sentences | sound AUC [95 % CI] | sound hit @ 5 % | word AUC | word hit @ 5 % | lowest word | sound ECE | median ms (audio s) | bias |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    aucs = {}
    for lang in LANGS:
        r = R["by_lang"].get(lang)
        if not r:
            continue
        n = r["native"]["0.5"]
        aucs[lang] = r["sound"]["auc"]
        L.append(f"| {lang} | {r['utts']} | {n['sounds']:.3f} | {n['words']:.3f} | {n['sentences']:.3f} | "
                 f"{f3(r['sound']['auc'])} [{f3(r['sound']['auc_ci'][0])}, {f3(r['sound']['auc_ci'][1])}] | "
                 f"{r['sound']['hit_at_5pct']:.3f} | {f3(r['word']['auc'])} | {r['word']['hit_at_5pct']:.3f} | "
                 f"{r['word']['lowest_is_corrupted']:.3f} | {r['sound']['ece_vs_controlled']:.4f} | "
                 f"{r['latency']['median_ms']:.0f} ({r['latency']['median_audio_s']:.1f}) | "
                 f"{cfg['bias'].get(lang, 0.0):+.3f} |")
    L.append("")
    med = float(np.median(list(aucs.values())))
    weak = [l for l, a in aucs.items() if a < med - 0.01]
    L.append(f"Median sound AUC {med:.3f}; languages more than 0.01 below the median: "
             f"{', '.join(weak) if weak else 'none'}.\n")
    L.append("## 2. Native acceptance vs strictness\n")
    L.append("| lang | sounds s=0 / 0.5 / 1 | words s=0 / 0.5 / 1 | sentences s=0 / 0.5 / 1 | sounds s=0.5 without bias |")
    L.append("|---|---|---|---|---|")
    for lang in LANGS:
        r = R["by_lang"].get(lang)
        if not r:
            continue
        n = r["native"]
        L.append(f"| {lang} | " + " | ".join(" / ".join(f"{n[str(s)][k]:.3f}" for s in S_LIST)
                                            for k in ("sounds", "words", "sentences"))
                 + f" | {r['native_before_bias']['0.5']['sounds']:.3f} |")
    L.append("")
    L.append("## 3. Sound-level AUC and hit rate by corruption kind\n")
    L.append("| lang | " + " | ".join(PHONE_KINDS) + " |\n|---|" + "---|" * len(PHONE_KINDS))
    for lang in LANGS:
        r = R["by_lang"].get(lang)
        if not r:
            continue
        L.append(f"| {lang} | " + " | ".join(
            f"{f3(r['sound']['by_kind'].get(k, {}).get('auc'))} / {r['sound']['by_kind'].get(k, {}).get('hit_at_5pct', float('nan')):.3f}"
            for k in PHONE_KINDS) + " |")
    L.append("\n(AUC / hit rate at the 5 %-native-flag threshold)\n")
    sw = R.get("swap")
    L.append("## 4. Same clips, one word swapped in the text: ours vs Azure\n")
    if sw is None:
        L.append("Not run: `results/azure_swap.md` and/or `bench/eval_swap.py` (the other agent's text-swap "
                 "benchmark) did not exist when this report was generated.\n")
    else:
        L.append(sw)
    r1p = repo_root() / "results" / "tables" / "round1" / "languages.json"
    if r1p.exists():
        R1 = json.loads(r1p.read_text())
        L.append("## 5. Rounds 2 + 3 vs round 1 — before → after (round 1 → current shipped)\n")
        L.append("Round 2 was designed AFTER the first held-out evaluation (these test tables motivated it); all "
                 "fitting and selection used train splits and FLEURS validation only (results/calibration.md §10). "
                 "Sound-level models are unchanged. Word = WordResult.prob at s = 0.5 (current: sounds x prosody "
                 "channel, explain-away on).\n")
        L.append("| lang | native words | word AUC | word hit @ 5 % | lowest word | native sentences |")
        L.append("|---|---|---|---|---|---|")
        for lang in LANGS:
            a, b = R1["by_lang"].get(lang), R["by_lang"].get(lang)
            if not a or not b:
                continue
            L.append(f"| {lang} | {a['native']['0.5']['words']:.3f} → {b['native']['0.5']['words']:.3f} | "
                     f"{a['word']['auc']:.3f} → {b['word']['auc']:.3f} | {a['word']['hit_at_5pct']:.3f} → {b['word']['hit_at_5pct']:.3f} | "
                     f"{a['word']['lowest_is_corrupted']:.3f} → {b['word']['lowest_is_corrupted']:.3f} | "
                     f"{a['native']['0.5']['sentences']:.3f} → {b['native']['0.5']['sentences']:.3f} |")
        L.append("")
        s1p = repo_root() / "results" / "tables" / "round1" / "swap_ours_vs_azure.json"
        s2p = repo_root() / "results" / "tables" / "swap_ours_vs_azure.json"
        if s1p.exists() and s2p.exists():
            S1, S2 = json.loads(s1p.read_text()), json.loads(s2p.read_text())

            def gv(r, k):
                x = r["metrics"].get(k) if r else None
                return float("nan") if not x or x.get("value") is None else float(x["value"])

            L.append("Azure swap benchmark, ours round 1 → round 2 (Azure in brackets):\n")
            L.append("| variant | lang | AUC | right word | sent AUC |\n|---|---|---|---|---|")
            for v in ("swap_far", "swap_near", "tone"):
                for lang in sorted(S2["ours"]):
                    if f"{v}:auc" not in S2["ours"][lang]["metrics"]:
                        continue
                    o1, o2, az = S1["ours"].get(lang), S2["ours"][lang], S2["azure"].get(lang)
                    L.append(f"| {v} | {lang} | {gv(o1, f'{v}:auc'):.3f} → {gv(o2, f'{v}:auc'):.3f} ({gv(az, f'{v}:auc'):.3f}) | "
                             f"{gv(o1, f'{v}:right_word'):.3f} → {gv(o2, f'{v}:right_word'):.3f} ({gv(az, f'{v}:right_word'):.3f}) | "
                             f"{gv(o1, f'{v}:sent_auc_prob'):.3f} → {gv(o2, f'{v}:sent_auc_prob'):.3f} ({gv(az, f'{v}:sent_auc_AccuracyScore'):.3f}) |")
            L.append("")
    exp_p = repo_root() / "results" / "tables" / "round3_explain" / "languages.json"
    RX = json.loads(exp_p.read_text()) if exp_p.exists() else R
    if RX.get("by_lang_no_explain"):
        L.append("## 6. Round 3 — designed after the earlier held-out evaluations\n")
        L.append("All round-3 fitting and selection used FLEURS validation and train splits only "
                 "(results/calibration.md §11); FLEURS test was then run once.\n")
        L.append("### 6a. Prosody channel (WordResult.prob = sounds_prob × P_pros)\n")
        L.append("Per language: native word acceptance at s = 0 / 0.5 / 1 (sounds only → with the channel); "
                 "prosodic-error word AUC and hit @ 5 % native flags (the word carrying a controlled tone / stress / "
                 "pitch error vs native words; WordResult.prob); sound-error word AUC (words with a controlled sound "
                 "error vs native words), sounds only → with the channel. dev = FLEURS validation, 5-fold "
                 "utterance-grouped CV of the fit; test = FLEURS test.\n")
        L.append("| lang | kind | set | native words s=0 / 0.5 / 1 | prosodic-error AUC (P_pros alone) | prosodic hit @ 5 % | sound-error AUC sounds → product |")
        L.append("|---|---|---|---|---|---|---|")
        tdir = repo_root() / "results" / "tables"
        for lang in LANGS:
            pj = tdir / f"prosody_{lang}.json"
            if pj.exists():
                d = json.loads(pj.read_text())["dev_cv"]
                na = d["native_accept"]
                L.append(f"| {lang} | {json.loads(pj.read_text())['kind']} | dev | " + " / ".join(
                    f"{na[k]['before']:.3f}→{na[k]['after']:.3f}" for k in ("0.0", "0.5", "1.0"))
                    + f" | {d['prosodic_error']['product']['auc']:.3f} ({d['prosodic_error']['p_pros']['auc']:.3f}) | "
                      f"{d['prosodic_error']['product']['hit_at_5pct']:.3f} | {d['sound_error']['before']['auc']:.3f} → {d['sound_error']['after']['auc']:.3f} |")
            r = R["by_lang"].get(lang)
            if r and r.get("prosody", {}).get("kind"):
                pr = r["prosody"]
                L.append(f"| {lang} | {pr['kind']} | test | " + " / ".join(
                    f"{r['native_sounds_only_words'][k]:.3f}→{r['native'][k]['words']:.3f}" for k in ("0.0", "0.5", "1.0"))
                    + f" | {pr['prosodic_error']['product']['auc']:.3f} | {pr['prosodic_error']['product']['hit_at_5pct']:.3f} | "
                      f"{pr['sound_error']['sounds_only']['auc']:.3f} → {pr['sound_error']['product']['auc']:.3f} |")
        L.append("")
        L.append("### 6b. Explain-away second pass, FLEURS test (without → with)\n")
        L.append("Run with the pass enabled (results/tables/round3_explain/); the pass is NOT shipped: it lowered the "
                 "speechocean762 correlations (english_vs_azure.md §8) and did not improve \"right word\" here or on "
                 "the swap benchmark, so `explain_away` is off by default (opt-in `Scorer(..., explain_away=True)`). "
                 "Every other table in this file is from the final run without the pass.\n")
        L.append("| lang | word AUC | word hit @ 5 % | lowest word | native words s=0.5 | sound AUC | utts with a 2nd pass | added ms (median) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for lang in LANGS:
            a, b = RX["by_lang_no_explain"].get(lang), RX["by_lang"].get(lang)
            if not a or not b:
                continue
            L.append(f"| {lang} | {a['word']['auc']:.3f} → {b['word']['auc']:.3f} | {a['word']['hit_at_5pct']:.3f} → {b['word']['hit_at_5pct']:.3f} | "
                     f"{a['word']['lowest_is_corrupted']:.3f} → {b['word']['lowest_is_corrupted']:.3f} | "
                     f"{a['native']['0.5']['words']:.3f} → {b['native']['0.5']['words']:.3f} | {a['sound']['auc']:.3f} → {b['sound']['auc']:.3f} | "
                     f"{b['latency']['share_utts_with_pass']:.2f} | {b['latency']['explain_away_median_ms']:.1f} |")
        L.append("")
        s2p = repo_root() / "results" / "tables" / "round3_explain" / "swap_ours_vs_azure.json"
        if s2p.exists():
            S2 = json.loads(s2p.read_text())
            if "ours_no_explain" in S2:
                def gv(rr, k):
                    x = rr["metrics"].get(k) if rr else None
                    return float("nan") if not x or x.get("value") is None else float(x["value"])

                L.append("### 6c. Azure swap benchmark — ours without → with explain-away (Azure)\n")
                L.append("| variant | lang | AUC | hit @ 5 % FA | right word |\n|---|---|---|---|---|")
                for v in ("swap_far", "swap_near", "tone"):
                    for lang in sorted(S2["ours"]):
                        if f"{v}:auc" not in S2["ours"][lang]["metrics"]:
                            continue
                        o0, o1, az = S2["ours_no_explain"].get(lang), S2["ours"][lang], S2["azure"].get(lang)
                        L.append(f"| {v} | {lang} | " + " | ".join(
                            f"{gv(o0, f'{v}:{m}'):.3f} → {gv(o1, f'{v}:{m}'):.3f} ({gv(az, f'{v}:{m}'):.3f})"
                            for m in ("auc", "hit@5%FA", "right_word")) + " |")
                L.append("")
    (repo_root() / "results" / "languages.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote results/languages.md")


def main():
    from bench.pipeline.common import parse_langs

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="run,eval,swap,report")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--lang", default="all")
    a = ap.parse_args()
    steps = a.steps.split(",")
    langs = parse_langs(a.lang)
    R = load_json("languages.json") or {"by_lang": {}}
    from pronunciada.calibrate import config

    cfg = config()
    for lang in langs:
        if "run" in steps or "eval" in steps:
            data = run_lang(lang, a.workers)
            if "eval" in steps:
                b = 0.0 if cfg["sound_model"].get(lang, cfg["universal_sound_model"]) != cfg["universal_sound_model"] \
                    else float(cfg["bias"].get(lang, 0.0))
                R["by_lang"][lang] = eval_lang(lang, data, b)
                R.setdefault("by_lang_no_explain", {})[lang] = eval_lang(lang, data, b, var="off")
                save_json("languages.json", R)
                r = R["by_lang"][lang]
                print(lang, "native", r["native"]["0.5"], "sound auc", round(r["sound"]["auc"], 4),
                      "word auc", round(r["word"]["auc"], 4), flush=True)
    if "swap" in steps:
        outs = swap_eval(a.workers)
        if outs is not None:
            md, res = swap_tables(outs)
            R["swap_outputs"] = len(outs)
            R["swap_md"] = md
            save_json("languages.json", R)
            save_json("swap_ours_vs_azure.json", res)
    if "report" in steps:
        R["swap"] = R.get("swap_md")
        report(R)


if __name__ == "__main__":
    main()
