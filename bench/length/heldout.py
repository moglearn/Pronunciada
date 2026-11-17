"""Round 15: held-out evaluation of an adopted length channel, paired before / after, ONCE per language.

    python -m bench.length.heldout run --lang ja [--parts fleurs,swap,cv]
    python -m bench.length.heldout report --lang ja      # -> results/tables/round15_length_heldout_<lang>.json

Every clip is scored ONCE by the public Scorer with the length channel off (Scorer(lang,
length=False)); "after" applies the shipped channel (pronunciada/data/length/<lang>.json) to the
same evidence offline: length_prob = pronunciada.length.score on the stored timing inputs, word
probability x pronunciada.length.word_factor, exactly the runtime arithmetic (checked against the
runtime Scorer on a few clips in `report`). Sound and sentence probabilities do not depend on the
channel.
Parts (all TEST data, never used for fitting or selection):
  fleurs  FLEURS test: natives, planted length swaps (bench/audit/contrasts.py generators, seeds
          "length-v1", as on dev) and the ten-language benchmark's controlled errors
          (bench/eval_languages.py: 5 kinds + the prosody kind, 2 errors, seeds "features-v1"),
          evaluated with bench.eval_languages.eval_lang (before / after)
  swap    fleurs-swap test records of the language (the existing swap benchmark), evaluated with
          bench.eval_swap.evaluate (before / after)
  cv      cv-real test half of the language's locale(s) (bench/eval_cv.py records; primary
          SentenceResult.prob is unchanged by construction; word mean / min change)
No Azure call is made.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from bench.length.fit_length import CONTRASTS, PER_DIR, VERSION, _fb

CHUNK = 25


def cdir(lang, part) -> Path:
    from bench.data.common import cache_root

    p = cache_root() / "length" / VERSION / lang / f"heldout_{part}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _inp(sc, res):
    from pronunciada import length as L

    c = res._pronunciada  # noqa: SLF001
    return L.frame_inputs(c["ref"], c["ev"], c["lpa"], _fb(sc, c))


def _pack_off(sc, res):
    from bench.eval_languages import _pack

    d = _pack(sc, res)
    d["ref"] = res._pronunciada["ref"]  # noqa: SLF001
    d["inp"] = _inp(sc, res)
    return d


def run_fleurs(lang, sc):
    from bench.audit.contrasts import apply, get_contrast, sites
    from bench.corrupt import NotApplicable, corrupt
    from bench.eval_languages import KINDS, PROS_KIND
    from bench.length.fit_length import _item
    from bench.pipeline.common import records, stable_seed
    from bench.pipeline.evalctl import inventory
    from pronunciada.audio import load_audio

    recs = sorted(records("fleurs", split="test", lang=lang), key=lambda r: r["id"])
    inv = inventory(lang)
    d = cdir(lang, "fleurs")
    for k0 in range(0, len(recs), CHUNK):
        out = d / f"chunk_{len(recs)}_{k0:05d}.pkl"
        if out.exists():
            continue
        rows = []
        for rec in recs[k0:k0 + CHUNK]:
            try:
                wav = load_audio(rec["audio"])
                res = sc.score((wav, 16000), rec["text"])
            except ValueError as e:
                rows.append({"id": rec["id"], "error": str(e)})
                continue
            ref = res._pronunciada["ref"]  # noqa: SLF001
            r = {"id": rec["id"], "dur": rec["dur"], "latency_ms": res.timing_ms.get("total", 0.0),
                 "timing": res.timing_ms, "warnings": res.warnings, "clean": _pack_off(sc, res), "cor": {},
                 "items": [_item(sc, res, "clean", rec["id"])]}
            kinds = KINDS + ((PROS_KIND[lang],) if lang in PROS_KIND else ())
            for kind in kinds:
                try:
                    cor = corrupt(ref, [kind], inv, n_errors=2, seed=stable_seed(rec["id"], kind, "features-v1"),
                                  strict=False)
                except NotApplicable:
                    continue
                if not cor.errors:
                    continue
                r2 = sc.score((wav, 16000), cor.ref)
                p = _pack_off(sc, r2)
                p["phone_wrong"] = np.asarray(cor.phone_wrong, bool)
                p["word_wrong"] = np.asarray(cor.word_wrong, bool)
                r["cor"]["prosody" if kind == PROS_KIND.get(lang) else kind] = p
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
                        r["items"].append(_item(sc, r2, "length", rec["id"], ww, e.flag,
                                                {"contrast": cn, "direction": dr}))
            rows.append(r)
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(rows))
        os.replace(tmp, out)
        print(f"{lang} fleurs-test {min(k0 + CHUNK, len(recs))}/{len(recs)}", flush=True)


def run_swap(lang, sc):
    from bench.data import load
    from bench.eval_swap import sentence_result_output

    recs = [r for r in load("fleurs-swap", split="test") if r["lang"] == lang]
    d = cdir(lang, "swap")
    for k0 in range(0, len(recs), CHUNK * 4):
        out = d / f"chunk_{len(recs)}_{k0:05d}.pkl"
        if out.exists():
            continue
        rows = {}
        for rec in recs[k0:k0 + CHUNK * 4]:
            try:
                res = sc.score(rec["audio"], rec["text"])
            except ValueError as e:
                rows[rec["id"]] = {"off": {"status": f"error: {e}", "words": []}}
                continue
            c = res._pronunciada  # noqa: SLF001
            spans = [tuple(w.char_span) for w in c["ref"].words]
            rows[rec["id"]] = {"off": sentence_result_output(res, spans), "ref": c["ref"], "inp": _inp(sc, res),
                               "wp": [w.prob for w in res.words]}
        out.write_bytes(pickle.dumps(rows))
        print(f"{lang} fleurs-swap {min(k0 + CHUNK * 4, len(recs))}/{len(recs)}", flush=True)


CV_LOCALES = {"ja": ("ja",), "it": ("it",), "de": ("de",)}


def run_cv(lang, sc):
    import bench.eval_cv as EC

    EC.HALF = "test"
    recs = EC.unique_clips(EC.records(CV_LOCALES[lang], half="test"))
    d = cdir(lang, "cv")
    for k0 in range(0, len(recs), CHUNK * 4):
        out = d / f"chunk_{len(recs)}_{k0:05d}.pkl"
        if out.exists():
            continue
        rows = {}
        for rec in recs[k0:k0 + CHUNK * 4]:
            try:
                res = sc.score(rec["audio"], rec["sentence"])
            except ValueError as e:
                rows[rec["id"]] = {"off": {"status": "error", "error": str(e)[:200]}}
                continue
            c = res._pronunciada  # noqa: SLF001
            rows[rec["id"]] = {"off": EC.summarize_result(res), "ref": c["ref"], "inp": _inp(sc, res),
                               "wp": [w.prob for w in res.words]}
        out.write_bytes(pickle.dumps(rows))
        print(f"{lang} cv-real {min(k0 + CHUNK * 4, len(recs))}/{len(recs)}", flush=True)


def run(lang, parts, threads=2):
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(threads)
    import warnings

    warnings.simplefilter("ignore")
    from pronunciada.scorer import Scorer

    sc = Scorer(lang, threads=threads, length=False)
    for p in parts:
        {"fleurs": run_fleurs, "swap": run_swap, "cv": run_cv}[p](lang, sc)


# ----------------------------------------------------------------------------- report


def _load(lang, part):
    out = [] if part == "fleurs" else {}
    for p in sorted(cdir(lang, part).glob("chunk_*.pkl")):
        x = pickle.loads(p.read_bytes())
        if part == "fleurs":
            out += x
        else:
            out.update(x)
    return out


def _factors(par, ref, inp, s_list=(0.0, 0.5, 1.0)):
    """Per word, the channel's factor at each strictness (1 where no sound is judged)."""
    from pronunciada import length as L

    pr = L.score(ref, inp, par)["prob"]
    out = []
    for w in ref.words:
        v = [pr[i] for i in range(*w.phones) if pr[i] is not None]
        out.append([L.word_factor(par, v, s) if v else 1.0 for s in s_list])
    return np.asarray(out, np.float64).reshape(-1, len(s_list)), pr


def _pack_on(par, d):
    import copy

    f, _ = _factors(par, d["ref"], d["inp"])
    e = copy.deepcopy({k: v for k, v in d.items() if k not in ("ref", "inp")})
    for k, s in enumerate((0.0, 0.5, 1.0)):
        e["V"]["on"]["wp"][s] = (e["V"]["on"]["wp"][s] * f[:, k]).astype(np.float32)
        e["V"]["off"]["wp"][s] = (e["V"]["off"]["wp"][s] * f[:, k]).astype(np.float32)
    return e


def report(lang, out_path):
    from bench.eval_languages import eval_lang
    from bench.length.fit_length import sound_eval, word_eval
    from pronunciada import length as L
    from pronunciada.scorer import Scorer

    par = L.params(lang)
    assert par is not None, f"no shipped length channel for {lang}"
    model = par["model"]
    res = {"lang": lang, "data": "FLEURS test / fleurs-swap test / cv-real test (held out)"}
    rows = _load(lang, "fleurs")
    if rows:
        ok = [r for r in rows if "error" not in r]
        items = [it for r in ok for it in r["items"]]
        se = sound_eval(items, model)
        we = word_eval(items, model, par)
        res["length_swaps"] = {"sound": se, "word": we}
        sc = Scorer(lang, threads=2, length=False)
        bias = sc.universal_bias
        before = {r["id"]: {k: r[k] for k in ("dur", "latency_ms", "timing", "warnings")} | {
            "clean": {k: v for k, v in r["clean"].items() if k not in ("ref", "inp")},
            "cor": {k: {kk: vv for kk, vv in v.items() if kk not in ("ref", "inp")} for k, v in r["cor"].items()}}
            for r in ok}
        after = {r["id"]: {k: r[k] for k in ("dur", "latency_ms", "timing", "warnings")} | {
            "clean": _pack_on(par, r["clean"]), "cor": {k: _pack_on(par, v) for k, v in r["cor"].items()}}
            for r in ok}
        eb, ea = eval_lang(lang, before, bias), eval_lang(lang, after, bias)
        keep = lambda e: {"native": e["native"], "sound": {k: e["sound"][k] for k in ("auc", "hit_at_5pct")},  # noqa: E731
                          "word": {k: e["word"][k] for k in ("auc", "hit_at_5pct", "lowest_is_corrupted")},
                          "prosody": e["prosody"]}
        res["ten_language_benchmark"] = {"before": keep(eb), "after": keep(ea)}
    sw = _load(lang, "swap")
    if sw:
        from bench.data import load
        from bench.eval_swap import evaluate

        recs = [r for r in load("fleurs-swap", split="test") if r["lang"] == lang and r["id"] in sw]
        off = {k: v["off"] for k, v in sw.items()}
        on = {}
        for k, v in sw.items():
            if "ref" not in v:
                on[k] = v["off"]
                continue
            f, _ = _factors(par, v["ref"], v["inp"], (0.5,))
            on[k] = {**v["off"], "words": [(a, b, float(p * f[j, 0])) for j, (a, b, p) in enumerate(v["off"]["words"])]}
        m = {}
        for name, o in (("before", off), ("after", on)):
            e = evaluate(recs, o, n_boot=200)[lang]["metrics"]
            m[name] = {k: v for k, v in e.items() if k.startswith(("swap_near", "swap_far", "native"))}
        res["swap_benchmark"] = m
    cv = _load(lang, "cv")
    if cv:
        import bench.eval_cv as EC
        from bench.metrics import roc_auc

        EC.HALF = "test"
        recs = EC.records(CV_LOCALES[lang], half="test")
        m = {}
        for test in sorted({r["test"] for r in recs}):
            rs = [r for r in recs if r["test"] == test and r["id"] in cv and cv[r["id"]]["off"]["status"] == "Success"]
            y = np.array([r["label"] for r in rs])
            if len(set(y)) < 2:
                continue
            ent = {"n": len(rs), "n_pos": int(y.sum())}
            for name in ("before", "after"):
                vals = {"prob": [], "word_mean": [], "word_min": []}
                for r in rs:
                    v = cv[r["id"]]
                    wp = np.asarray(v["wp"], np.float64)
                    if name == "after":
                        f, _ = _factors(par, v["ref"], v["inp"], (0.5,))
                        wp = wp * f[:, 0]
                    vals["prob"].append(v["off"]["prob"])
                    vals["word_mean"].append(wp.mean() if len(wp) else 0.0)
                    vals["word_min"].append(wp.min() if len(wp) else 0.0)
                ent[name] = {k: float(roc_auc(y, -np.asarray(x))) for k, x in vals.items()}
            m[test] = ent
        res["cv_real"] = m
    # runtime parity: the shipped Scorer on 5 test clips vs the offline arithmetic above
    if rows:
        from pronunciada.audio import load_audio
        from bench.pipeline.common import records

        recs = {r["id"]: r for r in records("fleurs", split="test", lang=lang)}
        sc_on = Scorer(lang, threads=2)
        diffs = []
        for r in [r for r in rows if "error" not in r][:5]:
            rec = recs[r["id"]]
            got = [w.prob for w in sc_on.score(load_audio(rec["audio"]), rec["text"]).words]
            f, _ = _factors(par, r["clean"]["ref"], r["clean"]["inp"], (0.5,))
            off = [it for it in r["items"] if it["tag"] == "clean"][0]["wp"][:, 1]
            diffs.append(float(np.max(np.abs(np.asarray(got) - off * f[:, 0]))))
        res["runtime_parity_max_abs_diff"] = max(diffs)
    Path(out_path).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float)[:6000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--lang", required=True)
    ap.add_argument("--parts", default="fleurs,swap,cv")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.lang, a.parts.split(","), a.threads)
    else:
        report(a.lang, a.out or f"results/tables/round15_length_heldout_{a.lang}.json")


if __name__ == "__main__":
    main()
