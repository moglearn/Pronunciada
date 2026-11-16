"""Round 8 (Russian only): reduction-aware vowel references. Development tooling, FLEURS-ru
VALIDATION only.

    python -m bench.calib.ru_round8 build-dev            # frozen dev swap set (validation audio)
    python -m bench.calib.ru_round8 audit [--adapted]    # vowel / consonant realisation audit
    python -m bench.calib.ru_round8 eval --tag NAME [--adapter PATH] [--workers 10]
    python -m bench.calib.ru_round8 table --tags a,b,c   # dev before / after table

build-dev: the fleurs-swap recipe (bench/data/fleurs_swap.py, bench/textswap.py) on FLEURS-ru
validation sources (sorted by id, dur <= 20 s, bench.textswap.source_ok), vocabulary = FLEURS-ru
train + validation transcripts + the shipped lexicon; built ONCE with the round-7 front-end and
frozen (calib cache round8_ru/devswap.jsonl), so before / after compare the same texts.

audit: for every vowel nucleus of the references of all validation clips, the candidate set
CANDS replaces its alternatives (equal weights) and the Viterbi path of the acoustic model (raw
ZIPA posteriors; --adapted: through the shipped adapter) picks one; counts per (letter, position
class) -> results/tables/ru_round8_audit.json. Same for t/tʲ/d/dʲ-type consonant slots.

eval: per validation clip, the public Scorer on the correct prompt, on the controlled-error
references of bench.eval_languages (5 sound kinds + stress, 2 errors, same seeds) and on the dev
swap texts; summarised with bench.eval_languages.eval_lang and bench.eval_swap.evaluate.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter, defaultdict

import numpy as np

LANG = "ru"
MAX_DUR = 20.0


def work_dir():
    from bench.calib.common import calib_cache

    d = calib_cache() / "round8_ru"
    d.mkdir(exist_ok=True)
    return d


# ----------------------------------------------------------------------------- dev swap set

def build_dev(n: int = 400):
    from bench import textswap as TS
    from bench.data import load
    from bench.data.fleurs_swap import lexicon_words
    from bench.pipeline.common import records
    from pronunciada.text import get_g2p

    g2p = get_g2p(LANG)
    val = sorted(records("fleurs-validation", lang=LANG), key=lambda r: r["id"])
    texts = [r["text"] for r in load("fleurs", "train", LANG)] + [r["text"] for r in val]
    t0 = time.time()
    vocab = TS.Vocab(LANG, texts, g2p, lexicon_words(LANG))
    print(f"vocab {len(vocab.entries)} entries ({time.time() - t0:.0f}s)", flush=True)
    srcs = [r for r in val if r["dur"] <= MAX_DUR and TS.source_ok(LANG, r["text"], g2p(r["text"]))][:n]
    out, cov = [], Counter()
    for r in srcs:
        vs = TS.make_variants(r["id"], LANG, r["text"], g2p, vocab)
        for v in ("orig", "swap_far", "swap_near"):
            if v in vs:
                text, lab = vs[v]
                cov[v] += 1
                out.append({"id": f"{r['id']}__{v}", "src": r["id"], "audio": r["audio"], "lang": LANG,
                            "text": text, "split": "dev", "dur": r["dur"], "labels": lab})
    p = work_dir() / "devswap.jsonl"
    p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    print("sources", len(srcs), dict(cov), "->", p, flush=True)


def dev_swap_records():
    p = work_dir() / "devswap.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ----------------------------------------------------------------------------- audit

CANDS = ("a", "ɐ", "ə", "ɑ", "ʌ", "o", "u", "ʊ", "i", "ɪ", "ɨ", "ɛ", "e")


def ref_keys(ref, g2p):
    """{phone index: key} with the keys of pronunciada.text.ru (vowel_keys / cons_key) for the
    words whose stress is known (the words the measured targets apply to)."""
    from pronunciada.text import ru as R

    out = {}
    for w in ref.words:
        p0, p1 = w.phones
        norm = g2p.normalise_word(w.text)
        if g2p._src.get(norm) not in R.KNOWN_STRESS:
            continue
        nuc = [i for i in range(p0, p1) if ref.phones[i].is_nucleus]
        s0, s1 = w.syllables
        st = [k for k in range(s1 - s0) if ref.syllables[s0 + k].stress == 1]
        keys = R.vowel_keys(norm, st[0] if st else None)
        if len(keys) == len(nuc) == s1 - s0:
            out.update(dict(zip(nuc, keys)))
        for i in range(p0, p1):
            ph = ref.phones[i]
            if ph.is_nucleus:
                continue
            alts = {a for a, _ in ph.alts}
            o = None
            if ph.ipa == "t" or ph.ipa == "d":
                o = ph.ipa
            elif ph.ipa in ("ts", "tsʲ") and "tʲ" in alts:
                o = "tʲ"
            elif ph.ipa in ("dz", "dzʲ") and "dʲ" in alts:
                o = "dʲ"
            if o:
                nx = ref.phones[i + 1] if i + 1 < p1 else None
                out[i] = R.cons_key(o, nx.ipa if nx else None, bool(nx and nx.is_nucleus))
    return out


CONS_CANDS = {"t": ("t", "tʲ", "ts", "tsʲ"), "d": ("d", "dʲ", "dz", "dzʲ", "z", "zʲ")}


def _audit_utt(args):
    rec, adapted = args
    import copy

    from bench.pipeline.common import _G2P, build_ref, load_lp, near_silent
    from pronunciada.adapter import apply
    from pronunciada.align import score_reference

    ref = build_ref(LANG, rec["text"])
    lp = load_lp("fleurs-validation", rec)
    if ref is None or lp is None or near_silent(lp, rec["dur"]):
        return None
    if adapted:
        lp = apply(lp, LANG)
    keys = ref_keys(ref, _G2P[LANG])
    r2 = copy.deepcopy(ref)
    cand_of = {}
    for i, key in keys.items():
        p = r2.phones[i]
        cs = CANDS if p.is_nucleus else CONS_CANDS[key[0][0]]
        cand_of[i] = [p.ipa] + [x for x in cs if x != p.ipa]
        p.alts = [(x, 0.0) for x in cand_of[i][1:]]
    ev = score_reference(lp, r2)
    rows = []
    for i, key in keys.items():
        a = ev.slots[i].alt_index
        rows.append({"key": key, "ref": ref.phones[i].ipa, "alts": [x for x, _ in ref.phones[i].alts],
                     "heard": cand_of[i][a] if a >= 0 else "",
                     "word": ref.words[ref.phones[i].word].text.lower()})
    return rows


def audit(adapted: bool, workers: int, ship: bool = False):
    from bench.pipeline.common import pmap, records, repo_root
    from pronunciada.text import get_g2p

    g = get_g2p(LANG)
    g.measured_targets = False            # audit on the round-7 references
    recs = records("fleurs-validation", lang=LANG)
    res = pmap(_audit_utt, [(r, adapted) for r in recs], workers=workers, chunksize=4)
    rows = [x for r in res if r for x in r]
    tab = defaultdict(Counter)
    for r in rows:
        tab[tuple(r["key"])][r["heard"]] += 1
    out = []
    for k, c in sorted(tab.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        out.append({"key": list(k), "n": n, "dist": {h: round(v / n, 4) for h, v in c.most_common() if h}})
    tag = "adapted" if adapted else "raw"
    (work_dir() / f"audit_rows_{tag}.pkl").write_bytes(pickle.dumps(rows))
    (repo_root() / "results" / "tables" / f"ru_round8_audit_{tag}.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    if ship and not adapted:
        doc = {"source": "bench/calib/ru_round8.py audit: FLEURS-ru validation (355 clips), raw ZIPA "
                         "posteriors, Viterbi choice among candidate symbols per slot",
               "classes": out}
        (repo_root() / "pronunciada" / "data" / "lex" / "ru_vowels.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=0))
    for o in out[:80]:
        print(o["n"], o["key"], " ".join(f"{h}:{v:.2f}" for h, v in list(o["dist"].items())[:6]))


# ----------------------------------------------------------------------------- dev eval

def _init(adapter_path):
    from bench.calib import api_run

    import os

    api_run._init(LANG, "fp32", 1)
    if os.environ.get("RU8_VOWEL_MIN"):          # dev sweep of pronunciada.text.ru.VOWEL_MIN
        api_run._SC.g2p.vowel_min = float(os.environ["RU8_VOWEL_MIN"])
    if os.environ.get("RU8_BIAS"):               # dev: per-language sound bias override
        api_run._SC.universal_bias = api_run._SC.sound_bias = float(os.environ["RU8_BIAS"])
    # measured targets on unless RU8_NO_TABLE (round-7 references)
    api_run._SC.g2p.measured_targets = not os.environ.get("RU8_NO_TABLE")
    if adapter_path:
        from pronunciada.adapter import Adapter

        api_run._SC.adapter = None if adapter_path == "none" else Adapter.from_file(adapter_path, LANG)


def _eval_clip(args):
    rec, wav, swaps = args
    from bench import eval_languages as EL

    r = dict(rec, audio=(wav, 16000))
    uid, out = EL._utt(r)
    sw = {}
    for s in swaps:
        sid, o = EL._swap_one(dict(s, audio=(wav, 16000)))
        sw[sid] = o
    return uid, out, sw


def run_eval(tag: str, adapter: str | None, workers: int):
    from concurrent.futures import ProcessPoolExecutor

    from bench.pipeline.common import iter_parquet_audio, records

    recs = records("fleurs-validation", lang=LANG)
    by_row = {int(r["audio"].split(":")[2]): r for r in recs}
    path = recs[0]["audio"].split(":")[1]
    swaps = defaultdict(list)
    for s in dev_swap_records():
        swaps[s["src"]].append(s)
    t0 = time.time()
    data, outs = {}, {}
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(adapter,)) as ex:
        futs = [ex.submit(_eval_clip, (by_row[i], wav, swaps.get(by_row[i]["id"], [])))
                for i, wav in iter_parquet_audio(path, set(by_row))]
        for k, f in enumerate(futs):
            uid, out, sw = f.result()
            data[uid] = out
            outs.update(sw)
            if (k + 1) % 50 == 0:
                print(f"{tag}: {k + 1}/{len(futs)} ({time.time() - t0:.0f}s)", flush=True)
    (work_dir() / f"eval_{tag}.pkl").write_bytes(pickle.dumps({"data": data, "swap": outs}))
    s = summarise(tag)
    print(json.dumps(s, indent=1), flush=True)


def summarise(tag: str) -> dict:
    from bench import eval_languages as EL
    from bench.eval_swap import evaluate
    from pronunciada.calibrate import config

    d = pickle.loads((work_dir() / f"eval_{tag}.pkl").read_bytes())
    bias = float(config()["bias"].get(LANG, 0.0))
    r = EL.eval_lang(LANG, d["data"], bias)
    recs = dev_swap_records()
    sw = evaluate(recs, {k: v["on"] for k, v in d["swap"].items()}, n_boot=500)[LANG]["metrics"]
    # share of replaced words scoring > 0.9 (near)
    pos = []
    for rr in recs:
        if rr["labels"]["variant"] != "swap_near":
            continue
        o = d["swap"].get(rr["id"], {}).get("on")
        if not o or o.get("status") != "Success":
            continue
        a, b = rr["labels"]["char_span_in_new_text"]
        ss = [w[2] for w in o["words"] if w[0] < b and a < w[1]]
        if ss:
            pos.append(min(ss))
    pos = np.array(pos)

    def m(k):
        x = sw.get(k, {})
        return [x.get("value"), x.get("lo"), x.get("hi")]

    out = {"tag": tag,
           "near_auc": m("swap_near:auc"), "near_hit5": m("swap_near:hit@5%FA"),
           "near_right_word": m("swap_near:right_word"), "near_n": sw.get("swap_near:n", {}).get("value"),
           "near_gt09": float(np.mean(pos > 0.9)) if len(pos) else None,
           "near_median": float(np.median(pos)) if len(pos) else None,
           "far_auc": m("swap_far:auc"), "far_hit5": m("swap_far:hit@5%FA"), "far_right_word": m("swap_far:right_word"),
           "native_fa60": m("native:fa<60"),
           "native_words_s05": r["native"]["0.5"]["words"], "native_sounds_s05": r["native"]["0.5"]["sounds"],
           "native_sents_s05": r["native"]["0.5"]["sentences"],
           "sound_auc": r["sound"]["auc"], "word_auc": r["word"]["auc"],
           "by_kind": {k: v["auc"] for k, v in r["sound"]["by_kind"].items()},
           "stress_auc": r["prosody"]["prosodic_error"]["product"]["auc"],
           "stress_hit5": r["prosody"]["prosodic_error"]["product"]["hit_at_5pct"],
           "word_hit5": r["word"]["hit_at_5pct"], "lowest_word": r["word"]["lowest_is_corrupted"],
           "utts": r["utts"]}
    (work_dir() / f"summary_{tag}.json").write_text(json.dumps(out, indent=1))
    return out


def _bias_utt(args):
    rec, adapter_path, vmin = args
    from bench.calib import finalize
    from bench.pipeline import common as PC
    from pronunciada import adapter as A

    if vmin is not None:
        g = PC._G2P.get(LANG)
        if g is None:
            from pronunciada.text import get_g2p

            g = PC._G2P[LANG] = get_g2p(LANG)
        g.vowel_min = vmin
        g.measured_targets = True
    if adapter_path:
        A._CACHE[LANG] = A.Adapter.from_file(adapter_path, LANG)
    return finalize._utt((rec, LANG))


def refit_bias(adapter: str | None, vmin: float | None, workers: int, write: bool = False):
    """bench.calib.finalize's rule for Russian only: offset = English median s=0.5 native logit
    (results/tables/calib_bias.json) - Russian median, 400 random FLEURS-train clips (seed 0)."""
    from bench.calib.common import load_json, save_json, ship_dir
    from bench.pipeline.common import pmap, records
    from pronunciada.calibrate import blend_logit

    tab = load_json("calib_bias.json")
    med_en = tab["by_lang"]["en"]["median_logit_s05"]
    recs = records("fleurs", split="train", lang=LANG)
    rng = np.random.default_rng(0)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), min(tab["per_lang"], len(recs)), replace=False))]
    res = [r for r in pmap(_bias_utt, [(r, adapter, vmin) for r in recs], workers=workers, chunksize=4)
           if r is not None]
    zl = np.concatenate([r[0] for r in res])
    zs = np.concatenate([r[1] for r in res])
    med = float(np.median(blend_logit(zl, zs, 0.5)))
    b = med_en - med
    acc = {tag: {str(s): float(np.mean(blend_logit(zl + off, zs + off, s) >= 0)) for s in (0.0, 0.5, 1.0)}
           for tag, off in (("before", 0.0), ("after", b))}
    print(f"ru median {med:.4f} bias {b:+.6f} ({len(res)} utts, {len(zl)} sounds) {acc}", flush=True)
    if write:
        cfgp = ship_dir() / "config.json"
        cfg = json.loads(cfgp.read_text())
        cfg["bias"][LANG] = round(b, 6)
        cfgp.write_text(json.dumps(cfg, indent=1))
        tab["by_lang"][LANG] = {"utts": len(res), "sounds": int(len(zl)), "median_logit_s05": med, "bias": b,
                                "native_acceptance": acc, "refit": "round 8 (bench/calib/ru_round8.py bias)"}
        save_json("calib_bias.json", tab)
    return b


def cv_ru(workers: int = 3):
    """cv-real ru TEST (Test A) with the current tree, without touching bench.eval_cv's shared
    caches / latest.json / cv_real.md: before = the outputs of the last full eval_cv run, after =
    scored now into the round-8 cache; Azure = its cached responses (no requests)."""
    from concurrent.futures import ProcessPoolExecutor

    from bench import eval_cv as E

    recs = E.records(["ru"])
    before, fp = E.ours_scores(recs)
    p = work_dir() / f"cv_ru_{E.fingerprint()}.json"
    if p.exists():
        after = json.loads(p.read_text())
    else:
        clips = E.unique_clips(recs)
        with ProcessPoolExecutor(workers, initializer=E._init, initargs=("ru",)) as ex:
            after = dict(ex.map(E._score_one, clips, chunksize=4))
        p.write_text(json.dumps(after, ensure_ascii=False))
    for o in after.values():
        if o["status"] != "Success":
            o.update({k: 0.0 for k in (E.OURS_PRIMARY,) + E.OURS_SECONDARY})
    az = E.azure_scores(recs)
    out = {"before_fingerprint": fp, "before": E.evaluate(recs, before, az), "after": E.evaluate(recs, after, az)}
    (work_dir() / "cv_ru_result.json").write_text(json.dumps(out, indent=1, default=float))
    for tag in ("before", "after"):
        for k, e in out[tag].items():
            if "primary" in e:
                pr = e["primary"]
                print(tag, k, "n", e["n_scored_both"], "ours %.3f %s azure %.3f %s delta %+.3f %s p %.3f" % (
                    pr["ours"], [round(x, 3) for x in pr["ours_ci"]], pr["azure"], [round(x, 3) for x in pr["azure_ci"]],
                    pr["delta"], [round(x, 3) for x in pr["delta_ci"]], pr["p_two_sided"]))


def table(tags):
    rows = [json.loads((work_dir() / f"summary_{t}.json").read_text()) for t in tags]
    keys = ["near_n", "near_auc", "near_gt09", "near_median", "near_hit5", "near_right_word", "far_auc",
            "far_hit5", "native_words_s05", "native_sounds_s05", "native_fa60", "sound_auc", "word_auc",
            "stress_auc"]
    print("| metric | " + " | ".join(tags) + " |")
    print("|---|" + "---|" * len(tags))
    for k in keys + ["by_kind"]:
        if k == "by_kind":
            for kk in rows[0]["by_kind"]:
                print(f"| sound AUC {kk} | " + " | ".join(f"{r['by_kind'].get(kk, float('nan')):.4f}" for r in rows) + " |")
            continue
        vals = []
        for r in rows:
            v = r[k]
            if isinstance(v, list):
                vals.append(f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" if v[1] is not None else f"{v[0]:.3f}")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        print(f"| {k} | " + " | ".join(vals) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build-dev", "audit", "eval", "summary", "table", "bias", "cv"])
    ap.add_argument("--vowel-min", type=float, default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--tags", default="base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--adapted", action="store_true")
    ap.add_argument("--ship", action="store_true", help="audit: write pronunciada/data/lex/ru_vowels.json")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.step == "build-dev":
        build_dev()
    elif a.step == "audit":
        audit(a.adapted, a.workers, a.ship)
    elif a.step == "eval":
        run_eval(a.tag, a.adapter, a.workers)
    elif a.step == "summary":
        print(json.dumps(summarise(a.tag), indent=1))
    elif a.step == "cv":
        cv_ru(min(a.workers, 3))
    elif a.step == "bias":
        refit_bias(a.adapter, a.vowel_min, a.workers, a.write)
    else:
        table(a.tags.split(","))


if __name__ == "__main__":
    main()
