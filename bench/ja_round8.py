"""Round 8 (Japanese): error analysis on the cv-real DEV half (never the test half).

    python -m bench.ja_round8 dump [--tag base]     # score ja dev clips, keep per-word evidence
    python -m bench.ja_round8 analyse [--tag base]  # cause breakdown (printed + results/tables/)

Per clip it keeps the sentence probability, every word's probabilities, its reference phones,
what was heard over each phone (greedy decode), the alternative chosen, and the whole greedy
decode of the utterance, so the analysis needs no listening and no re-scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench import cv_build as CB

_SC = None


def out_path(tag: str):
    d = CB.cache_root() / "cv-real-dev" / "ja_round8"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}.pkl"


def _init(kw: dict) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    global _SC
    import pronunciada

    _SC = pronunciada.Scorer("ja", threads=1, **kw)


def _one(rec: dict) -> dict:
    res = _SC.score(rec["audio"], rec["sentence"])
    c = res._pronunciada  # noqa: SLF001
    ref, ev = c["ref"], c["ev"]
    words = []
    for wi, (w, rw) in enumerate(zip(res.words, ref.words)):
        a, b = rw.phones
        words.append({
            "text": w.text, "prob": float(w.prob), "sounds": float(w.sounds_prob),
            "present": None if w.present_prob is None else float(w.present_prob),
            "kana": "".join(ref.syllables[s].text for s in range(*rw.syllables)),
            "ipa": [ref.phones[i].ipa for i in range(a, b)],
            "pp": [float(p.prob) for p in w.phones],
            "heard": [ev.slots[i].heard for i in range(a, b)],
            "alt": [int(ev.slots[i].alt_index) for i in range(a, b)],
            "alts": [[x for x, _ in ref.phones[i].alts] for i in range(a, b)],
            "nfr": [int(ev.slots[i].n_frames) for i in range(a, b)],
            "llr_del": [float(ev.slots[i].llr_del) for i in range(a, b)],
            "llr_sub": [float(ev.slots[i].llr_sub) for i in range(a, b)],
            "peak": [float(ev.slots[i].peak_post) for i in range(a, b)],
            "zl": [float(c["s_zl"][i]) for i in range(a, b)], "zs": [float(c["s_zs"][i]) for i in range(a, b)],
            "start": w.start, "end": w.end})
    dv = c.get("delivery") or {}
    return {"id": rec["id"], "label": rec["label"], "sentence": rec["sentence"], "dur": rec["dur"],
            "client_id": rec["client_id"], "prob": float(res.prob),
            "completeness": res.completeness, "scores": dict(res.scores),
            "greedy": "".join(t for t, _, _ in ev.free_tokens),
            "greedy_frames": [(t, int(s), int(e)) for t, s, e in ev.free_tokens],
            "ref": "".join(p.ipa for p in ref.phones), "words": words,
            "delivery": {k: float(v) for k, v in dv.items() if isinstance(v, (int, float))},
            "n_frames": int(ev.n_frames), "warnings": list(res.warnings),
            "sent_zl": float(c["sent_zl"]), "sent_zs": float(c["sent_zs"]),
            "w_zl": [float(x) for x in c["w_zl"]], "w_zs": [float(x) for x in c["w_zs"]],
            "S": {k: float(np.asarray(v).reshape(-1)[0]) for k, v in c["S"].items()
                  if np.asarray(v).size == 1 and np.issubdtype(np.asarray(v).dtype, np.number)}}


def dump(tag: str, workers: int = 3, kw: dict | None = None, half: str = "dev") -> list[dict]:
    assert half == "dev", "round 8 analysis runs on the dev half only"
    recs = CB.load_manifest(["ja"], half="dev")
    assert recs and all(r["half"] == "dev" for r in recs)
    t0 = time.time()
    with ProcessPoolExecutor(min(workers, 3), initializer=_init, initargs=(kw or {},)) as ex:
        out = list(ex.map(_one, recs, chunksize=4))
    pickle.dump(out, open(out_path(tag), "wb"))
    print(f"dumped {len(out)} dev clips to {out_path(tag).name} ({time.time() - t0:.0f} s)")
    return out


_SUD = None


def alt_prompts(text: str, limit: int = 12) -> list[tuple[str, str, str]]:
    """Hypothesis 1 probe: (prompt with one written word replaced by another dictionary reading
    of the same written form, word, reading). Readings: SudachiDict lookup of the analyser's word
    string; only words with a kanji; the analyser's own reading excluded."""
    global _SUD
    import pyopenjtalk
    from sudachipy import Dictionary

    from pronunciada.text._cjk_common import is_han

    if _SUD is None:
        _SUD = Dictionary(dict="core")
    out, pos = [], 0
    for x in pyopenjtalk.run_frontend(text):
        w = x["string"]
        j = text.find(w, pos)
        if j < 0:
            continue
        pos = j + len(w)
        if not any(is_han(c) for c in w):
            continue
        own = x["pron"].replace("’", "")
        seen = {own, x.get("read", "")}
        for m in _SUD.lookup(w):
            rf = m.reading_form()
            if not rf or rf in seen or len(rf) < 2 or not all("ァ" <= c <= "ヺ" or c == "ー" for c in rf):
                continue
            seen.add(rf)
            out.append((text[:j] + rf + text[j + len(w):], w, rf))
    return out[:limit]


def _alt_one(rec: dict) -> dict:
    base = _SC.score(rec["audio"], rec["sentence"])
    alts = []
    for t, w, rf in alt_prompts(rec["sentence"]):
        try:
            r = _SC.score(rec["audio"], t)
        except ValueError:
            continue
        alts.append({"word": w, "reading": rf, "prob": float(r.prob),
                     "word_probs": [(x.text, float(x.prob)) for x in r.words]})
    return {"id": rec["id"], "label": rec["label"], "client_id": rec["client_id"], "prob": float(base.prob),
            "word_probs": [(x.text, float(x.prob)) for x in base.words], "alts": alts}


def altread(tag: str = "altread", workers: int = 3) -> list[dict]:
    recs = CB.load_manifest(["ja"], half="dev")
    assert recs and all(r["half"] == "dev" for r in recs)
    with ProcessPoolExecutor(min(workers, 3), initializer=_init, initargs=({},)) as ex:
        out = list(ex.map(_alt_one, recs, chunksize=2))
    pickle.dump(out, open(out_path(tag), "wb"))
    best = [dict(r, prob=max([r["prob"]] + [a["prob"] for a in r["alts"]])) for r in out]
    print(f"alternative-reading probe: {sum(len(r['alts']) for r in out)} alternative prompts over "
          f"{sum(1 for r in out if r['alts'])} clips; AUC base {auc(out):.4f}, best-of {auc(best):.4f}")
    return out


# ----------------------------------------------------------------------------- analysis

def _ed(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[:], i
        for j, cb in enumerate(b, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (ca != cb))
    return d[-1]


def clip_per(r: dict) -> float:
    """Normalised edit distance greedy decode vs canonical reference (length marks ignored)."""
    g, f = r["greedy"].replace("ː", ""), r["ref"].replace("ː", "")
    return _ed(g, f) / max(1, len(f))


def _heard_word(r: dict, w: dict) -> str:
    s, e = w["start"] / 0.02, w["end"] / 0.02
    return "".join(t for t, a, b in r["greedy_frames"] if b > s - 1 and a < e + 1)


_G2P = None


def _reading_phones(kana: str) -> str:
    global _G2P
    if _G2P is None:
        from pronunciada.text import get_g2p

        _G2P = get_g2p("ja")
    return "".join(p.ipa for p in _G2P(kana).phones)


def word_cause(r: dict, w: dict) -> str:
    """Cause of a low (< 0.5) word, first matching rule (see results/round8_ja.md)."""
    import re

    from sudachipy import Dictionary

    from pronunciada.text._cjk_common import is_han

    global _SUD
    if r["per"] > 0.6:
        return "clip: decode does not match the prompt (PER > 0.6)"
    if w["present"] is not None and w["present"] < 0.5:
        return "word not heard (present < 0.5)"
    h = _heard_word(r, w).replace("ː", "")
    can = "".join(w["ipa"]).replace("ː", "")
    if any(is_han(c) for c in w["text"]):
        if _SUD is None:
            _SUD = Dictionary(dict="core")
        for m in _SUD.lookup(w["text"]):
            rf = m.reading_form()
            if not rf or not all("ァ" <= c <= "ヺ" or c == "ー" for c in rf):
                continue
            try:
                alt = _reading_phones(rf).replace("ː", "")
            except Exception:  # noqa: BLE001
                continue
            if alt != can and _ed(h, alt) + 1 <= _ed(h, can):
                return "another dictionary reading fits better"
    low = [i for i, p in enumerate(w["pp"]) if p < 0.5]
    if any("ː" in w["ipa"][i] for i in low):
        return "long vowel"
    if any(w["ipa"][i] in ("i", "u") and w["heard"][i] == "" for i in low):
        return "i/u devoiced or dropped"
    if any(w["ipa"][i][:2] in ("kk", "tt", "pp", "ss", "ʃʃ") for i in low):
        return "geminate"
    if w["text"] in ("は", "へ", "を"):
        return "particle は/へ/を"
    if "ン" in w["kana"] and any(w["ipa"][i] in ("ŋ", "n", "m") for i in low):
        return "moraic nasal"
    if re.fullmatch(r"[\u30a0-\u30ff]+", w["text"]):
        return "katakana loanword"
    return "other substitution"


def analyse(tag: str) -> dict:
    rows = pickle.load(open(out_path(tag), "rb"))
    for r in rows:
        r["per"] = clip_per(r)
    out = {"tag": tag, "auc": auc(rows), "words": {}, "examples": {}, "clips": {}}
    for lab, name in ((0, "accepted"), (1, "rejected")):
        cnt, ex = Counter(), {}
        for r in rows:
            if r["label"] != lab:
                continue
            for w in r["words"]:
                if w["prob"] < 0.5:
                    c = word_cause(r, w)
                    cnt[c] += 1
                    ex.setdefault(c, [])
                    if len(ex[c]) < 5:
                        ex[c].append(f"{w['text']} [{w['kana']}] heard {_heard_word(r, w) or '-'}")
        out["words"][name] = dict(cnt.most_common())
        out["examples"][name] = ex
        side = [r for r in rows if r["label"] == lab]
        bins = [(0, .2), (.2, .4), (.4, .6), (.6, 10.0)]
        out["clips"][name] = {
            "n": len(side), "with_low_word": sum(min(w["prob"] for w in r["words"]) < 0.5 for r in side),
            "prob_lt_0.5": sum(r["prob"] < 0.5 for r in side), "prob_ge_0.95": sum(r["prob"] >= 0.95 for r in side),
            "empty_decode(<=2 tokens)": sum(len(r["greedy_frames"]) <= 2 for r in side),
            "empty_decode_prob_ge_0.5": sum(len(r["greedy_frames"]) <= 2 and r["prob"] >= 0.5 for r in side),
            "too_long_warning": sum(any("too long" in x for x in r["warnings"]) for r in side),
            "per_bins": {f"[{a},{b})": sum(a <= r["per"] < b for r in side) for a, b in bins},
            "prob_ge_0.95_by_per": {f"[{a},{b})": sum(a <= r["per"] < b and r["prob"] >= 0.95 for r in side)
                                   for a, b in bins}}
    return out


def auc(rows: list[dict], key: str = "prob") -> float:
    from bench.metrics import roc_auc

    y = np.array([r["label"] for r in rows])
    return float(roc_auc(y, -np.array([r[key] for r in rows])))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("dump", "auc", "altread", "analyse"))
    ap.add_argument("--tag", default="base")
    ap.add_argument("--calib-dir", help="experiment: Scorer(calib_dir=...)")
    a = ap.parse_args(argv)
    if a.step == "analyse":
        res = analyse(a.tag)
        from pathlib import Path

        p = Path(__file__).resolve().parents[1] / "results" / "tables" / f"ja_round8_{a.tag}.json"
        p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return
    if a.step == "altread":
        altread()
        return
    if a.step == "dump":
        rows = dump(a.tag, kw={"calib_dir": a.calib_dir} if a.calib_dir else None)
    else:
        rows = pickle.load(open(out_path(a.tag), "rb"))
    print(f"ja dev AUC ({a.tag}): {auc(rows):.4f}  n={len(rows)} {Counter(r['label'] for r in rows)}")


if __name__ == "__main__":
    main()
