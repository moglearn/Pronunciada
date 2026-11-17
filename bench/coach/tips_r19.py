"""Round 19: pronunciation tips gated on the scorer's own acceptance (pronunciada.tips), test splits.

    PYTHONHASHSEED=0 python -m bench.coach.tips_r19 [--workers 2]

Nothing is fitted or selected. Every utterance is scored once (default strictness 0.5), coached
at "balanced" (res.coaching(), unchanged), and turned into tips twice with max_tips=3:
before = tips(..., include_accepted=True) (no gate), after = tips(...) (the round-19 gate: a
substitution / missing / mispronounced_word tip only when its sound or word has prob < 0.5).
A "pronunciation tip" is a tip made from one of those kinds (feature and contrast claims are
substitution items).

Sets: FLEURS test natives, the first 250 clips per language (every pronunciation tip is a false
alarm); speechocean762 test (expert-marked wrong sound = phone accuracy < 2 on a phone aligned to
ours, as in the Azure yardstick); OMPAL test learners (non-native, word-level labels kept;
teachers' marks: a syllable-initial consonant is wrong when its character's consonant mark is 0,
any other sound when the vowel mark is 0). Learner figures: (a) share of utterances with >= 1
wrong sound that get >= 1 pronunciation tip; (b) share of pronunciation tips that are right (a
sound-level tip on a wrong sound, a word-level tip on a word with a wrong sound); (c) share of
utterances with no wrong sound that get a pronunciation tip.
Writes results/tables/coaching_tips_r19.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")
PER_LANG = 250
VARIANTS = {"before": True, "after": False}          # include_accepted
_SC: dict = {}


def _scorer(lang):
    from pronunciada.scorer import Scorer

    if lang not in _SC:
        _SC.clear()
        _SC[lang] = Scorer(lang, threads=2)
    return _SC[lang]


def _pron_tips(res, items):
    """{variant: [(kind, word index, flat sound index or None)]} of the pronunciation tips."""
    from pronunciada.tips import PRONUNCIATION_KINDS, tips

    off = np.cumsum([0] + [len(w.phones) for w in res.words])
    out = {}
    for vn, inc in VARIANTS.items():
        T = tips(items, 3, res, include_accepted=inc)
        out[vn] = [(t["item"]["kind"], t["item"]["word_index"],
                    None if t["item"]["phone_index"] is None else int(off[t["item"]["word_index"]] + t["item"]["phone_index"]))
                   for t in T if t["item"]["kind"] in PRONUNCIATION_KINDS]
    return out


def _so762_wrong(res, rec):
    from bench.pipeline.labels import align_reference, arpa_to_ours

    W = rec["labels"]["words"]
    o2t, _ = align_reference(res._pronunciada["ref"], [w["text"] for w in W],  # noqa: SLF001
                             [[arpa_to_ours(p) for p in w["phones"]] for w in W])
    return {i for i, (j, k) in o2t.items() if W[j]["phones_accuracy"][k] < 2}


def _ompal_wrong(res, rec):
    from bench.eval_ompal import overlap, syllable_char_spans, word_char_spans
    from pronunciada import coach as Co

    X = Co.sound_inputs(res)
    spans = word_char_spans(rec["text"], [w.text for w in res.words])
    units = [u for u in rec["labels"]["units"] if u["span"] is not None]
    wrong = set()
    for i in range(len(X["expected"])):
        wi, k = int(X["word"][i]), int(X["syl"][i])
        if spans[wi] is None:
            continue
        a, b = syllable_char_spans(spans[wi], len(res.words[wi].syllables))[k]
        key = "consonant" if X["initial"][i] else "vowel"
        if any(u[key] == 0 and overlap(a, b, *u["span"]) for u in units):
            wrong.add(i)
    any_marked = any(u["consonant"] == 0 or u["vowel"] == 0 for u in units)
    return wrong, any_marked


def _job(job):
    kind, lang, recs = job
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "2"
    import soundfile as sf

    sc = _scorer(lang)
    out, t0 = [], time.time()
    for rec in recs:
        try:
            if kind == "fleurs":
                wav, _ = sf.read(rec["audio"], dtype="float32")
                res = sc.score(wav.mean(1) if wav.ndim > 1 else wav, rec["text"])
            else:
                res = sc.score(rec["audio"], rec["text"])
        except ValueError:
            continue
        items = res.coaching(sensitivity="balanced")
        o = {"id": rec["id"], "tips": _pron_tips(res, items)}
        if kind != "fleurs":
            if kind == "so762":
                wrong = _so762_wrong(res, rec)
                marked = bool(wrong)
            else:
                wrong, marked = _ompal_wrong(res, rec)
            off = np.cumsum([0] + [len(w.phones) for w in res.words])
            o["marked"] = marked
            o["wrong_sounds"] = sorted(wrong)
            o["wrong_words"] = sorted({wi for wi in range(len(res.words)) if any(off[wi] <= i < off[wi + 1] for i in wrong)})
        out.append(o)
    return kind, lang, out, time.time() - t0


def jobs():
    from bench.data import load
    from bench.pipeline.common import records

    for lang in LANGS:
        yield "fleurs", lang, records("fleurs", split="test", lang=lang)[:PER_LANG]
    S = load("speechocean762", split="test")
    for a in range(0, len(S), 250):
        yield "so762", "en", S[a:a + 250]
    O = [r for r in load("ompal", split="test") if not r["labels"]["native"] and not r["labels"]["word_level_dropped"]]
    for a in range(0, len(O), 250):
        yield "ompal", "zh", O[a:a + 250]


def native_summary(clips):
    n = max(len(clips), 1)
    return {vn: {"clips": len(clips), "share_clips_with_pron_tip": sum(1 for c in clips if c["tips"][vn]) / n,
                 "pron_tips_per_clip": sum(len(c["tips"][vn]) for c in clips) / n} for vn in VARIANTS}


def learner_summary(utts):
    out = {}
    M = [u for u in utts if u["marked"]]
    C = [u for u in utts if not u["marked"]]
    for vn in VARIANTS:
        right = tot = 0
        for u in utts:
            for kind, wi, i in u["tips"][vn]:
                tot += 1
                right += int(i in u["wrong_sounds"]) if i is not None else int(wi in u["wrong_words"])
        out[vn] = {"utterances": len(utts), "with_marked_sound": len(M),
                   "a_marked_utts_with_pron_tip": sum(1 for u in M if u["tips"][vn]) / max(len(M), 1),
                   "b_pron_tips_right": right / tot if tot else None, "n_pron_tips": tot,
                   "c_clean_utts_with_pron_tip": sum(1 for u in C if u["tips"][vn]) / len(C) if C else None,
                   "n_clean_utts": len(C)}
    out["a_relative_change"] = (out["after"]["a_marked_utts_with_pron_tip"] / out["before"]["a_marked_utts_with_pron_tip"] - 1
                                if out["before"]["a_marked_utts_with_pron_tip"] else None)
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor

    from bench.calib.common import save_json

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    R: dict = {}
    with ProcessPoolExecutor(min(a.workers, 3)) as ex:
        for kind, lang, o, dt in ex.map(_job, jobs()):
            R.setdefault(kind, {}).setdefault(lang, []).extend(o)
            print(f"{kind} {lang}: {len(o)} utterances, {dt:.0f}s", flush=True)
    out = {"setup": "default strictness 0.5, sensitivity 'balanced', max_tips 3; before = gate off "
                    "(include_accepted=True), after = gate on",
           "natives_fleurs_test": {l: native_summary(R["fleurs"][l]) for l in LANGS},
           "speechocean762_test": learner_summary(R["so762"]["en"]),
           "ompal_test": learner_summary(R["ompal"]["zh"])}
    save_json("coaching_tips_r19.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
