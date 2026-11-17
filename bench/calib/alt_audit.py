"""Round 16 A: how often natives use an accepted alternative (FLEURS validation, raw ZIPA posteriors).

    python -m bench.calib.alt_audit [--lang ko,ru,pt,it,de] [--adapted]

For every slot of a targeted class (below), the reference keeps its canonical symbol and gets the
contrast partner as the ONLY alternative, with equal weight (round-8 protocol, bench/calib/
ru_round8.py audit); the Viterbi path through the posteriors picks one. Reported per class and
context: n, share realised as the partner. The shipped references are used as they are for every
other slot.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict

import numpy as np

ASP = {"kʰ": "k", "tʰ": "t", "pʰ": "p", "tʃʰ": "tʃ"}
VOICED = {"p": "b", "t": "d", "k": "g", "f": "v", "s": "z", "ʃ": "ʒ", "ç": "ʝ", "x": "ɣ"}


def classes(lang, ref, i):
    """[(class, context, partner)] for slot i (partner '' = deleted)."""
    p = ref.phones[i]
    alts = {a for a, _ in p.alts}
    w = ref.words[p.word]
    p0, p1 = w.phones
    pos = "initial" if i == p0 else ("final" if i == p1 - 1 else "medial")
    out = []
    if lang == "ko" and p.ipa in ASP:
        out.append(("aspirated->plain", pos, ASP[p.ipa]))
    if lang == "ru" and p.ipa.endswith("ʲ") and not p.is_nucleus and p.ipa[:-1] in alts:
        nxt = ref.phones[i + 1] if i + 1 < p1 else None
        ctx = "final" if nxt is None else ("before_vowel" if nxt.is_nucleus else "before_cons")
        out.append(("soft->hard", ctx, p.ipa[:-1]))
    if lang == "pt" and p.ipa in ("n", "m", "ŋ") and "" in alts and i > p0 and ref.phones[i - 1].is_nucleus:
        out.append(("nasal coda->dropped", pos, ""))
    if lang == "it" and p.is_nucleus and p.ipa in ("e", "ɛ", "o", "ɔ"):
        partner = {"e": "ɛ", "ɛ": "e", "o": "ɔ", "ɔ": "o"}[p.ipa]
        if partner in alts:
            st = ref.syllables[p.syllable].stress
            out.append((f"{p.ipa}->{partner}", "stressed" if st == 1 else "unstressed", partner))
    if lang == "de" and i == p1 - 1 and p.ipa in VOICED and VOICED[p.ipa] in alts:
        out.append(("final devoiced->voiced", "word-final", VOICED[p.ipa]))
    return out


def _utt(args):
    rec, lang, adapted = args
    from bench.pipeline.common import build_ref, load_lp, near_silent
    from pronunciada.adapter import apply
    from pronunciada.align import score_reference

    ref = build_ref(lang, rec["text"])
    lp = load_lp("fleurs-validation", rec)
    if ref is None or lp is None or near_silent(lp, rec["dur"]):
        return []
    if adapted:
        lp = apply(lp, lang)
    r2 = copy.deepcopy(ref)
    sel = {}
    for i in range(len(ref.phones)):
        cl = classes(lang, ref, i)
        if cl:
            sel[i] = cl[0]
            r2.phones[i].alts = [(cl[0][2], 0.0)]
    if not sel:
        return []
    ev = score_reference(lp, r2)
    return [(c, ctx, ev.slots[i].alt_index != 0) for i, (c, ctx, _) in sel.items()]


def run(langs, adapted=False, workers=1):
    from bench.calib.common import save_json
    from bench.pipeline.common import pmap, records

    out = {}
    for lang in langs:
        recs = records("fleurs-validation", lang=lang)
        rows = [x for r in pmap(_utt, [(r, lang, adapted) for r in recs], workers=workers, chunksize=4) for x in r]
        c = defaultdict(Counter)
        for cl, ctx, alt in rows:
            c[(cl, ctx)]["n"] += 1
            c[(cl, ctx)]["alt"] += int(alt)
            c[(cl, "all")]["n"] += 1
            c[(cl, "all")]["alt"] += int(alt)
        out[lang] = {f"{k[0]} | {k[1]}": {"n": v["n"], "partner_share": v["alt"] / max(v["n"], 1)}
                     for k, v in sorted(c.items())}
        for k, v in out[lang].items():
            print(lang, k, v["n"], round(v["partner_share"], 3), flush=True)
    save_json(f"round16_alt_audit{'_adapted' if adapted else ''}.json", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="ko,ru,pt,it,de")
    ap.add_argument("--adapted", action="store_true")
    a = ap.parse_args()
    run(a.lang.split(","), a.adapted)


if __name__ == "__main__":
    main()
