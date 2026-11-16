"""Coaching diagnoses vs what annotators say real learners said. REPORT ONLY: nothing here is
fitted or selected; the gate comes from bench/coach/fit_gate.py (FLEURS validation).

    python -m bench.coach.eval_learners --steps run,report [--workers 2]

L2-ARCTIC test / EpaDB test (English): annotators give, per canonical phone, correct /
substituted by X / deleted. Our phones are aligned to theirs exactly as bench/eval_english.py
does (bench.pipeline.labels.align_reference). For each matched phone:
  coverage   = annotated substitutions that get a heard_ipa from us
  precision  = among our heard_ipa on matched phones (whatever the label), share equal to what
               the annotator heard (label correct -> the canonical sound; deleted -> nothing),
               exactly / within one articulatory feature
English vowel conventions are folded before comparing (ʌ -> ə, ə˞ -> ɜ˞: the front-end has no
separate stressed ʌ / unstressed ɚ).
OMPAL test (French learners of Taiwan Mandarin, teachers mark tone / consonant / vowel per
character): error-TYPE agreement of our items (tone item -> tone, syllable-initial consonant
-> consonant, any other sound -> vowel/final) with the teachers' marks, learners only.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.coach.gather import out_dir

FOLD = {"ʌ": "ə", "ə˞": "ɜ˞", "ɒ": "ɑ"}
_SC = {}


def _init():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "2"


def _scorer(lang):
    from pronunciada.scorer import Scorer

    if lang not in _SC:
        _SC[lang] = Scorer(lang, threads=2)
    return _SC[lang]


def _one(a):
    ds, rec = a
    lang = "zh" if ds == "ompal" else "en"
    try:
        res = _scorer(lang).score(rec["audio"], rec["text"])
    except ValueError as e:
        return rec["id"], {"error": str(e)}
    ref = res._pronunciada["ref"]  # noqa: SLF001
    out = {}
    for key, r in (("items", res), ("items_s1", _scorer(lang).rescore(res, 1.0))):
        items = r.coaching()
        for it in items:
            it.pop("gate", None)
            if it["phone_index"] is not None:
                w = ref.words[it["word_index"]]
                it["gi"] = w.phones[0] + it["phone_index"]
                it["syl_first"] = ref.syllables[ref.phones[it["gi"]].syllable].phones[0] == it["gi"]
        out[key] = items
    return rec["id"], {**out, "words": [w.text for w in res.words],
                       "n_syl": [len(w.syllables) for w in res.words], "audio": res.audio}


def step_run(workers):
    from concurrent.futures import ProcessPoolExecutor

    from bench.data import load

    for ds in ("l2arctic", "epadb", "ompal"):
        p = out_dir() / f"learners-{ds}.jsonl"
        done = set()
        if p.exists():
            done = {json.loads(l)["id"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
        recs = [r for r in load(ds, split="test") if r["id"] not in done]
        with open(p, "a", encoding="utf-8") as f, ProcessPoolExecutor(min(workers, 2), initializer=_init) as ex:
            for k, (uid, o) in enumerate(ex.map(_one, [(ds, r) for r in recs], chunksize=4)):
                f.write(json.dumps({"id": uid, **o}, ensure_ascii=False, default=float) + "\n")
                if (k + 1) % 200 == 0:
                    print(ds, k + 1, len(recs), flush=True)


def _load(ds):
    p = out_dir() / f"learners-{ds}.jsonl"
    return {(o := json.loads(l))["id"]: o for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def _fold(p):
    return FOLD.get(p, p) if p is not None else None


def within1(t, h):
    from pronunciada import articulatory as A

    if t == h:
        return True
    if not t or not h:
        return False
    try:
        d = A.diff(t, h)
        return len(d) <= 1 and "type" not in d
    except KeyError:
        return False


def english(ds, key="items"):
    from bench.data import load
    from bench.eval_english import _their_entries
    from bench.pipeline.common import build_ref
    from bench.pipeline.labels import align_reference, arpa_to_ours

    O = _load(ds)
    c = defaultdict(int)
    pairs = []                  # (label, truth, expected, heard) for diagnosed matched phones
    for rec in load(ds, split="test"):
        o = O.get(rec["id"])
        if o is None or "error" in o:
            continue
        ref = build_ref("en", rec["text"])
        ents = _their_entries(rec, ds)
        raw = [p for p in rec["labels"]["phones"]
               if (p["error"] != "a" if ds == "l2arctic" else p["ref"] != "0")]
        if ds == "l2arctic":
            words = [w["text"] for w in rec["labels"]["words"]]
            tp = [[e["ph"] for e in ents if e["word"] == j] for j in range(len(words))]
            pos = {}
            for k, e in enumerate(ents):
                pos.setdefault(e["word"], []).append(k)
            o2t, _ = align_reference(ref, words, tp)
            t2o = {pos[j][q]: i for i, (j, q) in o2t.items() if j in pos and q < len(pos[j])}
        else:
            o2t, _ = align_reference(ref, None, [[e["ph"] for e in ents]])
            t2o = {q: i for i, (_, q) in o2t.items()}
        by_phone = {it["gi"]: it for it in o[key] if it.get("gi") is not None}
        for k, e in enumerate(ents):
            if e["label"] == "unk" or k not in t2o:
                continue
            i = t2o[k]
            it = by_phone.get(i)
            exp = ref.phones[i].ipa
            if e["label"] == "sub":
                per = raw[k]["perceived"] if ds == "l2arctic" else raw[k]["annotated"]
                truth = _fold(arpa_to_ours(per))
            elif e["label"] == "del":
                truth = ""
            else:
                truth = _fold(exp)
            c[f"n_{e['label']}"] += 1
            if it is not None:
                c[f"nflag_{e['label']}"] += 1
            if it is not None and it["heard"] is not None:
                c[f"diagnosed_{e['label']}"] += 1
                pairs.append((e["label"], truth, _fold(exp), _fold(it["heard"])))
            if e["label"] == "sub" and truth is None:
                c["sub_unmappable"] += 1
    dg = [p for p in pairs if p[1] is not None]
    sub = [p for p in dg if p[0] == "sub"]
    out = dict(c)
    out.update(
        coverage_sub=c["diagnosed_sub"] / max(c["n_sub"], 1),
        flagged_sub=c["nflag_sub"] / max(c["n_sub"], 1),
        precision_exact=float(np.mean([t == h for _, t, _, h in dg])) if dg else None,
        precision_within1=float(np.mean([within1(t, h) for _, t, _, h in dg])) if dg else None,
        sub_exact=float(np.mean([t == h for _, t, _, h in sub])) if sub else None,
        sub_within1=float(np.mean([within1(t, h) for _, t, _, h in sub])) if sub else None,
        diagnosed_labels={l: sum(p[0] == l for p in dg) for l in ("sub", "del", "correct")})
    # most frequent confusions among diagnosed substitutions: (expected, annotator, ours)
    cnt = defaultdict(int)
    for _, t, e, h in sub:
        cnt[(e, t, h)] += 1
    out["top_sub"] = [[*k, v] for k, v in sorted(cnt.items(), key=lambda x: -x[1])[:12]]
    return out


def ompal(key="items"):
    from bench.data import load
    from bench.eval_ompal import overlap, syllable_char_spans, word_char_spans

    O = _load("ompal")
    rows = []
    for rec in load("ompal", split="test"):
        lab = rec["labels"]
        if lab["native"] or lab["word_level_dropped"]:
            continue
        o = O.get(rec["id"])
        if o is None or "error" in o:
            continue
        spans = word_char_spans(rec["text"], o["words"])
        types = defaultdict(set)          # syllable char span -> our item types
        for it in o[key]:
            sp = spans[it["word_index"]]
            if sp is None:
                continue
            if it["kind"] == "missing_word":
                for s in syllable_char_spans(sp, o["n_syl"][it["word_index"]]):
                    types[s] |= {"missing_word"}
                continue
            s = syllable_char_spans(sp, o["n_syl"][it["word_index"]])[it["syllable_index"]]
            if it["kind"] == "tone":
                types[s].add("tone")
            elif it["kind"] in ("substitution", "missing"):
                cons = it.get("syl_first") and it["expected"] not in ("j", "w", "ɥ") and it["expected"][0] not in "aeiouyəɛɔɑɤɨɯʊœ"
                types[s].add("consonant" if cons else "vowel")
        for u in lab["units"]:
            if u["span"] is None:
                continue
            ours = set()
            for s, t in types.items():
                if overlap(u["span"][0], u["span"][1], s[0], s[1]):
                    ours |= t
            rows.append({"tone": u["tone"] == 0, "consonant": u["consonant"] == 0, "vowel": u["vowel"] == 0,
                         "any": u["any_wrong"] == 1, "ours": sorted(ours)})
    out = {"units": len(rows), "units_any_wrong": sum(r["any"] for r in rows)}
    for t in ("tone", "consonant", "vowel"):
        em = [r for r in rows if t in r["ours"]]
        lab = [r for r in rows if r[t]]
        out[t] = {"ours": len(em), "teacher": len(lab),
                  "precision_type": float(np.mean([r[t] for r in em])) if em else None,
                  "precision_any": float(np.mean([r["any"] for r in em])) if em else None,
                  "recall": float(np.mean([t in r["ours"] for r in lab])) if lab else None}
    both = [r for r in rows if r["any"] and set(r["ours"]) & {"tone", "consonant", "vowel"}]
    out["type_agreement_when_both_flag"] = float(np.mean(
        [bool(set(r["ours"]) & {t for t in ("tone", "consonant", "vowel") if r[t]}) for r in both])) if both else None
    out["n_both"] = len(both)
    return out


def step_report():
    rep = {"l2arctic": english("l2arctic"), "epadb": english("epadb"), "ompal": ompal(),
           "strictness_1": {"l2arctic": english("l2arctic", "items_s1"), "epadb": english("epadb", "items_s1"),
                            "ompal": ompal("items_s1")}}
    Path("results/tables/coaching_learners.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="run,report")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if "run" in a.steps:
        step_run(a.workers)
    if "report" in a.steps:
        step_report()


if __name__ == "__main__":
    main()
