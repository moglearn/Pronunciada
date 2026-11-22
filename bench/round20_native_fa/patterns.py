"""Fixed pattern definitions, the same for every language, counted on the false alarms and on all correct words
(reads tables/rows_<lang>.jsonl written by taxonomy.py). Patterns, not word lists.

    python -m bench.round20_native_fa.patterns --lang en,de,ko,es,fr

For each pattern: n false alarms, share of all false alarms, the false-alarm rate among correct words with the
pattern, the rate among correct words without it (ratio = enrichment), and the most frequent lowest-sound pairs
(expected > heard) inside the pattern.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from bench.round20_native_fa.rebuild import ROOT

PATTERNS = {
    "within two words of an inserted (unsaid) word": lambda r: r["neighbour"] and r["kind"] == "word_omission",
    "within two words of a substituted / indel word": lambda r: r["neighbour"] and r["kind"] != "word_omission",
    "punctuation-only token": lambda r: bool(r.get("punct")),
    "number, abbreviation or multi-word expansion": lambda r: r["number"] and not r.get("punct"),
    "unknown word (letter rules)": lambda r: r["oov"] and not r["number"],
    "name (capitalised mid-sentence, in lexicon)": lambda r: r["name"] and not r["oov"] and not r["number"],
    "stress factor drives, sounds fine": lambda r: r["driver"] == "stress",
    "identity step drives": lambda r: r["driver"] == "identity",
    "function word judged absent (presence factor)": lambda r: r["function"] and r["driver"] == "present",
    "content word judged absent (presence factor)": lambda r: not r["function"] and r["driver"] == "present",
    "content word, a sound unsupported (deletion-type)": lambda r: not r["function"] and r["driver"] == "sounds"
    and r["low_kind"] == "deletion",
    "content word, another sound heard (substitution-type)": lambda r: not r["function"] and r["driver"] == "sounds"
    and r["low_kind"] == "substitution",
    "function word, sound evidence": lambda r: r["function"] and r["driver"] == "sounds",
    "last word of the clip": lambda r: r["last"],
    "first word of the clip": lambda r: r["first"],
    "long word (>= 8 sounds)": lambda r: r["n_snd"] >= 8,
    "lattice used a front-end alternative": lambda r: bool(r.get("has_alt")),
    "lexicon has another reading": lambda r: bool(r.get("has_lex")),
    "Whisper transcript differs at the word": lambda r: bool(r.get("asr_mismatch")),
}


FAMILIES = {   # candidate shared causes (unions of the patterns above), overlapping
    "presence / unsaid-word handling": lambda r: r["driver"] == "present"
    or (r["neighbour"] and r["kind"] == "word_omission"),
    "text front-end (number, unknown word, name, punctuation)": lambda r: bool(r["number"] or r["oov"] or r["name"]
                                                                                  or r.get("punct")),
    "stress factor": lambda r: r["driver"] == "stress",
    "identity step": lambda r: r["driver"] == "identity",
}


def families(lang: str) -> dict:
    from bench.round20_native_fa.taxonomy import share_ci

    rows = [json.loads(l) for l in (ROOT / "tables" / f"rows_{lang}.jsonl").read_text(encoding="utf-8").splitlines()]
    thr = 1 - json.loads((ROOT / "tables" / f"taxonomy_{lang}.json").read_text())["thr_score"]
    fa = [r for r in rows if not r["label"] and r["err"] >= thr]
    out = {}
    for name, f in FAMILIES.items():
        out[name] = {"n": sum(bool(f(r)) for r in fa), "pct": 100 * sum(bool(f(r)) for r in fa) / max(1, len(fa)),
                     "ci": [100 * x for x in share_ci(fa, f)]}
    a, b = FAMILIES["presence / unsaid-word handling"], FAMILIES["text front-end (number, unknown word, name, punctuation)"]
    u = sum(bool(a(r) or b(r)) for r in fa)
    out["union of the first two"] = {"n": u, "pct": 100 * u / max(1, len(fa)),
                                     "ci": [100 * x for x in share_ci(fa, lambda r: bool(a(r) or b(r)))]}
    return out


def run(lang: str) -> dict:
    rows = [json.loads(l) for l in (ROOT / "tables" / f"rows_{lang}.jsonl").read_text(encoding="utf-8").splitlines()]
    tax = json.loads((ROOT / "tables" / f"taxonomy_{lang}.json").read_text())
    thr = 1 - tax["thr_score"]
    neg = [r for r in rows if not r["label"]]
    fa = [r for r in neg if r["err"] >= thr]
    out = {}
    for name, f in PATTERNS.items():
        inn = [r for r in neg if f(r)]
        outn = [r for r in neg if not f(r)]
        n_fa = sum(r["err"] >= thr for r in inn)
        rate_in = 100 * n_fa / max(1, len(inn))
        rate_out = 100 * sum(r["err"] >= thr for r in outn) / max(1, len(outn))
        pairs = Counter(r["low_snd"] for r in inn if r["err"] >= thr and r["low_snd"]).most_common(5)
        out[name] = {"n_fa": n_fa, "share_fa": 100 * n_fa / max(1, len(fa)), "n_words": len(inn),
                     "fa_rate": rate_in, "fa_rate_rest": rate_out,
                     "ratio": (rate_in / rate_out) if rate_out > 0 else None, "low_pairs": pairs}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    a = ap.parse_args(argv)
    from bench.round20_native_fa import core as K

    for lang in a.lang.split(","):
        res = run(lang)
        fam = families(lang)
        K.save_json(f"patterns_{lang}.json", {"patterns": res, "families": fam})
        print(f"== {lang}")
        for name, v in fam.items():
            print(f"  family {name}: {v['n']} = {v['pct']:.1f} % [{v['ci'][0]:.1f}, {v['ci'][1]:.1f}]")
        for name, v in sorted(res.items(), key=lambda kv: -kv[1]["n_fa"]):
            print(f"  {v['n_fa']:4d} ({v['share_fa']:5.1f} % of FA)  rate {v['fa_rate']:5.1f} % vs {v['fa_rate_rest']:4.1f} %"
                  f"  n {v['n_words']:5d}  {name}  {v['low_pairs'][:3]}")


if __name__ == "__main__":
    main()
