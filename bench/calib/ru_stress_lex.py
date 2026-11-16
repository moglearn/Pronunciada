"""Round 10 (Russian): offline stress lexicon for word forms the shipped stress lexicon lacks.

    python -m bench.calib.ru_stress_lex build      # writes pronunciada/data/lex/ru_stress_extra.tsv.gz
    python -m bench.calib.ru_stress_lex agree      # silero vs our lexicon on words both know

OFFLINE / BENCH ONLY: uses silero-stress (MIT, torch) to put stress marks (and restore ё) on the
sentences of FLEURS-ru train + validation and the Common Voice ru sentences of the cv-real
metadata on disk (dev / test / invalidated tsv). FLEURS-ru TEST text is not used.
Per word form missing from pronunciada/data/lex/ru.tsv.gz (and not an unstressed clitic, not a
one-vowel word, no ё in the prompt): votes over its occurrences in context for (stressed vowel
index, spelling with ё restored). Kept: the majority variant; a second variant is kept as a
homograph alternative when it has >= 25 % of the votes and >= 2 occurrences.
File format: word \\t idx[:respelling][|idx[:respelling]]  (respelling only when ё was restored).
The runtime (pronunciada/text/ru.py) only reads the file; torch is not needed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

VOWELS = "аеёиоуыэюя"
WORD = re.compile(r"[а-яё+]+(?:-[а-яё+]+)*")
MIN_SHARE, MIN_COUNT = 0.25, 2


def sentences() -> dict[str, list[str]]:
    from bench.data import load
    from bench.data.common import cache_root
    from bench.pipeline.common import records

    out = {"fleurs_train": [r["text"] for r in load("fleurs", "train", "ru")],
           "fleurs_validation": [r["text"] for r in records("fleurs-validation", lang="ru")]}
    cv = []
    d = cache_root() / "data" / "cv-real" / "meta" / "ru"
    csv.field_size_limit(1 << 20)
    for n in ("dev.tsv", "test.tsv", "invalidated.tsv"):
        p = d / n
        if p.exists():
            with p.open(encoding="utf-8") as f:
                cv += [row["sentence"] for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
    out["common_voice"] = sorted(set(cv))
    return out


def _parse(acc_text: str, plain: str):
    """[(word as in plain, stressed vowel index or None, ё-respelling or None)] aligned by position."""
    a = WORD.findall(acc_text.lower())
    p = WORD.findall(plain.lower())
    if len(a) != len(p):
        return []
    out = []
    for wa, wp in zip(a, p):
        s = wa.replace("+", "")
        if s.replace("ё", "е") != wp.replace("ё", "е"):
            return []
        idx, nv = None, 0
        for k, ch in enumerate(wa):
            if ch == "+":
                if k + 1 < len(wa) and wa[k + 1] in VOWELS and (idx is None or "-" in wa):
                    idx = nv                     # hyphenated: the last part's stress is primary
            elif ch in VOWELS:
                nv += 1
        out.append((wp, idx, s if (s != wp and "ё" in s) else None))
    return out


def run_accentor(texts: list[str], batch: int = 64):
    from silero_stress import load_accentor

    acc = load_accentor()
    for t in texts:
        try:
            yield t, acc(t)
        except Exception:  # noqa: BLE001
            continue


def collect():
    from bench.calib.common import calib_cache

    cache = calib_cache() / "round8_ru" / "silero_votes.json"
    if cache.exists():
        return json.loads(cache.read_text())
    srcs = sentences()
    votes = defaultdict(Counter)
    n = 0
    for name, texts in srcs.items():
        for t, a in run_accentor(texts):
            for w, idx, resp in _parse(a, t):
                if idx is not None:
                    votes[w][f"{idx}:{resp or ''}"] += 1
            n += 1
    out = {"n_sentences": {k: len(v) for k, v in srcs.items()}, "votes": {w: dict(c) for w, c in votes.items()}}
    cache.write_text(json.dumps(out, ensure_ascii=False))
    return out


def _choose(c: dict[str, int]) -> list[str]:
    tot = sum(c.values())
    items = sorted(c.items(), key=lambda kv: -kv[1])
    keep = [items[0][0]]
    if len(items) > 1 and items[1][1] >= MIN_SHARE * tot and items[1][1] >= MIN_COUNT:
        keep.append(items[1][0])
    return keep


def agree():
    """Agreement of silero's majority stress with the shipped lexicon (digit stress entries,
    words with >= 2 vowels, not clitics), by type and by token."""
    from pronunciada.text import ru as R
    from pronunciada.text import get_g2p

    g = get_g2p("ru")
    V = collect()["votes"]
    t_ok = t_n = k_ok = k_n = 0
    for w, c in V.items():
        v = g.stress.get(w)
        if v is None or not v.isdigit() or w in R.CLITICS or sum(ch in VOWELS for ch in w) < 2:
            continue
        best = int(_choose(c)[0].split(":")[0])
        t_n += 1
        t_ok += best == int(v)
        for key, n in c.items():
            k_n += n
            k_ok += n * (int(key.split(":")[0]) == int(v))
    res = {"types": t_n, "type_agreement": t_ok / max(t_n, 1), "tokens": k_n, "token_agreement": k_ok / max(k_n, 1)}
    # stem-predicted stress (our fallback) vs silero, on words our lexicon lacks
    s_ok = s_n = 0
    for w, c in V.items():
        if w in g.stress or w in R.CLITICS or "ё" in w or sum(ch in VOWELS for ch in w) < 2:
            continue
        k = g._stem_stress(w, sum(ch in VOWELS for ch in w))
        if k is None:
            k = g.stress_model.predict(w, sum(ch in VOWELS for ch in w)) if g.stress_model else None
        if k is None:
            continue
        s_n += 1
        s_ok += k == int(_choose(c)[0].split(":")[0])
    res.update({"our_fallback_vs_silero_types": s_n, "our_fallback_agreement": s_ok / max(s_n, 1)})
    print(json.dumps(res, indent=1))
    return res


def build():
    from bench.pipeline.common import repo_root
    from pronunciada.text import ru as R
    from pronunciada.text import get_g2p

    g = get_g2p("ru")
    V = collect()
    rows, homo = [], 0
    for w, c in sorted(V["votes"].items()):
        if w in g.stress or w in R.CLITICS or "ё" in w or sum(ch in VOWELS for ch in w) < 2:
            continue
        keep = _choose(c)
        homo += len(keep) > 1
        enc = "|".join(k if not k.endswith(":") else k[:-1] for k in keep)
        rows.append(f"{w}\t{enc}")
    p = repo_root() / "pronunciada" / "data" / "lex" / "ru_stress_extra.tsv.gz"
    with gzip.open(p, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write("# silero-stress 1.5 (MIT), applied offline in context to FLEURS-ru train+validation and "
                "Common Voice ru (cv-real metadata) sentences; bench/calib/ru_stress_lex.py\n")
        f.write("\n".join(rows) + "\n")
    print(f"{len(rows)} word forms ({homo} with two stress variants), {p.stat().st_size / 1024:.0f} KB "
          f"sentences {V['n_sentences']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "agree"])
    a = ap.parse_args()
    build() if a.step == "build" else agree()


if __name__ == "__main__":
    main()
