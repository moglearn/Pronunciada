"""fleurs-swap: text-level planted errors on native FLEURS test audio (see bench/textswap.py).

    python -m bench.data.fleurs_swap [--langs en zh ...] [--n 150]

Source: bench.data.load("fleurs", "test", lang), sorted by id; the first N (150) utterances with
dur <= 20 s whose front-end warnings are only OOV notes (zh/ja/ko: no digits / Latin letters).
Per source utterance: `orig` + swap_far + swap_near (+ tone for zh) when possible.

Writes ~/.cache/pronunciada/data/fleurs-swap/manifest.jsonl (dataset "fleurs-swap", split "test",
id "<orig id>__<variant>", audio = the FLEURS file) and coverage.json. Deterministic: re-running
rewrites identical files. No audio is copied.

labels: {variant, src_id, word_index, char_span_in_new_text, orig_word, new_word,
         phone_edit_distance, n_phones, words: [{text, span}] (every evaluation unit of the new
         text), source ("fleurs"|"lexicon"|"jamo")}; zh adds orig_char, new_char,
         unit_span_in_new_text, orig/new_syllable, orig/new_tone (char_span_in_new_text is the
         replaced hanzi; the evaluation unit is the RefWord containing it).
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from bench import textswap as TS
from bench.data import load
from bench.data.common import data_dir, make_record, write_manifest

NAME = "fleurs-swap"
MAX_DUR = 20.0
LEX_DIR = Path(__file__).resolve().parents[2] / "pronunciada" / "data" / "lex"


def lexicon_words(lang: str) -> list[str]:
    fn = TS.LEXICON.get(lang)
    if not fn:
        return []
    out = []
    with gzip.open(LEX_DIR / fn, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            w = line.split("\t", 1)[0].strip()
            if w:
                out.append(w)
    return sorted(set(out))


def sources(lang: str, g2p, n: int) -> list[dict]:
    recs = sorted(load("fleurs", "test", lang), key=lambda r: r["id"])
    out = []
    for r in recs:
        if r["dur"] > MAX_DUR:
            continue
        if not TS.source_ok(lang, r["text"], g2p(r["text"])):
            continue
        out.append(r)
        if len(out) >= n:
            break
    return out


def build_lang(lang: str, n: int) -> tuple[list[dict], dict]:
    from pronunciada.text import get_g2p

    g2p = get_g2p(lang)
    texts = [r["text"] for r in load("fleurs", lang=lang)]          # train + test transcripts
    vocab = TS.Vocab(lang, texts, g2p, lexicon_words(lang) if lang not in ("zh", "ja", "ko") else None)
    srcs = sources(lang, g2p, n)
    recs, cov = [], {"sources": len(srcs), "vocab_entries": len(vocab.entries),
                     "vocab_hanzi": len(vocab.hanzi)}
    variants = ("orig", "swap_far", "swap_near") + (("tone",) if lang == "zh" else ())
    for v in variants:
        cov[v] = 0
    near_src: dict[str, int] = {}
    for r in srcs:
        vs = TS.make_variants(r["id"], lang, r["text"], g2p, vocab)
        for v in variants:
            if v not in vs:
                continue
            text, lab = vs[v]
            cov[v] += 1
            if v == "swap_near":
                near_src[lab.get("source", "?")] = near_src.get(lab.get("source", "?"), 0) + 1
            recs.append(make_record(f"{r['id']}__{v}", Path(r["audio"]), lang, text, "test", None,
                                    r["dur"], lab))
    cov["near_source"] = near_src
    cov["audio_hours"] = round(sum(x["dur"] for x in recs) / 3600, 3)
    return recs, cov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*", default=list(TS.LANGS))
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    root = data_dir(NAME)
    root.mkdir(parents=True, exist_ok=True)
    all_recs, cov = [], {}
    for lang in a.langs:
        recs, c = build_lang(lang, a.n)
        all_recs += recs
        cov[lang] = c
        print(lang, json.dumps(c, ensure_ascii=False), flush=True)
    write_manifest(root, all_recs)
    (root / "coverage.json").write_text(json.dumps(cov, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
