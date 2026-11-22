"""Round 19 (German): why do correct words score low? Audit of a dump, word by word and sound by sound.

    python -m bench.round19.audit dev-base [--fold fit] [--source cv] [--top 60]

For every correct word (all words of control items, and the unchanged words of changed items) the
word's error signal is 1 - benchmark word score. "Flagged" = at or above the dump's FA@80 threshold.
Each flagged word gets one cause, the first that applies:
  oov        the front-end had no lexicon entry (letter rules)        foreign  capitalised OOV / name
  neighbour  next to the changed word of a changed item                final    last word of the prompt
  sound:<expected>><heard>   the word's lowest sound and what the decoder heard there
Sound table: over the lowest-scoring sound of every flagged correct word, (expected, heard) pairs.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict

from bench.round19.evaluate import load_dump, manifest_for
from bench.third_party import lingolix as L

_P = re.compile(r"[^\w\s]", re.UNICODE)


def word_rows(name, fold="all", source=None):
    items = manifest_for(name)
    dump = load_dump(name)
    dev = not name.startswith("lingolix")
    if dev:
        items = [it for it in items if (fold == "all" or it["fold"] == fold) and (source is None or it["source"] == source)]
    items = [it for it in items if it["id"] in dump and not dump[it["id"]].get("error")]
    rows = []
    for it in items:
        rec = dump[it["id"]]
        lab = it.get("label") or {}
        if lab and not lab.get("word_level_scorable", True):
            continue
        toks = it["target"].split()
        al = L.align(toks, rec["words"])
        pos = lab.get("word_index")
        oov = {w.split(":", 1)[1] for w in rec.get("warnings", []) if w.startswith("oov:")}
        # map token -> internal words by character span
        spans, cur = [], 0
        for m in re.finditer(r"\S+", it["target"]):
            spans.append((m.start(), m.end()))
        for k, tok in enumerate(toks):
            a, b = spans[k]
            iw = [w for w in rec["rwords"] if w["span"][0] < b and w["span"][1] > a]
            rows.append({"item": it["id"], "k": k, "tok": tok, "label": int(k == pos),
                         "err": L.error_signal(al[k]), "bucket": lab.get("bucket") or lab.get("category") or "control",
                         "iw": iw, "n_tok": len(toks), "pos": pos,
                         "oov": any(w["text"].lower() in oov or _P.sub("", w["text"]).lower() in oov for w in iw),
                         "source": it.get("source", "lingolix")})
    return rows


def threshold(rows):
    pos = sorted((r["err"] for r in rows if r["label"]), reverse=True)
    return pos[min(math.ceil(0.8 * len(pos)) - 1, len(pos) - 1)]


def lowest_sound(r):
    best = None
    for w in r["iw"]:
        for p in w["phones"]:
            if p["zl"] is not None and (best is None or p["zl"] < best["zl"]):
                best = p
    return best


def cause(r):
    if r["oov"]:
        return "foreign/oov" if r["tok"][:1].isupper() and r["k"] > 0 else "oov"
    if r["pos"] is not None and abs(r["k"] - r["pos"]) == 1:
        return "neighbour"
    if r["k"] == r["n_tok"] - 1:
        return "final"
    if any(not w["phones"] for w in r["iw"]) or not r["iw"]:
        return "unmatched"
    if r["iw"] and min((w["present"] or 1.0) for w in r["iw"]) < 0.5:
        return "absent"
    return "sound"


def report(name, fold="all", source=None, top=60):
    rows = word_rows(name, fold, source)
    thr = threshold(rows)
    neg = [r for r in rows if not r["label"]]
    flagged = [r for r in neg if r["err"] >= thr]
    print(f"{name}: correct words {len(neg)}, changed {sum(r['label'] for r in rows)}, threshold err "
          f"{thr:.3f} (score {1 - thr:.3f}), flagged correct {len(flagged)} ({100 * len(flagged) / len(neg):.2f}%)")
    cz = Counter(cause(r) for r in flagged)
    base = Counter(cause(r) for r in neg)
    print("causes (flagged / all correct, rate):")
    for k, n in cz.most_common():
        print(f"  {k:12s} {n:4d} / {base[k]:5d}  {100 * n / base[k]:.1f}%")
    pairs = Counter()
    for r in flagged:
        if cause(r) != "sound":
            continue
        p = lowest_sound(r)
        if p:
            alt = "" if p["alt"] == 0 else f"[alt{p['alt']}]"
            pairs[(p["ipa"] + alt, p["heard"] or "-")] += 1
    print("lowest sound of 'sound' false flags (expected > heard):")
    for (a, b), n in pairs.most_common(40):
        print(f"  {a:10s} > {b:10s} {n}")
    print("flagged correct words (worst first):")
    for r in sorted(flagged, key=lambda r: -r["err"])[:top]:
        p = lowest_sound(r)
        ph = " ".join(x["ipa"] for w in r["iw"] for x in w["phones"])
        hd = " ".join((x["heard"] or "-") for w in r["iw"] for x in w["phones"])
        print(f"  {r['err']:.3f} {r['item'][:22]:22s} {r['tok'][:24]:24s} {cause(r):11s} /{ph}/ heard /{hd}/"
              + (f" low {p['ipa']}>{p['heard']} zl {p['zl']}" if p else ""))
    lab = sorted([r for r in rows if r["label"]], key=lambda r: r["err"])
    print("changed words scoring highest (hardest), lowest error first:")
    for r in lab[: max(10, len(lab) // 5)]:
        ph = " ".join(x["ipa"] for w in r["iw"] for x in w["phones"])
        hd = " ".join((x["heard"] or "-") for w in r["iw"] for x in w["phones"])
        print(f"  {r['err']:.3f} {r['item'][:22]:22s} {r['bucket']:13s} {r['tok'][:20]:20s} /{ph}/ heard /{hd}/")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--fold", default="all")
    ap.add_argument("--source")
    ap.add_argument("--top", type=int, default=60)
    a = ap.parse_args(argv)
    report(a.name, a.fold, a.source, a.top)


if __name__ == "__main__":
    main()
