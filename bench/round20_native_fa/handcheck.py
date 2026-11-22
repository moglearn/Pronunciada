"""Hand check of the automatic buckets: a seeded random sample of false alarms per language, printed with
the evidence a reader needs (prompt, target, the word, expected sounds with the aligner's chosen
alternative, what the decoder heard per sound, times, factors, the free decode around the word, the
Whisper transcript). The reader writes one bucket per sample into handlabels_<lang>.json
({sample id: bucket}); `agree` compares them with the automatic buckets.

    python -m bench.round20_native_fa.handcheck show --lang en [--n 50]
    python -m bench.round20_native_fa.handcheck agree --lang en

Reading rules used for the labels (written before labelling):
  noise      the transcript shows the speaker said something else at this word, or the clip is cut
             inside the word
  neighbour  within two words of the changed word and the decoded sounds of the word or its boundary sound belong
             to the changed word's region (shared or shifted boundary)
  variant    the decoded sounds are a known accepted pronunciation of the word (accent vowel, flap,
             reduction listed in a dictionary, regular assimilation) that the canonical form lacks
  text       a name, number, abbreviation or unknown word read in another plausible way than the front-end's
             (another reading of a name, a number read differently, a Roman numeral as a number)
  boundary   first / last word whose missing or wrong sounds are at the clip edge, or the decoded
             sounds of the word appear shifted into the neighbour's span
  function   a function word or a very short word reduced in fluent speech (sounds merged or dropped)
  tail       decoded sounds agree with the expected ones (at most one weak sound) and the score is only
             a little under the threshold
  residual   the decoder heard a clearly different sound in a content word with no reason above
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from bench.round20_native_fa import core as K
from bench.round20_native_fa.features import asr
from bench.round20_native_fa.rebuild import ROOT


def sample(lang: str, n: int = 50, seed: int = 20) -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "tables" / f"rows_{lang}.jsonl").read_text(encoding="utf-8").splitlines()]
    fa = [r for r in rows if r.get("_bucket_FA")]
    rng = random.Random(seed)
    return rng.sample(fa, min(n, len(fa)))


def show(lang: str, n: int) -> None:
    its = {it["id"]: it for it in K.practice_items(lang)}
    dump = K.dump_rec(lang)
    A = asr(lang)
    out = []
    for j, r in enumerate(sample(lang, n)):
        it, rec = its[r["item"]], dump[r["item"]]
        toks = it["target"].split()
        mark = " ".join(f"[{t}]" if q == r["k"] else t for q, t in enumerate(toks))
        lab = it.get("label") or {}
        lines = [f"#{j} {r['item']}  score {r['score']:.3f}  kind {r['kind']}  changed@{lab.get('word_index')}"
                 f" ({lab.get('original')}->{lab.get('replacement') or lab.get('added')})"]
        lines.append(f"  target: {mark}")
        if it["prompt"] != it["target"]:
            lines.append(f"  prompt: {it['prompt']}")
        a = A.get(it["clip"])
        if a:
            lines.append(f"  whisper (cer {a['cer']}): {a['transcript']}")
        spans = []
        import re

        for m in re.finditer(r"\S+", it["target"]):
            spans.append((m.start(), m.end()))
        s0, s1 = spans[r["k"]]
        for wi, w in enumerate(rec["rwords"]):
            if not (w["span"][0] < s1 and w["span"][1] > s0):
                continue
            ph = " ".join(p["ipa"] + (f"{{{p['alt_ipa']}}}" if p["alt"] else "") for p in w["phones"])
            hd = " ".join((p["heard"] if p["heard"] is not None else "?") or "∅" for p in w["phones"])
            pp = " ".join(f"{p['p']:.2f}" for p in w["phones"])
            lines.append(f"  word {w['text']!r} {w['start']:.2f}-{w['end']:.2f}s  prob {w['prob']}  sounds {w['sounds']}"
                         f" stress {w['stress']} len {w['length']} con {w['contrast']} present {w['present']}"
                         f" id {w['idf']} kod {w['ko_delta']} ins {w['ins']}")
            lines.append(f"    expected /{ph}/")
            lines.append(f"    heard    /{hd}/   p: {pp}")
            fs = rec["frame_shift"]
            t0, t1 = w["start"] - 0.25, w["end"] + 0.25
            fr = [t for t, a_, b_ in rec["free"] if t0 <= a_ * fs <= t1]
            lines.append(f"    free decode +-250 ms: {' '.join(fr)}")
        # the automatic bucket is NOT printed (labels are written blind to it)
        lines.append(f"  position {r['k'] + 1}/{r['n_tok']}  snr {r['snr']}")
        out.append("\n".join(lines))
    p = ROOT / "hand" / f"sample_{lang}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n\n".join(out), encoding="utf-8")
    print(p)


def agree(lang: str) -> dict:
    labels = json.loads((ROOT / "hand" / f"handlabels_{lang}.json").read_text())
    s = sample(lang, len(labels))
    pairs = [(r["_bucket_FA"], labels[str(j)]) for j, r in enumerate(s) if str(j) in labels]
    n = len(pairs)
    agree_n = sum(a == b for a, b in pairs)
    conf = Counter(pairs)
    hand = Counter(b for _, b in pairs)
    per = {}
    for b in sorted(set(a for a, _ in pairs) | set(hand)):
        auto_b = [h for a, h in pairs if a == b]
        per[b] = {"auto_n": len(auto_b), "hand_n": hand[b],
                  "precision": (sum(h == b for h in auto_b) / len(auto_b)) if auto_b else None}
    res = {"n": n, "agreement": agree_n / n if n else None, "hand_counts": dict(hand),
           "per_bucket": per, "confusion": {f"{a}->{b}": c for (a, b), c in conf.most_common()}}
    K.save_json(f"handcheck_{lang}.json", res)
    print(json.dumps(res, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["show", "agree"])
    ap.add_argument("--lang", required=True)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args(argv)
    if a.cmd == "show":
        show(a.lang, a.n)
    else:
        agree(a.lang)


if __name__ == "__main__":
    main()
