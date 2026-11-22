"""Turn the raw OpenPronounce outputs (bench.rivals.openpronounce) into the per-record files of the
external-system contract (bench.rivals.common), attached to Pronunciada's reference for each prompt.

    python -m bench.rivals.convert --mode default [--datasets speechocean762,fleurs-swap]

System names: openpronounce-1.1.0 (default), openpronounce-1.1.0-fast, openpronounce-1.1.0-student.

Scores (higher = better, 0-100, floor 0):
  sentence  the library's `score` (= scores["total"], the trained scorer).
  word      100 x (1 - p), p = the word model's probability that the word is mispronounced (`confidence` in
            differences.words); a word the library calls "not said" gets p = max(p, missing_confidence).
            OpenPronounce words (regex \\b[\\w']+\\b over the prompt) are attached to reference words by
            character span (mean when several overlap; a reference word overlapped by none gets the mean).
  sound     100 x (1 - p) of the phone model's probability for the expected sound: the sounds of an
            OpenPronounce word are aligned to the sounds of the one reference word it covers by edit
            distance on the IPA symbols (length and stress marks dropped); a reference sound left without
            a partner, or a reference word covered by several OpenPronounce words, takes its word's score.
  secondary the flags the app shows: word 0 when reported (mispronounced or not said), 100 otherwise.
Nothing is fitted or thresholded here.
"""
from __future__ import annotations

import argparse
import re
import unicodedata

from bench.rivals.openpronounce import MODES, VERSION, read_raw

WORD_RE = re.compile(r"\b[\w']+\b")
SYSTEM = {"default": f"openpronounce-{VERSION}", "fast": f"openpronounce-{VERSION}-fast",
          "student": f"openpronounce-{VERSION}-student"}
_STRIP = set("ːˑˈˌ.‿͡ ̃")


def _norm(p: str) -> str:
    p = unicodedata.normalize("NFD", p)
    return "".join(c for c in p if c not in _STRIP and unicodedata.category(c) != "Mn") or p


def _align(a: list[str], b: list[str]) -> list[int]:
    """For each element of a, the index of its partner in b (-1 if none): unit-cost edit distance, ties
    favour substitution so that a sound changed for another still finds its slot."""
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + (a[i - 1] != b[j - 1]), D[i - 1][j] + 1, D[i][j - 1] + 1)
    out, i, j = [-1] * n, n, m
    while i > 0 and j > 0:
        if D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def word_probs(o: dict) -> dict[int, float]:
    p = {}
    for w in o.get("words", []):
        x = float(w["confidence"])
        if w.get("missing"):
            x = max(x, float(w.get("missing_confidence", 1.0)))
        p[int(w["position"])] = x
    return p


def to_reference(ref, text: str, o: dict):
    spans = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
    wp = word_probs(o)
    reported = {int(e["position"]) for e in o.get("errors", [])}
    by_pos = {int(w["position"]): w for w in o.get("words", [])}
    w1 = {i: 100.0 * (1 - p) for i, p in wp.items()}
    w2 = {i: (0.0 if i in reported else 100.0) for i in wp}
    mean1 = sum(w1.values()) / len(w1) if w1 else float(o["score"])
    mean2 = sum(w2.values()) / len(w2) if w2 else 100.0
    cover, words1, words2, miss = [], [], [], 0
    for w in ref.words:
        a, b = w.char_span
        hit = [i for i, (s, e) in enumerate(spans) if s < b and e > a and i in w1]
        cover.append(hit)
        words1.append(sum(w1[i] for i in hit) / len(hit) if hit else mean1)
        words2.append(sum(w2[i] for i in hit) / len(hit) if hit else mean2)
        miss += not hit
    phone = [words1[p.word] for p in ref.phones]
    n_sound = 0
    for wi, w in enumerate(ref.words):
        if len(cover[wi]) != 1:
            continue
        op = by_pos[cover[wi][0]]["phones"]
        lo, hi = w.phones
        mine = [_norm(ref.phones[k].ipa) for k in range(lo, hi)]
        part = _align(mine, [_norm(p["expected"]) for p in op])
        for k, j in zip(range(lo, hi), part):
            if j >= 0:
                phone[k] = 100.0 * (1 - float(op[j]["confidence"]))
                n_sound += 1
    return words1, words2, phone, miss, n_sound


def convert(mode: str, datasets: list[str] | None = None) -> dict:
    from bench.rivals.common import build_ref, tasks, write_error, write_record

    system = SYSTEM[mode]
    T = tasks()
    stats = {}
    for ds, ts in T.items():
        if datasets and ds not in datasets:
            continue
        R = read_raw(mode, ds)
        if not R:
            continue
        c = {"records": 0, "errors": 0, "unmapped_words": 0, "sounds_own": 0, "sounds": 0}
        for t in ts:
            r = R.get(t["id"])
            if r is None:
                continue
            c["records"] += 1
            o = r["out"]
            ref = build_ref(t["lang"], t["text"])
            if ref is None:
                write_error(system, t, "front end could not read the prompt")
                c["errors"] += 1
                continue
            if "error" in o or o.get("score") is None:
                write_error(system, t, o.get("error", "no score"))
                c["errors"] += 1
                continue
            w1, w2, ph, miss, ns = to_reference(ref, r["text"], o)
            write_record(system, t, ref, word=w1, sentence=float(o["score"]), phone=ph,
                         secondary={"word": w2, "sentence": float(o["score"]), "phone": None},
                         extra={"scores": o["scores"], "formula_score": o["formula_score"], "reliable": o["reliable"],
                                "reasons": o["reasons"], "sec": o["sec"], "mode": mode, "rev": r["rev"]})
            c["unmapped_words"] += miss
            c["sounds_own"] += ns
            c["sounds"] += len(ref.phones)
        stats[ds] = c
        print(ds, c, flush=True)
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=MODES, default="default")
    ap.add_argument("--datasets", default=None)
    a = ap.parse_args(argv)
    convert(a.mode, a.datasets.split(",") if a.datasets else None)


if __name__ == "__main__":
    main()
