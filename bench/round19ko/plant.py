"""Round 19 (Korean): the dev set of planted reference-text edits (changed word vs correct word).

    python -m bench.round19ko.plant          # -> $CACHE/round19/korean/dev_items.jsonl

Audio (natives reading their own prompt), none of it held out anywhere:
  fleurs-val  FLEURS ko VALIDATION (226 clips; no speaker ids: the sentence id groups clips)
  cv          Common Voice 22.0 ko clips of bench.round19ko.cv_pool (39 speakers; no Lingolix speaker,
              no cv-real test-half speaker)
Every clip gives a control item (its own prompt) and one changed item: one word of the prompt is
replaced in the reference text only (the audio still says the original). The kinds follow the
Lingolix test 1 taxonomy, generated here by our own rules:
  sub_subtle    one substituted sound: a vowel for a vowel, or a consonant differing only in
                aspiration / tension (피어 -> 비어, 달 -> 딸)
  sub_moderate  one consonant substitution differing in at most 2 articulatory features, or two
                vowel substitutions
  sub_gross     any other substitution (1-2 sounds)
  indel         1-2 edits of which at least one adds or drops a sound
  word_omission a real word inserted into the reference (the speaker "dropped" it)
Replacements are real words (eojeols) of a vocabulary built from FLEURS ko transcripts (train /
validation / test texts) and the Common Voice ko TSV sentences, with every Lingolix ko prompt and
target removed. Distances are on our front-end's sounds (ZIPA conventions; tense consonants marked);
a candidate is kept only if, in the full new sentence, every other word keeps its sounds
(bench.textswap.verify). Positions are medial (never the first or last word). Deterministic
(seeded per clip). Kinds are assigned round-robin over clips in a seeded order so each kind gets
about a fifth (word_omission about half as many, as in the benchmark).
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from bench import textswap as TS
from bench.round19ko.cv_pool import audio_dir, work
from pronunciada import articulatory as A

_STRIP = re.compile(r"^\W+|\W+$")
_HANGUL = re.compile(r"^[가-힣]+$")
KINDS = ("sub_subtle", "sub_moderate", "sub_gross", "indel", "sub_subtle", "sub_moderate", "sub_gross",
         "indel", "word_omission")


def sounds(ref, w) -> tuple[str, ...]:
    return tuple(ref.phones[i].ipa + ("*" if getattr(ref.phones[i], "tense", None) else "")
                 for i in range(*w.phones))


def unit_sounds(ref, u_span) -> tuple[str, ...]:
    out = []
    for w in ref.words:
        if w.char_span[0] < u_span[1] and w.char_span[1] > u_span[0]:
            out.extend(sounds(ref, w))
    return tuple(out)


def vocabulary() -> Counter:
    from bench.pipeline.common import records
    from bench.third_party import lingolix as LX

    bad = set()
    for m in LX.tests()["tier1_ko"]:
        for t in (m["prompt"], m["target"]):
            bad |= {_STRIP.sub("", x) for x in t.split()}
    voc = Counter()
    texts = []
    for split in ("train", "test"):
        texts += [r["text"] for r in records("fleurs", split=split, lang="ko")]
    texts += [r["text"] for r in records("fleurs-validation", lang="ko")]
    meta = Path(work()).parents[1] / "data" / "cv-real" / "meta" / "ko"
    for f in ("dev", "test", "other", "invalidated"):
        with open(meta / f"{f}.tsv", encoding="utf-8") as fh:
            texts += [r["sentence"] for r in csv.DictReader(fh, delimiter="\t")]
    for t in texts:
        for tok in t.split():
            w = _STRIP.sub("", tok)
            if _HANGUL.match(w) and w not in bad:
                voc[w] += 1
    return voc


def _dels(t: tuple, depth: int = 2):
    out = {t}
    if depth >= 1:
        for i in range(len(t)):
            out.add(t[:i] + t[i + 1:])
    if depth >= 2 and len(t) > 1:
        for i, j in combinations(range(len(t)), 2):
            out.add(tuple(x for k, x in enumerate(t) if k not in (i, j)))
    return out


def ops(a: tuple, b: tuple) -> list[tuple[str, str | None, str | None]]:
    """Levenshtein alignment of two sound tuples: [(op, a_sound, b_sound)] for the non-equal steps."""
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]:
                out.append(("sub", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            out.append(("del", a[i - 1], None))
            i -= 1
        else:
            out.append(("ins", None, b[j - 1]))
            j -= 1
    return out[::-1]


def _is_vowel(s: str) -> bool:
    try:
        return A.features(s.rstrip("*"))["type"] == "vowel"
    except KeyError:
        return False


def grade(o: list) -> str | None:
    if not o or len(o) > 2:
        return None
    if any(op != "sub" for op, _, _ in o):
        return "indel"
    pairs = [(x, y) for _, x, y in o]
    vv = [(_is_vowel(x) and _is_vowel(y)) for x, y in pairs]
    if len(pairs) == 1:
        x, y = pairs[0]
        if vv[0]:
            return "sub_subtle"
        if _is_vowel(x) != _is_vowel(y):
            return "sub_gross"
        try:
            d = A.diff(x.rstrip("*"), y.rstrip("*"))
        except KeyError:
            return None
        if x.rstrip("*") == y.rstrip("*") or set(d) <= {"aspirated"}:
            return "sub_subtle"                    # tension or aspiration only
        return "sub_moderate" if len(d) <= 2 else "sub_gross"
    if all(vv):
        return "sub_moderate"
    return "sub_gross"


class Index:
    def __init__(self, voc: Counter, g2p):
        self.g2p = g2p
        self.snd = {}
        self.keys = defaultdict(set)
        for w in voc:
            try:
                ref = g2p(w)
            except Exception:  # noqa: BLE001
                continue
            if ref.warnings or not ref.words:
                continue
            s = tuple(x for ww in ref.words for x in sounds(ref, ww))
            if len(s) < 2:
                continue
            self.snd[w] = s
            for k in _dels(s):
                self.keys[k].add(w)
        self.voc = voc

    def near(self, s: tuple) -> set[str]:
        out = set()
        for k in _dels(s):
            out |= self.keys.get(k, set())
        return out


def plant_one(uid: str, text: str, kind: str, g2p, idx: Index, common: list[str]):
    ref = g2p(text)
    if not TS.source_ok("ko", text, ref):
        return None
    ou = TS.units(ref, "ko")
    rng = random.Random(f"r19ko|{uid}|{kind}")
    pos = [k for k in range(1, len(ou) - 1) if _HANGUL.match(ou[k].text)]
    rng.shuffle(pos)
    for k in pos:
        u = ou[k]
        if kind == "word_omission":
            cand = list(common)
            rng.shuffle(cand)
            for w in cand[:30]:
                a = u.span[0]
                new_text = text[:a] + w + " " + text[a:]
                nref = g2p(new_text)
                if not set(nref.warnings) <= set(ref.warnings):
                    continue
                nu = TS.units(nref, "ko")
                if len(nu) != len(ou) + 1 or nu[k].text != w:
                    continue
                if any(x.phones != y.phones for x, y in zip(ou[:k], nu[:k])) or \
                        any(x.phones != y.phones for x, y in zip(ou[k:], nu[k + 1:])):
                    continue
                return {"target": new_text, "label": {"bucket": kind, "word_index": k, "original": None,
                                                      "replacement": w, "n_ops": None}}
            continue
        s = unit_sounds(ref, u.span)
        cands = sorted(c for c in idx.near(s) if c != u.text)
        rng.shuffle(cands)
        for c in cands:
            o = ops(s, idx.snd[c])
            if grade(o) != kind:
                continue
            core_a = u.span[0]
            new_text = text[:core_a] + c + text[core_a + len(u.text):]
            got = TS.verify("ko", g2p, ou, k, new_text, c, ref.warnings)
            if got is None:
                continue
            nref, nu = got
            o2 = ops(s, unit_sounds(nref, nu[k].span))
            if grade(o2) != kind:
                continue
            return {"target": new_text, "label": {"bucket": kind, "word_index": k, "original": u.text,
                                                  "replacement": c, "n_ops": len(o2),
                                                  "ops": [[a or "", b or ""] for _, a, b in o2]}}
    return None


def sources():
    from bench.pipeline.common import records

    out = []
    for r in records("fleurs-validation", lang="ko"):
        out.append({"src": "fleurs-val", "id": r["id"], "audio": r["audio"], "prompt": r["text"],
                    "speaker": f"fleurs-sent-{r['labels']['fleurs_id']}", "dur": r["dur"]})
    pool = json.loads((work() / "cv_pool.json").read_text())["pool"]
    stats = json.loads((work() / "cv_audio_stats.json").read_text())
    for p in pool:
        st = stats.get(p["stem"], {})
        if "error" in st or st.get("active_s", 0) < 0.3:
            continue
        out.append({"src": "cv", "id": f"cv-{p['stem']}", "audio": str(audio_dir() / f"{p['stem']}.flac"),
                    "prompt": p["sentence"], "speaker": p["speaker"], "dur": p["dur"]})
    return out


def main():
    from pronunciada.text import get_g2p

    g2p = get_g2p("ko")
    voc = vocabulary()
    idx = Index(voc, g2p)
    common = [w for w, n in voc.most_common(400) if 2 <= len(w) <= 4]
    src = sources()
    order = sorted(src, key=lambda r: random.Random(f"r19ko-order|{r['id']}").random())
    items, tries = [], Counter()
    for n, r in enumerate(order):
        items.append({**r, "item": f"{r['id']}#ctl", "target": r["prompt"], "label": None})
        start = n % len(KINDS)
        for j in range(len(KINDS)):                  # the clip's kind, else the next one that fits
            kind = KINDS[(start + j) % len(KINDS)]
            tries[kind] += 1
            got = plant_one(r["id"], r["prompt"], kind, g2p, idx, common)
            if got:
                items.append({**r, "item": f"{r['id']}#chg", **got})
                break
    out = work() / "dev_items.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    ch = [i for i in items if i["label"]]
    print("vocab", len(voc), "indexed", len(idx.snd), "items", len(items), "changed", len(ch))
    print(Counter((i["src"], i["label"]["bucket"]) for i in ch))
    for i in ch[:25]:
        print(i["src"], i["label"]["bucket"], i["label"]["original"], "->", i["label"]["replacement"], i["label"].get("ops"))


if __name__ == "__main__":
    main()
