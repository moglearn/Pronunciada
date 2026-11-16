"""Our Mandarin front-end's surface tones vs AISHELL-3's human-checked pinyin tones (text only).

    .venv/bin/python -m bench.prosody.tone_agree      -> ~/.cache/pronunciada/prosody/tone_agree.json

Per character (erhua syllables skipped), categories:
  agree       same tone
  alt         AISHELL tone is one of our tone_alts
  reading     different base syllable (polyphone / reading choice)
  neutral     same base, one of the two is neutral (5)
  sandhi      same base, the difference is a documented sandhi: 3<->2 (third-tone sandhi applied by
              one side only), or 一 / 不 tone change; i.e. AISHELL tone == our lexical tone while we
              applied sandhi, or our tone == lexical and AISHELL's is the sandhi form
  other       same base, other tone difference
Training uses agree + alt + sandhi (label = our surface tone).
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from bench.data import load
from pronunciada.acoustic import cache_root

OUT = Path(cache_root()) / "prosody" / "tone_agree.json"


def norm(b: str) -> str:
    return b.replace("ü", "v").replace("u:", "v")


def categorize(ours: dict, ext_py: str) -> str:
    et = int(ext_py[-1]) if ext_py[-1:].isdigit() else None
    eb = norm(ext_py[:-1] if et else ext_py)
    t, lex, alts, ch = ours["tone"], ours["lex"], ours["alts"], ours["ch"]
    if et == t:
        return "agree"
    if et in alts:
        return "alt"
    if eb != norm(ours["base"]):
        return "reading"
    if et == 5 or t == 5:
        return "neutral"
    if ch in "一不" and {et, t} <= {1, 2, 4}:
        return "sandhi"
    if {et, t} == {2, 3} and lex == 3:
        return "sandhi"
    return "other"


def run(splits=("train", "test")) -> dict:
    from pronunciada.text.zh import ZhG2P

    g = ZhG2P()
    captured: list[list[dict]] = []
    orig = g._build

    def hook(S, wl, text, trad):
        captured.append([dict(s) for s in S])
        return orig(S, wl, text, trad)

    g._build = hook
    res = {}
    labels = {}
    for split in splits:
        cats = collections.Counter()
        pairs = collections.Counter()
        examples = collections.defaultdict(list)
        n_utt = n_skip = 0
        for r in load("aishell3", split=split):
            captured.clear()
            g(r["text"])
            S = [s for ph in captured for s in ph]
            hz, py = r["labels"]["hanzi"], r["labels"]["pinyin"]
            if len(py) != len(hz) or "".join(s["ch"] for s in S) != "".join(g.to_simplified(h) for h in hz):
                n_skip += 1
                continue
            n_utt += 1
            hz_of = [k for k, h in enumerate(hz) for _ in h]     # char -> hanzi entry
            per = []
            for ci, s in enumerate(S):
                k = hz_of[ci]
                if len(hz[k]) > 1 or (s["ch"] == "儿" and s["ci"] == s["wl"] - 1 and s["wl"] >= 2):
                    per.append(None)                              # erhua: skipped
                    continue
                c = categorize(s, py[k])
                cats[c] += 1
                per.append(c)
                if c != "agree":
                    et = py[k][-1]
                    pairs[(c, f"{s['tone']}->{et}")] += 1
                    if len(examples[c]) < 12:
                        examples[c].append(f"{s['word']}:{s['ch']} ours {s['base']}{s['tone']} "
                                           f"(lex {s['lex']}) vs {py[k]}")
            labels[r["id"]] = per
        tot = sum(cats.values())
        res[split] = {"utterances": n_utt, "skipped_text_mismatch": n_skip, "syllables": tot,
                      "agreement": cats["agree"] / tot,
                      "agreement_incl_alts": (cats["agree"] + cats["alt"]) / tot,
                      "categories": {k: [v, v / tot] for k, v in cats.most_common()},
                      "top_pairs": [[f"{c} {p}", v] for (c, p), v in pairs.most_common(15)],
                      "examples": examples}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": res, "per_char": labels}, open(OUT, "w"), ensure_ascii=False)
    return res


if __name__ == "__main__":
    res = run()
    for split, r in res.items():
        print(split, {k: v for k, v in r.items() if k != "examples"})
        for c, ex in r["examples"].items():
            print("  ", c, ex[:6])
