"""Round 19 (Korean, part 3): the identity class now comes from the reference sounds. For ordinary
text it must equal the spelling-based class of parts 1-2 on every sound, so no output changes.

    python -m bench.round19ko.class_check

Texts checked: the dev items (prompts and planted targets), FLEURS ko train / validation / test,
Zeroth test, and the Lingolix Korean targets (text only; no scoring). For the contrast-audit edits
the classes are expected to differ exactly at edited vowels.
"""
from __future__ import annotations

import json

from bench.round19ko.cv_pool import work
from pronunciada import ko_identity as KI
from pronunciada.text import get_g2p


def spelled_classes(ref):
    """The parts 1-2 rule (vowel class from the spelling)."""
    out = KI.classes(ref)
    for i, p in enumerate(ref.phones):
        if p.is_nucleus:
            out[i] = KI._vowel_of(ref.syllables[p.syllable].text) if len(p.ipa) == 1 else None
    return out


def main():
    from bench.pipeline.common import records
    from bench.third_party import lingolix as LX

    g = get_g2p("ko")
    texts = {}
    for x in (work() / "dev_items.jsonl").read_text(encoding="utf-8").splitlines():
        texts[json.loads(x)["target"]] = "dev"
    for split in ("train", "test"):
        for r in records("fleurs", split=split, lang="ko"):
            texts[r["text"]] = f"fleurs-{split}"
    for r in records("fleurs-validation", lang="ko"):
        texts[r["text"]] = "fleurs-val"
    for m in LX.tests()["tier1_ko"]:
        texts[m["target"]] = "lingolix"
    import pyarrow.parquet as pq

    t = pq.read_table(work() / "zeroth" / "test-00000-of-00001.parquet", columns=["text"])
    for x in t["text"].to_pylist():
        texts[x] = "zeroth-test"
    diff, n = {}, 0
    for text, src in texts.items():
        ref = g(text)
        a, b = spelled_classes(ref), KI.classes(ref)
        n += len(a)
        d = [(ref.phones[i].ipa, a[i], b[i]) for i in range(len(a)) if a[i] != b[i]]
        if d:
            diff.setdefault(src, []).append((text, d[:3]))
    out = {"n_texts": len(texts), "n_sounds": n, "texts_with_differences": {k: len(v) for k, v in diff.items()},
           "examples": {k: v[:5] for k, v in diff.items()}}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    main()
