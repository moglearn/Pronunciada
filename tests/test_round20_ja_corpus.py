"""Japanese length corpus tools: front-end -> mora units -> aligner phones -> segments."""
import pytest

pytest.importorskip("pyopenjtalk")

from bench.round20_ja_corpus.build import segments_for  # noqa: E402
from bench.round20_ja_corpus.units import PHONES, ref_units, token, variants  # noqa: E402


@pytest.fixture(scope="module")
def g2p():
    from pronunciada.text.ja import build

    return build()


def _fake_alignment(words, pause_after_word=None, variant=None):
    """Evenly spaced phones, 50 ms each; optional 200 ms pause after one word."""
    t = 0.1
    wt, pt = [], []
    for wi, wd in enumerate(words):
        phs = [p for u in wd["units"] for p in u["ph"]] if variant is None or wi != variant[0] else variant[1]
        w0 = t
        for p in phs:
            pt.append([round(t, 3), round(t + 0.05, 3), p])
            t += 0.05
        wt.append([round(w0, 3), round(t, 3), token(phs)])
        if pause_after_word == wi:
            t += 0.2
    return {"start": 0, "end": t + 0.1, "tiers": {"words": {"entries": wt}, "phones": {"entries": pt}}}


def test_units_long_geminate_nasal(g2p):
    words = ref_units(g2p("東京に行った。"))
    units = [u for w in words for u in w["units"]]
    assert [u["ph"] for u in units] == [["t", "oo"], ["ky", "oo"], ["n", "i"], ["i"], ["cl", "t", "a"]]
    assert units[0]["long"] and units[4]["gem"]
    words = ref_units(g2p("先生"))
    assert [p for u in words[0]["units"] for p in u["ph"]] == ["s", "e", "N", "s", "ee"]
    assert all(p in PHONES for w in words for u in w["units"] for p in u["ph"])


def test_variants_and_segments(g2p):
    rec = {"words": ref_units(g2p("東京に行った。"))}
    names = [v[0] for v in variants(rec["words"][0]["units"])]
    assert names[0] == "canon" and "short@0" in names
    segs, st = segments_for(rec, _fake_alignment(rec["words"], pause_after_word=0))
    assert st == "ok"
    kinds = [(s["s"], s["k"]) for s in segs]
    assert kinds[:4] == [("t", "C"), ("oː", "V"), ("kj", "C"), ("oː", "V")]
    q = next(s for s in segs if s["k"] == "Q")
    rel = segs[segs.index(q) + 1]
    assert q["gem"] and rel["gem"] and rel["s"] == "t"
    # pause after 東京: its last unit is phrase-final, its last segment pre-pausal
    assert segs[3]["pf"] and segs[3]["pp"] and segs[3]["next"] == "#"
    assert not segs[1]["pf"] and segs[-1]["uf"]
    # morae: to-o-kyo-o-ni-i-Q-ta = 8; the Q mora precedes the release
    assert q["mora"] == 6 and rel["mora"] == 7
    # aligner chose the shortened variant -> reported, not used
    short = [p if p != "oo" else "o" for p in ["t", "oo", "ky", "oo"]]
    _, st = segments_for(rec, _fake_alignment(rec["words"], variant=(0, ["t", "o", "ky", "oo"])))
    assert st == "variant:short"
    assert short == ["t", "o", "ky", "o"]
