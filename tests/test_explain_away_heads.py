"""The explain-away pass for languages whose phone table carries a head-based contrast cap (ko, ru, zh).

The column head_cap_z is added to the first pass's phone table outside features.phone_table; the second pass
must take it from the first pass like the other pass-1-only columns, instead of failing with KeyError.
"""
import numpy as np
import pytest

import pronunciada


@pytest.mark.model
@pytest.mark.parametrize("lang,text", [("ko", "오늘 아침에 빨래를 다 널고 나서 들어왔다"),
                                       ("ru", "Сегодня утром мы пошли гулять в парк")])
def test_explain_away_runs_with_head_caps(tts, monkeypatch, lang, text):
    monkeypatch.setenv("PRONUNCIADA_EA_BAND", "0,1")          # make the pass run whatever the scores
    wav = tts("The quick brown fox jumps over the lazy dog")
    sc = pronunciada.Scorer(lang, threads=2, explain_away=True)
    res = sc.score(wav, text)
    c = res._pronunciada
    assert "head_cap_z" in c["tab"]
    assert len(res.words) == len(text.split())
    assert all(np.isfinite(w.prob) for w in res.words)
    if c.get("explained"):
        assert len(c["tab"]["head_cap_z"]) == len(c["ref"].phones)
