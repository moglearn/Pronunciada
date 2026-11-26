"""pronunciada.word_match: heard tokens over a word vs its sounds (Japanese word step, round 19)."""
from pronunciada.word_match import _toks, align_word


class _Ph:
    def __init__(self, ipa, alts=()):
        self.ipa, self.alts = ipa, list(alts)


def _word(*ipas):
    return [_Ph(x) for x in ipas]


def test_affricate_heard_as_stop_plus_voiced_affricate_is_not_extra():
    # issue #3: ついたち, ち decoded as t + dʒ
    ins, dl, _ = align_word(_word("ts", "u", "i", "t", "a", "tʃ", "i"), _toks("suitatdʒi"))
    assert ins == 0 and dl == 1                  # つ heard as す is still a missing sound
    assert align_word(_word("tɕ", "i"), _toks("tdʑi"))[0] == 0
    assert align_word(_word("ts", "u"), _toks("tdzu"))[0] == 0


def test_real_extra_sounds_still_count():
    assert align_word(_word("t", "a"), _toks("tda"))[0] == 1          # a plain stop is not an affricate
    assert align_word(_word("tʃ", "i"), _toks("tʃiu"))[0] == 1
