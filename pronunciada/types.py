"""Shared data contract for pronunciada. Every module talks through these types.

Reference*  = what the learner was supposed to say (built from text by pronunciada.text).
*Result     = what the scorer returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- reference

@dataclass
class RefPhone:
    """One target sound.

    `ipa` is written in ZIPA token conventions: every character must be a token in
    pronunciada/data/zipa_tokens.txt (use pronunciada.text.ipa.normalize). A phone may span several
    tokens (e.g. "tʰ", "aː", "tʃ", "aʊ").
    """
    ipa: str
    word: int                      # index into Reference.words
    syllable: int                  # index into Reference.syllables
    is_nucleus: bool = False       # the vowel (or syllabic consonant) carrying the syllable
    # Acceptable alternative realisations: (ipa, log_prior). ipa == "" means "may be dropped".
    # G2P modules fill rule-based ones (weak forms, liaison, flapping ...); pronunciada.variants
    # appends the ones learned from native speech.
    alts: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class RefSyllable:
    word: int
    phones: tuple[int, int]        # [start, end) into Reference.phones
    text: str = ""                 # orthographic chunk when known (a hanzi, a kana mora group ...)
    # Mandarin: surface tone the learner should produce after sandhi, 1-4, 5 = neutral.
    tone: int | None = None
    tone_lexical: int | None = None        # dictionary tone before sandhi
    # Other tones that are also fine here (optional sandhi, regional neutral-tone differences ...).
    tone_alts: list[int] = field(default_factory=list)
    # Lexical stress: 1 primary, 2 secondary, 0 unstressed. None = language has no word stress.
    stress: int | None = None
    # Japanese pitch accent: True = high mora, False = low. None elsewhere.
    pitch_high: bool | None = None
    # Japanese only (round 13): the word's accent class "<pos>/<pos detail>/<head|chain>/c<content
    # words in its accent phrase>/<h|a|o|n>" (heiban / atamadaka / odaka / nakadaka); decides
    # whether the dictionary accent is dependable enough to judge (pronunciada.prosody.pitch_judge).
    pitch_class: str | None = None


@dataclass
class RefWord:
    text: str                      # exactly as written in the prompt (original script)
    syllables: tuple[int, int]     # [start, end) into Reference.syllables
    phones: tuple[int, int]        # [start, end) into Reference.phones
    char_span: tuple[int, int] = (0, 0)    # offsets into Reference.text


@dataclass
class Reference:
    lang: str                      # canonical code: en zh ja es fr de it pt ru ko
    text: str                      # original prompt text
    words: list[RefWord] = field(default_factory=list)
    syllables: list[RefSyllable] = field(default_factory=list)
    phones: list[RefPhone] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)   # e.g. out-of-vocabulary words, dropped symbols

    def validate(self) -> None:
        """Raise ValueError if indices/spans are inconsistent or a phone uses non-ZIPA symbols."""
        from .text.ipa import is_valid

        for wi, w in enumerate(self.words):
            s0, s1 = w.syllables
            p0, p1 = w.phones
            if not (0 <= s0 <= s1 <= len(self.syllables) and 0 <= p0 < p1 <= len(self.phones)):
                raise ValueError(f"word {wi} ({w.text!r}) has bad spans")
            for si in range(s0, s1):
                if self.syllables[si].word != wi:
                    raise ValueError(f"syllable {si} not owned by word {wi}")
        prev_end = 0
        for si, s in enumerate(self.syllables):
            p0, p1 = s.phones
            if p0 != prev_end or p1 <= p0:
                raise ValueError(f"syllable {si} phones not contiguous")
            prev_end = p1
            if sum(self.phones[i].is_nucleus for i in range(p0, p1)) != 1:
                raise ValueError(f"syllable {si} must have exactly one nucleus")
            for pi in range(p0, p1):
                if self.phones[pi].syllable != si or self.phones[pi].word != s.word:
                    raise ValueError(f"phone {pi} has wrong syllable/word index")
        if prev_end != len(self.phones):
            raise ValueError("phones not fully covered by syllables")
        for pi, p in enumerate(self.phones):
            if not p.ipa or not is_valid(p.ipa):
                raise ValueError(f"phone {pi} {p.ipa!r} is not valid ZIPA IPA")
            for alt, _ in p.alts:
                if alt and not is_valid(alt):
                    raise ValueError(f"phone {pi} alt {alt!r} is not valid ZIPA IPA")


# --------------------------------------------------------------------------- results

@dataclass
class PhoneResult:
    ipa: str                       # target sound
    prob: float                    # calibrated 0-1 probability this sound is acceptable
    start: float                   # seconds
    end: float
    heard: str | None = None       # what it sounded like instead, when prob is low ("" = missing)
    features: dict = field(default_factory=dict)        # raw evidence, for debugging/benchmarks


@dataclass
class SyllableResult:
    text: str
    start: float
    end: float
    tone_expected: int | None = None
    tone_heard: int | None = None
    tone_prob: float | None = None         # calibrated 0-1 probability the tone is acceptable
    stress_expected: int | None = None
    stress_prob: float | None = None       # probability this syllable carries the right stress level
    pitch_prob: float | None = None        # Japanese pitch accent


@dataclass
class WordResult:
    text: str
    prob: float                    # calibrated 0-1 probability the word is acceptable overall
    start: float
    end: float
    phones: list[PhoneResult] = field(default_factory=list)
    syllables: list[SyllableResult] = field(default_factory=list)
    sounds_prob: float | None = None       # sounds only
    stress_prob: float | None = None       # stress placement (None if not applicable)
    tone_prob: float | None = None         # tones (None if not applicable)
    present_prob: float | None = None      # probability the word was said at all
    omitted: bool = False                  # present_prob < 0.5


@dataclass
class SentenceResult:
    lang: str
    text: str
    prob: float                    # calibrated 0-1 probability the whole utterance is acceptable
    strictness: float
    words: list[WordResult] = field(default_factory=list)
    fluency: float | None = None           # 0-1, pacing / pausing
    completeness: float | None = None      # share of words actually said
    timing_ms: dict = field(default_factory=dict)       # per-stage latency
    warnings: list[str] = field(default_factory=list)
    # regression-style 0-1 sentence scores (accuracy, fluency, prosodic, total = the
    # speechocean762 0-10 scales / 10); empty when the scorer does not provide them
    scores: dict = field(default_factory=dict)
    # round 9: P(the recording is a reading of this prompt at all); already multiplied into
    # prob. None when the channel is not shipped / disabled
    match_prob: float | None = None
    # audio-quality gate (pronunciada.quality.check): measured values, "issues" and
    # "retry_recommended"; scores are returned either way. None when the check is disabled
    audio: dict | None = None

    def coaching(self, max_items: int | None = None, **gate) -> list[dict]:
        """Structured coaching items, most important first (pronunciada.coach.coaching)."""
        from .coach import coaching

        return coaching(self, max_items=max_items, **gate)
