"""Default English tip sentences for coaching items (pronunciada.coach). Separate from the data:
an app can ignore this module and map `tip_key` + `params` to its own or localised copy.

    tips(items, max_tips=3) -> [{"tip_key", "text", "params", "item"}]

One sentence per tip, at most one tip per sound / syllable / word item, at most `max_tips`
(default 3) per utterance, taken in the coaching order (lowest probability x highest
confidence first). Items without a diagnosis (heard None) get a generic "listen and repeat" tip
only when no diagnosed item is left to fill the cap.
"""
from __future__ import annotations

from . import articulatory as A

PLACE_WORDS = {
    "bilabial": "both lips", "labiodental": "the lower lip against the upper teeth",
    "dental": "the tongue tip at the teeth", "alveolar": "the tongue tip just behind the upper teeth",
    "postalveolar": "the tongue a little further back, behind the gum ridge",
    "retroflex": "the tongue tip curled back", "alveolopalatal": "the front of the tongue raised to the hard palate",
    "palatal": "the middle of the tongue against the hard palate",
    "labial-palatal": "rounded lips and the tongue raised to the hard palate",
    "velar": "the back of the tongue against the soft palate",
    "labial-velar": "rounded lips and the back of the tongue raised",
    "postalveolar-velar": "the tongue raised behind the gum ridge and at the back",
    "uvular": "the very back of the tongue, near the uvula", "pharyngeal": "the throat",
    "epiglottal": "the throat", "glottal": "the throat (vocal folds)",
}
TONE_WORDS = {1: "high and level", 2: "rising", 3: "low (dipping)", 4: "falling", 5: "light and short (neutral)"}
ORDINAL = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]

# tip_key -> default English template (the documented, stable list)
# round 4: feature-level tips (attribute heads), used when no heard sound is named
FEATURE_TEMPLATES = {
    "feature.voicing.add": "Let your voice buzz on /{expected}/ in “{word}”; it sounded voiceless.",
    "feature.voicing.remove": "Say /{expected}/ in “{word}” without voice (no buzz); it sounded voiced.",
    "feature.aspirated.add": "Add a small puff of air after /{expected}/ in “{word}”; it sounded unaspirated.",
    "feature.aspirated.remove": "Say /{expected}/ in “{word}” without a puff of air; it sounded aspirated.",
    "feature.manner.stop": "Close completely and release for /{expected}/ in “{word}”; the air kept flowing.",
    "feature.manner.continuant": "Keep the air flowing through /{expected}/ in “{word}”; it sounded stopped.",
    "feature.manner.affricate": "Start /{expected}/ in “{word}” with a short closure, then let the air hiss.",
    "feature.manner.nasal": "Let the air go through your nose for /{expected}/ in “{word}”; it did not sound nasal.",
    "feature.manner.oral": "Keep the air out of your nose for /{expected}/ in “{word}”; it sounded nasal.",
    "feature.manner.rhotic": "Make /{expected}/ in “{word}” a quick tap or roll of the tongue tip.",
    "feature.manner.other": "Check how you make /{expected}/ in “{word}”: the airflow sounded different.",
    "feature.place": "Make /{expected}/ in “{word}” {expected_place}; it sounded {heard_place}.",
    "feature.palatalised.add": "Raise the middle of your tongue (a slight “y” colour) on /{expected}/ in “{word}”.",
    "feature.palatalised.remove": "Say /{expected}/ in “{word}” without the “y” colour.",
    "feature.lateral.add": "Let the air pass along the sides of the tongue for /{expected}/ in “{word}”.",
    "feature.lateral.remove": "Don't let the air flow around the sides of the tongue for /{expected}/ in “{word}”; it sounded like an l.",
    "feature.height.close": "Raise your tongue a little (mouth less open) for /{expected}/ in “{word}”; it sounded too open.",
    "feature.height.open": "Open your mouth a little more for /{expected}/ in “{word}”; it sounded too closed.",
    "feature.backness.front": "Move your tongue forward for /{expected}/ in “{word}”; it sounded too far back.",
    "feature.backness.back": "Pull your tongue back for /{expected}/ in “{word}”; it sounded too far forward.",
    "feature.rounded.add": "Round your lips for /{expected}/ in “{word}”; they sounded spread.",
    "feature.rounded.remove": "Spread your lips (no rounding) for /{expected}/ in “{word}”; they sounded rounded.",
    "feature.nasal.add": "Let some air through your nose on /{expected}/ in “{word}”; it sounded oral.",
    "feature.nasal.remove": "Keep /{expected}/ in “{word}” oral (no air through the nose); it sounded nasal.",
    "feature.tense.add": "Make /{expected}/ in “{word}” tighter and stronger, with no puff of air.",
    "feature.tense.remove": "Relax /{expected}/ in “{word}”; it sounded tense.",
    "feature.long.add": "Hold /{expected}/ in “{word}” a little longer; it sounded short.",
    "feature.long.remove": "Make /{expected}/ in “{word}” shorter; it sounded long.",
}
PLACE_CLASS = {"labial": "with the lips", "front": "with the tongue tip behind the teeth",
               "post": "with the tongue further back, behind the gum ridge", "dorsal": "with the back of the tongue",
               "laryngeal": "in the throat"}
PLACE_HEARD = {"labial": "made with the lips", "front": "made with the tongue tip", "post": "made further back",
               "dorsal": "made with the back of the tongue", "laryngeal": "made in the throat"}


def feature_tip_key(claim: dict) -> tuple[str, dict]:
    """tip_key and extra params for one attribute-head claim (feature, expected class, heard class)."""
    h, e, x = claim["feature"], claim["expected"], claim["heard"]
    if h == "voicing":
        return ("feature.voicing.add" if e == "voiced" else "feature.voicing.remove"), {}
    if h in ("aspirated", "palatalised", "lateral", "rounded", "nasal", "long", "tense"):
        return (f"feature.{h}.add" if e == "yes" else f"feature.{h}.remove"), {}
    if h == "manner":
        if e == "nasal":
            return "feature.manner.nasal", {}
        if x == "nasal":
            return "feature.manner.oral", {}
        if e == "affricate":
            return "feature.manner.affricate", {}
        if e == "rhotic":
            return "feature.manner.rhotic", {}
        if e == "stop":
            return "feature.manner.stop", {}
        if e in ("fricative", "approximant") and x in ("stop", "affricate"):
            return "feature.manner.continuant", {}
        return "feature.manner.other", {}
    if h == "place":
        return "feature.place", {"expected_place": PLACE_CLASS.get(e, e), "heard_place": PLACE_HEARD.get(x, x)}
    if h in ("height", "backness"):
        order = {"height": ("close", "mid", "open"), "backness": ("front", "central", "back")}[h]
        closer = order.index(e) < order.index(x)          # expected is closer / fronter than heard
        if h == "height":
            return ("feature.height.close" if closer else "feature.height.open"), {}
        return ("feature.backness.front" if closer else "feature.backness.back"), {}
    return "sound.unclear", {}


def feature_tip(claim: dict, expected: str, word: str) -> dict:
    key, extra = feature_tip_key(claim)
    P = {"expected": expected, "word": word, "feature": claim["feature"], "heard_class": claim["heard"], **extra}
    return {"tip_key": key, "text": {**TEMPLATES, **FEATURE_TEMPLATES}[key].format(**P), "params": P}


TEMPLATES = {
    "voicing.add": "Let your voice buzz on /{expected}/ in “{word}”; it sounded like /{heard}/, without voice.",
    "voicing.remove": "Say /{expected}/ in “{word}” without voice (no buzz); it sounded like /{heard}/.",
    "aspiration.add": "Add a small puff of air after /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "aspiration.remove": "Say /{expected}/ in “{word}” without a puff of air; it sounded like /{heard}/.",
    "place": "Make /{expected}/ in “{word}” with {expected_place}; it sounded like /{heard}/, made with {heard_place}.",
    "manner.continuant": "Keep the air flowing through /{expected}/ in “{word}” instead of stopping it; it sounded like /{heard}/.",
    "manner.stop": "Close completely and release for /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "manner.affricate": "Start /{expected}/ in “{word}” with a short closure, then let the air hiss; it sounded like /{heard}/.",
    "manner.nasal": "Let the air go through your nose for /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "manner.oral": "Keep the air out of your nose for /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "manner.flap": "Make /{expected}/ in “{word}” a single quick tap of the tongue tip; it sounded like /{heard}/.",
    "manner.trill": "Let the tongue tip vibrate (roll) for /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "manner.other": "Say /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "lateral.add": "Let the air pass along the sides of the tongue for /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "lateral.remove": "Don't let the air flow around the sides of the tongue for /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "palatalisation.add": "Raise the middle of your tongue (a slight “y” colour) on /{expected}/ in “{word}”; it sounded like plain /{heard}/.",
    "palatalisation.remove": "Say /{expected}/ in “{word}” without the “y” colour; it sounded like /{heard}/.",
    "length.longer": "Hold /{expected}/ in “{word}” a little longer; it sounded short.",
    "length.shorter": "Make /{expected}/ in “{word}” shorter; it sounded long.",
    "vowel.open": "Open your mouth a little more for /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "vowel.close": "Raise your tongue a little (mouth less open) for /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "vowel.front": "Move your tongue forward for /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "vowel.back": "Pull your tongue back for /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "vowel.round": "Round your lips for /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "vowel.unround": "Spread your lips (no rounding) for /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "vowel.glide.add": "Glide towards /{glide}/ at the end of /{expected}/ in “{word}”; it sounded like a single /{heard}/.",
    "vowel.glide.remove": "Keep /{expected}/ in “{word}” steady without gliding; it sounded like /{heard}/.",
    "vowel.glide.change": "Glide towards /{glide}/ in /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "nasal.add": "Let some air through your nose on /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "nasal.remove": "Keep /{expected}/ in “{word}” oral (no air through the nose); it sounded like /{heard}/.",
    "rhotic.add": "Add the r-colour to /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "rhotic.remove": "Say /{expected}/ in “{word}” without r-colour; it sounded like /{heard}/.",
    "sound.substitution": "Say /{expected}/ in “{word}”; it sounded like /{heard}/.",
    "sound.unclear": "Listen to /{expected}/ in “{word}” and try it again; it did not sound quite right.",
    "sound.missing": "Don't skip /{expected}/ in “{word}”.",
    "sound.extra": "There seems to be an extra sound in “{word}”; try it without it.",
    "tone.wrong": "“{syllable}” should be {expected_tone} (tone {expected}); it sounded {heard_tone} (tone {heard}).",
    "tone.unclear": "Check the tone of “{syllable}”: it should be {expected_tone} (tone {expected}).",
    "stress.position": "In “{word}”, stress the {expected_ord} syllable; it sounded stressed on the {heard_ord}.",
    "stress.unclear": "In “{word}”, put the stress on the {expected_ord} syllable.",
    "word.mispronounced": "Try “{word}” again; some of its sounds were off.",
    "speech.extra": "There was extra speech {where}; read only the words of the sentence.",
    "speech.restart": "You started “{next_word}” twice; try to say it in one go.",
    "pause.long": "Try not to stop for so long ({duration} s) after “{word}”.",
    "pause.long_punctuation": "Keep the pause after “{word}” shorter ({duration} s).",
    "pause.mid_phrase": "Keep “{word}” and “{next_word}” together; there was a {duration} s pause between them.",
    "pause.in_word": "Say “{word}” in one go, without stopping in the middle.",
    "punctuation.rushed": "Make a short pause at the {mark} after “{word}”.",
    "pitch.flat": "Let your voice rise and fall more; it sounded quite flat.",
    "rate.slow": "Try reading a little faster and more smoothly.",
    "rate.fast": "Slow down a little so that every sound is clear.",
    "word.missing": "The word “{word}” seemed to be missing; try saying every word.",
    "length.vowel_long": "Hold the vowel /{sound}/ in “{word}” for two beats; it sounded short.",
    "length.vowel_short": "Keep the vowel /{sound}/ in “{word}” short (one beat); it sounded long.",
    "length.consonant_long": "Pause briefly before /{sound}/ in “{word}” (a double consonant); it sounded single.",
    "length.consonant_short": "No pause before /{sound}/ in “{word}”; it sounded doubled.",
}


def _ord(k: int) -> str:
    return ORDINAL[k] if 0 <= k < len(ORDINAL) else f"#{k + 1}"


def sound_tip_key(expected: str, heard: str, diff: list[str]) -> tuple[str, dict]:
    """(tip_key, extra params) for one diagnosed substitution: the single most salient
    difference, in a fixed priority order (so a sound never gets two contradictory tips)."""
    a, b = A.features(expected), A.features(heard)
    if not diff or "type" in diff:
        return "sound.substitution", {}
    if a["type"] == "consonant":
        if "manner" in diff:
            ma, mb = a["manner"], b["manner"]
            if ma == "nasal":
                return "manner.nasal", {}
            if mb == "nasal":
                return "manner.oral", {}
            if ma == "affricate":
                return "manner.affricate", {}
            if ma == "flap":
                return "manner.flap", {}
            if ma == "trill":
                return "manner.trill", {}
            if ma in ("fricative", "approximant") and mb in ("stop", "affricate", "flap"):
                return "manner.continuant", {}
            if ma == "stop" and mb in ("fricative", "approximant", "affricate"):
                return "manner.stop", {}
            return "manner.other", {}
        if "place" in diff:
            return "place", {"expected_place": PLACE_WORDS.get(a["place"], a["place"]),
                             "heard_place": PLACE_WORDS.get(b["place"], b["place"])}
        if "voicing" in diff:
            return ("voicing.add" if a["voicing"] == "voiced" else "voicing.remove"), {}
        if "aspirated" in diff:
            return ("aspiration.add" if a["aspirated"] else "aspiration.remove"), {}
        if "lateral" in diff:
            return ("lateral.add" if a["lateral"] else "lateral.remove"), {}
        if "palatalised" in diff:
            return ("palatalisation.add" if a["palatalised"] else "palatalisation.remove"), {}
        if "long" in diff:
            return ("length.longer" if a["long"] else "length.shorter"), {}
        if "rhotic" in diff:
            return ("rhotic.add" if a["rhotic"] else "rhotic.remove"), {}
        return "sound.substitution", {}
    # vowels
    if "glide" in diff and set(diff) <= {"glide", "long"}:     # only the glide differs
        if a["glide"] and not b["glide"]:
            return "vowel.glide.add", {"glide": a["glide"]}
        if b["glide"] and not a["glide"]:
            return "vowel.glide.remove", {}
        return "vowel.glide.change", {"glide": a["glide"]}
    if "height" in diff:
        return ("vowel.open" if A.height_steps(b["height"], a["height"]) > 0 else "vowel.close"), {}
    if "backness" in diff:
        order = A.BACKNESS
        return ("vowel.front" if order.index(a["backness"]) < order.index(b["backness"]) else "vowel.back"), {}
    if "rounded" in diff:
        return ("vowel.round" if a["rounded"] else "vowel.unround"), {}
    if "nasal" in diff:
        return ("nasal.add" if a["nasal"] else "nasal.remove"), {}
    if "long" in diff:
        return ("length.longer" if a["long"] else "length.shorter"), {}
    if "rhotic" in diff:
        return ("rhotic.add" if a["rhotic"] else "rhotic.remove"), {}
    return "sound.substitution", {}


def tip_for(item: dict, syllable_text: str | None = None, words: list[str] | None = None) -> dict:
    k = item["kind"]
    P = {"word": item["word"], "expected": item["expected"], "heard": item["heard"]}
    if k == "missing_word":
        key = "word.missing"
    elif k == "missing":
        key = "sound.missing"
    elif k == "extra":
        key = "sound.extra"
    elif k == "tone":
        P["syllable"] = syllable_text or item["word"]
        P["expected_tone"] = TONE_WORDS.get(item["expected"], str(item["expected"]))
        if item["heard"] is None:
            key = "tone.unclear"
        else:
            key = "tone.wrong"
            P["heard_tone"] = TONE_WORDS.get(item["heard"], str(item["heard"]))
    elif k == "stress":
        P["expected_ord"] = _ord(item["expected"])
        if item["heard"] is None:
            key = "stress.unclear"
        else:
            key = "stress.position"
            P["heard_ord"] = _ord(item["heard"])
    elif k == "mispronounced_word":
        key = "word.mispronounced"
    elif k == "extra_speech":
        W = words or []
        a, b = item.get("after_word", -1), item.get("before_word", len(W))
        nm = (lambda i: (W[i] or "") if 0 <= i < len(W) else "")
        P["next_word"] = nm(b)
        if item.get("restart") and P["next_word"]:
            key = "speech.restart"
        else:
            key = "speech.extra"
            P["where"] = ("at the start" if a < 0 else "at the end" if b >= len(W) or not W
                          else f"between “{nm(a)}” and “{nm(b)}”")
    elif k in ("long_pause", "pause_mid_phrase"):
        W = words or []
        P["duration"] = f"{item['duration']:.1f}"
        P["next_word"] = (W[item["before_word"]] or "") if 0 <= item["before_word"] < len(W) else ""
        if k == "long_pause":
            key = "pause.long_punctuation" if item.get("at_punctuation") else "pause.long"
        else:
            key = "pause.in_word" if item.get("in_word") else "pause.mid_phrase"
    elif k == "rushed_punctuation":
        key = "punctuation.rushed"
        P["mark"] = "full stop" if item["punctuation"] == "stop" else "comma"
    elif k == "length":                  # round 15: expected / heard are "long" / "short"
        from .text.ipa import is_vowel

        ipa = item.get("expected_ipa") or ""
        vowel = is_vowel(ipa)
        P["sound"] = ipa.replace("ː", "") if vowel else (ipa[1:] if len(ipa) > 1 and ipa[0] == ipa[1] else ipa)
        key = f"length.{'vowel' if vowel else 'consonant'}_{item['expected']}"
    elif k == "flat_pitch":
        key = "pitch.flat"
    elif k in ("slow", "fast"):
        key = f"rate.{k}"
    elif item["heard"] is None and item.get("features"):
        return feature_tip(item["features"][0], item["expected"], item["word"])
    elif item["heard"] is None:
        key = "sound.unclear"
    else:
        key, extra = sound_tip_key(item["expected"], item["heard"], item["diff_keys"])
        P.update(extra)
    return {"tip_key": key, "text": TEMPLATES[key].format(**P), "params": P}


PRONUNCIATION_KINDS = ("substitution", "missing", "mispronounced_word")


def _not_accepted(it: dict, items: list[dict], result) -> bool:
    """Round 19 gate: a pronunciation candidate becomes advice only when the scorer itself did not
    accept its sound or its word (prob < 0.5 at the result's current strictness). The sound's
    prob is the item's own when it has one; a mispronounced_word passes when its word, or one of
    the sounds flagged in it, is not accepted."""
    wi = it.get("word_index")
    if wi is None or not 0 <= wi < len(result.words):
        return True
    w = result.words[wi]
    if w.prob is not None and w.prob < 0.5:
        return True
    if it["kind"] == "mispronounced_word":
        sounds = [s for s in items if s["kind"] in ("substitution", "missing") and s.get("word_index") == wi]
    else:
        sounds = [it]
    for s in sounds:
        p = s.get("prob")
        if p is None:
            k = s.get("phone_index")
            p = w.phones[k].prob if k is not None and 0 <= k < len(w.phones) else None
        if p is not None and p < 0.5:
            return True
    return False


def tips(items: list[dict], max_tips: int = 3, result=None, include_accepted: bool = False) -> list[dict]:
    """Up to `max_tips` tips: diagnosed items first in coaching order, then flag-only items.
    `result` (optional SentenceResult) supplies syllable text for tone tips and, round 19, gates
    the pronunciation candidates (substitution, missing, mispronounced_word, including feature
    and contrast claims): such an item gives a tip only when its sound or its word has prob < 0.5
    in `result` (so the gate follows Scorer.rescore(strictness=...)). include_accepted=True, or no
    `result`, turns the gate off. Other kinds (missing_word, extra_speech, delivery, tone, stress,
    length) are never gated: they have their own calibrated thresholds."""
    if result is not None and not include_accepted:
        items = [it for it in items if it["kind"] not in PRONUNCIATION_KINDS or _not_accepted(it, items, result)]

    def target(it):
        return (it["word_index"], it["kind"] if it["word_index"] is None else it["kind"] in ("tone", "stress"),
                it["syllable_index"], it["phone_index"])

    diagnosed = [it for it in items if it["heard"] is not None or it.get("features")
                 or it["kind"] in ("missing", "missing_word", "extra")]
    diagnosed = [it for it in diagnosed if it["kind"] != "mispronounced_word"]
    rest = [it for it in items if it not in diagnosed]
    out, seen, words_done, kinds_done = [], set(), set(), set()
    words = [w.text for w in result.words] if result is not None else (
        [None] * (1 + max([i["word_index"] for i in items if i["word_index"] is not None] or [-1])))
    for it in items:                     # word texts for tips that name neighbouring words
        if it["word_index"] is not None and words[it["word_index"]] is None:
            words[it["word_index"]] = it["word"]
    for it in diagnosed + rest:
        if len(out) >= max_tips:
            break
        t = target(it)
        if t in seen or (it["word_index"] is not None and it["word_index"] in words_done) \
                or (it["kind"] in kinds_done and it["word_index"] is None):
            continue                     # one tip per sound, per word and per utterance-level kind
        seen.add(t)
        if it["word_index"] is not None:
            words_done.add(it["word_index"])
        kinds_done.add(it["kind"])
        syl = None
        if result is not None and it["kind"] == "tone":
            syl = result.words[it["word_index"]].syllables[it["syllable_index"]].text or None
        tp = tip_for(it, syl, words)
        tp["item"] = it
        out.append(tp)
    return out
