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
    "word.missing": "The word “{word}” seemed to be missing; try saying every word.",
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


def tip_for(item: dict, syllable_text: str | None = None) -> dict:
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
    elif item["heard"] is None:
        key = "sound.unclear"
    else:
        key, extra = sound_tip_key(item["expected"], item["heard"], item["diff_keys"])
        P.update(extra)
    return {"tip_key": key, "text": TEMPLATES[key].format(**P), "params": P}


def tips(items: list[dict], max_tips: int = 3, result=None) -> list[dict]:
    """Up to `max_tips` tips: diagnosed items first in coaching order, then flag-only items.
    `result` (optional SentenceResult) supplies syllable text for tone tips."""
    def target(it):
        return (it["word_index"], it["kind"] in ("tone", "stress"), it["syllable_index"],
                it["phone_index"])

    diagnosed = [it for it in items if it["heard"] is not None or it["kind"] in ("missing", "missing_word", "extra")]
    rest = [it for it in items if it not in diagnosed]
    out, seen, words_done = [], set(), set()
    for it in diagnosed + rest:
        if len(out) >= max_tips:
            break
        t = target(it)
        if t in seen or it["word_index"] in words_done:
            continue                     # one tip per sound, and per word (no pile-ups)
        seen.add(t)
        words_done.add(it["word_index"])
        syl = None
        if result is not None and it["kind"] == "tone":
            syl = result.words[it["word_index"]].syllables[it["syllable_index"]].text or None
        tp = tip_for(it, syl)
        tp["item"] = it
        out.append(tp)
    return out
