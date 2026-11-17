"""Azure-format output: a pronunciada result in the JSON *shape* of Microsoft Azure Speech
"Pronunciation Assessment" detailed results, for apps that already parse that format.

    from pronunciada.compat.azure import to_azure
    doc = to_azure(result)                       # Speech SDK layout (nested "PronunciationAssessment")
    doc = to_azure(result, layout="rest")        # REST short-audio layout (scores flat on each entry)

Only the shape is compatible. Every number is pronunciada's own (calibrated probabilities and
regression scores rescaled to 0-100), computed by a different model with different training
data; it is not Azure's output, not identical to it and not endorsed by Microsoft. Score
distributions differ, so thresholds tuned on Azure scores must be re-chosen on pronunciada
output, not copied. Field-by-field mapping: docs/AZURE_COMPAT.md.

Mapping (HundredMark; FivePoint divides by 20):
  NBest[0] AccuracyScore      100 x result.scores["accuracy"]  (speechocean762-style 0-10 / 10);
                              fallback 100 x mean word prob
  NBest[0] FluencyScore       100 x result.scores["fluency"] (= result.fluency)
  NBest[0] CompletenessScore  100 x result.completeness
  NBest[0] ProsodyScore       100 x result.scores["prosodic"] (omitted when absent)
  NBest[0] PronScore          100 x result.scores["total"]; fallback 100 x result.prob
  Word AccuracyScore          100 x WordResult.prob (a probability, not a rating)
  Word ErrorType              Omission (omitted / missing_word), Mispronunciation (a
                              mispronounced_word coaching item, else prob < 0.5), Insertion
                              (extra_speech items, as extra entries), else None
  Syllable AccuracyScore      100 x mean prob of the syllable's sounds
  Phoneme AccuracyScore       100 x PhoneResult.prob; Phoneme = our IPA symbol
  NBestPhonemes               expected sound (Score 100 x prob) then the heard sound, when one
                              was named (Score 100 x (1 - prob))
  Offset / Duration           seconds x 10^7 (100-ns ticks), from our alignment
"""
from __future__ import annotations

TICKS = 10_000_000
GRANULARITIES = ("Phoneme", "Word", "FullText")
LOCALES = {"en": "en-US", "zh": "zh-CN", "ja": "ja-JP", "es": "es-ES", "fr": "fr-FR", "de": "de-DE",
           "it": "it-IT", "pt": "pt-BR", "ru": "ru-RU", "ko": "ko-KR"}
NO_SPACE = ("zh", "ja")


def _ticks(sec) -> int:
    return max(0, int(round(float(sec or 0.0) * TICKS)))


def _scale(p, factor: float) -> float | None:
    if p is None:
        return None
    return round(min(1.0, max(0.0, float(p))) * factor, 1)


def _miscue_items(result, coaching_items):
    if coaching_items is not None:
        return coaching_items
    if getattr(result, "_pronunciada", None) is None:
        return None
    try:
        return result.coaching(sensitivity="balanced")
    except (ValueError, RuntimeError):
        return None


def _syllable_phones(result, wi: int, w) -> list[list[int]]:
    """Phone indices (within the word) of each syllable of word wi."""
    nsy = len(w.syllables)
    c = getattr(result, "_pronunciada", None)
    if c is not None:
        ref = c["ref"]
        s0 = ref.words[wi].syllables[0]
        p0 = ref.words[wi].phones[0]
        groups = [[] for _ in range(nsy)]
        for k in range(len(w.phones)):
            si = ref.phones[p0 + k].syllable - s0
            if 0 <= si < nsy:
                groups[si].append(k)
        return groups
    # rebuilt result (no reference): assign each sound to the syllable holding its midpoint
    groups = [[] for _ in range(nsy)]
    for k, p in enumerate(w.phones):
        mid = 0.5 * (p.start + p.end)
        best = min(range(nsy), key=lambda j: (0 if w.syllables[j].start <= mid <= w.syllables[j].end
                                              else min(abs(mid - w.syllables[j].start),
                                                       abs(mid - w.syllables[j].end))))
        groups[best].append(k)
    return groups


def _pa(layout: str, target: dict, scores: dict) -> dict:
    scores = {k: v for k, v in scores.items() if v is not None}
    if layout == "rest":
        target.update(scores)
    else:
        target["PronunciationAssessment"] = scores
    return target


def to_azure(result, granularity: str = "Phoneme", enable_miscue: bool = True, phoneme_alphabet: str = "IPA",
             *, layout: str = "sdk", grading_system: str = "HundredMark", nbest_phoneme_count: int = 5,
             coaching_items: list | None = None) -> dict:
    """A dict shaped like Azure Pronunciation Assessment's detailed result (see module docstring;
    the numbers are pronunciada's). layout "sdk": scores nested in "PronunciationAssessment"
    (the Speech SDK JSON); "rest": scores flat on NBest[0] / words / phonemes (the REST
    short-audio API). coaching_items: reuse result.coaching() output (computed if omitted and
    the result carries its evidence)."""
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}")
    if layout not in ("sdk", "rest"):
        raise ValueError('layout must be "sdk" or "rest"')
    if str(phoneme_alphabet).upper() != "IPA":
        raise ValueError("only phoneme_alphabet='IPA' is available (pronunciada has no SAPI phone set)")
    if grading_system not in ("HundredMark", "FivePoint"):
        raise ValueError('grading_system must be "HundredMark" or "FivePoint"')
    f = 100.0 if grading_system == "HundredMark" else 5.0
    words = list(result.words or [])
    sc = result.scores or {}
    items = _miscue_items(result, coaching_items) if enable_miscue else None
    mispron = {it["word_index"] for it in items or [] if it.get("kind") == "mispronounced_word"}
    missing = {it["word_index"] for it in items or [] if it.get("kind") == "missing_word"}
    extra = [it for it in items or [] if it.get("kind") == "extra_speech"] if enable_miscue else []

    mean_word = sum(w.prob for w in words) / len(words) if words else None
    nb = {"Lexical": ("" if result.lang in NO_SPACE else " ").join(
              w.text.lower() for w in words if not (enable_miscue and w.omitted)),
          "Display": result.text}
    _pa(layout, nb, {
        "AccuracyScore": _scale(sc.get("accuracy", mean_word), f),
        "FluencyScore": _scale(sc.get("fluency", result.fluency), f),
        "CompletenessScore": _scale(result.completeness, f),
        "ProsodyScore": _scale(sc.get("prosodic"), f),
        "PronScore": _scale(sc.get("total", result.prob), f),
    })

    if granularity != "FullText":
        out_words = []
        inserts: dict[int, list] = {}
        for it in extra:
            inserts.setdefault(int(it.get("after_word", len(words) - 1)), []).append(it)

        def add_inserts(k):
            for it in inserts.get(k, []):
                st, en = float(it.get("start") or 0.0), float(it.get("end") or 0.0)
                d = {"Word": (it.get("heard") or it.get("heard_ipa") or "").strip(),
                     "Offset": _ticks(st), "Duration": _ticks(max(0.0, en - st))}
                out_words.append(_pa(layout, d, {"AccuracyScore": 0.0, "ErrorType": "Insertion"}))

        add_inserts(-1)
        for wi, w in enumerate(words):
            omitted = bool(w.omitted) or wi in missing
            if omitted and enable_miscue:
                et = "Omission"
            elif wi in mispron or (items is None and w.prob < 0.5) or (omitted and not enable_miscue):
                et = "Mispronunciation"
            else:
                et = "None"
            d = {"Word": w.text, "Offset": _ticks(w.start), "Duration": _ticks(max(0.0, w.end - w.start))}
            _pa(layout, d, {"AccuracyScore": _scale(w.prob, f), "ErrorType": et})
            if granularity == "Phoneme" and not (omitted and enable_miscue):
                if w.syllables:
                    syls = []
                    for sy, ks in zip(w.syllables, _syllable_phones(result, wi, w)):
                        ps = [w.phones[k] for k in ks]
                        s = {"Syllable": "".join(p.ipa for p in ps), "Offset": _ticks(sy.start),
                             "Duration": _ticks(max(0.0, sy.end - sy.start))}
                        if sy.text:
                            s["Grapheme"] = sy.text
                        acc = sum(p.prob for p in ps) / len(ps) if ps else None
                        syls.append(_pa(layout, s, {"AccuracyScore": _scale(acc, f)}))
                    d["Syllables"] = syls
                phs = []
                for p in w.phones:
                    q = {"Phoneme": p.ipa, "Offset": _ticks(p.start), "Duration": _ticks(max(0.0, p.end - p.start))}
                    cand = [{"Phoneme": p.ipa, "Score": _scale(p.prob, f)}]
                    if p.heard:
                        cand.append({"Phoneme": p.heard, "Score": _scale(1.0 - p.prob, f)})
                        cand.sort(key=lambda x: -x["Score"])
                    scores = {"AccuracyScore": _scale(p.prob, f)}
                    if nbest_phoneme_count and nbest_phoneme_count > 0:
                        scores["NBestPhonemes"] = cand[:nbest_phoneme_count]
                    phs.append(_pa(layout, q, scores))
                d["Phonemes"] = phs
            out_words.append(d)
            add_inserts(wi)
        nb["Words"] = out_words

    spans = [(w.start, w.end) for w in words if w.end > w.start]
    t0 = min(a for a, _ in spans) if spans else 0.0
    t1 = max(b for _, b in spans) if spans else 0.0
    doc = {"RecognitionStatus": "Success", "Offset": _ticks(t0), "Duration": _ticks(max(0.0, t1 - t0)),
           "DisplayText": result.text}
    au = result.audio or {}
    if au.get("snr_db") is not None:
        doc["SNR"] = round(float(au["snr_db"]), 2)
    doc["NBest"] = [nb]
    return doc


def parse_header(value: str) -> dict:
    """Decode the `Pronunciation-Assessment` request header (base64 JSON) -> dict."""
    import base64
    import json

    raw = base64.b64decode(value.strip() + "=" * (-len(value.strip()) % 4))
    d = json.loads(raw.decode("utf-8"))
    if not isinstance(d, dict):
        raise ValueError("Pronunciation-Assessment header must be a JSON object")
    return d


def _bool(v, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def options_from_header(params: dict) -> dict:
    """to_azure keyword arguments from the decoded header (ReferenceText is read separately)."""
    gran = params.get("Granularity", "Phoneme")
    gran = {"phoneme": "Phoneme", "word": "Word", "fulltext": "FullText"}.get(str(gran).lower(), gran)
    return {"granularity": gran, "enable_miscue": _bool(params.get("EnableMiscue"), False),
            "phoneme_alphabet": params.get("PhonemeAlphabet", "IPA"),
            "grading_system": params.get("GradingSystem", "HundredMark"),
            "nbest_phoneme_count": int(params.get("NBestPhonemeCount", 5) or 0)}
