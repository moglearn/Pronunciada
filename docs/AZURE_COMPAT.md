# Azure-format output

`pronunciada.compat.azure` converts a Pronunciada result to the JSON layout returned by
Microsoft Azure Speech Pronunciation Assessment. `pronunciada serve --azure-compat` accepts the
Azure short-audio REST request and answers in that layout, so an existing client can be pointed
at it by changing the base URL.

Only the layout is compatible. The values are Pronunciada's calibrated probabilities and
regression scores, rescaled to 0–100. They come from a different model trained on different
data, and their distributions differ from Azure's. A threshold chosen on Azure output (for
example "accuracy < 60 means mispronounced") has to be chosen again on Pronunciada output.
This project is not affiliated with or endorsed by Microsoft.

## Python

```python
import pronunciada
from pronunciada.compat.azure import to_azure

res = pronunciada.score("clip.wav", "The quick brown fox jumps over the lazy dog", lang="en")
doc = to_azure(res)                 # granularity="Phoneme", enable_miscue=True, phoneme_alphabet="IPA"
doc["NBest"][0]["PronunciationAssessment"]
# {"AccuracyScore": 88.8, "FluencyScore": 92.8, "CompletenessScore": 100.0, "ProsodyScore": 89.2, "PronScore": 85.6}
doc["NBest"][0]["Words"][0]
# {"Word": "The", "Offset": 500000, "Duration": 400000,
#  "PronunciationAssessment": {"AccuracyScore": 58.7, "ErrorType": "Mispronunciation"},
#  "Syllables": [{"Syllable": "ðə", ..., "PronunciationAssessment": {"AccuracyScore": 83.4}}],
#  "Phonemes": [{"Phoneme": "ð", "Offset": 500000, "Duration": 200000,
#                "PronunciationAssessment": {"AccuracyScore": 87.0, "NBestPhonemes": [...]}}, ...]}
```

Arguments:

| argument | values | |
|---|---|---|
| `granularity` | `Phoneme` (default), `Word`, `FullText` | as Azure: words + syllables + phonemes / words only / sentence only |
| `enable_miscue` | `True` (default) | Omission / Insertion tags; with `False` only `None` / `Mispronunciation` |
| `phoneme_alphabet` | `IPA` | `SAPI` is not available (ValueError) |
| `layout` | `sdk` (default), `rest` | `sdk`: scores nested in `PronunciationAssessment` (Speech SDK JSON); `rest`: scores flat on each entry (REST short-audio JSON) |
| `grading_system` | `HundredMark` (default), `FivePoint` | FivePoint = HundredMark / 20 |
| `nbest_phoneme_count` | 5 | 0 drops `NBestPhonemes` |
| `coaching_items` | `None` | reuse `result.coaching()` output; computed when the result carries its evidence |

## Field-by-field mapping

"Ours" names `SentenceResult` fields (`pronunciada/types.py`). `x100` means the 0–1 value
times 100, rounded to 0.1.

| Azure field | status | pronunciada source |
|---|---|---|
| `RecognitionStatus` | present | always `"Success"` (errors are HTTP errors instead) |
| `Offset`, `Duration` (top level) | present | first word start, last word end − first start; seconds × 10⁷ (100-ns ticks) |
| `DisplayText` | approximated | the reference text: pronunciada does not transcribe |
| `SNR` | approximated | `audio["snr_db"]` of the audio-quality gate (a different estimator) |
| `NBest[0].Lexical` | approximated | the reference words, lower-cased, omitted words left out |
| `NBest[0].Display` | approximated | the reference text |
| `NBest[0].Confidence`, `ITN`, `MaskedITN` | not available | (no recognizer) |
| `AccuracyScore` (sentence) | present | `scores["accuracy"]` x100 (the speechocean762-style 0–10 accuracy / 10); without it the mean word prob x100 |
| `FluencyScore` | present | `scores["fluency"]` x100 (= `fluency`) |
| `CompletenessScore` | present | `completeness` x100 (share of words said) |
| `ProsodyScore` | present where we have one | `scores["prosodic"]` x100; omitted when absent (Azure has it for en-US only; ours exists for every language with sentence scores) |
| `PronScore` | approximated | `scores["total"]` x100 (our total, not Azure's weighted combination); without it `prob` x100 |
| `Words[].Word` | present | the prompt word as written |
| `Words[].Offset`, `Duration` | present | our alignment, ticks |
| `Words[].AccuracyScore` | approximated | `WordResult.prob` x100: a probability that the word is acceptable, not a rating; it falls steeply on errors |
| `Words[].ErrorType` | present | `Omission`: `omitted` or a `missing_word` coaching item; `Mispronunciation`: a `mispronounced_word` coaching item (without coaching evidence: prob < 0.5); `Insertion`: an `extra_speech` coaching item; else `None`. A word can have a low `AccuracyScore` and `ErrorType` `None` when no coaching item names an error in it: the coaching gate is tuned to keep false alarms on native speech low |
| Insertion entries | approximated | one extra word per `extra_speech` item, `Word` = the heard IPA, `AccuracyScore` 0 |
| `Words[].Confidence` | not available | |
| `Words[].Feedback.Prosody` (Break, Intonation) | not available | delivery flags are in our coaching items instead |
| `Syllables[]` | present where we have them | from `WordResult.syllables`; `Syllable` = the concatenated IPA of its sounds, `Grapheme` = the syllable text when known (CJK) |
| `Syllables[].AccuracyScore` | approximated | mean `prob` of its sounds x100 |
| `Phonemes[].Phoneme` | present | our IPA symbol (ZIPA conventions, e.g. `ɹ`, `tʰ`, `aʊ`) |
| `Phonemes[].Offset`, `Duration` | present | our alignment, ticks |
| `Phonemes[].AccuracyScore` | present | `PhoneResult.prob` x100 |
| `Phonemes[].NBestPhonemes[]` | approximated | the expected sound with `Score` = prob x100, then the heard sound (`PhoneResult.heard`, named only when prob < 0.5) with `Score` = (1 − prob) x100; ordered by `Score` |
| tone (Mandarin), stress, length | not in the Azure shape | see the native JSON (`SyllableResult.tone_*`, `stress_*`, `PhoneResult.length_*`) |

## REST short-audio compatibility (`serve --azure-compat`)

```
pronunciada serve --azure-compat
```

The server answers
`POST /speech/recognition/{conversation|interactive|dictation}/cognitiveservices/v1?language=<locale>&format=detailed`
with the `Pronunciation-Assessment` header (base64 JSON). It reads `ReferenceText` (required:
pronunciada scores against a known text), `Granularity`, `EnableMiscue` (Azure's default,
false, applies when absent), `GradingSystem`, `PhonemeAlphabet` (`IPA` when absent; `SAPI` →
400) and `NBestPhonemeCount`; `Dimension` and `EnableProsodyAssessment` are accepted and
ignored (`ProsodyScore` is included whenever we have one). Raw and chunked bodies both work.
The reply is the `layout="rest"` document. A subscription-key header is ignored and never
logged. The server has no authentication: see [SERVER.md](SERVER.md).

A client written against the Azure REST API, with only the base URL changed:

```python
import base64, json, urllib.request

BASE_URL = "http://127.0.0.1:8765"   # was: https://<region>.stt.speech.microsoft.com
params = {"ReferenceText": "The quick brown fox jumps over the lazy dog",
          "GradingSystem": "HundredMark", "Granularity": "Phoneme",
          "Dimension": "Comprehensive", "EnableMiscue": True, "PhonemeAlphabet": "IPA"}
req = urllib.request.Request(
    BASE_URL + "/speech/recognition/conversation/cognitiveservices/v1?language=en-US&format=detailed",
    data=open("clip.wav", "rb").read(),
    headers={"Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
             "Accept": "application/json",
             "Pronunciation-Assessment": base64.b64encode(json.dumps(params).encode()).decode()})
best = json.load(urllib.request.urlopen(req))["NBest"][0]
print("PronScore", best["PronScore"], "Accuracy", best["AccuracyScore"], "Fluency", best["FluencyScore"])
for w in best["Words"][:3]:
    print(w["Word"], w["AccuracyScore"], w["ErrorType"], [(p["Phoneme"], p["AccuracyScore"]) for p in w["Phonemes"]])
```

```
PronScore 85.6 Accuracy 88.8 Fluency 92.8
The 58.7 Mispronunciation [('ð', 87.0), ('ə', 79.7)]
quick 98.9 None [('k', 99.6), ('w', 99.7), ('ɪ', 99.8), ('k', 99.8)]
brown 99.3 None [('b', 99.8), ('ɹ', 99.5), ('aʊ', 99.6), ('n', 99.8)]
```

Differences a client may notice: no `Confidence` / `ITN` / `MaskedITN`; `DisplayText` is the
reference, not a transcript; words follow the prompt's segmentation (Mandarin and Japanese:
our word segmentation, not Azure's); phoneme symbols are IPA in ZIPA conventions; the
`language` locale only selects the language (`es-ES` and `es-MX` both score as `es`).
