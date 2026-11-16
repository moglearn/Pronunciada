# Pronunciada

On-device pronunciation checking for ten languages. Give it a recording and the text the speaker
was meant to say; it returns a 0–1 probability that each **sound**, **word** and the whole
**sentence** was pronounced acceptably, plus tones (Mandarin), word stress, fluency and which
words were skipped. One `strictness` knob moves every probability between "understandable" and
"native-like".

It runs locally on a laptop CPU: about 50 ms for a 3.5 s recording, 0.3–0.5 s for a 10–14 s one.
No audio leaves the device.

Languages: English, Mandarin Chinese (Simplified and Traditional), Japanese, Spanish (Latin
American), French, German, Italian, Brazilian Portuguese, Russian, Korean.

```python
import pronunciada

scorer = pronunciada.Scorer("es")                     # loads the model once
res = scorer.score("clip.wav", "Se recomienda a los viajeros que se informen.")

res.prob                      # whole sentence, 0–1
for w in res.words:
    w.text, w.prob            # word: sounds × stress/tone × "was it said at all"
    w.sounds_prob, w.stress_prob, w.tone_prob, w.present_prob, w.omitted
    for p in w.phones:
        p.ipa, p.prob, p.heard, p.start, p.end        # each sound, and what it sounded like instead

strict = scorer.rescore(res, strictness=0.9)          # re-grade without re-running the model
```

```bash
python -m pronunciada score clip.wav "The quick brown fox" --lang en --strictness 0.5
```

## Coaching feedback

`res.coaching(sensitivity="balanced")` lists the speaker's clearest real errors, most important
first: what was expected, what it sounded like (named only when that is reliable), which
articulatory features differ, and how sure we are. `pronunciada.tips.tips(items)` writes up to
three short English tips with stable `tip_key`s for an app's own copy. `res.audio` reports
recording problems (`clipped`, `too_noisy`, `too_quiet`, `too_short`, `mostly_silence`) and
`retry_recommended`. No score changes; coaching costs under 2 ms.

```bash
python -m pronunciada score clip.wav "FIRST PLACE IN THE DIVISION" --lang en --coach --sensitivity balanced
```

The sound operating point is fitted on English learners (speechocean762 train). On held-out
speechocean762 test, "balanced" flags sounds that experts also marked as errors 65 % of the time
and finds 33 % of all marked errors ("careful": 75 % / 20 %; round 1 found 18 %). When it names
the sound it heard on a real substitution, the name is right within one articulatory feature
77 % of the time. For example, on the test utterance “FIRST PLACE IN THE DIVISION”, the experts marked the /v/ of “division” as pronounced /f/; Pronunciada flags /v/ → /f/ (and
/t/ in “first” without naming it). Mandarin tones are calibrated on OMPAL learners but only the
most certain few percent reach 55 % precision on test; Mandarin consonants / vowels, missing
sounds and all other languages are not verified on learners. Details:
[docs/COACHING.md](docs/COACHING.md), [results/coaching.md](results/coaching.md).

## Install

```bash
pip install -e .
```

```bash
pip install -r requirements-text-eu.txt -r requirements-text-cjk.txt
```

The acoustic model ([anyspeech/zipa-small-crctc-500k](https://huggingface.co/anyspeech/zipa-small-crctc-500k),
64 M parameters, CC BY 4.0) is downloaded on first use: 248 MB, or 67 MB with
`Scorer(lang, precision="int8")`. Everything else this project trained (per-language adapters,
calibrators, tone and stress models) is in the package and totals about 8 MB; the pronunciation
dictionaries add 10 MB.

## How it compares with Microsoft Azure Pronunciation Assessment

Every number below is measured on held-out data through the public API shown above, with Azure
called on the same recordings and prompts (November 2026, region westus). Full tables with
confidence intervals are in [results/](results/).

### English: agreement with human raters (speechocean762 test, 2,500 recordings of learners)

Correlation with the human scores, higher is better.

| level | Pronunciada | Azure | |
|---|---|---|---|
| individual sounds | **0.660** | 0.538 | ahead (p = 0.002) |
| words | **0.609** | 0.589 | ahead, not significant (p = 0.33) |
| sentence accuracy | **0.761** | 0.673 | ahead (p = 0.002) |
| sentence fluency | **0.792** | 0.708 | ahead, borderline (p = 0.05) |
| sentence intonation ("prosodic") | 0.782 | **0.827** | behind, not significant (p = 0.16) |
| sentence overall | 0.773 | **0.796** | behind, not significant (p = 0.41) |

The sentence scores use delivery inputs added in round 5 ([results/round5.md](results/round5.md)); the word score uses word-pooled model activations since round 6 ([results/round6.md](results/round6.md)).

### English: finding real learner mistakes (area under the ROC curve, higher is better)

| data | Pronunciada | Azure |
|---|---|---|
| L2-ARCTIC test (expert-marked wrong sounds) | **0.712** | 0.631 |
| EpaDB test (Spanish speakers of English) | **0.721** | 0.587 |

### All ten languages: the same native recording, one word of the prompt changed

There is no public learner corpus with expert marks for most of these languages, so this is a
controlled stand-in: take a native recording (FLEURS test), replace one word of the prompt with a
clearly different word ("wrong word") or with a word one sound away ("near-miss"), and see whether
the system flags that word. 150 recordings per language; cells are **Pronunciada / Azure**, bold
where Pronunciada is ahead.

| language | wrong word: detection | wrong word: right word blamed | near-miss word: detection | near-miss: caught at 5 % false alarms | native words flagged (< 0.6) |
|---|---|---|---|---|---|
| en | **0.995** / 0.994 | **0.900** / 0.898 | **0.981** / 0.963 | **0.900** / 0.793 | 0.034 / 0.022 |
| zh | **0.991** / 0.989 | 0.887 / 0.899 | 0.964 / 0.966 | 0.860 / 0.896 | 0.046 / 0.036 |
| ja | **0.991** / 0.979 | **0.920** / 0.875 | **0.965** / 0.919 | **0.840** / 0.680 | 0.025 / 0.020 |
| es | **0.998** / 0.993 | **0.953** / 0.910 | **0.988** / 0.942 | **0.927** / 0.679 | 0.024 / 0.018 |
| fr | **0.997** / 0.986 | **0.920** / 0.823 | **0.986** / 0.937 | **0.947** / 0.647 | 0.026 / 0.020 |
| de | **0.996** / 0.982 | 0.913 / 0.921 | **0.953** / 0.887 | **0.773** / 0.517 | **0.044** / 0.049 |
| it | **0.998** / 0.990 | **0.987** / 0.929 | **0.971** / 0.970 | 0.833 / 0.913 | **0.025** / 0.037 |
| pt | **0.997** / 0.991 | **0.960** / 0.906 | **0.956** / 0.938 | **0.827** / 0.724 | 0.027 / 0.020 |
| ru | **0.990** / 0.982 | **0.907** / 0.761 | 0.901 / 0.909 | **0.607** / 0.342 | **0.050** / 0.054 |
| ko | 0.994 / 0.995 | 0.960 / 0.973 | **0.941** / 0.937 | **0.767** / 0.708 | **0.021** / 0.045 |

- Detection = area under the ROC curve. "Right word blamed" = the changed word gets the lowest
  score in the sentence. "Caught at 5 % false alarms" puts both systems at the threshold where
  each flags 5 % of the untouched words, so it is the fairest single column.
- The last column is each system at its own natural cut-off (probability 0.6 / score 60), lower is
  better; Azure is a little more forgiving of native speakers in six languages.
- Where Pronunciada is behind: Russian near-misses overall (0.90 against 0.91, although it catches
  more of them at 5 % false alarms), Italian and Mandarin near-misses at 5 % false alarms, and by
  about one point blaming the right word in German, Korean and Mandarin. Round 8 changed
  Russian in three ways: measured, reduction-aware vowel targets; an adapter retrained on them;
  and a stricter sound bias (+1.08 → +0.60, chosen on dev). Near-swap AUC rose 0.872 → 0.887 on
  test, which still trails Azure. cv-real ru got worse (0.793 → 0.772). Round 10 added an offline
  stress lexicon for 14 040 word forms (silero-stress, used offline only) and set the bias to 0.0
  by the same rule. Near-swap AUC is now 0.901 on test, still short of Azure's 0.909. cv-real ru
  is back to 0.788 (round 7: 0.793), at a cost of 0.7 points of native word acceptance on test
  ([results/round8_ru.md](results/round8_ru.md)).

### Mandarin tones

Same syllable, different tone in the prompt (so only the pitch tells them apart), FLEURS test:

| | Pronunciada | Azure |
|---|---|---|
| detection (AUC) | 0.845 | 0.846 |
| caught at 5 % false alarms | 37 % | **40 %** |

Real learners (OMPAL test: French learners of Taiwan Mandarin, every character marked by
teachers): word-level AUC for any error 0.795 vs Azure 0.704; tone-only errors 0.785 vs 0.679;
sentence accuracy Pearson with teachers 0.64 vs 0.36. Caveat: since round 7, Mandarin sentence
scores are recalibrated on OMPAL's DEV learners (same first language, different speakers), while
Azure is untuned ([results/round7.md](results/round7.md)).

The default tone model (`tone_zh_v3z`, 0.8 MB) reads the pitch contour, a low-frequency spectrum
patch and the acoustic model's per-syllable activations. It agrees with the dictionary tone on
76 % of syllables of fast crowd-sourced read speech (FLEURS) and 86 % of studio speech. Because
it partly recognises the syllable, it recovers a deliberately wrong pitch contour slightly less
often than the pitch-only v2 on synthetic tests. On real learner tone errors it is as good or
better (OMPAL dev). `PRONUNCIADA_TONE_MODEL=tone_zh_v2` selects the pitch-only model
([results/tone_v3.md](results/tone_v3.md)). Simplified and Traditional text give identical
references; Taiwan-accented speech is tested separately ([results/prosody.md](results/prosody.md)).

### Word stress (English, German, Spanish, Italian, Portuguese, Russian)

Azure offers no comparable per-word stress check in these languages, so there is no head-to-head.
On FLEURS test, a word whose stress is moved to another syllable is detected with AUC 0.92–0.95
(Russian 0.87). French and Korean have no word stress and nothing is scored there. Japanese pitch
accent is reported per syllable but is experimental and does not affect the word score.

### Native speakers are not punished

On native FLEURS test recordings at the default strictness, 99.3–99.9 % of sounds, 96.8–98.4 % of
words and 99.0–99.8 % of sentences are accepted in every language.

### Calibration

A probability of 0.9 should be right nine times in ten. On speechocean762 test the expected
calibration error is 1–2 % for sounds, 3–4 % for words and 2–4 % for sentences
([results/calibration.md](results/calibration.md)).

### Speed and size

| | Pronunciada (laptop CPU) | Azure (network request) |
|---|---|---|
| 3.5 s recording | 49 ms (38 ms with the int8 model) | – |
| 10–14 s recording | 0.32–0.50 s | 1.8–3.0 s |

## What is not done, or not good enough yet

- **Sentence-level intonation** in English trails Azure (0.77 against 0.83), which also pulls the
  overall sentence score slightly below Azure's.
- **Mandarin tones**: level with Azure on the FLEURS tone-swap AUC but still behind at 5 % false alarms.
- **Real learner mistakes outside English are untested**: the ten-language comparison uses native
  recordings with altered prompts. Detection of real mistakes is much harder (0.71–0.72 in English
  against 0.98–0.99 on the controlled test) and the same drop should be expected elsewhere.
- **Japanese pitch accent** is not reliable enough to grade.
- **Mobile**: everything is ONNX plus numpy, so it is portable, but no iOS / Android bindings have
  been built and nothing has been timed on a phone.
- Text front-ends handle ordinary prose; numbers, abbreviations and names outside the dictionaries
  fall back to spelling rules and are listed in `result.warnings`.

## How it works, briefly

1. The prompt is converted to the sounds a native speaker would produce (dictionaries plus rules
   per language, including tone changes in Mandarin and liaison in French).
2. A multilingual phone-recognition model (ZIPA) listens to the recording; a small per-language
   adapter corrects its habits for that language.
3. For every expected sound, the scorer compares "the speaker said this" with "the speaker said
   something else" and "the speaker skipped it", without forcing a rigid alignment.
4. Small calibration models trained on human-rated speech turn that evidence into probabilities
   for sounds, words and the sentence; separate small models judge tone and stress from pitch.

Design and file layout: [docs/DESIGN.md](docs/DESIGN.md). Benchmarks and how to re-run them:
[docs/benchmarks_data.md](docs/benchmarks_data.md), [results/](results/). The benchmarks were run
in four rounds; rounds 2–4 were designed after looking at an earlier held-out evaluation and are
marked as such in the result files.

## Licence

Code: Apache-2.0. The acoustic model is CC BY 4.0. The Italian, Portuguese and Russian
pronunciation files are derived from Wiktionary and stay under CC BY-SA. See [NOTICE](NOTICE).
