# Pronunciada

On-device pronunciation checking for ten languages. Give it a recording and the text the speaker
was meant to say; it returns a 0–1 probability that each **sound**, **word** and the whole
**sentence** was pronounced acceptably, plus tones (Mandarin), word stress, fluency and which
words were skipped. One `strictness` knob moves every probability between "understandable" and
"native-like".

It runs locally on a laptop CPU: about 53 ms for a 3.5 s recording, 0.3–0.5 s for a 10–14 s one.
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
| words | 0.585 | 0.589 | tie |
| sentence accuracy | **0.761** | 0.673 | ahead (p = 0.002) |
| sentence fluency | **0.792** | 0.708 | ahead, borderline (p = 0.05) |
| sentence intonation ("prosodic") | 0.782 | **0.827** | behind, not significant (p = 0.16) |
| sentence overall | 0.773 | **0.796** | behind, not significant (p = 0.41) |

The sentence scores use delivery inputs added in round 5 ([results/round5.md](results/round5.md)).

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
| ru | **0.991** / 0.982 | **0.913** / 0.761 | 0.872 / 0.909 | **0.580** / 0.342 | **0.048** / 0.054 |
| ko | 0.994 / 0.995 | 0.960 / 0.973 | **0.941** / 0.937 | **0.767** / 0.708 | **0.021** / 0.045 |

- Detection = area under the ROC curve. "Right word blamed" = the changed word gets the lowest
  score in the sentence. "Caught at 5 % false alarms" puts both systems at the threshold where
  each flags 5 % of the untouched words, so it is the fairest single column.
- The last column is each system at its own natural cut-off (probability 0.6 / score 60), lower is
  better; Azure is a little more forgiving of native speakers in six languages.
- Where Pronunciada is behind: Russian near-misses overall (0.87 against 0.91, although it catches
  more of them at 5 % false alarms), Italian and Mandarin near-misses at 5 % false alarms, and by
  about one point blaming the right word in German, Korean and Mandarin.

### Mandarin tones

Same syllable, different tone in the prompt (so only the pitch tells them apart):

| | Pronunciada | Azure |
|---|---|---|
| detection (AUC) | 0.816 | **0.846** |
| caught at 5 % false alarms | 30 % | **40 %** |

Azure is still ahead here. The tone model itself listens to pitch only (171 k parameters); it
agrees with the dictionary tone on 84 % of syllables of unseen studio speakers and 68–75 % on
harder crowd-sourced speech, and tells a wrong tone from a right one with AUC 0.90–0.97 on native
speech. Simplified and Traditional text give identical references; Taiwan-accented speech is
tested separately ([results/prosody.md](results/prosody.md)).

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
- **Mandarin tones** trail Azure on the head-to-head above.
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
