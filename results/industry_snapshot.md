# Pronunciation assessment systems, November 2026

Snapshot of 19 November 2026, Pronunciada 1.0.1. It covers systems built to grade pronunciation: commercial APIs and open-source packages. General speech recognizers (Whisper, wav2vec2 phone recognizers, forced aligners) are not pronunciation systems and are not included here.

> **Note, 22 November 2026.** OpenPronounce 1.0.0 and 1.1.0 were released on 18 and 19 November with new models. Its authors report a correlation of 0.79 between its overall score and the expert scores on speechocean762 (all speakers, sentence known). Pronunciada's sentence total on that test is 0.762. The OpenPronounce rows on this page are for version 0.4.0, so the open-source verdict below has not been checked against the new version yet. A rerun is in progress.

## Systems

| system | kind | languages | what it returns | how it was compared |
|---|---|---|---|---|
| **Pronunciada 1.0.1** | open-source library (Apache-2.0) | 10 | accuracy per sound, word and sentence; fluency, prosody and completeness; Mandarin tones; error flags; Python, command line, HTTP server, MCP server, Azure-format JSON | – |
| Microsoft Azure Pronunciation Assessment | commercial API | 10 tested | accuracy per sound, word and sentence; fluency, completeness, prosody; error flags | run by us on all 84 tests in [SCOREBOARD.md](SCOREBOARD.md) |
| Lingolix | commercial API | 6 in its benchmark | word scores, error flags and a sentence score (what its benchmark reads) | its own published benchmark; Pronunciada run on the same 1,700 recordings |
| TonePerfect | commercial API | 6 in that benchmark | the same | the same |
| OpenPronounce 0.4.0 | open-source app (MIT) | English; 5 more marked experimental | sentence score and flagged words | run by us, 18 November 2026 |
| E2E-GOP | open research model (MIT) | English | one score per expected sound | run by us, 18 November 2026 |
| CMUBYE | open research model | English | one quality score per expected sound | run by us, 18 November 2026 |

Not compared:

- **SpeechSuper**: its terms of service do not allow using the service for evaluation.
- **Speechace, ELSA (API), iFlytek, Tencent Cloud Smart Oral Evaluation**: each needs a commercial account, and none was opened for this snapshot.

### What E2E-GOP covers

E2E-GOP is the strongest open model on English single sounds (below), so its scope matters when reading the tables. From its model card (`anonym-coling2027/E2E-GOP-SpeechOcean762`, published 14 October 2026 by anonymous authors of a paper under review):

- Input: a recording and the expected sounds in ARPAbet, which the caller has to supply. It does not take text and has no pronunciation dictionary.
- Output: one number from 0 to 2 per sound. It has no word or sentence score, no fluency, prosody or completeness score, and no error flags. The word and sentence figures below are averages we computed from its sound scores.
- English only. It comes as a Hugging Face model with example code, not as a package, command-line tool or server.
- Its scoring head was trained on the training half of speechocean762, the same corpus as the main English test below.

CMUBYE has the same shape: English only, per-sound scores only. Its scoring head was chosen on the speechocean762 test split, so its speechocean762 figures are optimistic.

## Commercial systems

**Azure** (our runs, same recordings and text for both): 84 comparisons in ten languages. Pronunciada is significantly ahead in 40, not significantly different in 37 and significantly behind in 7. Details: [SCOREBOARD.md](SCOREBOARD.md).

**Lingolix and TonePerfect**: results on the [Lingolix benchmark](https://github.com/lingolix/pronunciation-benchmark) (v2026.10). Lingolix built it and is one of the vendors it compares. The Lingolix, Azure and TonePerfect figures are the ones Lingolix published, rounded and without intervals. Pronunciada's figures are from our run on the same items, with 95% intervals. Full method: [third_party_lingolix.md](third_party_lingolix.md).

Test 1: native recordings in which one word of the text was changed. Word-level AUC, higher is better.

| | Pronunciada | Lingolix | Azure | TonePerfect |
|---|---|---|---|---|
| English | **0.977** [0.963, 0.988] | 0.97 | 0.94 | 0.84 |
| French | **0.960** [0.924, 0.987] | **0.96** | 0.91 | 0.80 |
| Spanish | **0.979** [0.970, 0.987] | 0.97 | 0.91 | 0.79 |
| German | 0.940 [0.911, 0.965] | **0.95** | 0.87 | 0.79 |
| Japanese | 0.846 [0.798, 0.891] | **0.95** | 0.93 | 0.55 |
| Korean | 0.912 [0.874, 0.946] | **0.95** | 0.94 | 0.67 |

Test 1: share of correct words wrongly flagged when 80% of the changed words are caught. Lower is better.

| | Pronunciada | Lingolix | Azure | TonePerfect |
|---|---|---|---|---|
| English | 2.5% | **1%** | 6% | 26% |
| French | 1.6% | **1%** | 13% | 26% |
| Spanish | 2.4% | **2%** | 14% | 32% |
| German | 6.7% | **2%** | 22% | 42% |
| Japanese | 20.1% | 5% | **3%** | 74% |
| Korean | 8.4% | **3%** | 7% | 56% |

Test 2: 500 English learner recordings (speechocean762 test), Pearson correlation with the human scores. Higher is better.

| | Pronunciada | Lingolix | Azure | TonePerfect |
|---|---|---|---|---|
| sentence | **0.763** [0.658, 0.832] | 0.74 | 0.76 | 0.59 |
| word | 0.581 [0.489, 0.644] | 0.51 | **0.61** | 0.29 |

Pronunciada is ahead of TonePerfect on every figure. Against Lingolix it is level in English, French and Spanish, ahead on English learner words, and behind in German false alarms, Japanese and Korean. In Japanese its own error flag caught 19 of 40 vowel- and consonant-length errors (1 of 40 in 1.0.1; fixed in 1.0.2); Lingolix caught 37 and Azure 36.

## Open-source systems

English learners, speechocean762 test, Pearson correlation with expert scores. "+" means Pronunciada is significantly ahead of that system on the row, "−" significantly behind (paired tests, p < 0.05). "ʷ": the system has no per-sound output, so each sound carries its word's score.

| test | Pronunciada | E2E-GOP | CMUBYE | OpenPronounce |
|---|---|---|---|---|
| per sound | 0.685 | **0.796 −** | 0.707 − | 0.374 +ʷ |
| per word | 0.609 | **0.723 −** | 0.667 − | 0.431 + |
| sentence accuracy | **0.748** | 0.682 + | 0.686 + | 0.636 + |
| sentence fluency | **0.779** | 0.679 | 0.661 | 0.628 + |
| sentence prosody | **0.772** | 0.670 | 0.650 | 0.649 + |
| sentence total | **0.762** | 0.724 | 0.712 | 0.654 + |

Finding the sounds annotators marked as mispronounced (AUC). No system here was trained on either corpus.

| test | Pronunciada | E2E-GOP | CMUBYE | OpenPronounce |
|---|---|---|---|---|
| L2-ARCTIC | 0.712 | 0.731 − | **0.734** | – |
| EpaDB | 0.721 | **0.733 −** | 0.701 + | – |

Over all the tests each system covers:

| against | ahead | not different | behind |
|---|---|---|---|
| OpenPronounce (English learners and the changed-word test) | 28 | 0 | 0 |
| E2E-GOP (English learners) | 1 | 3 | 8 |
| CMUBYE (English learners) | 5 | 5 | 2 |

## Verdict

- **Commercial:** Pronunciada is not shown to be the most accurate system overall. Lingolix is ahead in Japanese, Korean and on German false alarms, Azure is ahead on 7 of 84 tests, and four large vendors were not tested.
- **Open source:** Pronunciada is the most accurate open-source pronunciation library we found. It is ahead of OpenPronounce on all 28 tests, and it is the only open package that gives sound, word and sentence scores, fluency, prosody and error flags, and covers more than one language. Two English-only research models, E2E-GOP and CMUBYE, score single English sounds and words better. Both return only per-sound scores.

## Reproducing

- Azure: see [Reproducing the benchmarks](../README.md#reproducing-the-benchmarks).
- Lingolix benchmark: `python -m bench.third_party.lingolix all` (downloads the test items, no account needed).
- The open-model runs used setup scripts that are not in this repository yet.
