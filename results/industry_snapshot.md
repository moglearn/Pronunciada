# Pronunciation assessment systems, November 2026

Snapshot of 19 November 2026, updated 22 November. Pronunciada 1.0.2 (its scores are those of 1.0.1; 1.0.2 changed one Japanese error flag). It covers systems built to grade pronunciation: commercial APIs and open-source packages. General speech recognizers (Whisper, wav2vec2 phone recognizers, forced aligners) are not pronunciation systems and are not included here.

> **Update, 22 November 2026.** OpenPronounce 1.0.0 and 1.1.0 came out on 18 and 19 November with new models. Every OpenPronounce figure on this page is now for 1.1.0, measured by us on 22 November on the same recordings as Pronunciada. The earlier result for 0.4.0 (Pronunciada ahead on all 28 tests) does not hold for 1.1.0: see [Open-source systems](#open-source-systems). Full tables: [round19_rivals.md](round19_rivals.md).

## Systems

| system | kind | languages | what it returns | how it was compared |
|---|---|---|---|---|
| **Pronunciada 1.0.2** | open-source library (Apache-2.0) | 10 | accuracy per sound, word and sentence; fluency, prosody and completeness; Mandarin tones; error flags; Python, command line, HTTP server, MCP server, Azure-format JSON | – |
| Microsoft Azure Pronunciation Assessment | commercial API | 10 tested | accuracy per sound, word and sentence; fluency, completeness, prosody; error flags | run by us on all 84 tests in [SCOREBOARD.md](SCOREBOARD.md) |
| Lingolix | commercial API | 6 in its benchmark | word scores, error flags and a sentence score (what its benchmark reads) | its own published benchmark; Pronunciada run on the same 1,700 recordings |
| TonePerfect | commercial API | 6 in that benchmark | the same | the same |
| OpenPronounce 1.1.0 | open-source library (AGPL-3.0) | 7: English, French, Spanish, German, Italian, Portuguese, Dutch | sentence scores (total, accuracy, fluency, prosody), a probability per word and sound, flagged words | run by us, 22 November 2026 |
| E2E-GOP | open research model (MIT) | English | one score per expected sound | run by us, 18 November 2026 |
| CMUBYE | open research model | English | one quality score per expected sound | run by us, 18 November 2026 |

Not compared:

- **SpeechSuper**: its terms of service do not allow using the service for evaluation.
- **Speechace**: its terms require written approval before benchmark results are published. None was requested.
- **ELSA (API), iFlytek, Tencent Cloud Smart Oral Evaluation**: each needs a commercial account, and none was opened for this snapshot.
- **Open packages found and not run yet**: hatsuon, bareum, Mimora and the Kaldi GOP recipe.

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

> **Update, 22 November 2026.** Lingolix has since been measured by us through its API (model version 2026-11) on the same 1,700 recordings: [lingolix_live.md](lingolix_live.md). Its measured figures are at or above the published ones in the tables below (German 0.973 and 1.0%, Korean 0.964 and 2.4%, Japanese 0.951 and 3.2%, 38 of 40 length errors). The Pronunciada column on that page is the current `main` branch, not 1.0.2. On it, the two systems are not significantly different in French, Spanish, German and Korean or on the English learner test; Lingolix is significantly ahead on English false alarms and in Japanese (false alarms and length errors).

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

English learners, speechocean762 test (2,500 recordings), Pearson correlation with expert scores. "+" means Pronunciada is significantly ahead of that system on the row, "−" significantly behind (paired tests, p < 0.05); no mark means the difference is not significant.

| test | Pronunciada | E2E-GOP | CMUBYE | OpenPronounce 1.1.0 |
|---|---|---|---|---|
| per sound | 0.685 | **0.796 −** | 0.707 − | 0.534 + |
| per word | 0.609 | **0.723 −** | 0.667 − | 0.569 + |
| sentence accuracy | **0.748** | 0.682 + | 0.686 + | 0.744 |
| sentence fluency | 0.779 | 0.679 | 0.661 | **0.827** |
| sentence prosody | 0.772 | 0.670 | 0.650 | **0.808** |
| sentence total | 0.762 | 0.724 | 0.712 | **0.790** |

OpenPronounce's 0.790 matches the 0.79 its authors report. Its lead on the sentence total is not significant (difference 0.028, 95% interval −0.017 to +0.094, p = 0.44, speakers resampled). By age, with its own age groups: adults 0.850 against Pronunciada's 0.844, children of 12 and under 0.670 against 0.558 (p = 0.57; without one speaker the difference turns in Pronunciada's favour). Its scorers were trained on the training half of speechocean762, and its authors compared design variants on the test split.

Finding the sounds annotators marked as mispronounced (AUC). No system here was trained on either corpus.

| test | Pronunciada | E2E-GOP | CMUBYE | OpenPronounce 1.1.0 |
|---|---|---|---|---|
| L2-ARCTIC | 0.712 | 0.731 − | 0.734 | **0.743 −** |
| EpaDB | 0.721 | 0.733 − | 0.701 + | **0.741 −** |

Changed-word tests in other languages, Pronunciada against OpenPronounce 1.1.0, which also covers several languages. Word-level AUC, higher is better; every difference is significant.

| | Lingolix test 1: Pronunciada | OpenPronounce | FLEURS, similar-sounding word: Pronunciada | OpenPronounce |
|---|---|---|---|---|
| English | **0.977** | 0.886 | **0.981** | 0.900 |
| French | **0.960** | 0.877 | **0.986** | 0.915 |
| Spanish | **0.979** | 0.929 | **0.990** | 0.894 |
| German | **0.940** | 0.887 | **0.960** | 0.885 |
| Italian | – | – | **0.979** | 0.854 |
| Portuguese | – | – | **0.963** | 0.820 |

Cost on a laptop CPU (Apple M3 Pro, 200 recordings of 3.4 s on average, fastest thread setting of each system):

| system | seconds per recording | peak memory | model files |
|---|---|---|---|
| **Pronunciada, 8-bit model** | **0.047** | **365 MB** | **91 MB** |
| Pronunciada, full-precision model | 0.085 | 532 MB | 281 MB |
| OpenPronounce, distilled English model | 0.170 | 1.4 GB | 396 MB |
| OpenPronounce `--fast` | 0.296 | 2.3 GB | 1.3 GB |
| OpenPronounce default | 0.735 | 3.7 GB | 2.6 GB |

OpenPronounce's default and `--fast` modes gave identical scores on every recording. Its distilled English model scores 0.802 on the speechocean762 sentence total, also not significantly different from Pronunciada (p = 0.09).

Over all the tests each system covers:

| against | ahead | not different | behind |
|---|---|---|---|
| OpenPronounce 1.1.0 (42 tests, six languages) | 32 | 8 | 2 |
| E2E-GOP (English learners) | 1 | 3 | 8 |
| CMUBYE (English learners) | 5 | 5 | 2 |

The 42 tests against OpenPronounce: the six speechocean762 rows above (ahead on 2, not different on 4), the two marked-sound corpora (behind on both), 18 changed-word rows on FLEURS in six languages (ahead on all), 8 rows of Lingolix test 1 in four languages (ahead on 7; German false alarms not different) and 8 Common Voice rows telling rejected from accepted clips and native from non-native speakers (ahead on 5, not different on 3). OpenPronounce covers Dutch, which Pronunciada does not; Pronunciada covers Japanese, Korean, Mandarin and Russian, which OpenPronounce does not.

## Verdict

- **Commercial:** Pronunciada is not shown to be the most accurate system overall. Lingolix is ahead in Japanese, Korean and on German false alarms, Azure is ahead on 7 of 84 tests, and four large vendors were not tested.
- **Open source:** no open system is ahead on everything. Against OpenPronounce 1.1.0, which also covers several languages, Pronunciada is ahead on 32 of 42 shared tests: word and sound scores for English learners and changed words in six languages. The two are not significantly different on the sentence scores for English learners, where OpenPronounce's figures are higher (0.790 against 0.762 on the total). OpenPronounce is ahead on the sounds annotators marked in L2-ARCTIC and EpaDB. Pronunciada runs 3.6 to 16 times faster with a quarter of the model size or less. Two English-only research models, E2E-GOP and CMUBYE, score single English sounds and words better; both return only per-sound scores.

## Reproducing

- Azure: see [Reproducing the benchmarks](../README.md#reproducing-the-benchmarks).
- Lingolix benchmark: `python -m bench.third_party.lingolix all` (downloads the test items, no account needed).
- OpenPronounce: `python -m bench.rivals.openpronounce --mode default --datasets speechocean762` runs it in its own environment with no network access; `bench/rivals/` holds the evaluation scripts and `python -m bench.ondevice` the cost measurements. Method and every row: [round19_rivals.md](round19_rivals.md).
- The E2E-GOP and CMUBYE runs used setup scripts that are not in this repository yet.
