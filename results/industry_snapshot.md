# Pronunciation assessment systems, November 2026

Snapshot of 19 November 2026, updated 22 November. Pronunciada 1.0.2 (its scores are those of 1.0.1; 1.0.2 changed one Japanese error flag). It covers systems built to grade pronunciation: commercial APIs and open-source packages. General speech recognizers (Whisper, wav2vec2 phone recognizers, forced aligners) are not pronunciation systems and are not included here.

> **Update, 22 November 2026.** OpenPronounce 1.0.0 and 1.1.0 came out on 18 and 19 November with new models. Every OpenPronounce figure on this page is now for 1.1.0, measured by us on 22 November on the same recordings as Pronunciada. The earlier result for 0.4.0 (Pronunciada ahead on all 28 tests) does not hold for 1.1.0: see [Open-source systems](#open-source-systems). Full tables: [round19_rivals.md](round19_rivals.md).

## Systems

| system | kind | languages | what it returns | how it was compared |
|---|---|---|---|---|
| **Pronunciada 1.0.2** | open-source library (Apache-2.0) | 10 | accuracy per sound, word and sentence; fluency, prosody and completeness; Mandarin tones; error flags; Python, command line, HTTP server, MCP server, Azure-format JSON | – |
| Microsoft Azure Pronunciation Assessment | commercial API | 10 tested | accuracy per sound, word and sentence; fluency, completeness, prosody; error flags | run by us on all 84 tests in [SCOREBOARD.md](SCOREBOARD.md) |
| Lingolix | commercial API | 21 offered, 6 in its benchmark; no Mandarin | word scores, error flags and a sentence score (what its benchmark reads) | run by us through its API on 22 November 2026 (model version 2026-11): its benchmark's 1,700 recordings and the other 2,000 speechocean762 test recordings |
| TonePerfect | commercial API | 6 in that benchmark | the same | not run by us: the figures Lingolix published for its benchmark on 11 October 2026 |
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

**Lingolix and TonePerfect**: results on the [Lingolix benchmark](https://github.com/lingolix/pronunciation-benchmark) (v2026.10). Lingolix built it and is one of the vendors it compares.

- The **Lingolix** column was measured by us through its API on 22 November 2026 (model version 2026-11) on the benchmark's 1,700 recordings. Intervals, every paired test and the figures Lingolix itself published: [lingolix_live.md](lingolix_live.md).
- The **Azure** and **TonePerfect** columns are the figures Lingolix published on 11 October 2026, rounded and without intervals.
- **Pronunciada** has two columns: the 1.0.2 release, with 95% intervals, and the current `main` branch, which is not released. Method of our run: [third_party_lingolix.md](third_party_lingolix.md).

A "−" after a Pronunciada figure means it is significantly behind Lingolix on that row (paired test on the same recordings, p < 0.05). No mark means the difference is not significant. Bold is the best figure in the row.

Test 1: native recordings in which one word of the text was changed. Word-level AUC, higher is better.

| | Pronunciada 1.0.2 | Pronunciada `main` | Lingolix | Azure | TonePerfect |
|---|---|---|---|---|---|
| English | 0.977 [0.963, 0.988] | 0.977 | **0.982** | 0.94 | 0.84 |
| French | 0.960 [0.924, 0.987] | 0.960 | **0.968** | 0.91 | 0.80 |
| Spanish | 0.979 [0.970, 0.987] | **0.985** | 0.977 | 0.91 | 0.79 |
| German | 0.940 [0.911, 0.965] − | 0.967 | **0.973** | 0.87 | 0.79 |
| Japanese | 0.846 [0.798, 0.891] − | 0.923 | **0.951** | 0.93 | 0.55 |
| Korean | 0.912 [0.874, 0.946] − | 0.957 | **0.964** | 0.94 | 0.67 |

Test 1: share of correct words wrongly flagged when 80% of the changed words are caught. Lower is better.

| | Pronunciada 1.0.2 | Pronunciada `main` | Lingolix | Azure | TonePerfect |
|---|---|---|---|---|---|
| English | 2.5% − | 2.5% − | **0.8%** | 6% | 26% |
| French | 1.6% | 1.7% | **1.5%** | 13% | 26% |
| Spanish | 2.4% − | 1.4% | **1.1%** | 14% | 32% |
| German | 6.7% − | 2.1% | **1.0%** | 22% | 42% |
| Japanese | 20.1% − | 9.7% − | 3.2% | 3% | 74% |
| Korean | 8.4% − | 3.1% | **2.4%** | 7% | 56% |

Japanese: Lingolix's measured 3.2% and Azure's published 3% cannot be ranked, since the Azure figure is rounded. Japanese vowel- and consonant-length errors caught by each system's own error flag, of 40: Pronunciada 1.0.2 19 (1 in 1.0.1), `main` 27, Lingolix 38, Azure 36 (published).

Test 2: 500 English learner recordings (speechocean762 test), Pearson correlation with the human scores. Higher is better.

| | Pronunciada 1.0.2 | Pronunciada `main` | Lingolix | Azure | TonePerfect |
|---|---|---|---|---|---|
| sentence | 0.763 [0.658, 0.832] | **0.788** | 0.713 | 0.76 | 0.59 |
| word | 0.581 [0.489, 0.644] | 0.581 | 0.527 | **0.61** | 0.29 |

All English learner recordings. Test 2 uses 500 of the 2,500 speechocean762 test recordings, a few from each speaker. On 22 November we sent Lingolix the other 2,000, so the comparison now covers the whole test split (125 speakers, 20 recordings each). `main` against Lingolix, with a paired test that resamples speakers:

| recordings | score | Pronunciada `main` | Lingolix | difference [95% interval] | p |
|---|---|---|---|---|---|
| the benchmark's 500 | sentence | 0.788 | 0.713 | +0.075 [−0.008, +0.168] | 0.074 |
| | word | 0.581 | 0.527 | +0.054 [−0.021, +0.129] | 0.163 |
| all 2,500 | sentence | **0.792** | 0.668 | +0.124 [+0.057, +0.190] | < 0.001 |
| | word | **0.573** | 0.490 | +0.083 [+0.043, +0.123] | < 0.001 |

Lingolix answered all 2,500. 1.0.2 was not scored in this format on the added recordings.

Mandarin learners: Lingolix does not offer Mandarin (a request with the code `zh` was refused on 22 November with its list of 21 languages), so the OMPAL test could not be run against it.

Pronunciada is ahead of TonePerfect on every figure. Against Lingolix as measured:

- **1.0.2** is not significantly different on English, French and Spanish AUC, on French false alarms and on the 500 learner recordings. It is behind on English and Spanish false alarms and on both figures in German, Japanese and Korean.
- **`main`** is not significantly different in French, Spanish, German and Korean or on the 500 learner recordings. It is behind on English and Japanese false alarms and on Japanese length errors, and ahead on the full set of 2,500 learner recordings, for sentence and word scores.

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

- **Commercial:** Pronunciada is not shown to be the most accurate system overall. Lingolix, as measured by us, is ahead of 1.0.2 in German, Japanese and Korean and on English and Spanish false alarms. Azure is ahead on 7 of 84 tests, and four large vendors were not tested. On the unreleased `main` branch the gap to Lingolix is down to English and Japanese false alarms and Japanese length errors, and `main` is ahead of Lingolix on the English learner test (all 2,500 speechocean762 test recordings).
- **Open source:** no open system is ahead on everything. Against OpenPronounce 1.1.0, which also covers several languages, Pronunciada is ahead on 32 of 42 shared tests: word and sound scores for English learners and changed words in six languages. The two are not significantly different on the sentence scores for English learners, where OpenPronounce's figures are higher (0.790 against 0.762 on the total). OpenPronounce is ahead on the sounds annotators marked in L2-ARCTIC and EpaDB. Pronunciada runs 3.6 to 16 times faster with a quarter of the model size or less. Two English-only research models, E2E-GOP and CMUBYE, score single English sounds and words better; both return only per-sound scores.

## Reproducing

- Azure: see [Reproducing the benchmarks](../README.md#reproducing-the-benchmarks).
- Lingolix benchmark: `python -m bench.third_party.lingolix all` (downloads the test items, no account needed).
- Lingolix through its API: `python -m bench.third_party.lingolix_live --plan-only` lists what would be sent and what it costs. Sending needs your own Lingolix key and `--quota-cap-min`, a cap on audio minutes.
- OpenPronounce: `python -m bench.rivals.openpronounce --mode default --datasets speechocean762` runs it in its own environment with no network access; `bench/rivals/` holds the evaluation scripts and `python -m bench.ondevice` the cost measurements. Method and every row: [round19_rivals.md](round19_rivals.md).
- The E2E-GOP and CMUBYE runs used setup scripts that are not in this repository yet.
