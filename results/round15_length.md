# Round 15: a length channel (long / short sounds) — Japanese adopted; Italian and German not

**Outcome.**
* **Japanese is adopted.** The new length channel judges Japanese long vs short vowels (including
  ー and おう / えい spelt long) and っ vs a single consonant from timing.
* **Planted length swaps.** At a threshold that flags 5 % of native judged sounds, it catches 85 %
  of planted length swaps on held-out FLEURS test (dev 87 %). Before, the scorer caught 5–6 %, and
  short-for-long was not even representable.
* **Swapped words.** The word-level factor flags 49 % of length-swapped words at the 5 % native
  word threshold (24 % before).
* **Costs.** Native word acceptance drops 0.1 point. The existing benchmarks move by ≤ 0.002 AUC.
* **Italian doubling misses the dev bars:** 33 % of swapped sounds caught (bar 50 %) and 34 % of
  words (bar 40 %).
* **German vowel length cleared every dev bar, but the held-out check found a significant
  regression** on cv-real: word-level AUC −0.009 / −0.011, p ≈ 0.01. It is therefore **not
  shipped**; the fitted candidate is kept outside the package.

Every language without a shipped channel is byte-identical to before. This was checked against
HEAD on 15 FLEURS test clips: es, zh, en, de, it, with all probabilities, spans and heard sounds
compared. Japanese with `Scorer(..., length=False)` is identical as well.

## What changed

* **Representation.** `RefPhone.length` is new: `"long"` / `"short"` / None. When a front-end
  leaves it None, `pronunciada.length.labels` derives it from the canonical symbol: Japanese `aː`
  or `tt` = long; a plain vowel, or a single consonant between two vowels of the same word =
  short.
  * The canonical symbol is what the prompt asks for. So おばあさん `aː{a}` expects a long vowel even
    though the front-end still accepts the short form as an alternative pronunciation.
  * The alternatives are untouched, so sound, word and sentence scores are unchanged when the
    channel is off.
  * Four places are deliberately not judged, because native speech is not length-contrastive there
    (they produced most native false alarms on dev):
    * the vowel of a one-mora word (particles を の で は: lengthened at phrase edges);
    * a short vowel next to another vowel of the same word (えい, おお, いう are often said as one
      long vowel);
    * a single voiceless consonant before i / u (devoicing: ます, です);
    * a vowel followed by ≥ 80 ms of silence (phrase-final lengthening).
* **Evidence** (`pronunciada/length.py`, computed from what the scorer already has):
  * For vowels:
    * the onset-to-onset interval from the canonical alignment (20 ms frames), relative to the
      utterance's median vowel;
    * the voiced part of that interval, cut at the first 30 ms of silence in the fbank energy;
    * the ratio to the neighbouring vowels;
    * the peak posterior of the acoustic model's length token `ː` over the vowel (AUC 0.97 alone on
      dev natives);
    * position in the word and the utterance, and the next consonant's class.
  * For consonants: the vowel-to-vowel span across the consonant relative to the utterance's
    single consonants, the closure silence, and the minimum energy.
* **Model.** 120 boosted depth-3 trees per class (vowel / consonant) predict P(long). They are
  fitted on FLEURS-ja **train** natives (1 000 clips; 4 765 long / 26 768 short vowels,
  510 / 13 319 consonants) with the text's own labels and balanced classes. The trees are exported
  as complete arrays, walked in numpy, and take 41 kB of JSON (`pronunciada/data/length/ja.json`).
  A logistic regression on the same features reached only 30 % of swaps at 1 % native false alarms;
  the trees reach 60 %.
* **Outputs.**
  * Per sound: `PhoneResult.length_expected` / `length_prob`, where `length_prob` = P(long) for an
    expected long sound and 1 − P(long) for a short one.
  * Per word: `WordResult.length_prob`, the factor multiplied into `WordResult.prob`.
  * The factor is 1 while the word's lowest length logit is above a knee at a low native quantile.
    Below the knee it falls linearly in log-odds to a floor. The knee moves with strictness (native
    quantile q/2 at s = 0, q at 0.5, 2q at 1).
  * The prosody channel's smooth logistic was tried first. It lowered too many native words a
    little, cost 0.014 planted sound-error word AUC and reached only 34–40 % word hit.
* **Coaching.** New kind `length` (`schema_minor` 4). `expected` / `heard` are `"long"` /
  `"short"`, and `expected_ipa` is the sound. Tips: `length.vowel_long` ("hold … for two beats"),
  `length.vowel_short`, `length.consonant_long` ("pause briefly before …"),
  `length.consonant_short` ("no pause before …").
  * The gate is a threshold on `length_prob`, set per sensitivity to the loosest native quantile
    whose estimated precision reaches the other kinds' targets (0.8 / 0.7 / 0.6). Precision is
    estimated for a 5 % share of wrong-length sounds, with planted swaps standing in for learner
    errors; no learner data with length labels exists.
  * Japanese dev values:

    | sensitivity | precision | catch | natives flagged |
    |---|---|---|---|
    | careful | 0.82 | 45 % | 0.5 % |
    | balanced | 0.71 | 70 % | 1.5 % |
    | eager | 0.66 | 75 % | 2.0 % |

* **Latency.** 0.77 ms per 10 s of audio (median over 60 FLEURS-ja test clips; p90 0.88 ms) against
  a total of 182 ms.

## Dev (FLEURS validation; planted swaps from `bench/audit/contrasts.py`, both directions)

Fixed rule for the word factor. The grid is native quantile q ∈ {0.2, 0.5, 1, 2} % × width
{1, 2, 4} × floor {0.2, 0.4}. A configuration is feasible when native word acceptance at s = 0.5
drops ≤ 0.5 points, and both the planted sound-error word AUC and the near-swap AUC drop
≤ 0.003. Among feasible configurations, the one with the highest word hit@5 % is taken.

| Japanese (266 clips) | before | after | bar |
|---|---|---|---|
| swapped sounds caught at 5 % native judged-sound FA (1 163 swaps) | 5–6 % (audit) | **87 %** (AUC 0.971) | ≥ 50 % |
| … vowels: said long where short expected / said short where long expected | | 97 % / 82 % | |
| … consonants: said double where single expected / said single where double (っ) expected | | 86 % / 81 % | |
| length-swapped words at 5 % native word FA (1 804) | 28 % | **50 %** | ≥ 40 % |
| native word acceptance, s = 0.5 | 98.28 % | 98.18 % (−0.10) | ≤ −0.5 |
| planted sound-error word AUC (5 kinds) | 0.9410 | 0.9393 (−0.0017) | ≤ −0.003 |
| near-swap AUC (frozen dev swap set) | 0.9655 | 0.9635 (−0.0020) | ≤ −0.003 |
| sound-level probabilities (`PhoneResult.prob`) | | unchanged by construction | |

Chosen: q = 2 %, width 4, floor 0.4. Natives flagged at the sound level make up 2.3 % of all
native sounds, since only judged sounds can be flagged.

| other languages, dev | Italian doubling (385 clips) | German stressed vowel length (359 clips) |
|---|---|---|
| swapped sounds caught at 5 % native FA | **33 %** (AUC 0.76) ✗ | 64 % (AUC 0.93) |
| length-swapped words at 5 % word FA (best feasible) | **34 %** (28 % before) ✗ | 47 % (28 % before) |
| native word acceptance drop | 0.09 points | 0.46 points |
| sound-error / near-swap word AUC drop | 0.0006 / 0.0024 | 0.0014 / −0.0006 |
| dev verdict | not adopted | dev bars met → held-out check |

Italian geminates in read FLEURS speech are often barely longer than single consonants (the audit
measured a timing AUC of 0.74). Neither the closure silence nor the acoustic model adds enough on
top of that.

## Held-out (FLEURS test, fleurs-swap test, cv-real test half; scored once, paired before / after)

`bench/length/heldout.py` scores every clip once with the channel off and applies the shipped
channel to the same evidence. This matches the runtime Scorer to 1e-16 on 5 clips. No Azure call
was made.

**Japanese (adopted)**

| | before | after |
|---|---|---|
| planted length swaps: sounds caught at 5 % native FA (1 701 swaps, 400 clips) | – | **85 %** (AUC 0.967) |
| … vowels: said long for short / said short for long | | 96 % / 81 % |
| … consonants: said double for single / said single for double (っ) | | 87 % / 72 % |
| length-swapped words at 5 % native word FA (2 672) | 24 % | **49 %** |
| native word acceptance, s = 0 / 0.5 / 1 | 97.95 / 97.69 / 97.47 % | 97.91 / 97.59 / 96.75 % |
| ten-language benchmark: sound AUC / hit@5 % | 0.9754 / 92.5 % | 0.9754 / 92.5 % |
| ten-language benchmark: word AUC / hit@5 % / lowest = corrupted | 0.9334 / 73.8 % / 74.2 % | 0.9313 / 73.4 % / 74.2 % |
| ten-language benchmark: native sounds / words / sentences | 99.65 / 97.69 / 99.25 % | 99.65 / 97.59 / 99.25 % |
| swap benchmark (fleurs-swap): near AUC / hit@5 % | 0.9652 / 84.0 % | 0.9638 / 82.0 % |
| swap benchmark: far AUC / hit@5 % | 0.9911 / 98.0 % | 0.9909 / 97.3 % |
| swap benchmark: natives below 0.6 | 2.5 % | 2.9 % |
| cv-real Test A (390 clips): primary AUC (sentence prob) | 0.802 | 0.802 (unchanged by construction) |
| cv-real Test A: word mean / word min AUC | 0.785 / 0.766 | 0.787 / 0.770 (paired Δ +0.002 / +0.004, n.s.) |

Every held-out bar holds:
* swapped sounds 85 % ≥ 50 %;
* swapped words 49 % ≥ 40 %;
* native word drop 0.10 point ≤ 0.5;
* sound AUC unchanged;
* word AUCs −0.002 or better.

At the strictest setting (s = 1) natives lose 0.7 points of word acceptance. That is the knee at
the 4 % native quantile doing its job; it is not an error.

**German (not adopted)**

| | before | after |
|---|---|---|
| planted length swaps: sounds caught (1 546) / words caught (1 554) | – / 25 % | 64 % / 48 % |
| native word acceptance, s = 0.5 | 96.78 % | 96.44 % (−0.34) |
| ten-language: word AUC / hit@5 % | 0.9455 / 75.1 % | 0.9446 / 73.9 % |
| ten-language: stress-error word AUC | 0.9386 | 0.9359 (−0.0027) |
| swap benchmark near AUC | 0.9563 | 0.9553 |
| **cv-real Test A word mean / word min AUC** | 0.809 / 0.797 | **0.800 / 0.786** (paired speaker bootstrap Δ −0.009 [−0.019, −0.002], p = 0.012; −0.011 [−0.021, −0.003], p = 0.008) |

German cleared the dev bars, but the held-out check found a statistically significant (if small)
loss on real Common Voice recordings, plus a 1.2-point drop in ten-language word hit@5 % and a
stress AUC drop at the edge of the 0.003 bar. With nothing allowed to regress, it stays off. The
German stressed vowels differ in quality as well as length, and on CV's mixed-quality recordings
the timing evidence lowers some accepted readings. The candidate is kept in the bench cache
(`$PRONUNCIADA_CACHE/length/len_v1/candidates/de.json`); shipping it would be a one-file change
after more work.

## Caveats

* Planted swaps edit the reference while the native audio stays the same. "Said long where short
  was expected" is therefore a native long sound scored against a short target, not a real
  learner. Real learner length errors are often partial (a vowel of 1.4 beats), so real catch rates
  will be lower than 85 %. No Japanese learner data with length labels was available.
* The coaching precision values assume that 5 % of judged sounds are wrong.
* Japanese long vowels written as two moras (えい, おう after a consonant) are judged only where the
  front-end writes a single long vowel (`oː`); the others are not judged.
* While this round ran, another agent's round-16 contrast caps changed the *shown* sound
  probabilities of some languages. They do not touch word probabilities or anything the length
  channel reads. All before / after comparisons here are paired on the same tree.

## Reproduce

    python -m bench.length.fit_length collect --lang ja --split validation
    python -m bench.length.fit_length collect --lang ja --split train
    python -m bench.length.fit_length fit --lang ja            # writes pronunciada/data/length/ja.json
    python -m bench.length.heldout run --lang ja && python -m bench.length.heldout report --lang ja

Tables: `results/tables/round15_length.json` (dev grids, coaching points, held-out for ja and de)
and `results/tables/round15_length_heldout_{ja,de}.json`. Tests: `tests/test_length.py`.
