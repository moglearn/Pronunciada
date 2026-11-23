# Round 20: why correct native words are flagged (diagnosis on the practice sets)

Diagnosis only. Nothing under `pronunciada/` was changed and no threshold was retuned. No held-out set was
used, and no Lingolix clip, prompt or result was looked at. No API was called. Scripts are in
`bench/round20_native_fa/`; tables are in `results/tables/round20_native_fa_*.json`.

## Summary

* **Baseline on the practice sets (current main).**

  | lang | AUC | FA@80 | false alarms |
  |---|---|---|---|
  | en | 0.9641 | 3.40 % | 641 |
  | de | 0.9757 | 2.03 % | 365 |
  | ko | 0.9728 | 2.39 % | 217 |
  | es | 0.9662 | 2.36 % | 499 |
  | fr | 0.9421 | 3.77 % | 768 |

  Intervals are in section 2. German reproduces the round-19 frozen dev numbers exactly.
* **One shared cause appears in all five languages: words the speaker did not say (the unsaid-word path).**
  * When the prompt contains a word the speaker did not say (the planted "word omission" changes), the
    word next to it is flagged in 27-35 % of items. The same word, on the same audio, in the control item
    is flagged in 1-3 %.
  * No other kind of change has this effect, and it is gone two to three words away.
  * Mechanism: the unsaid word's sounds must still be aligned. They take frames from the neighbours, which
    then look absent (presence factor) or wrong.
  * Evidence: run with the existing explain-away pass (its band removed), the unsaid word's sounds become a
    skippable wildcard. Neighbour FA then falls to 3.6-5.3 % (en/de/es/fr), and the inserted word is still
    caught (100 %, fr 97 %).
  * Together with function and content words that the presence factor judges absent, this family is 22-29 %
    of false alarms in en/de/es/fr and 14 % in ko.
* **A second shared cause, in en, es and fr (smaller in de and ko): the text front-end.**
  * Numbers and abbreviations read differently from our expansion: FA rate 16-29 %.
  * Unknown words read by letter rules: en 31 %, fr 20 %, es 10 %.
  * Names: en 8 %, fr 8 %.
  * For comparison, the remaining correct words have an FA rate of 2-3 %.
  * This family is 38 % of English false alarms. It is the largest identifiable difference between English
    and German.
* **Measurement artifacts, not scorer errors.**
  * French: punctuation-only tokens (« » ! ? : ;) are 19 % of false alarms. The metric counts them as words
    and no system scores them.
  * Numbers written with a no-break space are split into two tokens.
  * A round-19 label-index bug affects en 1, de 1, ko 1, es 10 and fr 22 changed items.
* **Valid pronunciation variants are a real but minor cause.** A variant lifts 12-16 % of false alarms over
  the threshold (es 2 %). It also lifts changed words, so taking the best of N variants gains nothing net,
  with one exception: German lexicon readings, FA@80 2.03 → 1.90 %, AUC +0.0006.
* **Reduced function words are not the main source.** Function words have a lower FA rate than content
  words in every language (1.2-1.8 % vs 2.5-5.7 %). When they are flagged, it is mostly through the presence
  factor, which belongs to the unsaid-word family above.
* **Language-specific factors also matter.**
  * The stress factor drives the score of 13 % (en), 10 % (de) and 22 % (es) of false alarms, on words whose
    sounds are fine.
  * The identity step drives 18 % (ko) and 13 % (es).
* **Korean:** reading the evidence by hand shows regular sound changes that the front-end does not produce.
  Examples are n-insertion and nasalisation, such as 꽃잎 [꼰닙] and 56 [오심뉵]. The automatic
  n-insertion rule rescues 7 of the 26 Korean words that a variant rescues.

## 1. Data

The round-19 practice ("dev") sets of planted changed words were rebuilt with the round-19 builders,
unchanged. `rebuild.py` only points their work folders at a new cache folder. Seeds and exclusions are the
same:

* no Lingolix clip, speaker or prompt;
* no cv-real test-half speaker;
* FLEURS test never read.

The Lingolix manifests are read only by the builders' exclusion code.

| lang | builder | clips | items | check against round 19 |
|---|---|---|---|---|
| en, es, fr | `bench/round19_rollout/sources.py` + `devset.py` | 460 CV + 260 FLEURS validation | 1,440 | same counts |
| de | `bench/round19/devset.py` | 460 CV + 260 FLEURS validation | 1,440 | Lingolix exclusion 200 clips / 171 speakers as in round 19. AUC 0.9757 and FA@80 2.03 % equal the round-19 frozen dev numbers. |
| ko | `bench/round19ko/cv_pool.py` + `plant.py` + Whisper filter (`asr.py`, same decoding as `asr_check.py`) | 737-clip pool from 39 speakers. 656 clips have sound; 317 pass CER ≤ 0.1. Plus 226 FLEURS validation. | 953 (543 controls + 410 changed) | identical to round 19 |

Notes on the rebuild:

* Only the Common Voice clips of the practice speakers were fetched.
* The German builder's Common Voice `validated.tsv` was downloaded again from the same mirror and revision.
* Korean main-code numbers are in-sample for the round-19 identity head. Round 19's 0.972 / 2.65 % was
  out-of-fold.

**Label problem in the round-19 builders.** `word_index` counts front-end units, but the metric counts
whitespace tokens.

* After a punctuation-only token the two differ. The changed word is then scored as a correct word, and a
  correct word as the changed one.
* Affected changed items (`index_check.py`): en 1, de 1, ko 1, es 10, fr 22.
* They stay in the baseline, which must match round 19. Their false alarms are counted as noise.

**Scoring.**

* Every item goes through `Scorer(lang).score` with defaults, then `to_azure` and the benchmark's Azure
  reduction.
* The metric functions come from `bench/third_party/lingolix.py` and are applied to practice items only
  (`collect.py`, `core.py`).
* Per word, the run also keeps:
  * the factors: sounds, stress, length, contrast, presence, identity step;
  * the times;
  * per sound: the chosen alternative, the shown probability, what the decoder heard, llr_sub and llr_del.
* Diagnostic only:
  * llr_ins, from a second alignment of the same posteriors with insertion on;
  * the free greedy decode.

**Independent read-the-prompt check.** The local Whisper large-v3 (MIT licence) was run on every clip with
greedy decoding and the language fixed.

| lang | clips | within CER 0.1 |
|---|---|---|
| ko | 882 | 513 |
| en | 720 | 663 |
| de | 720 | 700 |
| es | 720 | 690 |
| fr | 720 | 650 |

Word timestamps were tried first through cross-attention token times. They grew memory until the memory
guard stopped the job, so they were dropped. The planned alignment-jitter test therefore has no independent
estimate and is reported as not done.

## 2. Baselines (current main)

| lang | items | speakers | correct words | changed | AUC [95 % CI] | FA@80 % [95 % CI] | threshold score | n FA | n < 0.6 |
|---|---|---|---|---|---|---|---|---|---|
| en | 1440 | 670 | 18879 | 720 | 0.9641 [0.9560, 0.9713] | 3.40 [2.72, 4.39] | 0.528 | 641 | 746 |
| de | 1440 | 672 | 17975 | 720 | 0.9757 [0.9700, 0.9807] | 2.03 [1.67, 2.76] | 0.458 | 365 | 548 |
| ko | 953 | 162 | 9072 | 410 | 0.9728 [0.9631, 0.9809] | 2.39 [1.30, 4.33] | 0.587 | 217 | 220 |
| es | 1440 | 600 | 21182 | 720 | 0.9662 [0.9577, 0.9735] | 2.36 [1.60, 3.00] | 0.552 | 499 | 550 |
| fr | 1440 | 596 | 20376 | 720 | 0.9421 [0.9292, 0.9531] | 3.77 [2.84, 4.77] | 0.645 | 768 | 712 |

Other baselines for the later gate. All are on non-held-out data.

| measure | value |
|---|---|
| speechocean762 TRAIN, speaker-grouped 5-fold CV, 3 seeds. Round-19 scripts re-run on features re-extracted with main (`so762cv.sh`; output kept out of `round19/` and out of tracked tables). | Sentence total 0.8537 (round 19: 0.8536). Accuracy 0.8322, fluency 0.8784, prosodic 0.8677. Children ≤ 12: 0.552; others: 0.894. Sequence head alone: 0.8395. Sound (0-2) Pearson 0.7303 at blend 0.3. Word (0-10) Pearson 0.6952 at blend 0.4. |
| L2-ARCTIC TRAIN marked-sound detection. Shipped sound model, AUC of 1 − P(sound) at strictness 0.5, speaker bootstrap. Measurement only. | 0.714 [0.683, 0.745]. Substitutions 0.689, deletions 0.825. 89,210 sounds, 13,223 marked. |
| EpaDB TRAIN, same measure | 0.701 [0.683, 0.716]. Substitutions 0.694, deletions 0.827. 31,077 sounds. |
| CPU time per clip and size. `bench.ondevice` protocol, 4 threads, on the first 200 speechocean762 TRAIN clips (not test); shared machine, load ≈ 8. | fp32: p50 51 ms, p95 68 ms, RTF 0.015, RSS 571 MB. int8: p50 41 ms, p95 56 ms, RTF 0.013, RSS 394 MB. Package data 34.7 MB. Model files 297 MB (fp32) / 107 MB (int8). |

## 3. Tests and buckets

Populations, both made only of correct words (every word of control items and the unchanged words of
changed items):

* **FA**: the error signal is at or above the FA@80 threshold.
* **< 0.6**: the word score is below the app's threshold.

Every test is automatic and written in `features.py` / `taxonomy.py`.

**Exclusive buckets.** Priority order, where the first test that fires wins:

1. **Corpus noise.** Any of:
   * Whisper disagrees at the word: character alignment of prompt and transcript, more than 25 % of the
     word's characters edited. Not counted for digits, names or unknown words, where Whisper's spelling is
     not a reading check.
   * SNR below 15 dB.
   * The file starts or ends inside speech at the first or last word.
   * Punctuation-only token.
   * Round-19 label-index artifact.
   * Tokenisation artifact (no-break-space numbers).
2. **Neighbour of the change.** The word is within two words of the changed word, and the same word in the
   control item (same audio, the clip's own text) is *not* a false alarm. The change caused it.
3. **Variant rescues.** The best alternative pronunciation lifts the word over the threshold (for the < 0.6
   population: to ≥ 0.6) through the real pipeline.
4. **Text front-end.** Any of:
   * an unknown word (letter rules);
   * a name, meaning capitalised mid-sentence (de: capitalised and unknown);
   * a number, abbreviation or multi-word expansion.
5. **Boundary.** Any of:
   * first or last word;
   * clipped onset or offset;
   * an isolated 5 ms energy spike ≥ 20 dB within 100 ms of the word.
6. **Function / short.** Any of:
   * a spaCy 3.7.5 stop word (MIT licence);
   * Korean: a spaCy ko stop stem, alone or with one particle, of at most 3 syllables;
   * aligned duration under 120 ms, or at most 2 sounds.
7. **Tail.** None of the above, no other lexicon reading, and within 1 logit unit below the threshold.
8. **Residual.**

**Variant sources** (`variants.py`; all Apache-2.0 compatible):

* `lex`: the other readings in the shipped lexicon. CMUdict for en (BSD-2-Clause-style); ipa-dict for de and
  fr (MIT).
* `brit`: Britfone 3.0.1 (MIT), English only. Its IPA is mapped to the front-end's conventions.
* `rule:*`: written optional rules.
  * en: non-rhotic, glottal t, schwa drop.
  * de: -ig as [ɪk], schwa before a sonorant, vocalised r, final -e as schwa, short unstressed tense vowels.
  * es: seseo, distinción, yeísmo, -ado d drop, coda s → h, lenition.
  * fr: schwa drop, mid-vowel open / close, final cluster simplification, back a.
  * ko: coda place assimilation, n-insertion, h drop, ɯi → i, lax-stop voicing, ɔ → ʌ.
* `alt`: the alternatives that the aligner itself chose, promoted to canonical.

No permissive German or Korean pronunciation lexicon beyond the shipped one was found. Wiktionary is
CC BY-SA, so it was not used.

**Counterfactual re-scoring** (`counterfactual.py`):

* The word's sounds are replaced in the Reference, and `Scorer.score` runs again on the same cached
  posteriors.
* Sounds that keep their canonical symbol keep their front-end alternatives.
* Re-scored words: every changed word, plus every correct word scoring below 0.9.
* Check: the canonical form put through the same splice gives the original probability, max |diff| = 0
  (300 words per language).

**Hand check.** For each language, a seeded random sample of 50 false alarms was labelled by reading the
evidence. The reading rules are in the `handcheck.py` docstring, and the labels were written blind to the
automatic bucket.

Evidence printed per word:

* prompt and target;
* Whisper transcript;
* expected sounds with the chosen alternatives;
* what was heard per sound;
* factors and times;
* the free decode ±250 ms around the word.

## 4. Results

### Exclusive buckets, false alarms at FA@80 (% of the population [95 % CI])

| bucket | en (n 641) | de (n 365) | ko (n 217) | es (n 499) | fr (n 768) |
|---|---|---|---|---|---|
| corpus noise | 20.1 [16.1, 24.5] | 10.1 [6.3, 14.1] | 15.2 [8.0, 22.2] | 19.0 [13.6, 24.7] | 33.7 [27.5, 39.2] |
| neighbour of the change | 12.0 [9.1, 15.3] | 23.8 [18.4, 30.0] | 17.1 [8.3, 24.8] | 17.2 [13.5, 21.4] | 9.6 [7.4, 12.4] |
| variant rescues | 12.6 [9.0, 16.3] | 12.3 [7.9, 17.7] | 7.8 [4.9, 12.1] | 2.0 [0.6, 3.9] | 7.8 [5.6, 10.4] |
| text front-end | 29.5 [23.7, 35.5] | 9.9 [5.6, 14.1] | 8.8 [2.7, 19.5] | 28.1 [21.9, 33.9] | 28.3 [23.4, 33.2] |
| boundary | 4.8 [2.9, 7.3] | 6.3 [3.3, 9.5] | 18.4 [11.8, 26.7] | 3.6 [1.6, 5.8] | 4.0 [2.2, 6.1] |
| function / short | 8.0 [5.4, 10.9] | 9.3 [5.5, 13.8] | 6.0 [1.5, 9.9] | 13.4 [8.8, 18.5] | 6.5 [4.1, 9.2] |
| tail | 6.6 [4.3, 8.9] | 9.9 [6.1, 14.4] | 14.7 [9.6, 19.4] | 10.0 [6.6, 13.6] | 5.2 [3.5, 7.1] |
| residual | 6.4 [3.8, 9.5] | 18.4 [13.2, 23.3] | 12.0 [7.3, 16.8] | 6.6 [3.7, 9.8] | 4.8 [3.0, 7.1] |

### Exclusive buckets, correct words below 0.6 (% of the population [95 % CI])

| bucket | en (n 746) | de (n 548) | ko (n 220) | es (n 550) | fr (n 712) |
|---|---|---|---|---|---|
| corpus noise | 19.2 [15.0, 23.3] | 8.6 [5.6, 11.5] | 15.0 [7.9, 22.1] | 17.8 [12.5, 23.8] | 35.0 [28.9, 41.5] |
| neighbour of the change | 12.2 [9.2, 15.2] | 21.4 [17.0, 26.3] | 16.8 [8.2, 24.3] | 17.3 [13.9, 21.4] | 9.8 [7.2, 12.6] |
| variant rescues | 12.2 [9.2, 15.1] | 12.2 [8.7, 15.9] | 7.3 [4.7, 11.3] | 2.4 [0.9, 4.2] | 6.7 [4.5, 9.2] |
| text front-end | 30.0 [24.6, 35.3] | 9.5 [5.9, 13.1] | 8.6 [2.6, 19.3] | 28.4 [22.2, 34.1] | 28.4 [23.3, 33.9] |
| boundary | 4.6 [2.7, 6.6] | 7.7 [5.0, 10.6] | 18.2 [11.6, 26.3] | 3.8 [1.8, 6.0] | 4.4 [2.3, 6.5] |
| function / short | 7.9 [5.4, 10.6] | 10.9 [7.7, 14.5] | 6.8 [2.3, 10.7] | 13.3 [8.8, 18.1] | 6.6 [4.2, 9.4] |
| tail | 6.6 [4.5, 9.0] | 13.0 [9.5, 17.0] | 13.6 [8.5, 18.1] | 10.9 [7.3, 14.4] | 5.3 [3.2, 7.6] |
| residual | 7.4 [5.0, 10.2] | 16.8 [12.6, 20.8] | 13.6 [8.9, 18.7] | 6.2 [3.5, 9.2] | 3.8 [2.1, 5.8] |

### Overlapping hit rate of every test (% of FA; base rate on 3,000 random correct words in brackets)

| test | en | de | ko | es | fr |
|---|---|---|---|---|---|
| noise | 20.1 (3.2) | 10.1 (1.8) | 15.2 (3.9) | 19.0 (1.6) | 33.7 (3.8) |
| punct | 0.6 (0.0) | 1.6 (0.0) | 0.9 (0.0) | 6.6 (0.2) | 18.8 (0.8) |
| label_artifact | 0.2 (0.0) | 0.3 (0.0) | 0.5 (0.0) | 1.6 (0.0) | 2.1 (0.1) |
| tok_artifact | 0.0 (0.0) | 0.0 (0.0) | 4.1 (0.1) | 0.4 (0.0) | 1.3 (0.1) |
| asr_mismatch | 28.1 (3.4) | 8.2 (1.4) | 13.4 (3.6) | 16.4 (1.4) | 19.9 (3.6) |
| asr_noise | 17.6 (2.3) | 8.2 (1.0) | 12.4 (3.5) | 9.6 (0.8) | 11.1 (2.5) |
| low_snr | 0.6 (0.3) | 0.0 (0.0) | 0.0 (0.0) | 1.0 (0.2) | 0.3 (0.1) |
| truncated | 1.6 (0.7) | 0.0 (0.7) | 0.0 (0.3) | 0.0 (0.3) | 0.8 (0.4) |
| neighbour | 22.0 (13.3) | 32.1 (14.1) | 31.8 (15.6) | 28.1 (11.9) | 21.1 (12.5) |
| neighbour_twin_ok | 12.8 (13.0) | 24.4 (13.9) | 18.4 (15.3) | 18.6 (11.7) | 12.6 (12.2) |
| neighbour_omission | 12.0 (2.0) | 23.3 (2.4) | 12.9 (1.7) | 17.0 (1.7) | 10.3 (1.9) |
| has_lex | 18.3 (23.7) | 12.1 (6.0) | 0.0 (0.0) | 0.0 (0.0) | 3.8 (2.4) |
| has_brit | 32.8 (46.9) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) |
| has_rule | 41.2 (25.6) | 59.2 (57.7) | 73.3 (70.0) | 45.3 (45.9) | 58.1 (67.4) |
| has_alt | 21.4 (9.6) | 49.3 (32.6) | 48.8 (29.2) | 9.6 (2.3) | 18.5 (5.4) |
| _rescued | 15.8 | 14.2 | 12.0 | 2.2 | 10.0 |
| text_risk | 37.8 (10.8) | 12.3 (3.5) | 9.2 (1.9) | 33.3 (8.8) | 35.2 (9.3) |
| oov | 12.9 (1.5) | 5.5 (3.0) | 0.0 (0.0) | 20.0 (4.6) | 26.6 (5.6) |
| name | 28.1 (9.4) | 5.2 (2.0) | 0.0 (0.0) | 21.4 (6.4) | 25.3 (6.5) |
| number | 8.3 (1.0) | 6.8 (0.6) | 9.2 (1.9) | 7.4 (0.8) | 3.3 (0.9) |
| boundary | 18.9 (16.8) | 18.1 (20.9) | 35.0 (26.5) | 22.2 (14.7) | 24.2 (18.2) |
| first | 5.1 (6.9) | 3.8 (7.8) | 11.5 (11.4) | 6.8 (6.9) | 4.6 (7.4) |
| last | 11.2 (7.6) | 9.0 (8.7) | 17.1 (10.5) | 14.2 (6.0) | 16.3 (6.8) |
| clip_on | 2.3 (2.9) | 8.5 (13.1) | 1.4 (2.8) | 1.2 (1.8) | 3.3 (2.0) |
| clip_off | 6.7 (6.4) | 0.8 (0.7) | 0.5 (0.6) | 1.0 (0.9) | 0.9 (0.4) |
| click | 3.0 (3.3) | 5.2 (6.0) | 12.4 (8.1) | 1.4 (2.7) | 4.8 (5.5) |
| jitter_gt | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) |
| function | 23.7 (47.6) | 31.0 (50.9) | 8.8 (12.1) | 32.7 (53.9) | 22.8 (47.3) |
| short | 22.5 (32.7) | 27.9 (24.3) | 9.7 (4.6) | 29.7 (32.3) | 44.8 (39.4) |
| tail_zone | 48.8 (1.9) | 43.8 (1.0) | 51.2 (1.0) | 40.5 (0.9) | 33.9 (1.4) |

### Other priority orders (% of FA per bucket)

- en: variant_first: noise 17, neighbour 9, variant 16, text 32, boundary 5, function 8, tail 7, residual 6; boundary_first: noise 15, neighbour 10, variant 10, text 25, boundary 19, function 8, tail 7, residual 6; function_first: noise 11, neighbour 5, variant 12, text 27, boundary 4, function 28, tail 7, residual 6; text_last: noise 20, neighbour 12, variant 13, text 23, boundary 10, function 10, tail 7, residual 6
- de: variant_first: noise 9, neighbour 22, variant 14, text 10, boundary 6, function 9, tail 10, residual 18; boundary_first: noise 8, neighbour 21, variant 9, text 7, boundary 18, function 9, tail 10, residual 18; function_first: noise 4, neighbour 5, variant 10, text 10, boundary 5, function 38, tail 10, residual 18; text_last: noise 10, neighbour 24, variant 12, text 7, boundary 9, function 9, tail 10, residual 18
- ko: variant_first: noise 12, neighbour 16, variant 12, text 9, boundary 18, function 6, tail 15, residual 12; boundary_first: noise 9, neighbour 10, variant 5, text 9, boundary 35, function 6, tail 15, residual 12; function_first: noise 12, neighbour 13, variant 7, text 8, boundary 17, function 16, tail 15, residual 12; text_last: noise 15, neighbour 17, variant 8, text 8, boundary 18, function 6, tail 15, residual 12
- es: variant_first: noise 17, neighbour 14, variant 2, text 32, boundary 4, function 13, tail 10, residual 7; boundary_first: noise 14, neighbour 14, variant 1, text 19, boundary 22, function 13, tail 10, residual 7; function_first: noise 6, neighbour 5, variant 1, text 26, boundary 3, function 43, tail 10, residual 7; text_last: noise 19, neighbour 17, variant 2, text 18, boundary 13, function 15, tail 10, residual 7
- fr: variant_first: noise 31, neighbour 7, variant 10, text 32, boundary 4, function 7, tail 5, residual 5; boundary_first: noise 27, neighbour 8, variant 7, text 17, boundary 24, function 7, tail 5, residual 5; function_first: noise 5, neighbour 4, variant 5, text 25, boundary 2, function 49, tail 5, residual 5; text_last: noise 34, neighbour 10, variant 8, text 15, boundary 15, function 9, tail 5, residual 5

### Candidate shared causes (overlapping families, % of FA [95 % CI])

| family | en | de | ko | es | fr |
|---|---|---|---|---|---|
| presence / unsaid-word handling | 24.2 [19.9, 29.1] | 28.5 [22.7, 35.0] | 13.8 [8.4, 18.7] | 22.4 [18.1, 27.6] | 22.1 [18.2, 26.8] |
| text front-end (number, unknown word, name, punctuation) | 38.4 [32.1, 44.9] | 14.0 [9.2, 18.9] | 10.1 [3.8, 21.9] | 39.9 [33.1, 46.9] | 53.9 [48.2, 59.1] |
| stress factor | 13.4 [10.0, 17.2] | 10.1 [5.9, 14.8] | 0.0 [0.0, 0.0] | 22.2 [17.1, 27.2] | 0.0 [0.0, 0.0] |
| identity step | 0.0 [0.0, 0.0] | 3.3 [1.0, 6.3] | 18.0 [9.9, 26.5] | 12.6 [8.9, 16.6] | 0.0 [0.0, 0.0] |
| union of the first two | 60.2 [54.7, 65.6] | 38.9 [32.0, 45.9] | 23.5 [16.4, 34.1] | 58.7 [51.8, 65.3] | 72.0 [67.4, 76.3] |

### Which evidence drives the low score (FA; all correct words in brackets)

- en: sounds 67 % (11.8 %), present 19 % (1.3 %), stress 13 % (4.7 %), missing 1 % (0.0 %); lowest sound of sound-driven FA: deletion 212, substitution 210, weak 9; caught changed words: substitution 372, deletion 228, weak 15
- de: sounds 64 % (18.4 %), present 21 % (0.8 %), stress 10 % (2.7 %), identity 3 % (0.7 %), missing 2 % (0.0 %); lowest sound of sound-driven FA: substitution 132, deletion 103; caught changed words: substitution 403, deletion 218, weak 6
- ko: sounds 72 % (23.2 %), identity 18 % (2.5 %), present 5 % (0.5 %), missing 3 % (0.2 %), none 2 % (73.7 %); lowest sound of sound-driven FA: deletion 91, substitution 57, weak 8; caught changed words: substitution 129, deletion 77, weak 14
- es: sounds 42 % (5.8 %), stress 22 % (6.3 %), present 17 % (0.6 %), identity 13 % (0.6 %), missing 6 % (0.1 %), none 0 % (86.5 %); lowest sound of sound-driven FA: substitution 112, deletion 92, weak 4; caught changed words: substitution 272, deletion 172, weak 7
- fr: sounds 62 % (8.9 %), missing 19 % (0.7 %), present 18 % (1.5 %), none 1 % (88.9 %); lowest sound of sound-driven FA: substitution 290, deletion 183, weak 5; caught changed words: substitution 386, deletion 173, weak 6

Deletion-type evidence (presence factor drives, or the lowest sound is unsupported and nothing fits better) and extra-sound evidence (llr_ins <= -3), % of FA / % of caught changed words / % of all correct words:

- en: deletion-type 52 / 47 / 5.4; extra-sound 27 / 35 / 6.3
- de: deletion-type 49 / 38 / 4.3; extra-sound 22 / 44 / 9.2
- ko: deletion-type 47 / 27 / 11.1; extra-sound 31 / 39 / 8.0
- es: deletion-type 35 / 35 / 3.2; extra-sound 23 / 44 / 4.3
- fr: deletion-type 42 / 42 / 4.9; extra-sound 30 / 52 / 6.9

### Neighbour effect, paired with the same word of the control item (same audio)

| lang | change kind | pairs | FA % in changed item | FA % in control item |
|---|---|---|---|---|
| en | indel | 480 | 3.5 | 3.1 |
| en | sub_gross | 212 | 5.2 | 4.7 |
| en | sub_moderate | 220 | 2.7 | 3.6 |
| en | sub_subtle | 220 | 2.7 | 2.7 |
| en | word_omission | 194 | 31.4 | 1.0 |
| de | indel | 453 | 2.0 | 1.5 |
| de | sub_gross | 227 | 1.8 | 0.4 |
| de | sub_moderate | 228 | 1.3 | 2.2 |
| de | sub_subtle | 210 | 0.5 | 1.0 |
| de | word_omission | 206 | 35.0 | 1.0 |
| ko | indel | 172 | 3.5 | 2.3 |
| ko | sub_gross | 242 | 3.7 | 4.1 |
| ko | sub_moderate | 158 | 1.9 | 1.9 |
| ko | sub_subtle | 164 | 3.7 | 1.8 |
| ko | word_omission | 84 | 29.8 | 1.2 |
| es | indel | 459 | 3.3 | 2.2 |
| es | sub_gross | 234 | 2.6 | 1.7 |
| es | sub_moderate | 175 | 2.9 | 1.7 |
| es | sub_subtle | 241 | 1.7 | 1.7 |
| es | word_omission | 228 | 31.6 | 3.1 |
| fr | indel | 464 | 4.1 | 2.8 |
| fr | sub_gross | 238 | 3.8 | 1.7 |
| fr | sub_moderate | 207 | 2.4 | 2.4 |
| fr | sub_subtle | 204 | 2.5 | 2.5 |
| fr | word_omission | 228 | 26.8 | 1.8 |

### Word-omission items with the explain-away pass unbounded (existing option; diagnosis only)

| lang | items | neighbour FA % default -> explain-away | inserted word caught % | other words FA % |
|---|---|---|---|---|
| en | 97 | 31.4 -> 3.6 | 100.0 -> 100.0 | 5.6 -> 3.5 |
| de | 103 | 35.0 -> 5.3 | 100.0 -> 100.0 | 2.6 -> 1.9 |
| es | 114 | 31.6 -> 4.8 | 100.0 -> 100.0 | 2.6 -> 1.8 |
| fr | 114 | 26.8 -> 4.8 | 99.1 -> 97.4 | 3.7 -> 2.7 |

### Presence factor below 0.5 (share of words)

| group | en | de | ko | es | fr |
|---|---|---|---|---|---|
| inserted unsaid word (changed) | 96.9 (n 97) | 92.2 (n 103) | 66.7 (n 42) | 92.0 (n 113) | 93.0 (n 114) |
| other changed words | 8.7 (n 623) | 7.0 (n 617) | 3.3 (n 368) | 7.6 (n 606) | 11.3 (n 602) |
| correct function words, not neighbours | 0.7 (n 7910) | 0.1 (n 7955) | 0.1 (n 992) | 0.1 (n 10031) | 0.5 (n 8692) |
| correct content words, not neighbours | 0.2 (n 8471) | 0.1 (n 7508) | 0.1 (n 6599) | 0.0 (n 8580) | 0.2 (n 9013) |
| within two words of an inserted word | 12.6 (n 365) | 14.9 (n 396) | 7.1 (n 154) | 13.1 (n 428) | 10.8 (n 424) |

### Best of N alternative pronunciations (counterfactual re-scoring; every word re-scored takes its best)

| lang | variants | AUC | FA@80 % | new threshold | FA rescued % | caught changed words lifted out % | changed raised |
|---|---|---|---|---|---|---|---|
| en | none | 0.9641 | 3.40 | 0.528 | - | - | - |
| en | all | 0.9563 | 3.59 | 0.630 | 15.8 | 4.5 | 146 |
| en | lex | 0.9645 | 3.43 | 0.544 | 2.5 | 0.5 | 25 |
| en | brit | 0.9634 | 3.52 | 0.569 | 6.4 | 2.8 | 74 |
| en | rule | 0.9601 | 3.43 | 0.556 | 7.3 | 1.2 | 55 |
| en | alt | 0.9602 | 3.51 | 0.548 | 1.9 | 0.7 | 40 |
| de | none | 0.9757 | 2.03 | 0.458 | - | - | - |
| de | all | 0.9695 | 2.09 | 0.522 | 14.2 | 2.3 | 181 |
| de | lex | 0.9763 | 1.90 | 0.458 | 6.3 | 0.2 | 23 |
| de | rule | 0.9736 | 2.26 | 0.518 | 5.2 | 1.9 | 115 |
| de | alt | 0.9709 | 1.95 | 0.458 | 4.1 | 0.2 | 69 |
| ko | none | 0.9728 | 2.39 | 0.587 | - | - | - |
| ko | all | 0.9673 | 2.51 | 0.651 | 12.0 | 1.8 | 120 |
| ko | rule | 0.9725 | 2.68 | 0.651 | 5.5 | 1.8 | 74 |
| ko | alt | 0.9676 | 2.30 | 0.604 | 6.5 | 0.3 | 56 |
| es | none | 0.9662 | 2.36 | 0.552 | - | - | - |
| es | all | 0.9415 | 2.51 | 0.597 | 2.2 | 1.6 | 82 |
| es | rule | 0.9662 | 2.36 | 0.552 | 0.0 | 0.0 | 21 |
| es | alt | 0.9415 | 2.52 | 0.597 | 2.2 | 1.6 | 62 |
| fr | none | 0.9421 | 3.77 | 0.645 | - | - | - |
| fr | all | 0.9398 | 3.51 | 0.670 | 10.0 | 0.7 | 147 |
| fr | lex | 0.9425 | 3.67 | 0.645 | 2.7 | 0.0 | 1 |
| fr | rule | 0.9392 | 3.65 | 0.670 | 6.6 | 0.7 | 141 |
| fr | alt | 0.9422 | 3.72 | 0.645 | 1.4 | 0.0 | 10 |

### Rescues by variant kind (FA rescued by the best alternative)

- en: brit 38, rule:schwa_drop 21, rule:nonrhotic 18, lex 12, alt 11, rule:t_glottal 1
- de: lex 22, alt 15, rule:schwa_sonorant 7, rule:short_unstressed 6, rule:r_vocalised 1, rule:final_e_schwa 1
- ko: alt 14, rule:n_insertion 7, rule:place_assim 2, rule:h_drop 2, rule:lax_voicing 1
- es: alt 11
- fr: lex 19, rule:schwa_drop 18, rule:mid_close 17, rule:mid_open 12, alt 9, rule:a_back 2

### FA rate by number of sounds (correct words) and catch rate (changed words) at the one threshold

| sounds | en FA % / catch % | de FA % / catch % | ko FA % / catch % | es FA % / catch % | fr FA % / catch % |
|---|---|---|---|---|---|
| 1 | 2.8 / 50 (n 462) | 1.6 / 54 (n 189) | 0.0 / 0 (n 73) | 1.2 / 9 (n 1150) | 3.2 / 16 (n 1632) |
| 2 | 1.8 / 63 (n 5272) | 1.7 / 75 (n 3672) | 3.3 / 81 (n 363) | 1.4 / 71 (n 5526) | 1.5 / 74 (n 5888) |
| 3 | 1.7 / 79 (n 4092) | 1.5 / 80 (n 3483) | 1.9 / 75 (n 425) | 2.8 / 77 (n 2616) | 2.7 / 84 (n 3521) |
| 4 | 3.8 / 86 (n 4210) | 1.8 / 86 (n 3852) | 2.0 / 84 (n 2386) | 2.1 / 85 (n 3806) | 3.9 / 84 (n 4263) |
| 6 | 4.5 / 88 (n 2838) | 2.4 / 85 (n 2838) | 1.9 / 74 (n 2412) | 2.3 / 92 (n 4001) | 4.3 / 84 (n 3199) |
| 8 | 6.6 / 90 (n 1386) | 2.6 / 76 (n 2075) | 2.4 / 83 (n 1843) | 3.3 / 87 (n 2400) | 3.9 / 95 (n 1289) |
| 10 | 11.5 / 80 (n 459) | 2.3 / 74 (n 1022) | 3.3 / 100 (n 939) | 2.3 / 77 (n 1160) | 6.0 / 67 (n 317) |
| 12+ | 15.4 / 67 (n 156) | 3.8 / 70 (n 838) | 3.6 / 100 (n 617) | 5.1 / 93 (n 492) | 15.4 / 100 (n 123) |

### FA rate by aligned duration (ms): function words / content words (% FA, n in brackets)

| bin | en | de | ko | es | fr |
|---|---|---|---|---|---|
| 0-50 | 10.6 (479) / 42.9 (14) | 60.4 (48) / 100.0 (7) | 3.0 (67) / 28.0 (25) | 4.2 (1287) / 75.5 (49) | 5.9 (1497) / 88.0 (166) |
| 50-100 | 1.4 (3250) / 6.2 (48) | 2.9 (1069) / 34.8 (23) | 1.9 (156) / 5.3 (76) | 0.8 (4326) / 23.1 (26) | 0.9 (4638) / 10.2 (246) |
| 100-150 | 0.9 (2428) / 6.7 (134) | 0.5 (3194) / 2.4 (42) | 2.9 (136) / 7.9 (89) | 1.2 (2178) / 12.9 (62) | 0.8 (1922) / 8.0 (490) |
| 150-200 | 1.0 (1294) / 4.6 (413) | 0.6 (2251) / 5.4 (203) | 3.3 (121) / 3.0 (297) | 2.2 (1231) / 12.0 (166) | 0.8 (847) / 4.6 (821) |
| 200-250 | 0.7 (732) / 1.7 (725) | 0.5 (1109) / 3.2 (406) | 1.5 (199) / 1.6 (499) | 0.7 (893) / 3.0 (438) | 1.7 (357) / 3.7 (1064) |
| 250-300 | 1.2 (409) / 3.5 (1120) | 1.0 (705) / 3.0 (642) | 0.0 (208) / 2.7 (778) | 1.1 (526) / 1.5 (863) | 2.1 (235) / 3.3 (1201) |
| 300-350 | 1.8 (228) / 2.8 (1373) | 1.1 (438) / 2.3 (822) | 0.6 (167) / 2.6 (958) | 2.4 (331) / 2.4 (1021) | 1.9 (161) / 3.1 (1275) |
| 350-400 | 0.6 (155) / 4.3 (1327) | 1.0 (198) / 2.9 (931) | 3.0 (67) / 3.3 (1014) | 0.9 (230) / 3.8 (1132) | 0.0 (103) / 4.4 (1214) |
| 400-450 | 0.0 (59) / 4.2 (1136) | 2.7 (112) / 2.2 (955) | 0.0 (21) / 1.7 (959) | 0.0 (170) / 1.8 (1215) | 0.0 (73) / 3.4 (1027) |
| 450-500 | 6.2 (32) / 5.1 (806) | 1.3 (79) / 2.8 (702) | 0.0 (6) / 1.2 (688) | 0.0 (67) / 2.4 (1002) | 5.7 (35) / 2.0 (739) |
| 500-550 | 10.0 (20) / 5.8 (874) | 0.0 (37) / 2.2 (874) | 0.0 (4) / 1.3 (770) | 0.0 (44) / 2.5 (1180) | 10.5 (19) / 2.8 (793) |
| 550-600 | 0.0 (6) / 8.5 (529) | 0.0 (8) / 1.8 (720) | 0.0 (4) / 2.9 (523) | 0.0 (20) / 2.9 (823) | 6.7 (15) / 3.6 (475) |
| 600-650 | 8.3 (24) / 9.5 (1264) | 0.0 (3) / 3.1 (2397) | 0.0 (2) / 2.8 (1238) | 0.0 (15) / 3.8 (1887) | 33.3 (12) / 8.8 (951) |

### FA rate by sounds per second: function words / content words (% FA, n in brackets)

| bin | en | de | ko | es | fr |
|---|---|---|---|---|---|
| 0-5 | 10.0 (10) / 22.2 (18) | - / - | - / 18.2 (11) | 0.0 (4) / 60.0 (5) | 0.0 (2) / 0.0 (2) |
| 5-10 | 1.8 (336) / 6.4 (1022) | 1.3 (398) / 5.5 (365) | 0.0 (6) / 5.4 (56) | 0.0 (39) / 8.6 (175) | 6.1 (66) / 7.6 (473) |
| 10-15 | 1.1 (1606) / 4.6 (4811) | 0.4 (2525) / 2.2 (3921) | 1.3 (228) / 1.7 (1853) | 0.5 (651) / 2.6 (3632) | 1.7 (819) / 4.1 (4378) |
| 15-20 | 0.9 (2458) / 4.0 (3300) | 0.7 (3269) / 2.7 (3710) | 0.6 (509) / 1.8 (4457) | 1.0 (3170) / 2.5 (5307) | 0.6 (1987) / 2.8 (4300) |
| 20-25 | 0.8 (2778) / 6.8 (530) | 1.1 (2031) / 4.1 (665) | 1.4 (290) / 4.0 (1317) | 0.7 (4971) / 5.4 (633) | 0.9 (3729) / 6.8 (910) |
| 25-30 | 0.3 (672) / 26.0 (50) | 0.3 (593) / 17.1 (41) | 5.4 (56) / 8.6 (162) | 1.8 (842) / 18.0 (50) | 0.8 (918) / 9.7 (155) |
| 30-35 | 1.7 (811) / 36.4 (11) | 3.7 (294) / 0.0 (10) | 11.1 (27) / 22.2 (18) | 1.8 (811) / 62.5 (8) | 1.6 (1092) / 28.0 (50) |
| 35-40 | 5.3 (75) / 100.0 (4) | 6.6 (61) / 0.0 (2) | 0.0 (2) / 27.3 (11) | 19.2 (26) / 0.0 (2) | 2.2 (90) / 57.9 (19) |
| 40-45 | 16.5 (370) / 38.5 (13) | 42.3 (78) / 100.0 (4) | 7.5 (40) / 13.3 (15) | 7.4 (781) / 42.9 (21) | 6.9 (1211) / 35.5 (31) |

### Effect of the change by distance (FA % in the changed item vs the same word in the control item)

| change / distance | en | de | ko | es | fr |
|---|---|---|---|---|---|
| omission|1 | 31.4 vs 1.0 (n 194) | 35.0 vs 1.0 (n 206) | 29.8 vs 1.2 (n 84) | 31.6 vs 3.1 (n 228) | 26.8 vs 1.8 (n 228) |
| omission|2 | 9.4 vs 2.3 (n 171) | 6.8 vs 1.6 (n 191) | 4.3 vs 2.9 (n 70) | 6.5 vs 2.5 (n 200) | 9.1 vs 3.5 (n 198) |
| omission|3 | 5.9 vs 3.3 (n 153) | 1.9 vs 1.9 (n 158) | 6.2 vs 4.2 (n 48) | 1.7 vs 1.1 (n 174) | 2.9 vs 2.3 (n 175) |
| omission|4+ | 4.8 vs 3.6 (n 775) | 1.6 vs 1.9 (n 673) | 1.2 vs 0.6 (n 167) | 2.0 vs 1.6 (n 1040) | 2.8 vs 2.8 (n 998) |
| other|1 | 3.5 vs 3.4 (n 1132) | 1.5 vs 1.3 (n 1118) | 3.3 vs 2.7 (n 736) | 2.7 vs 1.9 (n 1109) | 3.4 vs 2.4 (n 1113) |
| other|2 | 2.4 vs 2.1 (n 997) | 1.5 vs 1.5 (n 992) | 2.9 vs 2.4 (n 577) | 2.5 vs 2.2 (n 1010) | 4.5 vs 3.6 (n 1003) |
| other|3 | 2.1 vs 2.3 (n 877) | 1.6 vs 1.5 (n 872) | 1.9 vs 1.7 (n 469) | 1.5 vs 1.4 (n 910) | 3.9 vs 4.6 (n 879) |
| other|4+ | 3.2 vs 3.2 (n 4829) | 1.7 vs 1.7 (n 4469) | 1.1 vs 1.1 (n 1583) | 2.0 vs 2.0 (n 5617) | 3.6 vs 3.6 (n 5291) |

### Function vs content words

- en: function words 48 % of correct words, FA rate 1.7 %; content words FA rate 5.0 %
- de: function words 51 % of correct words, FA rate 1.2 %; content words FA rate 2.9 %
- ko: function words 13 % of correct words, FA rate 1.6 %; content words FA rate 2.5 %
- es: function words 53 % of correct words, FA rate 1.4 %; content words FA rate 3.4 %
- fr: function words 49 % of correct words, FA rate 1.8 %; content words FA rate 5.7 %

### Tail: correct words by distance to the threshold (logit units of the score; % of correct words)

| logit(threshold score) - logit(score) (> 0: false alarm) | en | de | ko | es | fr |
|---|---|---|---|---|---|
| -3..-2 | 14.80 | 13.80 | 5.94 | 7.02 | 10.53 |
| -2..-1 | 5.28 | 5.05 | 3.59 | 3.53 | 4.11 |
| -1..-0.5 | 1.31 | 1.26 | 0.90 | 1.09 | 1.06 |
| -0.5..0 | 0.99 | 0.80 | 0.78 | 0.77 | 0.72 |
| 0..0.5 | 0.97 | 0.48 | 0.73 | 0.62 | 0.65 |
| 0.5..1 | 0.67 | 0.41 | 0.50 | 0.32 | 0.62 |
| 1..2 | 0.72 | 0.46 | 0.42 | 0.51 | 0.57 |
| 2..4 | 0.64 | 0.39 | 0.47 | 0.49 | 0.84 |
| 4..99 | 0.38 | 0.29 | 0.28 | 0.41 | 1.08 |
| FA@80 % if the threshold moved by -0.5 / -0.25 / 0 / +0.25 / +0.5 logit | 4.37 / 3.87 / 3.40 / 2.90 / 2.41 | 2.83 / 2.38 / 2.03 / 1.77 / 1.55 | 3.17 / 2.82 / 2.39 / 2.04 / 1.66 | 3.11 / 2.68 / 2.36 / 2.01 / 1.72 | 4.48 / 4.09 / 3.77 / 3.46 / 3.11 |

Hand check en: n 50, agreement 50 %; hand counts {'neighbour': 9, 'tail': 5, 'variant': 6, 'text': 14, 'residual': 9, 'noise': 4, 'function': 3}; precision of the automatic bucket: boundary 0.00 (5), function 0.17 (6), neighbour 0.88 (8), noise 0.33 (6), residual 1.00 (2), tail 0.25 (4), text 0.71 (14), variant 0.40 (5)

Hand check de: n 50, agreement 54 %; hand counts {'text': 15, 'neighbour': 12, 'residual': 10, 'function': 3, 'noise': 3, 'variant': 5, 'tail': 2}; precision of the automatic bucket: boundary 0.00 (4), function 0.00 (4), neighbour 0.79 (14), noise 0.60 (5), residual 1.00 (6), tail 0.14 (7), text 0.83 (6), variant 0.25 (4)

Hand check ko: n 50, agreement 40 %; hand counts {'neighbour': 10, 'noise': 10, 'residual': 11, 'tail': 6, 'variant': 9, 'text': 4}; precision of the automatic bucket: boundary 0.00 (7), function 0.00 (3), neighbour 1.00 (8), noise 0.71 (7), residual 0.40 (5), tail 0.11 (9), text 0.33 (6), variant 0.40 (5)

Hand check es: n 50, agreement 66 %; hand counts {'text': 20, 'variant': 5, 'tail': 2, 'noise': 7, 'residual': 4, 'neighbour': 12}; precision of the automatic bucket: boundary 0.00 (3), function 0.00 (5), neighbour 0.92 (12), noise 0.71 (7), residual 0.33 (6), tail 0.50 (2), text 1.00 (14), variant 0.00 (1)

Hand check fr: n 50, agreement 70 %; hand counts {'text': 20, 'variant': 4, 'neighbour': 5, 'tail': 4, 'noise': 12, 'residual': 4, 'function': 1}; precision of the automatic bucket: boundary 0.00 (1), function 0.00 (1), neighbour 1.00 (4), noise 0.69 (16), residual 0.00 (3), tail 0.67 (3), text 0.90 (20), variant 0.00 (2)

## 5. Ten informative patterns per language

The definitions are fixed and the same for every language (`patterns.py`). These are patterns, not word
lists. Each line gives the share of false alarms (%) and the FA rate inside the pattern against the rest.

**English** (641 false alarms)

1. Neighbour of an unsaid word: 31 % flagged vs 1.0 % in the control twin. In 40 of the 61 such false
   alarms at distance 1, the presence factor drives the score.
2. Unknown words read by letter rules: FA rate 31 % vs 3 %.
3. Numbers read another way: FA rate 29 %. Examples: "2010" as "two thousand and ten"; "116" with a reduced
   "and"; "World War I" read as "one".
4. Names in the lexicon: FA rate 7.7 % vs 3.1 %. Foreign names are read the native-language way
   (Versace, Lisboa, Raja, Guinness with [dʒ]).
5. Function words that the presence factor judges absent: 15 % of false alarms; FA rate 48 % once judged
   absent. The lowest sounds are ə, ɪ, ∅.
6. The stress factor drives the score while the sounds are fine (all sounds ≥ 0.9): 13 % of false alarms.
   Examples: "participated", "Alexander".
7. Whisper disagrees at the word: 28 % of false alarms (base 3.4 %). In the hand check, only about a third
   of these were plainly misread. Accented Common Voice clips make Whisper unreliable here.
8. Long words: FA rate rises from 1.7 % (3 sounds) to 11.5 % (10 sounds) and 15 % (12+), while the catch
   rate of changed words stays at 80-90 %. With names, unknown words and numbers removed, it still rises from
   1.6 % to 6.0 %.
9. Content words with an unsupported sound (deletion-type): FA rate 31 %. Coda and cluster consonants are
   dropped in fluent speech: k, h, n, t, d → ∅.
10. Accent vowels: water [wɔtə], Javanese [æ], "Antarctic" without the first c. Britfone rescues 38 false
    alarms, but it also lifts 74 changed words.

**German** (365 false alarms)

1. Neighbour of an unsaid word: 35 % vs 1.0 %. This is the largest single cause, at 24 % of false alarms in
   the exclusive buckets.
2. Function words judged absent by the presence factor: FA rate 59 %.
3. First lexicon reading is wrong while a later reading is right. Example: "tun" read as /tjuːn/ because
   ipa-dict lists it first. Best-of-N over lexicon readings is the one variant source with a net gain
   (FA@80 2.03 → 1.90 %).
4. Numbers and abbreviations: FA rate 25 %. Examples: "100" read as "hundert" rather than "einhundert"; "m"
   read as "Meter".
5. Loanwords and English names: Library, Fonds, Curry, Low, Burschi, Philip. Some come out of the letter
   rules or English-reading path wrong; others carry the wrong stress.
6. The stress factor drives the score in hyphenated or compound names (Nicht-Roma, Vinson-Massiv): 10 % of
   false alarms.
7. Lattice used a front-end alternative: 49 % of false alarms vs 33 % of words. Alternatives are taken but
   still cost probability.
8. Content words with another sound heard: ŋ̩ → ən, eː → ɛ, ŋ → g. These are mostly syllabic-nasal and vowel
   conventions.
9. The identity step drives the score: 3 % (FA rate 10 % when it drives).
10. Residual. 20 % in the exclusive buckets, the highest of the five languages: single-vowel quality
    differences and cluster reductions (Schildkröte without l).

**Korean** (217 false alarms)

1. Neighbour of an unsaid word: 30 % vs 1.2 %.
2. The identity step drives the score: 18 % of false alarms; FA rate 17 % when it drives (e → i, i → o).
3. Regular sound changes missing from the front-end: n-insertion and nasalisation, as in 꽃잎 [꼰닙],
   56가지 [오심뉵], 116명 [뱅심뉵], 진리 read as [진니]. The automatic rules rescue 12 false alarms; the
   hand check finds 9 of 50.
4. Whisper disagrees with the prompt even after the CER ≤ 0.1 filter: 13 % of false alarms. Examples:
   갔는가 read as 갔다는가, 끈을 as 끝을, parentheticals skipped.
5. Numbers read Sino-Korean vs native: 3개가 read as "세 개", 2개로 as "두 개". FA rate 13 %.
6. Clicks within 100 ms of the word: 12 % of false alarms vs 8 % of words. Weakly enriched; the hand check
   found none to be the cause.
7. Tokens mixing Latin and Hangul (Bhog(힌두교에서는, Siminoff)는): metric tokenisation artifacts, 4 %.
8. Tail: 15 % of false alarms sit within 1 logit below the threshold.
9. Loanwords (트레이드오프는, 컴퓨터를).
10. Last word of the clip: 17 % vs 10 %.

**Spanish** (499 false alarms)

1. The stress factor drives the score while the sounds are fine: 22 % of false alarms. Examples: "debe",
   "Adam", "%", "2000".
2. The identity step drives the score: 13 %; FA rate 50 % when it drives.
3. Neighbour of an unsaid word: 32 % vs 3.1 %.
4. Function words judged absent: FA rate 61 %.
5. The glide in "muy" and "cuando" is written w; ZIPA writes u. This w → u pair causes 35 false alarms. It
   is a front-end convention.
6. Unknown words and English or foreign names (Bulletin, Barnard, Flood, Jeffrey): FA rate 10 %.
7. Numbers and symbols ("1767", "%", "et al."): FA rate 25 %.
8. Punctuation-only tokens (¿ ¡): 7 % of false alarms.
9. Misreads that Whisper confirms ("sirve para guía" for "sirve de guía", "escapan"). The prompt typo
   "le jerga" is also counted here.
10. Last word of the clip: 14 % vs 6 %.

**French** (768 false alarms)

1. Punctuation-only tokens: 19 % of false alarms (metric artifact).
2. Unknown words read by letter rules: FA rate 20 %. Many are names: Hong, Greg, Shri, Siddhārtha,
   Jean read as /dʒin/, blog read as /blɔiŋ/.
3. Content words with schwa elided (ə → ∅) and other vowel changes: 33 % of false alarms.
4. Function words judged absent: FA rate 40 %.
5. Neighbour of an unsaid word: 27 % vs 1.8 %.
6. "six" and "dix" before a consonant, said [si] and [di]. These are regular variants.
7. Optional liaison not made ("ont été" without [t]).
8. Numbers with a no-break space ("1 400", "7 000") are split into two tokens. One token is always scored
   as missing.
9. Rule variants (schwa drop, mid vowels) rescue 47 false alarms; best-of-N lowers FA@80 3.77 → 3.51 % but
   also AUC.
10. Last word of the clip: 16 % vs 7 %.

## 6. Answer

**Do English, German and Korean share a root cause?** Yes, one: the handling of words the speaker did not
say.

* The paired control-twin test shows the same effect in all three languages: en 31 % vs 1 %, de 35 % vs 1 %,
  ko 30 % vs 1 %.
* The existing explain-away pass removes it where it could be run (en, de; also es, fr).
* For Korean, the explain-away pass crashes on main. It raises `KeyError: 'head_cap_z'` in
  `Scorer.explain_away`, because the head-cap column exists only in the first pass. The pass is off by
  default for Korean, so this is a latent bug.

**Spanish and French** show the same cause at the same size: 32 % and 27 % vs 2-3 %.

**This cause is not native-speech-specific, but it matters for this benchmark.** The benchmark's changed
items include the same "word omission" kind. Every such item creates a neighbour false alarm about a third
of the time. In our practice sets this family is 22-29 % of false alarms (en/de/es/fr) and 14 % (ko). Its
share in the benchmark depends on its share of omission items, which was not looked at.

**The second shared cause is the text front-end** (numbers, unknown words, names):

* 38 % of false alarms in English, 40 % in Spanish, 54 % in French (French including punctuation tokens);
* 14 % in German and 10 % in Korean.

It explains most of why English is worse than German at the same catch rate:

* English names have an FA rate of 7.7 %, unknown words 31 %, numbers 29 %.
* German names 5 %, unknown words 4 %, numbers 25 %.
* In German, the remaining correct words have an FA rate of 1.8 %; in English, 2.3 %.

**Worth a fix (my view; for you to decide):**

1. **Unsaid-word handling.** A word-level skip path, or a second pass triggered by strong omission
   evidence. This differs from today's explain-away, which wildcards the lowest word in general and has
   cost elsewhere: it lowered speechocean762 phone and word Pearson in round 3. German runs a bounded
   form whose band excludes exactly these clearly absent words. Best case on practice: 82-89 % of the
   neighbour false alarms (the neighbour FA drops from 27-35 % to 3.6-5.3 %), i.e. roughly 8-20 % of all
   false alarms.
2. **Text front-end readings.**
   * Alternative number readings: with and without "and"; year vs cardinal; Roman numerals; "hundert" vs
     "einhundert"; Sino vs native Korean.
   * Lexicon readings tried as variants (the German "tun" case).
   * Korean regular n-insertion and nasalisation.
   * The Spanish w/u glide convention.

   These are general rules of each language, not word lists.
3. **Measurement fixes in the practice sets / metric** (not the scorer): punctuation-only tokens, the
   word-index bug, and no-break-space numbers. Whether the benchmark itself has punctuation-only tokens was
   not checked (that would mean looking at its prompts).

**Probably irreducible, or not ours:**

* corpus noise: misreads confirmed by Whisper, skipped parentheticals, prompt typos;
* the tail: about 1 % of correct words lie within 0.5 logit of the threshold in every language, and the
  shape is similar across languages;
* residual single-sound differences.

**Not supported as main causes:**

* boundary effects: precision 0 % in every hand check, and only a mild enrichment of last words;
* reduced function words through the sound evidence: function words have the lower FA rate;
* broad best-of-N variants: they lift changed words almost as much as correct ones.

The stress factor (en, es) and the identity step (ko, es) are real, language-specific drivers. A fix there
would be a per-language decision, not a shared one.

## 7. What was not done

* **Alignment jitter against an independent word span:** not done. Whisper word times blew up memory and
  were dropped.
* **Breath detection:** not attempted. Clicks were tested.
* **Explain-away check for Korean:** crashes on main, as noted above. Code was not changed to work around
  it.
* **Function-word definition:** the spaCy stop lists are broader than strict closed classes (they include
  some frequent adverbs and verbs). No part-of-speech tagger was installed.
* **Hand-check agreement with the automatic buckets** is moderate: en 50 %, de 54 %, ko 40 %, es 66 %,
  fr 70 %.
  * The neighbour bucket is precise: 0.79-1.0.
  * The text bucket mostly is: 0.33-1.0, low in Korean.
  * Boundary (0.0), function (0-0.17) and tail are not reliable. Hand reading moves most of them to
    text, variant or residual.
  * The English Whisper test is noisy (precision 0.33).
