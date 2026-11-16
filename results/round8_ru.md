# Round 8: Russian reduction-aware references, retrained adapter, detection-aware sound bias

**Outcome.** The A + B + C′ package was adopted on dev, by +0.020 near-swap AUC (bar +0.02). It
was then frozen and run once on held-out data, Russian only:
* FLEURS-ru swap near-swap AUC: 0.872 → **0.887** [0.860, 0.914]. Azure is 0.909, so we still
  trail it.
* cv-real ru (Test A): **worse**, 0.793 → 0.772. Nothing was tuned on test.

The package:
* **A**: measured, reduction-aware vowel and t/d targets;
* **B**: an adapter retrained on those targets;
* **C′**: the Russian sound bias chosen by a detection-aware rule (+1.075 → **+0.60**), with the
  stress channel refitted.

Selection used FLEURS-ru **validation** (dev) only; the adapter was trained on FLEURS-ru train.
No Azure requests were made; Azure numbers are its cached responses.

## 1. Diagnosis

Round 7 on the swap test: near-swap AUC 0.872 for us, 0.909 for Azure.

* **The tail.** 36/150 replaced words score > 0.9 with us, 9/150 with Azure. Mostly these are
  unstressed inflectional endings.
* **Invisible by construction.** The round-7 front-end gives every unstressed vowel broad
  hand-written alternatives: unstressed `ɨ` accepts `(ɪ, a)`, unstressed `a` accepts
  `(ɐ, ə, ɑ, ʌ)`. So годы/года cannot be told apart.
* **Detected but forgiven.** On the misses that are distinct in the references (является →
  являются, имеет → имеют), the replaced sound gets strong evidence against it (llr_sub ≈ −4.5
  to −5.6, heard ɛ for ю). Yet its probability is still 0.95–0.98. The universal sound calibrator
  plus the largest per-language bias of any language (+1.075) maps that evidence to "fine".

## 2. What changed

### A. Measured references

Built with `python -m bench.calib.ru_round8 audit` (table in
`results/tables/ru_round8_audit_raw.json`, shipped as `pronunciada/data/lex/ru_vowels.json`).

How the audit works:
* It covers every vowel and every t/tʲ/d/dʲ slot of the 355 validation references, in words with
  known stress.
* Each slot is offered all candidate symbols with equal weight, and the Viterbi path through the
  **raw** ZIPA posteriors picks one.
* Counts are kept per (letter, position, left context) for vowels and per (origin, next sound)
  for t/d.

What it shows:
* ZIPA almost never emits ɐ / ə / ʊ for Russian.
* Final -ы and -а are separated: ы → ɨ 88 % / a 8 %; а → a 87 % / ɨ 4 %.
* Final -е after a soft consonant is mixed: ɛ 48 % / i 39 %.
* Final -т is heard as ts 25 % of the time, and final -ть as plain t 13 %.

The new rule (`pronunciada/text/ru.py`, `MEASURED_TARGETS = True`):
* The primary symbol is the most frequent realisation in the slot's class.
* An alternative is kept only if natives produce it in ≥ 15 % of cases (`VOWEL_MIN`, chosen on
  dev from 0.05 to 0.25). Its log prior is log(share / primary share).
* It applies only to known-stress words: lexicon, accent mark, ё, one vowel, or an unstressed
  clitic.
* Words with predicted stress keep the round-7 broad alternatives. That is 19 % of dev words:
  12.9 % stem-predicted, 6.2 % fully out of vocabulary.
* `get_g2p("ru").measured_targets = False` restores the round-7 references bit for bit (checked
  on 1 205 texts).

### B. Retrained adapter

* Trained with `bench.pipeline.train_adapter --lang ru`, FLEURS-ru train, existing recipe; conv
  was selected again.
* Controlled-error mean AUC 0.9848, against 0.9857 for the round-7 adapter on the round-7
  references.
* Selection record: `results/tables/adapter_ru.json`, reconstructed from the run's cached
  candidates.

### C′. Bias rule

The rule was fixed before the grid was looked at:
* grid 0 … +1.1 in steps of 0.1, plus the shipped +1.075, with A + B in place;
* for each value, the stress channel is refitted;
* a value is feasible only if, versus round 7, all of these hold: native words and sounds at
  default strictness drop ≤ 0.5 points, planted sound-error and word-error AUC drop ≤ 0.005, and
  stress-shift AUC drops ≤ 0.010;
* among feasible values, take the highest dev near-swap AUC; ties go to the larger bias.

How it was run (`bench/calib/bias_grid.py`):
* Alignment and features are computed once per reference. The calibrators are re-run per bias
  through `Scorer._calibrate`.
* The stress channel is refitted per bias by `fit_prosody`'s own `evaluate`/`fit`, on rows
  rebuilt in its exact format (same 200 clips, seeds and leniency rule).
* At the round-7 configuration this reproduces the shipped `prosody/ru.json` coefficients
  exactly, and the dev metrics of `ru_round8 eval` to 4 decimals.
* The frozen package run end to end through `Scorer` gives the same dev numbers: near AUC
  0.8915, native words 0.9765, stress AUC 0.8614.

Shipped:
* `pronunciada/data/adapters/ru.npz`;
* `pronunciada/data/calib/config.json`: bias ru = 0.6, plus a `bias_note`;
* `pronunciada/data/calib/prosody/ru.json`: refitted at bias 0.6, native quantiles
  (0.0025, 0.01).

No other language's files changed.

## 3. Dev results (FLEURS-ru validation)

The dev set is 355 clips with planted errors, plus a swap set of 349 near and 349 far items built
from validation sentences by the fleurs-swap recipe and frozen before any change.

### Part 1: references and adapter at the round-7 bias, round-7 stress channel

| | round 7 | A τ=.15 (old adapter) | A+B τ=.10 | A+B τ=.15 |
|---|---|---|---|---|
| near AUC | 0.871 | 0.877 | 0.881 | 0.883 |
| replaced > 0.9 | 24.9 % | 22.1 % | 20.6 % | 20.3 % |
| native words | 0.9722 | 0.9694 | 0.9711 | 0.9707 |
| planted sub_near AUC | 0.9679 | 0.9626 | 0.9707 | 0.9681 |

The adapter is part of the blur. With the same references, retraining it adds about +0.005 near
AUC and removes the planted-error loss that stricter references cause with the old adapter.

### Bias grid, A + B (τ = .15), stress channel refitted per value

| bias | near AUC | > 0.9 | hit@5 % | right word | native words | native sounds | sound AUC | word AUC | stress AUC | feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| round 7 (+1.075) | 0.8713 | 0.249 | 0.559 | 0.490 | 0.9722 | 0.9997 | 0.9811 | 0.9231 | 0.8710 | – |
| 0.0 | 0.9054 | 0.163 | 0.650 | 0.527 | 0.9773 | 0.9965 | 0.9814 | 0.9252 | 0.8273 | no (stress) |
| 0.1 | 0.8979 | 0.163 | 0.633 | 0.530 | 0.9751 | 0.9972 | 0.9814 | 0.9206 | 0.8531 | no (stress) |
| 0.2 | 0.8962 | 0.175 | 0.630 | 0.519 | 0.9762 | 0.9976 | 0.9814 | 0.9208 | 0.8557 | no (stress) |
| 0.3 | 0.8949 | 0.183 | 0.630 | 0.524 | 0.9762 | 0.9981 | 0.9814 | 0.9214 | 0.8565 | no (stress) |
| 0.4 | 0.8944 | 0.195 | 0.633 | 0.513 | 0.9765 | 0.9985 | 0.9814 | 0.9214 | 0.8583 | no (stress) |
| 0.5 | 0.8936 | 0.192 | 0.628 | 0.519 | 0.9771 | 0.9988 | 0.9814 | 0.9217 | 0.8598 | no (stress) |
| **0.6** | **0.8915** | 0.203 | 0.630 | 0.527 | 0.9765 | 0.9990 | 0.9814 | 0.9217 | 0.8614 | **yes** |
| 0.7 | 0.8893 | 0.212 | 0.628 | 0.521 | 0.9759 | 0.9993 | 0.9814 | 0.9221 | 0.8619 | yes |
| 0.8 | 0.8897 | 0.212 | 0.633 | 0.521 | 0.9756 | 0.9995 | 0.9814 | 0.9223 | 0.8623 | yes |
| 0.9 | 0.8872 | 0.215 | 0.630 | 0.521 | 0.9760 | 0.9997 | 0.9814 | 0.9226 | 0.8630 | yes |
| 1.0 | 0.8865 | 0.215 | 0.625 | 0.516 | 0.9754 | 0.9997 | 0.9814 | 0.9225 | 0.8623 | yes |
| 1.075 | 0.8871 | 0.218 | 0.622 | 0.530 | 0.9751 | 0.9998 | 0.9814 | 0.9229 | 0.8618 | yes |
| 1.1 | 0.8871 | 0.218 | 0.622 | 0.527 | 0.9749 | 0.9998 | 0.9814 | 0.9231 | 0.8618 | yes |

**Chosen: +0.60.** It is the most detecting feasible value. Every stricter value breaks the
stress guard, and the stress-shift AUC is the binding constraint: a stricter sound channel leaves
less room for the stress channel at fixed native acceptance.

| dev, versus round 7 | round 7 | A+B+C′ (0.6) | change |
|---|---|---|---|
| near AUC | 0.871 [0.852, 0.893] | 0.892 [0.871, 0.912] | **+0.0202** (bar +0.02: pass, barely) |
| native words / sounds | 0.9722 / 0.9997 | 0.9765 / 0.9990 | +0.43 / −0.06 points |
| sound / word AUC | 0.9811 / 0.9231 | 0.9814 / 0.9217 | +0.0003 / −0.0014 |
| stress-shift AUC | 0.8710 | 0.8614 | −0.0095 (limit 0.010) |

Native word acceptance rises because the refitted stress channel is more lenient.

## 4. Held-out, run once (Russian only)

Round-7 caches were moved to `calib/api_round7_ru/`. Azure answers are its cached responses.

| FLEURS-ru test | round 7 | round 8 | Azure |
|---|---|---|---|
| near-swap AUC | 0.872 [0.842, 0.900] | **0.887 [0.860, 0.914]** | 0.909 [0.892, 0.925] |
| near replaced > 0.9 | 36 / 150 | 32 / 150 | 9 / 150 |
| near hit@5 % FA | 0.580 [0.493, 0.667] | 0.607 [0.520, 0.693] | 0.342 [0.233, 0.458] |
| near right word | 0.500 [0.420, 0.580] | 0.507 [0.427, 0.587] | 0.373 [0.302, 0.447] |
| far-swap AUC | 0.991 [0.986, 0.995] | 0.991 [0.986, 0.995] | 0.982 [0.978, 0.987] |
| far hit@5 % FA | 0.993 [0.980, 1.000] | 0.987 [0.960, 1.000] | 1.000 |
| far right word | 0.913 [0.860, 0.953] | 0.893 [0.833, 0.940] | 0.761 [0.712, 0.808] |
| native false alarms < 0.6 / < 0.8 | 0.048 / 0.106 | 0.042 / 0.097 | 0.054 / 0.092 |
| ten-language: native sounds / words / sentences (s = .5) | 0.998 / 0.971 / 0.993 | 0.997 / 0.977 / 0.990 | |
| ten-language: planted sound AUC | 0.979 [0.975, 0.983] | 0.979 [0.975, 0.983] | |
| ten-language: planted word AUC | 0.922 [0.917, 0.927] | 0.923 [0.919, 0.928] | |
| ten-language: stress-shift AUC | 0.872 | 0.863 | |
| **cv-real ru Test A AUC** (400 clips) | 0.793 [0.749, 0.836] | **0.772 [0.726, 0.816]** | 0.809 [0.766, 0.851] |

Reading the held-out results:
* **Swap benchmark.** Near swaps improve (+0.015), which is less than the dev gain. The total-miss
  tail shrinks only from 36 to 32. We are still behind Azure on near-swap AUC; we remain ahead at
  5 % false alarms and on blaming the right word.
* **cv-real.** It gets worse: Δ vs Azure is −0.038 [−0.083, +0.004], against −0.016 before.
  cv-real labels are reviewer rejections (misreads, noise, truncation), not near-miss word swaps;
  the stricter sound channel evidently ranks those clips less well.
* **cv-real "before".** This is the last full `bench.eval_cv` run (fingerprint 3afad851c345e9d4).
  The "after" was scored separately (`ru_round8 cv`) so that `cv_real.md` and the shared
  `eval_cv` caches were not touched; another agent was editing `bench/eval_cv.py` at the time.
  **`results/cv_real.md` is therefore not updated.**

## 5. The same bias rule for the other languages (dev only, nothing changed)

Each language keeps its current references, adapter and finalize bias. The grid runs from 0 to the
current bias in steps of 0.1. The stress or tone channel is refitted per value; ja has no channel
in the product, and ko / fr have no stress. en uses its own sound model with no bias and is not
applicable.

| lang | current bias | rule's pick | near AUC (CI) now → pick | > 0.9 | hit@5 % | native words | native sounds | sound AUC | word AUC | prosody AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| zh | +0.089 | 0.0 | 0.963 [0.951, 0.973] → 0.966 | 0.076 → 0.062 | 0.838 → 0.845 | 0.9769 → 0.9762 | 0.9979 → 0.9975 | 0.9892 → 0.9892 | 0.9385 → 0.9410 | 0.880 → 0.876 (tone) |
| es | −0.722 | −0.7 | 0.987 → 0.987 | 0.038 → 0.038 | 0.941 → 0.941 | 0.9850 → 0.9849 | 0.9977 → 0.9978 | 0.9929 → 0.9929 | 0.9689 → 0.9689 | 0.955 → 0.956 |
| it | +0.078 | 0.0 | 0.960 → 0.962 | 0.110 → 0.098 | 0.792 → 0.795 | 0.9859 → 0.9853 | 0.9996 → 0.9995 | 0.9920 → 0.9920 | 0.9551 → 0.9569 | 0.949 → 0.946 |
| ja | +0.253 | +0.1 | 0.965 → 0.966 | 0.139 → 0.139 | 0.838 → 0.844 | 0.9828 → 0.9827 | 0.9979 → 0.9971 | 0.9787 → 0.9787 | 0.9390 → 0.9379 | – |
| ko | +0.787 | 0.0 | 0.928 [0.901, 0.954] → 0.931 | 0.187 → 0.168 | 0.800 → 0.781 | 0.9842 → 0.9831 | 0.9996 → 0.9982 | 0.9840 → 0.9840 | 0.9350 → 0.9317 | – |
| fr | −0.549 | −0.5 | 0.975 → 0.978 | 0.094 → 0.080 | 0.902 → 0.899 | 0.9804 → 0.9797 | 0.9909 → 0.9913 | 0.9874 → 0.9874 | 0.9608 → 0.9613 | – |
| de | +0.909 | +0.3 | 0.947 [0.934, 0.960] → 0.953 | 0.106 → 0.081 | 0.775 → 0.784 | 0.9701 → 0.9703 | 0.9977 → 0.9942 | 0.9798 → 0.9798 | 0.9438 → 0.9412 | 0.918 → 0.917 |
| pt | +0.697 | +0.1 | 0.964 → 0.966 | 0.093 → 0.082 | 0.820 → 0.828 | 0.9775 → 0.9767 | 0.9972 → 0.9937 | 0.9807 → 0.9807 | 0.9376 → 0.9350 | 0.934 → 0.930 |

**Answer.** Elsewhere, choosing the bias for native acceptance alone costs little detection:
+0.000 to +0.006 near AUC, all inside the CIs. The share of replaced words scoring > 0.9 falls
1–2.5 points in zh, it, ko, fr, de and pt.

Russian is the exception: +0.020, and only together with the retrained adapter. The large
positive biases (de +0.9, ko +0.8, pt +0.7) cost the most, but only about 0.003–0.006 AUC. No
bias change is proposed for these languages.

## 6. Pairs that remain indistinguishable

| pair | why | do natives distinguish it? |
|---|---|---|
| которые / которое, самые / самое, разные / разное, первые / первое | unstressed -ые vs -ое: ɨ vs ɑ/a overlap in fast speech | barely; in fluent speech close to homophones |
| открытие / открытия, создание / создания, мнение / мнения, побережье / побережья | -ие/-ия, -ье/-ья: after j natives produce ɛ / a / ɪ (audit) | in careful speech yes, often not in fluent speech |
| является / являются, имеет / имеют, считается / считаются | unstressed е vs ю after a vowel; the evidence is strong (llr ≈ −5) but still forgiven at bias 0.6 | **yes**: different grammatical person |
| обеспечит / обеспечить, взял / взять (final т/ть) | final т is affricated 25 % of the time, ть is plain t 13 % | yes in careful speech; the acoustic overlap is real |
| организации / организаций, ситуации / ситуаций | final i vs j | weakly |
| всё / все, ещё / еще, трёх / трех | spelling only (ё written as е) | no |
| тем / чем, часть / честь, каждый / каждой | consonant or stressed-vowel change, still > 0.9 (not inspected sound by sound) | **yes** |

## Files and reproduction

* `bench/calib/ru_round8.py`:
  * `build-dev`: the dev swap set;
  * `audit` (`--ship` writes the table);
  * `eval --tag`: end-to-end dev evaluation;
  * `bias`: the finalize rule for ru;
  * `cv`: cv-real ru with cached Azure.
* `bench/calib/bias_grid.py`: `build-dev`, `run` and `select` for any language. Outputs go to
  `calib/bias_grid/select_<lang>_<tag>.json`.
* `pronunciada/text/ru.py`, `pronunciada/data/lex/ru_vowels.json`,
  `tests/test_text_ru_round8.py`.
* Latency: the ru latency in the first table of `results/languages.md` (1061 ms) was measured
  while other jobs shared the CPU. It is not a regression measurement.
