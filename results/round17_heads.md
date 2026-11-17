# Round 17: attribute heads where the scorer was still blind

The coaching round-4 attribute heads (`pronunciada/attr_heads.py`) hear Korean and Mandarin
aspiration, Russian soft / hard and Japanese length well as stand-alone detectors. This round
turns them into **contrast evidence** for those four cases. Selection used FLEURS **validation**
only (the contrast audit's own 250 clips per language, its swap plan, plus one near and one far
planted substitution per clip as the planted-error guard; `bench/round17/collect.py`,
`fit.py`). Held-out data was run once, paired before / after in the same tree with
`PRONUNCIADA_HEAD_CAPS=0` for "before". Azure answers are cached; no requests were made.

## Summary

| contrast (learner error) | where | dev hit@5 % before → after | FLEURS test before → after [paired 95 % CI of Δ, pt] |
|---|---|---|---|
| ko: aspirated for plain | shown sound | 2 % → **90 %** | 5 % → **86 %** [+67, +88] |
| ko: plain for aspirated | shown sound | 22 % → **97 %** | 27 % → **98 %** [+59, +79] |
| ru: soft for hard | shown sound | 50 % → **67 %** | 47 % → **56 %** [+3, +14] |
| ru: hard for soft | shown sound | 31 % → **92 %** | 24 % → **94 %** [+59, +83] |
| zh: aspirated for plain | shown sound | 79 % → **96 %** | 90 % → **100 %** [+5, +15] |
| zh: plain for aspirated | shown sound | 62 % → **100 %** | 68 % → **99 %** [+22, +42] |
| ja: long for short vowel | length channel | 97 % → **97 %** | 94 % → **94 %** [-3, +4] |
| ja: short for long vowel | length channel | 78 % → **93 %** | 85 % → **96 %** [+5, +17] |
| ja: geminate for single | length channel | 88 % → **92 %** | 92 % → **90 %** [-10, +5] |
| ja: single for geminate | length channel | 88 % → **100 %** | 79 % → **100 %** [+11, +33] |

* **Korean aspiration** and **Russian hard for soft**: from blind to 86–98 % on test.
* **Russian soft for hard** gains 9 points (47 → 56 %; dev 50 → 67 %): under the 60 % bar on
  test.
* **Mandarin aspiration**: on top of round 16's cap; both directions reach 99–100 %.
* **Japanese length**: the head as an extra input to the timing model lifts the two weak
  directions (short for long 85 → 96 %, single for geminate 79 → 100 %); the other two are
  unchanged within noise.

## Method

**Evidence.** For a sound whose expected symbol belongs to the contrast:
* s = logit P(the head's class of the expected symbol). The heads are `aspirated` for ko / zh and
  `palatalised` for ru, all of which follow the audio in the round-4 leak check (honest 0.84 /
  0.76).
* z = slope · (s − s0), where s0 is the q-quantile of natives' s (FLEURS validation). sigmoid(z) is
  "the feature was realized as expected".

**Cap.** The shown sound log-odds are set to min(z_l, z) and min(z_s, z). This composes with
round 16's caps by the minimum (no product), and coaching inputs, words and sentences do not
read it.

**Rule** (fixed before test):
* grid q ∈ {0.2, 0.5, 1, 2, 5} %, slope ∈ {0.5, 1, 2, 4};
* guards: native sound acceptance drop ≤ 0.5 pt, planted sound AUC drop ≤ 0.003;
* the best lower-direction hit wins, ties going to the more lenient setting;
* adopt when the lower direction gains ≥ 10 pt.

The chosen settings were q 1 % / 1 % / 0.5 % and slope 0.5 for ko / ru / zh
(`pronunciada/data/coach/contrast_heads.json`).

**Japanese.** The length channel gets the head `long` as a second input:
* a logistic on [logit of the timing model's probability, s], fitted on dev natives against the
  planted length swaps;
* mapped back onto the timing model's own native distribution by quantiles, so the word knee and
  the coaching bars keep their native rates (`data/length/ja.json` "head",
  `length.head_combine`).

After the held-out run the stored quantile tables were rounded to 5 decimals, to keep the shipped file under the length channel's 50 KB size test; that moves the logits by at most 1e-5.

Adoption rule: dev hit +3 pt at the same native cost (5-fold by clip):
* vowels 87 → 94 %;
* doubled consonants 86 → 93 %.

## Guards

| lang | dev native sounds | dev planted sound AUC | dev native words | test native sounds | test planted sound AUC |
|---|---|---|---|---|---|
| ko | 99.96 → 99.75 % | 0.9785 → 0.9787 | 98.42 → 98.42 % | 99.54 → 99.33 % | 0.9716 → 0.9715 |
| ru | 99.31 → 98.92 % | 0.9727 → 0.9754 | 97.37 → 97.37 % | 99.01 → 98.58 % | 0.9731 → 0.9738 |
| zh | 99.75 → 99.65 % | 0.9881 → 0.9866 | 97.94 → 97.94 % | 99.68 → 99.59 % | 0.9841 → 0.9831 |
| ja | 99.59 → 99.59 % | 0.9721 → 0.9721 | 98.28 → 98.19 % | 99.38 → 99.38 % | 0.9737 → 0.9737 |

Ten-language benchmark (FLEURS test, `bench.eval_languages`, paired; sound AUC uses all its
corruption kinds):

| lang | native sounds | sound AUC | sound hit@5 % | native words | word AUC | word hit@5 % |
|---|---|---|---|---|---|---|
| ko | 0.9961 → 0.9940 | 0.9796 → 0.9751 | 0.954 → 0.923 | 0.9784 → 0.9784 | 0.9342 → 0.9342 | 0.741 → 0.741 |
| ru | 0.9912 → 0.9868 | 0.9751 → 0.9721 | 0.932 → 0.912 | 0.9679 → 0.9679 | 0.9293 → 0.9293 | 0.716 → 0.716 |
| zh | 0.9972 → 0.9964 | 0.9888 → 0.9884 | 0.991 → 0.984 | 0.9739 → 0.9739 | 0.9394 → 0.9394 | 0.799 → 0.799 |
| ja | 0.9936 → 0.9936 | 0.9724 → 0.9724 | 0.916 → 0.916 | 0.9759 → 0.9751 | 0.9313 → 0.9323 | 0.734 → 0.742 |

* **ko: sound AUC −0.0045, over the 0.003 dev guard.** ru is at −0.0030. On dev, with
  near / far substitutions, both were within. The benchmark also includes inserted sounds and
  word swaps, where an inserted aspirated / soft consonant now reads as "not as expected", and
  its 5 % threshold moves with the natives.
* **Words, sentences, swap benchmark, cv-real.** For ko / ru / zh they do not read the shown
  sound probability, so they cannot move (word columns above: identical).
* **Japanese** (its word probability moves through the length word factor):
  * swap benchmark near-swap AUC 0.9638 → **0.9681** (Azure 0.919);
  * hit@5 % 0.820 → 0.833;
  * native false alarms 0.0288 → 0.0280;
  * cv-real ja AUC 0.802 → 0.802 (identical; Azure 0.835).
* **Mandarin OMPAL** (word, sentence, coaching score): unchanged by construction; not re-run.

Cost: the Japanese "geminate for single" swapped word is caught less often at the word level
(test 18 → 8 %, dev 20 → 15 %), although the length channel's own hit stays about the same.

## Coaching

* **Korean and Russian.** A member sound below a per-sensitivity bar on sigmoid(z) gets the
  head's claim (`feature.aspirated.*` / `feature.palatalised.*` tips), or a new substitution item
  when it was not flagged. The bar allows ≤ 0.05 / 0.1 / 0.2 native items per clip.
  * Korean, dev swaps caught at "balanced" by these items: plain for aspirated
    55 %, aspirated for plain 21 %.
  * Russian: hard for soft 54 %, soft for hard 1 %.
  * Unverified on learners: there are no labeled Korean or Russian learner data.
* **Mandarin: not enabled** (OMPAL dev, learners; `bench/round17/ompal.py`):
  * the evidence alone has consonant-error AUC 0.80 on aspiration-member initials;
  * maxed into the round-12b coaching score it lowers the consonant AUC 0.956 → 0.935;
  * no bar reaches 0.7 precision (best 0.67 on 12 characters at 0.05). At the native-budget bars,
    "balanced" precision would fall 0.83 → 0.31.
  * French learners' consonant errors are mostly not aspiration errors the head hears.
* **Word probability: not changed for ko / ru / zh.** It would need a separate dev near-swap
  collection to check the guard. Swapped-word hit stays at 5–43 %, so a word containing only an
  aspiration or palatalisation error still scores well. This is the main gap left.

## Korean lax / tense

Can the activations separate lax ㄱㄷㅂㅅㅈ from tense ㄲㄸㅃㅆㅉ onsets among natives? The
front-end writes both as k t p s tʃ. A probe (logistic, pooled vector, labels from the spelling,
fitted on FLEURS train, tested on FLEURS validation natives; `bench/round17/ko_tense.py`; tense
n = 142 of 3662) gives:
* AUC **0.98** with the sound's own vector;
* 0.98 from the first layer alone;
* 0.44 for the aspiration head, which does not encode it.

So the contrast is **separable**, unlike the audit's simple acoustics (0.64). It was **not built**
this round. The references would need a tense marker that ZIPA's token set lacks (a RefPhone
attribute plus a tense head and planted swaps). That is the obvious next step.

## Latency

The activations are already computed for all ten languages, so nothing new runs in the acoustic
model. Extra time per ~10 s clip (median, features / length stage) comes from pooling and the
heads:
* ko +1.2 ms;
* ru +2.2 ms;
* zh +0.4 ms;
* ja +1.8 ms (length).

Every other language is byte-identical: 50 clips, 5 per language, `PYTHONHASHSEED=0`; en, es, fr,
de, it and pt are identical to the previous commit, and ko, ru, zh and ja differ as intended.

## What did not work

* **ru soft for hard**: 56 % on test, under the 60 % bar.
* **ko sound AUC on the ten-language benchmark**: −0.0045, over the dev guard.
* **Mandarin coaching**: no gain on OMPAL learners.
* **Word-level effect for ko / ru / zh**: none.
* **ja geminate-for-single at the word level**: costs 10 points.
