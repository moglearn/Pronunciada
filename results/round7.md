# Round 7: Mandarin tuned on real learners (OMPAL DEV)

**Disclosure.** After this round, OMPAL TEST compares a system with Mandarin defaults chosen on
OMPAL **DEV** with an **untuned** Azure. DEV is 28 French-L1 learners of Taiwan Mandarin, the
same first language and corpus as TEST but different speakers. The untuned system (round 6, tone
model `tone_zh_v2`) is kept beside it on the same TEST. All selection used OMPAL DEV and FLEURS-zh
validation only. The held-out sets were run once, after freezing:
* OMPAL TEST;
* FLEURS zh test (ten-language and tone-swap benchmarks);
* cv-real zh-CN / zh-TW test.

Azure numbers are its cached responses; no new Azure requests were made.

## What changed

| change | how it was chosen | adopted? |
|---|---|---|
| Default tone model `tone_zh_v2` → `tone_zh_v3z` (pitch + register + low mel + pooled ZIPA enc1/enc3) | coordinator, on OMPAL DEV (tone-channel AUC 0.877 vs 0.871; any-error 0.826 vs 0.816) | **yes** |
| Per-model word-channel calibrators: `zh.json` = v3z, `zh.tone_zh_v2.json`, `zh.tone_zh_v3.json` | `fit_prosody` (FLEURS validation) | yes |
| Learned word-level combination of sounds × tone, replacing the product (`bench/calib/fit_word_combo.py`) | speaker-grouped CV on OMPAL DEV + FLEURS guards | **no**: no variant met the guards, and none beat the bar |
| Per-language shift of the sounds channel | same | **no**: +0.0045 at best, below the +0.01 bar |
| Sentence `accuracy` and `prosodic` recalibrated on OMPAL DEV (`bench/calib/fit_sentence_zh.py`, `calib/sentence_zh_recal.json`) | speaker-grouped CV, gain ≥ 0.03, FLEURS native-median guard | **yes** (accuracy, prosodic); fluency kept (+0.005) |

Default behaviour and latency for other languages are unchanged. `PRONUNCIADA_TONE_MODEL=tone_zh_v2`
or `tone_zh_v3` selects the other models, each with its own calibrator.

The sentence recalibration is tied to v3z (its JSON names the tone model). With another tone model
selected, the round-4 sentence heads are used.

## 1. v3z parity and packaging

`python -m bench.prosody.parity_v3 tone_zh_v3z 150` compares the training pipeline (torch,
prepared float16 pitch track, cached features) with the runtime (onnxruntime, features from the
waveform, ZIPA activations computed live). On 150 FLEURS validation clips (5 065 syllables):
* max |ΔP| 0.019, 99th percentile 0.005, mean 0.0006;
* argmax agreement 99.94 %;
* the remaining difference is the float16 storage of the training-side pitch track.

The same spans were used on both sides. The runtime Scorer aligns with adapter-corrected
posteriors, the training prep with raw posteriors; that alignment skew is not covered here.

Everything v3z needs is in the package: `tone_zh_v3z.onnx` (712 KB), `.json` and
`.zemb.npz` (100 KB PCA). Its activations come from the acoustic model run the scorer already makes
(the default path already requests them for the sentence delivery features), so the extra cost is
the tone step only: about +3 ms per 10 s clip over v2 (19.5 vs 16.3 ms median, load ≈ 11).
Ten-language zh end-to-end median: 352 ms vs 347 ms per clip. If `ProsodyScorer` is called
directly without activations, v3z falls back to its "stream dropped" input (trained with 30 %
dropout).

## 2. Learned word combination (OMPAL DEV CV): not adopted

The data are OMPAL DEV (7 519 words, 12 518 units: 1 651 any-error, 1 318 tone-only, 333
consonant/vowel) and the FLEURS validation `fit_prosody` rows for v3z (26 327 words: natives,
planted tone errors, planted sound errors).
* CV: 5 folds, speaker-grouped for OMPAL and utterance-grouped for FLEURS.
* Unit score: minimum over overlapping words, as in `eval_ompal`.

| OMPAL DEV (CV) | any-error AUC | tone-only AUC | cons/vowel AUC | hit @ 10 % learner FA (any) | FLEURS native accept (s=.5) | FLEURS planted sound AUC | FLEURS planted tone AUC | guards |
|---|---|---|---|---|---|---|---|---|
| product (today) | 0.8095 | 0.8096 | 0.8089 | 0.368 | 0.979 | 0.942 | 0.890 | – |
| tone channel alone, min over the word | 0.8012 | 0.8442 | 0.6312 | 0.401 | | | | |
| product + sounds shift 0.5 / 1.0 | 0.8140 / 0.8175 | 0.818 / 0.824 | 0.800 / 0.791 | 0.390 / 0.399 | 0.983 / 0.986 | 0.937 / 0.930 | 0.896 / 0.901 | ok / fail (sound −0.013) |
| LR on logits, w_FLEURS 0.25 / 1 / 4 | 0.808 / 0.800 / 0.785 | 0.817 / 0.801 / 0.777 | 0.775 / 0.795 / 0.816 | | | 0.888 / 0.915 / 0.935 | 0.894 / 0.879 / 0.853 | fail |
| LR on logits + interaction | 0.815 / 0.810 / 0.801 | 0.823 / 0.813 / 0.797 | 0.784 / 0.799 / 0.814 | | | 0.916 / 0.929 / 0.939 | 0.899 / 0.892 / 0.880 | fail |
| LR on log-probabilities (± interaction) | 0.819 / 0.819 / 0.817 | 0.834 / 0.831 / 0.826 | 0.760 / 0.770 / 0.780 | 0.42 | 0.984–0.988 | 0.908–0.920 | 0.899–0.904 | fail (sound −0.02 to −0.03) |

The bar was max(product, tone channel alone) + 0.01 = 0.8195. Every learned variant failed a
guard. Fitting on learners up-weights the tone channel about 5:1, which costs planted sound-error
detection on FLEURS (−0.02 to −0.03) and real consonant/vowel errors on OMPAL (0.809 → 0.76–0.78).
The best variants sit 0.0005–0.001 below the bar anyway. **The product is kept.**

Why the coordinator's "tone channel alone 0.826" cannot be reached by a word score: that number
scores each character by its own syllable's tone. A word-level score gives every character of a
two-character word the same value, and with that granularity the tone channel is at 0.801. The
headroom is in character-level localisation, which the pre-specified unit metric does not give a
word score.

## 3. Sentence scores recalibrated on OMPAL DEV: adopted for accuracy and prosody

The model is a ridge regression (α = 1) per score on strictness-free inputs. The inputs are the
four round-4 sentence heads plus the utterance's mean syllable tone_prob, mean log tone_prob and
share of tone_prob < 0.5. The output is clip((prediction − 1) / 4, 0, 1). Evaluation is 5-fold
speaker-grouped CV on 1 105 DEV learner utterances; the FLEURS-validation native median is the guard.

| score | Pearson today | CV Pearson | FLEURS native median today → new | adopted |
|---|---|---|---|---|
| accuracy | 0.349 | **0.615** | 0.863 → 0.891 | yes |
| prosodic | 0.441 | **0.520** | 0.833 → 0.842 | yes |
| fluency | 0.670 | 0.675 | 0.864 → 0.877 | no (gain < 0.03) |

The four heads alone give 0.185 for accuracy; the tone summaries carry the gain. That is
expected for learners whose errors are 80 % tonal.

## 4. Held-out tests (once, frozen)

**OMPAL TEST** (21 speakers; 18 learners for sentence rows). 95 % speaker-bootstrap CIs. p is the
paired speaker bootstrap: round 7 vs untuned on the same resamples, and ours vs Azure as reported
by `eval_ompal`.

| | untuned (v2, round 6) | round 7 | Δ round 7 − untuned [CI], p | Azure zh-TW (untuned) | round 7 vs Azure p |
|---|---|---|---|---|---|
| word AUC, any error (primary) | 0.797 [0.757, 0.838] | 0.795 [0.753, 0.840] | −0.002 [−0.009, +0.008], p 0.72 | 0.704 [0.655, 0.749] | < 0.001 |
| word AUC, tone-only | 0.792 [0.749, 0.837] | 0.785 [0.739, 0.835] | −0.007 [−0.014, +0.004], p 0.22 | 0.679 [0.616, 0.737] | < 0.001 |
| word AUC, consonant/vowel | 0.820 [0.765, 0.868] | 0.839 [0.787, 0.882] | +0.019 [+0.008, +0.030], p < 0.001 | 0.822 [0.737, 0.874] | 0.42 |
| tone channel alone, any / tone-only | 0.815 / 0.854 | 0.825 / 0.857 | +0.010 [+0.003, +0.020], p 0.002 / +0.003, p 0.17 | – | |
| hit @ 10 % FA on native correct units, any error | 79.4 % | 86.6 % [81.2, 93.3] | | 74.2 % | 0.025 |
| hit @ 10 % FA on learner correct units, any error | 28.1 % | 25.9 % [19.3, 37.9] | | 23.8 % | 0.57 |
| sentence accuracy, Pearson | 0.388 [0.194, 0.486] | **0.635 [0.469, 0.742]** | +0.247 [+0.125, +0.428], p < 0.001 | 0.363 [0.175, 0.484] | < 0.001 |
| sentence prosody, Pearson | 0.419 [0.243, 0.571] | 0.532 [0.346, 0.677] | +0.113 [+0.003, +0.246], p 0.043 | 0.189 (PronScore) | < 0.001 |
| sentence fluency, Pearson | 0.594 | 0.594 (unchanged) | 0 | 0.343 | < 0.001 |

On real learner words the new tone model changes nothing measurable in the word score. The
tone-only word AUC is slightly (n.s.) lower. The tone channel itself is better (+0.010, p 0.002)
and consonant/vowel detection is better (+0.019). The gain on OMPAL is in the sentence scores,
which were fitted on DEV learners of the same population.

**FLEURS zh test, tone swap** (150 recordings; `results/languages.md`). Round 6 is the shipped
v2 + neutral4.

| | round 6 | round 7 | Azure |
|---|---|---|---|
| tone swap AUC | 0.824 [0.791, 0.859] | **0.845 [0.815, 0.873]** | 0.846 [0.815, 0.875] |
| tone swap hit @ 5 % FA | 0.313 | 0.373 [0.287, 0.467] | 0.402 [0.327, 0.514] |
| tone swap right word | 0.260 | 0.313 [0.240, 0.387] | 0.387 [0.313, 0.460] |
| near swap AUC / hit @ 5 % | 0.964 / 0.853 | 0.974 / 0.893 | 0.966 / 0.896 |
| far swap AUC / hit @ 5 % | 0.991 / 0.993 | 0.992 / 0.993 | 0.989 / 0.980 |
| native words < 0.6 / < 0.8 | 0.047 / 0.112 | 0.036 / 0.073 | 0.036 / 0.128 |
| ten-language: native words accepted / word-error AUC | 0.969 / 0.933 | 0.974 / 0.939 | |

**cv-real test** (`results/cv_real.md`): the primary score `SentenceResult.prob` is unchanged by
this round, so zh-CN 0.768 vs Azure 0.764 (p 0.85) and zh-TW 0.884 vs 0.861 (p 0.20) are
identical to before.

## What did not work

* A learned sounds × tone combination fitted on learners: up-weighting tone costs consonant/vowel
  and planted sound-error detection more than it gains (section 2).
* A sounds-channel shift: too small to clear the bar.
* The coordinator's premise that the word score "throws away" tone evidence is only half right.
  At word granularity the tone channel alone is 0.801 on DEV, below the product (0.810). The 0.826
  comes from character-level scoring.
* The OMPAL test word AUC did not improve with v3z.

## Files

* Calibration: `pronunciada/data/calib/prosody/zh*.json`, `pronunciada/data/calib/sentence_zh_recal.json`.
* Code: `pronunciada/scorer.py` (sentence recalibration, per-model calibrator), `pronunciada/prosody/__init__.py`
  (default v3z, fallback without activations), `bench/calib/fit_word_combo.py`,
  `bench/calib/fit_sentence_zh.py`, `bench/prosody/parity_v3.py`.
* CV tables: `results/tables/word_combo_zh_cv.json`, `results/tables/sentence_zh_recal_cv.json`.

Nothing was committed by me; the coordinator committed an intermediate state of this tree at 12:59.
