# Mandarin tones, round 3 ("tone v3")

**Disclosure.** This round was designed after earlier held-out evaluations (round 4 showed the
tone-swap gap to Azure). Every choice below was made on development data only: FLEURS zh
validation plus the AISHELL-3 test and zh-TW sets that `bench/prosody/tone_eval.py` already uses.
FLEURS zh **train** was used for training, including 742 more train clips than before. The
held-out test (FLEURS zh test: the ten-language benchmark and the Azure tone-swap set) was run
**once**, on the frozen configuration. The round-4 v2 results were already cached and are shown
next to it.

## Outcome

* **Shipped.** The model stays `tone_zh_v2`, with one new reference-side leniency rule,
  `neutral4` (below), and a re-fitted `pronunciada/data/calib/prosody/zh.json`.
* **Built but not the default.** `tone_zh_v3.onnx` can be selected with
  `PRONUNCIADA_TONE_MODEL=tone_zh_v3`. It was not made the default because every v3 candidate
  follows the pitch less than v2 on the leak tests (section 3). The adoption rule says it must
  not. Caveat: `zh.json` is fitted for v2, so the word channel is mis-calibrated when v3 is
  selected.
* **None of the dev targets was reached.** Best native-syllable agreement on FLEURS validation
  was 75 %, and only with ZIPA features that were then rejected; the target was 78 %. The best
  channel-alone AUC was 0.882 (target 0.92) and the best product hit@5 % was 0.383 (target 0.50).

## 1. What was tried

| change | where | kept? |
|---|---|---|
| FLEURS zh TRAIN: the 742 clips beyond the old 2 500 cap (`bench/prosody/fetch_fleurs_zh_rest.py`; validation and test sentences excluded) + FLEURS sampled ×3 | `tone3.py --data v3 --fleurs-weight 3` | in every v3 candidate |
| Local-register pitch input: lf0 minus its voicing-weighted ±0.75 s mean, plus that mean (`register_features`) | `pronunciada/prosody/features.py` | in v3 (did not help on its own) |
| Low-frequency log-mel patch: 20 of 80 Kaldi-style mel bins, ~20–650 Hz, numpy, 1.6 ms per 10 s (`lowmel_features`) | same | in v3 |
| Heavier contour transplant: share 0.8, 2–4 syllables, spectral dropout 0.5 (fine-tune) | `tone3.py --transplant 0.8 --tp-k 2-4` | = `tone_zh_v3` |
| B: pooled ZIPA enc1+enc3 activations per syllable (PCA 32 + flag, broadcast over the syllable's frames) | `zipa_emb_cache.py`, `tone3.py --zemb` | rejected (leak) |
| A: SwiftF0 pitch tracker | — | **not done**: it needs a third-party download (pip), which needs the owner's explicit OK |
| Leniency `neutral4`: a neutral-tone syllable after tone 1/2/4/5 also accepts tone 4 (a short fall) | `pronunciada/prosody/tone_leniency.py` | **shipped** |
| Leniency `neutral` (lexical tone for neutral syllables), `yibu` (一/不 sandhi tone 2 may stay 4) | same | rejected on dev |

Note that FLEURS zh train (2 500 clips, 7.5 h) was **already** in the v1/v2 training data. Item 1
of the plan was therefore only "the remaining 742 clips + upweighting".

## 2. Dev: native-speech accuracy and detection

Syllable model level (`python -m bench.prosody.tone3 eval …`; temperature and Platt fitted on
FLEURS validation). All candidates use the "v3" data mix. Steps: t3_reg / t3_reg_lm 12 k from
scratch; tp8 = t3_reg_lm + 4 k fine-tune steps; B = tp8 + 4 k steps (v2: 20 k).

| | v2 | t3_reg (pitch + register) | t3_reg_lm (+ low mel) | **tp8 = tone_zh_v3** | B (+ ZIPA enc1/enc3) |
|---|---|---|---|---|---|
| FLEURS val 5-way acc | 68.7 % | 67.9 % | 71.0 % | 70.6 % | 75.4 % |
| FLEURS val wrong-tone AUC | 0.902 | 0.897 | 0.911 | 0.909 | 0.932 |
| FLEURS val transplant AUC / hit@5 % / donor tone | .775 / .208 / .399 | .759 / .196 / .369 | .712 / .137 / .298 | .741 / .134 / .337 | .742 / .115 / .284 |
| aishell3 acc / AUC / hit@5 % | 83.8 / .969 / .849 | 82.1 / .964 / .826 | 83.6 / .969 / .849 | 83.0 / .966 / .832 | 85.8 / .974 / .880 |
| aishell3 recall 1/2/3/4/5 | .84/.83/.80/.88/.75 | .82/.81/.78/.88/.71 | .85/.83/.79/.87/.76 | .85/.82/.79/.87/.75 | .88/.84/.81/.88/.83 |
| aishell3 transplant AUC / hit@5 % | .899 / .496 | .884 / .471 | .871 / .396 | .896 / .455 | .902 / .479 |
| zhtw acc / AUC / hit@5 % | 75.0 / .927 / .637 | 72.3 / .917 / .617 | 72.7 / .921 / .620 | 71.6 / .916 / .600 | 75.6 / .934 / .661 |
| zhtw recall 1/2/3/4/5 | .77/.74/.64/.82/.61 | .73/.69/.62/.81/.55 | .73/.69/.63/.81/.66 | .73/.68/.60/.80/.61 | .79/.70/.67/.82/.73 |
| zhtw transplant AUC | .831 | .815 | .787 | .811 | .815 |

Word level (`python -m bench.calib.fit_prosody --lang zh [--dry-run] [--leniency …]`; 200 FLEURS
validation clips; 5-fold grouped CV; the native-drop rule ≤ 1 point applies in every row).
"Agreement" = the heard tone is the reference tone or an alt, with no leniency counted.

| | agreement (1/2/3/4/5) | channel-alone AUC | product AUC | product hit@5 % | sound-AUC cost | native accept s=.5 |
|---|---|---|---|---|---|---|
| v2 (round 4 shipped) | 67.0 % (.61/.67/.54/.76/.63) | 0.869 | 0.845 | 0.378 | 0.0268 | .985 → .977 |
| v2 + neutral | 67.0 % | 0.869 | 0.845 | 0.378 | 0.0267 | .985 → .977 |
| **v2 + neutral4 (shipped)** | 68.0 % with the rule (tone 5: .63 → .78) | **0.876** | **0.852** | **0.383** | **0.0252** | .985 → .977 |
| v2 + yibu | 67.1 % | 0.867 | 0.843 | 0.380 | 0.0269 | .985 → .976 |
| v2 + all three | 68.2 % | 0.875 | 0.850 | 0.383 | 0.0253 | .985 → .977 |
| t3_reg_lm | 70.6 % (.64/.72/.67/.74/.75) | 0.882 | 0.861 | 0.380 | 0.0253 | .985 → .977 |
| tp8 (tone_zh_v3) | 70.4 % (.63/.72/.63/.75/.76) | 0.878 | 0.857 | 0.367 | 0.0245 | .985 → .978 |

B was not run through the word-level fit. The runtime cannot get ZIPA activations without a
change to `pronunciada/scorer.py`, which is not my file. B was rejected on the leak test anyway.

Native-error analysis (`bench/prosody/tone_native_cats.py`, v2, FLEURS validation): neutral 们 is
heard as tone 4 60 % of the time (agreement 4 %), 子 and 得 are heard as 3/4, and tone-5 syllables
overall are heard as 4 17 % of the time. That motivated `neutral4`. 一 with tone 1 (numbers,
ordinals: 一九, 第一) is heard as 4 49 % of the time, but natives do say yī there, so no rule was
added. 不 with sandhi tone 2 is heard as 4 40 % of the time; the rule for it (`yibu`) lost AUC and
was dropped. Only 20 syllables were "3 before 3" without an alt.

## 3. Leak tests (pitch must decide)

Monotone test (`python -m bench.prosody.leak_test eval tone_v2 t3_reg t3_reg_lm t3_lm_tp8 t3_B_zemb`).
The data are 300 AISHELL-3 test utterances resynthesised with flat pitch, and the 2 433 syllables
whose label is tone 2/3/4. Every input (pitch, low mel, ZIPA activations) is recomputed from the
resynthesised audio.

| | acc on original | still the label tone on monotone (↓) | mono AUC (↑) | + transplant AUC FLEURS val / aishell3 / zhtw (↑) |
|---|---|---|---|---|
| **v2** | 86.2 % | **29.6 %** | 0.905 | **.775** / .899 / **.831** |
| t3_reg | 85.0 % | 33.6 % | 0.895 | .759 / .884 / .815 |
| t3_reg_lm | 86.5 % | 37.0 % | 0.892 | .712 / .871 / .787 |
| tp8 (tone_zh_v3) | 85.9 % | 32.7 % | 0.901 | .741 / .896 / .811 |
| B (ZIPA) | 87.9 % | 32.4 % | **0.908** | .742 / **.902** / .815 |

Every candidate leaves the dictionary tone in place on flat audio more often than v2. Every
candidate also recovers fewer transplanted FLEURS-validation contours, i.e. it listens to pitch
less on the in-domain dev set. B has a slightly higher mono AUC but is worse on the other two
measures. Under the adoption rule (no worse leak test) none is adopted. The spectral and ZIPA
streams buy native accuracy partly by recognising the syllable. Heavier transplant (tp8) pulled
the low-mel model back toward v2 but not past it.

## 4. Held-out test (once, frozen configuration), FLEURS zh

The ten-language benchmark and the Azure tone-swap set (`python -m bench.eval_languages --lang zh
--workers 4`). The round-4 zh caches were moved to `~/.cache/pronunciada/calib/api_round4_zh/`
and evaluated with the same code. Azure is its cached responses; no new Azure requests were made.
Utterance-clustered 95 % CIs.

| FLEURS-swap zh (150 recordings) | v2 (round 4) | **shipped: v2 + neutral4** | Azure |
|---|---|---|---|
| tone swap AUC | 0.816 [0.782, 0.852] | 0.824 [0.791, 0.859] | 0.846 [0.815, 0.875] |
| tone swap hit@5 % FA | 0.300 [0.213, 0.400] | 0.313 [0.220, 0.413] | 0.402 [0.327, 0.514] |
| tone swap hit@2 % FA | 0.067 | 0.067 | 0.213 |
| tone swap right word | 0.253 [0.180, 0.327] | 0.260 [0.187, 0.333] | 0.387 [0.313, 0.460] |
| near swap AUC / hit@5 % / right word | 0.964 / 0.860 / 0.787 | 0.964 / 0.853 / 0.780 | 0.966 / 0.896 / 0.747 |
| far swap AUC / hit@5 % / right word | 0.991 / 0.993 / 0.887 | 0.991 / 0.993 / 0.887 | 0.989 / 0.980 / 0.899 |
| native words score < 0.6 / < 0.8 | 0.046 / 0.116 | 0.047 / 0.112 | 0.036 / 0.128 |

| FLEURS zh test, controlled errors (400 clips) | v2 (round 4) | shipped |
|---|---|---|
| native acceptance sounds / words / sentences (s = 0.5) | .998 / .969 / .993 | .998 / .969 / .993 |
| sound-error AUC | 0.990 | 0.990 |
| word-error AUC | 0.932 | 0.933 |

Test moved in the direction dev predicted but by little: tone AUC +0.008 and hit@5 % +1.3
points, both well inside the CIs. Nothing else regressed beyond noise (near-swap hit@5 % −0.7
points). **We still trail Azure on Mandarin tone swaps.** No tuning was done on test.
`results/languages.md` was regenerated (zh rows). `results/azure_swap.md` holds Azure-only numbers
and is unchanged.

## 5. Size and latency

`tone_zh_v3.onnx` is 654 KB (178 k parameters). The package's prosody models total 2.1 MB, under
the 5 MB budget. Tone step on the densest 10 s FLEURS clip (48 syllables), `bench.prosody.latency`,
measured with the machine at load ≈ 11.6 (so absolute numbers are inflated):

| | 2 threads (as in Scorer) min / median | 4 threads min / median |
|---|---|---|
| v2 | 16.8 / 19.3 ms | 16.9 / 18.3 ms |
| v3 | 19.6 / 21.8 ms (+2.8) | 18.3 / 19.5 ms (+1.4) |

The leniency rule costs nothing measurable. B would need the scorer to pass ZIPA activations
(~0.3 ms extra, per acoustic.py) and a 768×32 projection.

## 6. Notes for whoever continues

* SwiftF0 (A) is untested. It needs `pip install swift-f0` (a third-party download), which I did
  not do without the owner's go-ahead. The probe is cheap once approved: F0 accuracy against Praat
  (`f0_eval.py`), then a t3_reg-style retrain. The tone, stress and pitch-accent models all
  consume `frame_features(F0Track)`, so a new tracker that returns an `F0Track` on the 10 ms grid
  would be a drop-in pitch source for all three after retraining.
* For B in production, `Scorer` would have to call `ProsodyScorer.score(..., embeddings=…)` with
  the enc1/enc3 activations it already has. That is a change to `pronunciada/scorer.py`, not made.
  The same applies to passing the fbank, though the numpy low mel is cheap anyway.
* The cheapest real gain left is probably counterfactual audio rather than more inputs: resynthesise
  training syllables with other tones (PSOLA, offline) so that segment cues stop predicting the label.

Files: `bench/prosody/{tone3,lowmel_cache,zipa_emb_cache,fetch_fleurs_zh_rest,tone_native_cats}.py`;
changes to `export.py` (`tone_v3`), `leak_test.py` (seq models), `latency.py` (threads),
`common.py` (unpickling caches written under the old `pronkit` name), `model.py` (`n_in`), and
`bench/calib/fit_prosody.py` (cache key covers the model/leniency env; `--dry-run`, `--leniency`;
agreement printout). Runtime: `pronunciada/prosody/{__init__,features,tone_leniency}.py`.
Test: `tests/test_prosody_v3.py`.

## 7. Second pass (dev only, no held-out run): selectable candidates for the OMPAL comparison

The default is unchanged (`tone_zh_v2` + `neutral4`). Each candidate below can be selected with
`PRONUNCIADA_TONE_MODEL=<name>` through the public `Scorer`. Each has its own word-channel
calibrator, `pronunciada/data/calib/prosody/zh.<name>.json`, fitted by
`PRONUNCIADA_TONE_MODEL=<name> python -m bench.calib.fit_prosody --lang zh`. The default uses
`zh.json`, and `Scorer` falls back to `zh.json` when no per-model file exists. All candidates
apply the `neutral4` rule.

| name | what | size | FLEURS-val agreement | channel AUC | product AUC | hit@5 % | sound-AUC cost | leak: still label / mono AUC / FLEURS-val transplant AUC |
|---|---|---|---|---|---|---|---|---|
| `tone_zh_v2` (default) | v2 + neutral4 | 626 KB | 68.0 % | 0.876 | 0.852 | 0.383 | 0.025 | 29.6 % / 0.905 / .775 |
| `tone_zh_v3` | pitch + register + low mel (tp8) | 654 KB | 70.4 % | 0.882 | 0.861 | 0.388 | 0.024 | 32.7 % / 0.901 / .741 |
| `tone_zh_v3z` | v3 + pooled ZIPA enc1+enc3 (PCA 32) | 712 KB + 100 KB PCA | **76.1 %** (1/2/3/4/5: .70/.77/.70/.79/.86) | **0.914** | **0.889** | **0.472** | **0.018** | 32.4 % / **0.908** / .742 |

The fit for each candidate keeps native word acceptance within 1 point
(.985 → .977 / .979 / .979). `tone_zh_v3z` comes close to the channel and hit targets (0.92 /
0.50). Its leak results are mixed, as in section 3: mono AUC better than v2, but more of the
dictionary tone left on flat audio and less transplant recovery. It is the candidate where real
learner data (OMPAL) can actually decide.

Tone step on the densest 10 s FLEURS clip, ZIPA activations precomputed, load ≈ 11
(min / median ms): v2 16.0 / 16.3, v3 18.1 / 18.6 (+2.3), v3z 19.2 / 19.5 (+3.2) at 2 threads;
at 4 threads 15.7 / 17.7 / 18.7 min.

Wiring (the smallest change):
* `pronunciada/scorer.py`:
  * requests the acoustic model's activations when the selected tone model needs them
    (`prosody.tone_needs_embeddings()`). The default delivery path already computes them, so
    nothing changes for v2.
  * passes `embeddings=emb` to `ProsodyScorer.score`.
  * loads `zh.<model>.json` for a non-default model.
* `ProsodyScorer` pools enc1+enc3 per syllable with `features.pool_syllables`, applies the
  shipped PCA (`<stem>.zemb.npz`) and raises if no activations are passed. `Scorer` reports
  this as a "prosody failed" warning.
* Tests: `tests/test_prosody_v3.py::test_selectable_tone_model_through_scorer[...]`, one per
  model.

## 8. SwiftF0 tracker probe (A, dev only): no clear advantage, stopped after step 1

The coordinator installed `swift-f0` 0.3.0 into the venv. I read the package code before
running it: it is numpy + onnxruntime and makes no network or file access beyond its own
`model.onnx`. `pronunciada/prosody/f0_swift.py` runs only that ONNX file and returns an
`F0Track` on our 10 ms grid:
* the clip is peak-normalised to 0.9 (for quiet audio);
* the search range is 50–600 Hz;
* log-F0 and confidence are interpolated from the 16 ms frames (centre k·16 ms, the best offset
  of −8 / 0 / +8 / +16 ms on VDE);
* energy is computed as in `f0.py`.

`f0.py` is unchanged and nothing selects the new module.

`python -m bench.prosody.f0_compare 60`: 60 FLEURS-zh validation clips (697 s), compared with
Praat autocorrelation (the `f0_eval.py` reference, voiced share 0.336):

| tracker | GPE | octave errors | FPE (cents) | VDE | voiced share | GPE / VDE, noisier half | 10 s clip |
|---|---|---|---|---|---|---|---|
| ours (YIN + Viterbi) | **1.8 %** | **1.3 %** | **28.2** | 7.3 % | 0.278 | **2.0 % / 9.3 %** | **9.7 ms** |
| SwiftF0 | 4.9 % | 3.7 % | 35.9 | 7.3 % | 0.306 | 5.9 % / 10.0 % | 73 ms (1 thread), 41 ms (2) |

SwiftF0 disagrees with our tracker on 3.3 % of the frames where both are voiced (2.0 % octave).
Caveat: our voicing gate was tuned against Praat on FLEURS validation, which favours us on VDE.
It does not explain SwiftF0's 2.7× gross and octave error rate. SwiftF0 is not cleaner here, and
it is 4–7× slower than the whole current pitch step. It would not fit the +15 ms budget as an
extra tracker. Per the instruction ("no clear advantage → stop after step 1"), no SwiftF0 tone
models were trained, so `tone_zh_v2s` / `tone_zh_v3zs` do not exist and nothing was vendored.
If it is ever used, it would be a drop-in pitch source for the tone, stress and pitch-accent
models (same `F0Track`), each needing a retrain.
