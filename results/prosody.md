# Prosody: tones, word stress, pitch accent — measured results

All numbers measured on this machine (M3 Pro, shared with other agents; load average 5–19 during
runs). Held-out sets were never used for fitting or tuning: aishell3 test (20 unseen speakers,
800 utts), FLEURS zh test (400), zhtw (Common Voice zh-TW, 400). Dev = FLEURS validation (used
for model selection, temperature and Platt calibration). Scripts: `bench/prosody/`; artefacts:
`~/.cache/pronunciada/prosody/` (results JSON in `results/`).

## 1. F0 tracker (`pronunciada/prosody/f0.py`)

YIN difference function at 8 kHz (one batched FFT cross-correlation for all frames), K = 4
candidate minima per frame + octave cost, logistic voicing probability (d' + energy), Viterbi over
{candidates, unvoiced} with an octave-jump cost. 10 ms hop on the fbank grid.
Reference: Praat autocorrelation (parselmouth, offline only), floor 50, ceiling 600, 10 ms.
Voicing / energy parameters tuned on 50 FLEURS-validation clips (5 × 10 languages); reported on 50
other clips (10 aishell3 train, 10 speechocean762 train, 3 × 10 FLEURS train languages; 452.8 s):

| set | GPE (>20 %) | octave errors | VDE | fine error (std, cents) | voiced share ours / Praat |
|---|---|---|---|---|---|
| dev (tuning, 587.6 s) | 0.91 % | 0.67 % | 6.93 % | 23.3 | 47.3 % / 47.5 % |
| **report (452.8 s)** | **0.81 %** | **0.56 %** | **5.74 %** | 24.7 | 52.6 % / 52.0 % |

Before tuning the voicing gate (defaults): GPE 0.52 %, VDE 14.4 % (under-voicing: 33.9 % voiced).
Speed, 10 s clip (F0 + frame features): 9.4–9.6 ms min/median at load 3.7 (section 8), 10.1–11.1 ms at load ≈ 5;
14–17 ms medians seen when the machine was at load 15–19. Praat on the same clips: 14–24 ms.
Synthetic tests (`tests/test_prosody_f0.py`, 14 pass): constant 60–560 Hz (median error < 1 %),
glides up/down/vibrato, weak-fundamental octave trap, ±10 % alternating-period trap (we report
200 Hz; Praat reports 100 Hz there; at ≥ 15 % modulation both report 100 Hz), silence/noise, 8 kHz
input, frame-grid length.

## 2. Mandarin tone labels: our front-end vs AISHELL-3 (`bench/prosody/tone_agree.py`)

Per character (erhua entries such as 孩儿 hair2 skipped; utterances whose text differs skipped).

| split | utts | syllables | exact agreement | incl. our tone_alts |
|---|---|---|---|---|
| train | 6032 (5 skipped) | 69 061 | **96.79 %** | 97.71 % |
| test | 797 (3 skipped) | 8 799 | **96.92 %** | 97.91 % |

Disagreements (train): other tone, same syllable 1.12 % · AISHELL tone is one of our alts 0.92 % ·
sandhi only 0.51 % · neutral tone 0.44 % · different reading (polyphone) 0.23 %.
Top patterns (ours→AISHELL, train): alt 2→3 (261; optional 3-3 sandhi across words), alt 5→4
(195; lexical neutral tone read as full tone), sandhi 3→2 (186; AISHELL applies 3-3 sandhi where we
keep 3, e.g. 有 before a 3rd tone in the next word), other 2→3 (152), other 1→4 (136), other 4→2
(126), other 4→1 (85), alt 5→1 (81), neutral 5→2 (68), other 4→3 (59). Polyphone examples:
长微博 zhang3/chang2, 证券 quan4/juan1, 调至 diao4/tiao2. Several "other" cases are AISHELL errors
(吴绮莉 wu1 qi1; 百 wai3).
Training uses only syllables that agree or differ only by sandhi (label = our surface tone).

## 3. Alignment timing offset (`bench/prosody/offset_check.py`, aishell3 train only)

All syllable windows shifted by δ × 10 ms; gradient-boosted classifier on the contour baseline
features, trained on half the train speakers, tested on the other half:

| δ (10 ms frames) | −8 | −6 | −4 | −2 | **0** | +2 | +4 | +6 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | .666 | .677 | .677 | .684 | **.686** | .688 | .679 | .677 | .674 |

No offset: the stated ZIPA frame timing (frame j starts at 0.02 j + 0.03 s) is used as is.

## 4. Mandarin tone models — first round

Labels: our surface tone (1–4, 5 = neutral). Train: aishell3 train (agree/sandhi syllables) +
FLEURS zh train = 155 639 syllables per epoch (8 537 utts). Augmentation fresh every epoch: tempo
0.85–1.15, normalised-contour range ×0.7–1.3, noisy pitch track (12–30 dB SNR) for 30 % of
utterances, boundary jitter ±20 ms. 20 epochs, AdamW + one-cycle, best dev epoch kept.

* **baseline**: lf0 resampled to 10 points over previous / target / next syllable + their mean
  voicing, energy, log duration (42 dims) → MLP 128-128 (22 661 params).
* **ProsodyNet**: 160 × 10 input frames (5 features + target / nucleus / prev / next / valid masks),
  conv5 → dilated conv5 (residual) → BiGRU → attention + mean pooling over the target → MLP.
  ch 96 / hid 64 (≈ 130 k params). **stride-2 variant**: first conv stride 2, ch 64 / hid 48
  (69 414 params) — 4.4 ms vs 15.9 ms ONNX time for 50 syllables at 2 threads.

Accuracy (label = our tone; 95 % CI by utterance bootstrap):

| model | dev (FLEURS val) | aishell3 test | FLEURS zh test | zhtw |
|---|---|---|---|---|
| baseline MLP | 52.9 % | 68.5 % [67.3, 69.7] | 53.3 % [52.0, 54.5] | 58.7 % [56.5, 60.9] |
| ProsodyNet | 65.9 % | 79.4 % [78.5, 80.5] | 65.6 % [64.4, 66.8] | 67.7 % [65.0, 69.9] |
| ProsodyNet stride-2 | 65.1 % | 79.7 % [78.8, 80.6] | 65.0 % [63.8, 66.3] | 67.3 % [64.7, 69.4] |

aishell3 test, ProsodyNet stride-2: against AISHELL's own tones 80.9 %; on syllables where both
sources agree 81.8 % (ProsodyNet: 80.5 % / 81.3 %; baseline 69.8 % / 70.3 %).

Wrong-tone detection (every syllable: negative = its tone or alts, positives = each other tone in
1–4 that is not an alt; score = 1 − P(acceptable)); ECE after temperature + Platt fitted on dev:

| model | set | AUC | EER | ECE (cal.) | 2↔3 AUC / EER | n neg / pos |
|---|---|---|---|---|---|---|
| baseline | aishell3 | 0.901 | 18.1 % | 3.2 % | 0.819 / 25.8 % | 8 854 / 26 895 |
| baseline | FLEURS | 0.795 | 28.1 % | 0.8 % | 0.739 / 33.0 % | 14 106 / 42 794 |
| baseline | zhtw | 0.828 | 24.9 % | 1.3 % | 0.746 / 31.1 % | 3 059 / 9 269 |
| ProsodyNet | aishell3 | 0.953 | 11.4 % | 1.7 % | 0.919 / 15.6 % | |
| ProsodyNet | FLEURS | 0.881 | 20.2 % | 0.7 % | 0.837 / 24.7 % | |
| ProsodyNet | zhtw | 0.891 | 18.9 % | 1.6 % | 0.815 / 25.5 % | |
| stride-2 | aishell3 | 0.953 | 11.5 % | 1.9 % | 0.917 / 15.8 % | |
| stride-2 | FLEURS | 0.878 | 20.3 % | 0.7 % | 0.829 / 24.8 % | |
| stride-2 | zhtw | 0.891 | 19.5 % | 1.5 % | 0.822 / 25.1 % | |

Per-tone recall (tones 1/2/3/4/5), stride-2: aishell3 .788/.790/.716/.862/.733;
FLEURS .561/.578/.620/.763/.672; zhtw .660/.656/.546/.781/.470.
(ProsodyNet: aishell3 .843/.797/.705/.820/.724; FLEURS .633/.605/.627/.709/.709;
zhtw .723/.670/.533/.741/.530. Baseline: aishell3 .744/.721/.513/.780/.358.)

Confusion, stride-2 (rows = label tone 1..5, columns = predicted 1..5):

```
aishell3            FLEURS zh test            zhtw
1410 133  14 225   7   1564 443  73 689  18   407 113  14  79   4
 102 1638 174 129  30   263 2084 550 604 104   48 495 116  75  20
  11 171 1024 207  17    37 232 1234 447  39   20  72 253 109   9
 173  97 103 2512  29   301 368 417 3715  68   78  73  57 825  24
  23  45  35  70 475    25  79  66 111 575    5  26  13  45  79
```

Pair AUCs (stride-2, aishell3 / FLEURS / zhtw): 1-2 .952/.850/.874 · 1-3 .984/.941/.933 ·
1-4 .944/.866/.900 · 2-3 .917/.829/.822 · 2-4 .974/.895/.929 · 3-4 .949/.893/.897 ·
neutral vs 1/2/3/4 .979-.935 / .956-.858 / .900-.735.
By position (detection AUC, stride-2, aishell3 / FLEURS / zhtw): monosyllabic word .956/.900/.871,
word-initial .958/.894/.898, medial .942/.843/.897, word-final .950/.857/.892,
utterance-final .950/.823/.862.

Robustness (stride-2): boundaries jittered ±40 ms → accuracy 77.7 % / 63.2 % / 66.0 % (−2.0 /
−1.8 / −1.3 points), AUC 0.946 / 0.866 / 0.885. Speaking rate (slowest / middle / fastest third;
aishell3 by speaker, others by utterance): aishell3 81.7 / 76.4 / 81.3 % (4.1 vs 4.9 syl/s),
FLEURS 63.7 / 66.7 / 64.6 % (3.9 vs 5.5 syl/s), zhtw 70.8 / 67.2 / 64.4 % (3.0 vs 4.7 syl/s).

Training curves (dev accuracy by epoch): baseline 45.9 → 52.9 % (plateau from epoch 9);
ProsodyNet 55.2 → 65.9 % (still +0.3 points over the last 5 epochs); stride-2 53.6 → 65.1 %.

Diagnosis of the ceiling: section 5.

## 5. Diagnosis of the tone ceiling

All variants: stride-2 ProsodyNet (ch 64, hid 48), 8 epochs (the shipped model: 20 epochs), same
data, augmentation and evaluation as section 4. Columns: accuracy / wrong-tone AUC / 2↔3 AUC.

| variant | dev acc | aishell3 test | FLEURS zh test | zhtw | aishell3 vs AISHELL tones / on agreeing syllables |
|---|---|---|---|---|---|
| F0-only, 20 ep (**shipped**) | 65.1 % | 79.7 / .953 / .917 | 65.0 / .878 / .829 | 67.3 / .891 / .822 | 80.9 / 81.8 % |
| F0-only, 8 ep (reference) | 64.0 % | 78.7 / .950 / .906 | 63.6 / .872 / .821 | 67.3 / .888 / .811 | 80.0 / 80.8 % |
| (b) rhyme span as target | 63.7 % | 78.8 / .951 / .906 | 63.6 / .870 / .815 | 67.5 / .886 / .804 | 80.1 / 80.8 % |
| (b) + 13 MFCC + 16-band low-freq spectrum | 69.1 % | 81.5 / .962 / .934 | 67.8 / .897 / .862 | 63.8 / .865 / .789 | 82.8 / 83.6 % |
| (b) + 16-band low-freq spectrum (31–531 Hz) | 67.4 % | 80.1 / .957 / .918 | 67.4 / .896 / .848 | 62.7 / .869 / .802 | 81.3 / 82.1 % |
| (d) ZIPA posteriors only (127 log-probs) | 77.2 % | 84.1 / .973 / .960 | 77.5 / .945 / .924 | 73.0 / .927 / .899 | 84.3 / 85.6 % |
| (d) ZIPA posteriors + F0 features | 75.3 % | 86.1 / .977 / .960 | 74.5 / .932 / .899 | 73.1 / .923 / .867 | 86.6 / 87.9 % |
| (c) 25 % of training speakers / utts | 59.9 % | 74.7 / .933 / .881 | 60.4 / .845 / .799 | 63.3 / .863 / .799 | 76.1 / 76.7 % |
| (c) 50 % | 61.5 % | 76.1 / .939 / .886 | 61.7 / .857 / .808 | 66.2 / .878 / .812 | 77.5 / 78.1 % |
| (c) 100 % (= reference) | 64.0 % | 78.7 / .950 / .906 | 63.6 / .872 / .821 | 67.3 / .888 / .811 | 80.0 / 80.8 % |

**Leak test (`bench/prosody/leak_test.py`).** The extra inputs also identify the *segments*, and
in natively read text the segments predict the dictionary tone. That is useless for learners, who
say the right syllable with the wrong tone. Measured text-only prior on aishell3 test syllables
with label tone 2/3/4 (n = 2 681): the most frequent tone of the syllable's phone string
(toneless pinyin) is right **66.4 %** of the time, and of the character **91.6 %**. Test: 300
aishell3 test utterances resynthesised with a flat pitch at the speaker's median (Praat PSOLA,
offline only). Segments and timing are unchanged, so the same spans are used; ZIPA posteriors,
pitch and spectra are recomputed. On syllables with label tone 2/3/4 (n = 2 433), a model that
reads pitch should stop giving the label tone. `mono AUC` = P(acceptable) on the original audio
vs on the monotone audio (the product case "learner speaks flat"):

| variant | acc on original | still the label tone on monotone (↓ better) | mono AUC (↑) |
|---|---|---|---|
| F0-only, shipped | 83.5 % | **29.5 %** | **0.883** |
| F0-only, 8 ep | 81.9 % | 27.5 % | 0.880 |
| rhyme span | 82.1 % | 28.5 % | 0.877 |
| + MFCC + low-freq spectrum | 83.2 % | 35.7 % | 0.849 |
| + low-freq spectrum | 82.2 % | 39.1 % | 0.843 |
| ZIPA posteriors only | 84.8 % | 54.1 % | 0.758 |
| ZIPA posteriors + F0 | 88.2 % | 47.2 % | 0.837 |
| 25 % / 50 % data | — | 26.0 % / 25.8 % | 0.868 / 0.872 |

Findings:
* (a) **Label noise is small.** On aishell3 test, accuracy against AISHELL-3's own human-checked
  tones is 80.9 % and on syllables where both sources agree 81.8 %, vs 79.7 % against our labels.
  Our labels disagree with AISHELL-3 on 3.1 % of syllables (2.1 % beyond our tone_alts). So label
  noise explains about 1–2 points, not the ceiling. FLEURS / zhtw use front-end labels only. If
  their label error rate is like AISHELL's (~3 %, probably more for FLEURS' names and numbers),
  their accuracies are lower bounds by up to ~3 points. On zhtw, where we already accept Taiwan
  variants as alts, the gap between accuracy and accuracy-with-alts is 0.6 points.
* (b) Rhyme span instead of the whole syllable: no change (±0.2 points). Spectral inputs raise
  native accuracy by 1.4–4.2 points on aishell3/FLEURS but *lower* zhtw by 3.5–4.6 points
  (microphone/channel mismatch). They also increase segment-based guessing (monotone: 35.7–39.1 %
  still the label vs 27.5 %; mono AUC 0.843–0.849 vs 0.880). Not shipped.
* (c) **Data helps moderately.** About +1.4 to +2.6 points per doubling (aishell3 74.7 → 76.1 →
  78.7 %; FLEURS 60.4 → 61.7 → 63.6 %). Caveat: smaller subsets also got proportionally fewer
  updates in 8 epochs.
* (d) **ZIPA posteriors raise native accuracy by 5–14 points, mostly through segment identity.**
  On flat-pitch audio the posterior-only model still gives the dictionary tone 54 % of the time.
  Its learner-relevant AUC is 0.758 vs 0.883 for F0-only. Some pitch information is present in
  the posteriors (accuracy falls 84.8 → 54.1 % when the pitch is flattened), but the gain on
  native test sets does not carry over to detecting wrong tones. Not shipped. Fixing this would
  need counterfactual training data (the same syllables resynthesised with other tones, labelled
  with the tone actually produced). Not done.
* **Shipped: F0-only stride-2 ProsodyNet** (section 4 numbers). Its native ceiling is set by the
  pitch signal itself, by FLEURS' fast read speech (5.5 syl/s in the fastest third) and by data
  quantity. It is not set by label noise.
(The diagnosis spectral-feature cache `~/.cache/pronunciada/prosody/extra/` was deleted afterwards to
save shared disk; `python -m bench.prosody.extra_feats` rebuilds it.)

## 6. Mandarin tones, round 2: more data, utterance-level model, contour transplant (`bench/prosody/tone2.py`)

**Data added** (training only; labels from our front-end; `bench/prosody/fetch_zh_extra.py`):

| source | licence | speakers | utterances | hours |
|---|---|---|---|---|
| AISHELL-1 train speakers S0002–S0715 (HF `AISHELL/AISHELL-1` per-speaker archives for S0002–S0101, then HF parquet mirror `carlot/AIShell` row groups for later speakers; ≤ 100 utts per speaker) | Apache-2.0 (corpus licence; the `carlot` mirror card states none) | 300 | 30 000 | 37.24 |
| Common Voice 25.0 zh-TW official train split (HF `OpenFormosa/common_voice_25_zh-TW`), contributors in our zhtw test set excluded (0 overlapping sentences), ≤ 50 utts per contributor | CC0-1.0 | 183 | 4 994 | 4.67 |
| **total new** | | **483** | 34 994 | **41.9** |

Disk: 1.9 GB + 0.26 GB FLAC under `~/.cache/pronunciada/data/{aishell1,cvtw}`, archives deleted. ZIPA
posteriors via bench.postcache; prep 0 failures. AISHELL-1/-3 speaker identities are separate
corpus id spaces (BAC009S… vs SSB…). Overlap of physical persons cannot be checked.
Old data = aishell3 train + FLEURS zh train (8 537 utts); new = old + the above (43 531 utts).

**Models.** *local* = shipped stride-2 ProsodyNet (69 k params). *seq* = the same syllable encoder,
then a BiGRU (hid 64) across all syllables of the utterance, head on [syllable embedding, sequence
state] (171 019 params). Exported for one utterance: x [S, 160, 10] → logits [S, 5].
**Contour transplant** (training, share 0.4 of utterances, k = 1–3 syllables). The syllable's
pitch + voicing track is replaced by the time-warped track of a donor syllable *of the same
utterance*. The donor's tone is outside the target's tone + alts, its duration ratio is 0.6–1.6,
and it is ≥ 30 % voiced. The donor is shifted so its local level (median log-F0 over donor ± 1
syllable) equals the target's. Label = donor tone; energy stays the target's.

**New held-out tests.**
* *Transplant detection*: on each test utterance one syllable gets a transplant (fixed seed;
  798 / 400 / 398 utterances got one). Positives = transplanted syllables, negatives =
  untouched syllables. Score = 1 − P(reference tone or alts). Also reported: hit rate at 5 %
  native false alarms, and donor-tone accuracy (argmax = the tone actually produced).
* *Monotone*: section 5's flat-pitch resynthesis, 2 433 tone-2/3/4 syllables. Values are AUC
  (original vs monotone audio) / share still labelled with the dictionary tone.

All variants: the same 8 000 optimiser steps × 256 syllables (old data ≈ 13 passes, new ≈ 3
passes), temperature + Platt on FLEURS-validation dev. Cells: 5-way acc % / wrong-tone AUC /
EER % / 2↔3 AUC / transplant AUC / transplant hit @ 5 % FA % / donor-tone acc %.

| data · model · transplant | aishell3 test | FLEURS zh test | zhtw | monotone AUC / still % |
|---|---|---|---|---|
| shipped (old · local · off, 20 ep) | 79.7 / .953 / 11.5 / .917 / .829 / 35.0 / 40.4 | 65.0 / .878 / 20.3 / .829 / .742 / 22.5 / 39.8 | 67.3 / .891 / 19.5 / .822 / .764 / 23.1 / 37.2 | .883 / 29.5 |
| old · local · off | 79.5 / .953 / 11.6 / .914 / .827 / 36.2 / 38.0 | 64.4 / .876 / 20.8 / .830 / .732 / 21.2 / 39.2 | 67.2 / .889 / 19.8 / .812 / .764 / 20.4 / 35.9 | .888 / 26.6 |
| old · local · on | 79.2 / .951 / 11.8 / .912 / .865 / 38.7 / 46.9 | 64.6 / .875 / 20.9 / .829 / .751 / 19.5 / 39.0 | 66.7 / .885 / 19.6 / .814 / .784 / 23.6 / 43.0 | .881 / 29.1 |
| old · seq · off | 82.1 / .964 / 9.8 / .941 / .823 / 39.3 / 35.7 | 66.2 / .882 / 20.1 / .843 / .723 / 20.5 / 35.8 | 69.0 / .900 / 18.0 / .821 / .764 / 19.8 / 37.4 | .872 / 38.1 |
| old · seq · on | 81.8 / .963 / 10.1 / .935 / .875 / 43.2 / 47.6 | 66.2 / .885 / 19.8 / .845 / .743 / 17.5 / 38.5 | 68.9 / .901 / 18.2 / .824 / .794 / 21.6 / 41.2 | .878 / 36.4 |
| new · local · off | 79.2 / .953 / 11.4 / .910 / .841 / 37.6 / 39.7 | 64.2 / .874 / 20.9 / .825 / .731 / 21.2 / 37.5 | 68.8 / .899 / 18.2 / .826 / .768 / 21.6 / 40.7 | .898 / 21.4 |
| new · local · on | 78.7 / .950 / 11.9 / .908 / .872 / 40.4 / 50.5 | 63.8 / .872 / 21.1 / .821 / .752 / 19.8 / 40.2 | 69.4 / .897 / 18.8 / .828 / .796 / 25.1 / 45.7 | .894 / 21.4 |
| new · seq · off | 82.4 / .965 / 9.8 / .937 / .842 / 41.0 / 39.3 | 66.7 / .887 / 19.6 / .846 / .740 / 23.0 / 37.2 | 73.2 / .920 / 16.0 / .851 / .777 / 23.1 / 36.7 | .893 / 31.7 |
| new · seq · on (8 k steps) | 82.1 / .964 / 10.0 / .935 / .885 / 46.5 / 50.6 | 66.4 / .887 / 19.8 / .847 / .768 / 22.5 / 43.8 | 72.3 / .918 / 16.5 / .848 / .816 / 28.9 / 45.2 | .895 / 30.9 |
| **new · seq · on, 20 k steps = tone_zh_v2** | **83.8 / .970 / 8.9 / .945 / .899 / 49.6 / 56.0** | **68.4 / .899 / 18.5 / .859 / .785 / 21.8 / 44.3** | **75.0 / .927 / 14.9 / .862 / .831 / 27.1 / 45.7** | **.905 / 29.6** |

Effects:
* **Sequence model**: +2.3 to +4.4 points accuracy and +.010 to +.021 wrong-tone AUC. Without
  transplant it learns the native tone grammar: monotone "still the label tone" rises to 38 % (old
  data) and transplant AUC does not improve.
* **Transplant**: transplant AUC +.02 to +.05 and donor-tone accuracy +6 to +12 points in every
  cell, at ≤ 0.6 points of native accuracy.
* **New data**: mostly helps zhtw (+1.8 to +4.3 points with Taiwan speakers in training) and the
  monotone test (AUC .893–.898 vs .872–.888). FLEURS and aishell3 move little at an equal step
  budget, since the new data was seen for only ~3 passes.
* **Chosen: new · seq · transplant**. It is best or tied on transplant AUC and best on wrong-tone
  detection for its budget, and its monotone AUC does not lose (.895 vs .883). Retrained for
  20 000 steps (dev 61.1 → 68.7 %, flat over the last 4 k steps) as **tone_zh_v2**.

**tone_zh_v2 vs the shipped tone_zh (round 1)**, temperature 1.002 / Platt (0.975, 0.062) from dev:

| | aishell3 test | FLEURS zh test | zhtw |
|---|---|---|---|
| 5-way accuracy | 83.8 % (was 79.7) | 68.4 % (65.0) | 75.0 % (67.3) |
| vs AISHELL-3's own tones | 84.9 % (80.9) | — | — |
| per-tone recall 1/2/3/4/5 | .841/.828/.802/.881/.753 | .616/.629/.582/.800/.710 | .773/.736/.635/.818/.607 |
| wrong-tone AUC / EER | 0.970 / 8.9 % (0.953 / 11.5) | 0.899 / 18.5 % (0.878 / 20.3) | 0.927 / 14.9 % (0.891 / 19.5) |
| wrong-tone hit @ 5 % FA | 84.9 % (77.2) | 55.7 % (49.4) | 63.7 % (53.6) |
| wrong-tone ECE (calibrated) | 1.7 % (1.9) | 0.9 % (0.7) | 1.2 % (1.5) |
| 2↔3 AUC | 0.945 (0.917) | 0.859 (0.829) | 0.862 (0.822) |
| transplant AUC / EER | 0.899 / 17.0 % (0.829 / 23.8) | 0.785 / 29.3 % (0.742 / 33.5) | 0.831 / 24.7 % (0.764 / 30.4) |
| transplant hit @ 5 % FA | 49.6 % (35.0) | 21.8 % (22.5) | 27.1 % (23.1) |
| donor tone recovered | 56.0 % (40.4) | 44.3 % (39.8) | 45.7 % (37.2) |
| accuracy, boundaries ±40 ms | 81.8 % (77.7) | 66.5 % (63.0) | 73.3 % (65.8) |
| monotone AUC / still label | 0.905 / 29.6 % (0.883 / 29.5) | | |

Transplant ECE is 22–39 % for both models. Platt is fitted on the wrong-tone pairs (25 %
acceptable); the transplant set is 90 % acceptable, and a calibrated probability does not
transfer across that prior shift. Use it for ranking, not for absolute rates. For context, the
coordinator's Azure figure on the FLEURS tone-swap benchmark (AUC 0.846, 40 % hits at 5 % FA) is
on a different protocol. It is not comparable to either table here without running that
benchmark on this model.
Runtime: `tone_zh_v2.onnx` 626 KB (+ JSON); all prosody models 1.4 MB. 10 s densest FLEURS zh
clip, 48 syllables: 14.5 / 14.6 ms min / median (round 1: 14.4 / 14.6), of which F0 + features
9.6 ms. ONNX vs torch max |Δlogit| 4e-7. The default is switched to `tone_zh_v2`
(`pronunciada.prosody.TONE_MODEL`); `tone_zh.onnx` is kept and selectable with
`PRONUNCIADA_TONE_MODEL=tone_zh`.

## 7. Word stress (en de es it pt ru)

Task: for each word with 2–8 syllables and exactly one primary stress in the front-end reference,
softmax over the word's syllables of a per-syllable score. Model: stride-2 ProsodyNet (ch 64, hid
48) on the syllable's 160-frame context + 7 word-relative cues (log duration, log nucleus duration,
nucleus energy, nucleus mean log-F0 and syllable max log-F0 minus the word mean, nucleus F0 slope,
nucleus voicing). **No position features**, on purpose: the score must come from the audio, not
from the language's default stress position. Train: FLEURS train with the final European
front-ends (1 000 utts per language); same augmentation as for tones; 15 epochs; dev = FLEURS
validation. Shift detection: negatives = P(expected syllable), positives = P(each other syllable)
of every word (`stress` corruption moves the primary stress to another syllable). ECE after
per-language Platt calibration on dev.

Shared multilingual model (language embedding, 70 162 params) — **shipped**:

| lang | test words | accuracy [95 % CI] | chance | majority position per word length* | shift AUC | EER | ECE (cal.) | n pos |
|---|---|---|---|---|---|---|---|---|
| en | 3 426 | 82.7 % [81.4, 84.1] | 42.0 % | 68.7 % | 0.933 | 14.0 % | 1.5 % | 5 393 |
| de | 4 745 | 82.6 % [81.3, 83.8] | 39.4 % | 64.9 % | 0.940 | 13.2 % | 0.8 % | 8 758 |
| es | 5 716 | 86.0 % [84.9, 87.1] | 38.2 % | 74.9 % | 0.961 | 10.2 % | 0.8 % | 10 790 |
| it | 6 181 | 85.1 % [84.1, 86.1] | 37.3 % | 91.0 % | 0.959 | 10.6 % | 0.7 % | 12 342 |
| pt | 5 607 | 87.8 % [86.9, 88.6] | 38.2 % | 74.2 % | 0.967 | 8.8 % | 1.0 % | 10 697 |
| ru | 5 525 | 76.0 % [74.8, 77.3] | 34.8 % | 54.5 % | 0.905 | 16.2 % | 1.4 % | 12 401 |

\* text-only reference fitted on the test words themselves (optimistic): always pick the most
frequent primary-stress position for the word's syllable count. Italian's default penultimate
stress makes this 91.0 %, above the acoustic model. The model never sees position, so its
detection does not just flag unusual positions. That is the desired property for learners.
Accuracy by syllable count (multi): en 2/3/4/5 syl 88.4/82.2/60.5/53.1 %; de 2/3/4 90.8/79.0/69.8 %;
es 89.1/85.2/81.3 %; it 92.0/80.4/79.1 %; pt 88.9/87.7/84.5 %; ru 77.6/74.6/77.9 %.

Per-language models (same architecture, no language embedding), accuracy / shift AUC:
en 80.8 % / 0.918, de 80.6 % / 0.931, es 84.7 % / 0.954, it 84.2 % / 0.953, pt 86.5 % / 0.962,
ru 76.3 % / 0.904. The shared model is better in 5 of 6 languages (+0.9 to +2.0 points; ru −0.3)
and is one file, so it is kept.
Training curves (dev acc): multi 77.7 → 83.3 % (flat over the last 4 epochs); per-language
models reach 75.8–85.8 %.

**speechocean762 test (English learners)**, shared model, Platt from FLEURS-en dev: 2 563 words
with ≥ 2 syllables matched to the human labels. Human word `stress` scores: **10 → 2 483 words,
5 → 80 words** (no other values). Pearson(stress_prob, human) 0.189, Spearman 0.142; ROC AUC for
"human gave less than 10" = **0.736** (80 positives). The English-only model: 0.218 / 0.146 / 0.742.
On these learner words the model puts the maximum on the dictionary-stressed syllable for 84.0 %.
With 80 positives the AUC has a wide uncertainty; the label is extremely skewed.

## 8. Japanese pitch accent (experimental)

Per mora (merged units as in the front-end) P(high) vs `pitch_high` (Open JTalk accent dictionary +
Tokyo rule). Stride-2 ProsodyNet, binary output, train FLEURS ja train (1 000 utts), 15 epochs.
FLEURS ja test, 22 558 morae (52.9 % labelled high):

| model | mora accuracy [95 % CI] | recall H / L | mora flip AUC / EER | word pattern AUC (all / ≥ 2 morae) | EER | ECE (cal.) |
|---|---|---|---|---|---|---|
| stride-1 (130 k) | 70.8 % [70.0, 71.7] | 70.4 / 71.3 % | 0.781 / 29.2 % | 0.765 / 0.785 | 29.9 % | 1.1 % |
| **stride-2 (shipped)** | 70.7 % [69.9, 71.6] | 70.5 / 71.0 % | 0.780 / 29.3 % | 0.764 / 0.785 | 30.4 % | 1.1 % |

Word-pattern test: 11 196 words, 18 015 alternative Tokyo patterns (bench.corrupt `pitch`
replacement set). **Agreement with the dictionary-derived labels on native speech is 70.7 %,
below 80 %, so pitch_prob is experimental.** The labels may be wrong (compounds, accent-phrase
chaining, speaker variation), or the model may be; this data cannot tell which. Dev curve: 63.0 →
71.0 %.

## 9. Runtime (`pronunciada/prosody/__init__.py`)

`ProsodyScorer(lang).score(wav, ref, evidence)` → one dict per syllable: zh `tone_probs`,
`tone_heard`, `tone_prob` (temperature + Platt; P(tone or tone_alt)); en/de/es/it/pt/ru
`stress_prob` on the primary-stressed syllable of every ≥ 2-syllable word (None elsewhere) +
`stress_dist`; ja `pitch_prob`, `pitch_high_prob`; fr/ko empty dicts. numpy + scipy +
onnxruntime only (checked: no torch / sklearn / parselmouth import under pronunciada/).

| file | size |
|---|---|
| tone_zh.onnx (69 k params, round 1, kept) | 276.6 KB |
| tone_zh_v2.onnx (171 k params, **default**) | 626.3 KB |
| stress.onnx (70 k params, 6 languages) | 280.4 KB |
| pitch_ja.onnx (69 k params) | 275.6 KB |
| JSON metadata (3 files) | 0.7 KB |
| **total** | **1.4 MB** (budget 5 MB) |

Latency, `bench.prosody.latency`: densest 9.5–10 s FLEURS test clip per language, 2 onnxruntime
threads, 30 runs, load ≈ 3.7, alignment excluded:

| lang | syllables | total min / median | of which F0 + features |
|---|---|---|---|
| zh | 48 | 14.3 / 14.4 ms | 9.5 / 9.6 ms |
| ja | 50 | 14.4 / 14.6 ms | 9.6 / 9.8 ms |
| en | 54 | 13.7 / 14.1 ms | 9.4 / 9.6 ms |
| de | 61 | 14.8 / 14.9 ms | |
| es | 52 | 13.7 / 13.7 ms | |
| it | 49 | 14.4 / 14.5 ms | |
| pt | 53 | 14.3 / 14.4 ms | |
| ru | 53 | 15.0 / 15.0 ms | |

ONNX vs torch max |logit difference| on random inputs: tone 7e-7, pitch 2e-7, stress 2e-6.
Tests (26 pass): `tests/test_prosody_f0.py` (synthetic pitch, glides, octave traps, silence,
speed), `tests/test_prosody_features.py` (shapes, normalisation, windows, dropped syllables,
cropping), `tests/test_prosody_runtime.py`. The runtime tests cover end-to-end runs on macOS
`say` clips (Tingting zh, Kyoko ja, Samantha en): a TTS voice is mostly accepted, and the same
audio scored against a wrong-tone reference gets a mean tone_prob lower by > 0.2. They also check
runtime vs training-pipeline parity (features rebuilt from the float16 track as in training,
max probability difference < 0.02), size ≤ 5 MB, latency and fr returning nothing.

## 10. Tried and dropped

* Contour-MLP baseline (section 4): 11–12 points below ProsodyNet.
* ProsodyNet stride 1, ch 96 / hid 64: same accuracy as stride 2, but 15.9 ms vs 4.4 ms model
  time for 50 syllables. A conv-only variant (no GRU) at full rate was just as slow (16.2 ms); not
  trained.
* Spectral inputs (MFCC, low-frequency spectrum), ZIPA posteriors: section 5. They raise native
  accuracy but rely on segment identity and hurt zhtw. Rhyme-only target: no effect.
* Per-language stress models: section 6.
* Constant alignment offset: none found (section 3).
* Voicing gate at the defaults (VDE 14.4 %): retuned on dev to 6.9 % (report set 5.7 %).

## 11. Known weaknesses

* Tone accuracy on fast read speech (FLEURS 68 % with v2) and phone-mic Taiwan speech (zhtw 75 %)
  is far below the clean AISHELL-3 voices (84 %). Transplant hits at 5 % FA on FLEURS stay at 22 %. 2↔3 is the weakest pair (AUC 0.83 FLEURS / 0.82 zhtw),
  and on zhtw tone 3 recall is 55 % and neutral-tone recall 47 %.
* Tone probabilities were validated only with *synthetic* errors: the reference changed, or a
  monotone resynthesis. No real learner Mandarin was available (LATIC / OMPAL not obtained).
  Under monotone audio, 29.5 % of tone 2/3/4 syllables are still classified as their label tone.
* Speaker normalisation is per utterance (log-F0 z-score over its voiced frames). Very short
  prompts (1–3 syllables) give unstable statistics; this was not evaluated.
* Stress: the speechocean762 agreement is weak (Pearson 0.19, AUC 0.74 on 80 positives). Russian
  is the weakest language (76 %). Long words (≥ 4 syllables) drop to 53–70 % in en/de. The labels
  come from the front-ends and include their errors.
* Japanese pitch: 70.7 % agreement with dictionary labels. Experimental.
* Latency 14–15 ms is measured on this M3 Pro. The F0 tracker's Viterbi is a Python loop
  (~2.5 ms of the ~9.5 ms).
