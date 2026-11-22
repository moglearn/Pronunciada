# Round 19: OpenPronounce 1.1.0 against Pronunciada 1.0.2, measured

Date of the runs: 22 November 2026. Every OpenPronounce number on this page was measured by us on the
same recordings as Pronunciada, with its released code and models. Pronunciada's numbers are its
cached per-recording outputs (the ones the scoreboard reads); a re-score of 20 speechocean762 clips and
24 cv-real clips with the current code (1.0.2) gave identical outputs.

## Summary

- **OpenPronounce's headline figure reproduces.** On all 2,500 speechocean762 test recordings, sentence
  known, its overall score follows the expert total at Pearson 0.790 (everyone), 0.850 (adults), 0.670
  (children of 12 and under): the 0.79 / 0.85 / 0.67 of its README, to the third decimal. Its own
  `benchmarks/speechocean762.py --report` and its own `openpronounce-evaluate` print the same 0.790.
- **On the sentence scores of speechocean762 the two systems are level.** Pronunciada 0.762 against
  0.790 on the total, difference −0.028 [−0.094, +0.017], p = 0.44 (speakers resampled). Accuracy,
  fluency and prosody: level as well. On adults the two are within 0.01 on every score. The children's
  gap (0.558 against 0.670) is not significant (p = 0.57) and comes from one speaker: with speaker 1404
  left out the difference is +0.024 for Pronunciada; by rank (Spearman) Pronunciada is slightly
  higher on children, 0.590 against 0.562 (p = 0.20, level).
- **Pronunciada is ahead on finding bad words and sounds of learners of English** (speechocean762: word
  AUC 0.908 against 0.886, word Pearson 0.609 against 0.569, sound Pearson 0.685 against 0.534, all
  p ≤ 0.006), **on every European changed-word test** (fleurs-swap, 18 of 18 rows; Lingolix test 1,
  7 of 8 rows, the eighth level) **and on telling rejected Common Voice clips from accepted ones** in
  English, Spanish, French, German and Italian (Portuguese level).
- **OpenPronounce is ahead on the sounds annotators marked in L2-ARCTIC (AUC 0.743 against 0.712) and
  EpaDB (0.741 against 0.721)**, both significant.
- **The distilled student** (`OpenPronounce/wav2vec2-base-phones-en`) reaches 0.802 on the total
  (README: 0.802); still level with Pronunciada (p = 0.09).
- **`--fast` changes no number.** Its outputs are identical to the default mode's on all 2,500
  speechocean762 recordings and all 2,700 fleurs-swap records in six languages: the second checkpoint
  and the reference voice only feed the transcript and the pre-1.0 formula score.
- **Cost on the device** (200 clips, CPU, fastest thread setting of each): Pronunciada int8 0.047 s per
  clip, 365 MB peak memory, 91 MB of weights; OpenPronounce student 0.170 s, 1.4 GB, 396 MB; `--fast`
  0.296 s, 2.3 GB, 1.3 GB; default 0.735 s, 3.7 GB, 2.6 GB.

## Provenance

| | |
|---|---|
| OpenPronounce | 1.1.0, tag `v1.1.0` = commit `bd51333cebb9e7eb791bce45b710e520ca97e690` (2026-11-19), AGPL-3.0-or-later, installed from the clone (non-editable) with its `tts-piper` extra in its own venv (Python 3.12, torch 2.15.0, transformers 5.17.0, piper-tts 1.8.0) |
| its models | phone recognizer `facebook/wav2vec2-lv-60-espeak-cv-ft` @ ae45363 (1.26 GB); word recognizer and acoustic embeddings `facebook/wav2vec2-large-960h` @ bdeaacd (1.26 GB, default mode only); non-English default mode also loads `jonatasgrosman/wav2vec2-large-xlsr-53-{french,spanish,german,italian,portuguese}` for the transcript; student `OpenPronounce/wav2vec2-base-phones-en` @ 2e12ec2 (379 MB); Piper voice per language from `rhasspy/piper-voices`; scoring models are the JSON files shipped in the package (`models/*.json`) |
| downloads | Hugging Face only (the student was the one new download; the other checkpoints were already in the cache of the 0.4.0 run and were linked, not copied) |
| Pronunciada | 1.0.2, cached outputs of the shipped scorer, fp32, default strictness |
| machine | Apple M3 Pro, 12 cores, 18 GB, macOS; CPU only for every OpenPronounce run (`OPENPRONOUNCE_DEVICE=cpu`, 4 torch threads); the machine was shared with other jobs, which affects timings only |
| runner | `bench/rivals/openpronounce.py` (raw outputs, one JSON line per record, resumable), `bench/rivals/convert.py`, evaluations `bench/rivals/eval_so762.py`, `eval_shared.py`, `eval_lingolix.py`, `evaluate_compare.py`, on-device `bench/ondevice.py`; tables `results/tables/rivals_*.json`, `results/tables/ondevice/` |

### No network while scoring

OpenPronounce's default reference voice is gTTS, i.e. Google Translate, called over the network the
first time a sentence is scored. That was not allowed here. We used the project's documented offline
backend instead, Piper (`OPENPRONOUNCE_TTS=piper`, the default voice of each language). In the
worker process the `gtts` module is made unimportable and every non-local socket connection raises,
and models are read from the local cache with `HF_HUB_OFFLINE=1`; no run hit either guard. This choice
cannot move any number we compare: the reference voice feeds only the acoustic distance, which feeds
only `formula_score` (the score of versions up to 0.4); `score`, `scores` and the word reports come from
the phone recognizer and the trained models, and are identical with the voice switched off (`--fast`,
below). Their own `benchmarks/speechocean762.py` calls gTTS directly, so its inference loop was not
run; its `--report` was run on our outputs (below).

Two other settings, neither of which touches an output: `prosody=False` (the pitch and energy curves
for display; the scorer computes its own prosody features either way; the on-device timing of the
default mode keeps them on), and the recognizers' outputs are memoised per audio file so the three
prompts of a fleurs-swap recording share one forward pass.

### What its scorers were trained and tuned on

From the code, the model files (`trained_on` keys) and `benchmarks/README.md`:

- English word model (`word_mdd_en.json`) and scorer (`scorer_en.json`): speechocean762 **train** split
  (2,500 utterances, plus 742 synthetic negatives for the scorer). The word threshold (0.35) is the
  best F1 on the train split; the scorer's linear recalibration is fitted on out-of-fold train
  predictions. The test split is used to report.
- Design choices were compared on the test split, though: the README's tables pick the shipped
  configuration (hidden layer 20 of 24, 16 PCA dimensions, embeddings on, synthetic negatives on, trees
  over ridge) among alternatives whose test-split numbers are shown side by side. No weight was fitted
  on the test split, but its 0.79 is not a fully untouched held-out figure. Pronunciada's 0.762 comes
  from a scorer fitted and selected under this project's held-out rules (no test-split selection).
- Other languages: generic word model and scorer trained on the same speechocean762 train split
  without phone identity; per-language accepted-phone variants, word thresholds and a "native
  reference" shift (`native_<lang>.json`) chosen on 120 FLEURS **dev** recordings per language. Our
  fleurs-swap test uses FLEURS **test** recordings, so there is no overlap.
- Student: distilled (no labels) from the two large checkpoints on 27,065 English Common Voice 27 clips
  and the speechocean762 train audio; its own word model and scorer refit on the train split. Its
  model card says the gain over the large model probably comes from having heard the train split's
  population. Its Common Voice clips may include clips of our cv-real English and Lingolix English
  test items (their selection list is not published), so the student's rows on those two tests carry
  that caveat.

## speechocean762 test (step 2 and 3)

All 2,500 test recordings, sentence known, OpenPronounce default mode (identical to `--fast`). Age
groups by OpenPronounce's rule (`benchmarks/groups.py`): children 12 and under, teenagers 13 to 17,
adults 18 and over. Intervals: 1,000 speaker resamples; difference = Pronunciada − OpenPronounce on
the same resamples, two-sided p (0.002 is the smallest p 1,000 resamples can give). Word-level: expert
word accuracy ≤ 6 (as asked) and < 5 (OpenPronounce's own label), error signal = minus the word score;
OpenPronounce's word score = 1 − its word model's probability (a word it calls "not said" takes the
larger of that and its missing-word probability), attached to our reference words by character span.
Sound level: its phone model's per-sound probability, aligned to our reference sounds inside each word
(97.6 % of the sounds found a partner; the rest take their word's score).

| group | test | n | Pronunciada 1.0.2 | OpenPronounce 1.1.0 | difference (P − O) [95 % CI] | p | verdict |
|---|---|--:|---|---|---|--:|---|
| everyone | sentence total | 2500 | 0.762 [0.700, 0.811] | 0.790 [0.728, 0.837] | -0.028 [-0.094, +0.017] | 0.436 | level |
| everyone | sentence accuracy | 2500 | 0.748 [0.685, 0.794] | 0.744 [0.679, 0.791] | +0.004 [-0.034, +0.035] | 0.729 | level |
| everyone | sentence fluency | 2500 | 0.779 [0.695, 0.836] | 0.827 [0.776, 0.862] | -0.048 [-0.147, +0.016] | 0.394 | level |
| everyone | sentence prosodic | 2500 | 0.772 [0.693, 0.830] | 0.808 [0.750, 0.850] | -0.036 [-0.132, +0.029] | 0.609 | level |
| everyone | sentence total, Spearman | 2500 | 0.734 [0.677, 0.778] | 0.727 [0.670, 0.772] | +0.006 [-0.015, +0.030] | 0.589 | level |
| everyone | word Pearson | 15967 | 0.609 [0.533, 0.663] | 0.569 [0.481, 0.629] | +0.040 [+0.011, +0.074] | 0.006 | Pronunciada ahead |
| everyone | word AUC, accuracy <= 6 | 15967 | 0.908 [0.888, 0.924] | 0.886 [0.862, 0.902] | +0.023 [+0.014, +0.034] | 0.002 | Pronunciada ahead |
| everyone | word AUC, accuracy < 5 | 15967 | 0.929 [0.911, 0.944] | 0.928 [0.910, 0.941] | +0.002 [-0.006, +0.011] | 0.669 | level |
| everyone | sound Pearson | 47369 | 0.685 [0.613, 0.731] | 0.534 [0.482, 0.577] | +0.150 [+0.122, +0.172] | 0.002 | Pronunciada ahead |
| adults (18+) | sentence total | 1220 | 0.844 [0.794, 0.881] | 0.850 [0.804, 0.885] | -0.006 [-0.021, +0.009] | 0.426 | level |
| adults (18+) | sentence accuracy | 1220 | 0.823 [0.772, 0.860] | 0.820 [0.772, 0.856] | +0.003 [-0.015, +0.020] | 0.709 | level |
| adults (18+) | sentence fluency | 1220 | 0.868 [0.827, 0.893] | 0.875 [0.835, 0.898] | -0.007 [-0.021, +0.006] | 0.284 | level |
| adults (18+) | sentence prosodic | 1220 | 0.848 [0.794, 0.883] | 0.850 [0.795, 0.883] | -0.002 [-0.018, +0.014] | 0.819 | level |
| adults (18+) | sentence total, Spearman | 1220 | 0.821 [0.746, 0.859] | 0.828 [0.753, 0.865] | -0.007 [-0.028, +0.016] | 0.559 | level |
| adults (18+) | word Pearson | 8701 | 0.656 [0.571, 0.713] | 0.606 [0.519, 0.667] | +0.051 [+0.024, +0.081] | 0.002 | Pronunciada ahead |
| adults (18+) | word AUC, accuracy <= 6 | 8701 | 0.920 [0.899, 0.937] | 0.901 [0.878, 0.920] | +0.019 [+0.011, +0.029] | 0.002 | Pronunciada ahead |
| adults (18+) | word AUC, accuracy < 5 | 8701 | 0.934 [0.911, 0.951] | 0.932 [0.912, 0.947] | +0.002 [-0.008, +0.012] | 0.655 | level |
| adults (18+) | sound Pearson | 25752 | 0.714 [0.641, 0.763] | 0.571 [0.507, 0.623] | +0.144 [+0.115, +0.170] | 0.002 | Pronunciada ahead |
| children (12 and under) | sentence total | 1040 | 0.558 [0.467, 0.652] | 0.670 [0.449, 0.790] | -0.112 [-0.261, +0.073] | 0.571 | level |
| children (12 and under) | sentence accuracy | 1040 | 0.540 [0.432, 0.632] | 0.580 [0.381, 0.703] | -0.040 [-0.166, +0.094] | 0.751 | level |
| children (12 and under) | sentence fluency | 1040 | 0.553 [0.410, 0.702] | 0.745 [0.566, 0.834] | -0.191 [-0.400, +0.045] | 0.354 | level |
| children (12 and under) | sentence prosodic | 1040 | 0.573 [0.435, 0.713] | 0.736 [0.534, 0.833] | -0.163 [-0.379, +0.079] | 0.621 | level |
| children (12 and under) | sentence total, Spearman | 1040 | 0.590 [0.510, 0.651] | 0.562 [0.475, 0.636] | +0.028 [-0.011, +0.073] | 0.200 | level |
| children (12 and under) | word Pearson | 5517 | 0.512 [0.361, 0.610] | 0.502 [0.277, 0.629] | +0.010 [-0.039, +0.103] | 0.659 | level |
| children (12 and under) | word AUC, accuracy <= 6 | 5517 | 0.874 [0.829, 0.906] | 0.845 [0.792, 0.882] | +0.029 [+0.007, +0.056] | 0.004 | Pronunciada ahead |
| children (12 and under) | word AUC, accuracy < 5 | 5517 | 0.902 [0.844, 0.933] | 0.899 [0.838, 0.929] | +0.003 [-0.021, +0.025] | 0.801 | level |
| children (12 and under) | sound Pearson | 16404 | 0.640 [0.493, 0.720] | 0.488 [0.418, 0.555] | +0.152 [+0.063, +0.177] | 0.002 | Pronunciada ahead |
| teenagers (13-17) | sentence total | 240 | 0.453 [0.346, 0.517] | 0.475 [0.302, 0.585] | -0.022 [-0.088, +0.064] | 0.581 | level |
| teenagers (13-17) | sentence accuracy | 240 | 0.430 [0.338, 0.515] | 0.428 [0.279, 0.523] | +0.002 [-0.071, +0.101] | 0.961 | level |
| teenagers (13-17) | sentence fluency | 240 | 0.422 [0.248, 0.528] | 0.434 [0.294, 0.532] | -0.011 [-0.095, +0.053] | 0.749 | level |
| teenagers (13-17) | sentence prosodic | 240 | 0.459 [0.281, 0.566] | 0.470 [0.297, 0.573] | -0.012 [-0.084, +0.054] | 0.659 | level |
| teenagers (13-17) | sentence total, Spearman | 240 | 0.447 [0.320, 0.532] | 0.431 [0.268, 0.545] | +0.016 [-0.049, +0.083] | 0.653 | level |
| teenagers (13-17) | word Pearson | 1749 | 0.400 [0.308, 0.490] | 0.236 [0.118, 0.323] | +0.164 [+0.049, +0.309] | 0.002 | Pronunciada ahead |
| teenagers (13-17) | word AUC, accuracy <= 6 | 1749 | 0.881 [0.822, 0.930] | 0.805 [0.738, 0.864] | +0.075 [+0.037, +0.117] | 0.002 | Pronunciada ahead |
| teenagers (13-17) | word AUC, accuracy < 5 | 1749 | 0.959 [0.929, 0.982] | 0.940 [0.903, 0.970] | +0.019 [-0.025, +0.064] | 0.398 | level |
| teenagers (13-17) | sound Pearson | 5213 | 0.462 [0.410, 0.507] | 0.384 [0.356, 0.413] | +0.079 [+0.043, +0.107] | 0.002 | Pronunciada ahead |

### The children's gap

OpenPronounce's 0.670 on children against Pronunciada's 0.558 is the largest gap in its favour, and it
is not robust: the speaker-resampled interval of the difference runs from −0.26 to +0.07 (p = 0.57),
and leaving one speaker out (1404, mean expert total 3.5 over 20 recordings) turns it into +0.024 for
Pronunciada. By rank the two are level with Pronunciada slightly ahead (Spearman 0.590 against 0.562).
Averaged per speaker, OpenPronounce follows the experts more closely on children (0.919 against 0.757):
it separates weak speakers from strong ones better, while within a speaker neither does well.

### Their own scripts

- `benchmarks/speechocean762.py --report` (their analysis code, run on a CSV of our run in their
  layout, `bench/rivals/their_csv.py`): score vs human total 0.790 (Spearman 0.727), accuracy 0.755,
  fluency 0.789, prosodic 0.778 (their report correlates the one total score with every dimension);
  826 of 15,967 words flagged, 52.5 % of them rated below 5 by the experts, 50.2 % of those found
  (README: "one word in twenty is reported, and the raters rejected 54 % of those").
- `openpronounce-evaluate` on the 2,500 recordings (score = expert total): all 0.790 [0.77, 0.81],
  adults 0.850, children 0.670, teenagers 0.475. Its interval resamples recordings, not speakers, so it
  is narrower than ours ([0.728, 0.837]); with that resampling Pronunciada's gap would look significant
  (p = 0.026 on everyone, 0.012 on children), which it is not once a speaker's recordings move together.
  Its Spearman (0.697) ranks ties in file order; scipy's tie-averaged Spearman is 0.727.

### Student and `--fast`

`--fast` (phone recognizer only): every score, every sub-score and every word probability is identical
to the default mode on all 2,500 recordings (and on all 2,700 fleurs-swap records, six languages), so
its table is the table above.

Student (`OPENPRONOUNCE_PHONEME_MODEL=OpenPronounce/wav2vec2-base-phones-en`, `--fast`, its own word model and scorer):

| group | test | n | Pronunciada 1.0.2 | OpenPronounce 1.1.0 | difference (P − O) [95 % CI] | p | verdict |
|---|---|--:|---|---|---|--:|---|
| everyone | sentence total | 2500 | 0.762 [0.700, 0.811] | 0.802 [0.747, 0.841] | -0.040 [-0.100, +0.003] | 0.092 | level |
| everyone | sentence accuracy | 2500 | 0.748 [0.685, 0.794] | 0.766 [0.714, 0.805] | -0.018 [-0.054, +0.008] | 0.210 | level |
| everyone | sentence fluency | 2500 | 0.779 [0.695, 0.836] | 0.837 [0.790, 0.868] | -0.058 [-0.149, +0.004] | 0.100 | level |
| everyone | sentence prosodic | 2500 | 0.772 [0.693, 0.830] | 0.821 [0.767, 0.859] | -0.049 [-0.138, +0.011] | 0.222 | level |
| everyone | sentence total, Spearman | 2500 | 0.734 [0.677, 0.778] | 0.730 [0.673, 0.776] | +0.004 [-0.017, +0.026] | 0.763 | level |
| everyone | word Pearson | 15967 | 0.609 [0.533, 0.663] | 0.609 [0.518, 0.669] | -0.000 [-0.031, +0.037] | 0.963 | level |
| everyone | word AUC, accuracy <= 6 | 15967 | 0.908 [0.888, 0.924] | 0.892 [0.868, 0.910] | +0.016 [+0.005, +0.029] | 0.004 | Pronunciada ahead |
| everyone | word AUC, accuracy < 5 | 15967 | 0.929 [0.911, 0.944] | 0.938 [0.920, 0.950] | -0.008 [-0.017, +0.001] | 0.106 | level |
| everyone | sound Pearson | 47369 | 0.685 [0.613, 0.731] | 0.563 [0.508, 0.608] | +0.122 [+0.100, +0.141] | 0.002 | Pronunciada ahead |
| adults (18+) | sentence total | 1220 | 0.844 [0.794, 0.881] | 0.850 [0.797, 0.884] | -0.006 [-0.018, +0.008] | 0.442 | level |
| adults (18+) | sentence accuracy | 1220 | 0.823 [0.772, 0.860] | 0.826 [0.779, 0.859] | -0.003 [-0.020, +0.013] | 0.767 | level |
| adults (18+) | sentence fluency | 1220 | 0.868 [0.827, 0.893] | 0.875 [0.832, 0.900] | -0.007 [-0.022, +0.009] | 0.408 | level |
| adults (18+) | sentence prosodic | 1220 | 0.848 [0.794, 0.883] | 0.855 [0.798, 0.891] | -0.007 [-0.022, +0.009] | 0.352 | level |
| adults (18+) | sentence total, Spearman | 1220 | 0.821 [0.746, 0.859] | 0.828 [0.755, 0.866] | -0.007 [-0.027, +0.014] | 0.496 | level |
| adults (18+) | word Pearson | 8701 | 0.656 [0.571, 0.713] | 0.637 [0.552, 0.693] | +0.019 [-0.011, +0.052] | 0.196 | level |
| adults (18+) | word AUC, accuracy <= 6 | 8701 | 0.920 [0.899, 0.937] | 0.908 [0.884, 0.926] | +0.013 [+0.004, +0.024] | 0.008 | Pronunciada ahead |
| adults (18+) | word AUC, accuracy < 5 | 8701 | 0.934 [0.911, 0.951] | 0.942 [0.924, 0.956] | -0.008 [-0.018, +0.002] | 0.096 | level |
| adults (18+) | sound Pearson | 25752 | 0.714 [0.641, 0.763] | 0.597 [0.529, 0.650] | +0.118 [+0.094, +0.138] | 0.002 | Pronunciada ahead |
| children (12 and under) | sentence total | 1040 | 0.558 [0.467, 0.652] | 0.709 [0.496, 0.816] | -0.151 [-0.277, +0.023] | 0.134 | level |
| children (12 and under) | sentence accuracy | 1040 | 0.540 [0.432, 0.632] | 0.636 [0.430, 0.747] | -0.096 [-0.198, +0.041] | 0.222 | level |
| children (12 and under) | sentence fluency | 1040 | 0.553 [0.410, 0.702] | 0.777 [0.601, 0.849] | -0.224 [-0.403, +0.003] | 0.058 | level |
| children (12 and under) | sentence prosodic | 1040 | 0.573 [0.435, 0.713] | 0.766 [0.572, 0.847] | -0.192 [-0.379, +0.041] | 0.224 | level |
| children (12 and under) | sentence total, Spearman | 1040 | 0.590 [0.510, 0.651] | 0.576 [0.485, 0.653] | +0.014 [-0.024, +0.056] | 0.484 | level |
| children (12 and under) | word Pearson | 5517 | 0.512 [0.361, 0.610] | 0.557 [0.281, 0.685] | -0.045 [-0.100, +0.093] | 0.458 | level |
| children (12 and under) | word AUC, accuracy <= 6 | 5517 | 0.874 [0.829, 0.906] | 0.851 [0.798, 0.890] | +0.023 [-0.000, +0.051] | 0.060 | level |
| children (12 and under) | word AUC, accuracy < 5 | 5517 | 0.902 [0.844, 0.933] | 0.911 [0.850, 0.943] | -0.009 [-0.030, +0.013] | 0.416 | level |
| children (12 and under) | sound Pearson | 16404 | 0.640 [0.493, 0.720] | 0.521 [0.440, 0.591] | +0.119 [+0.049, +0.153] | 0.002 | Pronunciada ahead |
| teenagers (13-17) | sentence total | 240 | 0.453 [0.346, 0.517] | 0.410 [0.243, 0.508] | +0.043 [-0.033, +0.139] | 0.304 | level |
| teenagers (13-17) | sentence accuracy | 240 | 0.430 [0.338, 0.515] | 0.377 [0.243, 0.474] | +0.053 [-0.049, +0.160] | 0.292 | level |
| teenagers (13-17) | sentence fluency | 240 | 0.422 [0.248, 0.528] | 0.420 [0.271, 0.516] | +0.002 [-0.062, +0.062] | 0.895 | level |
| teenagers (13-17) | sentence prosodic | 240 | 0.459 [0.281, 0.566] | 0.499 [0.345, 0.595] | -0.040 [-0.097, +0.004] | 0.082 | level |
| teenagers (13-17) | sentence total, Spearman | 240 | 0.447 [0.320, 0.532] | 0.362 [0.250, 0.439] | +0.085 [+0.031, +0.139] | 0.006 | Pronunciada ahead |
| teenagers (13-17) | word Pearson | 1749 | 0.400 [0.308, 0.490] | 0.213 [0.085, 0.347] | +0.187 [+0.046, +0.341] | 0.006 | Pronunciada ahead |
| teenagers (13-17) | word AUC, accuracy <= 6 | 1749 | 0.881 [0.822, 0.930] | 0.823 [0.779, 0.867] | +0.058 [-0.008, +0.115] | 0.086 | level |
| teenagers (13-17) | word AUC, accuracy < 5 | 1749 | 0.959 [0.929, 0.982] | 0.918 [0.858, 0.965] | +0.041 [-0.014, +0.110] | 0.192 | level |
| teenagers (13-17) | sound Pearson | 5213 | 0.462 [0.410, 0.507] | 0.396 [0.377, 0.414] | +0.066 [+0.012, +0.111] | 0.022 | Pronunciada ahead |

## Shared tests (step 4)

These were run in `--fast` mode after the identity above was established (the default-mode fleurs-swap
run gives the same 18 rows to the last digit). A record OpenPronounce could not score would get its
floor (0); none failed.

**OpenPronounce default / --fast (identical outputs)**

| test | n | Pronunciada 1.0.2 | OpenPronounce 1.1.0 | difference (P − O) [95 % CI] | p | verdict |
|---|--:|---|---|---|--:|---|
| marked sounds, L2-ARCTIC test | 29760 | 0.712 [0.647, 0.759] | 0.743 [0.688, 0.780] | -0.031 [-0.056, -0.012] | 0.002 | OpenPronounce ahead |
| marked sounds, EpaDB test | 20665 | 0.721 [0.707, 0.735] | 0.741 [0.726, 0.755] | -0.019 [-0.031, -0.009] | 0.006 | OpenPronounce ahead |
| fleurs-swap en far AUC | 150 | 0.995 [0.992, 0.997] | 0.976 [0.968, 0.983] | +0.019 [+0.012, +0.025] | 0.002 | Pronunciada ahead |
| fleurs-swap en near AUC | 150 | 0.981 [0.973, 0.987] | 0.900 [0.877, 0.921] | +0.081 [+0.061, +0.102] | 0.002 | Pronunciada ahead |
| fleurs-swap en near hit@5 % | 150 | 0.893 [0.840, 0.940] | 0.413 [0.280, 0.540] | +0.480 [+0.346, +0.620] | 0.002 | Pronunciada ahead |
| fleurs-swap es far AUC | 150 | 0.998 [0.996, 0.999] | 0.986 [0.981, 0.991] | +0.011 [+0.007, +0.015] | 0.002 | Pronunciada ahead |
| fleurs-swap es near AUC | 150 | 0.990 [0.985, 0.993] | 0.894 [0.874, 0.914] | +0.095 [+0.076, +0.116] | 0.002 | Pronunciada ahead |
| fleurs-swap es near hit@5 % | 150 | 0.967 [0.933, 0.993] | 0.400 [0.322, 0.500] | +0.567 [+0.460, +0.647] | 0.002 | Pronunciada ahead |
| fleurs-swap fr far AUC | 150 | 0.997 [0.995, 0.998] | 0.980 [0.973, 0.985] | +0.017 [+0.012, +0.023] | 0.002 | Pronunciada ahead |
| fleurs-swap fr near AUC | 150 | 0.986 [0.981, 0.991] | 0.915 [0.899, 0.930] | +0.072 [+0.057, +0.087] | 0.002 | Pronunciada ahead |
| fleurs-swap fr near hit@5 % | 150 | 0.927 [0.887, 0.973] | 0.456 [0.327, 0.571] | +0.471 [+0.360, +0.600] | 0.002 | Pronunciada ahead |
| fleurs-swap de far AUC | 150 | 0.996 [0.994, 0.997] | 0.987 [0.983, 0.991] | +0.008 [+0.004, +0.013] | 0.002 | Pronunciada ahead |
| fleurs-swap de near AUC | 150 | 0.960 [0.943, 0.975] | 0.885 [0.860, 0.912] | +0.075 [+0.054, +0.095] | 0.002 | Pronunciada ahead |
| fleurs-swap de near hit@5 % | 150 | 0.767 [0.700, 0.860] | 0.511 [0.420, 0.620] | +0.255 [+0.153, +0.360] | 0.002 | Pronunciada ahead |
| fleurs-swap it far AUC | 150 | 0.998 [0.997, 0.999] | 0.978 [0.972, 0.983] | +0.020 [+0.015, +0.026] | 0.002 | Pronunciada ahead |
| fleurs-swap it near AUC | 150 | 0.979 [0.971, 0.985] | 0.854 [0.834, 0.874] | +0.124 [+0.105, +0.145] | 0.002 | Pronunciada ahead |
| fleurs-swap it near hit@5 % | 150 | 0.887 [0.827, 0.933] | 0.273 [0.213, 0.348] | +0.613 [+0.519, +0.688] | 0.002 | Pronunciada ahead |
| fleurs-swap pt far AUC | 150 | 0.997 [0.996, 0.998] | 0.976 [0.969, 0.982] | +0.021 [+0.015, +0.028] | 0.002 | Pronunciada ahead |
| fleurs-swap pt near AUC | 150 | 0.963 [0.945, 0.979] | 0.820 [0.788, 0.852] | +0.143 [+0.114, +0.174] | 0.002 | Pronunciada ahead |
| fleurs-swap pt near hit@5 % | 150 | 0.853 [0.800, 0.907] | 0.347 [0.256, 0.440] | +0.507 [+0.410, +0.605] | 0.002 | Pronunciada ahead |
| cv-real en/A | 400 | 0.748 [0.698, 0.799] | 0.699 [0.646, 0.752] | +0.049 [+0.003, +0.098] | 0.045 | Pronunciada ahead |
| cv-real en/B | 260 | 0.517 [0.445, 0.588] | 0.472 [0.398, 0.547] | +0.044 [-0.038, +0.122] | 0.248 | level |
| cv-real es/A | 396 | 0.766 [0.718, 0.811] | 0.656 [0.599, 0.712] | +0.110 [+0.049, +0.175] | <0.001 | Pronunciada ahead |
| cv-real fr/A | 396 | 0.789 [0.743, 0.832] | 0.679 [0.627, 0.733] | +0.110 [+0.055, +0.165] | <0.001 | Pronunciada ahead |
| cv-real de/A | 396 | 0.823 [0.782, 0.863] | 0.676 [0.624, 0.726] | +0.147 [+0.094, +0.198] | <0.001 | Pronunciada ahead |
| cv-real de/B | 102 | 0.541 [0.423, 0.649] | 0.502 [0.377, 0.626] | +0.039 [-0.110, +0.172] | 0.587 | level |
| cv-real it/A | 398 | 0.767 [0.718, 0.814] | 0.687 [0.631, 0.739] | +0.080 [+0.026, +0.135] | 0.005 | Pronunciada ahead |
| cv-real pt/A | 400 | 0.729 [0.677, 0.779] | 0.677 [0.621, 0.729] | +0.053 [-0.009, +0.117] | 0.091 | level |

**OpenPronounce student (English only)**

| test | n | Pronunciada 1.0.2 | OpenPronounce 1.1.0 | difference (P − O) [95 % CI] | p | verdict |
|---|--:|---|---|---|--:|---|
| marked sounds, L2-ARCTIC test | 29760 | 0.712 [0.647, 0.759] | 0.753 [0.685, 0.796] | -0.040 [-0.065, -0.020] | 0.002 | OpenPronounce ahead |
| marked sounds, EpaDB test | 20665 | 0.721 [0.707, 0.735] | 0.739 [0.726, 0.752] | -0.018 [-0.028, -0.009] | 0.002 | OpenPronounce ahead |
| fleurs-swap en far AUC | 150 | 0.995 [0.992, 0.997] | 0.976 [0.967, 0.984] | +0.019 [+0.012, +0.026] | 0.002 | Pronunciada ahead |
| fleurs-swap en near AUC | 150 | 0.981 [0.973, 0.987] | 0.900 [0.875, 0.921] | +0.081 [+0.062, +0.103] | 0.002 | Pronunciada ahead |
| fleurs-swap en near hit@5 % | 150 | 0.893 [0.840, 0.940] | 0.500 [0.300, 0.612] | +0.393 [+0.284, +0.593] | 0.002 | Pronunciada ahead |
| cv-real en/A | 400 | 0.748 [0.698, 0.799] | 0.720 [0.668, 0.769] | +0.029 [-0.016, +0.073] | 0.215 | level |
| cv-real en/B | 260 | 0.517 [0.445, 0.588] | 0.529 [0.453, 0.606] | -0.012 [-0.087, +0.063] | 0.784 | level |

The default-mode file `rivals_shared_default.json` holds the 18 fleurs-swap rows run without `--fast`; they equal the rows above.

### Lingolix benchmark (v2026.10)

Same metric code as Pronunciada's own run (`bench/third_party/lingolix.py`). Test 1: 200 Common Voice
clips per language with one word of the prompt changed; FA@80 = share of correct words flagged at the
threshold that catches 80 % of the changed words (lower is better, so a negative difference favours
Pronunciada). Test 2: 500 speechocean762 test clips (taken from the run above). Differences on 1,000
resamples of utterances (test 1, as the benchmark resamples) or speakers (test 2).

| test | metric | Pronunciada 1.0.2 | OpenPronounce 1.1.0 | difference (P − O) [95 % CI] | p |
|---|---|---|---|---|--:|
| test 1, en | word AUC (higher better) | 0.977 [0.963, 0.988] | 0.886 [0.853, 0.919] | +0.090 [+0.059, +0.122] | 0.002 |
| test 1, en | FA@80 (lower better) | 0.025 [0.010, 0.039] | 0.173 [0.123, 0.269] | -0.148 [-0.242, -0.102] | 0.002 |
| test 1, fr | word AUC (higher better) | 0.960 [0.924, 0.987] | 0.877 [0.833, 0.914] | +0.083 [+0.054, +0.116] | 0.002 |
| test 1, fr | FA@80 (lower better) | 0.017 [0.009, 0.031] | 0.143 [0.098, 0.246] | -0.127 [-0.225, -0.081] | 0.002 |
| test 1, es | word AUC (higher better) | 0.980 [0.970, 0.988] | 0.929 [0.903, 0.951] | +0.050 [+0.032, +0.071] | 0.002 |
| test 1, es | FA@80 (lower better) | 0.024 [0.014, 0.039] | 0.090 [0.061, 0.136] | -0.066 [-0.110, -0.039] | 0.002 |
| test 1, de | word AUC (higher better) | 0.941 [0.911, 0.965] | 0.887 [0.844, 0.927] | +0.054 [+0.010, +0.100] | 0.016 |
| test 1, de | FA@80 (lower better) | 0.067 [0.029, 0.131] | 0.149 [0.074, 0.306] | -0.082 [-0.219, +0.004] | 0.060 |
| test 2 | word Pearson | 0.581 [0.489, 0.644] | 0.610 [0.494, 0.690] | -0.029 [-0.092, +0.030] | 0.394 |
| test 2 | utterance Pearson | 0.763 [0.658, 0.832] | 0.819 [0.734, 0.862] | -0.057 [-0.142, +0.003] | 0.084 |

Student, English only:

| test | metric | Pronunciada 1.0.2 | OpenPronounce 1.1.0 | difference (P − O) [95 % CI] | p |
|---|---|---|---|---|--:|
| test 1, en | word AUC (higher better) | 0.977 [0.963, 0.988] | 0.913 [0.884, 0.940] | +0.064 [+0.038, +0.092] | 0.002 |
| test 1, en | FA@80 (lower better) | 0.025 [0.010, 0.039] | 0.137 [0.082, 0.219] | -0.113 [-0.193, -0.061] | 0.002 |
| test 2 | word Pearson | 0.581 [0.489, 0.644] | 0.651 [0.529, 0.735] | -0.071 [-0.146, -0.004] | 0.042 |
| test 2 | utterance Pearson | 0.763 [0.658, 0.832] | 0.816 [0.717, 0.866] | -0.053 [-0.134, +0.009] | 0.148 |

OpenPronounce's own flags (the words it reports) catch 13 % (English), 28 % (French), 24 % (Spanish)
and 28 % (German) of the changed words, with 0.4 to 2.5 % of the correct words flagged; Pronunciada's
catch 77, 87, 83 and 42 % with 1.2 to 8.1 % flagged. At equal false alarms (the AUC and FA@80 rows)
Pronunciada separates them better in every language.

### `openpronounce-evaluate` on our European manifests

Its own command (`--fast` path: acoustic, transcription and prosody off) on the cv-real test manifests,
with score = up votes − down votes of the Common Voice validators, and Pronunciada's primary cv-real
score on the same rows with the same statistic (Pearson, 1,000 recording resamples, seed 0). Test B
rows are native against non-native accepted clips, whose net votes carry no pronunciation rating; they
are printed by the command but say nothing.

| manifest | rows | n | OpenPronounce Pearson [95 % CI] | Pronunciada Pearson [95 % CI] | difference (P − O) [95 % CI] | p |
|---|---|--:|---|---|---|--:|
| cv-real-de | all | 498 | 0.318 [0.25, 0.37] | 0.370 [0.33, 0.41] | +0.052 [+0.003, +0.107] | 0.038 |
| cv-real-de | test=A | 396 | 0.326 [0.27, 0.38] | 0.350 [0.31, 0.39] | +0.024 [-0.022, +0.073] | 0.286 |
| cv-real-en | all | 660 | 0.441 [0.38, 0.50] | 0.486 [0.43, 0.54] | +0.046 [-0.010, +0.104] | 0.116 |
| cv-real-en | test=A | 400 | 0.413 [0.35, 0.47] | 0.445 [0.37, 0.51] | +0.032 [-0.035, +0.094] | 0.340 |
| cv-real-es | all | 396 | 0.276 [0.21, 0.35] | 0.253 [0.22, 0.29] | -0.023 [-0.088, +0.048] | 0.527 |
| cv-real-es | test=A | 396 | 0.276 [0.21, 0.35] | 0.253 [0.22, 0.29] | -0.023 [-0.088, +0.048] | 0.527 |
| cv-real-fr | all | 396 | 0.328 [0.25, 0.40] | 0.357 [0.29, 0.41] | +0.029 [-0.025, +0.086] | 0.288 |
| cv-real-fr | test=A | 396 | 0.328 [0.25, 0.40] | 0.357 [0.29, 0.41] | +0.029 [-0.025, +0.086] | 0.288 |
| cv-real-it | all | 398 | 0.340 [0.27, 0.41] | 0.333 [0.28, 0.38] | -0.008 [-0.063, +0.056] | 0.775 |
| cv-real-it | test=A | 398 | 0.340 [0.27, 0.41] | 0.333 [0.28, 0.38] | -0.008 [-0.063, +0.056] | 0.775 |
| cv-real-pt | all | 400 | 0.293 [0.23, 0.36] | 0.301 [0.25, 0.35] | +0.008 [-0.051, +0.072] | 0.779 |
| cv-real-pt | test=A | 400 | 0.293 [0.23, 0.36] | 0.301 [0.25, 0.35] | +0.008 [-0.051, +0.072] | 0.779 |
| speechocean762-test | all | 2500 | 0.790 [0.77, 0.81] | 0.762 [0.74, 0.78] | -0.028 [-0.054, -0.005] | 0.026 |
| speechocean762-test | age_group=adult_18_plus | 1220 | 0.850 [0.83, 0.87] | 0.844 [0.83, 0.86] | -0.006 [-0.021, +0.010] | 0.452 |
| speechocean762-test | age_group=child_12_and_under | 1040 | 0.670 [0.60, 0.73] | 0.558 [0.51, 0.61] | -0.112 [-0.182, -0.028] | 0.012 |
| speechocean762-test | age_group=teen_13_17 | 240 | 0.475 [0.34, 0.59] | 0.453 [0.35, 0.55] | -0.022 [-0.106, +0.074] | 0.649 |

Dutch: OpenPronounce supports it (FLEURS-checked native thresholds and reference); Pronunciada does not,
so no shared test exists.

## On-device cost (step 5)

`python -m bench.ondevice --system <name> --threads <n>`, each run in a fresh process through the
heavy-job lock, CPU only, first 200 speechocean762 test clips (mean 3.4 s of audio), audio decoding
inside the timed call. The sweep below was run back to back (06:52 to 07:06); the machine was shared
with other jobs, and the last column is its 1-minute load average when each run started.
Pronunciada's runs happened to meet the heaviest load (8 to 10), OpenPronounce's default mode the
lightest (4 to 5), so the gaps are if anything understated. "load" = launch to models loaded, "cold
start" = launch to the first result; RTF = compute seconds per second of audio.

| system | threads | load s | cold start s | p50 s/clip | p95 s/clip | RTF | peak RSS MB | weights MB | machine load |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Pronunciada fp32 | 1 | 0.8 | 1.0 | 0.132 | 0.190 | 0.040 | 533 | 281 | 9.9 |
| Pronunciada fp32 | 4 | 0.8 | 0.9 | 0.085 | 0.118 | 0.026 | 532 | 281 | 8.3 |
| Pronunciada int8 | 1 | 0.6 | 0.7 | 0.068 | 0.098 | 0.021 | 359 | 91 | 8.7 |
| Pronunciada int8 | 4 | 0.8 | 0.9 | 0.047 | 0.078 | 0.015 | 365 | 91 | 10.0 |
| OpenPronounce default | 1 | 3.1 | 4.8 | 0.887 | 1.183 | 0.265 | 3681 | 2603 | 4.1 |
| OpenPronounce default | 4 | 3.1 | 6.3 | 0.735 | 0.946 | 0.221 | 3712 | 2603 | 4.7 |
| OpenPronounce --fast | 1 | 3.4 | 4.3 | 0.296 | 0.399 | 0.089 | 2378 | 1278 | 7.9 |
| OpenPronounce --fast | 4 | 3.3 | 6.3 | 0.444 | 0.713 | 0.140 | 2329 | 1278 | 7.5 |
| OpenPronounce student | 1 | 3.2 | 4.0 | 0.170 | 0.228 | 0.051 | 1445 | 396 | 7.3 |
| OpenPronounce student | 4 | 4.2 | 5.8 | 0.210 | 0.399 | 0.070 | 1439 | 396 | 6.9 |

Weights: Pronunciada = the ZIPA-small ONNX file (260 MB fp32 / 71 MB int8) + its package data (20 MB);
OpenPronounce default = the two wav2vec2-large checkpoints (1,264 + 1,262 MB) + the Piper voice (63 MB)
+ its package models; `--fast` = the phone checkpoint only; student = its 379 MB safetensors file (the
project also publishes a 123 MB int8 ONNX export of it, which the library does not use). The default
mode for a non-English language loads a third 1.26 GB checkpoint (the language's XLSR transcriber),
not measured here.

Two oddities, reported as measured: OpenPronounce `--fast` and student were slower with 4 threads than
with 1 (the same happened in a first pass an hour earlier, at load 4 to 9), and a first pass gave
Pronunciada fp32 0.060 s and int8 0.043 s per clip at 4 threads (`results/tables/ondevice/` holds the
second pass only). The fastest setting of each system, side by side: Pronunciada int8 0.047 s per clip,
OpenPronounce student 0.170 s (3.6 times slower), `--fast` 0.296 s (6.3 times), default 0.735 s (16 times);
peak memory 365 MB against 1.4, 2.3 and 3.7 GB; weights 91 MB against 396 MB, 1.3 GB and 2.6 GB.

## Where each is ahead

Pronunciada 1.0.2 against OpenPronounce 1.1.0 (default mode; `--fast` is the same system), counting
rows with p < 0.05 on the project's resampling rules:

- **Pronunciada ahead**: English learners' bad words and sounds on speechocean762 (word AUC at ≤ 6,
  word Pearson, sound Pearson, for everyone and adults; word AUC and sound Pearson for children;
  word and sound for teenagers); all 18 fleurs-swap rows in English, French, Spanish, German, Italian
  and Portuguese (changed words, near ones by 0.07 to 0.14 of AUC); Lingolix test 1 word AUC in
  English, French, Spanish and German and FA@80 in English, French and Spanish; cv-real accepted vs
  rejected in English, Spanish, French, German and Italian; speed (3.6 to 16 times faster per clip,
  fastest setting of each), peak memory (4 to 10 times less) and weights (4 to 29 times smaller).
- **OpenPronounce ahead**: marked mispronounced sounds in L2-ARCTIC (0.743 against 0.712) and EpaDB
  (0.741 against 0.721); the student too (0.753 and 0.739). The student adds a lead on Lingolix test 2 word Pearson (0.651 against 0.581,
  p = 0.042), on 500 speechocean762 clips; on the full 2,500 the same student is level with
  Pronunciada on word Pearson (0.609 against 0.609).
- **Level**: every sentence score on speechocean762 (total, accuracy, fluency, prosody) for everyone,
  adults, children and teenagers; word AUC at < 5; Lingolix test 2; German FA@80; cv-real Portuguese
  and the native-vs-non-native tests. OpenPronounce's point estimates are higher on the sentence
  scores (total 0.790 against 0.762, fluency 0.827 against 0.779, prosody 0.808 against 0.772), and the
  intervals do not exclude a real lead of a few hundredths; on adults the two are within 0.01.
- **Coverage**: OpenPronounce covers Dutch, Pronunciada does not. Pronunciada covers Japanese, Korean,
  Mandarin and Russian, OpenPronounce does not.

Consequence for the public comparison: "Pronunciada is ahead of OpenPronounce on all 28 tests" (the
0.4.0 snapshot) no longer holds. Against 1.1.0 it is ahead on the word, sound, changed-word and
rejected-clip tests and on cost, level on every sentence score, and behind on the two marked-sound
corpora.

## Files

- `bench/rivals/openpronounce.py` runner (OpenPronounce worker in its own venv, driver here), `convert.py`,
  `eval_so762.py`, `eval_shared.py`, `eval_lingolix.py`, `evaluate_manifests.py`, `evaluate_compare.py`,
  `their_csv.py`, `tables.py`, `common.py` (copied from the 2026-11-18 snapshot branch).
- `bench/ondevice.py` on-device harness.
- `results/tables/rivals_so762.json`, `rivals_shared_default.json`, `rivals_shared_fast.json`,
  `rivals_shared_student.json`, `rivals_lingolix.json`, `rivals_evaluate.json`, `results/tables/ondevice/*.json`.
- Raw OpenPronounce outputs (not in the repo): `$PRONUNCIADA_CACHE/round19/rivals/raw/<mode>/<dataset>.jsonl`.
