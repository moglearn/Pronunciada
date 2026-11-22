# Round 20: Japanese vowel length from a padded window of encoder activations

## Outcome

**Negative result. No held-out look was spent.**

* **The head works as a duration detector.** A small 1-D CNN reads a fixed ±160 ms window of the
  encoder's stack-1 activations around each vowel's CTC peak. On dev-pool natives (speakers and
  clips outside every fit and held-out set) it separates long from short vowels with AUC 0.965,
  against 0.948 for the shipped round-19 length model. It detects one-vowel WSOLA mismatches
  against natives with AUC 0.975 clean and 0.959 at 5 dB SNR. The shipped model scores 0.819 and
  0.787 on the same pairs.
* **Fed into the word score, it does not pass the dev guards.** 70 lower-only settings were run
  through the runtime on the round-19 dev sets. Every one raised the share of native words below
  0.6. That fails the round-19 Japanese native guard, which allows no rise.
  * The best dev gain is AUC 0.8674 → 0.8717 (+0.0043 [+0.0008, +0.0083], p = 0.01).
  * FA@80 moves 17.4 % → 15.9 % (p = 0.43, not significant).
  * The cost is native words below 0.6: +0.27 points on FLEURS validation and +0.21 points on CV
    dev.
  * The gentlest settings that still help cost 2 of 7,219 FLEURS-validation native words. Their
    gain is +0.001 AUC.
* **Held-out.** The brief says not to spend the held-out look when dev shows no clear gain within
  the guards. The Lingolix Japanese benchmark, FLEURS test, the length table on FLEURS test,
  cv-real ja test and the contrast audit were **not run**. None of the held-out targets (beat
  0.95 / 5 % / 37 of 40 and 0.93 / 3 % / 36 of 40; long-vowel length row ≥ 60 %) were tested.
  Held-out looks this round: **0**.
* **What ships.** The head and its parameters are in the package, **off by default**:
  * files: `pronunciada/ja_length20.py`, `data/length/ja_len20.{onnx,json}`, 149 kB;
  * `PRONUNCIADA_JA_LENGTH20=1` turns the setting described under "Architect's option" on;
  * unset or `=0`, every output equals current main. This was checked at max |diff| = 0 for fp32
    and int8 (see "Equivalence and parity").

## 1. Frame rates and which stack carries duration

**Frame rates.** Every stack's output is on the 20 ms output grid: T′ frames, the same as the CTC
log-probabilities (`ALL_LAYER_OUTPUTS`, graph surgery). The stacks inside are downsampled and then
upsampled, though. The relative step |x[t+1] − x[t]| / |x[t]| over 20 Common Voice clips,
averaged by t mod 8, shows a small step inside each block and a large one at block edges:

| stack | step pattern (t mod 8) | effective rate |
|---|---|---|
| enc0 | flat (0.89–0.91) | 50 Hz |
| enc1 | 0.39 / 0.91 alternating | 25 Hz |
| enc2 | large every 4th | 12.5 Hz |
| enc3 | large every 8th | 6.25 Hz |
| enc4 | large every 4th | 12.5 Hz |
| enc5 | 0.35 / 0.75–0.91 | 25 Hz |
| final (CTC input) | as enc5 | 25 Hz |

So only enc0 is a true 50 Hz stack. Everything after the CTC layer's input is effectively 25 Hz.

**Linear probe** (`bench/round20_ja_length/probe.py`).

* Data: 5,140 natural fit-pool vowels (every long one, a third of the short ones; 1,038 long).
* Model, per stack: PCA to 24 dimensions per frame, fitted inside each fold; the window flattened;
  logistic regression.
* Validation: 4 folds grouped by speaker.
* Target: the text's long / short label.

| stack | ±60 ms | ±100 ms | ±160 ms | ±200 ms |
|---|---|---|---|---|
| enc0 | 0.936 | 0.935 | 0.933 | 0.931 |
| enc1 | 0.956 | **0.957** | 0.953 | 0.953 |
| enc2 | 0.956 | **0.957** | 0.954 | 0.953 |
| enc3 | 0.953 | 0.953 | 0.951 | 0.950 |
| enc4 | 0.953 | 0.951 | 0.949 | 0.948 |
| enc5 | 0.949 | 0.950 | 0.945 | 0.942 |
| final | 0.949 | 0.950 | 0.945 | 0.942 |
| enc0 + enc1 + enc5 | 0.960 | 0.957 | 0.951 | 0.949 |

* The shipped round-19 length probability, read as P(long) on the same vowels, scores 0.911. That
  is in-sample, because it was fitted on these clips.
* Duration is carried best by the early-to-middle stacks enc1 and enc2. The full-rate enc0 is the
  worst. The head uses **enc1**, which the Japanese scorer already exports, so the acoustic model
  file is unchanged. enc2 would have needed one more graph output.

## 2. Data recipe

**Speech.** The round-19 fit pools (`bench/round19_ja/cv_pool.py`):

* Common Voice 22.0 ja `train` clips (`cv_fit` + `cv_fit2`, CC0). These pools exclude every
  Lingolix-ja speaker and prompt and every cv-real test-half speaker.
* FLEURS ja train (CC BY 4.0).
* 3,518 clips in total; 3 Common Voice clips were missing on disk and skipped.
* Dev pool: the round-19 dev-pool natives (`devnat_in.jsonl`), that is Common Voice ja `dev`
  speakers and FLEURS validation.
* No clip or speaker of any held-out set is in either pool.

**Examples** (`bench/round20_ja_length/extract.py`). One example per vowel the length channel
judges. Its window is the ±200 ms of enc1 and enc5 activations around the vowel's CTC peak
(float16, cached; whole-utterance activations are never stored).

| version | what changes | fit pool | dev pool |
|---|---|---|---|
| natural, clean | nothing; label = the text's | every long vowel, half of the short ones | all |
| WSOLA, clean | up to 3 judged vowels per clip, ≥ 450 ms apart | targets only | targets only |
| WSOLA + noise | as above, plus DEMAND noise | 5–20 dB, uniform | 20, 10 and 5 dB, one version each |
| natural + noise | noise only | every second clip, 5–20 dB | – |

**How the WSOLA versions are made:**

* The vowel's stable-spectrum span on the 10 ms filterbank (the round-19 `v_stable` run) is
  stretched or compressed in numpy (`wsola.py`). Short-vowel factors are ×1.3–2.6; long-vowel
  factors are ×0.4–0.8.
* 20 % of the targets get a near-identity control factor (×0.9–1.1) that keeps the natural label.
  This keeps WSOLA artefacts present in both classes.
* The rest of the clip is untouched, and the clip is re-scored with its ORIGINAL text, so audio
  and text disagree at exactly those vowels.
* Soft label "audio is long":
  * a short source becomes long around ×1.5;
  * a long source becomes short around ×0.67;
  * the label is a logistic in log factor with width 0.1.

**Noise.** DEMAND (Thiemann, Ito & Vincent 2013), Zenodo record 1227121, **CC BY 4.0** (licence
as listed on the record). It was downloaded without an account.

* Format: 16 kHz zips, channel 1 only (8 × 5 min, 77 MB kept; the zips were deleted).
* Training environments: DKITCHEN, DLIVING, OOFFICE, PSTATION, TBUS, NPARK.
* Held back for the dev WSOLA pairs: PCAFETER, STRAFFIC.
* SNR is measured against the speech-active frames.

**Size.** The fit pool has 80,285 examples, of which 15,649 are WSOLA targets away from the
control factors. The dev pool has 30,495, of which 8,115 are WSOLA targets. The cache, windows plus
dev runs, totals 3.9 GB.

## 3. Model and cross-validation

**Head** (`bench/round20_ja_length/train.py`, config `e1_w8`).

* Architecture:
  * LayerNorm and a linear map 256 → 48 per frame;
  * a learned position embedding;
  * two 1-D convolutions (kernel 5, GELU, residual);
  * masked attention pooling;
  * the log local mora duration (the utterance outside the word, as in `length_r19.rate_loc`);
  * one logit for "the vowel as said is long".
* Size: 36.9 k parameters, 148 kB ONNX (opset 17). The runtime uses onnxruntime only.
* Training: binary cross-entropy against the soft labels, with long and short balanced. AdamW
  with one-cycle learning rate, 12 epochs.

**CV.** Four folds grouped by speaker on Common Voice and by sentence on FLEURS. FLEURS has no
speaker ids, so sentence grouping is the closest available and FLEURS speaker leakage across
folds cannot be ruled out. Numbers are out-of-fold:

| config | natural AUC | natural + noise AUC | WSOLA acc. 5–9 dB | 10–14 dB | 15–20 dB | clean | long→short targets | short→long targets |
|---|---|---|---|---|---|---|---|---|
| **e1_w8** (enc1, ±160 ms, rate input) | 0.9738 | 0.9688 | 0.882 | 0.910 | 0.916 | 0.933 | 0.745 | 0.940 |
| e1_w5 (±100 ms) | 0.9713 | 0.9655 | 0.861 | 0.909 | 0.914 | 0.921 | 0.759 | 0.928 |
| e1_w10 (±200 ms) | 0.9740 | 0.9696 | 0.891 | 0.913 | 0.926 | 0.932 | 0.737 | 0.945 |
| e1_w8, no rate input | 0.9735 | 0.9685 | 0.881 | 0.909 | 0.918 | 0.933 | 0.754 | 0.939 |
| e1_w8, window warped to the mora rate | 0.9729 | 0.9657 | 0.877 | 0.906 | 0.923 | 0.922 | 0.741 | 0.935 |
| e15_w8 (enc1 + enc5) | 0.9750 | 0.9691 | 0.875 | 0.916 | 0.929 | 0.932 | 0.761 | 0.941 |
| e5_w8 (enc5) | 0.9734 | 0.9681 | 0.840 | 0.869 | 0.866 | 0.872 | 0.600 | 0.899 |

Notes on the sweep:

* "WSOLA acc." counts targets with an unambiguous soft label (< 0.1 or > 0.9).
* "long→short" means a long vowel compressed, with the text still long. These are the hardest
  pairs.
* **Speaking rate does not matter to the head.** Rate as an input, no rate, and a window warped to
  the local mora duration are within 0.001 AUC. The activations already encode duration relative
  to their context.
* Width ±160–200 ms beats ±100 ms. enc1 beats enc5, especially on noise and compressions.
* e1_w8 was kept: it is half the input of e15_w8 for −0.001 AUC.

**Dev pool** (fitted head; held-back speakers and noise environments):

| | head | shipped round-19 length model |
|---|---|---|
| natural vowels, long vs short AUC | **0.965** | 0.948 |
| … Common Voice dev part | **0.941** | 0.900 |
| … FLEURS validation part | **0.980** | 0.975 |
| WSOLA mismatch against natural natives, AUC, clean | **0.975** | 0.819 |
| … 20 dB | **0.972** | 0.814 |
| … 10 dB | **0.966** | 0.802 |
| … 5 dB | **0.959** | 0.787 |
| WSOLA target accuracy, clean / 20 / 10 / 5 dB | 0.937 / 0.942 / 0.890 / 0.856 | – |

The mismatch rows use the head's evidence against the text's length, and for the shipped model,
1 − P(the sound matches its text).

## 4. Integration (lower-only) and dev results

**Integration.** `pronunciada/ja_length20.py` is called right after the length channel in
`scorer.py`. The edit to `scorer.py` is 7 lines; `align.py` and `ja_round19.py` are untouched.

* For each judged vowel, zh is the head's log-odds that the vowel matches its text.
* The vowel's length log-odds become z′ = z + min(0, a·(zh − k)).
* k is a native quantile of out-of-fold zh on fit-pool natives, set separately for expected-long
  and expected-short vowels.
* It feeds `PhoneResult.length_prob` and through it the word step (z_len, z_long) and the error
  flag (`miscue_items`, length_prob ≤ 0.03). It can never raise a probability.

**How the grid was run.** `bench/round20_ja_length/devrun.py` scores every dev clip once with the
switch off. Then, for each setting, it lowers the cached length probabilities and re-assembles the
result through `Scorer._assemble`, the code path of `rescore`. Word probabilities, the step and
the flag are therefore exactly the runtime's. Main reproduces the round-19 final dev dumps with
max |diff| = 0.

**Families tried:**

| grid | settings | what varies |
|---|---|---|
| g1 | 12 | a ∈ {0.5, 1, 2} × q ∈ {0.5, 1, 2, 3} %, both classes |
| g2 | 14 | expected-short only, a few on expected-long at q = 0.25 %, caps on the drop (3 log-odds) |
| g3 | 14 | as g2, with position gates: skip the word's last sound; also skip a vowel next to a vowel |
| g4 | 15 | a "min" combination, z′ = min(z, a·(zh − k)): the head's verdict replaces a higher one only |
| g5 | 15 | only sounds the round-19 channel finds right (p ≥ 0.5), so the head only adds new detections |

All 70 are in `results/tables/round20_ja_length_dev.json`.

**Main against a selection of settings.** Dev set: 744 rows, 428 planted. Guards are the round-19
dev guards. Paired differences use a speaker bootstrap with 1,000 resamples.

| | main | g1 a=0.5 q=0.5 | **g1 a=1 q=1** | g1 a=1 q=2 | g5 a=1 S1 (p ≥ 0.5 only) |
|---|---|---|---|---|---|
| dev AUC | 0.8674 | 0.8686 | **0.8707** | 0.8717 | 0.8686 |
| paired ΔAUC | | +0.0012 [+0.0004, +0.0021], p = 0.002 | **+0.0033 [+0.0011, +0.0059], p = 0.002** | +0.0043 [+0.0008, +0.0083], p = 0.01 | – |
| FA@80 | 17.4 % | 16.4 % | **16.0 %** (Δ −1.4 [−2.9, +0.4], p = 0.53) | 15.9 % (p = 0.43) | 17.4 % |
| CV part AUC | 0.8337 | 0.8352 | 0.8378 | 0.8387 | – |
| length bucket | 0.839 | 0.844 | 0.855 | 0.862 | 0.844 |
| length, text long / audio short | 0.912 | 0.914 | 0.918 | 0.921 | 0.912 |
| length, text short / audio long | 0.749 | 0.758 | 0.777 | 0.789 | 0.760 |
| own flag: planted length units | 71 / 114 | 76 | **79** | 85 | 71 |
| own flag: control units flagged | 19.9 % | 20.1 % | 20.3 % | 20.8 % | 19.9 % |
| **guard: FLEURS-val natives < 0.6** (7,219 words) | 2.69 % | **2.72 % ✗** | **2.87 % ✗** | **2.96 % ✗** | **2.72 % ✗** |
| guard: FLEURS-val natives < 0.5 | 2.34 % | 2.37 % | 2.51 % | 2.56 % | 2.37 % |
| **guard: CV-dev natives < 0.6** (8,320 words) | 8.46 % | **8.49 % ✗** | **8.56 % ✗** | **8.67 % ✗** | 8.46 % |
| guard: CV-dev natives < 0.5 | 7.87 % | 7.87 % | **7.98 % ✗** | **8.10 % ✗** | **7.88 % ✗** |
| guard: FLEURS-swap dev far / near AUC | 0.9992 / 0.9786 | 0.9992 / 0.9785 | 0.9992 / 0.9783 | 0.9992 / 0.9784 | 0.9992 / 0.9788 |
| guard: cv-real dev half, word mean / min | 0.774 / 0.764 | 0.774 / 0.763 | 0.774 / 0.761 | 0.772 / 0.761 | 0.774 / 0.764 |

**Length table on FLEURS validation.** These are the planted long / short swaps of
`bench/length/heldout_runtime.py` with the same generators and seeds; the share is words caught at
5 % native word false alarms.

| | round 18 | main | g1 a=0.5 q=0.5 | **g1 a=1 q=1** | g1 a=1 q=2 | g2 short only, a=1 q=5 % |
|---|---|---|---|---|---|---|
| vowel_length A>B (audio long, text short) | 62.6 % | 50.8 % | 56.7 % | **65.6 %** | 68.8 % | 75.3 % |
| vowel_length B>A (audio short, text long) | 80.8 % | 88.9 % | 88.5 % | 89.8 % | 90.2 % | 87.6 % |
| gemination A>B / B>A | 5.7 / 48.9 % | 44.7 / 78.6 % | 44.7 / 78.6 % | 44.3 / 78.2 % | 44.3 / 78.2 % | 43.1 / 77.4 % |
| all | 56.2 % | 69.4 % | 70.9 % | 73.6 % | 74.6 % | 75.2 % |
| native word acceptance (≥ 0.5) | 97.89 % | 97.66 % | 97.63 % | 97.49 % | 97.44 % | 97.35 % |

The round-18 column was scored with `PRONUNCIADA_JA_ROUND19=0` on the same clips.

**About the direction names.** In `bench/audit/contrasts.py`, "A>B" means the audio has member A
and the edited text demands member B. For `vowel_length`, A = long, so **"vowel_length A>B" is
"said long, text short"**.

* The round-19 write-up and this brief describe the row that fell from 60.1 % to 43.8 % as "long
  vowel in the text, said short", but by the code it is the opposite direction.
* The head fixes exactly this row on validation: 50.8 % → 65.6–75 %, back above round 18's
  62.6 %. It also lifts the dev "text short / audio long" units most (0.749 → 0.78–0.81).

**Where the native cost comes from.**

* I listed the native dev words that crossed 0.6 under g1 a=1 q=1. Almost all are words whose
  vowel the round-19 channel already scored low (length_prob 0.02–0.2), for example 場所 /o/,
  思う /o/, 助手 /o/, オイル /o/ and 追いかける /o/.
* Many of these are word-final, or are vowels before another vowel.
* The head agrees with the old channel there: the vowel really is long. These are natives
  lengthening a short vowel in context, not detection errors.
* The step's length term is linear in the length logit without a floor, so pushing an already-low
  sound lower drags the word below 0.6.

**Every remedy that removed this cost also removed the gain:**

* **Gate to sounds the old channel finds right (g5).** It spares those natives. It also loses the
  whole length-table gain (A>B 51.2 %). The words the round-19 channel misses already have
  length_prob < 0.5; they are simply not low enough.
* **Skip word-final or vowel-adjacent sounds (g3).** This halves the native cost and halves the
  gain. Natives below 0.6 still rise.
* **Caps (g2/g3) and the "min" combination (g4).** Both trade the same way, and none reaches
  zero native cost.

## 5. What I think the limit is

**Duration itself is now well measured, through noise.**

* Sound level, natural: 0.965 vs 0.948.
* WSOLA mismatch: 0.96–0.975 vs 0.79–0.82.

**The remaining Japanese length errors are not a measurement problem. They are a decision
problem: a native's natural length variation looks like a learner's error.** In read Common Voice
and FLEURS speech, natives:

* shorten long vowels (about 5 % of text-long native vowels have zh < −2);
* lengthen short vowels at word ends and before vowels.

At the word level these natives are indistinguishable from planted swaps by duration alone.

**What could move it:**

* **A context model of native duration.** For example, the expected duration of this vowel in this
  word position and accent phrase, learned from natives, would let the lowering use "longer than
  natives make it here" rather than "long".
* **Reworking the word step's length term.** It is linear and unbounded, so any extra lowering of
  an already-low sound costs native words.
* **A different native guard.** The round-19 guard allows zero rise. The round-level rule in
  RULES.md allows ≤ 0.3 points on FLEURS validation. Under the RULES.md rule, g1 a=1 q=1 (+0.18
  points FLEURS validation, +0.10 points CV dev) would qualify; under the brief's guard it does
  not. Which guard applies is the architect's call, not mine.

## 6. Architect's option (not frozen, not held-out tested)

Packaged, off by default. `PRONUNCIADA_JA_LENGTH20=1` gives g1 a=1 q=1:

* enc1 head;
* a = 1;
* k_long = −5.65 and k_short = −2.72, the 1 % native quantiles;
* no gate.

On dev, against main:

* AUC +0.0033 (p = 0.002); FA@80 −1.4 points (n.s.);
* length bucket 0.839 → 0.855;
* own flag catches 79 instead of 71 of 114 planted length units; control units flagged
  19.9 → 20.3 %;
* FLEURS-validation length table: A>B words 50.8 → 65.6 %, all 69.4 → 73.6 %;
* natives below 0.6 rise **+0.18 points (FLEURS validation) and +0.10 points (CV dev)**.

It was **not** run on any held-out set. If it is wanted, the frozen candidate and these dev
numbers go to STATUS first, and then one look on each held-out row.

## 7. Size, latency, switches, parity

**Size:** +149 kB (`ja_len20.onnx` 148 kB, `ja_len20.json` 1 kB). The acoustic model and its
outputs are unchanged, because enc1 was already exported for Japanese.

**Latency with the switch on** (`bench/round19_merge/latency.py`, same tree, paired, interleaved):

* FLEURS validation, 100 clips, median 12.4 s, 4 threads: 161.4 → 162.5 ms. The paired median is
  +1.24 ms [1.11, 1.39].
* On 75 Common Voice dev clips (median 4.2 s), the head stage alone takes median 0.68 ms (p90
  1.49 ms) out of 57.8 ms in total.
* With the switch at its default (off), the added cost is one environment lookup.

**Equivalence and parity** (`bench/round20_ja_length/parity.py`, 60 dev clips, every word
probability, word flag and sound length probability):

* fp32: switch unset vs devrun main (= round-19 final dumps): max |diff| 0. Unset vs `=0`: 0.
* fp32, `=1` against the devrun record of the same setting: 0; 13 of the 60 clips change.
* int8: unset vs `=0`: max |diff| 0.

**Tests.** `tests/test_ja_length20.py`: lower-only, the switch, and the packaged head on a toy
input.

## Files

* **Package:**
  * `pronunciada/ja_length20.py`;
  * `pronunciada/data/length/ja_len20.{onnx,json}`;
  * `pronunciada/scorer.py`, a 7-line hook after the length channel.
* **Bench** (`bench/round20_ja_length/`):
  * `wsola.py`, `common.py` (noise, soft labels);
  * `extract.py`, `run_extract.sh`;
  * `probe.py`;
  * `train.py` (cv / fit / dev / knees), `run_sweep.sh`;
  * `devrun.py`, `run_devrun.sh`, `run_devrun_seq.sh`;
  * `deveval.py`, `parity.py`, `tables.py`.
* **Tables** (`results/tables/round20_ja_length_*.json`):
  * `probe`: frame-rate steps and the linear probe;
  * `cv`: CV sweep, dev-pool head numbers, native knees;
  * `dev`: all 70 settings with guards, and the FLEURS-validation length table;
  * `extra`: latency and the round-18 baselines.
* **Cache:** `$PRONUNCIADA_CACHE/round20/jalength/`: windows, heads, devrun dumps, DEMAND channel-1
  files.

## Held-out runs, in order

None.
