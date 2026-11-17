# Round 12: where single-sound evidence is lost, and four candidate fixes (dev only)

**Outcome.** No candidate meets the dev adoption bar, so nothing was adopted. The bar was: mean
near-swap AUC over the ten languages up ≥ 0.01, replaced words > 0.9 down by a quarter, all
guards met. Defaults are unchanged, and `pronunciada/scorer.py` was not edited. The candidates
exist only as bench code (`bench/calib/sound2.py`, `sound2_eval.py`, `sound_emb.py`).

One clear positive, outside the swap bar: on speechocean762 TRAIN cross-validation, candidate d(i)
lifts the English phone-level model:

| speechocean762 train CV | today | with d(i) |
|---|---|---|
| phone Pearson | 0.684 | 0.715 |
| precision at recall 0.20 / 0.33 / 0.47 | 0.82 / 0.71 / 0.61 | 0.86 / 0.76 / 0.65 |

d(i) adds per-sound ZIPA layer activations and native-prototype features. It is the strongest
lead for coaching, but it is not staged at runtime yet.

All data is FLEURS validation (dev) and speechocean762 train. No held-out set was touched. The work
ran under the post-crash limits: 2 worker processes, at most 2 threads, one language in memory at
a time.

## Data

`python -m bench.calib.sound2 run --lang all --workers 2` covers 3 563 validation clips:
* each clip on its correct prompt (the natives);
* the controlled-error references of `bench.eval_languages` (seeds "features-v1");
* a near-/far-swap dev set per language, built from validation sentences by the fleurs-swap recipe
  (`bench/calib/bias_grid.py build-dev`; 155–393 near items per language).

Stored per sound: the raw evidence, the inputs of the word calibrator, the universal sound logits
(with the language bias) and the shown logits.

The word calibrator is re-run offline (`features.word_table` + the shipped `word.npz`), so a
candidate changes word probabilities exactly as it would at runtime. With no candidate applied,
this reproduces the shipped word logits to 2e-7.

The baseline is the current tree. For Russian it gives a dev near AUC of 0.916; the round-10
record was 0.910, and the tree has changed since (stress-model work by another agent).

## 1. Diagnosis

A miss is a replaced word that still scores > 0.8 (the near swaps), or a planted near substitution
whose word stays > 0.8. Each miss is classified by the replaced sound's evidence and what happened
to it:
* **discounted**: the replaced sound has llr_sub ≤ −2 against it (the expected sound is at least
  7× less likely than the best alternative), but its calibrated sound P is ≥ 0.5;
* **lost in word model**: llr_sub ≤ −2 and sound P < 0.5, but the word stays > 0.8;
* **weak evidence**: −2 < llr_sub ≤ −0.5;
* **none**: llr_sub > −0.5.

**Near swaps:**

| lang | items | misses | discounted | lost in word model | weak | none | final sound: misses vs all | vowel: misses vs all | median llr / sound P of misses | natives with llr ≤ −2 |
|---|---|---|---|---|---|---|---|---|---|---|
| en | 388 | 28 (7 %) | 23 | 0 | 5 | 0 | 36 % vs 29 % | 46 % vs 30 % | −5.2 / 0.91 | 6.8 % |
| zh | 290 | 38 (13 %) | 30 | 0 | 7 | 1 | 13 % vs 15 % | 21 % vs 30 % | −4.3 / 0.96 | 5.2 % |
| es | 393 | 29 (7 %) | 28 | 0 | 1 | 0 | 69 % vs 32 % | 72 % vs 49 % | −5.4 / 0.85 | 2.7 % |
| it | 337 | 72 (21 %) | 65 | 0 | 6 | 1 | 90 % vs 62 % | 92 % vs 68 % | −4.9 / 0.90 | 4.0 % |
| ja | 173 | 32 (18 %) | 26 | 0 | 2 | 4 | 31 % vs 17 % | 66 % vs 36 % | −4.3 / 0.94 | 10.6 % |
| ko | 155 | 41 (26 %) | 24 | 0 | 13 | 4 | 34 % vs 31 % | 63 % vs 46 % | −3.2 / 0.98 | 8.2 % |
| fr | 287 | 32 (11 %) | 19 | 0 | 9 | 4 | 75 % vs 39 % | 78 % vs 42 % | −2.8 / 0.91 | 5.4 % |
| de | 320 | 59 (18 %) | 49 | 0 | 8 | 2 | 37 % vs 27 % | 44 % vs 38 % | −5.6 / 0.96 | 13.4 % |
| pt | 366 | 66 (18 %) | 59 | 0 | 6 | 1 | 59 % vs 23 % | 74 % vs 43 % | −5.0 / 0.91 | 7.5 % |
| ru | 344 | 84 (24 %) | 52 | 1 | 29 | 2 | 57 % vs 52 % | 79 % vs 46 % | −2.8 / 0.93 | 10.3 % |
| **all** | 3 053 | 481 | **375 (78 %)** | **1** | 86 (18 %) | 19 (4 %) | | | | |

**Planted near substitutions** (`bench/corrupt.py` phone_sub_near):

| | value |
|---|---|
| misses | 1 354 of 6 948 (19 %) |
| discounted by the sound calibrator | 1 240 (92 %) |
| lost in the word model | 9 |
| weak evidence | 84 |
| no acoustic evidence | 21 |

Per language, misses run from 10 % (fr) to 35 % (ja). The full table is in
`results/tables/round12_diagnosis.json`.

**Reading the diagnosis:**
* **The coordinator's hypothesis about where the evidence goes is right.** Almost every miss has
  strong acoustic evidence (median llr −3 to −6), and the sound calibrator maps it to P ≈ 0.9.
  The word model almost never loses evidence once the sound is flagged.
* **The misses lean the way described.** They are word-final (it 90 % vs 62 %, es 69 % vs 32 %,
  fr 75 % vs 39 %, pt 59 % vs 23 %) and vowels (es, it, ja, pt, ru, fr).
* **The discount is not simply wrong.** Natives' own correct sounds have llr ≤ −2 at 3–13 %, as
  often as or more often than the swapped sound's rank among them. The calibrator trades those
  native outliers against the errors. Everything below confirms it: an evidence-only second
  opinion cannot separate the two much better.

## 2. Candidates

Shared method:
* Every candidate keeps the shipped calibrators and only lowers sound log-odds:
  `z' = min(z_shipped, z_second)`.
* Each setting is one value for all languages, except c.
* The native guards are applied per language: native word and native sound acceptance at default
  strictness may drop ≤ 0.5 points.

The candidates:
* **a.** Logistic second opinion on the raw evidence (llr_sub, llr_del, z-scores, peak posterior,
  frames, position / nucleus / short-word flags and their interactions with llr_sub; monotone in
  llr_sub). It is fitted on natives' own sounds (acceptable) vs planted near / far substitutions,
  pooled over languages, **leave-one-language-out**. The offset puts a share q of the training
  natives below 0.
* **b.** The same model at word level: the word's probability is capped by the sigmoid of its
  lowest sound z.
* **c.** A position-specific bias: δ added to final / single-sound sounds only.
* **d(ii).** Candidate a plus three features from the ZIPA layers:
  * per expected sound, enc1 + enc3 + enc5 are mean-pooled over its frames ±1 frame (1024-d) and
    z-scored by language;
  * features are the cosine to the language's **native prototype** of the expected sound (best over
    its accepted alternatives), the best cosine to any other phone, and their margin;
  * prototypes come from 500 FLEURS **train** clips per language (`bench/calib/sound_emb.py`).
  * On its own, the margin separates planted near substitutions from native sounds with AUC 0.979
    (en), against 0.989 for llr_sub.

### Means over the ten languages, against today

| candidate | near AUC | replaced > 0.9 | worst native word drop | worst native sound drop | worst planted word AUC drop | worst stress AUC drop | passes guards |
|---|---|---|---|---|---|---|---|
| today | 0.9599 | 10.5 % | – | – | – | – | – |
| a, q = 0.002 | 0.9613 | 10.2 % | 1.11 pt | 0.53 pt | 0.0035 | 0.0061 | no |
| a, q = 0.005 | 0.9609 | 9.2 % | 2.03 | 1.15 | 0.0062 | 0.0118 | no |
| b, q = 0.002 | 0.9598 | 9.9 % | 1.28 | 0 | 0.0037 | 0.0099 | no |
| b, q = 0.01 | 0.9622 | 5.7 % | **6.99** | 0 | 0.0153 | 0.0618 | no |
| c, δ = −1 | 0.9600 | 10.1 % | 0.14 | 0.38 | 0.0047 | 0.0026 | yes |
| c, δ = −2 | 0.9581 | 10.0 % | 0.35 | 1.29 | 0.0126 | 0.0072 | no |
| **d(ii), q = 0.002** | **0.9625** | 10.2 % | 0.65 (de) | 0.36 | 0.0025 | 0.0036 | no (de only) |
| d(ii), q = 0.005 | 0.9619 | 9.0 % | 1.28 | 0.84 | 0.0055 | 0.0079 | no |

* **The best near-AUC gain is +0.0026** (d(ii), q = 0.002), against the +0.01 bar. Even then de
  native words drop 0.65 points.
* **Only b at q = 0.01 cuts replaced > 0.9 by a quarter.** It costs 7 points of native word
  acceptance.

### Per language, near AUC (q = 0.002, δ = −1)

| lang | today | a | b | c | d(ii) | > 0.9 today → d(ii) | native words / sounds, d(ii) |
|---|---|---|---|---|---|---|---|
| en | 0.979 | 0.981 | 0.979 | 0.978 | 0.982 | 3.1 → 3.4 % | −0.10 / −0.21 pt |
| zh | 0.963 | 0.962 | 0.962 | 0.961 | 0.964 | 7.6 → 7.6 % | −0.22 / −0.36 |
| es | 0.989 | 0.990 | 0.989 | 0.989 | 0.990 | 3.6 → 3.3 % | −0.10 / −0.09 |
| it | 0.964 | 0.967 | 0.966 | 0.967 | 0.969 | 12.2 → 12.5 % | −0.09 / −0.06 |
| ja | 0.966 | 0.966 | 0.965 | 0.963 | 0.966 | 13.9 → 12.7 % | −0.43 / −0.31 |
| ko | 0.928 | 0.929 | 0.928 | 0.928 | 0.930 | 18.7 → 20.0 % | −0.24 / −0.12 |
| fr | 0.976 | 0.979 | 0.976 | 0.976 | 0.980 | 9.4 → 7.3 % | −0.36 / −0.19 |
| de | 0.952 | 0.953 | 0.950 | 0.955 | 0.956 | 10.9 → 9.4 % | **−0.65** / −0.34 |
| pt | 0.966 | 0.968 | 0.968 | 0.967 | 0.969 | 9.6 → 7.9 % | −0.16 / −0.13 |
| ru | 0.916 | 0.919 | 0.916 | 0.916 | 0.919 | 16.0 → 17.8 % | −0.16 / −0.14 |

Every row is from `results/tables/round12_candidates.json`; models are in
`results/tables/round12_models.json`.

## 3. d(i): the speechocean762 sound heads with ZIPA-layer inputs (English)

Method:
* The shipped English feature set B is kept, and the same hyper-parameters and 5-fold
  speaker-grouped folds as `bench/calib/train_sound.py` are used.
* Added inputs: an **in-fold ridge projection** of the 1024-d pooled activations to the phone score
  (training rows get inner out-of-fold values), and the three prototype features.
* All 47 745 rows of the table were matched (1.000).

| speechocean762 train, CV | phone Pearson | AUC (expert < 2) | AUC str head | precision at recall 0.20 / 0.33 / 0.47 (str head) | same, reg head |
|---|---|---|---|---|---|
| B (shipped recipe) | 0.684 | 0.837 | 0.880 | 0.817 / 0.714 / 0.614 | 0.827 / 0.706 / 0.583 |
| B + prototypes | 0.702 | 0.844 | 0.885 | 0.845 / 0.738 / 0.640 | 0.849 / 0.738 / 0.602 |
| B + pooled activations | 0.698 | 0.841 | 0.883 | 0.825 / 0.735 / 0.627 | 0.846 / 0.719 / 0.592 |
| **B + both = d(i)** | **0.715** | **0.847** | **0.888** | **0.865 / 0.757 / 0.646** | 0.867 / 0.744 / 0.614 |

The coaching operating points measured on held-out are 0.75 / 0.65 / 0.55 for us and 0.56 / 0.46 /
0.39 for Azure. The train-CV values above are higher across the board, but the before → after gain
is the comparable quantity: about +0.04 / +0.04 / +0.03 precision.

Limits of this result:
* Not bootstrapped.
* It changes only English **shown** sound probabilities. The word model reads the universal model,
  so the near-swap numbers do not move.
* Native acceptance on FLEURS-en was **not** checked: that needs the full feature set for FLEURS
  clips, which this round's collection did not keep.
* Before it can ship, it needs that native check, a runtime path (the per-sound pooling is cheap;
  embeddings are already computed for the word score), and a held-out run.

## 4. Decision and what was not run

* **Not adopted: a, b, c, d(ii).** The best mean gain (+0.0026) is a quarter of the bar, and every
  setting that cuts the over-0.9 tail breaks native acceptance.
* **The diagnosis explains why.** The evidence is present but overlaps with natives' own variation,
  so any evidence-only second opinion buys detection only by flagging natives.
* **Real-learner guards not run.** No candidate passed the dev bar, so OMPAL dev and cv-real dev
  halves were not run.
* **d(i) is the lead for round 13**, as an English (sound_B) change: FLEURS-en native check, then
  OMPAL / cv-real dev guards, then held-out when allowed.
* **Prototype features are the most transferable new signal**, and are license-clean (FLEURS
  train): they gave the largest gain both in d(ii) and in d(i).

Reproduce:

    python -m bench.calib.sound_emb protos --lang all --n 500
    python -m bench.calib.sound2 run --lang all --workers 2
    python -m bench.calib.sound_emb so762
    python -m bench.calib.sound2_eval diagnose
    python -m bench.calib.sound2_eval candidates
    python -m bench.calib.sound2_eval so762

## 12b: candidate d(i) shipped for English (option A), and a Mandarin consonant / vowel model

### Dev (speechocean762 train CV, FLEURS validation, OMPAL dev)

**d(i) heads** (`bench/calib/train_sound_emb.py`): feature set B plus `emb_proj` (an in-fold ridge
projection of the pooled enc1/enc3/enc5 activations) and four prototype features. Same
hyper-parameters and the same 5-fold speaker-grouped folds as `train_sound.py`. Paired speaker
bootstrap, 1,000 resamples:

| speechocean762 train CV | sound_B | sound_E | Δ [95 % CI] |
|---|---|---|---|
| phone Pearson (0–2 head) | 0.684 | 0.714 | +0.030 [+0.023, +0.039] |
| AUC, expert score < 2 | 0.837 | 0.847 | +0.009 [+0.006, +0.013] |
| AUC, strict head | 0.880 | 0.889 | +0.009 [+0.006, +0.011] |
| precision at recall 0.20 / 0.33 / 0.47 (strict head) | 0.817 / 0.714 / 0.614 | 0.863 / 0.760 / 0.644 | +0.045 / +0.047 / +0.030 (all CIs > 0) |

**Option A vs B.** FLEURS-en validation natives, shown sound P ≥ 0.5 at default strictness:
* sound_B: 0.9933;
* sound_E: 0.9876, a drop of 0.57 points, against the 0.3-point limit for option B.

So **option A** was taken:
* sound_E gives the reported 0–2 phone score and the coaching inputs;
* `PhoneResult.prob` and every word / sentence input stay on sound_B / sound_U;
* the word / sentence calibrators were not retrained;
* config: `sound_detail_model: {en: sound_E}`.

**Runtime path.**
* `pronunciada/sound_emb.py` (numpy only) plus about 20 additive lines in `scorer.py`. The scorer
  already computes the activations for the round-6 word score.
* Training-vs-runtime parity on 200 train clips (3,891 sounds): max |ΔP| 0.096, 99th percentile
  0.0044, max |Δ 0–2| 0.155.
  * The shipped sound_B shows the same pattern on the same clips (max 0.083, p99 0.0008). That
    comes from the bench feature table vs the runtime table crossing tree splits.
  * The vectors themselves agree to 0.002 (float16 storage).
* Latency on a 3.5 s clip: 67.8 → 68.1 ms (median of 15).
* Files: `sound_E.npz` 317 KB, `sound_E_emb.npz` 95 KB.

**Coaching.** `bench/coach/fit_v2.py` reads the detail model's out-of-fold logits; the English
logistic is stored as `by_lang.en` and the other languages keep the global model. Train OOF
precision at recall 0.20 / 0.33 / 0.47 went from 0.811 / 0.711 / 0.613 to **0.850 / 0.756 / 0.648**.
* Native cost (FLEURS-en): flags per clip at balanced 3.0 → 4.2; named sounds per clip
  0.27 → 0.30, at the cap.
* `coach.py` edits: `error_prob(..., lang)`, the per-language logistic, and the zh hook.

**Mandarin** (`bench/calib/ompal_sound_emb.py`, OMPAL dev, 28 speakers, 5-fold speaker-grouped):

| CV AUC on OMPAL dev | today | + prototype margin | + activations | + both |
|---|---|---|---|---|
| initials (consonant wrong; 202 of 10,886) | 0.816 | 0.921 | 0.824 | **0.925** (+0.109 [+0.062, +0.168]) |
| finals (vowel wrong; 205 of 12,518) | 0.890 | 0.896 | 0.900 | **0.904** (+0.013 [+0.003, +0.025]) |

**Adopted**, since both parts clear the +0.01 bar.
* In runtime form (per-sound logistic, maximum over the unit) the OOF AUC is 0.931 / 0.915.
* Thresholds were fitted for OOF precision 0.8 / 0.7 / 0.6.
* Native acceptance (probabilities) is unchanged; FLEURS-zh native coaching flags per clip at
  balanced rose 0.02 → 0.14.
* Size: `data/coach/zh_sound.npz` 113 KB. Runtime parity with the training arrays: max 3e-5.

### Held-out, run once

English round-11 caches were moved to `calib/api_round11_en/`. Azure answers are cached; no
requests were made.

| speechocean762 test | round 11 | round 12b | Azure |
|---|---|---|---|
| phone Pearson (0–2 score) | 0.660 [0.581, 0.712] | **0.685 [0.613, 0.731]** | 0.538 |
| word, sentence rows; L2-ARCTIC / EpaDB detection | – | unchanged (they read sound_B / U) | – |
| coaching precision at recall 0.20 / 0.33 / 0.47 | 0.743 / 0.643 / 0.551 | **0.799 / 0.684 / 0.577** (paired Δ CIs all > 0) | 0.559 / 0.464 / 0.384 |
| coaching PR-AUC | 0.546 | **0.570** | 0.382 |
| balanced: precision / recall; named right within one feature | 0.65 / 0.33; 0.77 | 0.66 / 0.36; 0.79 | 0.44 at our recall |

| OMPAL test (balanced) | round 11 | round 12b |
|---|---|---|
| consonant precision / recall | 0.27 / 0.06 | **0.71 / 0.25** |
| vowel precision / recall | not flagged | **0.67 / 0.18** |

* The en rows of the swap benchmark were not re-run: under option A word probabilities do not
  change.
* OMPAL word / sentence AUCs were not re-run: the Mandarin change touches coaching only.
* One wiring bug was fixed during the held-out run: OMPAL units were first scored with the general
  coaching score. It was re-run with the Mandarin score; no threshold changed.
