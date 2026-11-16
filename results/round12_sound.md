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
* **Prototype features are the most transferable new signal**, and are licence-clean (FLEURS
  train): they gave the largest gain both in d(ii) and in d(i).

Reproduce:

    python -m bench.calib.sound_emb protos --lang all --n 500
    python -m bench.calib.sound2 run --lang all --workers 2
    python -m bench.calib.sound_emb so762
    python -m bench.calib.sound2_eval diagnose
    python -m bench.calib.sound2_eval candidates
    python -m bench.calib.sound2_eval so762
