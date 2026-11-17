# Round 18: contrast evidence reaches the word score; Korean lax / tense

Selection used FLEURS **validation**, speechocean762 **train** and OMPAL **dev** only. Held-out
data was run once, after the parameters were frozen; Azure answers are cached (no requests).

Method of comparison: the word factor is a deterministic function of (word probability, the
word's lowest contrast log-odds, strictness). So "before" and "after" come from the same scoring
run with the factor off: per word the probability and the lowest contrast log-odds are stored
(`bench/round18/collect.py`), and "after" is computed offline (`bench/round18/fit.py`), exactly
paired.

## A. Word-level contrast factor (`pronunciada/contrast_word.py`)

**Input.** Per sound, `contrast_z` (Scorer cache) is the lowest of:
* the round-16 cap log-odds where a cap fires;
* the round-17 / 18 head evidence (aspiration, palatalisation, tense).

**Factor.** f = 1 while the word's lowest z is above a knee, then falls linearly in log-odds to
`floor` over 4 units. This is the length channel's form. The knee at strictness 0.5 is the
q-quantile of native words' z; strictness 0 and 1 use q / 2 and 2q. Word probability is
multiplied by f; the sentence probability is not affected.

**Rule** (fixed in advance):
* settings within all dev guards: native word acceptance −0.5 pt, near- and far-swap AUC and
  planted word-error AUC not lower (tolerance 1e-4), learners −0.005;
* the strictest such setting (largest q, then lowest floor) is taken;
* adopted when the pooled contrast-swap word hit@5 % gains ≥ 15 pt.

The first version took the strictest setting by the native guard alone and then checked the
other guards. Every language then failed at least one of those guards by a small margin (for example zh
far-swap 0.9909 → 0.9905), so the guards were moved into the selection before any held-out run.

**Dev** (FLEURS validation; "near misses" = dev near-swap words missed at 5 % false alarms /
those containing a contrast-judged sound):

| lang | contrast word hit@5 % | near-swap AUC | far-swap AUC | planted word AUC | native words | setting | near misses / with contrast sound | decision |
|---|---|---|---|---|---|---|---|---|
| zh | 42 % → 76 % | 0.9629 → 0.9643 | 0.9909 → 0.9910 | 0.9392 → 0.9396 | 97.83 → 97.73 % | q 0.02, floor 0.6 | 47 / 32 | **adopted** |
| en | 80 % → 82 % | 0.9789 → 0.9789 | 0.9969 → 0.9969 | 0.9460 → 0.9460 | 97.07 → 97.07 % | q 0.05, floor 0.4 | 34 / 0 | not adopted: contrast swaps already caught at word level (80 %); +2 pt |
| ru | 29 % → 29 % | 0.9221 → 0.9220 | 0.9970 → 0.9970 | 0.9305 → 0.9304 | 97.56 → 97.56 % | q 0.002, floor 0.8 | 111 / 103 | not adopted: the planted word-error AUC guard binds at every useful setting |
| ja | 36 % → 57 % | 0.9666 → 0.9666 | 0.9976 → 0.9976 | 0.9391 → 0.9392 | 98.18 → 98.16 % | q 0.02, floor 0.4 | 29 / 2 | **adopted** |
| de | 21 % → 33 % | 0.9522 → 0.9523 | 0.9964 → 0.9966 | 0.9503 → 0.9507 | 97.42 → 97.37 % | q 0.05, floor 0.6 | 67 / 33 | not adopted: +12 pt < 15 |
| ko | 11 % → 22 % | 0.9276 → 0.9342 | 0.9974 → 0.9975 | 0.9350 → 0.9349 | 98.42 → 98.42 % | q 0.01, floor 0.8 | 31 / 28 | not adopted: the planted word-error AUC guard binds; +11 pt at the strictest passing setting |

Per contrast (dev, word hit@5 %, before → after at the chosen setting):

| lang | contrasts |
|---|---|
| zh | aspiration/A>B 40 % → 72 %; aspiration/B>A 43 % → 80 % |
| ja | voicing/A>B 26 % → 69 %; voicing/B>A 47 % → 46 % |
| ru | final_devoicing/A>B 12 % → 12 %; hard_soft/A>B 10 % → 10 %; hard_soft/B>A 29 % → 29 %; voicing/A>B 45 % → 45 %; voicing/B>A 57 % → 57 % |
| de | final_devoicing/A>B 6 % → 8 %; voicing/A>B 32 % → 62 %; voicing/B>A 28 % → 37 % |
| ko | aspiration/A>B 5 % → 6 %; aspiration/B>A 30 % → 49 %; tense/A>B 4 % → 7 %; tense/B>A 4 % → 28 % |
| en | dh_z/A>B 63 % → 63 %; dh_z/B>A 88 % → 94 %; th_s/A>B 75 % → 75 %; th_s/B>A 92 % → 95 % |

Learners (dev):
* OMPAL dev (zh), word AUC before → after:
  * any error 0.8095 → 0.8089;
  * tone-only 0.8096 → 0.8049;
  * consonant / vowel 0.8089 → 0.8245.
* speechocean762 train (en, at its strictest passing setting): word P Pearson
  0.7772 → 0.7738, AUC for expert word accuracy < 6.5
  0.9597 → 0.9592. Not adopted anyway (gain +2 pt).
* cv-real: its primary score is the sentence probability, which the factor does not touch (dev
  and test identical by construction).

### Held-out, once (adopted: zh, ja)

**zh** (adopted; kept):
* audit contrast swaps at word level (FLEURS test, hit@5 %): pooled 31 % → **62 %** (aspiration/A>B 35 % → 67 %; aspiration/B>A 27 % → 57 %); native words 97.43 → 97.33 %
* swap benchmark (Azure in brackets): near-swap AUC 0.9741 → **0.9741** (0.9663), paired Δ CI [-0.0003, +0.0005]; hit@5 % 0.893 → 0.893 (0.896); right word 0.787 → 0.787 (0.747); far-swap AUC 0.9918 → 0.9919 (0.9887); native words < 0.6 0.0365 → 0.0376 (0.0365)
* ten-language benchmark: native words 0.9739 → 0.9727; word AUC 0.9394 → 0.9401; word hit@5 % 0.7987 → 0.8037; lowest word 0.8160 → 0.8150; sound rows unchanged

**ja** (adopted; kept):
* audit contrast swaps at word level (FLEURS test, hit@5 %): pooled 40 % → **60 %** (voicing/A>B 32 % → 73 %; voicing/B>A 48 % → 47 %); native words 97.62 → 97.58 %
* swap benchmark (Azure in brackets): near-swap AUC 0.9681 → **0.9689** (0.9194), paired Δ CI [-0.0011, +0.0043]; hit@5 % 0.833 → 0.820 (0.680); right word 0.667 → 0.660 (0.539); far-swap AUC 0.9909 → 0.9909 (0.9791); native words < 0.6 0.0280 → 0.0290 (0.0204)
* ten-language benchmark: native words 0.9751 → 0.9739; word AUC 0.9323 → 0.9320; word hit@5 % 0.7420 → 0.7425; lowest word 0.7475 → 0.7470; sound rows unchanged

**OMPAL test** (zh), word AUC before → after:
* any error 0.7946 → 0.7938;
* tone-only 0.7852 → 0.7817;
* consonant / vowel 0.8394 → 0.8513.

**Unchanged by construction:**
* cv-real (sentence probability);
* `bench.eval_english` (no English change; its word rows read the 0–10 regression);
* every language without an adopted factor.

**Revert rule** (near-swap or cv-real drop with p < 0.10 and no contrast word gain): not
triggered. The zh and ja near-swap AUCs rose; ja near-swap hit@5 % fell one record
(0.833 → 0.820; p 0.96 that "after" is not better, i.e. a slight drop).

Byte-identical check: on 50 clips, 5 per language, `PYTHONHASHSEED=0`, against the previous
commit:
* en, es, fr, de, it, pt, ru and ja are identical (no ja clip had a voicing-judged sound below
  the knee);
* zh word probabilities differ in 2 clips;
* ko sound probabilities differ (B).

## B. Korean lax / tense (`RefPhone.tense`, `contrast_heads.py`, `data/coach/ko_tense.npz`)

**Reference.**
* The Korean front-end now marks each obstruent of the ㄱ/ㄲ ㄷ/ㄸ ㅂ/ㅃ ㅅ/ㅆ ㅈ/ㅉ series
  `tense = True / False`. The marker comes from the spelling **and from the front-end's own
  post-obstruent tensification** (학교 → 학꾜, 먹다 → 먹따), which the front-end already models.
* The symbol stays the same plain one (ZIPA has no tense token), so the acoustic scoring is
  unchanged: with the evidence off, every output is identical.

**Head.** A logistic regression on the pooled activations (own sound + neighbours), FLEURS train
natives (4,117 sounds, 531 tense). On FLEURS
validation natives it separates tense from lax with AUC **0.988**.

**Evidence and cap.** Exactly as round 17: z = slope (logit P(expected) − s0), and the shown
sound log-odds become min(·, z). Swaps flip the reference flag at one sound (120 per direction).
Since nothing else changes, the edited sound's probability is the native one capped by the
flipped evidence.

| | dev before → after | test before → after [Δ CI] |
|---|---|---|
| said tense, expected lax | 9 % → **85 %** | 7 % → **78 %** [+65, +77] |
| said lax, expected tense | 8 % → **99 %** | 5 % → **99 %** [+90, +97] |
| native sounds | 99.75 → 99.71 % | 99.33 → 99.26 % |
| planted sound AUC | 0.9787 → 0.9765 | 0.9715 → 0.9690 |

Chosen: q 0.002, slope 0.5. Both dev directions are ≥ 60 %, so it
is adopted.

**Ten-language benchmark, ko** (FLEURS test, against round 17's shipped state):
* native sounds 0.9940 → 0.9933;
* sound AUC 0.9751 → 0.9737;
* sound hit@5 % 0.923 → 0.915;
* words identical, because the Korean word factor was not adopted.

**Coaching.** A tense item below the bar gets the claim `feature: "tense"`, with tips
`feature.tense.add` ("Make /k/ … tighter and stronger, with no puff of air.") and
`feature.tense.remove` ("Relax /k/ …; it sounded tense."). Bars allow ≤ 0.05 / 0.1 / 0.2 native
items per clip. Unverified on learners.

**The tense evidence does reach the Korean word factor** (dev contrast rows above: lax for tense
4 → 28 % at the passing setting). The factor is not adopted for Korean, because the planted
word-error guard binds.

## What did not work

* **Word factor for en, ru, de, ko.**
  * en: its contrasts were already caught at word level.
  * ru, ko: the planted word-error AUC guard binds at every setting that helps; the factor also
    penalizes native words with a low contrast z, enough to cost planted-error AUC.
  * de: +12 pt, under the 15-pt bar.
* **Contrast hit, the opposite direction.** ja "voiceless for voiced" and ru "soft for hard" do
  not gain at word level.
* **ja near-swap hit@5 %**: one record fewer on test.
* **Coverage of the Korean tense marker.** It comes from spelling plus the front-end's
  tensification rule; other phonological tensification (for example after a ㄹ coda in Sino-Korean
  words, or after the adnominal -ㄹ) is not modeled.
