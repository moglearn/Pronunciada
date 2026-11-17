# Round 16: the hidden contrasts of the contrast audit

This round takes the two failure types of `results/contrast_audit.md` that need no new acoustic
measurement:
* **A, "accepted alternative"**: the front-end lists the learner's sound as an acceptable
  alternative, so the error passes whatever the audio;
* **B, "calibration hides it"**: the raw evidence separates the contrast, but the shown sound
  probability does not move.

Selection used FLEURS **validation** only: the audit's own items, clips and thresholds, with
2 workers. Held-out data was run once, Russian-style: paired before / after in the same tree,
using `PRONUNCIADA_CONTRAST_CAPS=0` and `PRONUNCIADA_PT_NASAL=0` for "before". Azure answers are
cached; no requests were made.

## Summary

| language: contrast (learner error) | type | dev hit@5 % before → after | held-out (FLEURS test) before → after [paired 95 % CI of Δ] | adopted |
|---|---|---|---|---|
| zh: aspirated for plain (怕 for 爸) | B cap | 18 % → **79 %** | 19 % → **90 %** [+60, +78] | **yes** |
| en: s for θ | B cap | 18 % → **100 %** | 24 % → **99 %** [+63, +84] | **yes** |
| en: z for ð | B cap | 23 % → **99 %** | 18 % → **98 %** [+69, +87] | **yes** |
| ru: soft for hard | B cap | 2 % → **50 %** | 3 % → 47 % [+35, +53] | **yes** (at the bar on dev, just under it on test) |
| ru: devoiced for voiced (final), voiceless for voiced | B cap | 43 % / 93 % → **82 % / 100 %** | 37 % / 81 % → **82 % / 99 %** | yes |
| ja: voiced for voiceless | B cap | 48 % → **95 %** | 52 % → **95 %** [+33, +52] | **yes** |
| de: voiced for voiceless (non-final) | B cap | 52 % → **88 %** | 38 % → **80 %** [+31, +52] | **yes** |
| de: voiceless for voiced; final devoiced for voiced | B cap | 62 % / 13 % → **94 % / 56 %** | 58 % / 5 % → **99 % / 44 %** | yes |
| ko: aspirated for plain | B cap | 2.5 % → 29 % within the guards | – | **no**: guards bind (planted AUC −0.022 and natives −0.6 pt at the useful offsets) |
| pt: oral for nasal (lá for lã) | A | error accepted by construction at 100 % of positions; strict hit 95 % | accepted 100 % → **42 %** of positions; strict hit 97 % → effective ≈ 56 % | **yes** (word-medial codas) |
| pt: nasal for oral | B | – (no cap fitted: the competitor is not a single partner token) | 19 % → 19 % | no |
| it: open for close / close for open (e/ɛ, o/ɔ) | A | natives use the partner: stressed e 26 %, ɛ 32–40 %, o 13 %, unstressed e 4 % | – | **no**: dropping the < 15 % alternatives leaves "open for close" at 38 % strict hit, under the 50 % bar |
| ko: plain for aspirated | A | natives (ZIPA) use plain at 47 % (35 % initial, 54 % medial) | – | **no**: natives demonstrably use it |
| ru: hard for soft | A | natives use hard at 29 % | – | **no**: natives use it |
| de: final devoicing | A | already devoiced in the reference; the voiced alternative is produced by natives 29 % | – | no change needed |

The opposite directions mostly lose a few points, because the 5 % native threshold moves (test):
* zh "plain for aspirated" 73 % → 68 %;
* ru "hard for soft" 33 % → 24 %;
* ru voicing "voiced for voiceless" 76 % → 72 %;
* ja "voiceless for voiced" 76 % → 73 %.

## A: the front-end alternatives (`bench/calib/alt_audit.py`)

**Method.** For every slot of the class, the canonical symbol is kept and the contrast partner is
the only alternative, with equal weight. The Viterbi path through the raw ZIPA posteriors picks one
(the round-8 protocol), on all FLEURS-validation clips. The bar from round 8 is reused: an
alternative is kept when natives produce it in ≥ 15 % of cases.

| class | n | natives realised as the partner |
|---|---|---|
| ko aspirated → plain | 876 | 47 % (initial 35 %, medial 54 %) |
| ru soft → hard | 2 645 | 29 % |
| pt nasal coda dropped | 2 587 | 11 %: word-final 16 %, **word-medial 7.5 %** |
| it e → ɛ | 3 981 | stressed 26 %, **unstressed 4 %** |
| it o → ɔ | 1 206 | **13 %** |
| it ɛ → e | 1 512 | 34 % |
| de final devoiced → voiced | 257 | 29 % |

**Changes adopted:**
* **pt** (`pronunciada/text/pt.py`, `measured_nasal = True`): a word-medial nasal coda is no longer
  deletable, so *canto* said as *cato* is now flagged. Word-final codas keep the alternative.
* **it** (`measured_mid`, staged off): drops the unstressed e → ɛ and the o → ɔ alternatives. Dev
  guards were met (native acceptance unchanged, near-swap +0.0003), but the contrast's hit stays
  under the bar, so it is not adopted.

**pt dev guards** (FLEURS validation, `bench/calib/sound2.py` collections):

| | before | after | Δ |
|---|---|---|---|
| native sounds | | | −0.15 pt |
| native words | | | −0.02 pt |
| planted sound AUC | | | −0.0007 |
| planted word AUC | | | −0.0003 |
| near-swap AUC | 0.9664 | 0.9664 | 0 |

The canonical symbols are unchanged, so the adapter was not retrained.

## B: contrast-specific caps (`bench/calib/contrast_fix.py`, runtime `pronunciada/calibrate.py`)

**The cap.** Per contrast class, a logistic on six raw-evidence inputs of the sound (llr_sub,
log_ratio_comp, z_llr_sub, peak_post, comp_mean, can_mean). It is fitted on the audit's swaps of
the hidden direction against natives' own sounds of the class.
* It is applied as a cap on the **shown** sound log-odds only, and only when two things hold: the
  expected sound is a member of the class, and the top competing token is its partner (ʰ for
  aspiration, ʲ for palatalisation, s / z for θ / ð, the other voicing class).
* Word and sentence inputs and the coaching inputs are untouched: the scorer keeps the pre-cap
  logits for coaching. So near-swap, words, sentences, cv-real and coaching thresholds cannot move.

**CV.**
* Aspiration (zh + ko) and voicing (ja + de, de + ru): leave-one-language-out.
* Single-language classes (en, ru palatalisation): 5-fold grouped by clip.

**Offset rule.** Per language, each class gets an equal share of two dev budgets: 0.5 pt of native
sound acceptance on all validation natives, and 0.003 of planted sound AUC. Within that, the most
lenient offset reaching the best hit is chosen. A class is adopted at hit ≥ 50 %.

**Dev** (all validation clips; the same rule computed from the shipped weights; end-to-end runtime
on the audit clips reproduces the hits above):

| lang | native sounds | planted sound AUC | words / near-swap / sentences |
|---|---|---|---|
| zh | 0.9979 → 0.9972 | 0.9892 → 0.9885 | unchanged |
| en | 0.9933 → 0.9932 | 0.9823 → 0.9825 | unchanged |
| ru | 0.9966 → 0.9935 | 0.9823 → 0.9795 | unchanged |
| ja | 0.9979 → 0.9957 | 0.9787 → 0.9758 | unchanged |
| de | 0.9977 → 0.9969 | 0.9798 → 0.9773 | unchanged |

**Real learners (dev).** speechocean762 train (English cap; the calibrator's out-of-fold logits):

| | before | after |
|---|---|---|
| detection AUC (expert < 2) | 0.7947 | 0.7946 |
| Pearson of the shown probability with the expert score | 0.682 | **0.668** |
| sounds touched | – | 341 of 46 727 |

* The reported 0–2 phone score (the project's phone-Pearson metric) and the coaching inputs do not
  change.
* The drop in the shown probability's Pearson is reported as a cost: experts often give s-for-θ a
  partial score, while the cap pushes it low.
* OMPAL dev / test and cv-real dev: word, sentence and coaching inputs are unchanged by
  construction. The Mandarin consonant coaching therefore does **not** gain from the aspiration
  cap; that would need the coaching score refitted on capped inputs.

**Shipped:**
* `pronunciada/data/calib/contrast_caps.json`: 5 classes, 6 language entries;
* `calibrate.contrast_caps` / `apply_contrast_caps`;
* about 5 lines in `scorer.py`;
* switch: `PRONUNCIADA_CONTRAST_CAPS=0`.

## Held-out, run once (paired before / after in the same tree)

**Swap benchmark** (FLEURS test; Azure from cache):

| | near-swap AUC before → after (Azure) | far-swap AUC | native false alarms < 0.6 |
|---|---|---|---|
| zh, en, ru, ja, de | identical | identical | identical |
| pt | 0.9599 → 0.9602 (Azure 0.9383) | 0.9970 → 0.9970 | 0.0251 → 0.0249 |

For zh, en, ru, ja and de the caps do not touch word scores.

**Ten-language benchmark:**

| lang | native sounds | planted sound AUC | words |
|---|---|---|---|
| zh | 0.9981 → 0.9972 | 0.9896 → 0.9888 | unchanged |
| en | 0.9930 → 0.9930 | 0.9820 → 0.9825 | unchanged |
| ru | 0.9950 → 0.9912 | 0.9788 → **0.9751** | unchanged |
| ja | 0.9965 → 0.9936 | 0.9754 → **0.9724** | unchanged |
| de | 0.9961 → 0.9956 | 0.9766 → 0.9742 | see below |
| pt | 0.9980 → 0.9977 | 0.9831 → 0.9830 | 0.9408 → 0.9407 |

On held-out data the ru planted sound AUC drop (−0.0037) slightly exceeds the 0.003 dev guard; ja
is at −0.0030.

**The de word columns moved** (native words 0.9644 → 0.9678, word AUC 0.9446 → 0.9455), which the
caps cannot cause. Another agent's in-progress edits in the shared files changed during the run.

**English** (`bench.eval_english`):
* phone / word / sentence Pearson rows: unchanged;
* speechocean762 test phone AUC of the shown probability: lenient 0.9438 → 0.9417, s = 0.5
  0.9240 → 0.9224;
* L2-ARCTIC detection AUC: 0.7121 → 0.7123;
* EpaDB detection AUC: 0.7212 → 0.7212.

**cv-real:**
* pt: 0.738 → 0.743 (Azure 0.760; p vs Azure 0.55);
* every other locale row identical.

OMPAL: not re-run, since nothing it reads changed.

**Contrast rows on FLEURS test** (new benchmark rows; audit generators, 250 test clips per
language, `results/tables/round16_test_contrasts.json`): see the summary table. The CIs are paired
clip bootstraps, 500 resamples, with the native threshold re-derived in each resample.

## What did not work

* **ko aspiration.** The cap separates the swaps (77 % hit at an aggressive offset), but Korean
  natives' own stops show ʰ / plain competitors so often that the planted-error and native guards
  bind first. Korean needs the f0 / VOT detector the audit sketches.
* **it mid vowels.** Removing the < 15 % alternatives is honest but does not reach the bar: the
  strict "open for close" hit is only 38 %.
* **pt nasal for oral.** There is no single partner token to gate on; still 19 %.
* **The first offset rule.** It broke ties towards the harshest cap; for English that cost
  0.08 Pearson on speechocean762. It was changed to the most lenient offset before any held-out
  run.
