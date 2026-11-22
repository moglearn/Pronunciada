# Round 19: German false flags on the Lingolix benchmark

**Result.** On the Lingolix test 1 German items (held out, run once with the frozen candidate),
word AUC rose from 0.9405 to **0.9565** [0.933, 0.978], paired Δ +0.016 [+0.009, +0.026], p < 0.001.
Lingolix published 0.95 and Azure 0.87. FA@80 fell from 6.68 % to **4.88 %** [1.80, 7.86],
paired Δ −1.80 points [−6.80, −0.51], p = 0.009. Lingolix published 2 % and Azure 22 %.

**Targets.** AUC ≥ 0.96 was **not met** (0.9565). FA@80 ≤ 1.5 % was **not met** (4.88 %).
The AUC point estimate is above Lingolix's published 0.95, but the interval includes it.

**Guards.** Most German guard rows improved, several of them significantly. Two held-out rows
got slightly worse:

* sound-error word AUC on FLEURS test: −0.0021, p = 0.03;
* stress-error word AUC on FLEURS test: −0.0028, p = 0.06.

Both of their hit @ 5 % rows went up. The brief's rule "must not get worse" is therefore
**not strictly met**, and section 6 asks the architect to decide.

Branch `worktree-agent-a2674ec67835464ea`; the frozen candidate is commit 66a0af2.
`PRONUNCIADA_DE_R19=0` restores the round-18 behaviour exactly: 100 of 100 dev items gave
identical benchmark word scores.

## 1. Data and method

**Dev set** (`bench/round19/devset.py`). 720 native clips, none of them from the benchmark:

* 460 Common Voice 22.0 de clips (CC0). They come from CV's dev split, from speakers in the
  cv-real *dev* half, accepted (≥ 2 up votes, 0 down votes), with no non-native accent label, and
  at most 2 clips per speaker.
* Excluded: all 200 Lingolix clips, their 171 speakers (looked up in `validated.tsv`) and their
  prompts.
* 260 FLEURS de validation clips (CC BY).

Each clip gives two items:

* a **control** item: the clip's own text;
* a **changed** item: one word replaced by a real word, in the benchmark's categories
  (sub_subtle / sub_moderate, graded by articulatory distance; sub_gross; indel, i.e. one or two
  sounds added or dropped; word_omission, i.e. a word inserted that was not said).

The replacement words come from a 35.6k-word list: Common Voice and FLEURS word frequencies run
through the German front-end, with Lingolix prompts excluded. Every edit was checked with the
front-end.

Speaker-disjoint folds:

* **fit**: 706 items, used for everything that is fitted;
* **check**: 734 items, used for selection checks.

**Metric.** The benchmark's own rules (`bench.third_party.lingolix`), computed after
`to_azure`. Dev intervals are bootstrap over speakers; Lingolix intervals are bootstrap over
clips, as in report.py.

**Baseline on dev.** Dev reproduces the benchmark's profile: AUC 0.961, FA@80 4.6 %. As on
Lingolix, sub_subtle and indel are the weakest categories.

## 2. Audit of false flags (before)

A correct word is flagged when it is at or above the FA@80 threshold. Causes are assigned in
this order: oov → neighbour of the changed word → last word → sound.

| cause | Lingolix (diagnosis) | dev |
|---|---|---|
| all flagged correct words | 104 / 1558 (6.7 %) | 819 / 17975 (4.6 %) |
| sound (the word model reads low sound evidence) | 52 | 453 |
| last word of the prompt | 21 | 56 |
| unknown word / name (letter-rule reference) | 18 | 176 |
| neighbour of the changed word | 13 | 111 |
| word judged absent / unmatched | 0 | 23 |

**Which factor sinks the word.** Word P = sounds × stress × presence.

* Of the 104 Lingolix false flags, the **stress factor** is the largest log-loss in 33. On dev it
  is the largest in 144 of 819. It is almost never what catches a changed word.
* Removing the stress factor would give dev FA@80 4.56 → 3.92 %.
* But every softening that was tried costs the stress guard (FLEURS validation stress-error word
  AUC 0.930; `bench/round19/stress_knee.py`, `tables/round19_de_stress_knee_sweep.json`):
  * a knee at 0.9 drops it to 0.891;
  * an exponent of 0.5 drops it to 0.907.
* **Not changed.**

**Rule-based alternatives** (`bench/round19/alt_usage.py`): r variants, vocalisation and
deletion, schwa loss, glottal stop, z/s and the rest.

* They are used by natives at the expected rates. Examples: r → ∅ 13.5 % of all r; final
  -er heard without r in 87 % of CV natives.
* They rarely absorb a real change: only 0–9 uses inside the 144 missed changed words, per pair.
* So the allophone lattice (owner idea 1) **is already the shipped mechanism**, and no rule was
  added.
* Frequent unlicensed native substitutions are few: v → f in medial position 7 %; everything
  else is ≤ 3 % (`native_subs.py`).

**Text.** Two clear reference bugs:

* A hyphenated word with an unknown piece went whole through the letter rules, e.g. "Kivu-See"
  → "… z eː eː".
* English names and loans were read with German letter rules (Hampton, Poole, share,
  Cyclone, …).

## 3. Fixes, each with its dev delta

All fixes were fitted and selected on dev, fit fold, then checked on the check fold.

| step | what | dev effect |
|---|---|---|
| A | **text** (`pronunciada/text/de.py`): a hyphenated word whose pieces are not all in the lexicon is read piece by piece. A word the German lexicon does not know (letter rules, compound split, pieces) that *is* in the English lexicon gets its English reading as an allowed whole-word variant (prior 0.25). | all: AUC +0.0012 [0.000, 0.002], FA@80 −0.24 [−0.51, −0.03], p = 0.02 |
| B | **German word recalibration** (`pronunciada/word_lang.py`, `data/calib/word_lang/de.json`, 853 bytes): see below | check fold, offline: AUC 0.9558 → 0.9674, FA@80 6.27 → 3.66 %; length-matched check 5.25 → 3.41 % |
| A+B | the frozen candidate, runtime, all dev | AUC 0.9609 → **0.9712**, Δ +0.0103 [0.008, 0.013]; FA@80 4.56 → **2.95 %**, Δ −1.60 [−2.75, −1.22], p < 0.001. Check fold: 0.9548 → 0.9675, 6.41 → 3.66 %. CV only: 0.9577 → 0.9701, 4.57 → 2.70 %. |

**How B works.**

* A logistic regression re-reads the pooled word model's log-odds together with inputs that
  model already reads: lowest and mean lenient / strict sound log-odds, and log(number of sounds).
* It is applied to both the lenient and the strict log-odds, so strictness still works.
* It is fitted on the fit fold: the planted real-word edits, plus `bench.corrupt` controlled
  sound errors on the same CV clips (2 per reference, 5 kinds).
* The weight of the controlled errors is 3. The rule, fixed before looking: take the largest FA
  gain whose controlled-error check AUC is ≥ before − 1e-4. Results by weight:

| weight | controlled-error AUC (before 0.9356) | FA@80 | outcome |
|---|---|---|---|
| 1 | 0.9347 | 3.19 % | fails the rule |
| **3** | **0.9358** | **3.66 %** | **taken** |
| 10 | 0.9359 | 4.12 % | |

* **Length is evidence, not position.** The planted words are slightly shorter than average.
  The positives are therefore re-weighted to the length distribution of correct words, and a
  length-matched check confirms the gain.
* **The learned effect.** At equal sound evidence, long words (compounds) are flagged less: log n
  coefficient 0.87.
* **Calibration is kept.** The intercept keeps native acceptance on fit-fold controls equal to
  the shipped value.
* **Unknown words.** A word read with letter rules gets +3.25 log-odds. This is the smallest
  shift that brings its native flag rate down to that of lexicon words: from 11.0 % to ≤ 1.0 %.
* **Untouched:** the stress and presence factors, the sentence level and the 0–10 word score.

**Tried and not adopted** (all measured on dev):

* **Explain-away second pass for German.** On top of A it gave FA@80 −0.47, p = 0.02, and it
  removed most of the neighbour flags. But it costs about **+20 ms** on clips that contain a wrong
  word: it ran on 35 / 40 such clips, median 94 vs 71 ms; correct clips were unchanged. That is
  over the 5 ms budget, so `config.json explain_away.langs` is empty and the code path stays
  available. Without it, B was refitted, so the frozen recalibration matches the shipped
  pipeline.
* **Stress knee / exponent**: fails the stress guard (section 2).
* **Per-sound native floor** (owner idea 2): each sound's 2nd-percentile log-odds on fit-fold
  natives is subtracted (the floors do differ: v 0.49 … x 4.60), and B is refitted. Check fold:
  AUC 0.9704 → 0.9726, FA@80 2.49 → 2.55 %. No FA gain, not adopted.
* **Function-word leniency** (owner idea 3): German function words get +δ log-odds.

  | δ | check FA@80 | function-word changes caught (of 271) |
  |---|---|---|
  | 0 | 2.49 % | 223 |
  | 0.5 | 2.63 % | 213 |
  | 1 | 3.02 % | 203 |
  | 3 | 4.18 % | 187 |

  It is worse at every δ: many real changes are function-word swaps (in / ihn, den / dann).
  Not adopted.
* **Unknown-word exponent without a rule**: monotone on dev only because planted words are never
  unknown, so it was replaced by the native-flag-rate rule above.

## 4. Held-out results, run once after freezing (commit 66a0af2)

**Lingolix test 1, German** (`tables/round19_de_heldout_lingolix.json`):

| | before (1.0.2) | after | paired Δ |
|---|---|---|---|
| word AUC | 0.9405 [0.911, 0.965] | **0.9565** [0.933, 0.978] | +0.016 [+0.009, +0.026], p < 0.001 |
| FA@80 | 6.68 % [2.85, 13.10] | **4.88 %** [1.80, 7.86] | −1.80 [−6.80, −0.51], p = 0.009 |
| sub_subtle / sub_moderate / sub_gross / indel / word_omission AUC | 0.872 / 0.966 / 0.985 / 0.909 / 0.999 | 0.894 / 0.983 / 0.991 / 0.937 / 0.999 | |
| own flag: recall / false alarms | 0.42 / 1.2 % | 0.43 / 1.3 % | |

Published for comparison: Lingolix AUC 0.95 and FA@80 2 %; Azure 0.87 and 22 %.

What is left on Lingolix (aggregate diagnosis, after the fix):

* 76 flagged correct words: sound 45, last word 14, neighbour 12, unknown word 5.
* The stress factor is the main factor in 32 of the 76.
* The 18 changed words below the threshold have a mean log sounds-probability of −0.11. These
  are mostly subtle one-sound changes in short words (diese / dieser, ihn / in).

**FLEURS test** (`tables/round19_de_heldout_fleurs_test.json`). Swap set is fleurs-swap test
(450 references); the error protocol is 400 clips with 2 planted errors per kind. Paired
bootstrap over clips.

| row | before | after | Δ [95 % CI], p |
|---|---|---|---|
| near-swap word AUC | 0.9565 | 0.9663 | +0.0098 [+0.0061, +0.0139], p < 0.001 |
| near-swap hit @ 5 % | 0.767 | 0.853 | +0.087 [+0.032, +0.137], p = 0.004 |
| far-swap word AUC | 0.9951 | 0.9954 | +0.0004, p = 0.18 |
| native words < 0.6 (swap set) | 4.64 % | 4.03 % | −0.61 pt [−1.13, −0.20] |
| native word acceptance (P ≥ 0.5) | 96.48 % | 97.04 % | +0.56 pt [+0.31, +0.79] |
| native words < 0.6 (error set) | 4.60 % | 4.02 % | −0.57 pt |
| **stress-error word AUC** | 0.9292 | 0.9264 | **−0.0028** [−0.0053, +0.0000], p = 0.056 |
| stress-error hit @ 5 % | 0.596 | 0.616 | +0.020 [+0.003, +0.040], p = 0.028 |
| **sound-error word AUC** (5 kinds) | 0.9401 | 0.9380 | **−0.0021** [−0.0039, −0.0003], p = 0.028 |
| sound-error hit @ 5 % | 0.749 | 0.753 | +0.004, p = 0.69 |

**cv-real, test half** (speaker bootstrap; protocol of eval_cv, our scores only):

| test | score | before | after | Δ, p |
|---|---|---|---|---|
| A | sentence prob (primary) | 0.8228 | 0.8284 | +0.0056 [−0.0005, +0.0132], p = 0.09 |
| A | word mean | 0.8120 | 0.8170 | +0.0050, p = 0.61 |
| A | word min | 0.8016 | 0.8054 | +0.0038, p = 0.67 |
| B | sentence prob | 0.5406 | 0.5344 | −0.0062 [−0.0270, +0.0122], p = 0.51 |
| B | word mean | 0.4921 | 0.4910 | −0.0012, p = 0.99 |
| B | word min | 0.4452 | 0.4391 | −0.0062, p = 0.80 |

**Validation versions of the same guards** (used while iterating, `tables/round19_de_guards_val.json`):

* near-swap AUC +0.009;
* native words < 0.6: −0.72 pt;
* stress AUC +0.0001;
* sound-error AUC −0.0016 (p = 0.17);
* cv-real dev: every AUC up (Test B word min +0.060, p = 0.03).

**Not run:**

* The contrast audit's German rows. Its runner writes into the shared audit cache and the shared
  results table. The near-swap rows above measure the same kind of one-sound swap.
* `bench/eval_languages.py` / `eval_swap.py` themselves. They rewrite the shared tables. The rows
  above re-implement their German measurements on the same test items: fleurs-swap test, and the
  error kinds of the prosody protocol.

## 5. Size and latency

* **Size.** One 853-byte JSON. No new model, and the backbone is unchanged.
* **Latency.** A 3.5 s clip on 4 threads:
  * median 68 ms after vs 77 ms before on correct clips, and 68 vs 71 ms on changed clips;
  * the difference is noise from other jobs on the machine;
  * the recalibration is a dot product per word.
* **One-time cost.** The English front-end loads lazily on the first unknown word of a process:
  80 ms once, then 0.5 ms per word.
* **No second pass ships.**

## 6. Decisions for the architect

1. **Guards.** Sound-error word AUC −0.002 (p = 0.03) and stress-error word AUC −0.003
   (p = 0.06) against large gains elsewhere:
   * Lingolix AUC +0.016;
   * near-swap hit @ 5 % +8.7 pt;
   * 0.6 pt fewer native false flags;
   * both hit @ 5 % rows up.

   The candidate is on by default because it passed its pre-set dev criteria.
   `PRONUNCIADA_WORD_LANG=0` keeps only the text fix; `PRONUNCIADA_DE_R19=0` reverts everything.
2. **Stress.** The stress channel is now the largest single cause of Lingolix false flags (32 of
   76). Any softening costs stress-error detection; the trade-off is in section 2.
3. **Explain-away for German** is worth about 0.5 pt of FA@80 on dev, but needs about 20 ms on
   clips with errors. Enabling it is one config entry plus a refit of B
   (`fit_rerank dev-text1ea --cor devcor-ea --cor-weight 10` gave check FA@80 3.30 %).
4. **Next lever for the missed changes** (subtle one-sound changes): the Korean "which sound was
   said" head of round 19 is the obvious candidate. For German it must count the native r /
   schwa variants as correct. Not tried here.

## 7. Language-general?

* **Recalibration (B).** The mechanism is language-general: `word_lang.py` reads
  `data/calib/word_lang/<lang>.json` for any language. Rolling it out needs, per language:
  * a dev set like this one (`devset.py` is German-specific only in its sources and vocabulary);
  * the controlled-error companion set;
  * a refit and the same guards.

  The long-word effect (log n) is likely strongest in compounding languages.
* **Hyphen pieces.** General. It could move into `_eu_base.py`.
* **English reading of unknown words.** Useful in fr, es, it and pt too. It needs the English
  front-end and a per-language check.

## Files

* `pronunciada/text/de.py`
* `pronunciada/word_lang.py`
* `pronunciada/scorer.py` (the call site, and the per-language explain-away option, empty by default)
* `pronunciada/data/calib/word_lang/de.json`
* `pronunciada/data/calib/config.json`
* `tests/test_round19_de.py`
* `bench/round19/*`: devset, collect, evaluate, audit, alt_usage, native_subs, chain, whatif,
  wordmodel, stress_knee, fit_rerank, rerank_variants, ideas, guards_dev, cv_guard, latency
* `results/tables/round19_de_*.json`

Dumps and logs are in `$PRONUNCIADA_CACHE/round19/german/`.
