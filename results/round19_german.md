# Round 19: German false flags on the Lingolix benchmark

**Current result (part 2, section 8; candidate 2 = commit fa3232c; the third look at the benchmark).**
On Lingolix test 1, German, word AUC is now **0.9621** [0.938, 0.981], up from 0.9405
(paired Δ +0.022 [+0.010, +0.037], p < 0.001), so the AUC ≥ 0.96 target is **met**. FA@80 is now
**2.70 %** [1.28, 7.01], down from 6.68 % (Δ −3.98 pt [−8.12, −0.73], p = 0.003), so the FA@80 ≤ 1.5 %
target is **not met**. For comparison, Lingolix published 0.95 and 2 %. Under the later "main" rules
the figures are 0.9636 and 2.57 %. On FLEURS test every German guard row is level or better except
the stress-error word AUC, at −0.0044 (p < 0.001; its hit @ 5 % is +1.1 pt, n.s.). The 42 false flags
that remain are listed in section 8.

Sections 1–7 below describe part 1 (commit 66a0af2) and its held-out look, the second.

**Part 1 result.** On the Lingolix test 1 German items (held out, run once with the frozen candidate),
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

## 8. Part 2: stress, sound identity, bounded second pass (candidate 2, commit fa3232c)

Part 1 is the base; the coordinator accepted its two small guard drops. Everything new was fitted
and selected on the dev set, a native pool and validation data, and then frozen (STATUS.md). After
that came the **third look at the Lingolix benchmark**. The looks so far: (1) baseline diagnosis,
(2) part-1 candidate, (3) this one.

### 8.1 Stress (no change shipped)

**Data** (`bench/round19/stress_collect.py`, `stress_fit.py`, `tables/round19_de_stress_curve.json`).
Every dev clip (CV dev and FLEURS validation natives) was scored on its own text and with 2 planted
stress errors. The check fold has 2,639 native words and 717 error words. Per word, the reliability
of the pitch / energy evidence was recorded:

* voiced share and mean voicing probability;
* word energy;
* number of syllables;
* clip SNR;
* speaking rate.

"Native words lowered by stress" means: word P < τ while sounds × presence ≥ τ, with τ = 0.8
(the Lingolix operating point).

**Shipped, check fold.** 4.02 % of native words are lowered by stress at τ = 0.8 (1.14 % at 0.5).
The stress-error word AUC is 0.878 and its hit@5 is 0.526.

**Families tried** (all fitted on the fit fold):

* a knee, f = min(1, P/k);
* a reliability-aware recalibration: a logistic on the stress logits and their interactions with
  the reliability features, with an intercept that keeps the share of natives below 0.5;
* that recalibration with a knee or a log-odds shift;
* a plain shift.

| setting | natives lowered @0.8 | stress-error AUC (Δ) | hit@5 |
|---|---|---|---|
| shipped | 4.02 % | 0.878 | 0.526 |
| reliability recal, no shift | 5.00 % | 0.897 (+0.020) | 0.551 |
| reliability recal, shift 0.5 | 3.15 % (−22 %) | 0.881 (+0.003) | 0.505 |
| reliability recal, shift 1.0 | 2.08 % (−48 %) | 0.860 (−0.018) | 0.456 |
| plain shift 1.0 | 1.86 % (−54 %) | 0.839 (−0.039) | 0.410 |
| knee 0.85 | 1.82 % (−55 %) | 0.792 (−0.086) | 0.488 |

The pre-set rule asked for at least half fewer natives lowered with no more than 0.01 of AUC lost.
**No setting meets it**, so nothing changes.

* The planted-error AUC is carried by mild penalties. Every family that halves the native cost
  loses 0.02 to 0.09 of AUC.
* The reliability features do help: at the same native cost, the reliability recalibration loses
  half as much AUC as a plain shift.
* Azure has no per-word stress, so there is no Azure figure to compare against.
* About 16 of the 42 remaining Lingolix false flags are driven by stress (8.6), so this is the
  architect's decision.

### 8.2 German sound-identity head (shipped)

Code and data: `pronunciada/de_identity.py`, `data/coach/de_identity.npz` (1.9 MB),
`data/calib/de_identity_word.json`. Pipeline: `bench/round19/cv_train.py`, `identity_collect.py`,
`identity_train.py`, `identity_fit.py`.

**Training data.**

* FLEURS de train: 1,000 clips.
* Common Voice 22.0 de **train**, shard 0: 2,612 clips. Excluded: every Lingolix clip, speaker and
  prompt; every cv-real speaker; the dev / test speakers. 20 % of these CV speakers are held out as
  a native pool.
* 148 k sounds in total, capped at 8,000 per class.

**The head.**

* An MLP, 2048 → 512 → classes, on the pooled enc1|enc3|enc5 vectors, as in Korean.
* The expected class is the reference sound being judged, after edits.
* Native variants are folded into one class: r-coloured long vowels count as the long vowel;
  z / s and ç / x are one class each; word-final obstruents count as devoiced.
* Not judged: r, schwa, ɐ, ʔ, and an unstressed final -e.
* Class accuracy: 97.3 % on dev controls (97.5 % CV, 97.2 % FLEURS) and 97.4 % on the native pool.

**Word factor.** The round-18 one-sided knee:

* per-class offset = the 5 % native quantile;
* knee = the 1 % native quantile of word minima;
* floor 0.3, width 4.

It was chosen by a rule fixed in advance: the lowest fit-fold FA@80 among settings that pass all
of these guards:

* native pool (held-out CV train speakers + fit controls): words below 0.5 and below 0.6 rise by at
  most 0.3 pt;
* controlled sound errors: word AUC does not fall;
* far swaps and word omission (wholly wrong or extra words): AUC does not fall.

For the chosen setting:

* native pool words below 0.5: +0.11 pt;
* controlled-error AUC: 0.9433 → 0.9464;
* fit-fold FA@80: 2.18 → 1.73 %.

**Tried, not used: the Korean two-sided step** (log-odds += b·[margin features], capped at +3,
offset logistic; `identity_fit2.py`, `tables/round19_de_identity_step_tried.json`).

* With an intercept it lifted every word (native pool −1.4 pt below 0.5) but lost ranking:
  fit FA@80 2.5–3.1 % against 2.18 % before, and far-swap and omission AUC fell.
* Without an intercept, or with weighted positives, it did not help either.
* A likely reason: part 1 had already recalibrated the German word log-odds, so there is little
  calibration left to win back.

### 8.3 Bounded explain-away (off)

One second pass costs a median of 26 ms on 9–11 s clips (p95 45 ms, max 50 ms) and 9 ms on
3–4.5 s clips (`ea_cost.py`). No band can keep the 10 s worst case under 12 ms whenever the pass
runs, so the pass stays off. The band code stays in for later use (`config explain_away.lang`,
`PRONUNCIADA_EA_BAND`).

### 8.4 Dev numbers for frozen candidate 2 (runtime)

| | base | part 1 | candidate 2 |
|---|---|---|---|
| dev, all: AUC / FA@80 | 0.9609 / 4.56 % | 0.9712 / 2.95 % | **0.9739 / 2.33 %** |
| dev, check fold: AUC / FA@80 | 0.9548 / 6.41 % | 0.9675 / 3.66 % | **0.9711 / 3.20 %** |
| sub_subtle AUC (all) | 0.922 | 0.949 | 0.963 |

Candidate 2 vs part 1: ΔAUC +0.0026 [0.001, 0.004], ΔFA −0.63 [−0.84, −0.12], p = 0.002.

**Validation guards** (switch off → candidate 2):

| row | off | candidate 2 |
|---|---|---|
| near-swap AUC | 0.9535 | 0.9724 |
| near-swap hit@5 | 0.800 | 0.894 |
| far-swap AUC | 0.9961 | 0.9977 |
| natives < 0.6 | 4.31 % | 3.77 % |
| native acceptance | 96.73 % | 97.16 % |
| sound-error AUC | 0.9433 | 0.9455 |
| sound-error hit@5 | 0.768 | 0.799 |
| stress-error AUC | 0.9276 | 0.9257 |
| stress-error hit@5 | 0.578 | 0.605 |

On cv-real dev every row went up.

**Size and latency.** +1.9 MB. The head is one MLP over the word's sounds and costs well under
1 ms; end-to-end medians with it on and off differ by less than the machine noise (77–83 ms).

### 8.5 Held-out look 3 (run once after freezing)

**Lingolix test 1, German** (`tables/round19_de_heldout3_lingolix*.json`):

| | base (1.0.2) | part 1 | **candidate 2** | paired Δ vs base |
|---|---|---|---|---|
| AUC, v2026.10 rules | 0.9405 [0.911, 0.965] | 0.9565 | **0.9621 [0.938, 0.981]** | +0.022 [+0.010, +0.037], p < 0.001 |
| FA@80, v2026.10 rules | 6.68 % [2.85, 13.10] | 4.88 % | **2.70 % [1.28, 7.01]** | −3.98 [−8.12, −0.73], p = 0.003 |
| AUC, main rules | 0.9437 | – | **0.9636 [0.940, 0.982]** | +0.020 [+0.009, +0.035] |
| FA@80, main rules | 6.03 % | – | **2.57 % [1.25, 6.33]** | −3.45 [−7.50, −0.54] |

Candidate 2 vs part 1: ΔAUC +0.0056 [+0.001, +0.013], p = 0.003; ΔFA −2.18 [−3.26, +0.38], p = 0.25.

AUC by bucket, base → candidate 2:

| bucket | base | candidate 2 |
|---|---|---|
| subtle | 0.872 | 0.905 |
| moderate | 0.966 | 0.985 |
| gross | 0.985 | 0.995 |
| indel | 0.909 | 0.945 |
| omission | 0.999 | 0.999 |

Own flag: recall 0.43 and false alarms 1.3 %, both unchanged. The factor does not set ErrorType.

**FLEURS test** (`tables/round19_de_heldout3_fleurs_test.json`; paired bootstrap over clips;
off → candidate 2):

| row | off | candidate 2 | Δ [95 % CI], p |
|---|---|---|---|
| near-swap AUC | 0.9565 | 0.9745 | +0.0180 [+0.009, +0.032] |
| near-swap hit@5 | 0.767 | 0.887 | +0.120 |
| far-swap AUC | 0.9951 | 0.9964 | +0.0013 |
| natives < 0.6, swap set | 4.64 % | 4.51 % | p = 0.67 |
| native acceptance | 96.48 % | 96.77 % | +0.29 pt, p = 0.04 |
| natives < 0.6 | 4.60 % | 4.34 % | p = 0.08 |
| sound-error AUC | 0.9401 | 0.9421 | +0.0020, p = 0.03 |
| sound-error hit@5 | 0.749 | 0.771 | |
| **stress-error AUC** | 0.9292 | 0.9248 | **−0.0044** [−0.0070, −0.0015], p < 0.001 |
| stress-error hit@5 | 0.596 | 0.608 | n.s. |

The part-1 drop in sound-error AUC is gone.

**cv-real test** (speaker bootstrap; off → candidate 2; all n.s.):

| row | off | candidate 2 | Δ, p |
|---|---|---|---|
| Test A sentence | 0.8228 | 0.8284 | p = 0.09 |
| Test A word mean | 0.8120 | 0.8208 | |
| Test A word min | 0.8016 | 0.8135 | |
| Test B sentence | 0.5406 | 0.5344 | −0.006, p = 0.51 |
| Test B word mean | 0.4921 | 0.4840 | |
| Test B word min | 0.4452 | 0.4398 | |

**Contrast audit, German rows** (FLEURS test, 250 clips; `round19_de_contrast_audit_test_{off,on}.json`).
The runner was pointed at a private cache copy (`$CACHE/round19/german/priv_cache`): the shared
audit plan was copied in, and nothing shared was written. Sound-level rows are unchanged (AUC
within ±0.003 per contrast). Word hit @ 5 % rose for every contrast:

| contrast | off | candidate 2 |
|---|---|---|
| voicing | 19 % | 66 % |
| s / ʃ | 31 % | 83 % |
| m / n | 63 % | 92 % |
| final n / ŋ | 57 % | 78 % |
| front rounded / unrounded | 72 % | 85 % |
| front / back rounded | 74 % | 86 % |
| ç / ʃ | 87 % | 97 % |
| long / short | 18 % | 22 % |
| final devoicing | 1 % | 4 % |

### 8.6 The 42 false flags that remain

This is an aggregate diagnosis on Lingolix after the look. At the FA@80 threshold (word score
0.665) there are 42 flagged correct words out of 1,558. Reaching 1.5 % needs 23 or fewer.

| kind | count | examples | note |
|---|---|---|---|
| stress factor is the largest loss | 16 | Somit, dadurch, darüber, amerikanischen, englischen, beschrieben, Projekt, angeboten, Vulkanismus | blocked by the stress guard (8.1) |
| loanword with a French / English lexicon entry | 6 | Management, Karriere, Karton, Saison, Make-up | lexicon entry differs from what speakers say |
| unknown name | 3 | Buenos, Hampton, Kivu-See | |
| next to the changed word | 7 | | alignment spill-over; the second pass would help but is over budget |
| reduced function word or ending | 5 | oder, einen, hinter, gegenüber ×2 | |
| other | 5 | | |

The 18 changed words still missed are mostly short-word -e / -er changes and one-sound changes that
ZIPA hears as the target (diese / dieser, ach / auch, ihn / in). Their mean log
sounds-probability is −0.13.

### 8.7 For the architect

1. **Stress guard.** The stress-error word AUC on FLEURS test is −0.0044 (hit@5 +1.1 pt). The stress
   channel itself is unchanged; the drop comes from the word-level changes moving native words
   around the stress errors. A stress change that halves the stress-driven false flags costs at
   least 0.018 of AUC on dev. It is the only route to FA@80 ≤ 1.5 % that I found.
2. **Switches.** `PRONUNCIADA_DE_IDENTITY=0` removes the head (back to part 1);
   `PRONUNCIADA_WORD_LANG=0` removes the recalibration; `PRONUNCIADA_DE_R19=0` reverts everything.
3. **Loanword lexicon entries** (Management, Karriere, Karton, Saison, …). Fixing them is a cheap
   follow-up that was not done. It would need a check that it fixes more than it breaks.

**Part 2 files:**

* `pronunciada/de_identity.py`
* `data/coach/de_identity.npz`
* `data/calib/de_identity_word.json`
* `pronunciada/scorer.py`: identity factor, and the bounded explain-away option
* `bench/round19/`: `stress_collect`, `stress_fit`, `cv_train`, `identity_collect`, `identity_train`,
  `identity_fit`, `identity_fit2`, `ea_cost`
* `tests/test_round19_de.py`
