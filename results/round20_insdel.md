# Round 20: extra and missing sounds from the free greedy decode (not adopted)

**Result: a clean negative in every language tested. Nothing ships and the runtime is unchanged.**

| language, dev set | shipped (main 8c74789) | with the greedy-decode term (out of fold) | paired Δ [95 %], p | verdict |
|---|---|---|---|---|
| ja, round-19 dev: word AUC | 0.8674 [0.840, 0.890] | 0.8676 [0.840, 0.890] | +0.0002 [−0.0002, +0.0007], p = 0.34 | no gain |
| ja: indel bucket AUC | 0.7321 [0.669, 0.790] | 0.7335 | +0.0014 [−0.0001, +0.0034], p = 0.08 | no gain |
| ja: FA@80 | 17.37 % | 17.43 % | +0.06 pt [−0.68, +0.16], p = 0.65 | slightly worse |
| ko, round-19 dev (Whisper-filtered): word AUC | 0.9734 [0.964, 0.982] | 0.9735 [0.964, 0.982] | +0.0001 [−0.0000, +0.0002], p = 0.26 | no gain |
| ko: indel bucket AUC | 0.9607 [0.946, 0.976] | 0.9609 | +0.0002 [−0.0002, +0.0005], p = 0.41 | no gain |
| ko: gross-swap bucket AUC | 0.9898 | 0.9898 | [−0.0002, −0.0000], p = 0.04 | guard fails (falls) |
| en, speechocean762 train, marked sounds (score < 1.5): AUC | 0.9706 | 0.9706 | 0 (a = 0 in all 5 folds) | no gain |
| en, FLEURS validation planted errors, sound AUC | 0.9872 | 0.9872 | 0 (a = 0 in all 5 folds) | no gain |

* Out of fold, the dev control words below 0.6 rose in both languages: ja 247 → 249, ko 111 → 113.
  The native guard therefore fails as well.
* French, Spanish and German were not run. The brief made that step conditional on a pass in
  Japanese or Korean.
* **No held-out look was made.** L2-ARCTIC, EpaDB, Lingolix, FLEURS test and the cv-real test half
  were not touched. No OpenPronounce model was run.
* No Azure request was made.

**Why it does not help.** The greedy string differs from the expected sounds far too often on
correctly read words:

* In Japanese and Korean the feature fires on 44–51 % of correct units. Among the correct units
  already above the 80 %-recall threshold, the shipped system's false alarms, it fires on 72–81 %.
* It does fire on most of the missed indel words: 77 % in ja, 81 % in ko. But it fires on caught
  ones just as often, so it does not separate the misses from the false alarms.
* Even unguarded, the best in-sample lower-only term adds only +0.0005 AUC (ja) and +0.0003 (ko).
  It pushes 35 (ja) and 14 (ko) native control words below 0.6.
* In English, "dropped in the greedy decode" fires on 3 % of correct sounds and on 84 % of the
  planted extra reference sounds. The sound model already ranks those sounds at AUC 0.984, so the
  CV never gave it a weight.

The two existing word steps already read a cheaper form of the same evidence:

* Japanese: the plain greedy-match `extra` count and `llr_ins`;
* Korean: `llr_ins`, `llr_del` and the word duration.

The weighted edit distance adds nothing on top of them.

## 1. Method

**Feature.** `bench/round20_insdel/greedy_match.py`. It is offline code; after this result it
lives outside the package.

1. **Decode.** Collapsed greedy decode of the adapter-corrected log-probabilities, as
   `Evidence.free_tokens` already computes it.
   * Tokens are grouped into units: a base symbol plus its modifiers.
   * Length marks are dropped and a repeated unit counts once, because length is the length
     channel's business.
2. **Assign units to words.** A unit belongs to a word when its middle frame falls inside the
   word's aligned span, widened by half the gap to the neighbouring words (`half`) or by 2 frames
   (`tol2`).
3. **Align per word.** A weighted edit distance aligns the word's units with its expected sounds.
   * Each sound may be said as its canonical form or as any front-end alternative. An allowed
     empty alternative makes dropping it free; this is the native-variant allowance.
   * Each sound may consume 0–3 heard units.
   * **Plain** variant: every edit costs 1.
   * **Weighted** variant:
     * substitution = 0.2 + 0.2 × the number of differing articulatory features, capped at 1;
       consonant vs vowel = 1;
     * insertion and deletion costs per unit base come from native rates:
       clip(1 + ln(mean rate / smoothed rate) / 3, 0.3, 1.5), so sounds that natives often drop or
       add cost less;
     * the rates are counted on the control clips of the other speaker folds only.
   * **Top-2** variant: a dropped sound whose token is among the top-2 tokens of some frame in the
     word's region counts half.
4. **Per-word features:**
   * weighted insertions and deletions;
   * their sum, and the sum per sound ("rate");
   * the number of edits;
   * the mean top-1 probability of the inserted units ("mass");
   * an edge-weighted sum.
5. **Per-sound feature:** "dropped in the greedy decode".

**Data.** Every dev clip was scored once with the current runtime (`collect.py`, main 8c74789).
Only small arrays were kept: per frame the top-4 tokens and their log-probabilities, the spans,
the aligner evidence and the shipped outputs. That is 8 kB per clip, under
`$PRONUNCIADA_CACHE/round20/insdel/`. The dev sets:

| language | dev sets |
|---|---|
| ja | round-19 dev (744 items); dev guard sets devnat, devswap, cvdev |
| ko | round-19 dev, Whisper rule as in round 19 (953 items); FLEURS validation with the eval_languages errors; Zeroth test (native pool) |
| en | speechocean762 TRAIN (2,500 clips, expert 0–2 sound scores and `<DEL>` marks); FLEURS validation with the eval_languages errors (2,358 items) |

The shipped dev numbers reproduce round 19 exactly: ja 0.8674 / 17.4 % / indel 0.732; ko natives
50 / 61 / 128 below 0.6.

**Word term (ja, ko).** Lower only: z' = z − a · max(0, f − k), added after the existing step.
* Grid: f ∈ {sum, rate, ins, del, edits, edge sum, mass}, a ∈ {0.25, 0.5, 1, 2, 4},
  k ∈ {0, 0.5, 1, 1.5, 2, 3}, and four variants.
* Speaker-grouped 4-fold CV. On the three fit folds, pick the setting with the highest dev AUC
  among those whose fit-fold native control words below 0.6 do not rise. Apply it to the held-out
  fold.
* Pool the out-of-fold outputs. Paired speaker bootstrap, 1,000 draws.
* The frozen candidate is the same rule run on the whole dev set. Its guards are the round-19
  rows.

**Sound term (en).** z' = z − a · d, where d = the drop indicator (weighted variants: × the
sound's deletion cost), a ∈ {0, 0.25, …, 3}.
* z is the uncapped sound_B log-odds at strictness 0.5, the same signal the marked-sound tests
  read.
* Speaker-grouped 5-fold CV on speechocean762 train; FLEURS-sentence-grouped 5-fold CV on FLEURS
  validation, where clean sounds below 0.5 may not rise.
* The held-out rows were to be run only for a frozen candidate with a paired significant dev
  gain. There was none.

## 2. Feature alone (unit AUC; positives = planted units of the bucket, negatives = all correct units)

| variant (feature: rate) | ja indel | ja all planted | ko indel | ko all planted | ko word omission |
|---|---|---|---|---|---|
| shipped word score (reference) | 0.732 | 0.867 | 0.961 | 0.973 | 0.996 |
| plain | 0.695 | 0.534 | 0.791 | 0.645 | 0.959 |
| plain, no variant allowance | 0.677 | 0.523 | 0.796 | 0.632 | 0.962 |
| weighted | 0.711 | 0.527 | 0.809 | 0.649 | 0.967 |
| weighted, no variant allowance | 0.697 | 0.522 | 0.801 | 0.638 | 0.968 |
| **weighted + top-2** | **0.730** | 0.529 | **0.824** | 0.658 | 0.966 |
| weighted, 2-frame tolerance | 0.558 | 0.469 | 0.743 | 0.635 | 0.941 |

* Weighted costs help a little (+0.02 indel AUC), and so do the top-2 allowance and the native
  variants.
* The 2-frame tolerance is worse. Neighbouring regions then overlap, and correct words collect
  their neighbours' sounds as insertions.
* No variant reaches the shipped score on its own indel bucket. The signal is specific to indels:
  near 0.5 on substitutions.

All variants and features: `results/tables/round20_insdel_{ja,ko}.json`, key `alone`.

## 3. As an extra input to the word step (ja, ko)

| | ja shipped | ja + term (OOF) | ko shipped | ko + term (OOF) |
|---|---|---|---|---|
| AUC | 0.8674 | 0.8676 (p = 0.34) | 0.9734 | 0.9735 (p = 0.26) |
| FA@80 | 17.37 % | 17.43 % | 2.36 % | 2.39 % |
| indel | 0.7321 | 0.7335 (p = 0.08) | 0.9607 | 0.9609 (p = 0.41) |
| length / subtle / moderate / gross | 0.839 / 0.886 / 0.933 / 0.954 | 0.839 / 0.886 / 0.933 / 0.955 | – / 0.948 / 0.977 / 0.990 | – / 0.948 / 0.977 / 0.990 (gross Δ < 0, p = 0.04) |
| dev control words < 0.6 | 247 | 249 | 111 | 113 |

**Fold picks:**
* ja: weighted-no-alt mass (0.5, 0.5); plain rate (1, 0.5); plain rate (0.25, 3) twice.
* ko: weighted rate (0.25, 0); weighted-top-2 del (0.5, 3), (1, 3) and (0.25, 3).

The fit folds gain at most 0.0003 AUC. The picks change from fold to fold, which is what a
near-zero effect looks like.

**Frozen candidates, dev guards** (no held-out look follows, since neither passed):

| guard | ja (plain rate, a 0.25, k 3) | ko (weighted top-2 del, a 0.25, k 3) |
|---|---|---|
| native words < 0.6 | FLEURS val 194 → 194, CV 704 → 704 | FLEURS val 50 → 50, CV 61 → 61, Zeroth 128 → 128 |
| far swaps / whole-wrong-word | FLEURS-swap dev far AUC 0.9992 → 0.9992 | planted word_swap AUC 0.99734 → 0.99734 |
| controlled errors | – | planted word AUC 0.92978 → 0.92978 (−6e-6); phone_del 0.7243 → 0.7243; phone_ins 0.97259 → 0.97257 |
| other | near AUC 0.97860 → 0.97860; cv-real dev word min AUC 0.7644 → 0.7626 | sub_far 0.99032 → 0.99030 |

**The two-sided alternative.** Also raise words whose feature is 0, by +0.25 / 0.5 / 1. In
sample, no setting improved AUC without native words below 0.6 rising
(`results/tables/round20_insdel_{ja,ko}_diag.json`).

## 4. English marked sounds

**speechocean762 train** (46,727 sounds; 3,864 with expert score < 1.5; 443 marked `<DEL>`):

* "Dropped in the greedy decode" fires on:
  * 2.9 % of sounds the experts scored 2;
  * 13 % of the marked sounds;
  * 29 % of the `<DEL>` sounds.
* On its own its AUC is 0.55 for marked sounds and 0.63 for deleted ones.
* All five folds chose a = 0, so the out-of-fold outputs equal the shipped ones:
  * marked-sound AUC 0.9706;
  * deleted 0.9829;
  * Pearson with the 0–2 score 0.884.
* Caveat: sound_B was fitted on speechocean762 train, so this baseline is in sample and
  optimistic. This is why a second, unseen dev set was added.

**FLEURS validation with planted errors** (sound_B has not seen it):

* The indicator fires on 84 % of the planted extra reference sounds (`phone_ins`) and on 3.0 %
  of clean sounds.
* sound_B alone already ranks those sounds at AUC 0.984. All five folds chose a = 0.

**No candidate, so no L2-ARCTIC / EpaDB run.** The OpenPronounce gap on marked sounds (0.743 vs
0.712, 0.741 vs 0.721) is not explained by missing sounds that we fail to flag. In speechocean762
train, deletions are 11 % of the marked sounds, and the sound model already ranks them at 0.98.

## 5. Cost

* **Size:** 0 bytes. Nothing was added to the package.
* **Speed:** the offline implementation (pure Python dynamic programming per word) takes about
  6 ms per Japanese dev clip. That is over the 1 ms budget, so it would need vectorising if it were
  ever adopted. It was not.

## 6. What this rules out, and what is left

* The unconstrained greedy string is not a usable indel detector on top of the round-19 word steps
  in Japanese and Korean. The reason is not "the greedy string agrees with the canonical string
  whenever the aligner already does": on the missed indel words it disagrees 77–81 % of the time.
  But it disagrees almost as often on correct words, because ZIPA's free decode of native speech
  routinely adds, drops or merges sounds (devoicing, reduced vowels, glides). The Spearman
  correlation between the feature and the shipped signal on planted indels is only 0.17 (ja) and
  0.05 (ko), so the information is new. It is just too noisy.
* A detector of extra or missing sounds would have to be judged against natives' own variation
  per context, not per word. One example is a learned "extra syllable here" head on the pooled
  activations, the extension the Korean write-up left open (`llr_ins` with identity-head evidence
  for the inserted sound). This was not tried.

## Files

* `bench/round20_insdel/`:
  * `collect.py`: one scoring pass, small arrays;
  * `greedy_match.py`: the feature;
  * `core.py`: variants, native cost tables, the term;
  * `ablate.py`: feature alone, CV, frozen candidate, guards;
  * `diag.py`: why it fails;
  * `en_sound.py`: speechocean762 train and FLEURS validation, sound level.
* `tests/test_round20_insdel.py` (5 tests).
* Tables: `results/tables/round20_insdel_{ja,ko,ja_diag,ko_diag,en_so762,en_valcor}.json`.
* Reproduce:

  ```
  PYTHONHASHSEED=0 python -m bench.round20_insdel.ablate --lang ja   # same for ko
  ```

  The first runs left tie-breaking between equal-length variants to Python's string hashing. That
  moved Korean fold picks and p by a few hundredths (p = 0.046–0.256) and never changed the
  verdict. Variants are now sorted deterministically, and the numbers above are from that version.
