# Round 19: Korean changed-word detection (Lingolix test 1, Korean)

**Result.** On the held-out Lingolix Korean test, word AUC rose from 0.912 [0.874, 0.946] to
**0.955 [0.931, 0.976]**. FA@80 fell from 8.4 % [4.5, 19.3] to **3.8 % [2.3, 6.2]**. Paired over clips,
ΔAUC was [+0.015, +0.072] and ΔFA@80 [−14.4, −1.5] pt, with p < 0.001 for both. The changed word now
gets the lowest score in its sentence in 81 % of clips, up from 71 %. Both targets were **missed**:
AUC ≥ 0.96 was missed by 0.005, and FA@80 ≤ 2.5 % by 1.3 pt. Against the published numbers
(not re-run here: Lingolix 0.95 / 3 %, Azure 0.94 / 7 %), the AUC is now above both. FA@80 beats
Azure but not Lingolix. One guard got worse on FLEURS test: native words below 0.6 went from
2.4 % to 3.3 %, as described below.

Built on: the round-17 / 18 head inputs (pooled enc1|enc3|enc5 vectors of a sound plus its
neighbours' mean, `sound_emb.phone_vectors`, `attr_heads.context_vectors`). It also reuses the
round-18 word-factor form (knee at a native quantile, linear fall to a floor; `contrast_word.py`)
and the Lingolix runner and metrics (`bench/third_party/lingolix.py`). New in this round:
* a Korean **sound-identity** head that covers every sound, not one contrast;
* a dev set of planted text edits on Common Voice style audio.

## 1. Diagnosis (held-out aggregates only; nothing was fitted on them)

These figures come from the existing 1.0.1 responses (`results/tables/round19_ko_diag_before.json`).
"Caught" means caught at the threshold that catches 80 % of the changed words.

| bucket | n | AUC | caught at the 80 % threshold |
|---|---|---|---|
| sub_subtle | 20 | **0.753** | 9 |
| sub_moderate | 20 | 0.946 | 16 |
| indel | 20 | 0.927 | 17 |
| sub_gross | 20 | 0.982 | 20 |
| word_omission | 10 | 0.996 | 10 |

* **Subtle bucket.** Almost every item is a one-vowel swap: a/ʌ, ʌ/o, e/i, ɯ/i, plus one aspiration
  swap. The shown sound probability accepted the wrong vowel: 나무 read as 너무 scored 0.94 on its
  "a". ZIPA writes ㅓ as ɔ / o / a and ㅗ as o / u, so the frame posteriors barely separate these vowels.
* **False alarms.** At the 80 % threshold (signal 0.139) there were 117 correct words above it:
  * 54 were in control clips;
  * 62 were in clips whose Whisper transcript differs from the prompt. In those clips the speaker
    really said something else, so no scorer can fix them.
* **Wrong lowest-scoring word.** In 26 of 90 clips another word scored lowest. In 20 of those 26
  the wrong pick was not next to the edit. Most were subtle swaps scoring above ordinary
  native words.

## 2. Dev set (none of it held out anywhere)

`bench/round19ko/cv_pool.py`, `plant.py`, `asr_check.py`:
* **Common Voice 22.0 ko (CC0).** Clips from dev / test / other whose speakers are neither Lingolix
  speakers nor cv-real test-half speakers. 197 of the 200 Lingolix clip ids map to 89 speakers;
  the 3 unmapped clips are not in the pool either. That leaves 737 clips from 39 speakers.
  * **Whisper filter.** Like the benchmark, only clips whose Whisper large-v3 transcript is within
    10 % CER of the prompt are kept: 317 of 656 clips with sound, from 33 speakers.
  * **Why the filter matters.** Without it, five speakers who read other text dominate. Dev AUC
    was then 0.81 instead of 0.936.
* **FLEURS ko validation:** 226 clips.
* **Planted edits.** Each clip gives a control item and one changed item. The edits are our own
  generator's, with Lingolix's kinds, and the replacements are real words.
  * The vocabulary is FLEURS and CV text, with every Lingolix prompt and target removed.
  * `sub_subtle`: a vowel swap, or a swap in aspiration / tension only.
  * `sub_moderate`: a consonant swap differing in ≤ 2 features, or two vowel swaps.
  * `sub_gross`, `indel`, `word_omission`: as in the benchmark.
* **Final dev:** 543 controls and 410 changed items. 60 % of the changed items are on CV audio.
* **Dev baseline:** AUC 0.936 [0.920, 0.951], FA@80 6.4 % [3.7, 8.7], subtle 0.834. This has the
  same profile as the benchmark.

## 3. Fixes, with dev deltas

Dev numbers in this section are pooled word AUC / FA@80.

**A. Korean sound-identity head** (`pronunciada/ko_identity.py`, `data/coach/ko_identity.npz`)
* It predicts the class of sound that was said, from 34 classes:
  * 7 vowels (a ʌ o u ɯ i e; ㅐ / ㅔ merged; ㅢ not judged);
  * the glides j and w;
  * 18 onsets, with lax, tense and aspirated kept apart;
  * 7 codas.
* Labels come from the front-end's standard-pronunciation reference of the prompt.
* Evidence per sound: s = log P(expected) − log max P(other), minus a per-class native offset.
* Class accuracy on FLEURS validation:

  | head | training data | accuracy |
  |---|---|---|
  | linear | FLEURS train | 0.928 |
  | MLP, 256 units | FLEURS train | 0.948 |
  | MLP | + 2.5 k Zeroth-Korean utterances (CC BY 4.0) | 0.955 |
  | **MLP, 512 units (shipped)** | FLEURS train + all 6 Zeroth shards, subsampled (~5 k utterances, ~105 speakers) | **0.960** |

  On CV dev clips the accuracy is 0.65, but that includes the unfiltered pool.
* A vowel-only head came first. It gave +0.015 AUC on an exploratory dev set (5 hand-dropped
  speakers instead of the Whisper filter). The identity head superseded it.

**B. Word factor** (`data/calib/ko_identity_word.json`, round-18 form)
* f = 1 while the word's lowest s is above a knee, then falls linearly to a floor of 0.5 over 8
  units. The knee is the 20 % native quantile at strictness 0.5, and 10 % / 40 % at strictness 0 / 1.
* Grid: knee quantile × multiplicative or logit form × floor / slope × width × per-class offset
  (off, or the 5 % quantile).
* Guards, fixed in advance:
  * FLEURS-validation native word acceptance falls by ≤ 0.3 pt;
  * planted-error word AUC (eval_languages kinds, FLEURS validation) does not fall;
  * FLEURS-validation dev AUC does not fall.
* Pick: the guarded setting with the highest pooled dev AUC. The native guard is the binding
  constraint: unguarded settings reach 0.975 / 1.9 %.

| step (Whisper-filtered dev) | AUC | FA@80 | sub_subtle |
|---|---|---|---|
| baseline | 0.936 | 6.4 % | 0.834 |
| MLP 512, FLEURS + 2 Zeroth shards | 0.971 | 3.0 % | – |
| **shipped head** (all Zeroth shards), in-sample | **0.971** | **2.9 %** | 0.948 |
| shipped head, speaker-grouped 2-fold (offsets, knee and grid choice refitted per fold) | 0.970 [0.960, 0.980] | 2.9 % [1.8, 4.7] | 0.946 |

Paired out-of-fold Δ: AUC [+0.023, +0.047], FA@80 [−4.7, −1.7] pt (p < 0.001;
`results/tables/round19_ko_cvcheck.json`). By bucket, out of fold:

| bucket | before | after |
|---|---|---|
| indel | 0.932 | 0.957 |
| sub_gross | 0.980 | 0.987 |
| sub_moderate | 0.948 | 0.970 |
| sub_subtle | 0.834 | 0.946 |
| word_omission | 0.996 | 0.996 |

**Tried and not used**
* **Counterfactual twins.** I tested the owner's idea as a rival-restricted log-likelihood ratio:
  the expected class against its minimal-pair rivals only (vowel neighbours, the three-way stop
  series, nasal and liquid neighbours). On the exploratory dev set the guarded AUC was **0.954**;
  rivals over all classes gave 0.959.
  * The restriction misses swaps that land outside the rival set, and it saves no native false
    alarms.
  * I did not build a posterior-level twin. ZIPA has no tense token, and its posteriors are what
    is blind here.
  * I did not try the VOT / F0 / duration cues.
* **Logit form of the factor.** On dev it was as good as the multiplicative form or slightly
  better, but it cost more native acceptance. It was not chosen.

## 4. Held-out runs, in order

1. **Lingolix ko before**, with `PRONUNCIADA_KO_IDENTITY=0`. This reproduces 1.0.1 exactly
   (0.912 / 8.4 %).
2. **Lingolix ko after.** The candidate was frozen in STATUS before this run.
3. **FLEURS ko test guard rows.** These are swap-test and fleurs-test-cor, collected with the
   switch off. "After" was computed offline with the runtime's own functions; it matches the
   runtime to a max |ΔP| of 7e-5 on 60 clips.

**Lingolix test 1, Korean.** Rules v2026.10, with clip bootstrap as in report.py. Responses are in
`$CACHE/round19/korean/lingolix/`; the table is `results/tables/round19_ko_lingolix.json`.

| | before | after | paired Δ [95 %] |
|---|---|---|---|
| word AUC | 0.912 [0.874, 0.946] | **0.955 [0.931, 0.976]** | [+0.015, +0.072], p < 0.001 |
| FA@80 | 8.4 % [4.5, 19.3] | **3.8 % [2.3, 6.2]** | [−14.4, −1.5] pt, p < 0.001 |
| changed word lowest | 0.711 | 0.811 | [+0.035, +0.170], p = 0.002 |
| AUC, main-branch rules | 0.915 | 0.958 | [+0.016, +0.073] |
| FA@80, main-branch rules | 7.7 % | 3.3 % | [−13.8, −1.5] pt |

By bucket, AUC before → after:

| bucket | before | after |
|---|---|---|
| sub_subtle | 0.753 | 0.909 |
| sub_moderate | 0.946 | 0.972 |
| indel | 0.927 | 0.934 |
| sub_gross | 0.982 | 0.983 |
| word_omission | 0.996 | 0.997 |

* **Own flag.** Recall 0.489 and false alarms 7.5 % are unchanged. The factor lowers the word
  probability but does not set ErrorType.
* **What remains.** At the new 80 % threshold (signal 0.43) there are 53 correct words above it,
  and 26 of them are in Whisper-mismatch clips. The changed words still missed are:
  * 9 of 20 subtle;
  * 6 of 20 indel. Most add a syllable the speaker said, which the next word absorbs.

## 5. Guards (our own Korean rows)

| set | row | before | after | paired Δ [95 %] | verdict |
|---|---|---|---|---|---|
| FLEURS **validation** (dev) | native word acceptance | 98.31 % | 98.01 % | [−0.50, −0.12] pt | at the 0.3-pt limit (passes by construction) |
| | planted word-error AUC | 0.918 | 0.924 | [+0.004, +0.009] | better |
| FLEURS **test**, swap benchmark (`bench.eval_swap` items) | near-swap word AUC | 0.931 | **0.971** | [+0.024, +0.061] | better |
| | near-swap hit@5 % | 0.753 | **0.913** | [+0.087, +0.227] | better |
| | far-swap word AUC | 0.994 | 0.995 | [+0.000, +0.003] | not worse |
| | far-swap hit@5 % | 0.993 | 0.993 | [0, +0.013] | not worse |
| | native words < 0.6 | 2.37 % | **3.34 %** | [+0.52, +1.47] pt | **worse** (Azure 4.5 %) |
| FLEURS **test**, eval_languages errors | native word acceptance | 97.66 % | 97.21 % | [−0.64, −0.28] pt | **worse** |
| | planted word-error AUC | 0.919 | 0.925 | [+0.003, +0.007] | better |
| cv-real ko (`eval_cv`) | sentence probability (its primary score) | – | – | – | unchanged by construction: only word probabilities change |
| contrast audit, ko sound rows | sound probabilities | – | – | – | unchanged by construction; word-level audit rows not re-run |

The near-swap AUC here is computed by `bench/round19ko/guards.py`, so it is not the scoreboard's
0.944. The hit@5 % (0.753) and native < 0.6 (0.023) baselines match the scoreboard. All rows are
in `results/tables/round19_ko_guards_{val,test}.json`.

The native cost was held to the 0.3-pt limit on FLEURS validation, but on FLEURS test it came out
at 0.45 pt of acceptance and +1.0 pt of words below 0.6. Some loss to noise between validation and
test was always possible, given the rule. I did not retune after seeing the test rows.

## 6. Size and latency

* Size: +1.9 MB (`ko_identity.npz`, float16 MLP 2048 → 512 → 34) and +1.5 kB of JSON.
* Latency: the head costs 0.6 ms per 41-sound clip (numpy). It reuses the existing ZIPA forward
  pass and the pooled vectors already computed for the round-17 / 18 heads.
* End-to-end medians on the shared laptop: 80.7 ms with the switch off, 75.3 ms with it on
  (3.5 s clip, 4 threads). Machine noise is larger than the effect.
* `python -m pytest -q tests`: 581 passed, including 3 new tests in `tests/test_round19_ko.py`.

## 7. Files

* Runtime:
  * `pronunciada/ko_identity.py` (new);
  * `pronunciada/scorer.py`: the evidence is computed when emb is available and lang is ko; the
    factor multiplies the word probability and is folded into `WordResult.contrast_prob`.
* Data: `pronunciada/data/coach/ko_identity.npz`, `pronunciada/data/calib/ko_identity_word.json`.
* Switch: `PRONUNCIADA_KO_IDENTITY=0` restores the old behaviour.
* Tooling: `bench/round19ko/*` (pool, planting, Whisper check, collection, heads, selection,
  2-fold check, guards, Lingolix before / after, runtime check).
* Docs: `docs/DESIGN.md` §7, `docs/LICENSING.md`, `NOTICE` (Zeroth-Korean attribution).
