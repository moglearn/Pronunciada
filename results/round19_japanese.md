# Round 19: Japanese word scores on Common Voice style audio (not adopted)

**Outcome.** The frozen candidate raised the Lingolix test-1 Japanese word AUC from 0.846 to 0.884.
That is +0.038 with p = 0.115, so it is not significant on 200 clips. It **failed two of our own
held-out guards**:

* FLEURS-swap far swaps: AUC 0.991 → 0.958 (p = 0.002).
* cv-real Test A Japanese word mean AUC: −0.046 (p = 0.012).
* cv-real Test A Japanese word min AUC: −0.093 (p = 0.001).

It also costs about 6 ms per benchmark clip, over the 5 ms budget. So **everything is off by
default** and the shipped behaviour is round 18's. `PRONUNCIADA_JA_ROUND19=1` turns the round-19
parts on. No target was met:

| target | before | after (candidate) | met? |
|---|---|---|---|
| Lingolix ja word AUC ≥ 0.96 | 0.846 [0.796, 0.889] | 0.884 [0.839, 0.924] | no |
| FA@80 ≤ 4 % | 20.0 % [12.8, 37.0] | 16.6 % [8.3, 30.0] | no |
| length items caught by own flag ≥ 37 / 40 | 19 | 21 | no |

Benchmark numbers come from `bench/round19_ja/unit_eval.py`, our re-implementation of the
benchmark's unit mapping and metrics. On the published responses it gives exactly the published
0.846 and 20.0 %. Intervals are a bootstrap over clips: the benchmark has no speaker ids, and the
CV speaker map gives 184 speakers for 200 clips.

## Diagnosis on the benchmark (aggregates only, before any change)

| bucket | n | AUC |
|---|---|---|
| length | 40 | 0.732 |
| sub_subtle | 20 | 0.910 |
| sub_moderate | 15 | 0.926 |
| sub_gross | 10 | 0.959 |
| indel | 15 | 0.909 |

* **FA@80 is a miss problem, not a false-alarm problem.** The threshold that catches 80 % of
  changed words sits at word score 0.898. 20 changed words score above it, and 14 of those are
  length items.
* **Length coverage.** The length channel judged a sound in 32 of the 40 length words. Its factor
  fell below 1 in only 24 of them.
* **Length false alarms on correct words.** Length probability < 0.03 on 6.1 % of expected-long
  vowels, 5.3 % of geminates and 0.7 % of short vowels. Common Voice speech is faster than FLEURS,
  and long vowels get shortened.
* **Where the 188 correct units above the threshold come from:**
  * 113: mild sound-score dips only;
  * 44: the length factor;
  * 13: low sounds;
  * 12: present < 0.5;
  * 5: the contrast factor;
  * 1: no system word overlapping the unit.
* **Mechanisms found in the code** (they confirm the brief's observations):
  1. **っ lands in the next word.** Open JTalk splits 行っ|た, and the front-end merges っ into the
     next mora, so `tt` sits in the word た with its vowel in the previous word. The length labels
     required both vowels in the same word, so the sound was never judged.
  2. **Repeated vowels.** Readings such as 場合 バアイ and 思惟 シイ came out as two short vowels.
     A short vowel next to a vowel is never judged, and for CTC two identical tokens in a row are
     awkward.
  3. **Vowel pairs are never judged.** No short vowel next to another vowel was judged, so 倍 /
     恋 against 場合 / 行為 could not be detected.

## Dev set (built first; no benchmark clip, speaker or prompt)

`bench/round19_ja/cv_pool.py` and `plant.py` built it.

* **Exclusions.**
  * The 200 benchmark clips were mapped to 184 Common Voice speakers through `validated.tsv`. Those
    speakers and the benchmark prompts are excluded everywhere.
  * Speakers whose cv-real half is "test" are excluded too.
* **Audio.**
  * Dev: CV ja `dev` split, 680 clips from 368 speakers, plus FLEURS validation.
  * Fit: CV ja `train` split, 1800 clips from 583 speakers, plus FLEURS train.
  * CV's splits are speaker-disjoint, and this was checked again. All CV audio is CC0.
  * Clips are 2–8 s with prompts of at least 8 characters, like the benchmark (2.1–7.9 s, 8–40
    characters).
* **Units.** Open JTalk accent phrases, which are close to the benchmark's bunsetsu units.
* **Edits.** One unit's core is replaced by a real word (cores seen in CV ja and FLEURS sentences).
  The new word differs from the original by a length change (ー, っ, repeated vowel), one
  substituted sound, or one sound added or dropped. Substitutions are graded by articulatory
  feature distance. The front-end must leave every other unit unchanged.
* **Size.**
  * Dev: 744 rows. 429 changed (length 114, subtle 128, moderate 79, gross 46, indel 62) and
    315 controls, 76 % CV.
  * Fit set: 2295 rows, 1061 controls.

The dev set is harder than the benchmark: baseline AUC 0.811 against 0.846, and 0.769 on its CV
part. Its indel bucket is much harder (0.685), because most of its indel edits drop a sound that
natives barely realise.

## Fixes, each with its dev delta

Dev deltas are paired, with a speaker-grouped bootstrap.

| step | dev AUC | Δ vs base | FA@80 |
|---|---|---|---|
| round 18 (base) | 0.811 [0.784, 0.839] | | 29.9 % |
| + merge: a repeated vowel mora becomes one long vowel (front-end) | 0.813 | +0.002 (p = 0.006) | 29.7 % |
| + labels: length judged across word boundaries (行っ\|た, 変え\|て) and for short vowels next to a vowel, except e+i, o+u and repeated vowels | 0.815 | +0.005 (p = 0.14); length bucket 0.740 → 0.769 | 29.9 % |
| + wordcal: word recalibration (below) | **0.870** [0.845, 0.893] | **+0.060 (p < 0.001)** | **19.7 %** (p < 0.001) |

On the dev parts: CV 0.769 → 0.844, FLEURS 0.881 → 0.923.

**wordcal** (`pronunciada/word_cal.py`, `data/calib/wordcal/ja.json`, 35 kB):

* **Model.** 200 boosted depth-3 trees over the word's existing pieces:
  * sounds logit, lowest length logit, contrast factor, present probability;
  * word-table statistics of the sound evidence;
  * a new free-decode match (`pronunciada/word_match.py`): greedy tokens over the word's region,
    aligned to the word's sounds and their alternatives, giving insertion / deletion /
    substitution counts.
* **Fit.** On the planted fit set.
* **Native map.** Out-of-fold logits are mapped by quantiles onto the shipped word logit of
  untouched FLEURS-train words. In-sample logits raised FLEURS-validation native false alarms by
  0.6 point.
* **Offset.** A uniform logit offset of +0.3 restores FLEURS-validation native rates; it does not
  change rankings:

  | FLEURS validation natives | round 18 | candidate |
  |---|---|---|
  | words with prob < 0.5 | 2.11 % | 1.99 % |
  | words with prob < 0.6 | 3.52 % | 2.81 % |

* **Ablation.** Without the match counts, dev AUC is 0.867 and the indel bucket 0.763 instead of
  0.822.
* **Parity.** Offline and runtime agree on 740 of 742 dev clips.

**Tried and dropped on dev:**

* **Length-model refit** on FLEURS train plus CV train natives (2797 clips) with the new labels,
  with or without the round-17 head evidence as a tree input. It was worse than the shipped
  channel on CV dev planted length words: sound-level AUC 0.903 against 0.930, catch at 2 %
  natives 37 % against 50 %. A first version leaked the label through the head-evidence sign and
  was fixed before this comparison. The shipped timing model was kept.
* **Lower word-factor floor** (0.4 → 0.02, with knee and width). It did not help with either length
  model (dev AUC within ±0.003).

## Held-out runs, in order

1. **Lingolix ja baseline:** the diagnosis dump of round-18 code. It reproduces 0.846 / 20.0 %.
2. **Frozen candidate 112446b on Lingolix ja,** with the baseline re-scored alongside for a paired
   comparison (`results/tables/round19_ja_lingolix.json`):

   | | before | after |
   |---|---|---|
   | AUC | 0.846 [0.796, 0.889] | 0.884 [0.839, 0.924] |
   | FA@80 | 20.0 % | 16.6 % |
   | length bucket | 0.732 | 0.816 |
   | sub_subtle | 0.910 | 0.894 |
   | sub_moderate | 0.926 | 0.990 |
   | sub_gross | 0.959 | 0.929 |
   | indel | 0.909 | 0.914 |
   | own flag: recall / false alarms | 0.56 / 8.4 % | 0.58 / 8.7 % |

   Paired differences: AUC Δ +0.038 [−0.008, +0.084], p = 0.115; FA@80 Δ −3.4 points
   [−20.4, +4.8], p = 0.21.

   The flag barely moves because the Azure-layout ErrorType comes from coaching items, not from
   the word probability.
3. **Guards, before / after on the same clips:** `bench/round19_ja/guards.py` →
   `results/tables/round19_ja_guards.json`, and `bench/length/heldout_runtime.py` →
   `results/tables/round19_ja_length_{before,after}.json`.

| guard | before | after | verdict |
|---|---|---|---|
| FLEURS-swap ja near AUC | 0.968 | 0.965 (Δ −0.003, p = 0.57) | holds |
| FLEURS-swap ja near hit@5 % | 84 % | 88 % (p = 0.05) | holds |
| **FLEURS-swap ja far AUC** | 0.991 | **0.958** (Δ −0.033, p = 0.002) | **fails** |
| **FLEURS-swap ja far hit@5 %** | 97 % | **77 %** | **fails** |
| FLEURS-swap natives below 0.6 | 3.27 % | 2.64 % (p = 0.02) | better |
| cv-real ja Test A sentence prob AUC | 0.798 | 0.803 (Δ +0.004, p = 0.06) | holds |
| **cv-real ja word mean AUC** | 0.788 | **0.742** (Δ −0.046 [−0.081, −0.013], p = 0.012) | **fails** |
| **cv-real ja word min AUC** | 0.775 | **0.682** (Δ −0.093 [−0.146, −0.040], p = 0.001) | **fails** |
| length table: planted length sounds caught at 5 % native FA | 95.5 % (1701) | 92.5 % (2136, more sounds judged) | |
| length table: length-swapped words caught | 54.0 % | 55.4 % | |
| length table: native word acceptance | 97.06 % | 97.87 % | |

**Why it failed.** The trees were fitted on one-sound edits only. A word whose whole text is
wrong (a far swap) sits outside that range and lands around 0.55, where the old product gave about
0.09: mean planted-word score 0.087 → 0.55. Trees cannot extrapolate below their lowest leaves.
On cv-real, the reviewers' rejected clips are often gross mismatches of the same kind.

**Combinations checked on dev after the failure (not run on held-out):**

| combination | dev AUC | FA@80 |
|---|---|---|
| min(shipped, recalibrated) | 0.850 | 24.2 % |
| geometric mean | 0.856 | 24.5 % |

Either keeps the gross-mismatch floor of the shipped product. They are the obvious next candidate,
but they need a far-swap dev check (the frozen FLEURS-swap dev set of `bench/calib/bias_grid.py`)
and the cv-real dev half before a second held-out run.

## Size and latency

* **Size:** +35 kB of data (`data/calib/wordcal/ja.json`).
* **Latency:** the median per benchmark clip rose 94.1 → 100.5 ms (+6.4 ms on the shared laptop;
  +3.4 ms on dev clips). Most of it is the pure-Python match alignment, which would need a
  vectorised version to fit the 5 ms budget.
* **Off by default:** every other language, and Japanese with the default switch, run the round-18
  code paths.

## Owner hypotheses (IDEAS.md) and the Korean identity head

* **Counterfactual long / short alignment.** Not built. The shipped channel already reads the
  acoustic model's own length-token posterior (`colon_post`) and the closure silence before a
  stop, the direct small-っ cue. After the new labels, 86 of 89 CV dev length units are judged.
  What limits the score is how the evidence is turned into a word score, not coverage.
* **Signal-processing data.** Not needed: native data did not limit the refit, which simply lost to
  the shipped model.
* **A Japanese "which sound was said" head**, like the Korean one, is the natural next step for
  the subtle / gross buckets. Not attempted this round.

## Files

* **Package:** `pronunciada/ja_round19.py` (switches), `text/ja.py` (merge), `length.py` (labels),
  `word_cal.py`, `word_match.py`, `scorer.py` (hooks), `data/calib/wordcal/ja.json`.
* **Bench:** `bench/round19_ja/`: cv_pool, plant, dump, unit_eval, collect, length_fit, lendiag,
  wordcal, parity, combine, guards, tables.
* **Tests:** `tests/test_ja_round19.py`.
* **Tables:** `results/tables/round19_ja_{lingolix,dev,guards,length_before,length_after}.json`.
* **Caches:** dev and fit data under `$PRONUNCIADA_CACHE/round19/japanese/` (about 0.6 GB of
  audio).
