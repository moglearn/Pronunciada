# Round 19 integration: merged code, equivalence checks, one frozen benchmark run

Branch `round19` was made from main c9773cc. Seven pieces of work were merged:
* the five round-19 branches (Japanese at its final candidate);
* the sound-identity rollout;
* main eaa4080 (the rival re-test).

Every language's outputs are byte-identical to its own branch. With every switch off, they are
byte-identical to main.

Then the full held-out suite was run frozen on the fully merged code, and that run is the new
scoreboard. Nothing was tuned: no threshold, data file or parameter of any branch was touched.

**Short version.**
* **Tally against Azure,** 84 comparisons:
  * 49 significantly ahead (was 40);
  * 32 not different (was 37): 16 ahead n.s., 1 level, 15 behind n.s.;
  * 3 significantly behind (was 7).
* **Regressions.** Every held-out row that got worse is listed in E.3. Two are significant:
  * Mandarin tone swaps, "right word blamed": 0.300 → 0.280. It is now significantly behind
    Azure (−0.107 [−0.207, −0.023], p = 0.02).
  * Portuguese contrast audit, nasal → oral, word hit@5 %: 26 % → 21 %
    (paired Δ −0.052 [−0.124, −0.008], p = 0.04).
* **Known costs**, stated in the branch write-ups and reproduced here:
  * Japanese "long vowel → short" words caught: 60 % → 44 %;
  * German planted stress-error word AUC: 0.932 → 0.919.

## What was merged

| order | source | commit | topic | write-up | switch (0 = old behaviour) |
|---|---|---|---|---|---|
| 1 | worktree-agent-a3633e76c1a06d754 | ac4bc76 | learner test suite (bench/learners; results only) | round19_learners.md | – |
| 2 | worktree-agent-a31e9bcfcc58e849b | b7c2850 | English sentence heads, sound / word blends, all-layer encoder outputs | round19_english.md | `PRONUNCIADA_SENTENCE_HEAD` |
| 3 | worktree-agent-a72249a2162e58229 | 1dc4f5b | Korean identity head, two-sided word step, no-raise rule, `llr_ins` | round19_korean.md | `PRONUNCIADA_KO_IDENTITY` |
| 4 | worktree-agent-a2674ec67835464ea | da9def0 | German text fixes, word recalibration, identity head, stress reliability, loanwords | round19_german.md | `PRONUNCIADA_DE_R19` (+ `_DE_IDENTITY`, `_DE_STRESS_REL`, `_DE_LOAN`, `_DE_TEXT`, `PRONUNCIADA_WORD_LANG`) |
| 5 | worktree-agent-a3e26ec0e6c91086b | 20d6260, then c5eb79c | Japanese candidate 2, then the final candidate (dfc8ab3 + write-up) | round19_japanese.md | `PRONUNCIADA_JA_ROUND19` |
| 6 | worktree-agent-ac79aa17d0dc794f2 | 51ba321 (built on da9def0) | shared `identity.py`; word steps for es, ru, pt, zh (it, en, fr ship nothing) | that branch's rollout write-up | `PRONUNCIADA_IDENTITY` / `PRONUNCIADA_IDENTITY_<LANG>` |
| 7 | main | eaa4080 | rival re-test (bench/rivals, bench/ondevice.py), industry snapshot edit | round19_rivals.md | – |

Merge commits:
* e5bb338, 6313c53, 3c10786, b2724d4, 0d43805 (first five);
* c31da85 (Japanese final);
* 1e91f8b (identity rollout);
* 836dba6 (main).

Check tooling and results are in `bench/round19_merge/`. The scoreboard header now names this run.

### Conflicts and how they were resolved

* **Learners, English, Korean, identity rollout, main:** no textual conflicts.
  * Each automatically merged hunk in scorer.py, delivery_coach.py and DESIGN.md was read. Each is
    guarded by its own language: `self.lang == "ko"`, config `sentence_head` (English only), or
    `identity.enabled(lang)`, which is true only where `data/calib/identity/<lang>.json` exists
    (es, pt, ru, zh).
  * The rollout's per-word hook sits after the German factor, as on its branch (it was built on
    da9def0).
* **German:**
  * `scorer.py`: the Korean and German identity-evidence blocks inserted at the same place. Both
    were kept.
  * `bench/round19/latency.py`: the English and German branches added different files with this
    name. The German one is now `bench/round19/latency_de.py`; `round19_german.md` still names
    the old module.
* **Japanese candidate 2:**
  * `scorer.py`, end of the per-word loop: the Korean factor and the Japanese word step were both
    kept, Korean first.
  * `DESIGN.md` §7: both bullets kept.
* **Japanese final (`llr_ins`):**
  * Both branches carry the **same** `align.py` implementation, byte-identical, so git merged it
    once. One implementation is kept.
  * The only conflict was the `score_reference` call. It now asks for the evidence when
    `lang == "ko"` (always, as on the Korean branch) or when `lang == "ja"` and the Japanese word
    step is on (as on the Japanese branch).
  * Korean outputs are unchanged by this merge: max |Δ| = 0 against the first-pass merge and
    against the Korean branch.
* **"Which sound was said" heads.** They are kept separate, as briefed:
  * `ko_identity.py`, `de_identity.py`, `ja_identity.py`;
  * the rollout's generic `identity.py`, used by es, pt, ru and zh.

  Nothing was refactored.

## A. Tests

`python -m pytest -q tests`:
* first-pass merge (0d43805): **605 passed**;
* fully merged code (836dba6): **609 passed**.

The rollout branch runs `test_prosody_channel` with the Mandarin identity step off. That change was
accepted by the coordinator.

## B. Equivalence: merged code = each branch; untouched languages = main

**Tool:** `bench/round19_merge/equiv.py` (dump / compare), `run_equiv.sh`, `run_equiv2.sh`,
`compare_all.py`, `compare_all2.py`.
* Every side was run from a `git archive` extract of the exact commit, put first on `PYTHONPATH`.
* The dumper checks where `pronunciada` was imported from.
* The branch worktrees themselves were not used, because the Japanese one kept changing during the
  night.

**Clips:** dev and validation data only.
* 100 FLEURS **validation** clips per language (seed 19). Each was scored with its own text and
  with one word replaced by the word at the same place in the next clip's text. The second
  version exercises identity steps, recalibration, flags and coaching.
* For English, also 100 speechocean762 **train** clips.

**Values compared, per score:**
* every field of `SentenceResult.to_dict()` except `timing_ms`, including raw evidence;
* word and sentence probabilities rescored at strictness 0 and 1;
* `Scorer.details()`;
* `coaching()`;
* `compat.azure.to_azure()`.

Strings must match exactly; numbers within 1e-6.

**Fully merged code** (836dba6; `results/tables/round19_integration_equiv2.json`). Each cell is
max |Δ| for fp32 / int8.

| language | compared with | clips (scores) | max \|Δ\| fp32 / int8 |
|---|---|---|---|
| en | English branch b7c2850 | FLEURS val 100 (200) + so762 train 100 (100) | 0 / 0 |
| ko | Korean branch 1dc4f5b | FLEURS val 100 (200) | 0 / 0 |
| de | German branch da9def0, and the rollout branch 51ba321 | FLEURS val 100 (200) | 0 / 0 (both) |
| ja | Japanese final c5eb79c | FLEURS val 100 (200) | 0 / 0 |
| es, ru, pt, zh | rollout branch 51ba321 | FLEURS val 100 (200) each | 0 / 0 |
| fr, it (nothing ships) | main c9773cc | FLEURS val 100 (200) each | 0 / 0 |

Every difference is exactly 0, about 600 k to 860 k values per language and precision.

The first-pass merge (0d43805, `round19_integration_equiv.json`) passed the same check:
* en, ko, de, ja-candidate-2 each equal to their branch;
* zh, es, fr, it, pt, ru equal to main.

Against that first pass, the final merge changed only ja, es, pt, ru and zh. en, ko, de, fr and it
are identical to the first pass.

**Do the other languages pay for the English all-layer outputs?** No.
* The extra stacks (enc0, enc2, enc4) and the `model.emb-enc0-…-final.onnx` variant are requested
  only when a language's sentence projection or sequence head names them. Only
  `sentence_head.en` does.
* Every other language keeps the round-5 output set and file. The latency workers report the
  outputs they hold: enc1, enc3, enc5 and final for every language but English (table F2).
* fr, which ships nothing, is level in time: fp32 +0.7 ms on 9.4 s clips, int8 +0.1 ms. Its peak
  memory is level too: 726 → 724 MB and 591 → 591 MB.

## C. Every switch off = main

The fully merged code was run with every round-19 switch at 0 and compared with main c9773cc on
the same clips:
* `PRONUNCIADA_SENTENCE_HEAD`, `_KO_IDENTITY`;
* `_DE_R19`, `_DE_IDENTITY`, `_DE_STRESS_REL`, `_DE_LOAN`, `_DE_TEXT`, `_WORD_LANG`;
* `_JA_ROUND19`, `_IDENTITY`.

**Result: max |Δ| = 0 for all ten languages, fp32 and int8,** on FLEURS val 100 (200 scores)
each, plus so762 train 100 for English. The first pass gave the same result for en, ko, de and ja.

* Korean (always) and Japanese (when its step is on) compute the extra-sound evidence (`llr_ins`).
  With the switches off, no output reads it.
* With the English switch **on**, every word and sound probability and every coaching item is
  still byte-identical to main. Only the four 0-10 sentence scores and the reported 0-2 sound and
  0-10 word regressions change.

## D. The frozen run

**Driver:** `bench/round19_merge/frozen_run.sh`. These are the steps and settings of the
2026-11-17 / 18 frozen runs:
* eval_english `run,so762,calib,errdet,int8,report` (the speed step was not rerun, as in those
  runs);
* eval_languages (FLEURS tests and the swap benchmark);
* eval_cv and eval_ompal;
* the coaching dumps and tables;
* the contrast audit on the test split;
* the length table;
* zipa_baseline;
* the scoreboard.

Added to them:
* the Lingolix benchmark (`bench.third_party.lingolix all`);
* the learner suite (Pronunciada side);
* `bench.coach.tips_r19`, whose table the scoreboard reads.

**Isolation:**
* Every Pronunciada output was computed fresh into an overlay cache
  (`bench/round19_merge/overlay.py`). Data, models, posteriors and cached Azure responses are
  linked. Output folders start empty. Nothing that already existed was read as output or
  overwritten.
* Multi-worker scoring went through `bench.heavy --lane cpu`.
* **Azure:** `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` and `AZURE_SPEECH_ENDPOINT` were removed
  from every job's environment, so the scripts could not send a request. No file under the Azure
  caches is newer than the start of the run.

**The two runs:**
* **Run 1** covered the first five merges (code 0d43805, results commit d4c2aac).
* **Run 2** covered the fully merged code and is the reference: results commit 28e62df, code
  836dba6 plus tooling, `pronunciada` 1.0.2 (version not changed).
* For run 2, run 1's outputs of the five changed languages (ja, es, pt, ru, zh) were moved aside,
  and every step was run again. The outputs of en, ko, de, fr and it were reused. B proves they are
  byte-identical under the final code.
* Every step whose cache is tied to the package fingerprint was re-scored in full: cv-real, OMPAL,
  learners, tips. **All rows of the new scoreboard come from run 2.**

**Run record:**
* Date: 2026-11-22. Run 1 05:27–08:08, run 2 08:36–09:39.
* Machine: Apple M3 Pro (Mac15,7), 12 cores, 18 GB, macOS 27.0.1, shared with other jobs (load
  average 3–10).
* Python 3.12.13; numpy 2.5.5, onnxruntime 1.32.0, kaldi-native-fbank 1.23.1, scipy 1.18.1,
  scikit-learn 1.9.1, pyarrow 26.0.1, pyopenjtalk-plus 0.4.1.post10.
* Model: zipa-small-crctc-500k (fp32 248 MB, int8 67 MB, unchanged).
* Logs: `$PRONUNCIADA_CACHE/round19/integrator/frozen_logs{_run1,2}/`.

**Reproduced from the branches** (frozen run against each write-up):
* **English:** `english_vs_azure.json` equals the English branch's own third held-out look in
  all 2,636 values: total 0.792, sound 0.700, word 0.640, int8 total 0.789.
* **Korean:**
  * Lingolix ko 0.957 [0.935, 0.977], FA@80 3.1 % (write-up: 0.957, 3.1 %);
  * swap near hit@5 % 0.867 (write-up 0.873, from its own guards script);
  * Lingolix main rules 0.960 / 2.6 % (write-up 0.960 / 2.6 %).
* **German:**
  * Lingolix de 0.967 [0.944, 0.985], FA@80 2.1 % (write-up 0.9670, 2.05 %);
  * swap near hit@5 % 0.907 (write-up 0.907);
  * near AUC 0.979 (write-up 0.9771, from its own script).
* **Japanese final:** Lingolix ja 0.923 [0.891, 0.951], FA@80 9.7 %, main rules 0.930 / 8.3 %.
  This matches the brief's 0.923 / 9.7 %.
* **Japanese candidate 2** (run 1): 0.917 / 13.4 %, matching its write-up.
* **es (rollout):** Lingolix es 0.985, FA@80 1.4 % (main 0.979 / 2.4 %).
* **Baselines:** `main_baseline.sh` re-ran main c9773cc on Lingolix and gave every branch's
  "before": de 0.940, ja 0.846, ko 0.912.

**Nothing failed to reproduce.** Two notes on data that does not reproduce bit for bit:
1. **Contrast-audit edit plans for de and ja were regenerated.** The 2026-11-17 plans index sounds
   in the old references. Under round 19's text front-ends:
   * 49 of 2,073 German sites point to a different sound;
   * 131 of 1,200 Japanese sites point to a different sound (`contrasts.apply` does not check).

   So de and ja got new plans with the same seeded rule. Every other language reuses the old plans:
   0 mismatches, except pt and it, where the 120 non-substitution sites are not indexed by sound,
   the same as on main. The de and ja audit rows therefore use different edits from the old table.
   Their before / after is not paired.
2. **The committed main tables for resampled audio** (cv-real, OMPAL, Lingolix) came from the
   11-17 package. Re-running main c9773cc on them (`bench/round19_merge/main_baseline.sh`)
   reproduced every value to ≤ 1e-4, and p-values to ≤ 0.004. So the before / after below is
   round 19 alone, not drift.

## E. Scoreboard: before (main, 2026-11-17 run) → after (this run)

`results/SCOREBOARD.md` was regenerated from run 2. The full row-by-row diff (170 rows, made by
`bench/round19_merge/scoreboard_diff.py`) is in
`$PRONUNCIADA_CACHE/round19/integrator/sb_diff.txt`.

### E.1 Tally against Azure (84 comparisons, each by its own table's paired test)

| | significantly ahead | ahead n.s. | level | behind n.s. | significantly behind |
|---|---|---|---|---|---|
| before | 40 | 20 | 0 | 17 | 7 |
| **after** | **49** | 16 | 1 | 15 | **3** |

By area, after (before in brackets), as sig. ahead / not different / sig. behind:

| area | after | before |
|---|---|---|
| speechocean762 | 4 / 2 / 0 | 2 / 3 / 1 |
| learner errors | 2 / 0 / 0 | 2 / 0 / 0 |
| coaching | 4 / 0 / 0 | 4 / 0 / 0 |
| OMPAL | 5 / 1 / 0 | 5 / 1 / 0 |
| tone swaps | 0 / 2 / 1 | 0 / 3 / 0 |
| swap benchmark | 32 / 17 / 1 | 25 / 21 / 4 |
| cv-real A | 2 / 9 / 0 | 2 / 8 / 1 |
| cv-real B | 0 / 1 / 1 | 0 / 1 / 1 |

Still significantly behind:
* zh tone swaps, "right word blamed" (new);
* en swap native false alarms;
* en cv-real B.

No longer significantly behind:
* en prosodic;
* ja, es and pt native false alarms;
* ja cv-real A.

**Classification changes, 20 rows.** 19 moved towards "ahead":
* **en, speechocean762:**
  * words: n.s. → significantly ahead;
  * fluency: n.s. → significantly ahead;
  * prosodic: significantly behind → behind n.s.
* **zh, swap benchmark:**
  * near AUC → significantly ahead;
  * far right word: behind → ahead n.s.;
  * native false alarms: behind n.s. → **significantly ahead** (0.026 vs 0.036, p = 0.006).
* **ja, swap benchmark:**
  * far right word → significantly ahead;
  * native false alarms: significantly behind → n.s.
* **es, swap benchmark:**
  * far right word → significantly ahead;
  * native false alarms: significantly behind → n.s.
* **de, swap benchmark:** far right word: behind → ahead n.s.
* **pt, swap benchmark:**
  * far right word → significantly ahead;
  * native false alarms: significantly behind → n.s.
* **ru, swap benchmark:** near AUC: behind n.s. → level.
* **ko, swap benchmark:**
  * far AUC → ahead n.s.;
  * near AUC → significantly ahead;
  * near hit → significantly ahead;
  * far right word → ahead n.s.
* **ja, cv-real A:** significantly behind → n.s.

**One row moved the other way:** zh tone swaps, "right word blamed": behind n.s. → significantly
behind.

### E.2 Rows that changed

Every changed row except the 67 contrast-audit rows and the 22 speed rows, both described below.

| section | row | before | after |
|---|---|---|---|
| speechocean762 vs experts | sounds / words | 0.685 / 0.609 | **0.700 / 0.640** (words vs Azure +0.050 [+0.011, +0.084], p = 0.02) |
| | accuracy / fluency / prosodic / total | 0.748 / 0.779 / 0.772 / 0.762 | **0.778 / 0.814 / 0.805 / 0.792** (total vs Azure −0.004 [−0.043, +0.032], p = 0.89) |
| ZIPA-alone Spearman rows (so762) | sounds / words / acc / flu / pro / total | 0.469 / 0.415 / 0.709 / 0.764 / 0.752 / 0.734 | 0.471 / 0.420 / 0.748 / 0.808 / 0.795 / 0.767 |
| OMPAL word AUC | any / tone-only / consonant-vowel | 0.792 / 0.778 / 0.856 | **0.787 / 0.772 / 0.854** (still ahead of Azure: +0.082, +0.093, +0.032) |
| OMPAL sentence accuracy (learners) | Pearson | 0.634 [0.468, 0.742] | 0.634 [0.467, 0.742] |
| zh tone swaps (FLEURS test) | AUC / caught at 5 % / right word | 0.834 / 0.367 / 0.300 | **0.831 / 0.333 / 0.280** |
| swap benchmark | zh near AUC / near hit / native FA | 0.975 / 0.900 / 0.040 | 0.980 / 0.927 / 0.026 |
| | ja far / near / near hit / far right word / native FA | 0.991 / 0.968 / 0.840 / 0.926 / 0.033 | 0.996 / 0.976 / 0.887 / 0.966 / 0.027 |
| | es near / near hit / far right word / native FA | 0.990 / 0.967 / 0.953 / 0.025 | 0.994 / 0.993 / 0.980 / 0.023 |
| | de near / near hit / far right word / native FA | 0.960 / 0.767 / 0.913 / 0.046 | 0.979 / 0.907 / 0.940 / 0.040 |
| | pt near / near hit / far right word / native FA | 0.963 / 0.853 / 0.940 / 0.028 | 0.967 / 0.887 / 0.960 / 0.024 |
| | ru near / near hit / far right word / native FA | 0.900 / 0.620 / 0.913 / 0.052 | 0.909 / 0.667 / 0.933 / 0.045 |
| | ko far / near / near hit / far right word / native FA | 0.994 / 0.944 / 0.753 / 0.960 / 0.023 | 0.996 / 0.974 / 0.867 / 0.980 / 0.017 |
| cv-real A | ja / de | 0.798 / 0.823 | 0.803 / 0.829 |
| cv-real B | de | 0.541 [0.423, 0.649] | **0.534 [0.420, 0.644]** |
| natives accepted, words (FLEURS test) | zh / de / pt / ru / ko | 97.1 / 96.5 / 97.6 / 96.9 / 97.7 % | 97.9 / 96.9 / 97.7 / 97.3 / 98.1 % |
| natives accepted, sounds | de | 99.6 % | **99.5 %** |
| planted word-error AUC | zh / ja / es / de / pt / ru / ko | 0.945 / 0.932 / 0.973 / 0.949 / 0.944 / 0.930 / 0.937 | 0.946 / 0.952 / 0.974 / 0.954 / **0.943** / 0.932 / 0.940 |
| planted sound-error AUC | ja / de | 0.972 / 0.974 | 0.973 / 0.975 |
| Japanese length, word caught | long → short / short → long / geminate → single / single → geminate / all | 60 / 80 / 8 / 42 / 54 % | **44** / 86 / 34 / 77 / 65 % (sound caught 97 → **96** % for long → short; n grew 1,701 → 2,136 judged sounds) |
| coaching tips on natives, "after" (share of clips with a tip) | zh / ja / es / de / ru / ko | 9 / 28 / 19 / 22 / 43 / 23 % | 8 / 23 / **22** / 17 / 40 / **24** % |

**Contrast audit, FLEURS test, word hit@5 %.** 67 rows changed; 59 went up. Largest gains:
* de voicing: 13 → 71 % and 25 → 76 %;
* de s → ʃ: 25 → 92 %;
* ja ʃ → s: 30 → 91 %;
* ko n → ŋ: 31 → 82 %;
* ru s → ʂ: 53 → 92 %.

Rows that went down are in E.3.

**Not on the scoreboard:**
* **Lingolix test 1** (`third_party_lingolix.json`), AUC / FA@80:

  | lang | before | after |
  |---|---|---|
  | es | 0.979 / 2.4 % | **0.985 / 1.4 %** |
  | de | 0.940 / 6.7 % | **0.967 / 2.1 %** |
  | ja | 0.846 / 20.1 % | **0.923 / 9.7 %** |
  | ko | 0.912 / 8.4 % | **0.957 / 3.1 %** |
  | en, fr | – | unchanged |

  Test 2: utterance PCC 0.763 → 0.788, word PCC 0.581 unchanged.
* **Learner suite** (`learners_suite.json`):
  * LS 0.714 → 0.721; against Azure +0.051 [+0.017, +0.085], p = 0.002;
  * LS-NL 0.771 → 0.829, +0.068, p = 0.004;
  * Speech Accent Archive native vs learner, C 0.678 → 0.851;
  * the speechocean762 C is now level with Azure: 0.855 vs 0.855.
* **Coaching tables** (azure yardstick, 12b, round 4): unchanged byte for byte.
* **Speed rows:** these are timings under machine load, not a comparison, so they were not read
  as gains or losses. The English speed block is the English branch's own measurement, carried in
  `english_vs_azure.json`. Latency was measured in part F.

### E.3 Rows that got worse

| row | before → after | interval / p | cause |
|---|---|---|---|
| **zh tone swaps, right word blamed** | 0.300 → 0.280 | vs Azure −0.107 [−0.207, −0.023], p = 0.02 (now significantly behind) | Mandarin identity step (rollout) |
| zh tone swaps, caught at 5 % | 0.367 → 0.333 | vs Azure −0.069 [−0.219, +0.060], p = 0.30 | same |
| zh tone swaps, AUC | 0.834 → 0.831 | vs Azure −0.015 [−0.062, +0.027], p = 0.51 | same |
| zh contrast audit, dictionary tone → other tone, word hit@5 % | 40 % → 38 % | paired Δ −0.028 [−0.071, +0.005], p = 0.31 | same |
| zh audit aspirated → plain / ŋ → n / ʐ → l | 68 → 66 / 39 → 37 / 95 → 94 % | paired p = 0.44 / 0.58 / 1.0 | same |
| OMPAL word AUC, any / tone-only / consonant-vowel | 0.792 → 0.787 / 0.778 → 0.772 / 0.856 → 0.854 | CIs [0.746, 0.833] / [0.726, 0.824] / [0.802, 0.893]; still significantly ahead of Azure on the first two; paired before / after not computed | same; main re-run reproduced the old values, so not drift |
| **pt audit nasal → oral, word hit@5 %** | 26 % → 21 % | paired Δ −0.052 [−0.124, −0.008], **p = 0.04** | Portuguese identity step |
| pt planted word-error AUC | 0.944 → 0.943 | n.s. | same |
| ru audit soft → hard, word hit@5 % | 11 % → 11 % (0.114 → 0.105) | paired p = 0.88 | Russian identity step |
| de natives, sounds accepted | 99.6 → 99.5 % | – | German text changes (new references) |
| de audit voiceless → voiced, **sound** hit@5 % | 99 → 98 % | new edit plan, not paired | German |
| de cv-real B (self-declared accents) | 0.541 → 0.534 | [0.420, 0.644]; vs Azure −0.097, p = 0.13 | German |
| **de planted stress-error word AUC** (FLEURS test, languages.md §5; not on the scoreboard) | 0.932 → 0.919; hit@5 % 0.600 → 0.546 | – | the known cost of the stress-reliability factor (round19_german.md §9.1) |
| **ja length: long vowel → short, words caught** | 60 % → 44 % (sounds 97 → 96 %) | n 585 → 620 | known; round19_japanese.md |
| ja native sentences accepted at strictness 1 | 0.838 → 0.833 | – | Japanese |
| ko lowest-word share (languages.md) | 0.839 → 0.833 | – | Korean |
| es word hit@5 % (languages.md) | 0.877 → 0.876 | – | Spanish |
| coaching tips on natives, after the gate: es / ko | 19 → 22 % / 23 → 24 % of clips | no interval in the table | identity steps (es); Korean mispronounced-word flags (ko "before" 52 → 56 %) |

**How the paired audit p-values were made.** They come from `bench/round19_merge/audit_paired.py`:
* a clip bootstrap over the identical edits of both runs, 2,000 replicates;
* each run's own 5 % native threshold.

Its point estimates differ from the scoreboard's by at most 1 point, because only edits present in
both runs are counted.

The de and ja audit rows use new edit plans (see D), so they have no paired p.

## F. Package size and latency

### F1. Size (`pronunciada/data`)

* main c9773cc: 20.67 MB.
* Fully merged: 36.39 MB (+15.72 MB).
* The backbone files are unchanged. No language adds more than 10 MB.

| language | files added | bytes |
|---|---|---|
| en | sentence_r19.npz, seqhead_en.onnx (2.45 MB), seqhead_en.npz | 2,732,383 |
| ko | coach/ko_identity.npz, calib/ko_identity_word.json | 1,985,049 |
| de | coach/de_identity.npz, de_identity_word.json, stress_rel/de.json, word_lang/de.json | 1,983,036 |
| ja | coach/ja_identity.npz, length/ja_r19.json (100 kB), word_step/ja.json | 1,096,791 |
| es | coach/identity/es.npz, calib/identity/es.json | 1,975,433 |
| pt | same, pt | 1,982,173 |
| ru | same, ru | 1,983,748 |
| zh | same, zh | 1,984,750 |
| shared | config.json | +1,048 |

`bench.ondevice` counts the model plus package data as "weights on disk":
* int8: 91.3 → 107.1 MB;
* fp32: 280.9 → 296.7 MB.

### F2. Latency: old (main c9773cc) vs new (merged), interleaved

**Tool:** `bench/round19_merge/latency.py`.
* Two warm worker processes, one per source tree, 4 threads.
* Each clip is sent to both, one at a time, with the order alternating.
* Waveforms are already in memory, and the posterior cache is cleared before each call.
* 100 clips each; median and paired median difference with a clip-bootstrap 95 % interval.
* Load average 3.4–4.9 throughout.

| lang | clips | precision | median clip (s) | old ms | new ms | paired Δ ms [95 %] | peak RSS MB old → new | encoder outputs read (new) |
|---|---|---|---|---|---|---|---|---|
| en | so762 train | fp32 | 3.6 | 54.4 | 55.6 | +1.6 [+1.4, +1.7] | 588 → 594 | enc0–enc5, final |
| en | so762 train | int8 | 3.6 | 42.0 | 43.3 | +1.3 [+1.2, +1.4] | 412 → 419 | enc0–enc5, final |
| en | FLEURS val | fp32 | 9.5 | 138.1 | 142.0 | +4.0 [+3.7, +4.3] | 851 → 858 | enc0–enc5, final |
| en | FLEURS val | int8 | 9.5 | 108.0 | 112.8 | +3.8 [+3.4, +4.2] | 666 → 678 | enc0–enc5, final |
| ko | FLEURS val | fp32 | 12.3 | 154.8 | 163.5 | **+10.5** [+8.8, +11.7] | 969 → 941 | enc1, enc3, enc5, final |
| ko | FLEURS val | int8 | 12.3 | 116.6 | 128.7 | **+10.9** [+9.3, +12.6] | 846 → 810 | " |
| de | FLEURS val | fp32 | 11.0 | 152.8 | 159.3 | +1.8 [+1.6, +2.2] | 1150 → 1144 | " |
| de | FLEURS val | int8 | 11.0 | 126.3 | 132.3 | +2.2 [+1.9, +2.7] | 1061 → 1102 | " |
| ja | FLEURS val | fp32 | 12.4 | 151.0 | 162.4 | **+14.4** [+12.0, +15.8] | 1111 → 1130 | " |
| ja | FLEURS val | int8 | 12.4 | 118.0 | 133.2 | **+14.8** [+13.1, +16.4] | 1023 → 1002 | " |
| es | FLEURS val | fp32 / int8 | 11.7 | 161.7 / 131.5 | 163.9 / 133.8 | +1.9 / +2.4 | 936 → 941 / 743 → 731 | " |
| pt | FLEURS val | fp32 / int8 | 11.2 | 161.8 / 130.4 | 160.8 / 132.9 | +1.4 / +1.5 | 957 → 964 / 790 → 789 | " |
| ru | FLEURS val | fp32 / int8 | 10.7 | 162.5 / 133.8 | 166.5 / 137.8 | +1.8 / +2.1 | 1046 → 1057 / 806 → 780 | " |
| zh | FLEURS val | fp32 / int8 | 10.5 | 145.6 / 112.3 | 143.2 / 114.8 | +0.6 / +0.9 | 959 → 977 / 763 → 811 | " |
| fr (nothing ships) | FLEURS val | fp32 / int8 | 9.4 | 113.6 / 84.6 | 115.4 / 84.5 | +0.7 / +0.1 | 726 → 724 / 591 → 591 | " |

Per-run JSON: `results/tables/round19_integration_latency/`. Peak RSS moves by ±40 MB between
otherwise identical runs, so only zh int8 (+48 MB) stands out.

**The ~5 ms budget is for a 3.5 s clip.** FLEURS clips are 3 to 3.5 times longer.
* **English on 3.6 s learner clips:** +1.3 to +1.6 ms.
* **Korean (+10.5 ms) and Japanese (+14.4 ms) on 12 s clips:** much of this scales with length.
  The extra-sound evidence (`llr_ins`) is a per-frame Python loop, and Japanese adds its length
  model and identity head. Scaled linearly, this is about 3 ms for Korean and 4 ms for Japanese at
  3.5 s. That is inside the budget, but it was **not measured on short clips** here. Both
  write-ups report +1 to +3 ms on 3.5 s clips.

### F3. `bench.ondevice` (round 19 vs 1.0.2 in the same session)

* 200 speechocean762 test clips, English.
* Each pair was run back to back through the lock, at 1-minute load average 2.5–4.1.
* Tables: `results/tables/ondevice_round19/`. The `pronunciada-1.0.2-same-session-*` files are
  the 1.0.2 runs. The committed `results/tables/ondevice/` was left unchanged; it was measured at
  load 8–10.

| system | threads | p50 ms 1.0.2 → round 19 | p95 ms | peak RSS MB | weights MB |
|---|---|---|---|---|---|
| int8 | 1 | 64.5 → 66.3 | 93.3 → 97.4 | 358 → 360 | 91.3 → 107.1 |
| int8 | 4 | 38.6 → 42.2 | 51.9 → 57.7 | 350 → 360 | 91.3 → 107.1 |
| fp32 | 1 | 97.5 → 97.2 | 141.2 → 142.3 | 518 → 538 | 280.9 → 296.7 |
| fp32 | 4 | 50.0 → 49.6 | 69.7 → 67.9 | 518 → 524 | 280.9 → 296.7 |

Load and cold start: 0.52–0.61 s, and 0.58–0.74 s first result. These are single unpaired runs
under shared load; F2 is the paired measurement.

## Rivals: OpenPronounce 1.1.0 on the round-19 outputs

**Method:**
* `bench.rivals.eval_so762`, `eval_shared` and `eval_lingolix` were re-run with the round-19
  frozen outputs (run 2) for Pronunciada and the **cached** OpenPronounce raw outputs. No
  OpenPronounce model was run.
* Results are in **new files**:
  * `results/tables/rivals_so762_round19.json` (default mode);
  * `rivals_shared_fast_round19.json` (all shared rows);
  * `rivals_lingolix_round19.json`.
* The 1.0.2 tables are untouched.
* Mode: OpenPronounce was run on L2-ARCTIC, EpaDB and Lingolix only in its fast (and student)
  mode, so those rows compare with fast.

Table from `bench/round19_merge/rivals_summary.py`; 1,000 speaker resamples (Lingolix test 1:
utterance resamples).

| test | Pronunciada 1.0.2 | Pronunciada round 19 | OpenPronounce 1.1.0 | round 19 − OpenPronounce [95 % CI] | p |
|---|---|---|---|---|---|
| so762, everyone: sentence total | 0.762 | 0.792 | 0.790 | +0.001 [−0.045, +0.040] | 0.89 |
| so762, everyone: accuracy | 0.748 | 0.778 | 0.744 | +0.034 [+0.009, +0.061] | 0.012 |
| so762, everyone: fluency | 0.779 | 0.814 | 0.827 | −0.012 [−0.087, +0.045] | 0.76 |
| so762, everyone: prosodic | 0.772 | 0.805 | 0.808 | −0.003 [−0.076, +0.054] | 0.97 |
| so762, everyone: word Pearson | 0.609 | 0.640 | 0.569 | +0.071 [+0.048, +0.097] | 0.002 |
| so762, everyone: sound Pearson | 0.685 | 0.700 | 0.534 | +0.165 [+0.135, +0.188] | 0.002 |
| so762, adults 18+: total / accuracy / fluency / prosodic | 0.844 / 0.823 / 0.868 / 0.848 | 0.863 / 0.839 / 0.889 / 0.873 | 0.850 / 0.820 / 0.875 / 0.850 | +0.013 (p 0.06) / +0.019 (0.014) / +0.014 (0.054) / +0.023 (0.010) | |
| so762, adults 18+: word / sound | 0.656 / 0.714 | 0.676 / 0.725 | 0.606 / 0.571 | +0.070 / +0.154 | 0.002 / 0.002 |
| so762, children ≤ 12: total | 0.558 | 0.614 | 0.670 | −0.056 [−0.148, +0.089] | 0.68 |
| so762, children ≤ 12: accuracy / fluency / prosodic | 0.540 / 0.553 / 0.573 | 0.599 / 0.633 / 0.640 | 0.580 / 0.745 / 0.736 | +0.019 (0.59) / −0.111 (0.58) / −0.096 (0.65) | |
| so762, children ≤ 12: word / sound | 0.512 / 0.640 | 0.561 / 0.660 | 0.502 / 0.488 | +0.059 / +0.172 | 0.002 / 0.002 |
| so762, teenagers 13–17: total / accuracy / fluency / prosodic | 0.453 / 0.430 / 0.422 / 0.459 | 0.466 / 0.483 / 0.435 / 0.423 | 0.475 / 0.428 / 0.434 / 0.470 | −0.010 (0.77) / +0.055 (0.09) / +0.001 (0.99) / −0.048 (0.21) | |
| so762, teenagers: word / sound | 0.400 / 0.462 | 0.411 / 0.474 | 0.236 / 0.384 | +0.175 / +0.090 | 0.002 / 0.002 |
| L2-ARCTIC test, error AUC (OP fast) | 0.712 | 0.712 | 0.743 | −0.031 [−0.056, −0.012] | 0.002 |
| EpaDB test, error AUC (OP fast) | 0.721 | 0.721 | 0.741 | −0.019 [−0.031, −0.009] | 0.006 |
| Lingolix 1 en: AUC / FA@80 (OP fast) | 0.977 / 0.025 | 0.977 / 0.025 | 0.886 / 0.173 | +0.090 / −0.148 | 0.002 / 0.002 |
| Lingolix 1 fr: AUC / FA@80 | 0.960 / 0.017 | 0.960 / 0.017 | 0.877 / 0.143 | +0.083 / −0.127 | 0.002 / 0.002 |
| Lingolix 1 es: AUC / FA@80 | 0.980 / 0.024 | 0.985 / 0.014 | 0.929 / 0.090 | +0.056 / −0.076 | 0.002 / 0.002 |
| Lingolix 1 de: AUC / FA@80 | 0.941 / 0.067 | 0.967 / 0.021 | 0.887 / 0.149 | +0.080 / −0.128 | 0.002 / 0.002 |
| Lingolix 2 (so762 500 clips): utterance / word PCC | 0.763 / 0.581 | 0.788 / 0.581 | 0.819 / 0.610 | −0.032 [−0.095, +0.025] / −0.029 [−0.092, +0.030] | 0.36 / 0.39 |

## Which rows come from which run

* `SCOREBOARD.md` and every table regenerated in D: run 2 (fully merged code). Run 1's tables are
  in commit d4c2aac for reference.
* The baseline columns of E (before) are the committed main tables (2026-11-17). For cv-real,
  OMPAL and Lingolix, the re-run of main c9773cc (`main_baseline.sh`) confirmed them.
* Rivals: run 2 for Pronunciada, cached OpenPronounce outputs.
* F: separate latency and on-device runs after the frozen run, not during scoring.

## Not done / open

* **Latency on 3.5 s clips for ko and ja** was not measured paired here. On 11–12 s clips they add
  10–15 ms; the `llr_ins` Python loop is the main candidate to vectorise.
* **The de and ja audit rows use regenerated edit plans,** so their before / after is not paired
  (D, note 1).
* **Not edited, as briefed:** README.md, the site, results/industry_snapshot.md and version
  numbers. The learner table still prints "Pronunciada 1.0.2", because the version was not bumped.
* **`results/README.md` still describes the 2026-11-17 run.** The new header of
  SCOREBOARD.md names this run.

## Merging later work

* **Japanese:** c5eb79c is merged. For a later Japanese commit:
  1. `git merge <commit>` on `round19`.
  2. Rerun `run_equiv2.sh` with fresh `src/ja_final` and `src/merged2` extracts, then
     `compare_all2.py`. The ja row must be 0 against the branch and, with switches off, against
     main; the other nine must be unchanged.
  3. Move ja's outputs aside in the overlay, as was done for run 2: the `fleurs-test-ja` and
     `fleurs-swap-ja` pickles, the audit ja chunks and `tier1_ja` responses. Then rerun
     `frozen_run.sh` and the scoreboard.
* **The same recipe** applies to any further rollout language.
