# Round 19 integration: merged code, equivalence checks, one frozen benchmark run

Branch `round19`, from main c9773cc. Five round-19 branches were merged in order. The
merged code reproduces every branch exactly, and so does main when every switch is off. Then one
frozen run of the full held-out suite was made on the merged code. Nothing was tuned: no threshold,
data file or parameter of any branch was touched.

## What was merged

| order | branch | commit | topic | write-up | switch (0 = old behaviour) |
|---|---|---|---|---|---|
| 1 | worktree-agent-a3633e76c1a06d754 | ac4bc76 | learner test suite (bench/learners; results only) | round19_learners.md | – |
| 2 | worktree-agent-a31e9bcfcc58e849b | b7c2850 | English sentence heads, sound / word blends, all-layer encoder outputs | round19_english.md | `PRONUNCIADA_SENTENCE_HEAD` |
| 3 | worktree-agent-a72249a2162e58229 | 1dc4f5b | Korean identity head, two-sided word step, no-raise rule, `llr_ins` | round19_korean.md | `PRONUNCIADA_KO_IDENTITY` |
| 4 | worktree-agent-a2674ec67835464ea | da9def0 | German text fixes, word recalibration, identity head, stress reliability, loanwords | round19_german.md | `PRONUNCIADA_DE_R19` (+ `_DE_IDENTITY`, `_DE_STRESS_REL`, `_DE_LOAN`, `_DE_TEXT`, `PRONUNCIADA_WORD_LANG`) |
| 5 | worktree-agent-a3e26ec0e6c91086b | 20d6260 | Japanese candidate 2: length model, identity head, capped word step | round19_japanese.md | `PRONUNCIADA_JA_ROUND19` |

Merge commits: e5bb338, 6313c53, 3c10786, b2724d4, 0d43805. Merge tooling: d313b1c and 331a58f
(`bench/round19_merge/`, plus the scoreboard header, which now names this run).

### Conflicts and how they were resolved

* **Learners, English, Korean:** no conflicts. Git merged scorer.py, delivery_coach.py and
  DESIGN.md automatically. Each hunk was read: each is guarded by its own language (`self.lang == "ko"`,
  config `sentence_head` = English only), and none touches another's lines.
* **German:**
  * `pronunciada/scorer.py`: the Korean (`cache["ko_identity_s"]`) and German
    (`cache["de_identity_s"]`) evidence blocks both inserted at the same place. Both were kept, one
    after the other; each runs only for its own language.
  * `bench/round19/latency.py`: the English and German branches each added a different file with
    this name in the shared `bench/round19/` package. The English one keeps the name. The German one
    is now `bench/round19/latency_de.py`, with its usage lines changed to match. Note that
    `round19_german.md` still calls it `bench.round19.latency`.
* **Japanese:**
  * `pronunciada/scorer.py`: the Korean word factor and the Japanese word step both inserted at
    the end of the per-word loop. Both were kept: Korean first, then Japanese. Each is gated by
    evidence that only its own language computes (`ko_identity_s`, `word_step_in`).
  * `docs/DESIGN.md` §7: the Korean and Japanese bullets were both kept.
* **"Which sound was said" heads.** The three heads stay separate, as briefed:
  * `ko_identity.py` + `data/coach/ko_identity.npz`;
  * `de_identity.py` + `de_identity.npz`;
  * `ja_identity.py` + `ja_identity.npz`.

  No code was shared or refactored.

## A. Tests

`python -m pytest -q tests` on the merged code: **605 passed** (main + 5 + 3 + 3 + 3 + 6 new test
files; no failures, no skips added).

## B. Equivalence: merged code = each branch; untouched languages = main

Tool: `bench/round19_merge/equiv.py` (dump / compare) and `run_equiv.sh`. Each side was run from a
`git archive` extract of the exact commit, imported first on `PYTHONPATH`, and the dumper checks
where `pronunciada` was imported from. The branch worktrees under `.claude/worktrees/` were not
used directly, because the Japanese worktree is still being worked on. The archive of 20d6260 is
what that worktree held at the merged commit.

**Clips (dev and validation only):**
* 100 FLEURS **validation** clips per language (seed 19). Each was scored twice: with its own text,
  and with one word replaced by the word at the same place in the next clip's text. The second
  version exercises the error paths: identity steps, word recalibration, flags and coaching.
* For English, also 100 speechocean762 **train** clips (learners).

**Values compared, per score:**
* every field of `SentenceResult.to_dict()` except `timing_ms` (words, sounds, syllables, raw
  evidence `features`, scores);
* the word and sentence probabilities rescored at strictness 0 and 1;
* `Scorer.details()` (0-2 sound, 0-10 word and sentence regressions, logits);
* `coaching()`;
* `compat.azure.to_azure()`.

Strings and flags must match exactly; numbers within 1e-6.

| check | language | clips (scores) | precision | values compared | max \|Δ\| | result |
|---|---|---|---|---|---|---|
| merged = English branch b7c2850 | en | FLEURS val 100 (200) | fp32 / int8 | 673,242 / 670,541 | 0 / 0 | identical |
| merged = English branch | en | so762 train 100 (100) | fp32 / int8 | 87,670 / 87,363 | 0 / 0 | identical |
| merged = Korean branch 1dc4f5b | ko | FLEURS val 100 (200) | fp32 / int8 | 735,905 / 739,026 | 0 / 0 | identical |
| merged = German branch da9def0 | de | FLEURS val 100 (200) | fp32 / int8 | 731,718 / 730,613 | 0 / 0 | identical |
| merged = Japanese branch 20d6260 | ja | FLEURS val 100 (200) | fp32 / int8 | 830,712 / 832,432 | 0 / 0 | identical |
| merged = main c9773cc | zh | FLEURS val 100 (200) | fp32 / int8 | 602,908 / 603,633 | 0 / 0 | identical |
| merged = main | es | " | fp32 / int8 | 858,504 / 858,586 | 0 / 0 | identical |
| merged = main | fr | " | fp32 / int8 | 662,690 / 661,918 | 0 / 0 | identical |
| merged = main | it | " | fp32 / int8 | 856,636 / 854,839 | 0 / 0 | identical |
| merged = main | pt | " | fp32 / int8 | 768,134 / 767,721 | 0 / 0 | identical |
| merged = main | ru | " | fp32 / int8 | 831,673 / 833,117 | 0 / 0 | identical |

Every difference is exactly 0, not merely below 1e-6. Table: `results/tables/round19_integration_equiv.json`.

**Do the other languages pay for the English all-layer outputs?** No.
* `AcousticModel` adds stacks 0, 2 and 4 to the graph's outputs, and writes the
  `model.emb-enc0-…-final.onnx` variant, only when a language's sentence projection or sequence
  head asks for those stacks. Only config `sentence_head.en` does.
* Every other language, and English with `PRONUNCIADA_SENTENCE_HEAD=0`, keeps the round-5 output
  set and the old file `model.emb.onnx`.
* The latency check (F) confirms it:
  * the unchanged-language workers report the same four outputs (enc1, enc3, enc5, final) on main
    and merged;
  * time and peak memory are level (table F2).

## C. Every switch off = main

The same clips were scored by the merged code with every round-19 switch at 0
(`PRONUNCIADA_SENTENCE_HEAD`, `_KO_IDENTITY`, `_DE_R19`, `_DE_IDENTITY`, `_DE_STRESS_REL`,
`_DE_LOAN`, `_DE_TEXT`, `_WORD_LANG`, `_JA_ROUND19` = 0), and the output was compared with main
c9773cc.

| language | clips | fp32 max \|Δ\| | int8 max \|Δ\| | result |
|---|---|---|---|---|
| en | FLEURS val 100 (200 scores) + so762 train 100 | 0 | 0 | identical |
| ko | FLEURS val 100 (200) | 0 | 0 | identical |
| de | FLEURS val 100 (200) | 0 | 0 | identical |
| ja | FLEURS val 100 (200) | 0 | 0 | identical |

Korean still computes the extra-sound evidence (`llr_ins`, about 1 ms) with the switch off, as its
write-up says, but no output reads it then.

With the English switch on, these are still byte-identical to main on both English sets:
* every word and sound probability;
* every coaching item.

The English change touches only the four 0-10 sentence scores and the reported 0-2 sound and 0-10
word regressions. That is why the coaching tables of the frozen run are expected to stay unchanged.

D_PLACEHOLDER

E_PLACEHOLDER

F_PLACEHOLDER

## Merging the later Japanese commit

The Japanese engineer is still working on the same branch. 20d6260 was merged as briefed. To merge
the final candidate:

1. Run `git merge <final commit>` on `round19`. Git takes only the commits after 20d6260. The
   conflict to expect is in the per-word loop of `pronunciada/scorer.py`, where Korean and
   Japanese both insert; keep both, Korean first.
2. Rerun `bash bench/round19_merge/run_equiv.sh` with a fresh `src/ja` extract of the new commit
   and a new `src/merged`, and rerun `compare_all.py`. The ja row must be exactly 0 against the
   branch and, with switches off, against main. Every other language must still be 0 against main.
3. Rerun the Japanese-affected frozen steps on the overlay with fresh output folders:
   eval_languages, eval_cv, length table, contrast audit (ja), Lingolix, tips. Then rerun the
   scoreboard.
