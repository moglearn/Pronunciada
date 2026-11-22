# Round 20 release check: the scoreboard now describes `main`

`main` at d7434aa holds round 19 plus two round-20 runtime changes:
* the Mandarin identity word step scaled by the word's tone factor (`PRONUNCIADA_IDENTITY_ZH_R20`,
  on by default; results/round20_mandarin.md);
* a speed-up of the Korean and Japanese paths with identical outputs (results/round20_latency.md).

Before this check, `SCOREBOARD.md` and `tables/*.json` still held the round-19 Mandarin numbers.
This check makes every row describe `main`, without re-running anything that `main` cannot change.
No scoring code, threshold, data file, version number, README.md, site page or
industry_snapshot.md was touched. Nothing was tuned.

**Short version.**
* **Nine languages are bitwise identical** to the code of the round-19 frozen run (836dba6), fp32
  and int8. Their frozen-run rows stand.
* **Every Mandarin row was re-run on `main`.** Every re-run number equals the "new" arm of
  results/round20_mandarin.md, to the last printed digit.
* **Tally against Azure, 84 comparisons** (significantly ahead / not different / significantly
  behind):

  | | sig. ahead | not different | sig. behind |
  |---|---|---|---|
  | 1.0.2 (2026-11-17 run) | 40 | 37 | 7 |
  | round 19 (frozen run 2) | 49 | 32 | 3 |
  | **main (this check)** | **48** | **34** | **2** |

  Two rows changed classification: zh tone swaps "right word blamed" is no longer significantly
  behind; zh swap-benchmark near AUC dropped from significantly ahead to ahead, n.s. (p = 0.05).
* **Size:** package data 36.39 MB, against 20.67 MB for 1.0.2. That is 55 bytes more than round 19.
* **On-device:** int8 at 4 threads, p50 41.2 ms (1.0.2) → 43.6 ms (`main`), same session, heavy
  machine load.

## 1. Equivalence: `main` = the frozen-run code, nine languages

**Tool:** `bench/round20_refresh/run_equiv.sh`, using `bench/round19_merge/equiv.py` and its
standard clip sets:
* 100 FLEURS **validation** clips per language (seed 19), each scored with its own text and with
  one word changed (200 scores);
* for English, also 100 speechocean762 **train** clips.

**Values compared:** every field of `SentenceResult.to_dict()` except timing, rescoring at
strictness 0 and 1, `details()`, `coaching()` and the Azure layout. **Tolerance 0.**

**Setup:**
* `main` was dumped from a `git archive` extract of d7434aa.
* It was compared with the round-19 merge check's dumps of the frozen-run code (836dba6,
  `merged2_*`).
* The same environment was used: Python 3.12, numpy 2.5.5, onnxruntime 1.32.0. A fresh dump of
  836dba6 would have been made if anything differed; it was not needed.

| language | set | scores | values compared, fp32 / int8 | max \|Δ\| fp32 / int8 |
|---|---|---|---|---|
| en | FLEURS val + so762 train | 200 + 100 | 760,912 / 757,904 | 0 / 0 |
| ko | FLEURS val | 200 | 735,905 / 739,026 | 0 / 0 |
| de | FLEURS val | 200 | 731,718 / 730,613 | 0 / 0 |
| ja | FLEURS val | 200 | 830,712 / 832,432 | 0 / 0 |
| es | FLEURS val | 200 | 858,504 / 858,586 | 0 / 0 |
| fr | FLEURS val | 200 | 662,690 / 661,918 | 0 / 0 |
| it | FLEURS val | 200 | 856,636 / 854,839 | 0 / 0 |
| pt | FLEURS val | 200 | 768,134 / 767,721 | 0 / 0 |
| ru | FLEURS val | 200 | 831,673 / 833,117 | 0 / 0 |
| zh (expected to differ) | FLEURS val | 200 | 602,908 / 603,633 | 20.5 / 14.3 |

**Nine of nine are identical:** no differing number and no differing string. So the frozen-run
outputs of en, ko, de, ja, es, fr, it, pt and ru in the round-19 overlay cache stay valid.

**Mandarin differs only where the round-20 rule acts:**
* word P (`words[].prob`, and at strictness 0 / 1);
* the Azure-layout word AccuracyScore (0–100 scale, hence the 20.5);
* 4 (fp32) / 6 (int8) coaching-item probabilities.

Sentence P, the 0–10 scores, and the sound and tone probabilities are identical. This matches
§7 of the Mandarin write-up.

Table: `tables/round20_release_equiv.json`.

## 2. The Mandarin re-run

**Driver:** `bench/round20_refresh/zh_run.sh`. It runs the frozen-run steps that produce Mandarin
rows, with the same settings.

**Overlay.** Work was done in a copy of the round-19 overlay cache under
`$PRONUNCIADA_CACHE/round20/refresh/frozen_cache`. The round-19 overlay was left untouched:
no file in it is newer than the start of the run.
* In the copy, these Mandarin outputs were moved aside:
  * the `fleurs-test-zh` and `fleurs-swap-zh` pickles;
  * the zh contrast-audit chunks (the edit plan is kept, so the edits are the same);
  * the OMPAL-test coaching dump.
* The fingerprint-keyed caches (cv-real, learner suite) were seeded under `main`'s fingerprint
  (dbcbc984e4d6da87) with the frozen run's outputs for every language except Mandarin
  (`bench/round20_refresh/seed.py`):
  * cv-real: 3,610 clips copied, 786 zh clips scored fresh;
  * learner suite: 3,300 copied, 745 zh clips scored fresh.
* OMPAL (all 1,850 utterances) and the coaching-tips table, which has no cache, were scored in full.

**Isolation and Azure:**
* Every job ran with `AZURE_SPEECH_KEY`, `_REGION` and `_ENDPOINT` removed. No file under the
  Azure cache is newer than the start of the run.
* Scoring went through `python -m bench.heavy --lane cpu`.

**Steps:**
* eval_languages (run, eval, swap, report; non-zh pickles reused);
* eval_cv;
* eval_ompal;
* coach.dump_r4 for ompal-test, then azure_yardstick, heldout_12b and fit_r4 eval;
* contrast_audit run for zh, then its report;
* learners.evaluate;
* coach.tips_r19;
* the eval_languages and eval_english reports;
* eval_zipa_baseline (eval, report);
* scoreboard.

**Not re-run:** the steps that have no Mandarin rows (English scoring and dumps, the Japanese length
table, Lingolix). Total time: 30 minutes, 2026-11-22 13:25–13:55.

**Only Mandarin values moved.** A key-by-key diff of every regenerated table against the round-19
version shows:
* changes only under zh keys;
* fingerprint and time stamps;
* the zh latency column (timing under load, see below).

The coaching r4 and yardstick tables are byte-identical. The learner suite changed only its
fingerprint: its OMPAL part is sentence-level, and the Mandarin rule leaves sentence scores alone.

### Mandarin rows: round 19 → `main`, and the Mandarin job's "new" arm

| row | round 19 | **main** | round20_mandarin.md "new" | match |
|---|---|---|---|---|
| tone swaps: AUC | 0.831 | **0.833** | .8334 | yes (0.83337 both) |
| tone swaps: caught at 5 % | 0.333 | **0.353** | .353 | yes |
| tone swaps: right word blamed | 0.280 | **0.307** | .307 | yes |
| right word vs Azure | −0.107 [−0.207, −0.023], p = 0.02 | **−0.080 [−0.180, +0.007], p = 0.08** | −.080 [−.180, +.007], p = .08 | yes |
| swap: near AUC (vs Azure p) | 0.980 (p = 0.03) | **0.979 (p = 0.05)** | .9794 | yes |
| swap: near hit@5 % | 0.927 | **0.913** | .9133 | yes |
| swap: far AUC / far right word | 0.992 / 0.900 | 0.992 / 0.900 | .9920 / .900 | yes |
| swap: native false alarms < 0.6 (vs Azure p) | 0.026 (p = 0.006) | **0.028 (p = 0.05)** | .0276 (p = .048) | yes |
| natives: words accepted (FLEURS test) | 97.9 % | **97.7 %** | 97.73 % | yes |
| planted word-error AUC (languages.md) | 0.9465 | 0.9455 | – | – |
| OMPAL word AUC: any / tone-only / consonant-vowel | 0.787 / 0.772 / 0.854 | **0.790 / 0.777 / 0.853** | .7900 / .7768 / .8525 | yes |
| cv-real zh-CN test A: word mean / word min AUC | 0.7496 / 0.7379 | 0.7483 / 0.7350 | .7483 / .7350 | yes |
| cv-real zh sentence AUC (scoreboard rows) | unchanged | unchanged | unchanged | yes |
| audit: tone → other tone | 38 % | **40 %** | .396 | yes |
| audit: aspirated → plain / plain → aspirated | 66 / 84 % | **60 / 80 %** | .600 / .800 | yes |
| audit: retroflex → dental / dental → retroflex | 75 / 70 % | **72 / 68 %** | .725 / .675 | yes |
| audit: n → ŋ / ŋ → n | 60 / 37 % | **57 / 32 %** | .567 / .317 | yes |
| audit: ʐ → l / m → n | 94 / 70 % | **93 / 67 %** | .933 / .667 | yes |
| coaching tips on natives, after the gate | 8 % of clips | **9 %** (0.084 → 0.088) | – | – |

**No mismatch was found.** One note on a different computation: the Mandarin job's "sound-error AUC
.9400 → .9392" comes from the rollout metrics (`bench.round19_rollout.heldout`). The languages table
has its own sound-error word AUC, which goes 0.9402 → 0.9393. That is the same direction and size,
but they are not the same computation.

Table of the scoreboard row diff (round 19 → `main`, 28 rows, all Mandarin or tally):
`tables/round20_release_scoreboard_diff.json`.

## 3. Scoreboard and tally

`SCOREBOARD.md` was regenerated. Its header now names this check, and the first paragraph of
`results/README.md` names it too.

| area | n | sig. ahead | ahead n.s. | level | behind n.s. | sig. behind |
|---|---|---|---|---|---|---|
| speechocean762 | 6 | 4 | 0 | 0 | 2 | 0 |
| learner errors | 2 | 2 | 0 | 0 | 0 | 0 |
| coaching | 4 | 4 | 0 | 0 | 0 | 0 |
| OMPAL | 6 | 5 | 1 | 0 | 0 | 0 |
| tone swaps | 3 | 0 | 0 | 0 | 3 | 0 |
| swap benchmark | 50 | 31 | 11 | 1 | 6 | 1 |
| cv-real test A | 11 | 2 | 5 | 0 | 4 | 0 |
| cv-real test B | 2 | 0 | 0 | 0 | 1 | 1 |
| **all** | 84 | **48** | 17 | 1 | 16 | **2** |

**Tally over time:** 1.0.2: 40 / 37 / 7; round 19: 49 / 32 / 3; `main`: **48 / 34 / 2**.

**Still significantly behind:**
* en swap benchmark, native false alarms;
* en cv-real B.

**Classification changes since round 19 (2):**
* **zh tone swaps, right word blamed:** significantly behind → behind, n.s. (p 0.02 → 0.08).
* **zh swap benchmark, near AUC:** significantly ahead → ahead, n.s.
  * 0.979 vs Azure 0.966, p 0.03 → 0.05.
  * In 1.0.2 this row was also ahead, n.s. (0.975, p = 0.29), so against the release it is unchanged.

**Classification changes since 1.0.2 (18):**
* the 19 round-19 moves towards "ahead" listed in results/round19_integration.md E.1, all of which
  stand, except zh near AUC (above);
* zh tone "right word blamed" is behind, n.s. in both 1.0.2 and `main`.

**Every row that got worse since round 19** (all Mandarin; none changes a classification except
near AUC above):
* swap near AUC 0.980 → 0.979;
* near hit@5 % 0.927 → 0.913;
* native false alarms 0.026 → 0.028 (still significantly ahead of Azure's 0.036);
* natives' words accepted 97.9 → 97.7 %;
* OMPAL consonant / vowel 0.854 → 0.853;
* audit word hit@5 %:
  * aspirated → plain 66 → 60 %;
  * plain → aspirated 84 → 80 %;
  * retroflex → dental 75 → 72 %;
  * dental → retroflex 70 → 68 %;
  * n → ŋ 60 → 57 % (the Mandarin job's paired p = .04);
  * ŋ → n 37 → 32 %;
  * ʐ → l 94 → 93 %;
  * m → n 70 → 67 %;
* coaching tips on natives 8 → 9 % of clips.

**Off the scoreboard:**
* planted word-error AUC 0.9465 → 0.9455;
* word hit@5 % 0.827 → 0.814;
* cv-real zh-CN word-min AUC 0.7379 → 0.7350 (the Mandarin job's p = .03).

## 4. Everything worse than 1.0.2

For the release notes. 1.0.2 = the scoreboard at `v1.0.2` (run of 2026-11-17); `main` = this
scoreboard. Built with `scoreboard_diff.py` (1.0.2 → `main`, 167 rows) and the languages table.

**On the scoreboard:**

| row | 1.0.2 → main | significance / note | cause |
|---|---|---|---|
| OMPAL word AUC, any error | 0.792 → 0.790 | still significantly ahead of Azure (+0.086); paired vs head-off −0.0020 [−0.0043, 0.0000], p = 0.05 (Mandarin job) | Mandarin identity step |
| OMPAL word AUC, tone-only | 0.778 → 0.777 | paired p = 0.18 | same |
| OMPAL word AUC, consonant / vowel | 0.856 → 0.853 | paired p = 0.08; vs Azure ahead n.s. | same |
| zh tone swaps, AUC | 0.834 → 0.833 | paired −0.0006, p = 0.72 | same |
| zh tone swaps, caught at 5 % | 0.367 → 0.353 | two items of 150, p = 0.88 | same |
| zh audit, aspirated → plain, word hit@5 % | 68 → 60 % | vs head-off p = 0.06 (Mandarin job) | same |
| zh audit, ŋ → n | 39 → 32 % | p = 0.10 | same |
| zh audit, n → ŋ | 58 → 57 % | n.s. | same |
| zh audit, ʐ → l | 95 → 93 % | n.s. | same |
| pt audit, nasal → oral, word hit@5 % | 26 → 21 % | paired p = 0.04 | Portuguese identity step (round 19) |
| ru audit, soft → hard | 11 → 10 % (0.114 → 0.105) | p = 0.88 | Russian identity step |
| de audit, voiceless → voiced, **sound** hit@5 % | 99 → 98 % | new edit plan, not paired (word hit 25 → 76 %) | German round 19 |
| de natives, sounds accepted | 99.6 → 99.5 % | – | German text changes |
| pt planted word-error AUC | 0.944 → 0.943 | n.s. | Portuguese identity step |
| de cv-real B | 0.541 → 0.534 | vs Azure −0.097, p = 0.13; vs the ZIPA baseline now −0.005 (p = 0.96) | German round 19 |
| ja length, long vowel → short, words caught | 60 → 44 % (sounds 97 → 96 %) | n 585 → 620 | known cost, round19_japanese.md |
| coaching tips on natives, after the gate: es / ko | 19 → 22 % / 23 → 24 % of clips | no interval | identity steps (es); Korean word flags (ko) |

**Mandarin rows back to the 1.0.2 level or better:**
* tone "right word blamed" 0.300 → 0.307;
* natives accepted 97.1 → 97.7 %;
* native false alarms 0.040 → 0.028;
* audit tone row 40 → 40 %;
* coaching tips on natives 9 → 9 %.

**Speed rows** are timings under different machine load, not comparisons; see §5.

**Off the scoreboard** (`languages.md`, Lingolix, the write-ups):

| row | 1.0.2 → main | note |
|---|---|---|
| de planted stress-error word AUC | 0.932 → 0.919 (hit@5 % 0.600 → 0.546) | known cost of the stress-reliability factor |
| **es planted stress-error word AUC** | **0.964 → 0.961 (hit@5 % 0.838 → 0.804)** | a round-19 change missing from round19_integration.md E.3; found in this check; cause not traced |
| it planted stress-error word AUC | 0.9391 → 0.9390 | −0.0001; it ships nothing, its outputs are identical to 1.0.2 (round 19 B / C); not traced |
| ja native sentences accepted at strictness 1 | 0.838 → 0.833 | Japanese round 19 |
| ko lowest-word share | 0.839 → 0.833 | Korean round 19 |
| es word hit@5 % | 0.877 → 0.876 | Spanish identity step |
| pt planted word-error AUC | 0.9436 → 0.9432 | as above |

Nothing on Lingolix got worse: es, de, ja and ko improved; en and fr are unchanged.

## 5. Cost

### Package data (`pronunciada/data`, from git)

| | bytes | MB |
|---|---|---|
| 1.0.2 (c9773cc) | 20,669,902 | 20.67 |
| round 19 (836dba6) | 36,394,313 | 36.39 |
| **main (d7434aa)** | **36,394,368** | **36.39** (+15.72 MB vs 1.0.2; +55 B vs round 19) |

The 55 bytes are the `"round20"` block in `calib/identity/zh.json` (1,533 → 1,588 bytes). The
backbone files are unchanged (int8 67 MB, fp32 248 MB).

| language | added over 1.0.2 | bytes |
|---|---|---|
| en | sentence_r19.npz, seqhead_en.onnx, seqhead_en.npz | 2,732,383 |
| ko | coach/ko_identity.npz, calib/ko_identity_word.json | 1,985,049 |
| de | coach/de_identity.npz, de_identity_word.json, stress_rel/de.json, word_lang/de.json | 1,983,036 |
| ja | coach/ja_identity.npz, length/ja_r19.json, word_step/ja.json | 1,096,791 |
| es | coach/identity/es.npz, calib/identity/es.json | 1,975,433 |
| pt | same, pt | 1,982,173 |
| ru | same, ru | 1,983,748 |
| zh | same, zh | 1,984,805 |
| fr, it | – | 0 |
| shared | calib/config.json | +1,048 |
| **total** | | **15,724,466** |

### `bench.ondevice`: main vs 1.0.2, same session

**Setup:**
* 200 speechocean762 test clips, English, mean 3.4 s.
* Each pair ran back to back through the lock: `bench/round20_refresh/cost_runs.sh`.
* 1.0.2 = an eaa4080 extract, whose `pronunciada/` equals c9773cc.
* Tables: `tables/ondevice_round20/`. The `pronunciada-1.0.2-same-session-*` files are the 1.0.2
  runs. The committed `tables/ondevice/` is unchanged.

**Machine load was high:** 1-minute load average 12.4–15.7. Other jobs were running, and Spotlight
indexing was at about 5 cores. Round 19's runs were made at 2.5–4.1. Absolute times are therefore
not comparable with round 19's table; the side-by-side pairs are.

| system | threads | p50 ms 1.0.2 → main | p95 ms | peak RSS MB | weights MB |
|---|---|---|---|---|---|
| int8 | 1 | 67.2 → 68.2 | 95.8 → 99.7 | 355 → 356 | 91.3 → 107.1 |
| int8 | 4 | 41.2 → 43.6 | 55.4 → 58.9 | 355 → 362 | 91.3 → 107.1 |
| fp32 | 1 | 100.7 → 102.4 | 149.0 → 150.3 | 515 → 525 | 280.9 → 296.7 |
| fp32 | 4 | 52.0 → 53.3 | 71.5 → 71.9 | 521 → 540 | 280.9 → 296.7 |

Load: 0.59–0.71 s. Cold start: 0.68–0.80 s. These are single unpaired runs. On English, `main`
adds +1.0 to +2.4 ms p50 on 3.4 s clips. This matches round 19's paired +1.3 to +1.6 ms (int8 /
fp32, 4 threads), since the round-20 changes do not touch English.

### Mandarin latency (sanity check only)

**Setup:** `bench/round19_merge/latency.py`, round-19 frozen-run code (old) against `main` (new).
* 100 FLEURS validation clips (median 10.5 s), 4 threads.
* Interleaved clip by clip; load average 8–10.
* Tables: `tables/round20_release_latency_zh_{fp32,int8}.json`.

| precision | old median ms | new median ms | paired Δ ms [95 %] | peak RSS MB old → new |
|---|---|---|---|---|
| fp32 | 152.1 | 150.0 | −0.10 [−1.20, +0.52] | 995 → 991 |
| int8 | 120.1 | 122.4 | −0.52 [−1.70, +0.38] | 763 → 720 |

The Mandarin rule costs no measurable time. Korean and Japanese were not re-measured (see
results/round20_latency.md).

## 6. Tests

`python -m pytest -q tests`: **692 passed**.

## Files

* `bench/round20_refresh/`:
  * `run_equiv.sh` and `equiv_table.py` (step 1);
  * `seed.py` and `zh_run.sh` (step 2);
  * `cost_runs.sh` (step 4).
* `bench/scoreboard.py`: header text only.
* `results/SCOREBOARD.md`, `results/README.md` (first paragraph), and the regenerated tables and
  pages:
  * `languages`, `swap_ours_vs_azure`, `cv_real`, `ompal`, `ompal_tone_zh_v3z`;
  * `contrast_audit_test`, `coaching_tips_r19`, `learners_suite`, `zipa_baseline`;
  * `english_vs_azure.md` (zh speed cell);
  * `scoreboard_tally`.
* New tables:
  * `round20_release_equiv.json`;
  * `round20_release_scoreboard_diff.json`;
  * `round20_release_latency_zh_*.json`;
  * `ondevice_round20/`.
* Logs and the overlay: `$PRONUNCIADA_CACHE/round20/refresh/`. The round-19 Mandarin outputs that
  were moved aside are in `zh_r19_outputs_moved/` there.

## Not done / open

* **Learner table label.** The table still prints "Pronunciada 1.0.2", because the version was not
  bumped. The cv-real, OMPAL, learner and ZIPA-baseline tables name `main`'s package fingerprint
  dbcbc984e4d6da87.
* **The es planted stress-error drop** (0.964 → 0.961, round 19) was not traced to a cause here. It
  should be stated in the release notes with the rest of §4.
* **Speed rows** of the scoreboard (`english_vs_azure.json` zh cell) are timings from this run under
  load average up to 16. They are not a comparison.
