# Round 20 false-alarm release check: one held-out run, a fixed rule, Spanish only

The native false-alarm work (results/round20_native_fa.md) left four fixes that passed on practice data:

* number readings (C) for en, de, es, fr and ko;
* German lexicon readings (E);
* Spanish glides;
* Korean n-insertion.

This check turned them on by default, froze the code, ran the held-out benchmarks once, and applied a decision
rule written down before the run. A latency budget was added after the run, before anything was decided on
latency.

**Result.**

| language | accuracy rule | latency budget (mean ≤ 2 ms, 4 threads) | final default |
|---|---|---|---|
| en | **fail** (c): 3 audit rows significantly worse | fail (+15.3 / +11.8 ms) | off (= main) |
| de | **fail** (c): 5 audit rows significantly worse | fail (+32.3 / +32.7 ms) | off (= main) |
| fr | **fail** (c): 1 audit row significantly worse | fail (+6.2 / +6.3 ms) | off (= main) |
| ko | pass | **fail** (+4.06 / +4.84 ms) | off (= main) |
| es | pass | pass (+0.76 / +0.61 ms) | **on: number readings, glides** |

Latency figures are int8 / fp32.

* **Tally against Azure** (84 rows; significantly ahead / not different / significantly behind):

  | | ahead | level | behind |
  |---|---|---|---|
  | main | 48 | 34 | 2 |
  | frozen run, all five languages on | 49 | 33 | 2 |
  | **final (Spanish on)** | **48** | **34** | **2** |

  * In the frozen run, two rows moved up: en swap "far right word" and de swap "native false alarms". Both
    belong to languages the rule then switched off.
  * No row moved down.
  * No Spanish row changed class.
* **Scores are the fp32 Scorer's.** The benchmark scripts build the Scorer at fp32, so the scoreboard and the
  tally are fp32 numbers. The identity proofs and latency below cover int8 too.
* **Nothing was sent anywhere.**
  * The Lingolix audio-minute ledger was 258.7612 min (5,700 entries, 3,700 responses) before and after the run.
  * Azure figures come from cached responses only.
  * No file of the earlier frozen-run overlays, the Azure cache or the Lingolix caches was changed.
* Package data is identical to main. The package grows by 14,149 bytes of code. pytest: 722 passed.

## 1. The release candidate and the identity proofs

Branch `round20-fa-rc` is the experimental branch `round20-native-fa` plus a merge of main (c1792d4).

**Shipped.** The four fixes, behind `pronunciada.r20.DEFAULTS`, a table of language → fixes on by default.
Each fix is switched by its environment variable: `PRONUNCIADA_R20_NUMALT`, `_DE_LEXALT`, `_ES_GLIDE` and
`_KO_NINS`. A value of `0` switches the fix off and `1` switches it on; unset, the table decides.

**Removed from the package.**

* The unsaid-word pass (B) and the guessed-word modes (D1, D2).
* Their parameters (`data/calib/r20.json`) and their tests.

They remain documented in results/round20_native_fa.md.

**Identity.** Tolerance 0 over every output of `bench/round19_merge/equiv.py`:

* `to_dict`;
* rescoring at strictness 0 / 1;
* `details`, `coaching` and the Azure layout.

Precision fp32 and int8. Table: `tables/round20_fa_release_check_equiv.json`.

| proof | languages | sets | comparisons | max \|Δ\| |
|---|---|---|---|---|
| (a) every switch 0 == main | en de es fr ko | FLEURS val 100 × 2; 100 practice items (about half with a number); en + 100 so762 train | 22 | 0 |
| (b) defaults == experimental branch with the four switches on, B / D off | en de es fr ko | same | 22 | 0 |
| (c) defaults == main | it pt ru ja zh | FLEURS val 100 × 2 | 10 | 0 |
| final table == main | en de fr ko it pt ru ja zh | as above | 26 | 0 |
| final table == frozen release candidate | es | FLEURS val, practice | 4 | 0 |

The defaults are not a no-op: they differ from main on all five languages.

**Learner practice check** (TRAIN splits, defaults against every switch at 0):

* speechocean762: 2,500 clips, 0 changed, 0 of 15,849 words changed. The word correlation is 0.78016 both ways.
* L2-ARCTIC: 2,699 clips, 0 changed.
* EpaDB: 1,903 clips, 0 changed.

## 2. The rule (written into the status file before the run)

For each language L in {en, de, es, fr, ko}, the fixes stay ON for L only if all of these hold:

* **(a)** No row of the 84-row Azure tally belonging to L moves down a class (ahead → level or behind,
  level → behind) against main's scoreboard.
* **(b)** On Lingolix's test 1, L's AUC and false alarms at 80 % are not significantly worse than main's
  (paired resampling as the harness does it, p < 0.05). For en, the test-2 and all-2,500 sentence and word
  correlations do not drop by more than 0.002.
* **(c)** No other held-out row for L in the scoreboard gets significantly worse.

If L fails, its fixes default off. That is main, whose numbers are known; no second run is needed. No partial
switches within a language, no retuning, no re-run.

Classes: "ahead" = significantly ahead; "level" = ahead n.s., level or behind n.s.; "behind" = significantly
behind.

The operational test for (c) was also fixed before the run. A row that moves the worse way counts as
significantly worse if a paired resampling of the release candidate against main, over the row's own units,
gives p < 0.05.

**Latency budget**, added by the coordinator after the run and before any latency decision:

* A language passes if its mean added latency on natural text is ≤ 2 ms at 4 threads, int8 and fp32.
* Natural text: 200 original FLEURS validation clips, defaults against every switch at 0.
* If a language is over, the pass must first be made cheaper with bit-identical outputs. Failing that, the
  language goes off.

## 3. The run

**Frozen commit 64d165b.** Runtime identical to 8f2c0f8; package fingerprint e74b7e91357671d2.

**How it was run.**

* Run once on 2026-11-23, 00:21–01:44, from a clean extract of that commit: `bench/round20_fa_rc/heldout_run.sh`.
* It used the steps and settings of the round-19 frozen run, through `python -m bench.heavy --lane cpu`.
* Every job ran with the Azure variables and `LINGOLIX_API_KEY` removed and with the round-20 switches unset.
* It ran in a copy of the overlay that holds main's held-out outputs (`bench/round20_fa_rc/overlay.py`).

**What was scored fresh.** Main's outputs for en, de, es, fr and ko were moved aside, and all five languages
were scored fresh:

* English: speechocean762, L2-ARCTIC and EpaDB test, and the coaching dumps;
* FLEURS test and swap;
* cv-real;
* the learner suite (English, and Spanish HEROICO);
* the contrast audit of the five languages;
* Pronunciada's answers on Lingolix's test 1 (five languages), on test 2, and on all 2,500 speechocean762 test
  clips.

**What was reused.** Italian, Portuguese, Russian, Japanese and Mandarin outputs were kept or seeded. Proof (c)
shows they are identical. OMPAL (Mandarin) and the Japanese length table were not re-run.

**The Lingolix comparison** was rebuilt with `lingolix_live --report-only`, into the extract's own results/ first.

## 4. Per-language numbers of the frozen run (all five languages on)

### (a) Tally rows

| row | main | frozen run |
|---|---|---|
| en swap, far "right word" | 0.887 vs Azure 0.898, behind n.s. (p 0.64) | 0.900, ahead n.s. (p 0.97) |
| de swap, native false alarms < 0.6 | 0.040 vs 0.049, ahead n.s. (p 0.07) | 0.037, **ahead** (p 0.02) |

No other tally row changed class.

### (b) Lingolix test 1, release candidate − main

Paired, the harness's 1,000 utterance resamples (seed 0), v2026.10 rules. The benchmark's main rules give the
same signs and p within 0.1.

| lang | AUC main → rc | Δ [95 %] | p | FA@80 main → rc | Δ [95 %] | p |
|---|---|---|---|---|---|---|
| en | 0.9768 → 0.9768 | 0 | 1.0 | 0.0247 → 0.0247 | 0 | 1.0 |
| fr | 0.9600 → 0.9600 | 0 | 1.0 | 0.0165 → 0.0165 | 0 | 1.0 |
| es | 0.9854 → 0.9858 | +0.0005 [+0.0000, +0.0012] | 0.044 (better) | 0.0139 → 0.0127 | −0.0012 [−0.0037, 0] | 0.36 |
| de | 0.9670 → 0.9672 | +0.0002 [−0.0008, +0.0009] | 0.54 | 0.0205 → 0.0180 | −0.0026 [−0.0044, +0.0065] | 0.39 |
| ko | 0.9567 → 0.9567 | +0.0000 [−0.0003, +0.0004] | 0.94 | 0.0308 → 0.0301 | −0.0007 [−0.0028, +0.0015] | 1.0 |
| ja | identical answers | | | | | |

* For en and fr, Pronunciada's answers differ from main's only in the latency field.
* English test 2 (500 clips) and all 2,500 speechocean762 test clips: the word and sentence correlations are
  identical to main's:
  * test 2: word 0.5805, sentence 0.7879;
  * all 2,500: word 0.5732, sentence 0.7920.
* The English learner tables are byte-identical in their numbers: speechocean762 test, L2-ARCTIC / EpaDB learner
  errors, and the coaching yardstick.

### (c) Other scoreboard rows

**The rows that fail the rule: the contrast audit, word hit at 5 % native false alarms.** Main against release
candidate, each with its own threshold. Paired resampling of clips: 2,000 draws, natives and edits drawn together,
both thresholds recomputed in each draw (`bench/round20_fa_rc/audit_paired.py`). Sound hits did not change in
any of these rows.

| lang | contrast (said → expected) | n | word hit@5 % main → rc | Δ [95 %] | p |
|---|---|---|---|---|---|
| en | d → ð | 120 | 0.967 → 0.892 | −0.075 [−0.138, −0.023] | 0.001 |
| en | n → ŋ | 120 | 0.950 → 0.908 | −0.042 [−0.082, −0.009] | 0.018 |
| en | θ → s | 120 | 0.825 → 0.767 | −0.058 [−0.104, −0.013] | 0.019 |
| de | s → ʃ | 120 | 0.917 → 0.800 | −0.117 [−0.192, −0.059] | < 0.001 |
| de | unrounded → rounded | 120 | 0.967 → 0.908 | −0.058 [−0.108, −0.018] | 0.009 |
| de | ʃ → ç | 120 | 0.983 → 0.933 | −0.050 [−0.089, −0.010] | 0.015 |
| de | ç → ʃ | 120 | 0.958 → 0.908 | −0.050 [−0.102, −0.009] | 0.020 |
| de | n → ŋ | 120 | 0.983 → 0.942 | −0.042 [−0.080, −0.008] | 0.030 |
| fr | s → ʃ | 134 | 0.993 → 0.963 | −0.030 [−0.062, −0.007] | 0.034 |

**Why.** The audit plants an error in the reference: one sound of a word is replaced, while the word's text and
the native audio stay as they were.

* When that word is a number (en, fr) or a German lexicon word with several readings, the pass scores it again
  against its other readings.
* Those readings are built from the unchanged text. They do not contain the planted error and match the native
  audio.
* So the word is no longer flagged.

Aggregate counts, main → release candidate:

| lang | audit edits whose word P rose | of edits | native audit words whose P changed | of which numbers |
|---|---|---|---|---|
| en | 60 | 2,880 | 8 | 8 |
| fr | 19 | 1,680 | 6 | 6 |
| de | 144 | 2,073 | 46 | 10 (the rest are lexicon words) |

A learner reading a real prompt does not produce this situation in the same way. The rule is about the
scoreboard as it stands, however, and these rows are on it.

**Other rows that moved in the frozen run.**

Better:

* en native false alarms 0.034 → 0.033 (still significantly behind Azure);
* de far AUC 0.997 → 0.998; near AUC 0.979 → 0.981; far right word 0.940 → 0.947;
* ko far right word 0.980 → 0.987;
* fr native false alarms 0.027 → 0.026;
* cv-real ko test A 0.802 → 0.843 (n = 22, behind ZIPA n.s. → ahead n.s.);
* coaching tips on natives after the gate: de 17 → 16 %, ko 24 → 23 %;
* the Spanish rows in §5.

Worse, none significant:

* the other audit word-hit drops: en ɛ → æ, v → b, s → ʃ, n → m; de m → n (p 0.05), voicing, front / back;
  ko ɔ → o, ŋ → n; all p ≥ 0.05.

Full row list: `tables/round20_fa_release_check_verdict_frozen_run.json`; audit rows:
`tables/round20_fa_release_check_audit_paired.json`.

### Verdict of the rule

* **en:** fails (c).
* **de:** fails (c).
* **fr:** fails (c).
* **es:** passes (a), (b) and (c).
* **ko:** passes (a), (b) and (c).

## 5. Latency budget

**Setup.**

* Natural text: 200 original FLEURS validation clips per language.
* Defaults against every switch at 0, same tree, paired and interleaved clip by clip (`bench/round19_merge/latency.py`).
* Load average 2–5.
* Tables: `tables/round20_fa_release_check_latency_natural.json`, and for the worst-case practice items
  `tables/round20_fa_release_check_latency_practice.json`.

Added ms per clip, mean [95 %] / median / p95 of the paired difference:

| lang | int8, 4 threads | fp32, 4 threads | int8, 1 thread | fp32, 1 thread |
|---|---|---|---|---|
| en | 15.3 [5.6, 27.5] / 0.09 / 59.8 | 11.8 [5.8, 18.9] / 0.17 / 116.2 | 13.3 / 0.49 / 59.3 | 11.7 / 0.27 / 73.0 |
| de | 32.3 [17.1, 53.9] / 0.39 / 119.3 | 32.7 [17.1, 55.1] / 0.38 / 124.9 | 33.4 / 0.49 / 122.7 | 34.2 / 1.0 / 130.7 |
| **es** | **0.76** [0.21, 1.58] / 0.25 / 2.4 | **0.61** [0.06, 1.40] / 0.29 / 1.8 | 0.62 / 0.24 / 3.0 | 0.16 / −0.04 / 2.3 |
| fr | 6.2 [0.7, 16.3] / −0.06 / 3.8 | 6.3 [0.7, 16.2] / −0.14 / 2.4 | 6.5 / 0.25 / 3.6 | 6.2 / −0.25 / 3.5 |
| ko | 4.06 [0.30, 8.60] / −0.30 / 23.9 | 4.84 [1.84, 8.97] / 0.06 / 24.5 | 5.4 / 0.76 / 6.2 | 5.8 / 0.74 / 25.4 |

**How often the pass re-scores** (int8; fp32 within one clip):

| lang | clips re-scored | number tokens | numbers below the 0.9 gate | numbers re-scored | de multi-reading words re-scored |
|---|---|---|---|---|---|
| en | 16 / 200 (8 %) | 64 | 27 | 19 | – |
| de | 49 / 200 (25 %) | 40 | 21 | 10 | 50 |
| es | 1 / 200 (0.5 %) | 44 | 20 | 1 | – |
| fr | 8 / 200 (4 %) | 57 | 13 | 12 | – |
| ko | 10 / 200 (5 %) | 96 | 8 | 14 | – |

Spanish rarely re-scores: of 20 Spanish numbers below the gate, 19 had no reading other than the canonical one
(identical readings are skipped).

**Korean: making it cheaper.** One Korean re-score costs 73.7 ms:

* align 58.7 ms;
* features 8.8 ms;
* calibration 4.0 ms.

The cost is the re-alignment of the whole clip against the edited reference. The alignment band and the
forward-backward sums depend on the whole lattice. Scoring only the word's span from the cached posteriors
therefore gives different numbers, not identical ones.

Identical readings are already skipped. A work cap changes outputs whenever it binds. No output-identical change
came close to the factor of 2 the budget needs, so **ko goes off** under the budget rule.

English, German and French also exceed the budget, but they were already off under the accuracy rule.

**Spanish costs**, as a mean on natural text:

* +0.76 ms int8 and +0.61 ms fp32 at 4 threads;
* +0.62 / +0.16 ms at 1 thread.

## 6. Final per-language table and its numbers

`pronunciada.r20.DEFAULTS` (commit 7dd16c6): es: number readings, glides; en, de, fr, ko: none.

**Proofs.**

* final == main for every other language, and final Spanish == frozen release candidate (§1, 32 / 32,
  max |Δ| 0);
* tests: 722 passed;
* final package fingerprint fe95440dbacec9ac.

**How the tables were rebuilt.** No scoring was repeated:

* An overlay was built from the per-clip outputs that already existed (`bench/round20_fa_rc/final_overlay.py`):
  main's for en, de, fr and ko, the frozen run's for es.
* Only the report and evaluation steps ran (`bench/round20_fa_rc/final_tables.sh`). Their log shows no clip
  scored.
* The coaching-tips table has no per-clip cache. It is main's table with the frozen run's Spanish natives entry
  (`compose_tips.py`).

**Check of the rebuilt tables.** A key-by-key diff of every table against main
(`tables/round20_fa_release_check_table_diff_final.json`) shows only:

* Spanish values;
* the OMPAL cache name (fingerprint);
* NaN = NaN entries.

The scoreboard body regenerated in the repository equals the one from the rebuild.

**Spanish, main → final** (every changed scoreboard row):

| row | main | final |
|---|---|---|
| swap far AUC (vs Azure) | 0.998, p 0.008 | 0.999, p 0.008 |
| swap native false alarms < 0.6 | 0.023 (behind n.s., p 0.06) | 0.021 (behind n.s., p 0.18) |
| swap near AUC vs ZIPA | +0.078 | +0.079 |
| natives: words accepted (FLEURS test) | 98.4 % | 98.6 % |
| planted word-error AUC (natives table) | 0.974 | 0.975 |
| cv-real test A | 0.766 [0.718, 0.811] | 0.768 [0.722, 0.813] (vs Azure +0.043, p 0.13) |
| coaching tips on natives after the gate | 22 % (0.28 per clip) | 18 % (0.24) |
| Lingolix test 1 AUC / FA@80 | 0.9854 / 0.0139 | 0.9858 / 0.0127 |
| audit trill → tap, sound hit | 72 % | 71 % (p 1.0) |
| audit voiced → voiceless / voiceless → voiced, word hit | 100 / 99 % | 99 / 98 % (p 0.71 / 0.70) |
| audit m → n, word hit | 100 % | 98 % (p 0.32) |
| audit tap → trill, word hit | 91 % | 92 % |

**Everything Spanish that got worse**, none significant:

* the four audit cells above;
* swap near "hit at 80 %" 0.993 → 0.987;
* cv-real test A fluency-score AUC (secondary) 0.710 → 0.708;
* native median word P 0.9919 → 0.9917.

Off the scoreboard, Spanish improved: planted word AUC 0.9737 → 0.9746 and word hit@5 % 0.877 → 0.881 (FLEURS
test natives); HEROICO learner test C 0.618 → 0.618 (+0.0007).

**The final tally** is 48 / 34 / 2, as on main. The two rows that moved up in the frozen run belong to
languages that are now off.

## 7. Cost

* **Package data:** 36,394,368 bytes, as on main (the round-20 calibration file was removed).
* **Code:** +14,149 bytes.
* **Latency:** §5. Spanish adds a mean of under 1 ms per clip on natural text.
* On the practice items, where half the prompts carry a number, the worst case is: es +1.0 / +0.1 ms
  (fp32 / int8) mean, median under 1 ms.

## Files

* **Runtime:** `pronunciada/r20.py` (DEFAULTS, `enabled`), `pronunciada/text/es.py`, `pronunciada/text/ko.py`,
  `pronunciada/scorer.py` (comment only); `tests/test_r20.py`.
* **Tools, in `bench/round20_fa_rc/`:**
  * item list and identity proofs: `practice_items.py`, `run_equiv.sh`, `run_equiv_final.sh`;
  * learner check and latency: `learner_check.py`, `latency.sh`, `latency_natural.sh`, `triggers.py`;
  * the run and the rule: `overlay.py`, `heldout_run.sh`, `verdict.py`, `audit_paired.py`;
  * final tables: `final_overlay.py`, `final_tables.sh`, `compose_tips.py`, `table_diff.py`.
* **Shared tools extended:** `bench/round19_merge/equiv.py` (an `items:<file>` set) and
  `bench/round19_merge/latency.py` (mean and p95 of the paired difference).
* **Header text:** `bench/scoreboard.py`.
* **Tables:** `tables/round20_fa_release_check_*.json`, the regenerated held-out tables and pages (Spanish values
  only), and `SCOREBOARD.md`.
  * The regenerated `lingolix_live.json` records no git commit because it was built from an extract; the commit
    is 4ef2339.

## Not done / open

* **Korean.** A cheaper Korean pass would need a word-span re-alignment. Its outputs would differ from the frozen
  ones, so it is a new candidate, not an identity-preserving speed-up, and would need its own practice gate and
  held-out run.
* **The audit failure.** It is specific to planted reference errors inside multi-reading words. Whether the audit
  should exclude such words is a question about the audit, not something to change after this run.
* **Not re-run here:** OMPAL scoring, the Japanese length table and Lingolix's Japanese answers. Their outputs
  are identical by proof (c).
