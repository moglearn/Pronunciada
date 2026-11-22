# Round 20: Mandarin tone rows after the identity step (plus Portuguese nasal and German stress)

**Result.** A new Mandarin rule scales each raise of the identity step by the word's tone factor. It brings the
tone-swap rows back to roughly their head-off level and keeps most of the native gain:

* FLEURS test natives below 0.6: 4.11 % (off), 2.60 % (round 19), **2.86 %** (new).
* Tone swaps, right word blamed: 0.300, 0.280, **0.307**. Against Azure the gap is −0.080 [−0.180, +0.007],
  p = 0.08, so it is no longer significantly behind.
* Two tone rows are still a little under head-off, neither significant: AUC −0.0006 (p = 0.72) and caught at 5 %
  −0.013 (two items of 150, p = 0.88).
* Against round 19 the new rule gives back a little on sound errors (sound-error AUC −0.0009, p < 0.001), on
  near swaps (−0.0007, p = 0.004) and on three audit rows.

Portuguese and German: no clean gate was found, so both are left as they are.

| arm | switch | what it is |
|---|---|---|
| off | `PRONUNCIADA_IDENTITY_ZH=0` | no identity rule for Mandarin |
| r19 | `PRONUNCIADA_IDENTITY_ZH_R20=0` | round-19 rule, current main |
| **new** | default | round-19 step, raise × clip((T − 0.2) / (0.98 − 0.2), 0, 1); T = the word's tone factor |

## 1. What was wrong (dev only)

Dev and validation data, `python -m bench.round20_mandarin.dev diagnose`:

* **The round-19 rule raises almost every planted tone swap.** It raised 97 % of them (290 FLEURS-validation swaps,
  674 dev items), by +0.71 log-odds on average. A tone swap leaves the consonants and vowels right, so the identity
  evidence is confident.
* **The tone channel is weak on those words.** Their median tone factor is 0.80; for native words the 5 % quantile
  is 0.86. The raise hides that weak evidence and changes which word is lowest in the sentence. This confirms the
  architect's reading.
* **The native gain has two sources.** Of the 80 native dev words lifted over 0.6, about half were held down by
  the sounds factor S (median 0.75) and half by the tone factor T (median 0.73). The contrast factor was the cause
  in 4 %.

**Design (a), the step on sounds only, does not work.** It moves the step to S (or to S × contrast) and keeps T
multiplicative. It restores the invariant and gives the best tone rows, but it fails G3:

* controlled sound-error AUC 0.9247 → 0.9218;
* the near-swap and sound-error gains are lost (validation sound-error AUC 0.9487 → 0.9421).

The reason: the step's coefficients are fitted on word log-odds that include the tone factor.

## 2. Selection (dev, rule fixed before the grid; `bench/round20_mandarin/dev.py`)

**Guards:**

* G1–G5 of `results/round19_rollout.md` §3, against off.
* Two new families, each of which must not fall below off:
  * the FLEURS-validation tone swaps (290; scoreboard definitions from `bench.eval_swap._metrics`): AUC, caught at
    5 %, right word;
  * the same three measures on the dev set's tone items.

**Pick:** the lowest mean of the three dev native rows below 0.6.

**Grid:** 102 settings: on = word P / sounds / S × contrast; tone gate 0.2–0.95; tone ramp (t0, t1) over
{0.1 … 0.8} × {0.5 … 1.0}; and combinations. Full table: `tables/round20_mandarin_dev_grid.json`.

* The round-19 rule fails all six new tone guards on dev, so the new guards do see the held-out problem.
* 26 settings pass. Every "sounds" and "segmental" variant fails G3.
* Picked: **ramp 0.2–0.98**.

| dev row | off | r19 | **new** |
|---|---|---|---|
| natives < 0.6: pool / CV dev / FLEURS val | 5.39 / 4.89 / 3.27 % | 4.27 / 4.07 / 2.02 % | 4.57 / 4.37 / 2.29 % |
| val tone swaps: AUC / caught@5 % / right word | .8478 / .369 / .328 | .8473 / .341 / .303 | .8501 / .369 / .334 |
| dev tone items: AUC / caught@5 % / right word | .8651 / .390 / .496 | .8634 / .362 / .488 | .8664 / .393 / .504 |
| dev check fold AUC (Δ vs off, p) | .9393 | .9406 (+.0013, p = .03) | .9411 (+.0018 [+.0007, +.0031], p = .002) |
| dev check fold FA@80 (p) | 7.22 % | 6.66 % (p = .23) | 6.55 % (p = .034) |
| controlled sound-error AUC | .9247 | .9258 | .9253 |

* **Fragility.** Validation caught at 5 % passes with a tie (.369 = .369).
* **Runtime = offline.** Max |ΔP| is 6.4e-5 on 80 dev items (float16 stored vectors); see
  `tables/round20_mandarin_verify.json`.

## 3. Held out (one run per row, after the candidate and its dev numbers were in STATUS)

**How the run was made.**

* Three arms were scored in one process on the same items, switches toggled per item.
* Azure comes from cached responses only; no request was sent.
* Commit of the frozen candidate: 8d1952e. The later commits leave Mandarin outputs unchanged (§5).

### FLEURS test, tone swaps (150) and swap benchmark (scoreboard definitions, utterance-clustered bootstrap)

| row | off | r19 | **new** | new − off [95 %], p | new − r19 [95 %], p | Azure |
|---|---|---|---|---|---|---|
| tone AUC | .8340 | .8305 | **.8334** | −.0006 [−.0034, +.0022], .72 | +.0028 [+.0020, +.0038], .002 | .8456 |
| tone caught at 5 % | .367 | .333 | **.353** | −.013 [−.053, +.027], .88 | +.020 [−.013, +.067], .40 | .402 |
| tone right word blamed | .300 | .280 | **.307** | +.007 [−.027, +.040], .83 | +.027 [+.007, +.053], .05 | .387 |
| native words < 0.6 (swap set) | 3.98 % | 2.58 % | **2.76 %** | −1.22 pt, .002 | +0.18 pt, .02 | 3.65 % |
| near-swap AUC | .9749 | .9801 | **.9794** | +.0046, .004 | −.0007, .002 | .9663 |
| far-swap AUC | .9920 | .9920 | .9920 | 0 | 0 | .9887 |

**Paired against Azure** (same items):

| row | off | r19 | new |
|---|---|---|---|
| tone AUC | −.012, p = .62 | −.015, p = .51 | −.012, p = .61 |
| tone caught at 5 % | −.035, p = .46 | −.069, p = .30 | −.049, p = .44 |
| tone right word | −.087 [−.180, .000], p = .07 | **−.107 [−.207, −.023], p = .02** | −.080 [−.180, +.007], p = .08 |
| native false alarms < 0.6 | +.003, p = .55 | −.011, p = .006 (ahead) | −.009, p = .048 (ahead) |

**Rollout FLEURS-test rows** (`bench.round19_rollout.heldout` metrics; paired clip bootstrap):

| row | off | r19 | **new** | new − r19, p |
|---|---|---|---|---|
| natives below 0.6 (protocol clean clips) | 4.11 % | 2.60 % | **2.86 %** | +0.26 pt, p < .001 |
| native acceptance | 97.06 % | 97.89 % | 97.73 % | −0.15 pt, p < .001 |
| sound-error AUC / hit@5 % | .9379 / .803 | .9400 / .819 | .9392 / .809 | −.0009 / −.0095, both p < .001 |
| planted tone AUC (native negatives) | .8339 | .8306 | .8334 | +.0028, p < .001 (new − off −.0005, p = .75) |

`tables/round20_mandarin_heldout_fleurs.json`

### OMPAL test (real learners): word AUC, paired speaker bootstrap (2,000 draws, 21 speakers)

| class | off | r19 | **new** | r19 − off | new − off | new − r19 |
|---|---|---|---|---|---|---|
| any error | .7920 | .7866 | .7900 | −.0054 [−.0086, −.0027], p < .001 | −.0020 [−.0043, .0000], p = .05 | +.0034, p < .001 |
| tone-only | .7784 | .7724 | .7768 | −.0060 [−.0098, −.0029], p < .001 | −.0016 [−.0041, +.0006], p = .18 | +.0044, p < .001 |
| consonant / vowel | .8565 | .8540 | .8525 | −.0025, p = .41 | −.0040 [−.0106, +.0005], p = .08 | −.0014, p = .48 |

* The round-19 loss on OMPAL is significant with a speaker bootstrap; round 19 did not compute it.
* The new rule removes most of that loss.
* `tables/round20_mandarin_heldout_ompal.json`

### Contrast audit zh (FLEURS test, 250 clips): word hit@5 %, paired clip bootstrap

| row | off | r19 | **new** | new − r19, p | new − off, p |
|---|---|---|---|---|---|
| dictionary tone → other tone (n 240) | .400 | .375 | **.396** | +.023, .61 | −.005, .74 |
| aspirated → plain | .683 | .658 | .600 | −.053, .08 | −.079, .06 |
| plain → aspirated | .675 | .842 | .800 | −.044, .14 | +.123, .04 |
| retroflex → dental | .642 | .750 | .725 | −.027, .10 | +.081, .04 |
| dental → retroflex | .658 | .700 | .675 | −.027, .54 | +.018, .50 |
| n → ŋ | .583 | .600 | .567 | **−.041, .04** | −.031, .25 |
| ŋ → n | .392 | .367 | .317 | −.062, .14 | −.083, .10 |
| y → u / u → y | .933 / .967 | .983 / .983 | .983 / .983 | 0 | +.03 / +.02 |
| ʐ → l / l → ʐ | .950 / .600 | .942 / .675 | .933 / .675 | −.011 / 0 | −.02 / +.08 |
| m → n / n → m | .542 / 1.0 | .700 / 1.0 | .667 / 1.0 | −.030, .12 | +.101, < .01 |

The threshold is each arm's own 5th percentile of native word P. Fewer natives are raised under the new rule, so its
threshold is lower, and that alone costs some sound rows. `tables/round20_mandarin_heldout_audit_zh.json`

### cv-real zh test half (speaker bootstrap)

| test A | off | r19 | **new** | new − r19, p |
|---|---|---|---|---|
| word mean AUC | .7373 | .7496 | .7483 | −.0013, p = .35 |
| word min AUC | .7256 | .7379 | .7350 | −.0029, p = .03 |
| sentence P | .7708 | .7708 | .7708 | 0 |

`tables/round20_mandarin_heldout_cvreal.json`

## 4. Everything that got worse (held out)

**New against round 19 (current main):**

* natives below 0.6 +0.26 pt (2.60 → 2.86 %; still −1.25 pt against off);
* native acceptance −0.15 pt;
* sound-error AUC −0.0009 and hit@5 % −0.95 pt;
* near-swap AUC −0.0007;
* cv-real word-min AUC −0.0029 (p = .03);
* OMPAL consonant / vowel −0.0014 (n.s.);
* audit n → ŋ −4.1 pt (p = .04), aspirated → plain −5.3 pt (p = .08), ŋ → n −6.2 pt (n.s.), and smaller
  non-significant drops on retroflex and m → n.

**New against off** (rows the new rule does not fully recover):

* tone AUC −0.0006 and caught at 5 % −1.3 pt (n.s.);
* OMPAL any −0.0020 (p = .05), tone-only −0.0016 (p = .18), consonant / vowel −0.0040 (p = .08);
* audit aspirated → plain −7.9 pt (p = .06), ŋ → n −8.3 pt (p = .10).

**Not met.** The brief's target was tone rows at least at their head-off values. Right word is met. AUC and caught at
5 % are within noise but numerically under head-off.

**The invariant.** Word P = sounds × prosody × presence still does not hold for Mandarin with the head on: the ramp
still moves the whole word P. `tests/test_scorer.py::test_prosody_channel` therefore keeps running with
`PRONUNCIADA_IDENTITY=0`.

## 5. Switches, exactness, size, latency

**Switches:**

* `PRONUNCIADA_IDENTITY_ZH_R20=0` restores the round-19 rule;
* `PRONUNCIADA_IDENTITY_ZH=0` and `PRONUNCIADA_IDENTITY=0` work as before.

**Code.** The rule lives in `data/calib/identity/zh.json` under `"round20"`; `identity.rule()` merges it. The scorer
now passes the word's tone factor, contrast factor, sound classes and extra-sound evidence to `apply_word`. No
shipped rule uses the last three.

**Exactness.** Equivalence results are in §7. Size: +40 bytes of JSON. Latency: a few arithmetic operations per
word; see §7.

## 6. Portuguese and German (time-boxed; nothing ships, no held-out look)

### Portuguese nasal → oral (`bench/round20_mandarin/pt_nasal.py`, about 75 min)

**The validation audit reproduces the held-out drop.** On FLEURS validation, 120 edits, the word hit@5 % is .267 with
the identity step off and .192 with it on. The step raises 83 % of the edited words.

**Why.** The edit deletes the coda nasal from the reference while the audio keeps it. The head has no nasal-vowel
class and judges only reference sounds. The edited vowel's evidence is ordinary: median +4.7, about the native 25th
percentile.

**Class-specific no-raise for vowels: fails.** With thresholds −1 to +5 the audit row never recovers (best .242), and
natives rise.

**An extra-sound gate separates the edits well.** The rule: no raise when the word's lowest llr_ins is below t, with
`pronunciada.align` computing insertion evidence for pt. At −3:

* 79 % of the edits and 4.3 % of native words are below it;
* audit row .275 (off .267);
* dev fit AUC .9498 → .9553 and FA@80 4.91 → 4.52 %;
* controlled sound-error AUC .9244 → .9357;
* validation sound-error AUC .9383 → .9510.

It still fails the pre-declared rule, which allowed no native row to rise against the shipped rule:

* pool natives below 0.6: 2.83 → 2.92 %;
* FLEURS-validation natives: 2.66 → 2.85 %;
* validation stress-error AUC: .9590 → .9574.

So it is not shipped. The gate code is in `identity.py` (`no_raise_ins_below`) and is inert: no rule file sets it.
`tables/round20_mandarin_pt_nasal_dev.json`

### German stress (`bench/round20_mandarin/de_stress.py`, about 20 min)

**The test.** On the round-19 German dev stress rows, use the full stress channel where the pitch evidence is reliable
(voiced share ≥ v, mean voicing ≥ u) and the sounds are confident (sounds × presence ≥ s0), and the reliability
factor elsewhere. Grid: 48 settings.

* Full channel: natives below 0.6 3.94 %, below 0.8 9.57 %, stress AUC .8797.
* Shipped reliability factor: 3.24 %, 7.74 %, .8616.

**No gate passes.** Every gate that recovers stress AUC also raises natives below 0.6 and 0.8, roughly in proportion:

* +0.011 AUC costs +0.4 pt below 0.6;
* the narrowest gate (8 % of words) gives +0.002 AUC for +0.06 pt.

The flagged natives are just as common among reliable, confidently pronounced words. Left as is.
`tables/round20_mandarin_de_stress_dev.json`

## 7. Exactness and latency

**Setup.** `bench/round20_mandarin/run_equiv.sh` and `equiv_table.py`, the dump of `bench/round19_merge/equiv.py`:
100 FLEURS-validation clips per language, each scored twice (own text and one changed word). Every output is
compared: SentenceResult, rescoring at strictness 0 and 1, details, coaching, Azure layout. Tolerance is 0. Table:
`tables/round20_mandarin_switch_equivalence.json`.

**Results.**

* **Switch off equals main.** `PRONUNCIADA_IDENTITY_ZH_R20=0` gives max |diff| = 0 against main (8c74789), with 0
  differing values, in fp32 and in int8 (about 603,000 values each).
* **Other languages are untouched.** For pt, es, ru, en, ko, de, ja, fr and it, the branch with default settings
  equals main: max |diff| = 0 in fp32 and int8, 0.66–0.86 M values per language.
* **The held-out run used the frozen candidate.** The branch head equals commit 8d1952e on zh (max |diff| = 0, fp32
  and int8).
* **What the rule changes.** With the rule on, zh differs from main only in:
  * word P (`words[].prob`, and at strictness 0 / 1);
  * the Azure-layout word AccuracyScore;
  * the probability of 4 coaching items (of 200 clips).

  Sentence P, the 0–10 scores, phone, sounds and tone probabilities are identical.

**Latency.** `tables/round20_mandarin_latency.json`: 40 clips of 2.5–4.5 s, 4 threads, interleaved, 80 runs per arm.
The median is 17.5 ms with the new rule and 17.2 ms with the round-19 rule; p95 is 21.8 / 22.2 ms. That is within
noise.

**Tests.** `pytest -q tests`: 615 passed (6 new in `tests/test_round20_identity.py`).

## 8. Recommendation

1. **Ship the new Mandarin rule.** It removes the one row that was significantly behind Azure (tone right word,
   p = 0.02 → 0.08) and most of the significant OMPAL loss. It keeps 83 % of the FLEURS-test native gain and stays
   significantly ahead of Azure on native false alarms (p = 0.048).
   * **The price:** small sound-row losses against round 19 (sound-error AUC −0.0009; audit n → ŋ −4 pt).
   * **Turning the Mandarin head off is not the better answer.** It would give back 1.25 pt of native false alarms
     and the near-swap gain (+0.0046 AUC, p = 0.004) for about one tone item in 150.
2. **The architect has to decide whether "tone rows at least head-off" is a hard gate.** If it is, the honest
   answer is "head off for zh": AUC −0.0006 and caught at 5 % −0.013 are numerically under head-off, though far
   from significant.
3. **Portuguese has a promising lever.** The extra-sound gate is the strongest result here: dev AUC +0.0055, FA@80
   −0.4 pt, validation sound-error AUC +0.013, and the nasal audit row recovered. It costs 0.1–0.2 pt of native
   words below 0.6, still under head-off. It probably needs the step's coefficients refitted with the extra-sound
   feature, which is a new round rather than a gate.

## Files

* `pronunciada/identity.py`, `pronunciada/scorer.py`, `pronunciada/data/calib/identity/zh.json`
* `bench/round20_mandarin/`: collect, dev, verify, heldout, heldout_zh.sh, audit_arms.sh, audit_tables, overlay,
  pt_nasal, de_stress, run_equiv.sh
* `tests/test_round20_identity.py`
* `results/tables/round20_mandarin_*.json`

Logs and caches: `$PRONUNCIADA_CACHE/round20/mandarin/`.
