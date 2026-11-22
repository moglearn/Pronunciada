# Round 19: rolling the sound-identity head out to more languages

**Result.** Five languages ship: Spanish, Russian, Portuguese, Italian and Mandarin. French and English do not.

| lang | ships | dev check fold: AUC / FA@80 (paired p) | held-out headline |
|---|---|---|---|
| es | **yes** | .9636 → .9686 (p < .001) / 3.39 → 2.33 % (p < .001) | Lingolix es AUC .9795 → **.9854**, FA@80 2.42 → **1.39 %** |
| ru | **yes** | .8870 → .8960 (p < .001) / 15.1 → 11.6 % (p = .14) | FLEURS test: every row better, stress-error AUC +.0195 |
| pt | **yes** | .9248 → .9333 (p < .001) / 8.02 → 5.25 % (p = .004) | FLEURS test: every row better |
| it | **yes** (architect, see 6) | .9653 → .9708 (p < .001) / 3.37 → 1.80 % (p = .002) | stress-error AUC on FLEURS test **−0.0063** |
| zh | **yes** | .9393 → .9406 (p = .03) / 7.22 → 6.66 % (p = .23) | natives below 0.6 4.11 → 2.60 %; planted tone AUC −0.0032 |
| fr | no | .9411 → .9420 / 3.82 → 3.37 % (p = .17) | not run (coordinator: gain too small, natives rose 0.2 pt) |
| en | no | .9589 → .9632 (p < .001) / 3.34 → 2.67 % (p = .006) | not run: the stress guard set before the run failed (−0.0058 vs −0.005 allowed) |

**What this builds on.**

* The German head of round 19: branch `worktree-agent-a2674ec67835464ea`, da9def0. This branch starts from it.
* The Korean two-sided step: `worktree-agent-a72249a2162e58229`.
* The Japanese dev-guard list.

`pronunciada/de_identity.py` is untouched; the Korean and Japanese modules are not on this branch.

## 1. What ships

**Runtime.** One generic module, `pronunciada/identity.py`, plus a scorer hook. It runs for any language except de
that has both of its data files:

* `data/coach/identity/<lang>.npz`: the head, about 1.9 MB;
* `data/calib/identity/<lang>.json`: classes and the word rule, about 1.3 kB.

**The head.** The German recipe: an MLP 2048 → 512 → classes, on the pooled enc1|enc3|enc5 vector of a sound and
its neighbours' mean. It is fitted on native speech only:

* FLEURS train (CC BY);
* Common Voice 22.0 dev-split natives (CC0).

**Classes.** The front-end's reference sounds, with a few merges (es β/ð/ɣ → b/d/g; en dark l → l).

* **Native variants count as the canonical class, sound by sound.** The margin is computed against the best class
  that is neither the expected one nor one of that sound's allowed alternatives in the front-end.
* **Not judged:** schwa (fr, en), and sounds the front-end lets natives drop with prior ≥ e^−2.

**Word rule.** Every shipped language ended up with the Korean two-sided step:

* word log-odds += c·[min(s,0), min(s+4,0), mean min(s,4), soft count of s<0, 1];
* capped at +3;
* no raise when the word's weakest sound has s < −2.

ru and pt apply the step to the sounds probability only, which keeps the stress factor as it is. es, it and zh apply
it to the word P. The German knee factor was tried in every language and never passed the strict native guard.

**What it changes.** Only `WordResult.prob`, and through it the Azure-layout word score. The sentence probability,
the 0–10 scores, the phone probabilities and `sounds_prob` are identical with the head on or off: max |diff| = 0 on
108 clips. English sentence scoring is untouched. The English sentence-head branch
(`worktree-agent-a31e9bcfcc58e849b`) blends `w_reg` and the sentence scores and never reads `WordResult.prob`, so the
two changes are independent. English ships nothing here in any case.

**Switches.**

| switch | effect |
|---|---|
| `PRONUNCIADA_IDENTITY_<LANG>=0` (ES, RU, PT, IT, ZH) | that language back to the old behaviour |
| `PRONUNCIADA_IDENTITY=0` | every language back to the old behaviour |

With the switches off, every output of all ten languages equals the German branch da9def0: max |diff| = 0.0 over
108 clips (fr es en ru pt it zh: 12 dev items each; de ko ja: 8 FLEURS-train clips each). The check covers sentence
P, the 0–10 scores, word P, sounds P and phone P (`tables/round19_rollout_switch_equivalence.json`). With the
switches on, only the five shipped languages differ, and only in word P.

**Size and latency.**

* 1.9 MB plus 1.3 kB per shipped language; the backbone is unchanged.
* The head plus the word rule take a median of 0.4–0.8 ms per clip (p95 ≤ 1.5 ms), measured on stored vectors.
* End to end on 3.5 s clips (4 threads, 80 interleaved runs), the on / off medians are: es 62.1 / 61.0, ru 93.5 /
  89.5, pt 93.8 / 91.4, it 81.6 / 82.2 and zh 60.5 / 60.8 ms. These differences are within machine noise, and the
  machine was shared.
* Runtime equals the offline rule: max |ΔP| ≤ 1.1e-4 on 80 dev items per language.

## 2. Data and dev sets (nothing held out is used)

* **Common Voice 22.0, CV's own DEV split only:**
  * speakers in the cv-real dev half;
  * no Lingolix clip, speaker or prompt;
  * no cv-real clip;
  * accepted clips, not marked non-native, 1–12 s.
* **Speaker split, by hash:** 35 % dev set, 52 % head training, 13 % native pool (pool used only for quantiles and
  guards).
* **FLEURS:** train for the head; validation for the dev set and the validation guards.
* **Dev set, space-separated languages** (the German builder, generalised): 460 CV and 260 FLEURS-validation clips
  (ru has 329 CV clips). Each clip gives a control item and one planted real-word change: subtle / moderate / gross
  swap, indel, or word omission. Scored with the Lingolix rules (`bench.third_party.lingolix`).
* **Dev set, Mandarin:** the swap protocol of `bench/textswap.py` (control, swap_near, tone, swap_far per clip;
  2,760 items). Scored on character-span units with the same metric code.
* Folds fit / check are by speaker. Intervals are 1,000 speaker-bootstrap draws (dev) or clip-bootstrap draws
  (held out). Paired differences have two-sided p.

## 3. Selection and ship rule (fixed before each language's run)

**Guards, all on non-held-out data:**

* **G1:** the native pool's words below 0.5 and below 0.6 do not rise.
* **G2:** natives below 0.6 do not rise on FLEURS validation (swap originals and clean clips) or on the CV dev
  controls. French was run with +0.3 pt allowed; from Spanish on, under the coordinator's rule, no rise at all.
* **G3:** controlled-error word AUC does not fall.
* **G4:** far-swap and whole-wrong-word AUC (dev, and FLEURS-validation far swaps) do not fall.
* **G5:** planted stress-error AUC (FLEURS-validation protocol) falls by ≤ 0.005. For Mandarin, the planted-tone dev
  bucket falls by ≤ 0.005.

**Pick:** the lowest dev fit-fold FA@80 among the passing settings. Grid:

* knee: 32 settings;
* step: on word P or sounds P × positive weight 1 / 2 × no-raise −2 / 0 / never × caps 1 / 2 / 3.

**Ship rule:** on the check fold, a gain in AUC or a fall in FA@80 with paired p < 0.05, and no native row below 0.6
rises.

## 4. Per-language tables

Held-out looks: each row was run **once**, after the candidate and its dev numbers were written to STATUS.md.

### Spanish: ships (4 held-out looks: Lingolix, FLEURS test, cv-real test, contrast audit)

| row | off | on | Δ [95 %], p |
|---|---|---|---|
| dev check AUC | .9636 [.952,.974] | .9686 [.957,.978] | +.0050 [+.0036,+.0065], p < .001 |
| dev check FA@80 | 3.39 % | 2.33 % | −1.07 [−1.82,−0.57], p < .001 |
| dev all AUC / FA@80 | .9614 / 3.45 % | .9662 / 2.36 % | p < .001 / p < .001 |
| dev natives below 0.6: pool / CV / FLEURS val | 2.32 / 2.03 / 2.46 % | 2.16 / 1.86 / 2.34 % | all fall |
| **Lingolix es AUC** (v2026.10) | .9795 | **.9854** [.976,.992] | +.0059 [+.0032,+.0088], p < .001 |
| **Lingolix es FA@80** (v2026.10) | 2.42 % | **1.39 %** [0.73,2.26] | −1.03 [−2.00,−0.30], p = .002 |
| Lingolix es, main rules: AUC / FA@80 | .9825 / 1.91 % | .9873 / 1.02 % | |
| Lingolix subtle / indel AUC | .971 / .970 | .984 / .974 | |
| FLEURS test near-swap AUC | .9888 | .9936 | +.0048, p < .001 |
| FLEURS test far-swap AUC | .9980 | .9989 | +.0008, p < .001 |
| FLEURS test natives below 0.6 / swap-set natives | 2.13 / 2.46 % | 2.04 / 2.28 % | n.s. / n.s. |
| **FLEURS test stress-error AUC** | .9657 | .9626 | **−.0032** [−.0039,−.0024], p < .001 |
| FLEURS test stress-error hit@5 | .833 | .815 | −.0175, p = .004 |
| FLEURS test sound-error AUC | .9619 | .9630 | +.0011, p < .001 |
| cv-real test A: word mean / word min AUC | .7444 / .7460 | .7628 / .7619 | +.018 (p = .002) / +.016 (p = .002) |
| contrast audit word hit@5: tap–trill / voicing / m–n | .554 / .971 / .988 | .842 / .996 / 1.0 | sound rows identical |

Published for comparison (not re-run here): Lingolix 0.97 / 2 %. The baseline here reproduces the brief's 0.979 /
2.4 %.

### Russian: ships (3 looks: FLEURS test, cv-real test, audit; no Lingolix ru)

| row | off | on | Δ, p |
|---|---|---|---|
| dev check AUC | .8870 [.866,.907] | .8960 [.874,.915] | +.0089 [+.0044,+.0131], p < .001 |
| dev check FA@80 | 15.1 % | 11.6 % | −3.5, p = .14 |
| dev natives below 0.6: pool / CV / val | 4.03 / 4.91 / 3.70 % | 3.24 / 4.09 / 3.00 % | all fall |
| FLEURS test natives below 0.6 / swap set | 4.20 / 4.82 % | 3.62 / 4.09 % | p < .001 / p < .001 |
| FLEURS test acceptance | 96.92 % | 97.25 % | +0.33 pt, p < .001 |
| FLEURS test near-swap AUC / hit@5 | .8951 / .627 | .9076 / .693 | p = .006 / p < .001 |
| FLEURS test **stress-error AUC** | .8882 | .9078 | **+.0195**, p < .001 |
| FLEURS test sound-error AUC | .9245 | .9278 | p < .001 |
| cv-real test A: word mean / min | .7859 / .7916 | .8020 / .8041 | p = .01 / p = .07 |
| audit word hit@5: hard–soft / unstressed vowel / voicing / final devoicing / s–ʃ / m–n | .171 / .325 / .508 / .037 / .571 / .812 | .225 / .412 / .738 / .172 / .925 / .963 | sound rows identical |

### Portuguese: ships (3 looks)

| row | off | on | Δ, p |
|---|---|---|---|
| dev check AUC | .9248 [.908,.941] | .9333 [.916,.951] | +.0085, p < .001 |
| dev check FA@80 | 8.02 % | 5.25 % | −2.77 [−4.89,−0.99], p = .004 |
| dev natives below 0.6: pool / CV / val | 3.15 / 2.63 / 2.98 % | 2.83 / 2.44 / 2.66 % | all fall |
| FLEURS test natives below 0.6 / swap set | 3.26 / 2.84 % | 2.83 / 2.37 % | p < .001 / p < .001 |
| FLEURS test near-swap AUC | .9601 | .9648 | p = .004 |
| FLEURS test far-swap AUC | .9971 | .9981 | |
| FLEURS test **stress-error AUC** | .9512 | .9561 | **+.0049**, p < .001 |
| FLEURS test sound-error AUC | .9354 | .9372 | |
| cv-real test A: word mean / min | .7482 / .7427 | .7740 / .7720 | p < .001 / p < .001 |
| audit word hit@5: nasal / mid vowels / tap–trill / voicing / s–ʃ / m–n | .517 / .304 / .758 / .754 / .717 / .783 | .521 / .438 / .804 / .892 / .817 / .942 | |

### Italian: ships by the dev rule; the held-out stress row is over the dev limit (3 looks)

| row | off | on | Δ, p |
|---|---|---|---|
| dev check AUC | .9653 [.954,.975] | .9708 [.960,.981] | +.0055, p < .001 |
| dev check FA@80 | 3.37 % | 1.80 % | −1.57 [−2.37,−0.72], p = .002 |
| dev natives below 0.6: pool / CV / val | 2.43 / 2.58 / 1.80 % | 2.42 / 2.35 / 1.67 % | none rise |
| dev validation stress-error AUC | .9503 | .9454 | −.0049 (limit .005) |
| **FLEURS test stress-error AUC** | .9469 | .9406 | **−.0063** [−.0074,−.0054], p < .001 |
| FLEURS test stress-error hit@5 | .713 | .671 | −.041, p < .001 |
| FLEURS test swap-set natives below 0.6 | 2.26 % | 2.37 % | +0.12 pt, n.s. (p = .57) |
| FLEURS test natives below 0.6 (protocol set) / acceptance | 2.29 % / 98.39 % | 2.17 % / 98.31 % | n.s. / n.s. |
| FLEURS test near-swap AUC | .9768 | .9830 | p < .001 |
| FLEURS test sound-error AUC | .9507 | .9534 | p < .001 |
| cv-real test A: word mean / min | .7647 / .7563 | .7797 / .7714 | p = .04 / p = .07 |
| audit word hit@5: gemination / mid vowels / voicing / s–ʃ / m–n | .325 / .379 / .754 / .771 / .971 | .325 / .629 / .946 / .963 / 1.0 | |

### Mandarin: ships (3 looks)

| row | off | on | Δ, p |
|---|---|---|---|
| dev check AUC (character units) | .9393 | .9406 | +.0013 [+.0001,+.0027], p = .03 |
| dev check FA@80 | 7.22 % | 6.66 % | p = .23 |
| dev buckets: near / tone / far | .964 / .872 / .989 | .968 / .870 / .990 | |
| dev natives below 0.6: pool / CV / val | 5.39 / 4.89 / 3.27 % | 4.27 / 4.07 / 2.02 % | all fall |
| FLEURS test natives below 0.6 / swap set | 4.11 / 3.98 % | 2.60 / 2.58 % | p < .001 / p < .001 |
| FLEURS test acceptance | 97.06 % | 97.89 % | +0.82 pt, p < .001 |
| FLEURS test near-swap AUC | .9727 | .9787 | p = .002 |
| FLEURS test far-swap AUC | .9915 | .9913 | −.0001, n.s. |
| **FLEURS test planted-tone AUC** | .8339 | .8306 | **−.0032** [−.0060,−.0003], p = .03 |
| FLEURS test planted-tone hit@5 | .367 | .333 | n.s. |
| FLEURS test sound-error AUC | .9379 | .9400 | p < .001 |
| cv-real test A: word mean / min | .7373 / .7256 | .7496 / .7379 | p = .006 / p = .03 |
| audit word hit@5: aspiration / retroflex / n–ŋ / y–u / r–l / tone / m–n | .679 / .650 / .487 / .950 / .775 / .400 / .771 | .750 / .725 / .483 / .983 / .808 / .375 / .850 | |

### French: not shipped (0 held-out looks)

| row | off | on (knee, natives +0.3 pt allowed) |
|---|---|---|
| dev check AUC | .9411 | .9420 (p < .001) |
| dev check FA@80 | 3.82 % | 3.37 % (p = .17) |
| natives below 0.6: pool / CV / val | 2.01 / 1.92 / 2.65 % | 2.21 / 2.10 / 2.85 % (rise) |

**Why the gain is so small** (dev aggregates):

* The head is as accurate as the German one (96.8 % vs 97.3 %).
* Liaison and schwa are not the cause.
* 99 of the 143 changed words still missed at recall 80 % are indels, which a "which sound" head does not judge.
* The audit shows the aligner already catches French one-sound swaps: pooled word miss@5 is 0.13 for fr, against
  0.46 for de and 0.55 for ko.

### English: not shipped (0 held-out looks)

* **Head.** Class accuracy 94.5 % on dev and 94.0 % on the pool.
* **Closest candidate** (step on word P): check AUC .9589 → .9632 (p < .001), FA@80 3.34 → 2.67 % (p = .006).
  Natives below 0.6 fall everywhere (pool 3.23 → 2.84, CV 3.24 → 2.90, val 3.78 → 3.36 %).
* **Why not:** the FLEURS-validation stress-error AUC falls by 0.0058, over the 0.005 limit, and the architect
  kept the guard.
* **The sounds-only variant** passes the stress guard but lets natives rise by 0.02–0.10 pt.
* No English data is in the package; the tables are `round19_rollout_en_*.json`.

## 5. Everything that got worse anywhere (held out)

* **it:** FLEURS test stress-error AUC −0.0063 and hit@5 −4.1 pt. On validation it was −0.0049, so the held-out
  value is over the 0.005 dev limit. Swap-set natives below 0.6 rose +0.12 pt (n.s.).
* **es:** FLEURS test stress-error AUC −0.0032 and hit@5 −1.75 pt, within the limit.
* **zh:**
  * FLEURS test planted-tone AUC −0.0032 (p = .03) and hit@5 −3.3 pt (n.s.);
  * audit tone word hit@5 .400 → .375 and n–ŋ .487 → .483;
  * far-swap AUC −0.0001 (n.s.).
* Nothing else fell on any held-out row of es, ru, pt, it or zh. Every native row below 0.6 fell or was flat, except
  the Italian swap set above.

## 6. For the architect

1. **Italian stress.** The held-out stress-error cost (−0.0063) is over the dev limit. The architect asked for the
   looks to be run as frozen; nothing was changed afterwards. Options: keep it, or set `PRONUNCIADA_IDENTITY_IT=0`.
   Applying the step to the sounds probability (the ru / pt form) would likely remove the stress cost. It was in the
   Italian grid, did not pass the native guards there, and was not re-tried after the look.
2. **English** is one guard away (stress −0.0058), with large native gains.
3. **The pattern.** The head pays off where the contrast audit shows one-sound swaps the aligner misses (ru, pt, de,
   ko) and adds little where it does not (fr, and partly zh). Indels are the main remaining miss in every language.
   Korean's extra- / missing-sound evidence is the next lever.

## Files

* `pronunciada/identity.py`
* `pronunciada/scorer.py` (hook)
* `pronunciada/data/coach/identity/{es,ru,pt,it,zh}.npz`
* `pronunciada/data/calib/identity/{es,ru,pt,it,zh}.json`
* `bench/round19_rollout/{sources,devset,collect,train,fit,verify,heldout}.py`
* `tests/test_round19_rollout.py`
* `tests/test_scorer.py`: `test_prosody_channel` now runs with `PRONUNCIADA_IDENTITY=0`. Its invariant, word P =
  sounds × prosody × presence, no longer holds for Mandarin with the step on.

`pytest -q tests`: 588 passed.
* `results/tables/round19_rollout_*.json`

Logs and the private cache roots are under `$PRONUNCIADA_CACHE/round19/rollout/`.
