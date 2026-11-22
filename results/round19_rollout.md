# Round 19: rolling the sound-identity head out to more languages

**Result.** Spanish ships. French and English do not. The other languages are in section 6.

| language | ships | why |
|---|---|---|
| es | **yes** | Dev: significant gain, and native false flags fall. Lingolix held out: AUC 0.9795 → **0.9854**, FA@80 2.42 → **1.39 %**. |
| fr | no | Dev: AUC +0.001 and FA@80 −0.5 pt (n.s.), while native words below 0.6 rise 0.2 pt (coordinator's decision). |
| en | no | Dev is strong (check FA@80 3.34 → 2.67 %, p = 0.006; natives fall), but the planted stress-error AUC falls by 0.0058 (limit 0.005). This is left to the architect. |

This builds on the German head (branch `worktree-agent-a2674ec67835464ea`; this branch starts from da9def0) and on
the Korean two-sided step (`worktree-agent-a72249a2162e58229`, `pronunciada/ko_identity.py`).

## 1. What was built

**Runtime.** One generic module, `pronunciada/identity.py`, with one data pair per language:

* `data/coach/identity/<lang>.npz`: the head, 1.9 MB;
* `data/calib/identity/<lang>.json`: classes and the word rule, about 1 kB.

The German and Korean modules are untouched. The scorer hook computes the evidence for any language except de that
has both files, and changes **WordResult.prob only**. Switch: `PRONUNCIADA_IDENTITY_<LANG>=0`
(or `PRONUNCIADA_IDENTITY=0` for all languages).

* **Head.** The German MLP: pooled enc1|enc3|enc5 vector of a sound plus its neighbours' mean, 2048 → 512 → classes.
* **Classes.** The front-end's reference sounds, with per-language merges (es β/ð/ɣ → b/d/g; en dark l → l, ə˞ → ɜ˞;
  fr ʀ → ʁ, ɑ → a).
* **Native variants count as the canonical class.** This is done per sound, generically: the margin is
  log P(expected) − log max P(class that is neither the expected class nor one of the sound's alternatives in the
  front-end). So e / ɛ in French, or t / ɾ / d in English, are never judged where the front-end allows both.
* **Not judged:**
  * skipped classes (fr and en schwa);
  * a sound the front-end lets natives drop with prior ≥ e^−2 (for example French liaison consonants, Spanish
    synalepha vowels, English final t / d).
* **Word rule.** Either form, chosen per language on dev:
  * the German one-sided **knee** factor;
  * the Korean two-sided **step**: word log-odds += c·[min(s,0), min(s+4,0), mean min(s,4), soft count of s<0, 1],
    capped, with no raise when the weakest sound's s < −2. c comes from an L2 logistic regression with the old
    log-odds as offset. A variant applies the step to the sounds probability only, keeping the stress factor
    (added for English).

**Training and selection** (`bench/round19_rollout/`):

* `sources.py`, `devset.py`: data and dev set;
* `collect.py`: one scoring pass that stores the vectors, so every choice is an offline what-if;
* `train.py`, `fit.py`: head and word rule;
* `verify.py`: runtime equals offline, plus latency;
* `heldout.py`: private-cache held-out runs.

## 2. Data (nothing held out is used)

* **Common Voice 22.0, CV's own DEV split only** (CC0):
  * speakers in the cv-real *dev* half;
  * no Lingolix clip, speaker (looked up through dev / test / invalidated.tsv) or prompt;
  * no cv-real clip;
  * accepted clips (≥ 2 up, 0 down), not marked non-native, 1–12 s.
* **Speaker split, by hash:** 35 % dev set (≤ 2 clips each), 52 % head training (≤ 4), 13 % native pool (≤ 4;
  used only for quantiles and guards). For ru, pt and it, which have few speakers, the caps are 4 / 16 / 16.
* **FLEURS:** train for the head (CC BY); validation for the dev set and the validation guards. FLEURS test is read
  only by the held-out runs.
* **Dev set per language:** 460 CV clips and 260 FLEURS-validation clips. Each gives a control item and one planted
  real-word change in the Lingolix categories (subtle / moderate / gross swap, indel, word omission), built as in the
  German dev set. Folds are fit / check by speaker.
* **Controlled sound errors** (`bench.corrupt`) on the CV dev controls.

## 3. Selection rule and guards (fixed before looking)

Guards, all on non-held-out data:

* **G1 native pool:** words below 0.5 and below 0.6.
* **G2 natives below 0.6:** FLEURS validation (swap originals and clean protocol clips) and CV dev controls.
  * French: may rise ≤ 0.3 pt.
  * From Spanish on, under the coordinator's stricter rule: may not rise at all.
* **G3:** controlled-error word AUC does not fall.
* **G4:** far-swap and whole-wrong-word AUC (dev fit fold and FLEURS-validation far swaps) do not fall.
* **G5:** for stress languages, the FLEURS-validation planted stress-error word AUC falls by ≤ 0.005.

**Pick:** the lowest dev fit-fold FA@80 among the passing settings, both forms.

**Ship rule (coordinator):** on the check fold, a gain in AUC or a fall in FA@80 with paired p < 0.05, and no native
row below 0.6 rises.

(Sections 4–7 are filled in below as the languages finish.)
