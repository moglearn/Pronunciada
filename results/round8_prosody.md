# Round 8 (prosody): the tone-v3z recipe applied to word stress and Japanese pitch accent

Dev only. Selection and evaluation used FLEURS validation; no held-out set was read. The
candidates can be selected but are **not** defaults. The shipped `stress.onnx`, `pitch_ja.onnx`
and `pronunciada/data/calib/prosody/<lang>.json` are unchanged: the "old" columns were measured
with `fit_prosody --dry-run`. Russian numbers use whatever `pronunciada/text/ru.py` was in the tree
at run time (another agent is changing it). Old and new were run back to back, so they are
comparable to each other but may move once that work lands.

## Recipe (`bench/prosody/prosody3.py`)

Same training data as the shipped models: FLEURS train, 1 000 utterances per language, front-end
labels. Inputs per 10 ms frame: the 5 pitch / energy features, local-register pitch (2),
low-frequency log-mel (20), and the syllable's pooled ZIPA enc1+enc3 activations, broadcast over
its window. Each syllable's context window therefore also carries its neighbours'. The activations
use a 32-dimensional PCA fitted on FLEURS train of the task's languages, plus a flag; 30 % dropout
is applied to the spectral and ZIPA streams.

Counterfactual augmentation, in feature space:
* stress: for 40 % of words, the pitch + energy frames of the stressed syllable are swapped with
  another syllable's (time-warped), and the label moves to that syllable;
* pitch: in 40 % of utterances, the pitch frames of 1–3 high/low mora pairs are swapped, and both
  labels flip.

Models:
* `stress_v3`: StressNet (word softmax, language embedding), 87 762 parameters, 7 000 steps.
* `pitch_ja_v3`: utterance-level SeqToneNet with 1 output, all inputs.
* `pitch_ja_v3r`: the same with pitch + register only (no spectrum, no ZIPA), transplant 0.6.

## 1. Word stress: `fit_prosody` dev numbers (200 FLEURS validation clips per language, 5-fold CV)

| lang | stress-shift AUC, channel alone | product AUC | hit @ 5 % | native word acceptance s=.5 (before → after) | sound-error AUC cost | model accuracy (dev) |
|---|---|---|---|---|---|---|
| en | 0.966 → **0.970** | 0.933 → **0.937** | 0.596 → **0.639** | .979→.972 / .979→.974 | 0.0067 → 0.0051 | 81.7 → 85.7 % |
| de | 0.959 → **0.977** | 0.919 → **0.937** | 0.485 → **0.620** | .977→.970 / .977→.971 | 0.0084 → 0.0049 | 81.9 → 88.6 % |
| es | 0.972 → **0.976** | 0.960 → **0.968** | 0.770 → **0.850** | .992→.986 / .992→.986 | 0.0078 → 0.0071 | 86.6 → 90.3 % |
| it | 0.960 → **0.967** | 0.949 → **0.957** | 0.748 → **0.790** | .996→.988 / .996→.988 | 0.0085 → 0.0075 | 85.1 → 86.6 % |
| pt | 0.962 → **0.977** | 0.945 → **0.964** | 0.672 → **0.812** | .990→.982 / .990→.982 | 0.0106 → 0.0067 | 86.2 → 90.2 % |
| ru | 0.893 → **0.931** | 0.862 → **0.900** | 0.325 → **0.395** | .986→.978 / .986→.978 | 0.0194 → 0.0142 | 76.2 → 83.6 % |

Every language improves on every word-level number. The native-acceptance cost stays within
the 1-point rule, and the sound-error cost shrinks. Each language has its own fitted calibrator,
`<lang>.stress_v3.json`.

## 2. Japanese pitch accent

| | `pitch_ja` (shipped) | `pitch_ja_v3` (all inputs) | `pitch_ja_v3r` (pitch + register only) |
|---|---|---|---|
| mora agreement with the dictionary labels, FLEURS-ja validation (14 480 morae) | 71.0 % | **80.6 %** (best dev checkpoint) | 73.2 % |
| recall high / low | 70.5 / 71.7 % | 80.2 / 80.9 % | 80.3 / 65.1 % |
| mora flip AUC | 0.779 | 0.879 | 0.802 |
| `fit_prosody --lang ja`: channel-alone AUC | 0.766 | 0.866 | 0.788 |
| product AUC | 0.752 | **0.847** | 0.770 |
| hit @ 5 % | 0.182 | 0.343 | 0.210 |
| native word acceptance s=.5 | .987 → .980 | .987 → .979 | .987 → .977 |
| sound-error AUC cost | 0.046 | 0.035 | 0.048 |

On paper `pitch_ja_v3` meets the bar for leaving the experimental flag: agreement ≥ 80 %, a
product AUC gain, native drop 0.8 points. But it fails the leak checks below. Most of the gain
comes from recognizing the word, not from hearing its pitch. **I would not take it out of
experimental.** The flag was not flipped. `pitch_ja_v3r` follows the pitch best but stays far
below 80 %.

## 3. Leak checks (does the decision follow the acoustics or the dictionary?)

**(a) Feature-space swap / flatten**, all dev words, `python -m bench.prosody.prosody3 eval …`.
The manipulation is the same as the training augmentation, so the new models were trained for
exactly this test. Treat (a) as a sanity check, not independent evidence.

| | swap: argmax follows the swapped cues (↑) | swap: stays on the dictionary syllable (↓) | flatten pitch + energy: still on the dictionary syllable (chance) |
|---|---|---|---|
| stress, old (en/de/es/it/pt/ru) | .50/.48/.47/.30/.41/.40 | .38/.38/.40/.52/.46/.40 | .67/.64/.59/.61/.62/.51 (≈ .37–.42) |
| stress_v3 | **.85/.88/.88/.88/.89/.84** | **.14/.11/.11/.11/.10/.14** | .23/.11/.14/.12/.12/.11 |
| ja: swap follows / flatten still label (majority 53 %) | old .42 / .60 | v3 **.25 / .80** | v3r **.51** / .63 |

**(b) Audio-level monotone resynthesis** (independent of training), `python -m bench.prosody.leak_r8 60`.
Praat PSOLA flattens the pitch at the speaker's median on 60 dev clips per language; loudness
and duration stay. All inputs are recomputed from the audio through the runtime `ProsodyScorer`.
The table reports accuracy on the original → share still on the dictionary label on monotone
audio, and the mono AUC (P(expected) original vs monotone, ↑ = more sensitive to pitch).

| | old | new |
|---|---|---|
| en stress | 81.9 → 79.7 %, AUC 0.566 | 87.8 → 85.2 %, AUC 0.530 |
| de | 81.5 → 75.1 %, 0.603 | 89.2 → 85.5 %, 0.579 |
| es | 87.1 → 69.9 %, **0.745** | 89.5 → 79.2 %, 0.672 |
| it | 87.9 → 84.0 %, 0.611 | 88.6 → 84.3 %, 0.576 |
| pt | 83.0 → 79.8 %, 0.560 | 87.1 → 84.7 %, 0.549 |
| ru | 74.5 → 64.3 %, 0.608 | 83.0 → 80.2 %, 0.582 |
| ja pitch accent (old / v3 / v3r) | 71.3 → 58.0 %, 0.578 | v3 82.3 → **67.9 %**, 0.592; v3r 72.2 → 61.9 %, **0.613** |

Reading:
* Stress also lives in loudness, duration and vowel quality, so a flat pitch should not erase it.
  Both stress models keep most decisions on monotone audio, and `stress_v3` reacts less to the
  flattening than the old model: mono AUC −0.01 to −0.07, largest in Spanish.
* Combined with (a), `stress_v3` follows swapped pitch + energy cues strongly but leans on
  duration / segmental evidence (ZIPA: vowel reduction) more than the old model. For learners who
  misplace stress with full vowels this is a real risk. Real learner stress data (like OMPAL for
  tones) should decide, as it did for v3z.
* Japanese pitch accent is pitch only. `pitch_ja_v3` still keeps the dictionary label on 68 % of
  monotone morae, against a 53 % majority and 58 % for the old model. On feature-space swaps it
  follows the pitch less than the old model (25 % vs 42 %). That is the segment leak the tone
  work warned about, and transplant augmentation at 0.4 did not prevent it. `pitch_ja_v3r` is the
  most pitch-faithful (mono AUC 0.613, swap follow 51 %) but only reaches 73 % agreement.

## 4. Selectable names, size, latency

| env variable = name | files (package) | size | per-language calibrator | tone/stress step, densest 10 s FLEURS test clip, 2 threads, min / median |
|---|---|---|---|---|
| `PRONUNCIADA_STRESS_MODEL=stress_v3` | `stress_v3.onnx/.json/.zemb.npz` | 349 + 100 KB | `<lang>.stress_v3.json` for en de es it pt ru | en 54 syllables: 14.4 / 14.7 (old) → 17.2 / 17.5 ms |
| `PRONUNCIADA_PITCH_MODEL=pitch_ja_v3` | `pitch_ja_v3.onnx/.json/.zemb.npz` | 694 + 100 KB | `ja.pitch_ja_v3.json` | ja 50 morae: 14.7 / 15.3 (old) → 18.2 / 18.5 ms |
| `PRONUNCIADA_PITCH_MODEL=pitch_ja_v3r` | `pitch_ja_v3r.onnx/.json` | 628 KB | `ja.pitch_ja_v3r.json` | 15.2 / 15.5 ms |

The latency is the prosody step with the ZIPA activations given. The Scorer already computes them
on the default path, and requests them whenever a selected model needs them
(`prosody.needs_embeddings`). The prosody data directory is now 4.8 MB, just under the 5 MB test
budget: the next candidate needs a clean-up first.

Wiring:
* `pronunciada/prosody/__init__.py`: `STRESS_MODEL`, `PITCH_MODEL`, `selected_model(lang)`,
  `default_model(lang)`, `needs_embeddings(lang)`, and the sequence path for the pitch model.
* `pronunciada/scorer.py`: loads `<lang>.<model>.json` for a non-default model of any language.
* `bench/calib/fit_prosody.py`: the cache key covers the selected model, and a non-default model's
  calibrator is written to its own file.
* `bench/prosody/export.py`: `stress3` / `pitch3`.
* `lowmel_cache.py` / `zipa_emb_cache.py` now take a language.

Tests: `tests/test_prosody_r8.py` covers defaults unchanged plus one Scorer load-and-score test per
candidate. The full suite gives 464 passed, 1 skipped (not mine).

## What did not work

* `pitch_ja_v3` reaches 80.6 % mostly by recognizing words. Its pitch-following is worse than the
  shipped model's on both leak checks, so the ZIPA stream leaks in Japanese despite transplant
  augmentation.
* Without the spectral and ZIPA streams (`pitch_ja_v3r`), the gain shrinks to +2 points of
  agreement and +0.02 product AUC.
* Stress gains are real on native dev data, but the audio-level monotone check shows reduced pitch
  sensitivity. Whether that matters for learners is untested here.

## 5. Decision step: real learner stress marks (speechocean762 TRAIN, used as dev; test untouched)

`python -m bench.prosody.so762_stress` runs the public `Scorer("en")` with the shipped `stress`
and with `stress_v3` (its own calibrator).

Data:
* words with ≥ 2 syllables in our reference, matched to corpus words: 2 518 words, 125 speakers;
* label: the corpus ships one aggregated word stress score per word (10 or 5); stress wrong = 5,
  which gives 104 positives (4.1 %);
* negatives in (b): stress-correct words with accuracy ≥ 8 (2 072).

Metric: AUC of 1 − `WordResult.stress_prob`, with speaker-bootstrap 95 % CIs and the paired p.

| | old `stress` | `stress_v3` | v3 − old [CI], p |
|---|---|---|---|
| (a) vs all other words | 0.770 [0.724, 0.820] | 0.797 [0.753, 0.844] | +0.027 [−0.005, +0.058], p 0.09 |
| (b) vs stress-correct words with accuracy ≥ 8 | 0.777 [0.730, 0.825] | 0.808 [0.763, 0.853] | +0.032 [−0.002, +0.064], p 0.06 |
| stress-correct words pushed below 0.5 | 0.6 % | 0.5 % | |

The rule (v3 not lower on (b) by more than 0.01) says **adopt**. The duration / vowel-quality
reliance seen on the monotone check does not hurt on these real learner marks; the point estimate
improves, though not significantly with 104 positives.

The switch is prepared but **not applied**, because a Russian job depends on the current default.
`python -m bench.prosody.switch_stress_v3 [--dry-run]` touches exactly:
* `pronunciada/prosody/__init__.py`: `STRESS_MODEL = "stress"` → `"stress_v3"`;
* for en de es it pt ru, `pronunciada/data/calib/prosody/<lang>.json` is copied to
  `<lang>.stress.json`, so the old model stays selectable with its own calibrator;
* for the same languages, `<lang>.stress_v3.json` becomes `<lang>.json`.

If `ru.json` was re-fitted for the new Russian front-end after `ru.stress_v3.json` was made,
re-fit the latter first.
