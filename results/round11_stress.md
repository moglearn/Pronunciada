# Round 11: `stress_v3` becomes the word-stress default (en de es it pt ru)

**Decision.** `stress_v3` is the default stress model for all six languages. The per-language
keep / revert rule reverted none. Japanese pitch accent stays `pitch_ja` (experimental) and is
unchanged. `PRONUNCIADA_STRESS_MODEL=stress` selects the previous model with its own calibrator
(`<lang>.stress.json`). Per-language exceptions would go in `pronunciada.prosody.STRESS_MODEL_BY_LANG`;
it is empty.

**Held-out discipline.** The model and its calibrators were chosen on FLEURS validation (round 8)
and on speechocean762 TRAIN learner stress marks (the round-8 decision step). The held-out sets
were run once per condition: the old default first, then the switched tree, on the same clips:
* FLEURS test: the ten-language controlled errors and the fleurs-swap benchmark;
* cv-real test;
* speechocean762 / L2-ARCTIC / EpaDB test.

Azure numbers are cached responses only; no new Azure requests were made.

Interruptions:
* A machine crash interrupted the first attempt. The refit and the held-out runs were redone from
  scratch.
* The coaching agent committed changes to `pronunciada/scorer.py` and `text/ru.py` while the
  "new" run was in flight. Both runs were checked on every clean and corrupted reference of the six
  FLEURS test sets (2 800 references per language): phone scores and sounds-based word scores are
  identical between them. Only the stress channel differs.

## 1. Dev refit against the current tree (FLEURS validation, `fit_prosody`, 200 clips, 5-fold CV)

The fit_prosody row caches were keyed only by model files. After the Russian front-end changed,
the first refit silently reused stale rows. The caches were moved to
`~/.cache/pronunciada/calib/prosody_rows_pre_r11/` and everything was re-collected.

| lang | stress-shift AUC, channel alone, old → v3 | product AUC | hit @ 5 % | native word acceptance s=.5 (old / v3) | sound-error AUC cost (old / v3) |
|---|---|---|---|---|---|
| en | 0.966 → 0.970 | 0.933 → 0.937 | 0.596 → 0.639 | .979→.972 / .979→.974 | 0.0067 / 0.0051 |
| de | 0.959 → 0.977 | 0.919 → 0.937 | 0.485 → 0.620 | .977→.970 / .977→.971 | 0.0084 / 0.0049 |
| es | 0.972 → 0.976 | 0.960 → 0.968 | 0.770 → 0.850 | .992→.986 / .992→.986 | 0.0078 / 0.0071 |
| it | 0.960 → 0.967 | 0.949 → 0.957 | 0.748 → 0.790 | .996→.988 / .996→.988 | 0.0085 / 0.0075 |
| pt | 0.962 → 0.977 | 0.945 → 0.964 | 0.672 → 0.812 | .990→.982 / .990→.982 | 0.0106 / 0.0067 |
| ru (round-10 front-end) | 0.921 → 0.959 | 0.875 → 0.915 | 0.465 → 0.542 | .985→.980 / .985→.979 | 0.0091 / 0.0055 |

v3 is ahead in every language on every number. The switch then ran as prepared
(`python -m bench.prosody.switch_stress_v3`):
* `STRESS_MODEL = "stress_v3"`;
* `<lang>.json` was copied to `<lang>.stress.json`;
* `<lang>.stress_v3.json` became `<lang>.json`;
* the `results/tables/prosody_<lang>.json` dev tables were swapped the same way.

## 2. Held-out, paired on the same clips (old default → `stress_v3`)

**fleurs-swap** (150 recordings per language). Paired cluster bootstrap over source utterances,
1 000 resamples; Δ [95 % CI], two-sided p. Azure is untuned, from cached responses.

| lang | near-swap AUC | near-swap hit @ 5 % FA | near-swap right word | far-swap AUC | Azure near AUC / hit / right word |
|---|---|---|---|---|---|
| en | 0.981 → 0.981 (+0.001 [−0.000, +0.002], p 0.22) | 0.900 → 0.900 (p 1.0) | 0.727 → 0.733 (p 0.75) | 0.995 → 0.995 | 0.963 / 0.793 / 0.686 |
| de | 0.953 → **0.956** (+0.003 [+0.001, +0.005], p < 0.01) | 0.773 → 0.767 (p 1.0) | 0.653 → 0.640 (p 0.57) | 0.996 → 0.996 | 0.887 / 0.517 / 0.460 |
| es | 0.988 → **0.989** (+0.001 [+0.000, +0.003], p 0.04) | 0.927 → 0.953 (+0.027, p 0.08) | 0.840 → **0.893** (+0.053 [+0.013, +0.093], p 0.01) | 0.998 → 0.998 | 0.942 / 0.679 / 0.563 |
| it | 0.971 → **0.976** (+0.005 [+0.002, +0.008], p < 0.01) | 0.833 → 0.847 (p 0.33) | 0.720 → 0.760 (p 0.19) | 0.998 → 0.998 | 0.970 / 0.913 / 0.740 |
| pt | 0.956 → **0.960** (+0.004 [+0.001, +0.007], p 0.01) | 0.827 → 0.833 (p 0.88) | 0.713 → 0.707 (p 0.88) | 0.997 → 0.997 | 0.938 / 0.724 / 0.590 |
| ru | 0.901 → 0.902 (+0.001 [−0.007, +0.008], p 0.78) | 0.607 → 0.607 (p 0.88) | 0.540 → 0.513 (−0.027 [−0.067, +0.007], p 0.21) | 0.990 → 0.991 | 0.909 / 0.342 / 0.373 |

**Ten-language controlled errors** (FLEURS test, 400 clips per language). The metrics are
`WordResult.prob` at s = 0.5, with a paired utterance bootstrap:
* stress-shift AUC: the word carrying a moved primary stress vs the untouched words;
* word-error AUC: all five sound-error kinds;
* native word acceptance.

| lang | stress-shift word AUC | word-error AUC | native word acceptance |
|---|---|---|---|
| en | 0.925 → **0.941** (+0.015, p < 0.01) | 0.943 → 0.945 (+0.002, p < 0.01) | 0.973 → 0.971 (p 0.28) |
| de | 0.924 → **0.941** (+0.017, p < 0.01) | 0.943 → 0.946 (+0.003, p < 0.01) | 0.968 → 0.968 (p 0.73) |
| es | 0.954 → **0.969** (+0.015, p < 0.01) | 0.968 → 0.971 (+0.002, p < 0.01) | 0.984 → 0.985 (p 0.26) |
| it | 0.949 → 0.949 (−0.000, p 0.95) | 0.950 → 0.953 (+0.003, p < 0.01) | 0.984 → 0.985 (p 0.42) |
| pt | 0.941 → **0.954** (+0.013, p < 0.01) | 0.939 → 0.941 (+0.002, p 0.02) | 0.978 → 0.979 (p 0.19) |
| ru | 0.868 → **0.898** (+0.030, p < 0.01) | 0.926 → 0.929 (+0.003, p < 0.01) | 0.970 → 0.968 (−0.002, p 0.09) |

Channel alone (`results/languages.md` §6a, test): stress-shift AUC is en 0.936, de 0.939, es 0.968,
it 0.946, pt 0.952, ru 0.896 (round 10: .919 / .920 / .952 / .946 / .938 / .872).

**cv-real test** (primary `SentenceResult.prob`, paired speaker bootstrap): identical before and
after in every locale (Δ 0.000, p 1.0). Values: en A 0.744 / B 0.522, de A 0.827 / B 0.544,
es 0.770, it 0.781, pt 0.738, ru 0.788. The sentence probability does not read the word stress
channel.

**English (`bench.eval_english`)**: the speechocean762 / L2-ARCTIC / EpaDB word- and sentence-level
tables use the sounds-based word scores (`word_reg`, `word_z`) and the sentence scores. All are
unchanged: sentence prob max |Δ| = 0 on all three test sets. `WordResult.prob` itself changed on
67 % / 95 % / 75 % of the utterances of the three sets, but no pre-specified English held-out metric
reads it. Only the latency rows of `results/english_vs_azure.md` moved, from re-running on a
less loaded machine.

**Keep / revert rule** (revert a language if its near-swap AUC or cv-real AUC drops with p < 0.10
while its stress-shift AUC does not improve):
* No language has a near-swap AUC drop; the five positive changes are +0.001 to +0.005.
* cv-real is identical.
* Italian's stress-shift AUC did not improve (0.949 → 0.949), but nothing dropped, so it is kept.
* Russian native word acceptance moved −0.002 (p 0.09). That is not a rule metric; noted only.

**All six keep `stress_v3`.**

## 3. Untouched languages are byte-identical

`python -m bench.prosody.identity_check dump old|new` / `compare old new` covers the first 5 FLEURS
validation clips per language through the public Scorer: every sentence, word, syllable and
phone float as raw float64 bytes, before the switch and after the switch plus the model-folder
clean-up.

| lang | values | SHA-256 (first 16) before = after |
|---|---|---|
| zh | 1 188 | a2587f5914d8f2f9 = a2587f5914d8f2f9 (IDENTICAL) |
| ja | 1 981 | ae1716900f4aeb2a = ae1716900f4aeb2a (IDENTICAL) |
| fr | 1 923 | 776a8be0d9b3f624 = 776a8be0d9b3f624 (IDENTICAL) |
| ko | 2 216 | 3c626071213cbe70 = 3c626071213cbe70 (IDENTICAL) |

## 4. Model folder clean-up, size, latency

These models were moved out of `pronunciada/data/prosody/` to
`~/.cache/pronunciada/prosody/retired/`:
* `tone_zh` (round-1 local tone model);
* `tone_zh_v3`;
* `pitch_ja_v3` (+ `.zemb.npz`);
* `pitch_ja_v3r`.

No default selects them. They stay selectable at no cost through a path stem, e.g.
`PRONUNCIADA_TONE_MODEL=~/.cache/pronunciada/prosody/retired/tone_zh_v3` (now accepted for every
prosody kind); their calibrators remain in `calib/prosody/`. The tests resolve retired models
the same way and skip if they are absent.

| | before | after |
|---|---|---|
| `pronunciada/data/prosody/` | 4.83 MB | **2.46 MB** (budget 5 MB) |
| contents | 13 model files | `tone_zh_v3z` (+PCA), `tone_zh_v2`, `stress_v3` (+PCA), `stress`, `pitch_ja` |

Latency: the stress step on the densest 10 s FLEURS English clip (2 threads) is 14.4 → 17.2 ms
min. The ZIPA activations it reads are already computed on the default path, so there is no extra
acoustic-model cost. End-to-end FLEURS test medians in `results/languages.md`: en 313 ms,
de 389, es 378, it 453, pt 405, ru 365 per clip (2 worker processes; not comparable with earlier
runs at 5 workers).

Other changes:
* `pronunciada/prosody/__init__.py`: `STRESS_MODEL_BY_LANG`, path stems for every kind.
* `bench/prosody/switch_stress_v3.py`: applied.
* `bench/prosody/r11_compare.py`: paired held-out comparison.
* `bench/prosody/identity_check.py`.
* The retired-model test helpers.
* Caches: old-default outputs in `~/.cache/pronunciada/calib/api_r11_old/`, new in `api_r11_new/`,
  pre-round outputs in `api_pre_r11/`.
