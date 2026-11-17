# Round 4: omission channel, prosody leniency rule, tone model v2

Designed after looking at the round-3 held-out evaluation (error analysis of the Azure text-swap benchmark), fitted on FLEURS validation only, then evaluated once on the held-out sets. Round-3 tables are kept in `results/tables/round3/`.

## What changed

1. **Omission channel** (`bench/calib/fit_omission.py`, `pronunciada/data/calib/omission.json`). In about three quarters of the far-swap records where the wrong word was blamed, the blamed word was a one-sound function word (pt "é", fr "à", ja "を") that the hard rule `every sound has llr_del < -2` had flagged as omitted, also in the untouched native recording; the benchmark adapter scored such a word 0. The rule is replaced by a calibrated `present_prob` (one logistic model for all languages, fitted on planted extra words), multiplied into `WordResult.prob`; `omitted = present_prob < 0.5`; the adapter now uses `WordResult.prob` as is.
   Dev CV (25396 said / 3549 planted words): AUC 0.996; sounds=1: false omission 0.048 → 0.010, caught 0.87 → 0.65; sounds=2: false omission 0.009 → 0.005, caught 0.82 → 0.78; sounds>=3: false omission 0.001 → 0.003, caught 0.63 → 0.91.
2. **Prosody channel leniency** (`bench/calib/fit_prosody.py`). Round 3 let the channel alone put 1 % (lenient) / 5 % (strict) of native words below 0.5, which cost 1.4–2.8 points of native word acceptance at the default strictness. Now the pair is chosen per language by a fixed rule on dev CV: the strictest pair of the grid that costs at most 1.0 point.
3. **Japanese pitch accent** no longer enters `WordResult.prob` unless `Scorer(..., experimental=True)`; the per-syllable pitch probability is still reported. (Both were removed in round 15: Japanese pitch accent is not scored.)
4. **Mandarin tone model v2** (`tone_zh_v2.onnx`, sequence model, 42 h of extra training speech, contour-transplant augmentation) is the default; details in `results/prosody.md` section 6.

## Azure text-swap benchmark, FLEURS test clips: round 3 → round 4 (Azure)

| language | far: right word | near: AUC | near: hit @ 5 % FA | native units < 0.6 |
|---|---|---|---|---|
| en | 0.839 → 0.900 (0.898) | 0.973 → 0.981 (0.963) | 0.873 → 0.900 (0.793) | 0.039 → 0.034 (0.022) |
| zh | 0.850 → 0.887 (0.899) | 0.955 → 0.964 (0.966) | 0.813 → 0.860 (0.896) | 0.076 → 0.046 (0.036) |
| ja | 0.719 → 0.920 (0.875) | 0.920 → 0.965 (0.919) | 0.760 → 0.840 (0.680) | 0.106 → 0.025 (0.020) |
| es | 0.880 → 0.953 (0.910) | 0.981 → 0.988 (0.942) | 0.900 → 0.927 (0.679) | 0.036 → 0.024 (0.018) |
| fr | 0.729 → 0.920 (0.823) | 0.979 → 0.986 (0.937) | 0.940 → 0.947 (0.647) | 0.029 → 0.026 (0.020) |
| de | 0.841 → 0.913 (0.921) | 0.942 → 0.953 (0.887) | 0.733 → 0.773 (0.517) | 0.055 → 0.044 (0.049) |
| it | 0.897 → 0.987 (0.929) | 0.962 → 0.971 (0.970) | 0.753 → 0.833 (0.913) | 0.036 → 0.025 (0.037) |
| pt | 0.769 → 0.960 (0.906) | 0.942 → 0.956 (0.938) | 0.780 → 0.827 (0.724) | 0.038 → 0.027 (0.020) |
| ru | 0.751 → 0.913 (0.761) | 0.854 → 0.872 (0.909) | 0.540 → 0.580 (0.342) | 0.073 → 0.048 (0.054) |
| ko | 0.964 → 0.960 (0.973) | 0.939 → 0.941 (0.937) | 0.764 → 0.767 (0.708) | 0.020 → 0.021 (0.045) |

Mandarin tone swaps (same syllable, different tone): AUC 0.804 → 0.816 (Azure 0.846); hit @ 5 % FA 0.307 → 0.3 (Azure 0.402); right word 0.247 → 0.253 (Azure 0.387).

speechocean762, L2-ARCTIC and EpaDB numbers are unchanged from round 3 (they use the regression heads and the sound probabilities, which round 4 does not touch); see `results/english_vs_azure.md`.
