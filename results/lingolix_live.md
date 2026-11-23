# Lingolix API measured by us, against Pronunciada

Lingolix: POST https://api.lingolix.com/api/pronunciation/v3/check; model_version 2026-11 (3700); tested 2026-11-22T16:06:24Z to 2026-11-22T22:00:55Z (UTC); 3700 responses.
Pronunciada outputs read from $PRONUNCIADA_CACHE/round19/integrator/frozen_cache (installed package 1.0.2); repository commit 39e3cdc4aa; benchmark v2026.10 (e021b08).

| test | metric | Pronunciada [95% CI] | Lingolix measured [95% CI] | Pronunciada − Lingolix [95% CI], p | Lingolix published |
|---|---|---|---|---|---|
| test 1 en (v2026.10 rules) | word AUC | 0.977 [0.963, 0.988] | 0.982 [0.966, 0.993] | -0.005 [-0.020, +0.013], p 0.489 | 0.97 |
| test 1 en (v2026.10 rules) | FA@80 | 2.5% [1.0, 3.9] | 0.8% [0.4, 1.7] | +1.7 [+0.2, +2.8], p 0.020 | 1% |
| test 1 fr (v2026.10 rules) | word AUC | 0.960 [0.924, 0.987] | 0.968 [0.947, 0.986] | -0.008 [-0.042, +0.018], p 0.585 | 0.96 |
| test 1 fr (v2026.10 rules) | FA@80 | 1.7% [0.9, 3.1] | 1.5% [0.9, 3.0] | +0.1 [-0.8, +1.4], p 0.853 | 1% |
| test 1 es (v2026.10 rules) | word AUC | 0.985 [0.976, 0.993] | 0.977 [0.957, 0.991] | +0.008 [-0.003, +0.024], p 0.184 | 0.97 |
| test 1 es (v2026.10 rules) | FA@80 | 1.4% [0.8, 2.3] | 1.1% [0.6, 1.9] | +0.2 [-0.4, +0.8], p 0.462 | 2% |
| test 1 de (v2026.10 rules) | word AUC | 0.967 [0.945, 0.985] | 0.973 [0.951, 0.989] | -0.005 [-0.029, +0.019], p 0.657 | 0.95 |
| test 1 de (v2026.10 rules) | FA@80 | 2.1% [0.9, 5.4] | 1.0% [0.4, 6.0] | +1.0 [-1.9, +3.6], p 0.196 | 2% |
| test 1 ja (v2026.10 rules) | word AUC | 0.923 [0.891, 0.951] | 0.951 [0.922, 0.978] | -0.028 [-0.064, +0.006], p 0.098 | 0.95 |
| test 1 ja (v2026.10 rules) | FA@80 | 9.7% [3.8, 20.6] | 3.2% [1.9, 4.7] | +6.5 [+0.7, +17.3], p 0.032 | 5% |
| test 1 ko (v2026.10 rules) | word AUC | 0.957 [0.935, 0.977] | 0.964 [0.940, 0.984] | -0.007 [-0.021, +0.007], p 0.270 | 0.95 |
| test 1 ko (v2026.10 rules) | FA@80 | 3.1% [1.9, 9.7] | 2.4% [1.6, 5.2] | +0.7 [-1.3, +5.9], p 0.657 | 3% |
| test 1 en (main rules) | word AUC | 0.981 [0.968, 0.991] | 0.984 [0.969, 0.994] | -0.003 [-0.018, +0.014], p 0.613 | — |
| test 1 en (main rules) | FA@80 | 1.9% [0.6, 3.2] | 0.6% [0.3, 1.3] | +1.3 [-0.1, +2.4], p 0.088 | — |
| test 1 fr (main rules) | word AUC | 0.963 [0.926, 0.989] | 0.971 [0.949, 0.988] | -0.008 [-0.042, +0.018], p 0.599 | — |
| test 1 fr (main rules) | FA@80 | 1.1% [0.6, 2.2] | 1.3% [0.7, 2.2] | -0.1 [-0.7, +0.9], p 0.967 | — |
| test 1 es (main rules) | word AUC | 0.987 [0.979, 0.994] | 0.980 [0.959, 0.994] | +0.007 [-0.004, +0.023], p 0.284 | — |
| test 1 es (main rules) | FA@80 | 1.0% [0.4, 1.8] | 0.8% [0.3, 1.4] | +0.3 [-0.2, +0.7], p 0.414 | — |
| test 1 de (main rules) | word AUC | 0.968 [0.946, 0.986] | 0.976 [0.954, 0.991] | -0.008 [-0.031, +0.017], p 0.553 | — |
| test 1 de (main rules) | FA@80 | 2.0% [0.9, 4.8] | 0.6% [0.2, 4.8] | +1.4 [-1.4, +3.5], p 0.128 | — |
| test 1 ja (main rules) | word AUC | 0.930 [0.899, 0.957] | 0.958 [0.930, 0.983] | -0.028 [-0.063, +0.006], p 0.110 | — |
| test 1 ja (main rules) | FA@80 | 8.3% [3.1, 19.1] | 1.8% [0.8, 3.0] | +6.6 [+1.2, +17.2], p 0.020 | — |
| test 1 ko (main rules) | word AUC | 0.960 [0.939, 0.979] | 0.968 [0.944, 0.987] | -0.008 [-0.021, +0.007], p 0.258 | — |
| test 1 ko (main rules) | FA@80 | 2.6% [1.5, 9.1] | 2.1% [1.3, 4.3] | +0.5 [-1.1, +5.8], p 0.587 | — |
| test 1 ja | length errors caught (of 40) | 27 | 38 | -11 [-16, -6], p 0.002 | 37 |
| test 2 (speechocean762) | word Pearson | 0.581 [0.489, 0.644] | 0.527 [0.400, 0.630] | +0.054 [-0.029, +0.128], p 0.236 | 0.51 |
| test 2 (speechocean762) | sentence Pearson | 0.788 [0.706, 0.847] | 0.713 [0.575, 0.792] | +0.075 [-0.014, +0.174], p 0.100 | 0.74 |

FA@80: share of correct words flagged at the threshold that catches 80% of the planted errors (lower is better); its differences are in percentage points. Intervals: 1,000 resamples of utterances (test 1), groups of clips sharing the first four digits of the utterance id (test 2; close to speakers but not identical, see the speaker resampling below) or recordings (fleurs-swap); p two-sided, paired on the same resamples. Length errors caught: each system's own error flag on the 40 Japanese length items (Lingolix: report.py's flag, expected vs detected syllable IPA). Failed or unanswered requests: see the JSON.

## speechocean762, more speakers

All 2,500 speechocean762 test clips (125 speakers, 20 clips each) as test 2 items, built with build_tier2.py's rules; the builder reproduces 500 of the benchmark's 500 items exactly. Lingolix gets the same request as in test 2, so the benchmark's 500 clips were not sent again. Pronunciada's answers (read from $PRONUNCIADA_CACHE/third_party/lingolix/responses/pronunciada/tier2x.jsonl) equal those of test 2 above on all 500 clips.

| clips | metric | Pronunciada [95% CI] | Lingolix measured [95% CI] | Pronunciada − Lingolix [95% CI], p | p, test 2 resampling |
|---|---|---|---|---|---|
| a. the benchmark's 500 clips (500 clips, 124 speakers) | sentence Pearson | 0.788 [0.731, 0.839] | 0.713 [0.606, 0.785] | +0.075 [-0.008, +0.168], p 0.074 | 0.100 |
| a. the benchmark's 500 clips (500 clips, 124 speakers) | word Pearson | 0.581 [0.486, 0.655] | 0.527 [0.403, 0.629] | +0.054 [-0.021, +0.129], p 0.163 | 0.236 |
| b. every clip with a Lingolix answer (2,500 clips, 125 speakers) | sentence Pearson | 0.792 [0.741, 0.833] | 0.668 [0.581, 0.736] | +0.124 [+0.057, +0.190], p < 0.001 | 0.002 |
| b. every clip with a Lingolix answer (2,500 clips, 125 speakers) | word Pearson | 0.573 [0.504, 0.630] | 0.490 [0.409, 0.560] | +0.083 [+0.043, +0.123], p < 0.001 | 0.004 |
| c. complete speakers only (125 of 125 speakers with all 20 clips answered, 2,500 clips) | sentence Pearson | 0.792 [0.741, 0.833] | 0.668 [0.581, 0.736] | +0.124 [+0.057, +0.190], p < 0.001 | — |
| c. complete speakers only (125 of 125 speakers with all 20 clips answered, 2,500 clips) | word Pearson | 0.573 [0.504, 0.630] | 0.490 [0.409, 0.560] | +0.083 [+0.043, +0.123], p < 0.001 | — |

p: two-sided, from 10,000 resamples of speakers (with replacement; both systems on the same resample; the intervals come from the same resamples). p, test 2 resampling: the method of the test 2 rows above (1,000 resamples of groups formed by the first four digits of the utterance id), repeated for continuity.

To finish the split: 0 of 2,500 clips not yet sent, 0.00 audio minutes. If all of it is billed as overage at the Starter rate (EUR 0.05 per minute): EUR 0.00, or EUR 0.00 for 0 whole minutes. Source: https://lingolix.com/en/pricing (read 2026-11-22): Starter plan, overage billed per full minute used; usage measured by the second, only full minutes charged.

## Mandarin

2026-11-22: Lingolix does not accept Mandarin. One OMPAL clip (2.5 s) sent with `language_code=zh` at 20:33:46 UTC was refused with HTTP 400 and the message "Language 'zh' not supported", followed by the supported codes: bg, cs, da, de, el, en, es, fi, fr, hu, it, ja, ko, nb, nl, pl, pt, ru, sv, tr, uk. Nothing was billed. The OMPAL (Mandarin) comparison could therefore not be run.

