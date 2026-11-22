# Lingolix API measured by us, against Pronunciada

Lingolix: POST https://api.lingolix.com/api/pronunciation/v3/check; model_version 2026-11 (1700); tested 2026-11-22T16:06:24Z to 2026-11-22T18:55:28Z (UTC); 1700 responses.
Pronunciada outputs read from $PRONUNCIADA_CACHE/round19/integrator/frozen_cache (installed package 1.0.2); repository commit d7434aa0d7 (uncommitted changes); benchmark v2026.10 (e021b08).

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

FA@80: share of correct words flagged at the threshold that catches 80% of the planted errors (lower is better); its differences are in percentage points. Intervals: 1,000 resamples of utterances (test 1), speakers (test 2) or recordings (fleurs-swap); p two-sided, paired on the same resamples. Length errors caught: each system's own error flag on the 40 Japanese length items (Lingolix: report.py's flag, expected vs detected syllable IPA). Failed or unanswered requests: see the JSON.

