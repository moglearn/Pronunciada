# Lingolix API measured by us, against Pronunciada

Lingolix: POST https://api.lingolix.com/api/pronunciation/v3/check; model_version 2026-11 (200); tested 2026-11-22T16:06:24Z to 2026-11-22T16:09:59Z (UTC); 200 responses.
Pronunciada outputs read from $PRONUNCIADA_CACHE/round19/integrator/frozen_cache (installed package 1.0.2); repository commit be8155222e (uncommitted changes); benchmark v2026.10 (e021b08).

| test | metric | Pronunciada [95% CI] | Lingolix measured [95% CI] | Pronunciada − Lingolix [95% CI], p | Lingolix published |
|---|---|---|---|---|---|
| test 1 ja (v2026.10 rules) | word AUC | 0.923 [0.891, 0.951] | 0.951 [0.922, 0.978] | -0.028 [-0.064, +0.006], p 0.098 | 0.95 |
| test 1 ja (v2026.10 rules) | FA@80 | 9.7% [3.8, 20.6] | 3.2% [1.9, 4.7] | +6.5 [+0.7, +17.3], p 0.032 | 5% |
| test 1 ja (main rules) | word AUC | 0.930 [0.899, 0.957] | 0.958 [0.930, 0.983] | -0.028 [-0.063, +0.006], p 0.110 | — |
| test 1 ja (main rules) | FA@80 | 8.3% [3.1, 19.1] | 1.8% [0.8, 3.0] | +6.6 [+1.2, +17.2], p 0.020 | — |
| test 1 ja | length errors caught (of 40) | 27 | 38 | -11 [-16, -6], p 0.002 | 37 |

FA@80: share of correct words flagged at the threshold that catches 80% of the planted errors (lower is better); its differences are in percentage points. Intervals: 1,000 resamples of utterances (test 1), speakers (test 2) or recordings (fleurs-swap); p two-sided, paired on the same resamples. Length errors caught: each system's own error flag on the 40 Japanese length items (Lingolix: report.py's flag, expected vs detected syllable IPA). Failed or unanswered requests: see the JSON.

