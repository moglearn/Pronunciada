# Round 9: prompt-match channel for the sentence probability (not adopted)

**Outcome.** The channel works as a detector of gross prompt mismatch on FLEURS, but it does not
pass the adoption rule and it breaks a guard:

* **Dev gain too small.** cv-real DEV Test A AUC rises by +0.010 on average over the 11 locales.
  Without Korean (11 pairs per side) the mean is +0.006. The bar was +0.015.
* **speechocean762 TRAIN guard fails clearly.** Sentence-probability ECE goes 0.032 → 0.140 and
  AUC 0.983 → 0.953.

Nothing is shipped: there is no `pronunciada/data/calib/match.json`, so `SentenceResult.prob` is
unchanged and `match_prob` is `None`. The held-out sets (cv-real test, swap sentence rows,
speechocean762 test, OMPAL test) were **not run**, and every published number stands. No Azure
requests were made this round.

## Design

**Runtime.**
* `pronunciada/match.py` computes recognizer-side evidence per utterance (`evidence`). From it,
  `inputs` builds six clipped inputs:
  * insertion / substitution / deletion rates of the greedy decode against the reference (the
    round-5 disfluency definition);
  * share of words with `present_prob < 0.5`;
  * share of sounds with `llr_sub < −3`;
  * a "too few recognized tokens" term, max(0, −log(tokens heard / tokens expected)).
* `prob` is a logistic model on those inputs.

**Scorer change** (`pronunciada/scorer.py`, a few lines):
* The evidence is stored in the per-utterance cache.
* When `match.json` exists, `SentenceResult.prob = sentence head × P_match`, and the new
  `SentenceResult.match_prob` exposes P_match.
* `Scorer(..., match=False)` disables it.
* Word and sound outputs never read it.

**Fit** (`bench/calib/fit_match.py`): FLEURS **validation** only, 150 clips × 10 languages, pooled,
1500 rows per kind.

| kind | what is scored | label |
|---|---|---|
| own | the clip with its own sentence | match |
| slip | its own sentence with one word replaced | match (mild cases are kept up on purpose) |
| heavy | its own sentence with 30–60 % of words replaced / deleted / inserted | no match |
| other | another sentence of similar length | no match |
| noise | −50 dBFS white noise, own sentence | no match |
| trunc | the first third of the clip, own sentence | no match |

The model is a logistic regression. The negatives' total weight is `neg_weight` × the positives'.
It is monotone: an input whose coefficient comes out positive is dropped and the model refitted,
which happened to "too few tokens". `neg_weight` was selected on cv-real DEV from
{0.01, 0.03, 0.1, 0.25, 0.5, 1, 2}; the best was 0.5.

**Coefficients at 0.5:** intercept 5.65; insertion −5.70; substitution −4.94; deletion −1.38;
absent words −6.10; strong-against sounds −11.3; too-few-tokens 0. Full results:
`results/tables/match_round9.json`.

## FLEURS validation, leave-one-language-out

AUC match vs no match is ≥ 0.999 in every held-out language.

| held-out language | P_match own | slip | heavy | other / noise / trunc | native acceptance (prob ≥ 0.5) before → after |
|---|---|---|---|---|---|
| en | 0.985 | 0.954 | 0.177 | ≤ 0.001 | 0.993 → 0.987 |
| zh | 0.992 | 0.968 | 0.222 | ≤ 0.001 | 1.000 → 0.993 |
| ja | 0.979 | 0.953 | 0.171 | ≤ 0.002 | 1.000 → 1.000 |
| es | 0.994 | 0.976 | 0.289 | ≤ 0.001 | 1.000 → 1.000 |
| fr | 0.991 | 0.974 | 0.210 | ≤ 0.001 | 1.000 → 1.000 |
| de | 0.970 | 0.910 | 0.158 | ≤ 0.001 | 0.993 → 0.993 |
| it | 0.994 | 0.980 | 0.314 | ≤ 0.001 | 1.000 → 1.000 |
| pt | 0.989 | 0.972 | 0.344 | ≤ 0.001 | 1.000 → 1.000 |
| ru | 0.982 | 0.927 | 0.156 | ≤ 0.001 | 1.000 → 1.000 |
| ko | 0.986 | 0.938 | 0.252 | ≤ 0.001 | 1.000 → 1.000 |

One pooled model transfers to every language. On FLEURS, though, the existing sentence heads
already reject almost all gross mismatches: before the channel, acceptance is 0.000 for other,
0.004 for noise and 0.003 for trunc, and 0.151 for heavy (0.021 after). Native acceptance drops by
one clip of 150 in en and zh, 0.7 points each, which fails the ≤ 0.3-point guard at this sample
size.

## cv-real DEV (selection), ours only

Before = shipped sentence prob, after = × P_match. Paired speaker bootstrap, 1000 resamples.

| locale / test | n | before | after | Δ [95 % CI] | p | rejected / accepted with P_match < 0.5 |
|---|---|---|---|---|---|---|
| en A | 396 | 0.781 | 0.784 | +0.003 [−0.009, +0.016] | 0.59 | 24 % / 1.5 % |
| zh-CN A | 380 | 0.771 | 0.780 | +0.008 [−0.001, +0.018] | 0.07 | 29 % / 6 % |
| zh-TW A | 400 | 0.810 | 0.825 | +0.015 [+0.007, +0.024] | 0.004 | 31 % / 4 % |
| ja A | 388 | 0.763 | 0.771 | +0.008 [−0.003, +0.020] | 0.20 | 36 % / 10 % |
| es A | 398 | 0.757 | 0.761 | +0.004 [−0.003, +0.011] | 0.31 | 4 % / 0.5 % |
| fr A | 400 | 0.801 | 0.810 | +0.009 [−0.000, +0.018] | 0.06 | 16 % / 1 % |
| de A | 400 | 0.817 | 0.816 | −0.001 [−0.014, +0.012] | 0.92 | 16 % / 0 % |
| it A | 400 | 0.764 | 0.765 | +0.001 [−0.007, +0.010] | 0.78 | 16 % / 0.5 % |
| pt A | 400 | 0.768 | 0.771 | +0.003 [−0.016, +0.021] | 0.73 | 21 % / 1.5 % |
| ru A | 400 | 0.782 | 0.788 | +0.006 [−0.010, +0.023] | 0.57 | 22 % / 0 % |
| ko A | 22 | 0.752 | 0.810 | +0.058 [−0.017, +0.145] | 0.17 | 9 % / 0 % |
| ko A-weak | 54 | 0.775 | 0.796 | +0.021 [−0.037, +0.074] | 0.52 | 7 % / 0 % |
| en B | 192 | 0.578 | 0.574 | −0.004 | 0.65 | – |
| de B | 142 | 0.513 | 0.538 | +0.024 | 0.12 | – |

**Mean Test A Δ** is +0.0104 over 11 locales and +0.0056 without ko. No locale drops by more than
0.01, the worst being de −0.001, so that part of the rule holds. Still, the bar is missed by a wide
margin: 0.0104 against +0.015.

**Variants tried, all at the same plateau** (dev mean Test A Δ):

| variant | dev mean Test A Δ |
|---|---|
| negatives = heavy only | +0.0096 |
| negatives = heavy + other | +0.0103 |
| each input dropped in turn | +0.0015 to +0.0105 |
| coefficients × 0.5 / 2 / 3 | +0.006 / +0.007 / +0.004 |

The channel does what it was built for: 16–36 % of rejected clips are flagged against ≤ 10 % of
accepted ones. But most rejected Common Voice clips that the sentence heads pass are not gross
mismatches. They are one-word misreadings, insertions and accents. Accepted clips with noisy audio
carry the same recognizer evidence, so pushing harder costs as much as it gains. This is the
cross-language version of the round-8 Japanese finding.

## Guards

| guard | result |
|---|---|
| FLEURS validation natives, acceptance at s = 0.5, ≤ 0.3-point drop per language | **fails** for en and zh: −0.7 points each (1 clip of 150); 0 elsewhere |
| speechocean762 TRAIN sentence prob vs accuracy ≥ 7: ECE no worse than +0.5 points, AUC no drop | **fails**: ECE 0.032 → 0.140, AUC 0.983 → 0.953, mean p 0.792 → 0.671 |
| OMPAL DEV sentence metrics | not run (the channel was already rejected) |

The speechocean762 failure is the decisive one. Read-aloud L2 English has high substitution and
"strong evidence against" rates while reading the right sentence. So the recognizer-side evidence
calls ordinary accented speech a mismatch, which FLEURS natives never show. A version fitted with
L2 positives (e.g. speechocean762 train as extra "match" rows) is the obvious next step. It would
need its own selection.

## Byte-identity check

The tree with the round-9 code (no `match.json`) was compared with the committed `scorer.py` /
`types.py` on three FLEURS validation clips per language (30 clips). Every probability, word,
sound, `heard` string and sentence score is identical in 9 languages. In ru, two clips differ by
at most 2.3e-15. The committed code differs from itself between two runs by up to 4.7e-15 on the
same clips, so that is pre-existing floating-point run-to-run noise, not this change.

## Reproduce

    python -m bench.calib.fit_match collect --n 150     # FLEURS validation rows (~30 min, 3 workers)
    python -m bench.calib.fit_match dev                 # cv-real DEV halves, ours only (~5 min)
    python -m bench.calib.fit_match so762               # speechocean762 TRAIN guard rows (~4 min)
    python -m bench.calib.fit_match fit --neg-weight 0.5   # LOLO, dev table, guard; --write would ship match.json

Tests: `tests/test_match.py` covers evidence on synthetic posteriors, clipping, monotonicity, the
shipped-coefficient sign check (skipped while nothing is shipped), and the Scorer leaving words
untouched and multiplying the sentence prob.
