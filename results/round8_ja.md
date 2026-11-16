# Round 8: Japanese on cv-real (not adopted)

**Outcome.** None of the three hypotheses is supported by the dev data, and nothing tried came close
to the +0.02 dev bar. Nothing in `pronunciada/` changed. The held-out sets (cv-real ja test, the
FLEURS-ja swap and ten-language rows) were **not run**, so the one held-out look is still unused,
and every Japanese row in README.md, cv_real.md, languages.md and azure_swap.md stands. The FLEURS
guards were not needed because no candidate reached them.

All analysis uses the cv-real **ja dev half** only: 194 rejected and 194 accepted clips, with speakers
disjoint from the test half. No test-half clip or output was inspected.

## Tooling added

* `python -m bench.eval_cv --half dev ...` scores the dev half and sends it to Azure. It uses its
  own caches (`cv-real-dev/ours/`, `azure/cv-real-dev/`) and writes only
  `results/tables/cv_real_dev.json`, never `cv_real.md`. The default (test) behaviour, caches and
  `latest.json` are untouched. Tests: `tests/test_eval_cv_half.py`.
* `python -m bench.ja_round8 dump | analyse | altread` keeps per-word evidence for every dev clip
  (phones, what was heard, greedy decode, probabilities). It also builds the cause breakdown
  (`results/tables/ja_round8_base3.json`) and runs the alternative-reading probe.

## Dev baseline, ours vs Azure (like for like)

| ja dev, Test A | AUC [95 % CI, speaker bootstrap] |
|---|---|
| Pronunciada (`SentenceResult.prob`) | 0.763 [0.712, 0.811] |
| Azure (`PronScore`) | 0.753 [0.703, 0.800] |
| ours − Azure | +0.010 [−0.031, +0.048], p = 0.66 |

On dev we are level with or slightly ahead of Azure, while test has Azure ahead by 0.034
(p = 0.085). Both CIs are about ±0.05 wide, so the test gap is consistent with noise between the
halves and is not a stable Japanese weakness. Secondary scores on dev: ours `total` 0.778,
`accuracy` 0.766; Azure Accuracy 0.729, Completeness 0.709. The length-only baselines are 0.505
and 0.514.

## Cause breakdown (dev)

**Clip level.** PER is the normalised edit distance between the greedy phone decode and the
reference.

| | accepted (194) | rejected (194) |
|---|---|---|
| clips with a word < 0.5 | 62 | 133 |
| sentence prob < 0.5 | 16 | 53 |
| sentence prob ≥ 0.95 | 142 | 56 |
| PER < 0.2 / 0.2–0.4 / 0.4–0.6 / > 0.6 | 109 / 63 / 11 / 11 | 35 / 78 / 30 / 51 |
| near-empty decode (≤ 2 phone tokens) | 5 | 11 |
| ↳ of which sentence prob ≥ 0.5 | 1 | 4 |

**Words below 0.5, by cause** (first matching rule, `bench.ja_round8.word_cause`):

| cause | accepted | rejected | examples (accepted side unless noted) |
|---|---|---|---|
| clip decode does not match the prompt (PER > 0.6) | 212 | 1011 | 見物人… read as "i" in a 2.0 s clip; noisy or misassigned audio |
| word not heard (present < 0.5) | 56 | 133 | で, いる, て, は |
| other substitution | 46 | 83 | わたし heard "bafaeb", 八百屋 heard "taka" |
| another dictionary reading fits the decode better | 16 | 11 | 身体 シンタイ heard "karada", 角 カド heard "kakku"; rejected side: 側 ガワ heard "soba", 下 モト heard "ʃita" |
| i/u devoiced or dropped | 15 | 19 | 飛び出し, 向け |
| particle は/へ/を | 4 | 5 | は heard "ha" |
| long vowel | 3 | 4 | 大きな heard "okika" |
| moraic nasal | 3 | 3 | みんな heard "mina" |
| katakana loanword | 4 | 2 | リニューアル |

Most low words on accepted clips (212 of 359) sit in clips whose audio does not decode as the
prompt at all. These include recordings far too short for the text, as well as noise. Azure scores
most of these low too. Every Japanese-specific cause is small and about as common on the rejected
side.

## Hypothesis 1: kanji reading ambiguity (refuted)

* **Breakdown.** Only 16 low accepted words are better explained by another dictionary reading,
  against 11 on rejected clips.
* **Probe** (`bench.ja_round8 altread`). For every kanji word, each other SudachiDict reading of
  the same written form was substituted into the prompt and scored. The probe kept the best
  sentence over the original and all alternatives: 1331 alternative prompts over 299 clips.
* **Result.** Best-of AUC falls from 0.763 to **0.758**. An alternative reading raises the score by
  more than 0.05 on 35 rejected clips but only on 9 accepted ones. The rejected side has real
  misreadings that are dictionary-attested readings of the same characters: 十 ジュッ, 町 チョウ
  for まち, 県 アガタ, 下手 ベタ, 一宮 イチミヤ, 墓石 ボセキ. Tolerating readings this way hides the
  very errors reviewers reject, so the word-level N-best design was not built.

## Hypothesis 2: colloquial variation (not supported)

Devoicing, long vowels, は/へ/を and ん together explain 25 low accepted words, against 31 on
rejected clips. Loosening them would raise both sides, so no reference change was made.

## Hypothesis 3: rejected clips we pass (partly true, not fixable by a rule)

**What is true.**
* Sentence prob ignores the omission channel. For example, 人間は考える葦である has an empty
  decode and four of six words with present < 0.5, yet scores 0.64.
* Local misreadings and insertions are accepted:
  * 北本 read "hoku-kitamoto": the insertion is invisible.
  * 磐田 heard "ebada": phones with llr_sub −5.9 and peak posterior 0.01 still get p ≈ 0.9.
  * 鰺ヶ沢 is missing its final "wa".

**Rules tried on dev, all failed:**

| change | dev AUC |
|---|---|
| baseline | 0.763 |
| × product / mean / min of word present_prob | 0.735 / 0.757 / 0.736 |
| × share of words present | 0.760 |
| × geometric mean / mean / min word prob | 0.753 / 0.755 / 0.746 |
| floor for near-empty decodes (≤ 0–5 tokens) | 0.764–0.756 |
| ja sound bias −1.5 / −0.5 / +1.0 (shipped +0.25) | 0.766 / 0.763 / 0.765 |
| sentence strictness 0 … 1 | 0.761–0.763 |
| rank blend with clip PER | ≤ 0.773 |

Raw clip-level aggregates also land at the same ceiling:

| aggregate | dev AUC |
|---|---|
| mean llr_sub | 0.765 |
| 10th-percentile llr_sub | 0.764 |
| share of llr_sub < −2 | 0.762 |
| clip PER | 0.763 |

The penalties fail because accepted Common Voice clips show the same omission and substitution
evidence (fast function words, noise, accents). The only thing above +0.02 was a logistic model
over the sentence prob and the decoder insertion / substitution / deletion rates, **fitted on dev
itself**. In speaker-grouped 5-fold CV it reached 0.785 (+0.022). That is selection and fitting
on the same 388 clips, right at the bar, so it was not adopted. Fitting such a recalibration on
planted-error FLEURS-ja validation and then checking it on dev is the obvious next step.

## Azure usage

388 ja dev clips sent once each: 388 HTTP requests, no retries or failures, 0.55 h of audio
(ja-JP). Only public CC0 Common Voice audio and prompts were sent. Responses are cached in
`$PRONUNCIADA_CACHE/azure/cv-real-dev/`. No test-half or other requests were made.
