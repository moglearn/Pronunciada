# Round 6: outside speech encoders (not adopted) and an embedding-based word score (adopted)

Round 6 was designed after the round-5 test evaluation. All design and selection used
speechocean762 TRAIN only, with 5-fold speaker-grouped CV × 3 seeds plus an age-band split. The
guard rail used FLEURS VALIDATION. Adoption bars were fixed beforehand by the coordinator and
were not lowered. Test was evaluated once, after freezing (`python -m bench.eval_english`,
cached Azure responses, no new Azure requests). The round-5 test outputs are kept in
`~/.cache/pronunciada/calib/api_round5_backup/`.

## 1. Outside pre-trained encoders for the sentence heads: not adopted

**Licences** (Hugging Face model cards, checked 2026-11-16):

| model | licence | used |
|---|---|---|
| `openai/whisper-tiny` | Apache-2.0 | yes |
| `openai/whisper-base` | Apache-2.0 | yes |
| `facebook/wav2vec2-base` | Apache-2.0 | yes |
| `facebook/hubert-base-ls960` | Apache-2.0 | yes |
| `microsoft/wavlm-base`, `microsoft/wavlm-base-plus` | no licence on the card | skipped |

**Extraction** (`bench/calib/ext_encoders.py`, torch on MPS, 1 process, about 1 min per model):

- Every hidden layer is pooled to mean ++ std over the speech frames of the ZIPA alignment.
- Whisper runs without the 30 s padding: the clip's own frames plus the first T positional
  embeddings, which is the only form a short-clip runtime could afford.

**Pipeline** (`bench/calib/train_ext.py`):

- Each candidate is the round-5 configuration plus one outside source: a layer, or the
  top-2 layers concatenated.
- The outside source gets its own in-fold ridge projection onto the four targets, feeding the
  same trees + ridge blend.
- Layers were shortlisted by ridge-only out-of-fold Pearson (`results/tables/round6_ext.json`,
  `screen`). The best single layers on prosodic were whisper-base L3 / L4 (0.830 / 0.831),
  wav2vec2 L7 (0.824), hubert L10 (0.826) and whisper-tiny L3 (0.822). ZIPA stack 1 alone
  scores 0.811.

**Adoption bar:** prosodic ≥ round 5 + 0.015 and total not lower.

3-seed CV. Each cell is the mean, followed by (paired Δ vs round 5 ± SE):

| config | accuracy | fluency | prosodic | total | clears bar |
|---|---|---|---|---|---|
| round 5 (shipped) | 0.8228 | 0.8637 | 0.8570 | 0.8434 | – |
| + whisper-tiny L3 | 0.8242 (+0.0015 ± 0.0009) | 0.8681 (+0.0044 ± 0.0013) | 0.8616 (+0.0046 ± 0.0012) | 0.8454 (+0.0019 ± 0.0009) | no |
| + whisper-tiny L2 | 0.8221 (−0.0007 ± 0.0005) | 0.8659 (+0.0022 ± 0.0006) | 0.8588 (+0.0018 ± 0.0010) | 0.8443 (+0.0009 ± 0.0007) | no |
| + whisper-tiny L2+3 | 0.8237 (+0.0009 ± 0.0006) | 0.8677 (+0.0040 ± 0.0010) | 0.8618 (+0.0047 ± 0.0014) | 0.8455 (+0.0021 ± 0.0009) | no |
| + whisper-base L3 | 0.8278 (+0.0051 ± 0.0003) | 0.8694 (+0.0057 ± 0.0011) | 0.8641 (+0.0071 ± 0.0007) | 0.8496 (+0.0062 ± 0.0006) | no |
| + whisper-base L4 | 0.8283 (+0.0055 ± 0.0003) | 0.8690 (+0.0053 ± 0.0010) | 0.8640 (+0.0069 ± 0.0010) | 0.8492 (+0.0057 ± 0.0007) | no |
| + whisper-base L3+4 | 0.8301 (+0.0074 ± 0.0002) | 0.8703 (+0.0066 ± 0.0009) | 0.8658 (+0.0088 ± 0.0017) | 0.8509 (+0.0075 ± 0.0005) | no |
| + wav2vec2-base L7 | 0.8263 (+0.0035 ± 0.0007) | 0.8667 (+0.0030 ± 0.0003) | 0.8634 (+0.0064 ± 0.0006) | 0.8478 (+0.0043 ± 0.0011) | no |
| + wav2vec2-base L9 | 0.8269 (+0.0042 ± 0.0004) | 0.8676 (+0.0039 ± 0.0007) | 0.8629 (+0.0059 ± 0.0010) | 0.8486 (+0.0052 ± 0.0007) | no |
| + wav2vec2-base L7+9 | 0.8269 (+0.0042 ± 0.0003) | 0.8683 (+0.0046 ± 0.0002) | 0.8649 (+0.0079 ± 0.0007) | 0.8491 (+0.0057 ± 0.0006) | no |
| + hubert-base L10 | 0.8267 (+0.0039 ± 0.0008) | 0.8683 (+0.0047 ± 0.0005) | 0.8638 (+0.0068 ± 0.0007) | 0.8488 (+0.0054 ± 0.0016) | no |
| + hubert-base L11 | 0.8281 (+0.0053 ± 0.0010) | 0.8680 (+0.0043 ± 0.0005) | 0.8626 (+0.0056 ± 0.0009) | 0.8490 (+0.0055 ± 0.0005) | no |
| + hubert-base L10+11 | 0.8287 (+0.0059 ± 0.0006) | 0.8682 (+0.0045 ± 0.0005) | 0.8633 (+0.0063 ± 0.0012) | 0.8499 (+0.0064 ± 0.0012) | no |

**Result: none clears the bar.** The best was whisper-base L3+4 at +0.009 prosodic. Every
outside source helps a little and consistently (+0.002 to +0.009 on all heads). However, most
of what these models carry about delivery is already in the ZIPA stack-1 projection and the
disfluency features.

The sentence heads stay as shipped in round 5. Nothing was exported to ONNX, so there is no
size or latency to report for an outside encoder. The age-band check (part 3) was not needed
for the sentence level, since nothing changed there.

## 2. Word 0–10 score: adopted

**Method** (`bench/calib/train_word_emb.py`):

- The ZIPA stack-1 activations are pooled (mean ++ std) over each word's frames
  (`delivery.word_frame_spans` / `pool_words`; frames from the word's first sound start to its
  last sound end).
- A ridge projection onto word accuracy is fitted in-fold (inner out-of-fold values for the
  trees' training rows).
- It is added to the shipped word-regression inputs (`WORD_NUM`, rebuilt exactly; a refit
  reproduces `word_reg.npz` to 3e-8), using the same s2 trees.

**Bar:** word Pearson improves by ≥ 0.010.

| variant | speaker CV, 3 seeds (Δ ± SE) | age-band split, s2 | age-band split, s3 (more regularised) |
|---|---|---|---|
| shipped inputs | 0.6491 | 0.4536 | 0.4407 |
| + word-pooled projection | 0.6735 (+0.0245 ± 0.0017) | **0.4708** | 0.4542 |
| + word-pooled + utterance-pooled projection | 0.6854 (+0.0364 ± 0.0010) | 0.4647 | 0.4510 |

The age-band split trains on adults (age ≥ 15) and tests on children, then the reverse; the
table shows pooled out-of-fold Pearson. Two observations from it:

- **The utterance context does not survive the harsh split.** It wins on speaker CV but loses
  on the age-band split.
- **More regularised trees are worse.** s3 is below s2 in every row.

**Chosen:** the word-pooled projection only, with s2 trees, shipped as `word_reg_r6.npz`.

**What it changes:**

- It sets only the reported 0–10 word score (`Scorer.details()["word_accuracy"]`).
- The sentence level still reads the round-2 word regression (cached as `w_reg_r2`), so the
  sentence inputs are unchanged.
- The word acceptance heads and every probability are untouched.

**Native check** (100 FLEURS-validation clips per language, `bench/calib/native_word_check.py`):
the median word score drops by 0.02–0.13 in every language, all within the 0.3 rule. It is
therefore on for all ten languages (config `word_score_model`). Native medians sit at about
9.9 / 10, so this check is weak: it shows the score does not get harsher on native speech,
not that it is better outside English.

## speechocean762 TEST: round 5 → round 6 (Pearson, speaker-cluster bootstrap 95 % CI; Δ = ours − Azure, paired bootstrap, two-sided p)

| level | round 5 | round 6 | Azure | round 6 − Azure |
|---|---|---|---|---|
| phone | 0.660 | 0.660 [0.581, 0.712] | 0.538 [0.458, 0.600] | +0.122 [+0.099, +0.141], p = 0.002 |
| word | 0.585 | **0.609** [0.533, 0.663] | 0.589 [0.515, 0.647] | +0.020 [−0.022, +0.059], p = 0.33 |
| sentence accuracy | 0.761 | 0.761 [0.703, 0.804] | 0.673 [0.613, 0.720] | +0.088 [+0.052, +0.123], p = 0.002 |
| sentence fluency | 0.792 | 0.792 [0.717, 0.846] | 0.708 [0.628, 0.760] | +0.084 [0.000, +0.175], p = 0.05 |
| sentence prosodic | 0.782 | 0.782 [0.709, 0.836] | 0.827 [0.788, 0.854] | −0.045 [−0.115, +0.007], p = 0.16 |
| sentence total | 0.773 | 0.773 [0.717, 0.820] | 0.796 [0.743, 0.834] | −0.023 [−0.074, +0.021], p = 0.41 |

- **Word improved; Pearson is now ahead of Azure, not significantly.** Pearson rose
  0.585 → 0.609 (CV gain +0.024, test gain +0.024). On Spearman the word score is level with
  Azure (0.415 vs 0.417).
- **Sentence rows** are identical to round 5, as designed.
- **L2-ARCTIC / EpaDB** error-detection AUCs are unchanged (0.712 / 0.721).
- **int8:** word 0.611; sentence scores 0.760 / 0.792 / 0.782 / 0.773.

## Parity, size, speed, tests

- **Parity:** `results/tables/delivery_parity.json`, 200 train clips through the default
  public `Scorer`. The word-pooled projection input matches the training cache exactly (max
  abs 0.0), as do all round-5 sentence inputs.
- **Size:** `word_reg_r6.npz` is 32 kB. No new model files.
- **Latency** for a 3.5 s clip, median of 100: fp32 4 threads 49.6 ms (round 5: 53.0 ms on a
  busier machine), int8 4 threads 40.2 ms. The word pooling itself costs well under 1 ms.
- **Tests:** `pytest -q` passes 424 tests, including 2 new ones in `tests/test_delivery.py`.
  The suite includes the Mandarin tone agent's in-progress tests; none failed.

## Files

- **Code:**
  - `pronunciada/delivery.py`: `word_frame_spans`, `pool_words`
  - `pronunciada/scorer.py`: config `word_score_model`, `w_reg` / `w_reg_r2`
- **Data:**
  - `pronunciada/data/calib/word_reg_r6.npz`
  - `pronunciada/data/calib/config.json`
- **Bench scripts** (`bench/calib/`): `ext_encoders.py`, `train_ext.py`, `train_word_emb.py`,
  `native_word_check.py`; `delivery_extract.py` and `parity_delivery.py` were extended.
- **Tables** (`results/tables/`): `round6_ext.json`, `round6_word_ablation.json`,
  `round6_word_harsh.json`, `round6_word_fit.json`, `round6_native_word.json`.
- **Dependency:** `transformers` 4.57.1 was added to the venv for bench extraction only; it is
  not a runtime dependency.
