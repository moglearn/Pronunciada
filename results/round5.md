# Round 5: delivery inputs for the English sentence scores

I designed round 5 after the earlier held-out evaluations had shown that we were behind Azure
on speechocean762 sentence prosodic and total. All design, feature selection and fitting used
speechocean762 TRAIN only, plus FLEURS VALIDATION for the other-language guard rail. Once the
configuration was frozen I evaluated speechocean762 TEST once (`python -m bench.eval_english`,
cached Azure responses, no new Azure requests). Nothing was changed after that. The round-4
tables are kept in `results/tables/round4/`.

## What changed

1. **Delivery features**: `pronunciada/delivery.py`, numpy only. There is one function per
   utterance, computed from the waveform, the fbank, the adapted log-posteriors, the alignment and
   the shared F0 track. It covers 40 features in five groups (rhythm, pauses, pitch, loudness,
   recogniser disfluency evidence), plus mean and std pooling of encoder activations over speech
   frames.
2. **Encoder activations**: four intermediate tensors (stacks 1, 3, 5 and the encoder output
   before the CTC layer) are added as extra outputs of the ZIPA ONNX graph. This is done by
   editing the model protobuf directly (`acoustic.add_graph_outputs`), so the runtime does not
   need the `onnx` package. The modified model is written next to the original in the cache as
   `model.emb.onnx` (fp32) and `model.int8.emb.onnx` (int8).
   - Log-probs are bit-identical for fp32 and int8 (unit test).
   - Model latency overhead is about 0.3 ms.
   - The graph passes `onnx.checker`.
   - If the modified model cannot be built, the scorer falls back to the plain model and the
     embedding inputs become NaN.
3. **Sentence regression heads**: `bench/calib/train_delivery.py` retrains `reg_accuracy`,
   `reg_fluency`, `reg_prosodic` and `reg_total` in `pronunciada/data/calib/sentence.npz`.
   - Inputs: `SENT_NUM`, the 7 disfluency features, and 4 ridge projections of the pooled
     stack-1 activations (`emb_accuracy` … `emb_total`). The projection is fitted in-fold and
     shipped as plain matrices inside `sentence.npz` (`proj/*`).
   - Output: 0.7 × trees + 0.3 × ridge. The ridge model uses the same inputs, with NaN replaced
     by the train median and every input clipped to its train 1–99 % range (`lin/*`).
   - The acceptance heads (`len`, `str`) are byte-for-byte the round-4 arrays. The script
     asserts that they give identical outputs.
   - Sound and word levels are untouched.
4. **Per-language choice**: config `sentence_model`. Eight languages keep `sentence_r4.npz`
   (guard rail below). For those languages the scorer computes no delivery features and loads
   the plain acoustic model.
5. **Shared pitch track**: `ProsodyScorer.score(..., track=)` reuses the pitch track. The pitch
   group was not adopted, so the track is only computed when the prosody channel needs it.

**Baseline correction.** `results/tables/calib_levels.json`, which the task quoted (CV 0.809 /
0.849 / 0.831 / 0.824), comes from `train_levels.py`. That is not the shipped sentence model,
which `bench/calib/train_word2.py` trained in round 2 on round-2 word inputs. My first ablation
mirrored `train_levels.py`. After I found this, I rebuilt the inputs exactly as `train_word2.py`
builds them (`round2_sentence_inputs()`). That rebuild reproduces the shipped trees to 1e-8 and
the round-2 CV of 0.8075 / 0.8494 / 0.8311 / 0.8217. I then reran the ablation and refitted.
Only the rerun is reported below.

## CV ablation (speechocean762 train, 5-fold speaker-grouped, 3 seeds)

Each cell is the mean Pearson over 3 seeds, followed by (paired Δ vs base ± standard error over
seeds). The model is trees only, s2 hyperparameters.

**Adoption rule:** a group is adopted if both of these hold:

- prosodic or total improves by Δ ≥ 0.003 and by more than 2 SE;
- no head has Δ ≤ −0.002.

**Embedding layer:** I took the best `emb_*` configuration by mean prosodic/total Δ, which was
stack 1 (`enc1`).

| config | accuracy | fluency | prosodic | total | adopted |
|---|---|---|---|---|---|
| base (round-4 inputs) | 0.8068 | 0.8457 | 0.8296 | 0.8255 | – |
| + rhythm | 0.8092 (+0.0024 ± 0.0010) | 0.8518 (+0.0062 ± 0.0003) | 0.8333 (+0.0037 ± 0.0004) | 0.8278 (+0.0023 ± 0.0004) | yes |
| + pauses | 0.8121 (+0.0054 ± 0.0010) | 0.8501 (+0.0044 ± 0.0002) | 0.8334 (+0.0037 ± 0.0002) | 0.8284 (+0.0029 ± 0.0005) | yes |
| + pitch | 0.8080 (+0.0013 ± 0.0004) | 0.8429 (−0.0027 ± 0.0012) | 0.8269 (−0.0027 ± 0.0015) | 0.8248 (−0.0007 ± 0.0008) | no |
| + loudness | 0.8095 (+0.0028 ± 0.0007) | 0.8485 (+0.0029 ± 0.0015) | 0.8338 (+0.0041 ± 0.0016) | 0.8294 (+0.0039 ± 0.0009) | yes |
| + disfluency | 0.8120 (+0.0052 ± 0.0008) | 0.8520 (+0.0063 ± 0.0001) | 0.8364 (+0.0067 ± 0.0003) | 0.8300 (+0.0045 ± 0.0008) | yes |
| + emb, final output | 0.8082 (+0.0014 ± 0.0008) | 0.8577 (+0.0121 ± 0.0006) | 0.8449 (+0.0153 ± 0.0006) | 0.8288 (+0.0033 ± 0.0003) | yes |
| + emb, stack 3 | 0.8083 (+0.0015 ± 0.0009) | 0.8583 (+0.0127 ± 0.0021) | 0.8459 (+0.0162 ± 0.0009) | 0.8283 (+0.0027 ± 0.0008) | yes |
| + emb, stack 1 | 0.8144 (+0.0076 ± 0.0009) | 0.8574 (+0.0118 ± 0.0012) | 0.8496 (+0.0200 ± 0.0005) | 0.8342 (+0.0086 ± 0.0006) | yes |
| + emb, stack 5 | 0.8064 (−0.0003 ± 0.0021) | 0.8551 (+0.0094 ± 0.0015) | 0.8434 (+0.0138 ± 0.0007) | 0.8268 (+0.0013 ± 0.0010) | yes |
| + emb, all four | 0.8107 (+0.0039 ± 0.0008) | 0.8601 (+0.0145 ± 0.0009) | 0.8496 (+0.0200 ± 0.0010) | 0.8323 (+0.0068 ± 0.0003) | yes |
| + emb, stack 3 + final | 0.8085 (+0.0017 ± 0.0015) | 0.8601 (+0.0144 ± 0.0012) | 0.8472 (+0.0176 ± 0.0006) | 0.8299 (+0.0043 ± 0.0009) | yes |
| + all five hand-made groups | 0.8147 (+0.0080 ± 0.0017) | 0.8580 (+0.0123 ± 0.0004) | 0.8417 (+0.0120 ± 0.0015) | 0.8351 (+0.0096 ± 0.0011) | (reference) |

**Forward selection.** I added the qualifying groups one at a time, largest gain first. Each Δ
is measured against the previous accepted set.

| step | accuracy | fluency | prosodic | total | adopted |
|---|---|---|---|---|---|
| base + emb (stack 1) | 0.8144 (+0.0076) | 0.8574 (+0.0118) | 0.8496 (+0.0200) | 0.8342 (+0.0086) | yes |
| + disfluency | 0.8182 (+0.0038) | 0.8596 (+0.0022) | 0.8525 (+0.0029) | 0.8390 (+0.0048) | yes |
| + loudness | 0.8183 (+0.0002) | 0.8608 (+0.0012) | 0.8546 (+0.0021) | 0.8408 (+0.0019) | no |
| + pauses | 0.8214 (+0.0032) | 0.8627 (+0.0031) | 0.8552 (+0.0027) | 0.8418 (+0.0028) | no (just below 0.003) |
| + rhythm | 0.8179 (−0.0003) | 0.8638 (+0.0042) | 0.8540 (+0.0015) | 0.8403 (+0.0013) | no |

**Blend with ridge.** The shipped configuration, 3-seed CV:

| model | accuracy | fluency | prosodic | total |
|---|---|---|---|---|
| trees only | 0.8182 | 0.8596 | 0.8525 | 0.8390 |
| 0.7 × trees + 0.3 × ridge (shipped) | 0.8228 | 0.8637 | 0.8570 | 0.8434 |

I chose the 0.7 weight on CV during the first, mis-specified ablation, where 0.6 and 0.7 were
tied.

**Gain over round 4:** CV of the shipped model minus the base, per head: +0.016 accuracy,
+0.018 fluency, +0.027 prosodic, +0.018 total.

**Pitch was a negative result.** Adding the pitch group alone lowered fluency and prosodic.

## speechocean762 TEST: before / after (Pearson, speaker-cluster bootstrap 95 % CI; Δ = ours − Azure, paired bootstrap, two-sided p)

| level | round 4 | round 5 | Azure | round 5 − Azure |
|---|---|---|---|---|
| phone | 0.660 [0.581, 0.712] | 0.660 [0.581, 0.712] | 0.538 [0.458, 0.600] | +0.122 [+0.099, +0.141], p = 0.002 |
| word | 0.585 [0.505, 0.644] | 0.585 [0.505, 0.644] | 0.589 [0.515, 0.647] | −0.004 [−0.043, +0.033], p = 0.82 |
| sentence accuracy | 0.747 [0.684, 0.793] | **0.761** [0.703, 0.804] | 0.673 [0.613, 0.720] | +0.088 [+0.052, +0.123], p = 0.002 |
| sentence fluency | 0.783 [0.703, 0.838] | **0.792** [0.717, 0.846] | 0.708 [0.628, 0.760] | +0.084 [0.000, +0.175], p = 0.05 |
| sentence prosodic | 0.772 [0.697, 0.829] | **0.782** [0.709, 0.836] | 0.827 [0.788, 0.854] | −0.045 [−0.115, +0.007], p = 0.16 |
| sentence total | 0.761 [0.697, 0.811] | **0.773** [0.717, 0.820] | 0.796 [0.743, 0.834] | −0.023 [−0.074, +0.021], p = 0.41 |

**Round 4 − Azure, for comparison:** prosodic −0.055 (p = 0.02), total −0.035 (p = 0.17).

All four sentence scores improved by +0.009 to +0.014, and phone and word are unchanged.

**The goal was not reached.** Prosodic and total are still below Azure. Prosodic is no longer
significantly behind (p = 0.16, was 0.02). On Spearman, prosodic is level with Azure: 0.780 vs
0.782.

**Most of the CV gain did not carry over.** The CV gain on prosodic was +0.027, but the test gain
was only +0.010, so the train-CV → test drop is still about 0.07.

**int8 sentence Pearson** (speechocean762 test): accuracy 0.760, fluency 0.792, prosodic 0.782,
total 0.773. In round 4 these were 0.742 / 0.780 / 0.771 / 0.760.

## Other languages: native FLEURS validation, 100 clips per language

Each cell is the median on the 0–10 scale, round-4 heads → round-5 heads, on the same calls.
**Rule:** if any of fluency, prosodic or total drops by more than 0.3, the language keeps the
round-4 heads.

| lang | fluency | prosodic | total | ships |
|---|---|---|---|---|
| en | 8.69 → 8.48 | 8.30 → 8.06 | 8.07 → 7.98 | round 5 |
| zh | 8.45 → 7.79 | 8.11 → 7.37 | 8.01 → 7.36 | round 4 |
| es | 8.71 → 8.10 | 8.47 → 7.80 | 8.18 → 7.85 | round 4 |
| it | 7.93 → 7.58 | 7.76 → 7.27 | 7.87 → 7.48 | round 4 |
| ja | 8.43 → 7.72 | 8.24 → 7.29 | 8.02 → 7.26 | round 4 |
| ko | 8.53 → 7.74 | 8.33 → 7.33 | 7.94 → 7.35 | round 4 |
| fr | 8.62 → 8.40 | 8.32 → 8.00 | 8.08 → 7.96 | round 4 |
| de | 8.76 → 8.24 | 8.33 → 7.77 | 8.08 → 7.84 | round 4 |
| pt | 7.99 → 7.91 | 7.79 → 7.54 | 7.91 → 7.62 | round 5 (largest drop 0.29) |
| ru | 8.64 → 8.24 | 8.35 → 7.78 | 8.12 → 7.87 | round 4 |

**The new heads only travel to English.** The English-trained projection of the encoder
activations reads non-English phonetics as accent, and native medians drop by 0.4–1.0. As a
result, eight languages keep round 4. English also drops a little on native FLEURS (0.1–0.24);
I report this but it is within the rule. A first fit without input clipping on the ridge part
dropped every language, English included, by about 0.5. That fit was also the mis-specified one
above, and I discarded it before evaluating test.

The ten-language benchmarks were not re-run, for two reasons. Sound and word outputs and the
sentence probability are unchanged, and the sentence regression outputs only changed for en
and pt.

## Parity, speed, tests

- **Training/runtime parity** (`bench/calib/parity_delivery.py`, `results/tables/delivery_parity.json`):
  200 train clips run through the default public `Scorer`. Every adopted delivery input,
  including the projection outputs, has a maximum absolute difference of 0.0 from the training
  cache, with no NaN mismatches. The tree walker matches sklearn to 3e-8.
- **Latency** for a 3.5 s clip (speechocean762 test, median of 100):

  | setting | round 4 | round 5 |
  |---|---|---|
  | fp32, 4 threads | 48.8 ms | 53.0 ms |
  | fp32, 1 thread | 98.7 ms | 101.1 ms |
  | int8, 4 threads | 37.7 ms | 39.0 ms |

  The new "delivery" stage takes 5.0 ms (pitch track for the stress channel included; it was
  inside "prosody" before), and 53.0 ms is within the 60 ms limit.
- **Tests:** `pytest -q` passes 407 tests (399 before, plus 8 in `tests/test_delivery.py`).

## Files

- **Code:**
  - `pronunciada/delivery.py`
  - `pronunciada/acoustic.py`: `EMBED_OUTPUTS`, `add_graph_outputs`, `embeddings=`
  - `pronunciada/scorer.py`: `delivery=`, config `sentence_model`
  - `pronunciada/calibrate.py`: `load_projection`, `load_linear`, `apply_linear_blend`
  - `pronunciada/prosody/__init__.py`: optional `track`
- **Data:**
  - `pronunciada/data/calib/sentence.npz` (round 5)
  - `pronunciada/data/calib/sentence_r4.npz` (unchanged)
  - `config.json`: `sentence_model`
- **Bench scripts:** `bench/calib/delivery_extract.py`, `train_delivery.py`,
  `native_sentence_check.py`, `parity_delivery.py`
- **Tables:** `results/tables/round5_ablation.json`, `round5_fit.json`, `round5_native.json`,
  `delivery_parity.json`, `english_vs_azure.json`
- **Caches** (`~/.cache/pronunciada/calib/`):
  - `delivery_speechocean762_train.npz`
  - `round2_sentence_inputs.npz`
  - round-4 test API outputs moved to `api_round4_backup/`
