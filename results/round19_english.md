# Round 19: English sentence scores (layer fusion, delivery inputs, a sequence head)

**Second pass (later the same night): where things now stand.**

The third held-out look was the full benchmark run on the frozen second candidate.

| measure | shipped baseline | first pass | second pass |
|---|---|---|---|
| sentence total, Everyone | 0.762 | 0.791 | **0.792** [0.743, 0.832] |
| sentence total, Adults 13+ | 0.839 | 0.858 | 0.857 |
| sentence total, Children ≤ 12 | 0.558 | 0.608 | **0.614** [0.523, 0.684] |
| sound | 0.685 | 0.685 | **0.700** |
| word | 0.609 | 0.609 | **0.640** |

- The total-score target of 0.80 is **not reached**.
- Second pass against first pass: total +0.001 (p = 0.41), children +0.005 (p = 0.06).
- Against the shipped baseline:
  - total +0.030 [+0.015, +0.051], p = 0.001;
  - children +0.056 [+0.001, +0.121], p = 0.048.
- The gains of this pass are at the sound and word levels.
- Word is now significantly ahead of Azure: +0.050, p = 0.022.

Details are in the section "Second pass" at the end. The first-pass write-up follows unchanged.

**Outcome.** On speechocean762 TEST the sentence total score's Pearson rises from **0.762 to 0.791**
(paired Δ +0.029 [+0.014, +0.050], p = 0.001). That is short of the 0.80 target. Adults rise
from 0.839 to 0.858, and children of 12 and under from 0.558 to 0.608 (Δ +0.051, p = 0.11).
Sound and word scores are unchanged, because only the sentence regression heads changed. The change
is on by default for English only. `PRONUNCIADA_SENTENCE_HEAD=0` restores the previous heads.

Design, fitting and selection used speechocean762 TRAIN (speaker-grouped CV) and FLEURS
VALIDATION (native guard) only. TEST was run twice: once for the shipped baseline, once for the
frozen candidate. Nothing was changed after the second run. The full English benchmark was run
before and after the change; it repeats these two systems on TEST and adds int8, L2-ARCTIC / EpaDB
and speed.

## Held-out: speechocean762 TEST, sentence total (Pearson, speaker-cluster bootstrap 95 % CI)

| group | utts / speakers | before (shipped) | after (round 19) | paired Δ [95 % CI], p | OpenPronounce 1.1.0 (reported) |
|---|---|---|---|---|---|
| Everyone | 2500 / 125 | 0.762 [0.699, 0.812] | **0.791** [0.741, 0.831] | +0.029 [+0.014, +0.050], p = 0.001 | 0.79 |
| Adults (13+) | 1460 / 73 | 0.839 [0.787, 0.876] | **0.858** [0.812, 0.889] | +0.018 [+0.011, +0.028], p = 0.001 | 0.85 |
| Adults (18+) | 1220 / 61 | 0.844 [0.794, 0.881] | 0.864 [0.819, 0.896] | +0.020 [+0.012, +0.029], p = 0.001 | |
| Children ≤ 12 | 1040 / 52 | 0.558 [0.466, 0.654] | **0.608** [0.517, 0.677] | +0.051 [−0.004, +0.115], p = 0.11 | 0.67 |

The rival's numbers are its own report and were not reproduced here. Its "adults" group may be
defined differently from either of the two adult rows above.

**All four sentence scores, Everyone** (Azure on the same clips, cached responses):

| score | before | after | paired Δ, p | Azure | after − Azure, p |
|---|---|---|---|---|---|
| accuracy | 0.748 | **0.778** [0.724, 0.819] | +0.030, p = 0.001 | 0.673 | +0.104, p = 0.002 |
| fluency | 0.779 | **0.809** [0.736, 0.859] | +0.030, p = 0.001 | 0.708 | +0.101, p = 0.016 |
| prosodic | 0.772 | **0.800** [0.733, 0.850] | +0.028, p = 0.002 | 0.827 | −0.027, p = 0.42 (was −0.055, p = 0.02) |
| total | 0.762 | **0.791** [0.741, 0.831] | +0.029, p = 0.001 | 0.796 | −0.006, p = 0.85 |

Children ≤ 12: accuracy 0.540 → 0.597 (p = 0.017), fluency 0.553 → 0.610, prosodic 0.573 → 0.618.

**Unchanged:** sound 0.685, word 0.609, and L2-ARCTIC / EpaDB error-detection AUC 0.712 / 0.721.

**int8 backbone** (full benchmark, same frozen system):

| | accuracy | fluency | prosodic | total |
|---|---|---|---|---|
| before | 0.745 | 0.777 | 0.771 | 0.761 |
| after | 0.774 | 0.809 | 0.800 | 0.789 |

With the int8 model the new head loses 0.002 on total compared with fp32 (0.791 → 0.789). The old
heads lost 0.001. No tapped layer was found to be sensitive, so none was investigated further.

## What ships

English sentence scores = 0.65 × `sentence_r19` + 0.35 × `seqhead_en`. The config key is
`sentence_head`, set for English only; every other language is unchanged.

- **`sentence_r19.npz`** (193 kB): the shipped s2 boosted trees per score. Inputs:
  - the round-2/4 sentence inputs (`SENT_NUM`);
  - all 40 delivery features of `pronunciada/delivery.py`;
  - an F0 summary (median, 10th and 90th percentile of voiced F0, Hz);
  - a 4-output ridge projection of the **std-pooled** activations of **every encoder stack**
    (enc0 … enc5 + final, 2,496 dims; fitted in-fold as in round 5).
  - The acceptance heads (`len`, `str`) are byte-identical to `sentence_r4` and read the same 24
    columns, so `SentenceResult.prob` is unchanged.
- **`seqhead_en.onnx`** (815 kB) + **`seqhead_en.npz`** (25 kB): 205 k parameters, fp32,
  `pronunciada/sentence_head.py`, onnxruntime.
  - Sound tokens: a softmax-weighted sum of per-stack linear maps of each sound's mean activation
    (all seven stacks), plus a linear map of the 48 aligner-evidence inputs of the sound models,
    plus embeddings of the expected sound and its position in the word.
  - One utterance token: `SENT_NUM`, delivery and F0.
  - Four score queries, then 1 pre-LN transformer layer (d 64, 4 heads).
  - Trained multi-task: sentence MSE, plus 1.0 × (1 − batch Pearson), plus 0.5 × sound MSE, plus
    0.5 × word MSE.
  - The ONNX output matches torch to 3e-7.
- **Backbone:** the ZIPA files are unchanged. `acoustic.ALL_LAYER_OUTPUTS` adds the stack-0/2/4
  outputs to the graph by the same protobuf append as round 5. The variant is written next to the
  model as `model[.int8].emb-enc0-…-final.onnx`; its log-probs are bit-identical (unit test).
- **Size:** +1.03 MB for English.
- **Latency:** fp32, 4 threads, 3.35 s clips on a busy shared machine (`bench/round19/runtime_check.py`):
  - fp32: 60.7 → 64.9 ms (+4.2 ms), within the ~5 ms allowance;
  - int8: 53.2 → 53.2 ms;
  - the `sentence_head` stage alone has a median of 0.87 ms;
  - most of the rest comes from the extra graph outputs and the voice / F0 summaries.
  - The benchmark's speed step gave 74 → 62 ms, but load from other jobs dominates that comparison.
- **Parity:** runtime tree inputs match the training cache to 1.7e-3. That is the float16 storage
  of the cached pooled activations; the projection inputs are computed in float32 at runtime.
  Tree walker vs sklearn: 2.5e-8.

## Native guard (FLEURS VALIDATION, the 100 seeded English clips of round 5)

Median 0–10 score, new − old, through the public `Scorer`:

| accuracy | fluency | prosodic | total |
|---|---|---|---|
| +0.15 | +0.05 | +0.17 | +0.09 |

The rule allows a drop of up to 0.3, so the guard passes. Natives score slightly higher.

## Experiment ledger (speechocean762 TRAIN CV)

The CV design is the same for every experiment:

- 5 folds, grouped by speaker; children (≤ 12) and the others are dealt separately, so each fold
  has the same share of children;
- 3 seeds;
- Δ is against the shipped recipe (s2 trees on `SENT_NUM`) refitted in the same folds.

Tables: `results/tables/round19_english_cv_utt.json`, `_cv_seq.json`, `_cv_blend.json`,
`_guard.json`.

| # | hypothesis / config | total | Δ total | prosodic Δ | child ≤ 12 | adult | native guard | kept |
|---|---|---|---|---|---|---|---|---|
| 0 | shipped recipe (sentence_r4 inputs, s2 trees) | 0.824 | – | – | 0.493 | 0.873 | – | baseline |
| 1 | + all delivery features | 0.836 | +0.011 | +0.014 | 0.530 | 0.879 | pass (−0.12) | yes |
| 2 | + F0 summary alone | 0.825 | +0.001 | +0.002 | 0.514 | 0.869 | pass | with 1 |
| 3 | round-5 set (disfluency + enc1 projection) | 0.840 | +0.015 | +0.024 | 0.523 | 0.882 | pass (−0.27) | no |
| 4 | single-stack mean ++ std projections (+ disfluency): enc0 / 2 / 3 / 4 / final | 0.840 / 0.832 / 0.832 / 0.833 / 0.834 | +0.008 … +0.016 | | | | enc3, enc4, enc5, final, enc0 fail (−0.33 … −0.75) | no |
| 5 | delivery + F0 + all seven stacks, mean ++ std (layer fusion) | 0.844 | +0.020 | +0.029 | 0.543 | 0.886 | **fail** (fluency −0.65) | no |
| 6 | same, **std-pooled** stacks only | 0.841 | +0.017 | +0.024 | 0.537 | 0.883 | pass (−0.18) | **yes** |
| 7 | 6 + mean-pooled enc2 | 0.840 | +0.016 | | | | | no |
| 8 | voice cues (spectral centroid, roll-off) on base / base + F0 | 0.824 / 0.825 | −0.000 / +0.000 | | 0.502 / 0.515 | 0.870 / 0.868 | | no |
| 9 | worst-k evidence (5th / 10th pct, worst 3 of llr_sub, z_llr_sub, llr_del, peak_post) | 0.824 | +0.000 | | 0.501 | 0.872 | | no |
| 10 | rater-agreement item weights (5 raters, `scores-detail.json`), base / on 6 | 0.826 / 0.839 | +0.002 / −0.002 vs 6 | | | | | no |
| 11 | F0 / voice-gated child model (p(child) from voice, AUC 0.76), on base + F0 + voice / on #5 + voice | 0.810 / 0.831 | −0.014 vs base / −0.012 vs ungated | | 0.517 / 0.545 | 0.847 / 0.868 | | no |
| 12 | sequence head, 2 layers d 96, all stacks (1 seed) | 0.817 | −0.007 | | 0.512 | 0.861 | | – |
| 13 | … stronger regularisation, corr. loss 1.0 (1 seed) | 0.825 | | | 0.501 | 0.869 | | – |
| 14 | … 1 layer d 64 (1 seed) | 0.827 | | | 0.520 | 0.868 | | – |
| 15 | … no encoder stacks (evidence + utterance only, 1 seed) | 0.796 | | | 0.464 | 0.846 | | layer fusion worth +0.03 here |
| 16 | … stacks enc0–2 only (1 seed) | 0.820 | | | | | | – |
| 17 | **seq head 1 layer d 64, regularised (3 seeds)** | 0.833 | +0.008 | +0.023 | 0.502 | 0.878 | natives +0.46 (higher) | as blend |
| 18 | **blend 0.65 × #6 + 0.35 × #17 (frozen)** | **0.853** | **+0.029** | **+0.038** | **0.550** | **0.893** | pass (+0.05 … +0.17) | **shipped** |

Other blend weights for #18:

| seq weight | 0.15 | 0.25 | 0.35 | 0.5 |
|---|---|---|---|---|
| total | 0.849 | 0.852 | 0.853 | 0.852 |

The sequence head's out-of-fold sound and word scores are 0.67–0.71 and 0.66–0.69 Pearson. That
is close to `sound_E` and above `word_reg_r6` in their own CV rounds (0.715, 0.674). They were not
pursued, because this round was about the sentence score and the shipped sound and word levels
were to stay as they are.

### Why the round-5 inputs were lost, and the guard-safe version

The licence refit failed the guard only narrowly in English: prosodic −0.32 with the 0.7 / 0.3
ridge blend. Refitted here as trees only, the round-5 set passes (−0.27), but it is the weakest
of the projection options.

The cause is the **mean**-pooled activations of the upper stacks:

- enc3, enc4, enc5 and final each lower native medians by 0.45–0.75 on their own;
- enc0 and enc1 lower them by 0.2–0.33;
- delivery and F0 groups by at most 0.18.

The likely reason is that a mean over a whole clip carries recording-channel and speaker
information that speechocean762 does not vary, so the projection reads read-aloud FLEURS speech
as "worse". The **std** half (how much the representation moves within the clip) keeps most of
the CV gain (+0.017 vs +0.020) and passes the guard.

### Children

Children are where the gap is. The children's labels themselves are much noisier:

- one rater against the mean of the other four: r = 0.53 for children, 0.87 for adults (train);
- children's total scores have half the spread of adults' (sd 0.97 vs 1.94).

So their ceiling is lower; a Spearman-Brown estimate puts it near 0.86. The test offsets between
groups are small (mean label vs prediction within 0.05 in both groups, aggregate diagnosis only),
so the deficit is ranking within children, not calibration across groups.

Speaker cues did not help:

- F0, spectral centroid / roll-off and the F0-gated child model are all neutral or harmful in CV.

What did help children:

- delivery features;
- std-pooled stack activations;
- the sequence-head blend.

CV for children went 0.493 → 0.550, and test 0.558 → 0.608.

## Not done

- **Mel-warp (vocal-tract-length) ablation** for high-F0 voices: not run. It needs a full
  re-extraction with a modified front-end, and the speaker-cue results above gave no sign that
  voice normalisation is the missing piece.
- **Pairwise ranking loss and a CCC loss:** only a batch-Pearson term was used, at weights 0.5
  and 1.0.
- **Joint bottleneck** (predict accuracy / fluency / prosodic, derive total): not run as a
  separate model. The sequence head shares one representation across the four scores.
- **Direction 5** (per-language low-rank adaptation of the top encoder blocks): not started.
  The torch checkpoint was not downloaded.
- **Ensembling** several seeds of the sequence head: not tried. Only one seed (0) ships.

## Files

- **Runtime:**
  - `pronunciada/sentence_head.py` (new);
  - `pronunciada/acoustic.py`: `ALL_LAYER_OUTPUTS`, `embed_outputs_for`, layer-set-specific
    embedding model files;
  - `pronunciada/delivery.py`: `pooled_part`, `.mean` / `.std` projection keys;
  - `pronunciada/scorer.py`: config `sentence_head`, `PRONUNCIADA_SENTENCE_HEAD`, F0 / voice
    summaries, the blend;
  - `pronunciada/data/calib/sentence_r19.npz`, `seqhead_en.onnx`, `seqhead_en.npz`, `config.json`.
- **Bench** (`bench/round19/`):
  - feature cache: `extract_en.py`, `voice_extract.py`;
  - CV: `data.py`, `exp_utt.py`, `seqhead.py`, `exp_seq.py`, `cv_blend.py`;
  - fitting and export: `fit_final.py`, `train_seq_final.py`;
  - checks: `seq_guard.py`, `runtime_check.py`, `eval_en.py`;
  - `bench/calib/api_run.py` gains `PRONUNCIADA_API_DIR`.
- **Tests:** `tests/test_sentence_head.py`. `tests/test_delivery.py` is adjusted for the new
  config key. The suite passes: 584 tests.
- **Tables:** `results/tables/round19_english_*.json`. `results/english_vs_azure.md` /
  `.json` were regenerated by the after run.
- **Caches** (`~/.cache/pronunciada/round19/english/`):
  - feature caches for train / test / FLEURS validation, about 0.75 GB;
  - API outputs;
  - the five-rater score file of speechocean762 (`download/scores-detail.json`, from the corpus's
    GitHub repository, CC BY 4.0; train ids only used).

## Second pass

This pass followed the coordinator's second brief:

1. seeds, losses and a bottleneck for the sequence head;
2. its sound and word outputs;
3. a VTLN ablation for children.

The CV design is unchanged: the same folds, 3 seeds, train only. The guards use FLEURS
validation. The held-out set was looked at **a third time**, once, after freezing, through the
full `bench.eval_english` run. The first two looks were the baseline and the first frozen candidate.

### 1. Sentence scores: sequence-head variants (CV total, 3 seeds)

The reference is the single first-pass model `s6`, at 0.8326.

| variant | total | child | adult | sound OOF | word OOF |
|---|---|---|---|---|---|
| single model (shipped in pass 1) | 0.8326 | 0.502 | 0.878 | 0.668 | 0.659 |
| CCC loss instead of Pearson | 0.8339 | 0.512 | 0.878 | 0.656 | 0.646 |
| + pairwise logistic ranking loss (0.5) | 0.8266 | 0.500 | 0.876 | 0.652 | 0.648 |
| joint bottleneck (accuracy / fluency / prosodic / completeness → total) | 0.8012 | 0.480 | 0.847 | 0.670 | 0.662 |
| **3 seeds, outputs averaged** | **0.8392** | 0.514 | 0.883 | **0.678** | **0.666** |

The 3-seed average ships. It is one ONNX graph of 3 models, 2.45 MB, 615 k parameters.

Weight averaging was not used, because each seed has its own initialisation and so does not
share a basin with the others. CCC was left out: its +0.001 is within noise, and it weakens the
sound and word outputs.

**Blend with the trees, re-picked by CV** (`round19_english_cv_blend.json`, sequence-head weight):

| weight | 0.15 | 0.25 | 0.35 | 0.5 |
|---|---|---|---|---|
| total | 0.849 | 0.852 | 0.854 | 0.854 |

0.35 is kept. CV total is 0.8536, against 0.8529 with the single model. Inside the blend, the
ensemble adds almost nothing to the sentence scores.

### 2. Sound and word outputs of the sequence head

Both sides are out-of-fold on the same folds:

- **sound:** the round-12b `sound_E` recipe, refitted per fold (in-fold `emb_proj`), shipped
  hyper-parameters;
- **word:** the round-6 `word_reg_r6` recipe, refitted per fold.

Table: `round19_english_cv_level.json`.

| level | shipped recipe | blend (seq weight) | blended | detection on train folds |
|---|---|---|---|---|
| sound (0-2, Pearson) | 0.720 | 0.3 | **0.730** | AUC score < 2: 0.847 → 0.849; < 1: 0.956 → 0.962 |
| word (0-10, Pearson) | 0.675 | 0.4 | **0.695** | AUC acc < 8: 0.928 → 0.935; < 6: 0.934 → 0.941 |

**What is wired in.** For English only:

- the reported 0-2 sound score (`details()["phone_score"]`) is 0.7 × `sound_E` + 0.3 × the head;
- the 0-10 word score (`word_accuracy`) is 0.6 × `word_reg_r6` + 0.4 × the head.

**What is not touched:**

- the sound, word and sentence probabilities;
- the coaching inputs (`coach_zl` / `coach_zs`);
- the sentence calibrator's inputs.

On the 100 FLEURS-validation natives, every probability is bit-identical with the switch on and
off. Native medians went up: word 9.90 → 9.94, sound 1.965 → 1.974. Sentence medians changed
by +0.06 to +0.17.

### 3. Children: VTLN ablation (dropped)

**Method** (`round19_english_vtln.json`):

- a log-mel frequency warp (`audio.vtln_warp`, α = 0.91, formants moved down);
- applied only to clips whose median F0 is above 220 Hz, decided from the audio;
- experiment switch `PRONUNCIADA_VTLN=0.91:220`, off by default.

**Result.** On the 1,155 train clips it touches, agreement between the aligner evidence and the
experts did not improve:

- children: `llr_sub` sound-level Pearson 0.379 → 0.377, `peak_post` 0.333 → 0.338, utterance
  means within ±0.013;
- adults: slightly worse (`llr_sub` 0.584 → 0.575).

The rule also catches 34 % of adult clips (higher-pitched voices). It was dropped without
training on it, as the brief asked.

### Held-out look 3 (frozen second candidate; full `bench.eval_english`, before = `PRONUNCIADA_SENTENCE_HEAD=0`)

Sentence scores by group, with paired differences against the shipped baseline:

| group | score | before | after | paired Δ [95 % CI], p |
|---|---|---|---|---|
| Everyone | total | 0.762 | **0.792** [0.743, 0.832] | +0.030 [+0.015, +0.051], p = 0.001 |
| Everyone | accuracy | 0.748 | 0.778 | +0.030, p = 0.001 |
| Everyone | fluency | 0.779 | 0.814 | +0.036, p = 0.001 |
| Everyone | prosodic | 0.772 | 0.805 | +0.033, p = 0.001 |
| Adults 13+ | total | 0.839 | 0.857 [0.812, 0.890] | +0.018 [+0.011, +0.028], p = 0.001 |
| Children ≤ 12 | total | 0.558 | 0.614 [0.523, 0.684] | +0.056 [+0.001, +0.121], p = 0.048 |

Against Azure (cached responses): prosodic −0.022 (p = 0.45), total −0.004 (p = 0.89),
fluency +0.106 (p = 0.004).

| measure | before | after |
|---|---|---|
| sound, Pearson | 0.685 [0.613, 0.731] | **0.700** [0.628, 0.747] |
| word, Pearson | 0.609 [0.533, 0.663] | **0.640** [0.563, 0.696] |
| word minus Azure | +0.020, p = 0.33 | **+0.050, p = 0.022** |
| int8: total / sound / word | 0.761 / 0.684 / 0.611 | 0.789 / 0.699 / 0.639 |
| L2-ARCTIC / EpaDB detection AUC | 0.712 / 0.721 | 0.712 / 0.721 (unchanged) |

Against look 2 (first-pass candidate): total +0.001 (p = 0.41). The second pass's held-out gain
is at the sound level (+0.015) and the word level (+0.031).

**Latency** (`round19_english_latency.json`; machine quieter, 1-minute load 3.8; old and new
scorers interleaved call by call; 3.5 s clips; 4 threads):

| backbone | median old → new | paired added time | `sentence_head` stage |
|---|---|---|---|
| fp32 | 53.2 → 54.4 ms | **+1.2 ms** | 1.1 ms |
| int8 | 42.6 → 44.1 ms | **+1.4 ms** | 1.1 ms |

The benchmark's own speed step ran on a loaded machine: fp32 74.3 → 75.2 ms, int8 58.7 → 58.1 ms.

**Size** added for English: 2.67 MB in total (`sentence_r19.npz` 193 kB, `seqhead_en.onnx`
2.45 MB, `seqhead_en.npz` 25 kB).

**The parallel `encoder` engineer** had no frozen winner when this candidate froze. Per their
STATUS, LoRA and full fine-tuning of stack 5 under the sequence head gave CV 0.824, below their
frozen control at 0.830. If a variant wins later, its head would replace or join the 3-model
ensemble inside the 0.35 blend weight, and the blend weight would need re-picking on the same folds.

### Still not done

- Direction 5 in the form briefed here: no backbone fine-tuning on my side; the `encoder`
  engineer covers it.
- Children remain the gap: 0.614, against the rival's reported 0.67. Every speaker-normalisation
  cue tried was neutral. The label noise for children (single rater against the other four:
  r = 0.53) caps what any model can reach, but not at 0.61.
