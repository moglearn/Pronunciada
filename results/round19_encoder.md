# Round 19: adapting the top ZIPA-small encoder stack to English scoring

Question: does training the top of the encoder for pronunciation scoring (instead of reading frozen
activations) lift agreement with the speechocean762 experts beyond a small head on frozen features,
inside the on-device limits?

**Answer: no.** On the shared 5-fold, speaker-grouped, child-balanced folds (3 seeds), every way of
adapting the last encoder stack (low-rank adapters, full fine-tuning, a 3-run soup, joint training
from scratch) scores the same as, or slightly below, the identical training run with the stack frozen.
Paired differences on the sentence total are -0.001 to -0.003 (speaker-bootstrap intervals touch or
include zero). Nothing was close to the bar for a held-out run (the English engineer's guard-safe CV
total 0.853 + 0.01), so **the speechocean762 test split was not used**. Encoder adaptation is not
worth shipping at this data size.

## The cut

The Zipformer's six stacks run one after the other at 50 Hz; the encoder output is
concat(stack 5 [256], stack 4 channels 256:384, stack 3 channels 384:512), then the CTC layer. The
smallest cut above which only the last stack and the output layers remain is four tensors: stack 4
channels 0:256 (the input of stack 5), stack 4 channels 256:384, stack 3 channels 384:512, and the
stride-2 key-padding mask (shared with stack 1). Graph names are in `bench/round19_encoder/cut.py`.

| piece | parameters | file | CPU, 3.5 s clip, 4 threads |
|---|---|---|---|
| full ZIPA-small (shared, unchanged) | 62 M | 248 MB fp32 / 67 MB int8 | 43.6 ms fp32 / 26.6 ms int8 |
| top part (stack 5 + output layers) | 3.89 M | 16.3 MB fp32 / **4.7 MB int8** | 5.4 ms fp32 / **3.4 ms int8** |
| two-stack top (stacks 4 + 5) | 15.5 M | 61 MB fp32, ~15.5 MB int8 | not deployable (> 10 MB), not trained |

(Timings on the shared laptop with other jobs running; medians of 30 runs.) A per-language copy of
stack 5 fits the envelope: 4.7 MB and ~3.4 ms extra, run on the shared model's cut tensors beside the
unchanged top.

Checks: the extracted top graph reproduces the full model exactly (max abs difference 0 on
stack 5, final output and log-posteriors). For training, the top graph runs in torch through a small
ONNX interpreter (`onnx_torch.py`; float initializers become parameters, LoRA hooks on the MatMuls),
which matches onnxruntime to 1e-6, also in padded batches (edge padding keeps the stride-2
downsampler exact). One export quirk had to be handled: softplus-like activations are computed as
log(1 + exp(x)) inside a Where; exp overflows to inf in the unused branch and makes gradients NaN, so
the interpreter uses an exp whose backward is 0 where the incoming gradient is 0 (forward unchanged).
Cut tensors were cached once (float16; train 2,500 clips 497 MB, FLEURS-validation natives 46 MB).

## Design

Head, inputs, folds, seeds and metrics are the English engineer's (copied unchanged into
`bench/round19_encoder/en_*.py`): the GOPT-like sequence head "s6" (d 64, 1 layer) over per-sound means
of all seven encoder outputs + the aligner evidence + utterance cues, 25 epochs, corr weight 1.
Stage 1 is that head on frozen features. Stage 2 continues from it for 6 epochs with the per-sound
stack-5 and final-encoder vectors recomputed from the cut tensors by the (trainable) top graph:

- **frozen_e6**: stage 2 with the stack frozen: the control (same extra epochs, no adaptation)
- **lora8_e6**: rank-8 adapters on all 36 weight matrices of stack 5 (attention, feed-forward, conv
  in/out, non-linear attention); top lr 3e-4, head lr 3e-4
- **lora8_hi**: same, top lr 2e-3, head lr 1e-4 (makes the stack do the work)
- **full_e6**: all 3.89 M stack-5 weights trainable, lr 3e-5, L2 pull to the released weights (0.01)
- **soup3_lora8_e6**: three lora8_e6 runs from the same stage-1 head, head and merged stack weights
  averaged (seed 0 only)
- **joint_lora8 / joint_frozen**: no stage 1; fresh head + LoRA (top lr 1e-3) trained jointly for 25
  epochs, against the same loop with the stack frozen

Stage 2 uses length-bucketed batches (to bound GPU memory); the joint runs show this hurts the head
on its own (joint_frozen 0.801 vs stage 1 0.830 with the same recipe), so every variant is compared with
the control trained in the same loop, not with stage 1. Feature-level augmentation was not added:
the adapted stack barely moved even without it (see below), so there was no overfitting signal to fight.

## CV results (speechocean762 train, 5 folds x 3 seeds, Pearson, mean over seeds)

| variant | accuracy | fluency | prosodic | **total** | total, children <= 12 | total, others | sound | word |
|---|---|---|---|---|---|---|---|---|
| stage 1 (English s6, frozen) | .810 | .859 | .851 | **.832** | .502 | .877 | .668 | .659 |
| frozen_e6 (control) | .806 | .856 | .847 | **.827** | .498 | .873 | .668 | .660 |
| lora8_e6 | .804 | .855 | .847 | **.825** | .497 | .872 | .667 | .659 |
| lora8_hi | .803 | .851 | .842 | **.824** | .494 | .871 | .668 | .659 |
| full_e6 | .804 | .854 | .846 | **.825** | .496 | .872 | .668 | .659 |
| soup3_lora8_e6 (seed 0) | .800 | .856 | .850 | **.825** | .494 | .875 | .669 | .662 |
| joint_frozen (seed 0) | .781 | .826 | .825 | **.801** | .482 | .855 | .647 | .639 |
| joint_lora8 (seed 0) | .777 | .820 | .809 | **.802** | .451 | .865 | .647 | .637 |

(Joint rows: JOINT_NOTE.) Per seed and all metrics: `results/tables/round19_encoder_cv.json`.

Paired speaker bootstrap of the sentence total (2,000 speaker resamples; `round19_encoder_paired.json`):

| comparison | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| lora8_e6 - frozen_e6 | -.0019 [-.0044, .0007] p .14 | -.0012 [-.0028, .0004] p .16 | -.0008 [-.0015, -.0001] p .02 |
| full_e6 - frozen_e6 | -.0020 [-.0046, .0005] p .13 | -.0015 [-.0037, .0006] p .14 | -.0010 [-.0022, .0000] p .04 |
| lora8_hi - frozen_e6 | -.0028 [-.0066, .0006] p .12 | -.0006 [-.0041, .0027] p .66 | -.0034 [-.0068, -.0003] p .04 |
| soup3 - frozen_e6 | -.0013 [-.0034, .0007] p .20 | | |
| joint_lora8 - joint_frozen | +.0011 [-.0131, .0144] p .89 | JOINT_S1 | JOINT_S2 |

Children vs adults: no variant helps children (lora8_e6 - control on children: -.0006, -.0004,
-.0022; joint_lora8 on children -.031 [-.078, .012] on seed 0, adults +.010 [-.001, .023]).

Why nothing moves: with adapters the stack-5 weights change by a median 0.4 % (relative Frobenius
norm; at most 12.6 % for one matrix), with full fine-tuning by 0.07 %; the head reaches its training
loss without needing the stack to change, and the stack's per-sound vectors are one of seven inputs,
next to the frozen lower stacks (which already carry the accent detail the upper stack might discard).
Pushing the stack harder (lora8_hi, or joint training at lr 1e-3) does not help either.

## Native guard (FLEURS validation English, 100 natives; results/licence_refit.md check)

lora8_e6 fitted on all of train (stage 1 + stage 2, seed 0), median predicted sentence score minus the
shipped score: accuracy +0.73, fluency +0.55, prosodic +0.71, total +0.64 (stage-1 head alone: +0.51,
+0.29, +0.39, +0.46). Natives are scored higher, never lower (1 % put more than 0.5 below shipped on
accuracy/total). **Passes** (limit: no drop beyond 0.3). `round19_encoder_guard.json`.

## int8 cost

The deployable form (adapted stack 5 exported to ONNX with the deltas merged, run by onnxruntime on
the cut tensors; per-sound means in numpy; the fold's head) reproduces the torch CV numbers exactly in
fp32 (seed 0 total .8240 = .8240) and in dynamic int8 costs .0004 on the total (.8236; children
.4946 vs .4947). int8 piece: 4.6 MB. `round19_encoder_int8.json`.

## Held-out

None. No candidate beat its own frozen control, let alone the bar (CV total >= .863).

## What a larger run would add

The adapters had little to learn from: 2,500 utterances from 125 speakers, with children's labels so
noisy that one rater agrees with the other four at r 0.53. The frozen lower stacks are already
available to the head, so the top stack's "ignoring accent" is not a bottleneck at this size. A larger
run would only plausibly help with (1) much more scored learner speech (tens of thousands of utterances,
e.g. adding the permissively licensed parts of other scored corpora), and (2) adapting lower stacks,
which do not fit the 10 MB envelope (stack 4 alone is 11.7 M parameters). More GPU time on the same
data would buy more seeds, not a different answer: the paired differences are already within +-0.003.

## Files

- `bench/round19_encoder/cut.py` (cut, top graph, timing), `extract_cut.py` (cache), `onnx_torch.py`
  (torch interpreter, LoRA / full / export), `verify_torch.py`, `train_cv.py` (stage 1 / 2 CV),
  `guard.py`, `int8_cost.py`, `report.py` (paired bootstrap), `en_*.py` (English harness, copied).
- Tables: `results/tables/round19_encoder_{cv,paired,guard,int8}.json`.
- Nothing is wired into the runtime (no candidate). The shared backbone file is unchanged.

Compute: stage 1 and the guard on the laptop (CPU); stage 2 sweeps on one cloud A10G (two jobs side by
side, ~12 s per epoch each; the torch interpreter is bound by per-op dispatch, not by GPU arithmetic).
