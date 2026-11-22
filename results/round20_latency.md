# Round 20: Korean and Japanese latency (same outputs)

Round 19 made Korean and Japanese slower than the other languages: +10.5 / +14.4 ms (fp32) on 12 s
FLEURS clips, against +0.6 to +2.4 ms elsewhere (results/round19_integration.md, F2). This round
changes speed only. Every output number is bitwise identical to main 8c74789 in all ten languages,
fp32 and int8. No scoring logic, threshold, model file, switch or dependency changed.

**Result:** paired latency added by round 19 (against c9773cc, 4 threads):

| | 12 s clips, before | 12 s clips, after | short clips, before | short clips, after |
|---|---|---|---|---|
| Korean | +11.0 to +13.4 ms | **+1.4 to +1.9 ms** | +1.8 to +2.2 ms | **+0.7 to +0.8 ms** |
| Japanese | +13.9 to +16.1 ms | **+4.5 ms** | +3.3 to +3.4 ms | **+1.8 to +1.9 ms** |

Korean meets the ~2 ms target. Japanese meets it on short clips but not on 12 s clips. The rest
is work the round-19 Japanese model really does (see "What is left").

## 1. Profile before

Method:
* `Scorer.score` with cProfile on 50 FLEURS validation clips (own text), plus the scorer's own stage
  timings.
* The posterior cache was filled first, so the filterbank and the model are not included.
* 2 threads, on a shared machine.

Tables: `tables/round20_latency_profile.json`.

| | pre-19 c9773cc | main 8c74789 | where the difference went |
|---|---|---|---|
| ko, post-model median | 44.9 ms | 60.6 ms | **extra-sound evidence `llr_ins`: +9.9 ms** (aligner with insertion on vs off, paired); `ko_identity.evidence` 1.8 + `word_deltas` 1.1 ms; assemble ~0.5 |
| ja, post-model median | 33.1 ms | 48.8 ms | **`llr_ins`: +7.6 ms**; `length_r19.extra` 5.6 ms (of which `_best_path` 3.3 ms in ~120 calls per clip); `word_match` 1.6; `ja_identity` 1.1; assemble ~0.5 |

The suspicion in the integration write-up was right. `llr_ins` was a second per-frame Python loop:
* 2(n − 1) rows, each going through 12 `logaddexp` passes per frame;
* the two rows of each gap (1-sound and 2-sound chains) had identical first states.

Measured on its own:

| aligner (`score_reference`), paired CPU time | ja | ko |
|---|---|---|
| main, insertion on, minus the pre-19 aligner | +10.5 ms | +7.4 ms |

## 2. What changed

* **`align._fast_scores`**:
  * **Reused chain states.** An extra-sound chain between slot i and slot i + 1 is fed exactly like
    slot i + 1's wildcard chain. Chain states 0 and 1 do not depend on the chain's length. So any
    wildcard row of slot i + 1 with chain weight 0 already holds them, and no second recursion runs.
  * **Extra rows only where needed.** A gap with no such row gets an extra row in the same recursion.
    This happened in none of the 41,600 gaps of 200 Korean and Japanese clips; only pre-wildcarded slots need it.
  * **Kept occupancies.** The recursion forms each chain occupancy log(W_k + b_k) one frame later
    anyway (it feeds state k + 1). It is now kept (`Oh`) instead of computed again with `logaddexp`
    for the chain ends. This also makes the aligner without insertion about 0.4 ms faster (all
    languages).
  * **Only the frames that count.** The occupancies still needed are computed only where the drain is
    above −inf. Elsewhere the frame's term is −inf whatever the occupancy.
  * **Same summation order.** Array layouts are kept, so the frame sums run in the same order as
    before. A row-major vs column-major change alone moved `llr_ins` by about 1e-14; this was found
    and removed.
* **`length_r19.extra`:** the long / short counterfactual best paths (~120 per clip) run as one
  padded batch. It uses the same operations element by element, and each job is read at its own
  last frame.
* **`ja_identity` / `ko_identity.evidence`:** the best rival class is found with a masked row max
  instead of `np.delete` per sound. Korean offsets are applied in float32, exactly as the Python-float
  scalar arithmetic did.
* **`word_match`:**
  * each word's free-decode tokens are found by bisection (the tokens come in time order; a plain
    scan is the fallback);
  * the word alignment DP is memoised.
* **`scorer` (ko):** the phone vectors already computed for the contrast heads are reused, and the
  Korean sound classes are computed once.

CPU-time cost of the pieces after the change (30 clips of ~12 s, 2 threads):

| piece | ja | ko |
|---|---|---|
| aligner insertion on, minus the pre-19 aligner | +0.97 ms | +0.44 ms |
| `length.score` (Japanese round-19 model) | 5.99 → 3.50 ms | n/a |

Not changed, and why:
* **The identity-head MLP** (0.8 ms ko, 0.4 ms ja; `A ** 3` in the GELU is 0.3 ms of it). Rewriting
  `A ** 3` as `A * A * A`, or using float64, is not bitwise equal to `powf`.
* **The vowel-stability scan in `length_r19`** (~0.6 ms). Turning its per-frame `sdot` calls into one
  `sgemv` can change the last bit and flip a threshold test.

## 3. Equivalence

Method:
* Tool: `bench/round20_latency/run_equiv.sh`, then `compare.py` with tolerance 0. It uses the dump of
  `bench/round19_merge/equiv.py`.
* Outputs compared: `to_dict` (without timing), rescore at strictness 0 and 1, `details`, coaching
  items and `to_azure`.
* Clips: FLEURS validation. Each clip is scored with its own text and with one word swapped.
* Comparison: main 8c74789 vs this branch.

Table: `tables/round20_latency_equivalence.json`.

| languages | clips (scores) | values compared per precision | result, fp32 and int8 |
|---|---|---|---|
| ko | 200 (400) | 1.47–1.48 M | max \|diff\| = 0.0, no differing value |
| ja | 200 (400) | 1.61 M | max \|diff\| = 0.0, no differing value |
| en, de, zh, es, fr, it, pt, ru | 100 (200) each | 0.60–0.86 M each | max \|diff\| = 0.0, no differing value |

Also:
* `tests/test_align_insertion.py` checks `llr_ins` against explicit lattices: a nothing / 1-sound /
  2-sound "any speech" slot inserted between slots i and i + 1. There are 48 cases, including
  single-sound slots (one chain state), slots with no weight-0 chain, and pre-wildcarded slots.
  The test passes on main and on this branch.
* `python -m pytest -q tests`: 657 passed.

## 4. Latency

Method:
* Tool: `bench/round19_merge/latency.py`. It runs two warm workers, with one scorer running at a
  time and the clip order alternating. The posterior cache is cleared before each call. 4 threads,
  100 clips.
* Each row reports the paired median difference with a 95 % clip-bootstrap interval.
* Machine: shared with other jobs. The 1-minute load average was 3–10 (it is recorded per run in the
  JSON).
* Short clips:
  * **ko:** FLEURS validation clips cut in the pause after the last word ending by ~3.8 s, with the
    text cut to the same words (`bench/round20_latency/make_cut.py`; 176 clips, 100 used, median
    3.66 s). Also all 34 cv-real dev-half clips of 2.5–5 s (median 3.98 s).
  * **ja:** cv-real dev-half clips of 2.5–5 s (median 3.46 s). These are the `fleurs-val-cut` and
    `cv-dev-short` sets of `equiv.py`.
* The three trees were compared pairwise. A tree's median shifts between runs, so the paired column
  is the one to read.

Table: `tables/round20_latency_paired.json`.

| lang | clips | prec | pre-19 → main (before) | main → branch | pre-19 → branch (after) | branch median |
|---|---|---|---|---|---|---|
| ko | FLEURS 12.3 s | fp32 | +13.38 [+11.87, +15.26] | −10.02 [−12.72, −8.15] | **+1.36** [+0.46, +1.90] | 168.7 ms |
| ko | FLEURS 12.3 s | int8 | +11.00 [+9.56, +12.11] | −8.40 [−9.88, −7.32] | **+1.87** [+1.71, +2.07] | 118.5 ms |
| ko | FLEURS cut 3.66 s | fp32 | +1.83 [+1.77, +1.94] | −1.24 [−1.33, −1.16] | **+0.73** [+0.69, +0.81] | 46.6 ms |
| ko | FLEURS cut 3.66 s | int8 | +1.80 [+1.73, +1.92] | −1.13 [−1.20, −1.09] | **+0.66** [+0.59, +0.73] | 36.3 ms |
| ko | cv dev 3.98 s (n = 34) | fp32 | +2.23 [+1.70, +2.57] | −1.33 [−1.89, −1.00] | **+0.76** [+0.64, +0.84] | 52.1 ms |
| ko | cv dev 3.98 s (n = 34) | int8 | +2.15 [+1.72, +2.60] | −1.70 [−2.13, −1.24] | **+0.69** [+0.53, +1.07] | 42.9 ms |
| ja | FLEURS 12.4 s | fp32 | +16.08 [+14.34, +18.34] | −12.11 [−12.59, −10.80] | **+4.51** [+3.83, +4.98] | 151.5 ms |
| ja | FLEURS 12.4 s | int8 | +13.85 [+12.45, +15.98] | −10.03 [−10.89, −8.82] | **+4.47** [+3.81, +5.09] | 121.3 ms |
| ja | cv dev 3.46 s | fp32 | +3.39 [+3.15, +3.63] | −1.85 [−1.96, −1.66] | **+1.75** [+1.67, +1.90] | 48.3 ms |
| ja | cv dev 3.46 s | int8 | +3.32 [+3.15, +3.67] | −1.68 [−1.95, −1.47] | **+1.91** [+1.71, +2.12] | 40.3 ms |

The "before" column repeats the round-19 F2 measurement on a second day. It came out 1–3 ms higher
than F2's +10.5 / +10.9 (ko) and +14.4 / +14.8 (ja).

## What is left

* **Japanese 12 s: +4.5 ms.** Measured CPU time per 12 s clip:
  * the round-19 length model's inputs and trees: about 3.5 ms in `length.score`, about 2 ms of which
    is new work (`length_r19.extra`: filterbank stability scan, mora rates, the batched
    counterfactual paths);
  * the identity head: 0.55 ms;
  * the extra-sound evidence: about 1 ms;
  * the word-match DP;
  * the word step in assemble.

  All of these scale with the number of sounds. Getting below 2 ms would need changing what the
  length model computes, which this job does not do.
* **Korean 12 s int8: +1.9 ms** (CI to +2.1). This is the identity-head MLP plus about 0.4 ms of
  extra-sound evidence. The other languages show the same order (+1.5 to +2.4 ms).
