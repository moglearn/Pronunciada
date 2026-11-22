# Round 19: Japanese word scores on Common Voice style audio

## Final result (round-19 final candidate, commit dfc8ab3; the fourth and last look at the benchmark)

**What ships.** The final candidate is on by default; `PRONUNCIADA_JA_ROUND19=0` restores round 18.
It is candidate 2 (part 2) plus two new inputs to the word step, both chosen on dev under every dev
guard:

* **extra-sound evidence:** `llr_ins`, ported from the Korean round-19 `align.py`;
* **expected-long vowels:** a separate term on the length logit of expected-long vowels.

**Targets: none met.**

| target | result (v2026.10 rules) | met? |
|---|---|---|
| AUC ≥ 0.96 | 0.923 | no |
| FA@80 ≤ 4 % | 9.7 % | no |
| length items flagged by our own flag ≥ 37 / 40 | 27 | no |

### Side by side

* Lingolix test 1 Japanese runs through the benchmark runner's code path (`bench/round19_ja/lx_run.py`).
* Intervals: bootstrap over clips.
* Guards: held-out FLEURS-swap ja test, cv-real ja Test A test half, and the length table on FLEURS
  test.
* Candidate 1's benchmark numbers are from `bench/round19_ja/unit_eval.py`, which reproduces the
  runner. Its main-rules numbers were not computed.

| | round 18 | candidate 1 | candidate 2 | **final** |
|---|---|---|---|---|
| **AUC, v2026.10 rules** | 0.846 [0.798, 0.891] | 0.884 [0.839, 0.924] | 0.917 [0.885, 0.948] | **0.923 [0.891, 0.951]** |
| **FA@80, v2026.10 rules** | 20.1 % [12.7, 37.4] | 16.6 % | 13.4 % [3.6, 26.4] | **9.7 % [3.8, 20.6]** |
| AUC, main rules | 0.855 | – | 0.925 | **0.930 [0.900, 0.957]** |
| FA@80, main rules | 18.5 % | – | 11.9 % | **8.3 % [3.1, 19.1]** |
| length items flagged (own ErrorType) | 19 / 40 | 21 / 40 | 27 / 40 | 27 / 40 |
| flag recall / precision / false alarms on correct words | 0.56 / 0.41 / 8.4 % | 0.58 / – / 8.7 % | 0.64 / 0.44 / 8.6 % | 0.64 / 0.44 / 8.6 % |
| length bucket AUC | 0.732 | 0.816 | 0.872 | 0.878 |
| subtle / moderate / gross | 0.910 / 0.926 / 0.960 | 0.894 / 0.990 / 0.929 | 0.947 / 0.953 / 0.989 | 0.946 / 0.951 / 0.987 |
| indel | 0.909 | 0.914 | 0.916 | **0.940** |
| guard: FLEURS-swap far AUC | 0.991 | **0.958 ✗** | 0.995 | 0.996 |
| guard: FLEURS-swap near AUC | 0.968 | 0.965 | 0.976 | 0.976 |
| guard: natives below 0.6 (FLEURS-swap; Azure 2.0 %) | 3.27 % | 2.64 % | 2.41 % | 2.69 % |
| guard: cv-real ja word mean AUC | 0.788 | **0.742 ✗** | 0.803 | **0.810** (+0.022 [0.002, 0.045], p = 0.04) |
| guard: cv-real ja word min AUC | 0.775 | **0.682 ✗** | 0.791 | 0.795 (+0.020, p = 0.07) |
| guard: cv-real ja sentence AUC | 0.798 | 0.798 | 0.803 | 0.803 |
| length table: length-swapped words caught at 5 % native FA | 54.0 % | 55.4 % | 66.0 % | 65.3 % |
| length table: native word acceptance | 97.06 % | 97.87 % | 97.62 % | 97.12 % |
| length table sub-row "vowel_length A>B" (words) | 60.1 % | 49.7 % | 45.8 % | **43.8 %** |
| length table sub-row "vowel_length B>A" (words) | 79.5 % | 71.6 % | 84.1 % | 86.5 % |
| length table sub-row "gemination A>B" (words) | 8.1 % | 12.0 % | 37.2 % | 33.9 % |
| length table sub-row "gemination B>A" (words) | 41.9 % | 62.6 % | 78.6 % | 77.4 % |
| length table: sounds caught | 95.5 % | – | 97.3 % | 97.3 % |
| added latency (4 threads, dev clips) | – | ≈ 6 ms | +3.0 ms | +4.2 ms (stage sum) |

**Final against round 18 (paired):**

| rules | AUC | FA@80 |
|---|---|---|
| v2026.10 | [+0.036, +0.120], p = 0.001 | [−24.2, −0.4] points, p = 0.04 |
| main | [+0.035, +0.119], p = 0.001 | [−24.0, −0.6] points, p = 0.04 |

**Final against candidate 2 (paired, v2026.10):**

* AUC +0.006 [+0.001, +0.011], p = 0.02;
* FA@80 [−7.3, +1.2] points, p = 0.50.

**Remaining misses.** At the threshold that catches 80 %, the planted words still above it are:

* 10 length;
* 4 subtle;
* 4 indel;
* 2 moderate.

That threshold now sits at word score 0.960.

Not flagged by our own flag:

* 13 of 40 length items;
* about half of the subtle swaps.

### Two held-out losses to state plainly

* **The "vowel_length A>B" words sub-row of the length table is below round 18:** 60.1 % → 43.8 %.
  It already fell in candidate 2 (45.8 %); the final candidate moves it by −2 points.
  * The other three sub-rows and the total rose (54.0 → 65.3 %).
  * The direction this sub-row measures is the opposite of the one the dev set showed weakest. On
    dev, planted "text long, audio short" items rose 0.894 → 0.912 with the new term, and "text
    short, audio long" stayed at 0.743 → 0.749.
  * So the loss was not visible on dev and is not explained.
* **Natives below 0.6 on FLEURS-swap test** are 2.69 % for the final against 2.41 % for candidate 2.
  Both are better than round 18's 3.27 %.

### Part 3: what was done and what was learnt

**(1) The step fitted properly (log-loss, base log-odds as offset)** (`bench/round19_ja/stepfit.py`).

* **Fit pool.**
  * The Common Voice ja train split has no speakers outside the fit pool under the round's rules:
    all 583 are in the pool that trained the head and the length model.
  * So the fit used the 718 train **clips** the head never saw, from 186 of those speakers. 664
    survived the planting rules: 359 edited, 305 controls.
  * Plus FLEURS train (planted edits and controls; the head did see FLEURS train).
  * Total: 25,262 words, 920 planted.
* **Inputs.**
  * identity margins as native z-scores per class;
  * length logits separately for expected-long vowels, expected-short vowels and consonants;
  * the counterfactual relative to the expected form;
  * the word duration ratio;
  * extra-sound evidence (`llr_ins`, free-decode count);
  * missing-sound evidence (`llr_del`, present).
* **Result.** Signed weights, capped at +3.

  | fit | dev AUC |
  |---|---|
  | CV clips only | 0.812 |
  | with FLEURS | 0.826 |
  | with FLEURS, planted words up-weighted 5× | 0.834 |
  | with FLEURS, planted words up-weighted 20× | 0.837 |
  | candidate 2 | 0.862 |

  The two up-weighted fits failed the native guard.
* **Why the head is under-used.**
  * It is **not** the pool size, and **not** missing per-class calibration. With z-scores per class
    and 25k words, the weight on the identity hinge rose (0.08 → 0.24 → 0.40) but stayed small.
  * The log-loss fit spends its capacity on calibration. The constant pushes every native word up,
    and the many native words with one low-margin sound make a heavy identity term costly in
    log-loss even where it helps ranking.
  * Candidate 2's grid, chosen for dev AUC under the native guard, uses the identity margin much
    more steeply (1.5 per unit below the class's 2 % native quantile + 2). It ranks better.
* **So the final candidate keeps candidate 2's grid form.** A coordinate search added the two new
  terms on dev (`bench/round19_ja/stepgrid2.py`):

  | term | weight chosen | verdict |
  |---|---|---|
  | extra-sound evidence | +0.25·min(llr_ins + 2, 0) | chosen |
  | expected-long vowel length logit | +0.5·min(z_long, 0) | chosen |
  | missing-sound term (lowest llr_del), any weight | – | failed the native guard |
  | word-duration ratio term, any weight | – | failed the native guard |

**(2) Length.** The word duration ratio (word against its mora count at the local mora rate, this
sound counted short) and the 10 ms spectral-stability span were already inputs of the round-2
length model. The vowel sound-level AUC stayed at 0.972 with them.

Neither the cap nor the 0.03 flag threshold explains the length-table loss:

* The table measures word probabilities, not the flag, so the flag threshold does not enter it.
* On dev, the expected-long-vowel direction was not the weak one.
* The new expected-long term raised it on dev but not on the held-out sub-row.

**(3) Extra / missing sounds.**

* `llr_ins` was ported (`pronunciada/align.py` is now the Korean branch's version, with insertion
  stitching; it is computed for Japanese only).
* Dev indel AUC rose 0.707 → 0.732; held-out indel rose 0.916 → 0.940.
* Missing-sound evidence did not pass the native guard.

### Dev numbers of the final candidate

Runtime dumps; `results/tables/round19_ja_devguards.json`.

| dev | round 18 | candidate 2 | final |
|---|---|---|---|
| AUC | 0.811 | 0.862 | 0.867 |
| FA@80 | 29.9 % | 18.4 % | 17.4 % |

Paired, final − candidate 2: AUC +0.006 [+0.003, +0.009], p < 0.001; both speaker folds rose.

Dev guards (all pass against round 18):

| guard | final |
|---|---|
| FLEURS-validation natives < 0.5 / < 0.6 | 2.34 / 2.69 % (round 18: 2.11 / 3.52) |
| CV-dev natives < 0.5 / < 0.6 | 7.87 / 8.46 % (round 18: 8.83 / 10.58) |
| FLEURS-swap dev far / near AUC | 0.9992 / 0.9786 |
| cv-real dev half word mean / min AUC | 0.774 / 0.764 |

### Size and latency (Japanese, round 19 in total)

**Size: about 1.07 MB.**

| file | size |
|---|---|
| identity head | 0.97 MB |
| length model | 98 kB |
| step parameters | 3 kB |

**Latency, round 19 against round 18 (median per stage):**

* 60 dev clips, 4 threads: alignment with insertion +1.7 ms, length +1.7 ms, word step +0.7 ms,
  assembly +0.2 ms. The stage sum is +4.2 ms.
* 80 benchmark clips, 2 threads: the stage sum is +5.1 ms.
* Medians of the total moved by +3.0 to +7.5 ms between runs. The model stage alone moved by
  ±2.5 ms, because other jobs shared the machine.

## Part 2 (candidate 2)

Two candidates were built this round:

* **Candidate 1 (part 1 below) was not adopted.** It replaced the word score with a learned model
  and failed two held-out guards.
* **Candidate 2 (this part) is adopted and on by default.** It keeps the round-18 word score as the
  base and adds:
  * a discriminative length model with new duration inputs;
  * a Japanese "which sound was said" head;
  * a capped two-sided step on the word log-odds.

It passed every dev guard and every held-out guard. The benchmark targets are **not** met.

**Switches.** `PRONUNCIADA_JA_ROUND19=0` restores round 18. `PRONUNCIADA_JA_ROUND19_OFF=merge,labels,lenmodel,step`
turns single parts off.

**Held-out looks.** This was the **third look** at the Lingolix benchmark this round:

1. the round-18 baseline, for the diagnosis;
2. candidate 1;
3. candidate 2.

## Part 2 results (candidate 2, frozen at commit 7a3dfde before its held-out run)

### Lingolix test 1, Japanese

* Both runs go through the benchmark runner's own code path: Scorer → `to_azure` → the benchmark's
  Azure reduction → its metrics (`bench/round19_ja/lx_run.py`; table
  `results/tables/round19_ja_lingolix_runner.json`).
* Intervals: bootstrap over clips.
* Paired p-values: bootstrap over clips, after − before.
* "Main" rules: the later "main" rules (f8750aa) drop words that Whisper heard differently from the
  prompt.

| | v2026.10 before | v2026.10 after | main before | main after | target | met? |
|---|---|---|---|---|---|---|
| word AUC | 0.846 [0.798, 0.891] | **0.917 [0.885, 0.948]** | 0.855 | **0.925 [0.895, 0.953]** | ≥ 0.96 | **no** |
| FA@80 | 20.1 % [12.7, 37.4] | **13.4 % [3.6, 26.4]** | 18.5 % | **11.9 % [2.7, 24.6]** | ≤ 4 % | **no** |
| length items flagged by own ErrorType | 19 / 40 | **27 / 40** | 19 / 40 | 27 / 40 | ≥ 37 | **no** |
| flag recall / precision / FA on correct words | 0.56 / 0.41 / 8.4 % | 0.64 / 0.44 / 8.6 % | 0.56 / 0.47 / 7.4 % | 0.64 / 0.50 / 7.5 % | | |

Paired differences (v2026.10 rules):

* AUC +0.071 [+0.031, +0.113], p = 0.001.
* FA@80 −6.7 points [−22.8, +1.8], p = 0.12.

The main rules give the same picture: AUC [+0.030, +0.113], p = 0.001.

By bucket (v2026.10):

| bucket | before | after |
|---|---|---|
| length | 0.732 | 0.872 |
| sub_subtle | 0.910 | 0.947 |
| sub_moderate | 0.926 | 0.953 |
| sub_gross | 0.960 | 0.989 |
| indel | 0.909 | 0.916 |

**Remaining misses.** These are the planted words still scored above the threshold that catches
80 %, an aggregate count:

* 10 length;
* 5 indel;
* 3 subtle;
* 2 moderate.

That threshold now sits at word score 0.982 (0.898 before), because correct words are nudged up.
FA@80 is therefore set by the 20 % of planted words that still look almost perfect, not by false
alarms.

Not flagged:

* 13 of 40 length items;
* 12 of 20 subtle;
* 7 of 15 indel.

### Held-out guards

The "before" outputs are the round-18 runs from part 1, which use the same code path with the
switch at 0. Tables: `results/tables/round19_ja_guards2.json` and `round19_ja_length_after2.json`.

| guard | round 18 | candidate 2 | verdict |
|---|---|---|---|
| FLEURS-swap ja test, far AUC | 0.991 | **0.995** (p = 0.002, better) | holds |
| far hit@5 % | 97.3 % | 98.7 % | holds |
| near AUC | 0.968 | **0.976** (p = 0.03, better) | holds |
| near hit@5 % | 84.0 % | 89.3 % (p = 0.08) | holds |
| natives below 0.6 (published weakness; Azure 2.0 %) | 3.27 % | **2.41 %** (p = 0.01, better) | holds |
| natives below 0.8 | 6.9 % | 3.4 % | holds |
| cv-real ja Test A, sentence prob AUC | 0.798 | 0.803 (+0.004, p = 0.06) | holds |
| cv-real ja word mean AUC | 0.788 | 0.803 (+0.014 [−0.007, +0.036], p = 0.18) | holds |
| cv-real ja word min AUC | 0.775 | 0.791 (+0.016 [−0.005, +0.038], p = 0.13) | holds |
| length table (FLEURS test): planted length sounds caught at 5 % native FA | 95.5 % (1,701) | 97.3 % (2,136, more sounds judged) | holds |
| length-swapped words caught at 5 % native word FA | 54.0 % | **66.0 %** | holds |
| native word acceptance | 97.06 % | 97.62 % | holds |

* **One length-table row fell:** "long vowel expected, short said" words fell from 60.1 % to
  45.8 %. Geminate rows rose from 8.1 % to 37.2 % and from 41.9 % to 78.6 %.
* **FLEURS-swap natives below 0.6.** The native false alarms are now 2.41 % on FLEURS-swap ja test,
  against Azure's 2.0 %.

### What candidate 2 is

**(A) Structure.** Round 18's word probability (sounds × present × contrast) stays the base. Its
log-odds move by a capped two-sided step (`pronunciada/word_step.py`, the Korean round-19 form):

```
step = min(3, 3 + 1.5·min(s − 2, 0) + 1.0·min(z_len, 0) − 0.5·extra)
```

* `s`: the word's lowest identity margin. Each sound's margin is taken relative to the 2 % native
  quantile of its class and position (first / middle / final), as the Korean branch learned.
* `z_len`: the word's lowest length logit. It replaces the round-15 length factor.
* `extra`: sounds the free decode hears over the word beyond its text (`word_match.py`).

A word whose sounds all clearly match gains at most +3 log-odds. Clear evidence of a wrong sound,
wrong length or extra sound pushes it down without a floor. Whole-word errors keep their low base,
and far swaps improved. The form and settings were chosen on dev:

* by a grid with the native guard applied first (`bench/round19_ja/stepgrid.py`);
* both speaker folds picked the same setting;
* then the dev guards below were checked.

An unconstrained logistic with the old log-odds as a fixed offset was tried first. It saturated the
constant at the cap and gave the identity evidence zero weight. Sign-constrained fits did the
same. The grid was used instead.

**(B) Length** (`pronunciada/length_r19.py`, `data/length/ja_r19.json`, 98 kB).

* **New inputs.** All are measured as "how long", whatever the text expects:
  * the stable-spectrum vowel span on the 10 ms filterbank (cosine ≥ 0.85 to the centre frames);
  * the closure energy dip between the neighbouring vowels;
  * the local mora duration, excluding the word;
  * the word's duration against its mora count (counting this sound short);
  * the counterfactual: best-path log-likelihood of the long form against the short form (aː / a,
    tt / t) on the posteriors around the sound.
* **Lesson from the first try: text-relative inputs cannot be learnt from natives.** Inputs relative
  to the text (the counterfactual "expected form vs twin", the head evidence for the expected
  value) never vary with the label on natives. A native-only fit therefore lost to the shipped
  channel: CV word AUC 0.836 against 0.930.
* **Fitting.** The model is fitted **discriminatively**: natives against planted long / short swaps
  aligned as the swapped text (`bench/round19_ja/swapcollect.py`, `length_fit3.py`). Data: Common
  Voice ja train (1,800 clips, 583 speakers) plus FLEURS ja train. The model gives P(the sound
  matches its text) directly.
* **Sound level, dev pool:** natives against planted swaps, sounds aligned as at runtime.

  | | CV dev (19,797 natives / 2,978 swaps) | FLEURS validation |
  |---|---|---|
  | shipped | 0.970 | 0.981* |
  | round 18 inputs + head, refitted | 0.980 | |
  | + durations | 0.981 | |
  | **+ counterfactual (shipped as `lenmodel`)** | **0.984** | **0.991** |
  | catch at 2 % native FA, shipped → new | 72.7 % → 87.2 % | |
  | vowels | 0.971 → 0.972 | |
  | consonants | 0.971 → **0.995** | |

  *The shipped head combination was fitted on FLEURS validation, so its FLEURS figure is
  optimistic.

  The ≥ 0.97 aim is met on CV. The counterfactual is what fixed geminates. Vowel length is still
  at 0.97.
* **Error flag.** A judged sound with length_prob ≤ 0.03 marks the word mispronounced. This was
  chosen on dev at a native word flag rate no higher than the 1.0.2 rule's:

  | | 1.0.2 rule (shipped model) | new threshold |
  |---|---|---|
  | dev native words flagged | 1.36 % | 1.18 % |
  | planted length units caught | 49 / 114 | 61 / 114 |

  The coaching thresholds are the new model's native quantiles at the shipped native rates.

**(C) Identity head** (`pronunciada/ja_identity.py`, `data/coach/ja_identity.npz`, 0.97 MB).

* **Model:** one 256-unit GELU layer on enc1|enc3|enc5 plus the neighbours' mean, 32 classes:
  * the vowels short and long;
  * N and j;
  * consonants, with kː and tː the only geminate classes common enough to keep.
* **Training:** native CV ja train plus FLEURS train, 172k sounds; the training loop is the Korean
  branch's.
* **Dev accuracy:** 90.5 % (CV 87.9 %, FLEURS 93.4 %).
* **Expected class:** taken from the reference sounds being judged, as the Korean branch advised.

**(D) Latency.** 60 dev clips, 4 threads, warm, median:

* round 18: 64.5 ms; candidate 2: 67.6 ms (+3.0 ms);
* by stage: length +1.8 ms (filterbank spans and counterfactual Viterbi), word step +0.8 ms
  (identity MLP and free-decode match), assembly +0.1 ms;
* on the benchmark clips: median 86.9 → 88.4 ms per request (+1.5 ms, 2 threads).

Candidate 1's 6.4 ms was never broken down by stage at the time. It ran the free-decode alignment
on every word plus 200 trees per word; both are gone or cheaper now. Size added for Japanese:
about 1.07 MB.

### Dev numbers (runtime dumps; `results/tables/round19_ja_devguards.json`)

| dev | round 18 | fixes 1–2 only | candidate 2 |
|---|---|---|---|
| dev set AUC (744 rows, 428 planted) | 0.811 [0.784, 0.839] | 0.815 | **0.862 [0.837, 0.887]** (p < 0.001) |
| … CV part | 0.769 | 0.776 | 0.829 |
| FA@80 | 29.9 % | 29.9 % | 18.4 % (p < 0.001) |
| length / subtle / moderate / gross / indel | 0.740 / 0.831 / 0.912 / 0.924 / 0.685 | 0.769 / 0.827 / 0.910 / 0.924 / 0.677 | 0.827 / 0.890 / 0.935 / 0.955 / 0.706 |
| own flag: planted length units | 55 / 114 | 58 / 114 | 71 / 114 |
| own flag: flag rate on control units | 20.0 % | 20.2 % | 19.9 % |
| **dev guard** FLEURS-validation natives < 0.5 / < 0.6 | 2.11 / 3.52 % | 2.23 / **3.70 %** ✗ | 2.02 / 2.41 % |
| **dev guard** CV-dev natives < 0.5 / < 0.6 | 8.83 / 10.58 % | 8.89 / 10.76 % ✗ | 7.33 / 7.90 % |
| **dev guard** FLEURS-swap dev far / near AUC | 0.9975 / 0.9689 | 0.9974 / 0.9662 | 0.9990 / 0.9782 |
| **dev guard** cv-real ja DEV half word mean / min AUC | 0.742 / 0.741 | 0.747 / 0.745 | 0.766 / 0.757 |

* **Fixes 1–2 alone do not ship.** On their own they slightly raise native words below 0.6, which
  fails the native guard, so they ship only inside candidate 2.
* **Dev guards added after candidate 1's failure:**
  * the frozen FLEURS-swap dev set (bench/calib/bias_grid.py), with near and far swaps;
  * the cv-real ja DEV half;
  * native word rates on both FLEURS validation and CV dev.

## Part 1 (candidate 1, not adopted)

**Outcome (candidate 1).** The frozen candidate 1 raised the Lingolix test-1 Japanese word AUC from 0.846 to 0.884.
That is +0.038 with p = 0.115, so it is not significant on 200 clips. It **failed two of our own
held-out guards**:

* FLEURS-swap far swaps: AUC 0.991 → 0.958 (p = 0.002).
* cv-real Test A Japanese word mean AUC: −0.046 (p = 0.012).
* cv-real Test A Japanese word min AUC: −0.093 (p = 0.001).

It also costs about 6 ms per benchmark clip, over the 5 ms budget. It was switched off and then
removed: its "wordcal" part is no longer in the package, and part 2 replaced it. The rest of this
part is kept as written at the time. No target was met:

| target | before | after (candidate) | met? |
|---|---|---|---|
| Lingolix ja word AUC ≥ 0.96 | 0.846 [0.796, 0.889] | 0.884 [0.839, 0.924] | no |
| FA@80 ≤ 4 % | 20.0 % [12.8, 37.0] | 16.6 % [8.3, 30.0] | no |
| length items caught by own flag ≥ 37 / 40 | 19 | 21 | no |

Benchmark numbers come from `bench/round19_ja/unit_eval.py`, our re-implementation of the
benchmark's unit mapping and metrics. On the published responses it gives exactly the published
0.846 and 20.0 %. Intervals are a bootstrap over clips: the benchmark has no speaker ids, and the
CV speaker map gives 184 speakers for 200 clips.

## Diagnosis on the benchmark (aggregates only, before any change)

| bucket | n | AUC |
|---|---|---|
| length | 40 | 0.732 |
| sub_subtle | 20 | 0.910 |
| sub_moderate | 15 | 0.926 |
| sub_gross | 10 | 0.959 |
| indel | 15 | 0.909 |

* **FA@80 is a miss problem, not a false-alarm problem.** The threshold that catches 80 % of
  changed words sits at word score 0.898. 20 changed words score above it, and 14 of those are
  length items.
* **Length coverage.** The length channel judged a sound in 32 of the 40 length words. Its factor
  fell below 1 in only 24 of them.
* **Length false alarms on correct words.** Length probability < 0.03 on 6.1 % of expected-long
  vowels, 5.3 % of geminates and 0.7 % of short vowels. Common Voice speech is faster than FLEURS,
  and long vowels get shortened.
* **Where the 188 correct units above the threshold come from:**
  * 113: mild sound-score dips only;
  * 44: the length factor;
  * 13: low sounds;
  * 12: present < 0.5;
  * 5: the contrast factor;
  * 1: no system word overlapping the unit.
* **Mechanisms found in the code** (they confirm the brief's observations):
  1. **っ lands in the next word.** Open JTalk splits 行っ|た, and the front-end merges っ into the
     next mora, so `tt` sits in the word た with its vowel in the previous word. The length labels
     required both vowels in the same word, so the sound was never judged.
  2. **Repeated vowels.** Readings such as 場合 バアイ and 思惟 シイ came out as two short vowels.
     A short vowel next to a vowel is never judged, and for CTC two identical tokens in a row are
     awkward.
  3. **Vowel pairs are never judged.** No short vowel next to another vowel was judged, so 倍 /
     恋 against 場合 / 行為 could not be detected.

## Dev set (built first; no benchmark clip, speaker or prompt)

`bench/round19_ja/cv_pool.py` and `plant.py` built it.

* **Exclusions.**
  * The 200 benchmark clips were mapped to 184 Common Voice speakers through `validated.tsv`. Those
    speakers and the benchmark prompts are excluded everywhere.
  * Speakers whose cv-real half is "test" are excluded too.
* **Audio.**
  * Dev: CV ja `dev` split, 680 clips from 368 speakers, plus FLEURS validation.
  * Fit: CV ja `train` split, 1800 clips from 583 speakers, plus FLEURS train.
  * CV's splits are speaker-disjoint, and this was checked again. All CV audio is CC0.
  * Clips are 2–8 s with prompts of at least 8 characters, like the benchmark (2.1–7.9 s, 8–40
    characters).
* **Units.** Open JTalk accent phrases, which are close to the benchmark's bunsetsu units.
* **Edits.** One unit's core is replaced by a real word (cores seen in CV ja and FLEURS sentences).
  The new word differs from the original by a length change (ー, っ, repeated vowel), one
  substituted sound, or one sound added or dropped. Substitutions are graded by articulatory
  feature distance. The front-end must leave every other unit unchanged.
* **Size.**
  * Dev: 744 rows. 429 changed (length 114, subtle 128, moderate 79, gross 46, indel 62) and
    315 controls, 76 % CV.
  * Fit set: 2295 rows, 1061 controls.

The dev set is harder than the benchmark: baseline AUC 0.811 against 0.846, and 0.769 on its CV
part. Its indel bucket is much harder (0.685), because most of its indel edits drop a sound that
natives barely realise.

## Fixes, each with its dev delta

Dev deltas are paired, with a speaker-grouped bootstrap.

| step | dev AUC | Δ vs base | FA@80 |
|---|---|---|---|
| round 18 (base) | 0.811 [0.784, 0.839] | | 29.9 % |
| + merge: a repeated vowel mora becomes one long vowel (front-end) | 0.813 | +0.002 (p = 0.006) | 29.7 % |
| + labels: length judged across word boundaries (行っ\|た, 変え\|て) and for short vowels next to a vowel, except e+i, o+u and repeated vowels | 0.815 | +0.005 (p = 0.14); length bucket 0.740 → 0.769 | 29.9 % |
| + wordcal: word recalibration (below) | **0.870** [0.845, 0.893] | **+0.060 (p < 0.001)** | **19.7 %** (p < 0.001) |

On the dev parts: CV 0.769 → 0.844, FLEURS 0.881 → 0.923.

**wordcal** (`pronunciada/word_cal.py`, `data/calib/wordcal/ja.json`, 35 kB):

* **Model.** 200 boosted depth-3 trees over the word's existing pieces:
  * sounds logit, lowest length logit, contrast factor, present probability;
  * word-table statistics of the sound evidence;
  * a new free-decode match (`pronunciada/word_match.py`): greedy tokens over the word's region,
    aligned to the word's sounds and their alternatives, giving insertion / deletion /
    substitution counts.
* **Fit.** On the planted fit set.
* **Native map.** Out-of-fold logits are mapped by quantiles onto the shipped word logit of
  untouched FLEURS-train words. In-sample logits raised FLEURS-validation native false alarms by
  0.6 point.
* **Offset.** A uniform logit offset of +0.3 restores FLEURS-validation native rates; it does not
  change rankings:

  | FLEURS validation natives | round 18 | candidate |
  |---|---|---|
  | words with prob < 0.5 | 2.11 % | 1.99 % |
  | words with prob < 0.6 | 3.52 % | 2.81 % |

* **Ablation.** Without the match counts, dev AUC is 0.867 and the indel bucket 0.763 instead of
  0.822.
* **Parity.** Offline and runtime agree on 740 of 742 dev clips.

**Tried and dropped on dev:**

* **Length-model refit** on FLEURS train plus CV train natives (2797 clips) with the new labels,
  with or without the round-17 head evidence as a tree input. It was worse than the shipped
  channel on CV dev planted length words: sound-level AUC 0.903 against 0.930, catch at 2 %
  natives 37 % against 50 %. A first version leaked the label through the head-evidence sign and
  was fixed before this comparison. The shipped timing model was kept.
* **Lower word-factor floor** (0.4 → 0.02, with knee and width). It did not help with either length
  model (dev AUC within ±0.003).

## Held-out runs, in order

1. **Lingolix ja baseline:** the diagnosis dump of round-18 code. It reproduces 0.846 / 20.0 %.
2. **Frozen candidate 112446b on Lingolix ja,** with the baseline re-scored alongside for a paired
   comparison (`results/tables/round19_ja_lingolix.json`):

   | | before | after |
   |---|---|---|
   | AUC | 0.846 [0.796, 0.889] | 0.884 [0.839, 0.924] |
   | FA@80 | 20.0 % | 16.6 % |
   | length bucket | 0.732 | 0.816 |
   | sub_subtle | 0.910 | 0.894 |
   | sub_moderate | 0.926 | 0.990 |
   | sub_gross | 0.959 | 0.929 |
   | indel | 0.909 | 0.914 |
   | own flag: recall / false alarms | 0.56 / 8.4 % | 0.58 / 8.7 % |

   Paired differences: AUC Δ +0.038 [−0.008, +0.084], p = 0.115; FA@80 Δ −3.4 points
   [−20.4, +4.8], p = 0.21.

   The flag barely moves because the Azure-layout ErrorType comes from coaching items, not from
   the word probability.
3. **Guards, before / after on the same clips:** `bench/round19_ja/guards.py` →
   `results/tables/round19_ja_guards.json`, and `bench/length/heldout_runtime.py` →
   `results/tables/round19_ja_length_{before,after}.json`.

| guard | before | after | verdict |
|---|---|---|---|
| FLEURS-swap ja near AUC | 0.968 | 0.965 (Δ −0.003, p = 0.57) | holds |
| FLEURS-swap ja near hit@5 % | 84 % | 88 % (p = 0.05) | holds |
| **FLEURS-swap ja far AUC** | 0.991 | **0.958** (Δ −0.033, p = 0.002) | **fails** |
| **FLEURS-swap ja far hit@5 %** | 97 % | **77 %** | **fails** |
| FLEURS-swap natives below 0.6 | 3.27 % | 2.64 % (p = 0.02) | better |
| cv-real ja Test A sentence prob AUC | 0.798 | 0.803 (Δ +0.004, p = 0.06) | holds |
| **cv-real ja word mean AUC** | 0.788 | **0.742** (Δ −0.046 [−0.081, −0.013], p = 0.012) | **fails** |
| **cv-real ja word min AUC** | 0.775 | **0.682** (Δ −0.093 [−0.146, −0.040], p = 0.001) | **fails** |
| length table: planted length sounds caught at 5 % native FA | 95.5 % (1701) | 92.5 % (2136, more sounds judged) | |
| length table: length-swapped words caught | 54.0 % | 55.4 % | |
| length table: native word acceptance | 97.06 % | 97.87 % | |

**Why it failed.** The trees were fitted on one-sound edits only. A word whose whole text is
wrong (a far swap) sits outside that range and lands around 0.55, where the old product gave about
0.09: mean planted-word score 0.087 → 0.55. Trees cannot extrapolate below their lowest leaves.
On cv-real, the reviewers' rejected clips are often gross mismatches of the same kind.

**Combinations checked on dev after the failure (not run on held-out):**

| combination | dev AUC | FA@80 |
|---|---|---|
| min(shipped, recalibrated) | 0.850 | 24.2 % |
| geometric mean | 0.856 | 24.5 % |

Either keeps the gross-mismatch floor of the shipped product. They are the obvious next candidate,
but they need a far-swap dev check (the frozen FLEURS-swap dev set of `bench/calib/bias_grid.py`)
and the cv-real dev half before a second held-out run.

## Size and latency

* **Size:** +35 kB of data (`data/calib/wordcal/ja.json`).
* **Latency:** the median per benchmark clip rose 94.1 → 100.5 ms (+6.4 ms on the shared laptop;
  +3.4 ms on dev clips). Most of it is the pure-Python match alignment, which would need a
  vectorised version to fit the 5 ms budget.
* **Off by default:** every other language, and Japanese with the default switch, run the round-18
  code paths.

## Owner hypotheses (IDEAS.md) and the Korean identity head

* **Counterfactual long / short alignment.** Not built. The shipped channel already reads the
  acoustic model's own length-token posterior (`colon_post`) and the closure silence before a
  stop, the direct small-っ cue. After the new labels, 86 of 89 CV dev length units are judged.
  What limits the score is how the evidence is turned into a word score, not coverage.
* **Signal-processing data.** Not needed: native data did not limit the refit, which simply lost to
  the shipped model.
* **A Japanese "which sound was said" head**, like the Korean one, is the natural next step for
  the subtle / gross buckets. Not attempted this round.

## Files

* **Package:** `pronunciada/ja_round19.py` (switches), `text/ja.py` (merge), `length.py` (labels),
  `word_cal.py`, `word_match.py`, `scorer.py` (hooks), `data/calib/wordcal/ja.json`.
* **Bench:** `bench/round19_ja/`: cv_pool, plant, dump, unit_eval, collect, length_fit, lendiag,
  wordcal, parity, combine, guards, tables.
* **Tests:** `tests/test_ja_round19.py`.
* **Tables:** `results/tables/round19_ja_{lingolix,dev,guards,length_before,length_after}.json`.
* **Caches:** dev and fit data under `$PRONUNCIADA_CACHE/round19/japanese/` (about 0.6 GB of
  audio).
