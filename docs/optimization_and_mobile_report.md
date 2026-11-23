# Size and CPU speed: measurements

Status (2026-11-23): baselines recorded on commit cbc6d6a (`main` 04e5c8d plus the harness);
candidates measured. No package default and no shipped file has been changed by this work.

## How this was measured

**Configurations.** A configuration is a source tree (a git revision or the working tree), a
precision (fp32 is the package default and the precision of `results/SCOREBOARD.md`; int8 is the
optional small model), and
optionally a different backbone ONNX file, a folder of per-language adapters and ONNX Runtime
session options. The package reads these through three environment variables
(`PRONUNCIADA_BACKBONE`, `PRONUNCIADA_ADAPTER_DIR`, `PRONUNCIADA_ORT_OPTIONS`,
`pronunciada/_overrides.py`). With the variables unset every output is unchanged.

**Latency** (`python -m bench.opt.profile latency`). One fresh process per configuration, language
and thread count (1, 2, 4 ONNX Runtime threads). Load = process start to Scorer built; cold
start = process start to the first score returned. Steady state: every clip scored in several
rounds; within a round the configurations take turns clip by clip and the order flips each round.
Each call decodes the audio file and runs the whole pipeline (the posterior cache is cleared).
Stage times are the Scorer's own `timing_ms`. Two configurations are compared on the same clips
with a paired interval: per clip the median over rounds, the difference of the means, and a 95 %
clip-bootstrap interval.

**Memory** (`profile rss`). Peak resident set size of a fresh process, read from the operating
system (`getrusage`): after loading only, and after scoring the clips.

**Disk** (`profile footprint`). Every file of the package and of the model folder. Which files a
language needs is recorded while a Scorer of that language is built and scores one clip with
`details()`, `coaching()`, tips and the Azure layout (every `open()` and every ONNX Runtime
session).

**Clips.** `bench/opt/clips.json`: 20 FLEURS train recordings per language for all ten languages
(practice data, ids only), and the 200 speechocean762 test recordings `bench/ondevice.py` times.
The speechocean762 clips are used for timing only.

**Quality gate** (`python -m bench.opt.gate`). The held-out suite behind the scoreboard is run for
the candidate in isolation (`bench/opt/suite.py`) with cached Azure responses only, and compared
with a baseline run of the same precision row by row. A candidate fails if a speechocean762
Pearson correlation drops by more than 0.002, if an Azure tally row moves down a class, or if a
language degrades materially: a row significantly worse than the baseline in its own paired test,
a detection AUC down by more than 0.002, or native false alarms up by more than 0.1 point. A
candidate whose per-clip outputs equal the baseline's passes by identity.

**Numerical parity** (`python -m bench.opt.parity`). The same practice clips scored by the
reference and the candidate; max and mean absolute difference of the backbone posteriors and of
every score, and the number of clips where any score differs.

## Machine

Apple M3 Pro, 12 cores, 18 GB, macOS; Python 3.12, numpy 2.5.5, onnxruntime 1.32.0 (CPU). The
machine was shared with other jobs; the 1-minute load average is printed with every table. Timings
taken at different load levels are not compared with each other; comparisons are the paired,
interleaved ones.

## The fp32 baseline run

`python -m bench.opt.gate record-baseline --code cbc6d6a --precision fp32`, steps run three at a
time (`bench/opt/suite.py`), every credential variable removed, Azure numbers from cached responses
only. Wall time 2,480 s (41 min) while other jobs ran; longest steps: eval_languages 2,459 s,
contrast audit 1,507 s, coaching tips 896 s, eval_english 526 s.

**Contrast-audit edit plans.** The frozen runs behind the committed tables used German and Japanese
edit plans regenerated on 2026-11-22. Those plans exist only in the frozen runs' cache overlays;
the main cache still holds the 2026-11-18 plans, which `bench/round19_merge/overlay.py` copies. A
first comparison therefore differed in 29 German and Japanese audit rows and nowhere else. The plans
of the frozen run are now pinned (`$PRONUNCIADA_CACHE/opt/audit_plans/<lang>/plan_250.json`, copied
from the round-20 false-alarm release check's overlay) and copied into every run
(`suite.pin_audit_plans`). SHA-1 of the plans used: de `22a54bbc4a`, ja `5e53e0ab69`, en `0c4eac2f2d`,
es `5b80a50016`, fr `b8df19cbe1`, it `45f2ecc70a`, ko `348c3ba171`, pt `2f925469e1`, ru `3173eb111c`,
zh `80006e7e66` (first ten hex digits). After the German and Japanese audit was re-run with them,
**all eleven tables equal the tables committed on `main` key by key, and all 416 scoreboard rows are
equal** (`python -m bench.opt.gate tables-vs-rev --rev 04e5c8d`). The committed tables came from
serial frozen runs, so this also shows that running steps side by side gives identical tables.

## Baseline

### Latency

<!-- opt:baseline-latency -->
Clips: practice, 3 rounds; machine: Apple M3 Pro, 12 cores, load average 6.4 at the start, 4.1 at the end; onnxruntime 1.32.0; 2026-11-23 03:45.

| config | lang | threads | load s | cold start s | wall p50 ms | mean | p90 | peak RSS MB |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| fp32 | en | 1 | 0.57 | 1.01 | 356.8 | 346.3 | 439.1 | 706 |
| int8 | en | 1 | 0.58 | 0.91 | 241.3 | 235.1 | 304.8 | 554 |
| fp32 | en | 2 | 0.58 | 0.89 | 227.9 | 220.4 | 278.9 | 703 |
| int8 | en | 2 | 0.57 | 0.82 | 170.2 | 164.5 | 205.9 | 551 |
| fp32 | en | 4 | 0.58 | 0.86 | 184.4 | 179.3 | 232.4 | 713 |
| int8 | en | 4 | 0.61 | 0.86 | 145.9 | 145.2 | 189.7 | 564 |
| fp32 | de | 1 | 0.82 | 1.19 | 323.1 | 379.2 | 581.5 | 970 |
| int8 | de | 1 | 0.83 | 1.11 | 220.7 | 261.8 | 407.0 | 823 |
| fp32 | de | 2 | 0.82 | 1.09 | 210.3 | 246.1 | 379.6 | 972 |
| int8 | de | 2 | 0.82 | 1.05 | 156.6 | 186.6 | 292.1 | 824 |
| fp32 | de | 4 | 0.81 | 1.04 | 151.4 | 177.0 | 269.5 | 985 |
| int8 | de | 4 | 0.82 | 1.02 | 120.3 | 146.3 | 224.2 | 831 |
| fp32 | es | 1 | 0.51 | 0.84 | 300.6 | 324.6 | 471.4 | 723 |
| int8 | es | 1 | 0.50 | 0.74 | 201.4 | 220.3 | 322.4 | 569 |
| fp32 | es | 2 | 0.55 | 0.79 | 195.4 | 210.3 | 299.1 | 723 |
| int8 | es | 2 | 0.55 | 0.74 | 145.2 | 155.8 | 221.9 | 568 |
| fp32 | es | 4 | 0.53 | 0.71 | 140.6 | 152.8 | 215.2 | 724 |
| int8 | es | 4 | 0.57 | 0.73 | 115.8 | 124.3 | 175.3 | 572 |
| fp32 | fr | 1 | 0.59 | 0.82 | 291.1 | 329.1 | 527.6 | 677 |
| int8 | fr | 1 | 0.58 | 0.74 | 188.4 | 214.2 | 349.6 | 542 |
| fp32 | fr | 2 | 0.58 | 0.73 | 175.5 | 201.0 | 322.4 | 675 |
| int8 | fr | 2 | 0.59 | 0.71 | 123.7 | 142.7 | 230.4 | 542 |
| fp32 | fr | 4 | 0.59 | 0.72 | 124.2 | 141.1 | 226.3 | 679 |
| int8 | fr | 4 | 0.60 | 0.70 | 96.6 | 109.5 | 176.8 | 545 |
| fp32 | it | 1 | 0.51 | 1.01 | 333.8 | 341.9 | 445.3 | 685 |
| int8 | it | 1 | 0.52 | 0.87 | 224.9 | 231.4 | 302.6 | 532 |
| fp32 | it | 2 | 0.59 | 0.93 | 222.2 | 233.9 | 305.4 | 675 |
| int8 | it | 2 | 0.54 | 0.81 | 165.7 | 173.9 | 228.8 | 522 |
| fp32 | it | 4 | 0.56 | 0.82 | 152.0 | 157.4 | 202.2 | 688 |
| int8 | it | 4 | 0.53 | 0.75 | 124.3 | 128.7 | 164.5 | 534 |
| fp32 | pt | 1 | 0.51 | 1.14 | 411.7 | 421.0 | 584.6 | 800 |
| int8 | pt | 1 | 0.49 | 0.95 | 283.9 | 288.8 | 401.2 | 624 |
| fp32 | pt | 2 | 0.50 | 0.92 | 266.9 | 274.1 | 375.5 | 801 |
| int8 | pt | 2 | 0.48 | 0.80 | 198.3 | 204.5 | 284.9 | 630 |
| fp32 | pt | 4 | 0.50 | 0.81 | 190.8 | 195.5 | 272.5 | 802 |
| int8 | pt | 4 | 0.51 | 0.79 | 156.9 | 161.1 | 227.9 | 625 |
| fp32 | ru | 1 | 0.65 | 1.06 | 313.0 | 323.6 | 488.8 | 770 |
| int8 | ru | 1 | 0.65 | 0.96 | 210.5 | 222.1 | 346.0 | 587 |
| fp32 | ru | 2 | 0.76 | 1.07 | 203.0 | 213.4 | 325.7 | 770 |
| int8 | ru | 2 | 0.69 | 0.93 | 151.8 | 161.3 | 247.9 | 589 |
| fp32 | ru | 4 | 0.65 | 0.88 | 147.5 | 155.1 | 238.2 | 766 |
| int8 | ru | 4 | 0.63 | 0.83 | 123.9 | 128.7 | 200.1 | 591 |
| fp32 | ja | 1 | 0.72 | 1.04 | 343.5 | 353.5 | 480.9 | 925 |
| int8 | ja | 1 | 0.71 | 0.92 | 227.3 | 234.6 | 322.8 | 773 |
| fp32 | ja | 2 | 0.72 | 0.93 | 215.9 | 221.6 | 291.4 | 926 |
| int8 | ja | 2 | 0.72 | 0.88 | 158.5 | 161.4 | 218.8 | 771 |
| fp32 | ja | 4 | 0.71 | 0.87 | 152.8 | 156.2 | 205.6 | 926 |
| int8 | ja | 4 | 0.72 | 0.86 | 123.3 | 125.3 | 171.9 | 773 |
| fp32 | ko | 1 | 0.49 | 0.96 | 377.7 | 394.2 | 520.6 | 826 |
| int8 | ko | 1 | 0.48 | 0.81 | 246.6 | 262.4 | 347.1 | 660 |
| fp32 | ko | 2 | 0.46 | 0.75 | 232.6 | 248.1 | 326.9 | 835 |
| int8 | ko | 2 | 0.43 | 0.66 | 169.0 | 179.7 | 239.6 | 660 |
| fp32 | ko | 4 | 0.46 | 0.69 | 160.5 | 169.6 | 217.8 | 830 |
| int8 | ko | 4 | 0.45 | 0.63 | 128.0 | 135.1 | 172.4 | 672 |
| fp32 | zh | 1 | 0.94 | 1.28 | 298.2 | 326.7 | 447.2 | 845 |
| int8 | zh | 1 | 0.93 | 1.17 | 201.0 | 220.5 | 302.3 | 695 |
| fp32 | zh | 2 | 0.93 | 1.16 | 191.0 | 209.1 | 282.8 | 845 |
| int8 | zh | 2 | 0.92 | 1.10 | 140.6 | 154.0 | 206.8 | 695 |
| fp32 | zh | 4 | 0.92 | 1.10 | 137.5 | 149.8 | 197.3 | 845 |
| int8 | zh | 4 | 0.93 | 1.08 | 111.7 | 121.4 | 159.8 | 697 |

Per stage, p50 ms (SentenceResult.timing_ms; load = audio decode + resample, model = ZIPA backbone):

| config | lang | threads | load | g2p | audio_check | fbank | model | adapter | align | features | delivery | calibrate | sentence_head | prosody | length | word_step | assemble |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| fp32 | en | 1 | 1.79 | 0.48 | 0.58 | 10.05 | 281.52 | 0.26 | 18.74 | 4.16 | 15.13 | 8.51 | 4.31 | 7.04 | – | – | 0.72 |
| int8 | en | 1 | 1.80 | 0.46 | 0.57 | 9.95 | 167.46 | 0.26 | 18.98 | 4.20 | 14.82 | 8.60 | 4.27 | 7.17 | – | – | 0.75 |
| fp32 | en | 2 | 1.81 | 0.46 | 0.58 | 10.03 | 157.23 | 0.26 | 18.59 | 4.10 | 14.76 | 8.29 | 3.75 | 5.80 | – | – | 0.72 |
| int8 | en | 2 | 1.79 | 0.46 | 0.57 | 10.04 | 98.15 | 0.27 | 18.42 | 4.19 | 14.72 | 8.18 | 3.69 | 5.74 | – | – | 0.74 |
| fp32 | en | 4 | 1.88 | 0.55 | 0.62 | 9.98 | 107.26 | 0.27 | 20.64 | 4.30 | 15.20 | 8.48 | 3.81 | 6.33 | – | – | 0.74 |
| int8 | en | 4 | 1.83 | 0.49 | 0.60 | 10.01 | 71.73 | 0.27 | 19.61 | 4.29 | 14.93 | 8.59 | 3.81 | 6.89 | – | – | 0.72 |
| fp32 | de | 1 | 1.58 | 0.70 | 0.53 | 9.14 | 252.79 | 0.37 | 25.68 | 3.50 | 13.40 | 3.69 | – | 8.47 | – | – | 1.26 |
| int8 | de | 1 | 1.59 | 0.72 | 0.54 | 9.19 | 149.50 | 0.38 | 26.19 | 3.53 | 13.71 | 3.70 | – | 8.51 | – | – | 1.26 |
| fp32 | de | 2 | 1.59 | 0.72 | 0.54 | 9.33 | 140.28 | 0.37 | 26.06 | 3.55 | 13.60 | 3.70 | – | 6.86 | – | – | 1.27 |
| int8 | de | 2 | 1.61 | 0.72 | 0.53 | 9.07 | 87.33 | 0.38 | 26.50 | 3.57 | 13.57 | 3.72 | – | 6.79 | – | – | 1.27 |
| fp32 | de | 4 | 1.56 | 0.69 | 0.50 | 8.88 | 80.49 | 0.37 | 27.23 | 3.44 | 13.14 | 3.55 | – | 6.51 | – | – | 1.24 |
| int8 | de | 4 | 1.56 | 0.69 | 0.51 | 8.76 | 54.46 | 0.37 | 24.92 | 3.48 | 12.92 | 3.60 | – | 6.64 | – | – | 1.25 |
| fp32 | es | 1 | 1.54 | 0.43 | 0.52 | 8.80 | 241.88 | 0.55 | 16.72 | 3.04 | 12.63 | 3.41 | – | 9.27 | – | – | 0.99 |
| int8 | es | 1 | 1.55 | 0.43 | 0.51 | 8.84 | 142.85 | 0.55 | 17.08 | 3.20 | 13.13 | 3.42 | – | 9.40 | – | – | 1.04 |
| fp32 | es | 2 | 1.58 | 0.45 | 0.52 | 8.82 | 134.99 | 0.58 | 17.26 | 3.21 | 12.86 | 3.58 | – | 7.41 | – | – | 1.11 |
| int8 | es | 2 | 1.60 | 0.45 | 0.53 | 8.93 | 85.55 | 0.57 | 17.77 | 3.12 | 13.06 | 3.47 | – | 7.29 | – | – | 1.02 |
| fp32 | es | 4 | 1.55 | 0.43 | 0.53 | 8.77 | 81.58 | 0.57 | 17.70 | 3.09 | 12.93 | 3.51 | – | 7.28 | – | – | 1.02 |
| int8 | es | 4 | 1.56 | 0.44 | 0.51 | 8.83 | 55.40 | 0.58 | 17.85 | 3.17 | 13.10 | 3.55 | – | 7.31 | – | – | 1.02 |
| fp32 | fr | 1 | 1.63 | 0.37 | 0.50 | 9.18 | 255.87 | 0.37 | 15.31 | 2.59 | 2.09 | 3.12 | – | – | – | – | 0.60 |
| int8 | fr | 1 | 1.62 | 0.38 | 0.50 | 9.21 | 150.89 | 0.38 | 15.33 | 2.65 | 2.10 | 3.08 | – | – | – | – | 0.56 |
| fp32 | fr | 2 | 1.62 | 0.37 | 0.50 | 9.22 | 140.22 | 0.38 | 15.54 | 2.63 | 2.07 | 3.02 | – | – | – | – | 0.56 |
| int8 | fr | 2 | 1.62 | 0.39 | 0.52 | 9.25 | 88.18 | 0.38 | 15.30 | 2.62 | 2.11 | 3.10 | – | – | – | – | 0.58 |
| fp32 | fr | 4 | 1.66 | 0.38 | 0.51 | 9.31 | 88.41 | 0.39 | 15.77 | 2.67 | 2.09 | 3.02 | – | – | – | – | 0.57 |
| int8 | fr | 4 | 1.63 | 0.37 | 0.50 | 9.25 | 60.41 | 0.38 | 15.58 | 2.69 | 2.06 | 2.98 | – | – | – | – | 0.59 |
| fp32 | it | 1 | 1.71 | 0.49 | 0.55 | 9.59 | 265.37 | 0.59 | 18.40 | 3.55 | 13.98 | 3.67 | – | 11.51 | – | – | 0.85 |
| int8 | it | 1 | 1.65 | 0.49 | 0.54 | 9.40 | 157.22 | 0.60 | 18.61 | 3.50 | 14.02 | 3.68 | – | 11.52 | – | – | 0.86 |
| fp32 | it | 2 | 1.91 | 0.54 | 0.58 | 9.71 | 154.86 | 0.65 | 20.44 | 3.65 | 14.52 | 3.86 | – | 9.68 | – | – | 0.88 |
| int8 | it | 2 | 1.73 | 0.51 | 0.58 | 9.93 | 98.08 | 0.63 | 19.78 | 3.68 | 14.59 | 3.83 | – | 9.51 | – | – | 0.90 |
| fp32 | it | 4 | 1.69 | 0.49 | 0.55 | 9.67 | 87.86 | 0.61 | 18.93 | 3.57 | 14.05 | 3.73 | – | 8.87 | – | – | 0.85 |
| int8 | it | 4 | 1.67 | 0.48 | 0.54 | 9.43 | 59.89 | 0.61 | 19.19 | 3.54 | 14.36 | 3.75 | – | 9.05 | – | – | 0.86 |
| fp32 | pt | 1 | 1.99 | 0.62 | 0.61 | 11.64 | 324.89 | 0.44 | 33.02 | 3.39 | 16.54 | 3.66 | – | 11.35 | – | – | 1.13 |
| int8 | pt | 1 | 2.00 | 0.61 | 0.62 | 11.60 | 196.05 | 0.46 | 34.14 | 3.42 | 16.43 | 3.67 | – | 11.46 | – | – | 1.12 |
| fp32 | pt | 2 | 1.99 | 0.61 | 0.63 | 11.93 | 181.85 | 0.46 | 34.72 | 3.43 | 16.52 | 3.66 | – | 9.17 | – | – | 1.12 |
| int8 | pt | 2 | 2.00 | 0.62 | 0.65 | 12.30 | 113.09 | 0.45 | 34.62 | 3.46 | 16.26 | 3.65 | – | 8.93 | – | – | 1.15 |
| fp32 | pt | 4 | 2.00 | 0.61 | 0.62 | 11.83 | 105.01 | 0.47 | 33.71 | 3.45 | 16.23 | 3.60 | – | 9.04 | – | – | 1.12 |
| int8 | pt | 4 | 1.97 | 0.63 | 0.61 | 11.74 | 71.81 | 0.46 | 33.52 | 3.43 | 15.82 | 3.65 | – | 8.97 | – | – | 1.12 |
| fp32 | ru | 1 | 1.62 | 0.70 | 0.51 | 8.68 | 241.62 | 0.56 | 20.76 | 5.08 | 12.79 | 3.49 | – | 9.87 | – | – | 0.89 |
| int8 | ru | 1 | 1.59 | 0.70 | 0.51 | 8.94 | 144.00 | 0.56 | 20.38 | 4.95 | 12.98 | 3.45 | – | 10.00 | – | – | 0.87 |
| fp32 | ru | 2 | 1.73 | 0.75 | 0.53 | 8.81 | 135.39 | 0.57 | 20.76 | 4.97 | 12.89 | 3.47 | – | 7.79 | – | – | 0.87 |
| int8 | ru | 2 | 1.64 | 0.71 | 0.52 | 8.76 | 85.11 | 0.58 | 20.93 | 4.97 | 13.46 | 3.50 | – | 7.92 | – | – | 0.89 |
| fp32 | ru | 4 | 1.58 | 0.70 | 0.51 | 8.70 | 81.96 | 0.58 | 20.63 | 4.96 | 12.67 | 3.42 | – | 7.69 | – | – | 0.87 |
| int8 | ru | 4 | 1.60 | 0.73 | 0.52 | 8.91 | 56.17 | 0.58 | 20.94 | 4.88 | 12.52 | 3.40 | – | 7.82 | – | – | 0.89 |
| fp32 | ja | 1 | 1.86 | 1.85 | 0.58 | 10.18 | 283.07 | 0.41 | 24.19 | 3.42 | 2.63 | 3.68 | – | – | 7.32 | 0.94 | 1.19 |
| int8 | ja | 1 | 1.80 | 1.83 | 0.58 | 10.29 | 168.54 | 0.42 | 23.43 | 3.44 | 2.56 | 3.70 | – | – | 7.45 | 0.94 | 1.16 |
| fp32 | ja | 2 | 1.76 | 1.84 | 0.56 | 10.14 | 157.90 | 0.41 | 24.09 | 3.34 | 2.54 | 3.63 | – | – | 7.25 | 0.94 | 1.17 |
| int8 | ja | 2 | 1.80 | 1.82 | 0.56 | 10.00 | 98.63 | 0.42 | 23.38 | 3.31 | 2.55 | 3.67 | – | – | 7.39 | 0.93 | 1.18 |
| fp32 | ja | 4 | 1.81 | 1.80 | 0.58 | 10.19 | 93.85 | 0.42 | 24.29 | 3.38 | 2.59 | 3.72 | – | – | 7.38 | 0.95 | 1.23 |
| int8 | ja | 4 | 1.80 | 1.85 | 0.58 | 10.32 | 63.64 | 0.42 | 23.26 | 3.37 | 2.57 | 3.63 | – | – | 7.33 | 0.94 | 1.18 |
| fp32 | ko | 1 | 2.00 | 0.42 | 0.62 | 11.35 | 318.77 | 0.70 | 31.99 | 6.46 | 2.33 | 3.33 | – | – | – | – | 0.62 |
| int8 | ko | 1 | 1.94 | 0.38 | 0.61 | 11.20 | 188.32 | 0.68 | 32.43 | 6.35 | 2.34 | 3.38 | – | – | – | – | 0.61 |
| fp32 | ko | 2 | 1.93 | 0.39 | 0.61 | 11.23 | 175.37 | 0.72 | 32.21 | 6.18 | 2.29 | 3.24 | – | – | – | – | 0.58 |
| int8 | ko | 2 | 1.92 | 0.38 | 0.61 | 11.17 | 108.69 | 0.71 | 31.78 | 6.35 | 2.31 | 3.28 | – | – | – | – | 0.59 |
| fp32 | ko | 4 | 1.93 | 0.38 | 0.60 | 11.60 | 102.31 | 0.70 | 31.19 | 6.17 | 2.31 | 3.19 | – | – | – | – | 0.58 |
| int8 | ko | 4 | 1.91 | 0.37 | 0.58 | 11.08 | 69.57 | 0.70 | 31.10 | 6.16 | 2.27 | 3.21 | – | – | – | – | 0.58 |
| fp32 | zh | 1 | 1.55 | 0.70 | 0.52 | 8.59 | 240.78 | 0.36 | 16.34 | 4.18 | 12.70 | 3.13 | – | 9.17 | – | – | 0.93 |
| int8 | zh | 1 | 1.54 | 0.69 | 0.51 | 8.60 | 142.69 | 0.36 | 16.79 | 4.23 | 13.02 | 3.14 | – | 9.20 | – | – | 0.96 |
| fp32 | zh | 2 | 1.58 | 0.71 | 0.52 | 8.67 | 135.34 | 0.36 | 16.45 | 4.17 | 12.70 | 3.09 | – | 7.17 | – | – | 0.95 |
| int8 | zh | 2 | 1.57 | 0.71 | 0.52 | 8.58 | 84.59 | 0.37 | 16.38 | 4.19 | 12.64 | 3.06 | – | 7.33 | – | – | 0.97 |
| fp32 | zh | 4 | 1.59 | 0.71 | 0.53 | 8.72 | 81.03 | 0.37 | 16.71 | 4.14 | 12.69 | 3.10 | – | 7.19 | – | – | 0.96 |
| int8 | zh | 4 | 1.57 | 0.71 | 0.52 | 8.75 | 55.38 | 0.36 | 16.45 | 4.14 | 12.62 | 3.08 | – | 7.31 | – | – | 0.97 |

Paired comparison (per clip median over rounds, B - A; 95 % clip-bootstrap interval of the mean):

| lang | threads | A | B | mean B - A ms [95 % CI] | median ms | relative |
|---|--:|---|---|--:|--:|--:|
| en | 1 | fp32 | int8 | -112.53 [-128.32, -97.66] | -113.08 | -32.3 % |
| en | 2 | fp32 | int8 | -56.41 [-64.15, -49.18] | -56.87 | -25.7 % |
| en | 4 | fp32 | int8 | -34.91 [-42.54, -27.90] | -32.88 | -19.3 % |
| de | 1 | fp32 | int8 | -118.00 [-135.60, -101.92] | -101.93 | -31.1 % |
| de | 2 | fp32 | int8 | -59.40 [-68.35, -51.17] | -52.08 | -24.2 % |
| de | 4 | fp32 | int8 | -31.21 [-36.85, -26.35] | -26.00 | -17.8 % |
| es | 1 | fp32 | int8 | -105.43 [-118.30, -93.70] | -99.91 | -32.3 % |
| es | 2 | fp32 | int8 | -54.39 [-61.14, -48.19] | -50.14 | -26.0 % |
| es | 4 | fp32 | int8 | -28.64 [-31.92, -25.56] | -27.18 | -18.8 % |
| fr | 1 | fp32 | int8 | -115.86 [-132.34, -101.86] | -105.23 | -35.3 % |
| fr | 2 | fp32 | int8 | -58.20 [-66.55, -51.05] | -52.38 | -29.1 % |
| fr | 4 | fp32 | int8 | -30.34 [-34.95, -26.39] | -26.95 | -22.0 % |
| it | 1 | fp32 | int8 | -111.26 [-123.79, -99.81] | -109.18 | -32.5 % |
| it | 2 | fp32 | int8 | -58.38 [-64.12, -53.18] | -56.14 | -25.8 % |
| it | 4 | fp32 | int8 | -28.96 [-32.55, -25.86] | -26.86 | -18.6 % |
| pt | 1 | fp32 | int8 | -132.80 [-153.03, -112.98] | -130.49 | -31.6 % |
| pt | 2 | fp32 | int8 | -70.73 [-81.38, -60.33] | -69.12 | -25.9 % |
| pt | 4 | fp32 | int8 | -33.91 [-38.96, -28.96] | -32.83 | -17.5 % |
| ru | 1 | fp32 | int8 | -101.55 [-115.81, -86.37] | -99.10 | -31.5 % |
| ru | 2 | fp32 | int8 | -51.86 [-59.58, -43.89] | -50.26 | -24.5 % |
| ru | 4 | fp32 | int8 | -26.02 [-29.60, -22.25] | -25.26 | -16.9 % |
| ja | 1 | fp32 | int8 | -118.60 [-135.13, -102.36] | -114.64 | -33.6 % |
| ja | 2 | fp32 | int8 | -60.52 [-69.22, -52.13] | -57.67 | -27.3 % |
| ja | 4 | fp32 | int8 | -31.21 [-35.21, -27.23] | -30.33 | -20.0 % |
| ko | 1 | fp32 | int8 | -131.00 [-147.40, -116.42] | -127.64 | -33.3 % |
| ko | 2 | fp32 | int8 | -67.76 [-76.62, -60.04] | -64.44 | -27.4 % |
| ko | 4 | fp32 | int8 | -35.80 [-41.13, -31.35] | -34.87 | -21.1 % |
| zh | 1 | fp32 | int8 | -106.61 [-121.05, -93.30] | -98.59 | -32.7 % |
| zh | 2 | fp32 | int8 | -55.28 [-62.68, -48.38] | -50.58 | -26.5 % |
| zh | 4 | fp32 | int8 | -28.47 [-32.22, -25.02] | -26.33 | -19.1 % |

Clips: so762, 1 rounds; machine: Apple M3 Pro, 12 cores, load average 4.1 at the start, 4.5 at the end; onnxruntime 1.32.0; 2026-11-23 03:59.

| config | lang | threads | load s | cold start s | wall p50 ms | mean | p90 | peak RSS MB |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| fp32 | en | 1 | 0.60 | 0.73 | 102.4 | 106.6 | 136.1 | 521 |
| int8 | en | 1 | 0.55 | 0.65 | 67.0 | 69.6 | 89.0 | 349 |
| fp32 | en | 4 | 0.56 | 0.65 | 51.5 | 52.6 | 65.0 | 527 |
| int8 | en | 4 | 0.58 | 0.65 | 41.7 | 42.6 | 51.2 | 348 |

Per stage, p50 ms (SentenceResult.timing_ms; load = audio decode + resample, model = ZIPA backbone):

| config | lang | threads | load | g2p | audio_check | fbank | model | adapter | align | features | delivery | calibrate | sentence_head | prosody | length | word_step | assemble |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| fp32 | en | 1 | 1.90 | 0.13 | 0.28 | 3.03 | 82.62 | 0.11 | 2.61 | 0.93 | 4.78 | 2.99 | 1.09 | 1.36 | – | – | 0.17 |
| int8 | en | 1 | 0.67 | 0.13 | 0.27 | 3.04 | 48.54 | 0.11 | 2.62 | 0.95 | 4.76 | 2.99 | 1.09 | 1.36 | – | – | 0.17 |
| fp32 | en | 4 | 0.67 | 0.13 | 0.28 | 3.07 | 32.87 | 0.12 | 2.67 | 0.96 | 4.87 | 2.97 | 1.06 | 1.29 | – | – | 0.17 |
| int8 | en | 4 | 0.66 | 0.13 | 0.28 | 3.06 | 23.13 | 0.12 | 2.65 | 0.94 | 4.83 | 2.97 | 1.05 | 1.29 | – | – | 0.17 |

Paired comparison (per clip median over rounds, B - A; 95 % clip-bootstrap interval of the mean):

| lang | threads | A | B | mean B - A ms [95 % CI] | median ms | relative |
|---|--:|---|---|--:|--:|--:|
| en | 1 | fp32 | int8 | -37.01 [-38.14, -35.97] | -35.48 | -34.7 % |
| en | 4 | fp32 | int8 | -10.07 [-10.45, -9.72] | -9.69 | -19.1 % |
<!-- /opt:baseline-latency -->

### Memory

<!-- opt:baseline-rss -->
Fresh process per row; peak resident set size from getrusage; 20 clips (practice); 2026-11-23 04:00.

| config | lang | threads | load only MB | after scoring MB | load s |
|---|---|--:|--:|--:|--:|
| fp32 | en | 1 | 424 | 694 | 0.54 |
| fp32 | en | 4 | 425 | 691 | 0.55 |
| fp32 | ja | 1 | 584 | 909 | 0.68 |
| fp32 | ja | 4 | 586 | 910 | 0.69 |
| fp32 | zh | 1 | 554 | 831 | 0.93 |
| fp32 | zh | 4 | 555 | 832 | 0.92 |
| int8 | en | 1 | 246 | 538 | 0.55 |
| int8 | en | 4 | 247 | 539 | 0.56 |
| int8 | ja | 1 | 407 | 755 | 0.69 |
| int8 | ja | 4 | 409 | 757 | 0.69 |
| int8 | zh | 1 | 380 | 680 | 0.93 |
| int8 | zh | 4 | 379 | 684 | 0.95 |
<!-- /opt:baseline-rss -->

### Disk footprint

<!-- opt:baseline-footprint -->
Source tree cbc6d6ac7226; 2026-11-23 04:01. models/* sizes are the files in the local model cache; the backbone is downloaded on first use, not shipped in the wheel. A '.emb' file is the copy with extra encoder outputs that the first Scorer builds next to the downloaded model.

| category | files | MB |
|---|--:|--:|
| backbone (runtime-built extra-outputs copy) | 4 | 661.89 |
| backbone | 3 | 462.55 |
| coaching | 14 | 13.19 |
| lexicon | 18 | 10.46 |
| adapter | 10 | 4.57 |
| calibration | 43 | 4.31 |
| prosody | 10 | 2.20 |
| native statistics | 10 | 1.51 |
| code | 68 | 0.71 |
| length | 2 | 0.15 |
| backbone tokens | 1 | 0.00 |
| other data | 1 | 0.00 |

Package (pronunciada/, as in the wheel): 37.10 MB.

Smallest install that can score each language (code + backbone + what it opens):

| lang | code MB | backbone MB | adapter | lexicon | calibration | coaching | other data | total MB | third-party data MB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| de | 0.71 | 260.27 | 0.32 | 5.46 | 0.63 | 1.99 | 0.69 | 270.07 | 0.00 |
| en | 0.71 | 260.27 | 0.07 | 0.97 | 3.97 | 0.01 | 0.61 | 266.61 | 0.50 |
| es | 0.71 | 260.27 | 0.72 | 1.43 | 0.63 | 1.99 | 0.60 | 266.34 | 0.00 |
| fr | 0.71 | 260.27 | 0.32 | 1.13 | 0.63 | 0.01 | 0.15 | 263.22 | 0.00 |
| it | 0.71 | 260.27 | 0.72 | 0.62 | 0.63 | 0.19 | 0.59 | 263.72 | 0.00 |
| ja | 0.71 | 260.27 | 0.32 | 0.00 | 0.63 | 1.18 | 0.29 | 263.40 | 331.20 |
| ko | 0.71 | 260.27 | 0.72 | 0.00 | 0.63 | 2.18 | 0.10 | 264.60 | 0.00 |
| pt | 0.71 | 260.27 | 0.32 | 0.18 | 0.63 | 1.99 | 0.62 | 264.73 | 0.00 |
| ru | 0.71 | 260.27 | 0.72 | 1.64 | 0.63 | 2.17 | 0.64 | 266.77 | 0.00 |
| zh | 0.71 | 260.27 | 0.32 | 0.00 | 0.63 | 2.28 | 0.96 | 265.18 | 42.88 |

Data files no language opened while scoring (20): `models/model.fp16.onnx`, `models/model.int8.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.onnx`, `models/model.int8.emb.onnx`, `models/model.int8.onnx`, `models/model.onnx`, `pronunciada/data/calib/prosody/de.stress.json`, `pronunciada/data/calib/prosody/en.stress.json`, `pronunciada/data/calib/prosody/es.stress.json`, `pronunciada/data/calib/prosody/it.stress.json`, `pronunciada/data/calib/prosody/pt.stress.json`, `pronunciada/data/calib/prosody/ru.stress.json`, `pronunciada/data/calib/prosody/zh.tone_zh_v2.json`, `pronunciada/data/calib/prosody/zh.tone_zh_v3.json`, `pronunciada/data/calib/sentence.npz`, `pronunciada/data/lex/LICENSE.txt`, `pronunciada/data/lex/SOURCES-eu.txt`, `pronunciada/data/prosody/stress.json`, `pronunciada/data/prosody/stress.onnx`, `pronunciada/data/prosody/tone_zh_v2.json`, `pronunciada/data/prosody/tone_zh_v2.onnx`

Source tree cbc6d6ac7226; 2026-11-23 04:02. models/* sizes are the files in the local model cache; the backbone is downloaded on first use, not shipped in the wheel. A '.emb' file is the copy with extra encoder outputs that the first Scorer builds next to the downloaded model.

| category | files | MB |
|---|--:|--:|
| backbone (runtime-built extra-outputs copy) | 4 | 661.89 |
| backbone | 3 | 462.55 |
| coaching | 14 | 13.19 |
| lexicon | 18 | 10.46 |
| adapter | 10 | 4.57 |
| calibration | 43 | 4.31 |
| prosody | 10 | 2.20 |
| native statistics | 10 | 1.51 |
| code | 68 | 0.71 |
| length | 2 | 0.15 |
| backbone tokens | 1 | 0.00 |
| other data | 1 | 0.00 |

Package (pronunciada/, as in the wheel): 37.10 MB.

Smallest install that can score each language (code + backbone + what it opens):

| lang | code MB | backbone MB | adapter | lexicon | calibration | coaching | other data | total MB | third-party data MB |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| de | 0.71 | 70.68 | 0.32 | 5.46 | 0.63 | 2.17 | 0.69 | 80.66 | 0.00 |
| en | 0.71 | 70.68 | 0.07 | 0.97 | 3.97 | 0.01 | 0.61 | 77.02 | 0.50 |
| es | 0.71 | 70.68 | 0.72 | 1.43 | 0.63 | 1.99 | 0.60 | 76.75 | 0.00 |
| fr | 0.71 | 70.68 | 0.32 | 1.13 | 0.63 | 0.19 | 0.15 | 73.80 | 0.00 |
| it | 0.71 | 70.68 | 0.72 | 0.62 | 0.63 | 0.19 | 0.59 | 74.13 | 0.00 |
| ja | 0.71 | 70.68 | 0.32 | 0.00 | 0.63 | 1.18 | 0.29 | 73.81 | 331.20 |
| ko | 0.71 | 70.68 | 0.72 | 0.00 | 0.63 | 2.18 | 0.10 | 75.01 | 0.00 |
| pt | 0.71 | 70.68 | 0.32 | 0.18 | 0.63 | 1.99 | 0.62 | 75.14 | 0.00 |
| ru | 0.71 | 70.68 | 0.72 | 1.64 | 0.63 | 2.17 | 0.64 | 77.18 | 0.00 |
| zh | 0.71 | 70.68 | 0.32 | 0.00 | 0.63 | 2.28 | 0.96 | 75.59 | 42.88 |

Data files no language opened while scoring (20): `models/model.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.onnx`, `models/model.emb.onnx`, `models/model.fp16.onnx`, `models/model.int8.onnx`, `models/model.onnx`, `pronunciada/data/calib/prosody/de.stress.json`, `pronunciada/data/calib/prosody/en.stress.json`, `pronunciada/data/calib/prosody/es.stress.json`, `pronunciada/data/calib/prosody/it.stress.json`, `pronunciada/data/calib/prosody/pt.stress.json`, `pronunciada/data/calib/prosody/ru.stress.json`, `pronunciada/data/calib/prosody/zh.tone_zh_v2.json`, `pronunciada/data/calib/prosody/zh.tone_zh_v3.json`, `pronunciada/data/calib/sentence.npz`, `pronunciada/data/lex/LICENSE.txt`, `pronunciada/data/lex/SOURCES-eu.txt`, `pronunciada/data/prosody/stress.json`, `pronunciada/data/prosody/stress.onnx`, `pronunciada/data/prosody/tone_zh_v2.json`, `pronunciada/data/prosody/tone_zh_v2.onnx`
<!-- /opt:baseline-footprint -->

## Before and after

<!-- opt:before-after -->
Code before = cbc6d6a, after = 7834007 (opt-phase1-code; passes the full gate by identity). Timings paired and interleaved on the 20 practice clips per language (about 10 s each).

**before-fp32 -> after-fp32** (practice clips, 2 rounds, load average 3.9; 2026-11-23 04:47). Per thread count: median over the ten languages of each language's value.

| threads | load s, before -> after | cold start s | wall p50 ms | paired mean diff ms (languages faster / slower, 95 % CI excludes 0) | relative |
|--:|---|---|---|---|--:|
| 1 | 0.55 -> 0.33 | 0.99 -> 0.76 | 314.7 -> 310.8 | -2.74 (8 / 0 of 10) | -0.8 % |
| 2 | 0.53 -> 0.33 | 0.84 -> 0.62 | 201.8 -> 199.3 | -2.51 (10 / 0 of 10) | -1.1 % |
| 4 | 0.53 -> 0.33 | 0.76 -> 0.54 | 146.4 -> 143.7 | -2.55 (9 / 0 of 10) | -1.7 % |

Per stage, p50 ms, median over languages (before -> after):

| threads | load | g2p | fbank | model | adapter | align | features | delivery | calibrate | prosody | assemble | wall |
|--:|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.27 -> 1.59 | 0.54 -> 0.56 | 9.03 -> 9.21 | 249.60 -> 249.49 | 0.41 -> 0.42 | 18.46 -> 17.04 | 3.40 -> 2.33 | 12.64 -> 12.66 | 3.37 -> 3.42 | – | 0.88 -> 0.88 | 314.68 -> 310.78 |
| 2 | 1.61 -> 1.59 | 0.54 -> 0.57 | 9.03 -> 9.05 | 138.83 -> 139.49 | 0.42 -> 0.41 | 18.59 -> 17.34 | 3.37 -> 2.32 | 12.76 -> 12.62 | 3.41 -> 3.41 | – | 0.90 -> 0.90 | 201.76 -> 199.28 |
| 4 | 1.61 -> 1.60 | 0.55 -> 0.56 | 9.12 -> 9.13 | 83.55 -> 83.80 | 0.43 -> 0.43 | 19.02 -> 17.58 | 3.38 -> 2.33 | 12.58 -> 12.49 | 3.39 -> 3.37 | – | 0.89 -> 0.89 | 146.43 -> 143.69 |

**before-int8 -> after-int8** (practice clips, 2 rounds, load average 3.4; 2026-11-23 04:57). Per thread count: median over the ten languages of each language's value.

| threads | load s, before -> after | cold start s | wall p50 ms | paired mean diff ms (languages faster / slower, 95 % CI excludes 0) | relative |
|--:|---|---|---|---|--:|
| 1 | 0.59 -> 0.29 | 0.90 -> 0.60 | 217.6 -> 213.7 | -3.28 (8 / 0 of 10) | -1.4 % |
| 2 | 0.58 -> 0.27 | 0.78 -> 0.50 | 153.1 -> 149.8 | -3.35 (10 / 0 of 10) | -2.3 % |
| 4 | 0.59 -> 0.27 | 0.75 -> 0.47 | 121.5 -> 117.3 | -3.84 (9 / 0 of 10) | -3.0 % |

Per stage, p50 ms, median over languages (before -> after):

| threads | load | g2p | fbank | model | adapter | align | features | delivery | calibrate | prosody | assemble | wall |
|--:|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.65 -> 1.66 | 0.57 -> 0.57 | 9.43 -> 9.30 | 154.23 -> 153.74 | 0.42 -> 0.42 | 19.23 -> 17.66 | 3.57 -> 2.46 | 12.77 -> 12.82 | 3.48 -> 3.49 | – | 0.89 -> 0.90 | 217.62 -> 213.75 |
| 2 | 1.64 -> 1.66 | 0.57 -> 0.58 | 9.28 -> 9.30 | 89.93 -> 89.30 | 0.42 -> 0.41 | 18.98 -> 17.51 | 3.61 -> 2.49 | 12.49 -> 12.66 | 3.50 -> 3.49 | – | 0.89 -> 0.91 | 153.07 -> 149.83 |
| 4 | 1.65 -> 1.66 | 0.55 -> 0.57 | 9.34 -> 9.35 | 59.13 -> 59.19 | 0.43 -> 0.43 | 19.30 -> 17.87 | 3.53 -> 2.46 | 12.70 -> 12.57 | 3.50 -> 3.43 | – | 0.89 -> 0.90 | 121.50 -> 117.34 |

Peak RSS, fresh process, 20 clips (before = profile of record, after = this run):

| precision | lang | threads | load only MB, before -> after | after scoring MB, before -> after |
|---|---|--:|---|---|
| fp32 | en | 1 | 424 -> 395 | 694 -> 688 |
| fp32 | en | 4 | 425 -> 397 | 691 -> 697 |
| fp32 | ja | 1 | 584 -> 578 | 909 -> 926 |
| fp32 | ja | 4 | 586 -> 575 | 910 -> 925 |
| fp32 | zh | 1 | 554 -> 543 | 831 -> 846 |
| fp32 | zh | 4 | 555 -> 542 | 832 -> 847 |
| int8 | en | 1 | 246 -> 213 | 538 -> 506 |
| int8 | en | 4 | 247 -> 211 | 539 -> 506 |
| int8 | ja | 1 | 407 -> 402 | 755 -> 736 |
| int8 | ja | 4 | 409 -> 402 | 757 -> 737 |
| int8 | zh | 1 | 380 -> 362 | 680 -> 670 |
| int8 | zh | 4 | 379 -> 360 | 684 -> 668 |

Disk: smallest install per language, MB (code + backbone file actually loaded + data opened; before -> after):

| lang | fp32 | int8 |
|---|---|---|
| de | 270.1 -> 269.5 | 80.7 -> 79.7 |
| en | 266.6 -> 266.0 | 77.0 -> 76.1 |
| es | 266.3 -> 265.8 | 76.8 -> 75.8 |
| fr | 263.2 -> 262.6 | 73.8 -> 72.9 |
| it | 263.7 -> 263.1 | 74.1 -> 73.2 |
| ja | 263.4 -> 262.8 | 73.8 -> 72.9 |
| ko | 264.6 -> 264.0 | 75.0 -> 74.1 |
| pt | 264.7 -> 264.1 | 75.1 -> 74.2 |
| ru | 266.8 -> 266.2 | 77.2 -> 76.2 |
| zh | 265.2 -> 264.6 | 75.6 -> 74.6 |

Package (pronunciada/, the wheel's contents): 37.10 -> 37.12 MB.

First-ever load (a model cache holding only the downloaded files; en, 4 threads) against later loads (median of 3 later fresh processes on the same cache), and the files the first load writes:

| config | first load s | first cold start s | later load s | later cold start s | files written by the first load (MB) |
|---|--:|--:|--:|--:|---|
| before-fp32 | 0.96 | 1.21 | 0.56 | 0.81 | `model.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.onnx` 260.3 |
| after-fp32 | 1.30 | 1.54 | 0.36 | 0.60 | `model.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.ort1.32.0-arm64-b6694ac0.onnx` 259.7 |
| before-int8 | 0.65 | 0.87 | 0.56 | 0.78 | `model.int8.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.onnx` 70.7 |
| after-int8 | 0.73 | 0.95 | 0.29 | 0.50 | `model.int8.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.ort1.32.0-arm64-b6694ac0.onnx` 69.7 |
<!-- /opt:before-after -->

## Quality gate results

<!-- opt:gate -->
**Code of `opt-phase1-code` at 7834007** (11 commits on cbc6d6a: one cached backbone file with every extra
output, compact lexicon storage, faster alignment Viterbi, pre-optimized backbone graph loaded with the
optimizer off, smaller per-clip savings), fp32, default models and settings (so the pre-optimized graph
path runs), against the fp32 baseline at cbc6d6a: **PASS by identity**. All 6,376 per-clip output files of
the suite (scoring caches of eval_english, eval_languages, cv-real, OMPAL, the coaching dumps, the contrast
audit, the length and coaching-tips dumps) equal the baseline's, timing fields aside, and all eleven tables
equal the baseline's key by key; no scoreboard row moved. Suite 2,019 s (34 min), comparison 9 s.
Result file: `$PRONUNCIADA_CACHE/opt/results/code-7834007.gate.json`.
<!-- /opt:gate -->

## Numerical parity

<!-- opt:parity -->
<!-- /opt:parity -->

## Candidate log

Every candidate was built outside the repository (`$PRONUNCIADA_CACHE/opt/candidates/<name>/`) by
`bench/opt/build_candidates.py` (backbone files; tools virtualenv with onnx 1.23.2, onnxslim 0.1.99,
onnxruntime 1.32.0) or `bench/opt/build_adapters.py` (adapters). Specs: `bench/opt/candidates.json`.
Parity: the 200 practice clips at 2 threads against the default configuration of the same precision
(`python -m bench.opt.candidates parity <name> --code cbc6d6a`). Latency: paired and interleaved
against the default, en and zh, 10 clips, 3 rounds (`opt_profile_cand_fp32.json`); mean
difference, negative = faster. RSS: fresh process, en, 10 clips (`opt_rss_cand_fp32.json`).
Rebuilding the int8 model with `quantize_dynamic(MatMul, QInt8)` reproduces the shipped
`model.int8.onnx` byte for byte, so the int8 variants below differ from the shipped file only in
what is stated.

<!-- opt:candidates -->
| candidate | kind | precision | size MB (default fp32 260.3, int8 70.7; adapters 4.57 in total) | result | latency vs default, mean | peak RSS vs default |
|---|---|---|--:|---|---|---|
| ort-arena-off | a. ONNX Runtime options | fp32 | – | exactly identical | en t1 +0.2 ms [-0.8, +1.0]; en t4 -0.1 ms [-0.8, +0.4]; zh t1 -0.2 ms [-1.1, +0.9]; zh t4 +0.1 ms [-0.5, +0.6] | t1 +146 MB; t4 +147 MB |
| ort-mempattern-on | a. ONNX Runtime options | fp32 | – | exactly identical | en t1 -0.4 ms [-1.7, +1.1]; en t4 -1.9 ms [-2.8, -1.2]; zh t1 -1.6 ms [-4.7, +1.3]; zh t4 -0.1 ms [-1.4, +1.4] | t1 +3 MB; t4 +2 MB |
| ort-nospin | a. ONNX Runtime options | fp32 | – | exactly identical | en t1 +0.3 ms [-0.7, +1.2]; en t4 +11.7 ms [+9.8, +13.6]; zh t1 -2.1 ms [-4.4, -0.7]; zh t4 +14.6 ms [+12.7, +17.0] | t1 +1 MB; t4 -0 MB |
| ort-blockbase4 | a. ONNX Runtime options | fp32 | – | not identical (200/200 clips differ; raw posteriors max 5.72e-05) | – | – |
| ort-ext-level | a. ONNX Runtime options | fp32 | – | exactly identical | en t1 +1.3 ms [-1.7, +6.4]; en t4 -1.0 ms [-1.7, -0.3]; zh t1 -3.6 ms [-8.1, -1.2]; zh t4 -0.6 ms [-1.3, +0.0] | t1 +1 MB; t4 +1 MB |
| slim-fp32 | b. lossless graph | fp32 | 259.9 | exactly identical | en t1 -0.3 ms [-1.3, +0.6]; en t4 -1.0 ms [-1.9, -0.3]; zh t1 -3.3 ms [-8.0, -0.6]; zh t4 -0.5 ms [-1.2, +0.1] | t1 +17 MB; t4 +18 MB |
| slim-int8 | b. lossless graph | int8 | 69.8 | exactly identical | en t1 -1.9 ms [-3.1, -0.5]; en t4 -0.3 ms [-1.0, +0.5]; zh t1 +3.7 ms [+0.2, +9.0]; zh t4 +0.1 ms [-0.3, +0.7] | t1 -12 MB; t4 -13 MB |
| ortopt-basic-fp32 | b. lossless graph | fp32 | 259.7 | exactly identical | en t1 +0.5 ms [-0.8, +2.1]; en t4 -0.8 ms [-1.7, -0.1]; zh t1 -2.6 ms [-6.4, -0.5]; zh t4 -0.2 ms [-1.1, +0.6] | t1 +27 MB; t4 +29 MB |
| ortopt-basic-int8 | b. lossless graph | int8 | 70.1 | exactly identical | en t1 -0.4 ms [-1.7, +0.8]; en t4 -0.7 ms [-1.8, +0.3]; zh t1 +2.1 ms [-1.3, +6.8]; zh t4 +0.5 ms [+0.1, +0.9] | t1 -6 MB; t4 -8 MB |
| ortopt-ext-fp32 | b. lossless graph | fp32 | 259.7 | exactly identical | en t1 +1.4 ms [+0.5, +2.5]; en t4 -0.8 ms [-1.9, +0.1]; zh t1 -2.9 ms [-7.3, -0.6]; zh t4 +0.2 ms [-0.5, +0.8] | t1 +18 MB; t4 +19 MB |
| ortopt-ext-int8 | b. lossless graph | int8 | 69.7 | exactly identical | en t1 -0.3 ms [-1.6, +0.6]; en t4 -0.5 ms [-1.2, +0.4]; zh t1 +1.5 ms [-1.6, +6.2]; zh t4 -0.2 ms [-0.6, +0.3] | t1 -13 MB; t4 -12 MB |
| prep-fp32 | b. lossless graph | fp32 | 260.0 | exactly identical | en t1 +1.0 ms [+0.2, +1.9]; en t4 -0.8 ms [-1.7, -0.1]; zh t1 -1.5 ms [-3.5, -0.2]; zh t4 -0.3 ms [-0.9, +0.3] | t1 +21 MB; t4 +21 MB |
| f16store-fp32 | c. float16 storage | fp32 | 132.3 | not identical (200/200 clips differ; raw posteriors max 0.0486) | en t1 -1.2 ms [-3.0, +0.4]; en t4 +1.7 ms [-0.0, +4.1]; zh t1 -1.2 ms [-2.8, +0.2]; zh t4 -0.5 ms [-1.1, +0.1] | t1 +72 MB; t4 +71 MB |
| int8x-embed+out | e. int8, MatMuls kept fp32 | int8 | 72.3 | not identical (200/200 clips differ; raw posteriors max 5.42) | en t1 +3.8 ms [+2.3, +5.2]; en t4 +1.2 ms [+0.2, +2.2]; zh t1 +4.3 ms [+3.6, +5.0]; zh t4 +1.1 ms [+0.6, +1.4] | t1 +3 MB; t4 +3 MB |
| int8x-embed+enc0+out | e. int8, MatMuls kept fp32 | int8 | 78.3 | not identical (200/200 clips differ; raw posteriors max 4.81) | en t1 +18.2 ms [+13.2, +23.5]; en t4 +5.7 ms [+4.2, +7.2]; zh t1 +19.1 ms [+15.5, +23.7]; zh t4 +5.0 ms [+4.1, +5.9] | t1 +6 MB; t4 +7 MB |
| nbits4-fp32 | e. 4-bit weights | fp32 | 46.0 | not identical (200/200 clips differ; raw posteriors max 8.06) | en t1 +3.9 ms [+1.9, +5.4]; en t4 +1.2 ms [+0.2, +2.5]; zh t1 +2.8 ms [+0.8, +4.5]; zh t4 -0.1 ms [-1.8, +1.5] | t1 -202 MB; t4 -200 MB |
| adapters-fp16 | d. adapters | fp32 | 2.0 | not identical (200/200 clips differ; adapted posteriors max 0.0109) | – | – |
| adapters-int8 | d. adapters | fp32 | 0.9 | not identical (200/200 clips differ; adapted posteriors max 0.699) | – | – |
| adapters-lowrank32 | d. adapters | fp32 | 1.4 | not identical (200/200 clips differ; adapted posteriors max 5.31) | – | – |
| adapters-lowrank16 | d. adapters | fp32 | 0.7 | not identical (200/200 clips differ; adapted posteriors max 21.2) | – | – |

How each was made:

- **ort-arena-off**: {"enable_cpu_mem_arena": false, "sessions": ["backbone"]}
- **ort-mempattern-on**: {"enable_mem_pattern": true, "sessions": ["backbone"]}
- **ort-nospin**: {"config": {"session.intra_op.allow_spinning": "0"}}
- **ort-blockbase4**: {"config": {"session.dynamic_block_base": "4"}, "sessions": ["backbone"]}
- **ort-ext-level**: {"graph_optimization_level": "ORT_ENABLE_EXTENDED", "sessions": ["backbone"]}
- **slim-fp32**: onnxslim.slim(model) default passes
- **slim-int8**: onnxslim.slim(model) default passes
- **ortopt-basic-fp32**: InferenceSession(optimized_model_filepath, GraphOptimizationLevel.ORT_ENABLE_BASIC)
- **ortopt-basic-int8**: InferenceSession(optimized_model_filepath, GraphOptimizationLevel.ORT_ENABLE_BASIC)
- **ortopt-ext-fp32**: InferenceSession(optimized_model_filepath, GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
- **ortopt-ext-int8**: InferenceSession(optimized_model_filepath, GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
- **prep-fp32**: quant_pre_process(skip_symbolic_shape=True): onnx shape inference + ORT basic optimisation
- **f16store-fp32**: 627 float32 initializers (>= 256 values) stored as float16 + Cast to float32
- **int8x-embed+out**: quantize_dynamic(MatMul, QInt8), 2 MatMul nodes kept fp32 (['embed', 'out'])
- **int8x-embed+enc0+out**: quantize_dynamic(MatMul, QInt8), 48 MatMul nodes kept fp32 (['embed', 'enc0', 'out'])
- **nbits4-fp32**: MatMulNBitsQuantizer(block_size=32, is_symmetric=True), default accuracy level
- **adapters-fp16**: fp16 storage of all ten adapters, compressed npz (pronunciada/data/adapters: 4.57 MB)
- **adapters-int8**: int8 storage of all ten adapters, compressed npz (pronunciada/data/adapters: 4.57 MB)
- **adapters-lowrank32**: lowrank32 storage of all ten adapters, compressed npz (pronunciada/data/adapters: 4.57 MB)
- **adapters-lowrank16**: lowrank16 storage of all ten adapters, compressed npz (pronunciada/data/adapters: 4.57 MB)
<!-- /opt:candidates -->

## int8 against fp32

<!-- opt:int8 -->
The int8 baseline (`record-baseline --precision int8`, same code cbc6d6a, same suite and audit plans;
1,628 s) compared with the fp32 baseline in the gate's own terms (`python -m bench.opt.gate compare
--baseline <fp32 run> --candidate <int8 run>`). This is a description of the small model, not a gate on
anything. Result file: `$PRONUNCIADA_CACHE/opt/results/int8_vs_fp32.json`.

**In the gate's terms int8 would FAIL:** 59 failing row checks, 336 of 416 rows move, every language has at
least one failing row; 361 rows had a paired test, and every recomputed value equalled its table.

speechocean762 Pearson, fp32 -> int8:

| level | fp32 | int8 | change |
|---|--:|--:|--:|
| individual sounds | 0.6999 | 0.6987 | -0.0012 |
| words | 0.6395 | 0.6394 | -0.0001 |
| sentence accuracy | 0.7776 | 0.7743 | **-0.0033** |
| sentence fluency | 0.8144 | 0.8139 | -0.0005 |
| sentence prosodic | 0.8050 | 0.8047 | -0.0003 |
| sentence total | 0.7918 | 0.7894 | **-0.0024** |

Azure tally (84 rows), fp32 -> int8: significantly ahead 48 -> 47, ahead n.s. 17 -> 15, level 1 -> 1,
behind n.s. 16 -> 17, significantly behind 2 -> 4. Ten rows change class:

- zh tone swaps, right word blamed: behind n.s. -> **behind (significant)**
- fr swap benchmark, native false alarms: behind n.s. -> **behind (significant)**
- zh swap benchmark, native false alarms: ahead (significant) -> ahead n.s.
- en cv-real A: ahead (significant) -> ahead n.s.
- zh swap benchmark, far right word: ahead n.s. -> behind n.s.
- ko swap benchmark, far right word: ahead n.s. -> behind n.s.
- zh-CN cv-real A: ahead n.s. -> behind n.s.
- ko swap benchmark, far AUC: ahead n.s. -> level
- de swap benchmark, native false alarms: ahead n.s. -> ahead (significant) (better)
- ru swap benchmark, near AUC: level -> ahead n.s. (better)

Other material moves:
- Native false alarms rise in every language: on the swap benchmark by 0.2 to 0.4 points (en, zh, es, fr, it,
  ru, ko); native coaching tips by 0.4 to 4.8 points of clips (ru 4.8, ko 4.0); natives' words accepted fall
  by 0.1 to 0.26 points and sentences accepted by up to 0.5 points (de).
- Detection AUCs drop by more than 0.002 on cv-real A (en, zh-CN, zh-TW, es, de, ru), cv-real B (de),
  EpaDB (-0.0045, paired p = 0.002), and OMPAL word any error / tone-only.
- Learner errors L2-ARCTIC -0.0015 (p = 0.016); coaching precision-recall AUC -0.0064 (p = 0.002).

int8 is 3.6x smaller on disk (70.7 vs 260.3 MB), about 150 MB less memory, and 19 % (4 threads) to 35 %
(1 thread) faster on speechocean762 clips. These quality costs fit the fragility measured in the next
section.
<!-- /opt:int8 -->

## Numerical fragility of the int8 backbone

<!-- opt:fragility -->
`python -m bench.opt.fragility --eps <e>`: 20 practice clips (en, de, ja, zh, 5 each, 12,082 frames). The fbank features of each clip are perturbed by Gaussian noise (sd e) or a constant shift (+e), and the backbone's log-posteriors on the perturbed features are compared with those on the original ones. Table: `results/tables/opt_fragility_<e>.json`.

| backbone | e | perturbation | max abs change, nats | mean | 99.9th percentile | frames whose top token changes |
|---|--:|---|--:|--:|--:|--:|
| fp32 | 1e-4 | gaussian | 0.002775 | 2.55e-05 | 0.000562 | 0.00 % |
| fp32 | 1e-4 | shift | 0.0001669 | 2.98e-06 | 4.96e-05 | 0.00 % |
| int8 | 1e-4 | gaussian | 2.001 | 0.182 | 1.08 | 1.03 % |
| int8 | 1e-4 | shift | 2.83 | 0.182 | 1.12 | 1.19 % |
| int8x-embed+out | 1e-4 | gaussian | 2.657 | 0.152 | 0.914 | 1.11 % |
| int8x-embed+out | 1e-4 | shift | 2.864 | 0.153 | 0.955 | 1.17 % |
| int8x-embed+enc0+out | 1e-4 | gaussian | 3.036 | 0.152 | 0.946 | 1.05 % |
| int8x-embed+enc0+out | 1e-4 | shift | 2.551 | 0.152 | 0.906 | 1.04 % |
| f16store-fp32 | 1e-4 | gaussian | 0.002761 | 2.55e-05 | 0.000558 | 0.00 % |
| f16store-fp32 | 1e-4 | shift | 0.0001888 | 3e-06 | 4.77e-05 | 0.00 % |
| nbits4-fp32 | 1e-4 | gaussian | 0.004193 | 2.63e-05 | 0.000744 | 0.00 % |
| nbits4-fp32 | 1e-4 | shift | 0.0002005 | 3.44e-06 | 6.1e-05 | 0.00 % |
| fp32 | 3e-4 | gaussian | 0.008251 | 7.59e-05 | 0.00168 | 0.00 % |
| fp32 | 3e-4 | shift | 0.0004711 | 6.76e-06 | 0.000136 | 0.00 % |
| int8 | 3e-4 | gaussian | 2.39 | 0.178 | 0.967 | 1.15 % |
| int8 | 3e-4 | shift | 2.773 | 0.179 | 1.02 | 1.29 % |
| int8x-embed+out | 3e-4 | gaussian | 2.612 | 0.153 | 0.947 | 1.20 % |
| int8x-embed+out | 3e-4 | shift | 2.823 | 0.152 | 0.855 | 1.05 % |
| int8x-embed+enc0+out | 3e-4 | gaussian | 3.389 | 0.152 | 0.942 | 0.98 % |
| int8x-embed+enc0+out | 3e-4 | shift | 2.791 | 0.151 | 0.875 | 1.17 % |
| f16store-fp32 | 3e-4 | gaussian | 0.008198 | 7.59e-05 | 0.00168 | 0.00 % |
| f16store-fp32 | 3e-4 | shift | 0.000452 | 6.78e-06 | 0.00013 | 0.00 % |
| nbits4-fp32 | 3e-4 | gaussian | 0.01304 | 7.77e-05 | 0.00219 | 0.00 % |
| nbits4-fp32 | 3e-4 | shift | 0.0005083 | 7.23e-06 | 0.000167 | 0.00 % |

The audit's finding is reproduced: feature changes of 1e-4 move the shipped int8 backbone's log-posteriors by up to 2.0 to 2.8 nats (mean 0.18), and the top token of about 1 % of frames changes; fp32 moves by at most 0.003 nats (0.008 at 3e-4) and no top token changes. The int8 change does not grow from 1e-4 to 3e-4: it is a jump, not a slope. The likely mechanism is the dynamic quantization of the activations (DynamicQuantizeLinear before each of the 264 quantized MatMuls rounds activations to 8 bits with a scale computed per call), where a tiny input change flips roundings. Keeping the feature-embedding, first-stack and CTC-output MatMuls in fp32 (int8x-embed+out, int8x-embed+enc0+out) lowers the mean change only to 0.15 nats; the largest changes stay the same. Models whose activations stay fp32 (f16store-fp32: float16 weights; nbits4-fp32: 4-bit weights, MatMulNBits) are as stable as fp32.
<!-- /opt:fragility -->
