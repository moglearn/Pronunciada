# Size and CPU speed: measurements

Status: harness only. No baseline has been recorded yet and no optimization has been made. The
tables below are empty until the scripts in `bench/opt/` fill them.

## How this was measured

**Configurations.** A configuration is a source tree (a git revision or the working tree), a
precision (int8 is the product default; fp32 is the precision of `results/SCOREBOARD.md`), and
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

<!-- opt:machine -->
<!-- /opt:machine -->

## Baseline

### Latency

<!-- opt:baseline-latency -->
<!-- /opt:baseline-latency -->

### Memory

<!-- opt:baseline-rss -->
<!-- /opt:baseline-rss -->

### Disk footprint

<!-- opt:baseline-footprint -->
<!-- /opt:baseline-footprint -->

## Before and after

<!-- opt:before-after -->
<!-- /opt:before-after -->

## Quality gate results

<!-- opt:gate -->
<!-- /opt:gate -->

## Numerical parity

<!-- opt:parity -->
<!-- /opt:parity -->
