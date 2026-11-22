# Round 19: OpenPronounce 1.1.0 against Pronunciada 1.0.2, measured

Date of the runs: 22 November 2026. Every OpenPronounce number on this page was measured by us on the
same recordings as Pronunciada, with its released code and models. Pronunciada's numbers are its
cached per-recording outputs (the ones the scoreboard reads); a re-score of 20 speechocean762 clips and
24 cv-real clips with the current code (1.0.2) gave identical outputs.

## Summary

- **OpenPronounce's headline figure reproduces.** On all 2,500 speechocean762 test recordings, sentence
  known, its overall score follows the expert total at Pearson 0.790 (everyone), 0.850 (adults), 0.670
  (children of 12 and under): the 0.79 / 0.85 / 0.67 of its README, to the third decimal. Its own
  `benchmarks/speechocean762.py --report` and its own `openpronounce-evaluate` print the same 0.790.
- **On the sentence scores of speechocean762 the two systems are level.** Pronunciada 0.762 against
  0.790 on the total, difference −0.028 [−0.094, +0.017], p = 0.44 (speakers resampled). Accuracy,
  fluency and prosody: level as well. On adults the two are within 0.01 on every score. The children's
  gap (0.558 against 0.670) is not significant (p = 0.57) and comes from one speaker: with speaker 1404
  left out the difference is +0.024 for Pronunciada; by rank (Spearman) Pronunciada is ahead on
  children, 0.590 against 0.562 (p = 0.20).
- **Pronunciada is ahead on finding bad words and sounds of learners of English** (speechocean762: word
  AUC 0.908 against 0.886, word Pearson 0.609 against 0.569, sound Pearson 0.685 against 0.534, all
  p ≤ 0.006), **on every European changed-word test** (fleurs-swap, 18 of 18 rows; Lingolix test 1,
  7 of 8 rows, the eighth level) **and on telling rejected Common Voice clips from accepted ones** in
  English, Spanish, French, German and Italian (Portuguese level).
- **OpenPronounce is ahead on the sounds annotators marked in L2-ARCTIC (AUC 0.743 against 0.712) and
  EpaDB (0.741 against 0.721)**, both significant.
- **The distilled student** (`OpenPronounce/wav2vec2-base-phones-en`) reaches 0.802 on the total
  (README: 0.802); still level with Pronunciada (p = 0.09).
- **`--fast` changes no number.** Its outputs are identical to the default mode's on all 2,500
  speechocean762 recordings and all 2,700 fleurs-swap records in six languages: the second checkpoint
  and the reference voice only feed the transcript and the pre-1.0 formula score.
- **Cost on the device**: ONDEVICE_SUMMARY

## Provenance

| | |
|---|---|
| OpenPronounce | 1.1.0, tag `v1.1.0` = commit `bd51333cebb9e7eb791bce45b710e520ca97e690` (2026-11-19), AGPL-3.0-or-later, installed from the clone (non-editable) with its `tts-piper` extra in its own venv (Python 3.12, torch 2.15.0, transformers 5.17.0, piper-tts 1.8.0) |
| its models | phone recognizer `facebook/wav2vec2-lv-60-espeak-cv-ft` @ ae45363 (1.26 GB); word recognizer and acoustic embeddings `facebook/wav2vec2-large-960h` @ bdeaacd (1.26 GB, default mode only); non-English default mode also loads `jonatasgrosman/wav2vec2-large-xlsr-53-{french,spanish,german,italian,portuguese}` for the transcript; student `OpenPronounce/wav2vec2-base-phones-en` @ 2e12ec2 (379 MB); Piper voice per language from `rhasspy/piper-voices`; scoring models are the JSON files shipped in the package (`models/*.json`) |
| downloads | Hugging Face only (the student was the one new download; the other checkpoints were already in the cache of the 0.4.0 run and were linked, not copied) |
| Pronunciada | 1.0.2, cached outputs of the shipped scorer, fp32, default strictness |
| machine | Apple M3 Pro, 12 cores, 18 GB, macOS; CPU only for every OpenPronounce run (`OPENPRONOUNCE_DEVICE=cpu`, 4 torch threads); the machine was shared with other jobs, which affects timings only |
| runner | `bench/rivals/openpronounce.py` (raw outputs, one JSON line per record, resumable), `bench/rivals/convert.py`, evaluations `bench/rivals/eval_so762.py`, `eval_shared.py`, `eval_lingolix.py`, `evaluate_compare.py`, on-device `bench/ondevice.py`; tables `results/tables/rivals_*.json`, `results/tables/ondevice/` |

### No network while scoring

OpenPronounce's default reference voice is gTTS, i.e. Google Translate, called over the network the
first time a sentence is scored. That was not allowed here. We used the project's documented offline
backend instead, Piper (`OPENPRONOUNCE_TTS=piper`, the default voice of each language). In the
worker process the `gtts` module is made unimportable and every non-local socket connection raises,
and models are read from the local cache with `HF_HUB_OFFLINE=1`; no run hit either guard. This choice
cannot move any number we compare: the reference voice feeds only the acoustic distance, which feeds
only `formula_score` (the score of versions up to 0.4); `score`, `scores` and the word reports come from
the phone recognizer and the trained models, and are identical with the voice switched off (`--fast`,
below). Their own `benchmarks/speechocean762.py` calls gTTS directly, so its inference loop was not
run; its `--report` was run on our outputs (below).

Two other settings, neither of which touches an output: `prosody=False` (the pitch and energy curves
for display; the scorer computes its own prosody features either way; the on-device timing of the
default mode keeps them on), and the recognizers' outputs are memoised per audio file so the three
prompts of a fleurs-swap recording share one forward pass.

### What its scorers were trained and tuned on

From the code, the model files (`trained_on` keys) and `benchmarks/README.md`:

- English word model (`word_mdd_en.json`) and scorer (`scorer_en.json`): speechocean762 **train** split
  (2,500 utterances, plus 742 synthetic negatives for the scorer). The word threshold (0.35) is the
  best F1 on the train split; the scorer's linear recalibration is fitted on out-of-fold train
  predictions. The test split is used to report.
- Design choices were compared on the test split, though: the README's tables pick the shipped
  configuration (hidden layer 20 of 24, 16 PCA dimensions, embeddings on, synthetic negatives on, trees
  over ridge) among alternatives whose test-split numbers are shown side by side. No weight was fitted
  on the test split, but its 0.79 is not a fully untouched held-out figure. Pronunciada's 0.762 comes
  from a scorer fitted and selected without the test split.
- Other languages: generic word model and scorer trained on the same speechocean762 train split
  without phone identity; per-language accepted-phone variants, word thresholds and a "native
  reference" shift (`native_<lang>.json`) chosen on 120 FLEURS **dev** recordings per language. Our
  fleurs-swap test uses FLEURS **test** recordings, so there is no overlap.
- Student: distilled (no labels) from the two large checkpoints on 27,065 English Common Voice 27 clips
  and the speechocean762 train audio; its own word model and scorer refit on the train split. Its
  model card says the gain over the large model probably comes from having heard the train split's
  population. Its Common Voice clips may include clips of our cv-real English and Lingolix English
  test items (their selection list is not published), so the student's rows on those two tests carry
  that caveat.

## speechocean762 test (step 2 and 3)

All 2,500 test recordings, sentence known, OpenPronounce default mode (identical to `--fast`). Age
groups by OpenPronounce's rule (`benchmarks/groups.py`): children 12 and under, teenagers 13 to 17,
adults 18 and over. Intervals: 1,000 speaker resamples; difference = Pronunciada − OpenPronounce on
the same resamples, two-sided p (0.002 is the smallest p 1,000 resamples can give). Word-level: expert
word accuracy ≤ 6 (as asked) and < 5 (OpenPronounce's own label), error signal = minus the word score;
OpenPronounce's word score = 1 − its word model's probability (a word it calls "not said" takes the
larger of that and its missing-word probability), attached to our reference words by character span.
Sound level: its phone model's per-sound probability, aligned to our reference sounds inside each word
(97.6 % of the sounds found a partner; the rest take their word's score).

SO762_DEFAULT

### The children's gap

OpenPronounce's 0.670 on children against Pronunciada's 0.558 is the largest gap in its favour, and it
is not robust: the speaker-resampled interval of the difference runs from −0.26 to +0.07 (p = 0.57),
and leaving one speaker out (1404, mean expert total 3.5 over 20 recordings) turns it into +0.024 for
Pronunciada. By rank the two are level with Pronunciada slightly ahead (Spearman 0.590 against 0.562).
Averaged per speaker, OpenPronounce follows the experts more closely on children (0.919 against 0.757):
it separates weak speakers from strong ones better, while within a speaker neither does well.

### Their own scripts

- `benchmarks/speechocean762.py --report` (their analysis code, run on a CSV of our run in their
  layout, `bench/rivals/their_csv.py`): score vs human total 0.790 (Spearman 0.727), accuracy 0.755,
  fluency 0.789, prosodic 0.778 (their report correlates the one total score with every dimension);
  826 of 15,967 words flagged, 52.5 % of them rated below 5 by the experts, 50.2 % of those found
  (README: "one word in twenty is reported, and the raters rejected 54 % of those").
- `openpronounce-evaluate` on the 2,500 recordings (score = expert total): all 0.790 [0.77, 0.81],
  adults 0.850, children 0.670, teenagers 0.475. Its interval resamples recordings, not speakers, so it
  is narrower than ours ([0.728, 0.837]); with that resampling Pronunciada's gap would look significant
  (p = 0.026 on everyone, 0.012 on children), which it is not once a speaker's recordings move together.
  Its Spearman (0.697) ranks ties in file order; scipy's tie-averaged Spearman is 0.727.

### Student and `--fast`

`--fast` (phone recognizer only): every score, every sub-score and every word probability is identical
to the default mode on all 2,500 recordings (and on all 2,700 fleurs-swap records, six languages), so
its table is the table above.

SO762_STUDENT

## Shared tests (step 4)

These were run in `--fast` mode after the identity above was established (the default-mode fleurs-swap
run gives the same 18 rows to the last digit). A record OpenPronounce could not score would get its
floor (0); none failed.

SHARED

### Lingolix benchmark (v2026.10)

Same metric code as Pronunciada's own run (`bench/third_party/lingolix.py`). Test 1: 200 Common Voice
clips per language with one word of the prompt changed; FA@80 = share of correct words flagged at the
threshold that catches 80 % of the changed words (lower is better, so a negative difference favours
Pronunciada). Test 2: 500 speechocean762 test clips (taken from the run above). Differences on 1,000
resamples of utterances (test 1, as the benchmark resamples) or speakers (test 2).

LINGOLIX

OpenPronounce's own flags (the words it reports) catch 13 % (English), 28 % (French), 24 % (Spanish)
and 28 % (German) of the changed words, with 0.4 to 2.5 % of the correct words flagged; Pronunciada's
catch 77, 87, 83 and 42 % with 1.2 to 8.1 % flagged. At equal false alarms (the AUC and FA@80 rows)
Pronunciada separates them better in every language.

### `openpronounce-evaluate` on our European manifests

Its own command (`--fast` path: acoustic, transcription and prosody off) on the cv-real test manifests,
with score = up votes − down votes of the Common Voice validators, and Pronunciada's primary cv-real
score on the same rows with the same statistic (Pearson, 1,000 recording resamples, seed 0). Test B
rows are native against non-native accepted clips, whose net votes carry no pronunciation rating; they
are printed by the command but say nothing.

EVALUATE

Dutch: OpenPronounce supports it (FLEURS-checked native thresholds and reference); Pronunciada does not,
so no shared test exists.

## On-device cost (step 5)

ONDEVICE

## Where each is ahead

PLAIN

## Files

- `bench/rivals/openpronounce.py` runner (OpenPronounce worker in its own venv, driver here), `convert.py`,
  `eval_so762.py`, `eval_shared.py`, `eval_lingolix.py`, `evaluate_manifests.py`, `evaluate_compare.py`,
  `their_csv.py`, `tables.py`, `common.py` (copied from the 2026-11-18 snapshot branch).
- `bench/ondevice.py` on-device harness.
- `results/tables/rivals_so762.json`, `rivals_shared_default.json`, `rivals_shared_fast.json`,
  `rivals_shared_student.json`, `rivals_lingolix.json`, `rivals_evaluate.json`, `results/tables/ondevice/*.json`.
- Raw OpenPronounce outputs (not in the repo): `$PRONUNCIADA_CACHE/round19/rivals/raw/<mode>/<dataset>.jsonl`.
