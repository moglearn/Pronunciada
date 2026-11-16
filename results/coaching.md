# Coaching feedback and audio-quality gate

Design and schema: [docs/COACHING.md](../docs/COACHING.md). Code: `pronunciada/coach.py`, `pronunciada/tips.py`, `pronunciada/articulatory.py`, `pronunciada/quality.py`; benchmarks `bench/coach/`. Everything was fitted or chosen on FLEURS **validation** (planted errors and synthetic degradations) and Common Voice cv-real **dev**; L2-ARCTIC test, EpaDB test and OMPAL test are reported only (nothing was tuned on them). No cv-real test clip was read.

## 1. Reliability gate for "what it sounded like" (FLEURS validation, planted substitutions)

Protocol (`bench/coach/gather.py`): 250 validation clips per language (Korean: 226), the native audio scored against references in which `bench.corrupt` replaced a sound X by Y (`phone_sub_near` and `phone_sub_far`, 2 per reference), so the true "heard" sound is X. A planted sound is *flagged* when its probability is < 0.5 (default strictness); a flagged sound gets a *diagnosis* (`heard_ipa`) when it passes the gate. The heard candidate is the language's sound of the same type (vowel / consonant) with the highest geometric-mean span posterior of its tokens (adapted model output, the sound's aligned span ± 1 frame), never the expected sound or one of its accepted alternatives.

Rule, fixed before the numbers were seen: the grid setting (log-ratio heard/expected ∈ {-1 … 3}, heard posterior ∈ {0 … 0.3}, frames ∈ {0, 1, 2}) with pooled "within one articulatory feature" precision ≥ 0.85 and the highest pooled coverage; then, per language whose own precision is below 0.85, the loosest heard-posterior bar reaching 0.85 with ≥ 20 diagnoses. "Within one feature" = `articulatory.diff(truth, heard)` has at most one entry (a vowel/consonant change never counts).

Chosen: `min_log_ratio` -1.0, `min_heard_post` 0.0, `min_frames` 0, per-language overrides {"de": {"min_heard_post": 0.25}}. The pooled target is met without any gating: once a planted sound is flagged, the best competing sound is nearly always the true one. The gate binds only in German.

| lang | planted | flagged | diagnosed | coverage (of planted) | coverage (of flagged) | exact | within 1 feature | near-miss: cov / within 1 | far: cov / within 1 | native flagged sounds with a (wrong) diagnosis |
|---|---|---|---|---|---|---|---|---|---|---|
| en | 1000 | 192 | 192 | 19.2 % | 100.0 % | 0.823 | 0.875 | 14.6 % / 0.863 | 23.8 % / 0.882 | 147 of 161 |
| zh | 1000 | 177 | 177 | 17.7 % | 100.0 % | 0.853 | 0.921 | 11.6 % / 0.897 | 23.8 % / 0.933 | 32 of 32 |
| ja | 1000 | 192 | 192 | 19.2 % | 100.0 % | 0.807 | 0.958 | 15.8 % / 0.987 | 22.6 % / 0.938 | 55 of 55 |
| es | 1000 | 491 | 491 | 49.1 % | 100.0 % | 0.945 | 0.980 | 35.6 % / 0.978 | 62.6 % / 0.981 | 68 of 76 |
| fr | 1000 | 522 | 522 | 52.2 % | 100.0 % | 0.912 | 0.954 | 46.4 % / 0.922 | 58.0 % / 0.979 | 187 of 207 |
| de | 1000 | 70 | 26 | 2.6 % | 37.1 % | 0.846 | 0.885 | 1.6 % / 0.750 | 3.6 % / 0.944 | 11 of 79 |
| it | 1000 | 164 | 164 | 16.4 % | 100.0 % | 0.951 | 1.000 | 11.2 % / 1.000 | 21.6 % / 1.000 | 6 of 6 |
| pt | 1000 | 70 | 70 | 7.0 % | 100.0 % | 0.929 | 0.986 | 10.2 % / 0.980 | 3.8 % / 1.000 | 43 of 48 |
| ru | 1000 | 349 | 349 | 34.9 % | 100.0 % | 0.785 | 0.943 | 23.6 % / 0.898 | 46.2 % / 0.965 | 80 of 91 |
| ko | 904 | 112 | 112 | 12.4 % | 100.0 % | 0.929 | 0.955 | 11.7 % / 0.981 | 13.1 % / 0.932 | 9 of 10 |
| pooled | 9904 | 2339 | 2295 | 23.2 % | 98.1 % | 0.882 | 0.953 | 18.3 % / 0.939 | 28.0 % / 0.961 | 638 of 765 |

Coverage is limited by flagging, not by the gate: only 7–52 % of the planted substitutions are flagged at all (German and Portuguese 7 %). The last column counts sounds of the *unaltered* native clips that were flagged (false alarms) and still received a heard sound, which is then necessarily wrong; per clip that is about 0.26 wrong diagnoses (2,476 clips).

Precision / coverage as the log-ratio bar rises (pooled and per language):

| lang | lr ≥ -1 | lr ≥ 0 | lr ≥ 1 | lr ≥ 2 | lr ≥ 3 |
|---|---|---|---|---|---|
| en | 19.2 % / 0.875 | 19.2 % / 0.875 | 19.0 % / 0.884 | 18.8 % / 0.888 | 18.2 % / 0.896 |
| zh | 17.7 % / 0.921 | 17.7 % / 0.921 | 17.7 % / 0.921 | 17.7 % / 0.921 | 17.7 % / 0.921 |
| ja | 19.2 % / 0.958 | 19.2 % / 0.958 | 19.2 % / 0.958 | 19.2 % / 0.958 | 18.9 % / 0.963 |
| es | 49.1 % / 0.980 | 49.1 % / 0.980 | 49.0 % / 0.980 | 49.0 % / 0.980 | 48.2 % / 0.979 |
| fr | 52.2 % / 0.954 | 52.2 % / 0.954 | 52.2 % / 0.954 | 52.0 % / 0.958 | 51.6 % / 0.961 |
| de | 2.6 % / 0.885 | 2.6 % / 0.885 | 2.6 % / 0.885 | 2.6 % / 0.885 | 2.6 % / 0.885 |
| it | 16.4 % / 1.000 | 16.4 % / 1.000 | 16.4 % / 1.000 | 16.4 % / 1.000 | 16.3 % / 1.000 |
| pt | 7.0 % / 0.986 | 7.0 % / 0.986 | 6.8 % / 0.985 | 6.8 % / 0.985 | 6.8 % / 0.985 |
| ru | 34.9 % / 0.943 | 34.9 % / 0.943 | 34.9 % / 0.943 | 34.8 % / 0.943 | 34.7 % / 0.942 |
| ko | 12.4 % / 0.955 | 12.4 % / 0.955 | 12.4 % / 0.955 | 12.4 % / 0.955 | 12.4 % / 0.955 |
| pooled | 23.2 % / 0.953 | 23.2 % / 0.953 | 23.1 % / 0.953 | 23.1 % / 0.954 | 22.8 % / 0.956 |

By sound class of the true sound (pooled, diagnosed items):

| class | n | within 1 feature |
|---|---|---|
| affricate | 22 | 0.636 |
| approximant | 174 | 0.931 |
| flap | 2 | 1.000 |
| fricative | 258 | 0.965 |
| nasal | 239 | 0.967 |
| stop | 318 | 0.978 |
| trill | 68 | 0.971 |
| vowel | 1214 | 0.948 |

Confidence (`item.confidence`): within-one-feature precision by log-ratio quartile of the diagnosed rows, made monotone: edges [6.755, 7.917, 8.998], values [0.865, 0.974, 0.976, 0.976].

**Missing sounds.** Planted extra reference consonants (`phone_ins`: the audio lacks the sound) cannot be told apart from substitutions and false alarms: the best rule over frames ≤ 1–3, nothing decoded in the span, `llr_del` ≤ −4…−15 and blank share reached 0.67 precision on 33 items. Below the 0.85 bar, so `kind: "missing"` is **off** by default (`missing_min_llr_del` = −1e9); a skipped sound is reported as a flagged substitution, usually without `heard_ipa`. Missing *words* use the shipped omission channel (`present_prob` < 0.5). `kind: "extra"` is in the schema but not produced yet.

**Tones (Mandarin).** Items need the syllable probability and the word's calibrated channel (`WordResult.tone_prob`) both < 0.5; heard = the most probable tone / stressed syllable. Bar on its posterior (planted: precision = heard equals the original):

| bar | planted flagged | diagnosed | precision | coverage of planted | native items with a diagnosis |
|---|---|---|---|---|---|
| 0.3 | 73 / 500 | 73 | 0.959 | 14.6 % | 34 of 8407 (0.4 %) |
| 0.4 | 73 / 500 | 73 | 0.959 | 14.6 % | 34 of 8407 (0.4 %) |
| 0.5 | 73 / 500 | 73 | 0.959 | 14.6 % | 33 of 8407 (0.4 %) |
| 0.6 | 73 / 500 | 69 | 0.971 | 13.8 % | 31 of 8407 (0.4 %) |
| 0.7 | 73 / 500 | 68 | 0.971 | 13.6 % | 28 of 8407 (0.3 %) |
| 0.8 | 73 / 500 | 64 | 0.969 | 12.8 % | 26 of 8407 (0.3 %) |
| 0.9 | 73 / 500 | 57 | 0.965 | 11.4 % | 22 of 8407 (0.3 %) |

By language at the chosen bar: zh 0.959 precision, 14.6 % coverage.

**Word stress (en de es it pt ru).** Items need the syllable probability and the word's calibrated channel (`WordResult.stress_prob`) both < 0.5; heard = the most probable tone / stressed syllable. Bar on its posterior (planted: precision = heard equals the original):

| bar | planted flagged | diagnosed | precision | coverage of planted | native items with a diagnosis |
|---|---|---|---|---|---|
| 0.3 | 993 / 2995 | 991 | 0.920 | 33.1 % | 180 of 18717 (1.0 %) |
| 0.4 | 993 / 2995 | 986 | 0.923 | 32.9 % | 177 of 18717 (0.9 %) |
| 0.5 | 993 / 2995 | 972 | 0.932 | 32.5 % | 168 of 18717 (0.9 %) |
| 0.6 | 993 / 2995 | 938 | 0.948 | 31.3 % | 160 of 18717 (0.9 %) |
| 0.7 | 993 / 2995 | 902 | 0.959 | 30.1 % | 154 of 18717 (0.8 %) |
| 0.8 | 993 / 2995 | 872 | 0.967 | 29.1 % | 144 of 18717 (0.8 %) |
| 0.9 | 993 / 2995 | 778 | 0.979 | 26.0 % | 121 of 18717 (0.6 %) |

By language at the chosen bar: en 0.901 precision, 24.4 % coverage; es 0.948 precision, 42.4 % coverage; de 0.895 precision, 34.2 % coverage; it 0.891 precision, 46.0 % coverage; pt 0.949 precision, 35.6 % coverage; ru 0.949 precision, 15.8 % coverage.

## 2. Real learners (report only)

`bench/coach/eval_learners.py`: the shipped gate on L2-ARCTIC test (900 utterances, 24 speakers of 6 L1s) and EpaDB test (1,263, Spanish L1), phones aligned to the annotators' as in `bench/eval_english.py`. Precision is over every diagnosis we give on a matched phone: right when it equals what the annotator heard (canonical sound when marked correct, nothing when marked deleted). English vowel labels folded ʌ→ə, ə˞→ɜ˞.

| corpus | annotated substitutions | flagged | with heard_ipa (coverage) | our diagnoses on matched phones | precision exact | within 1 feature | on substitutions: exact / within 1 | diagnoses on phones marked correct / deleted |
|---|---|---|---|---|---|---|---|---|
| l2arctic | 3168 | 2.8 % | 2.8 % | 263 | 0.183 | 0.270 | 0.565 / 0.659 | 62 / 116 |
| epadb | 3848 | 2.9 % | 2.9 % | 198 | 0.242 | 0.389 | 0.522 / 0.663 | 63 / 43 |
| l2arctic, strictness 1.0 | 3168 | 8.1 % | 7.0 % | 591 | 0.218 | 0.357 | 0.606 / 0.732 | 187 / 191 |
| epadb, strictness 1.0 | 3848 | 9.3 % | 7.8 % | 472 | 0.356 | 0.496 | 0.656 / 0.750 | 157 / 59 |

Coverage is limited by flagging: at the default strictness only ~3 % of the annotated substitutions get a sound probability below 0.5 (the scores are calibrated on speechocean762 raters, who accept most accented sounds). Overall precision is low mainly because many of our diagnoses fall on sounds the annotators marked *deleted* (we cannot say "missing" reliably) or *correct*.

Most frequent diagnosed substitutions (expected, annotator, ours, count):

- l2arctic: /ʃ/ → s vs s (6), /oʊ/ → ɔ vs ɑ (2), /p/ → t vs t (2), /oʊ/ → aʊ vs ʊ (2), /s/ → t vs t (2), /d/ → n vs n (2), /p/ → f vs f (2), /ʊ/ → ɔ vs ɑ (1), /m/ → b vs b (1), /ɑ/ → ɛ vs ɛ (1)
- epadb: /eɪ/ → aɪ vs aɪ (7), /eɪ/ → i vs i (4), /ə/ → i vs i (4), /ə/ → ɛ vs ɛ (3), /z/ → s vs s (3), /dʒ/ → ʃ vs ʃ (3), /aɪ/ → ɔ vs ɔ (3), /ɛ/ → i vs i (3), /ə/ → aɪ vs i (3), /eɪ/ → ɛ vs ə (2)

OMPAL test, learners only (7257 characters, 1220 marked wrong by the teachers): does the *type* of our item match the teachers' mark? Tone item → tone; syllable-initial consonant → consonant; any other sound → vowel (final).

| type | our items (characters) | teachers' marks | P(teacher marks this type \| our item) | P(teacher marks any error \| our item) | recall |
|---|---|---|---|---|---|
| tone | 210 | 1113 | 0.476 | 0.490 | 0.090 |
| consonant | 81 | 143 | 0.407 | 0.605 | 0.231 |
| vowel | 102 | 142 | 0.324 | 0.471 | 0.232 |
| tone, strictness 1.0 | 521 | 1113 | 0.451 | 0.463 | 0.211 |
| consonant, strictness 1.0 | 143 | 143 | 0.287 | 0.441 | 0.287 |
| vowel, strictness 1.0 | 189 | 142 | 0.243 | 0.392 | 0.324 |

Where both we and the teachers flag a character (174), our type matches one of theirs in 86.8 % (strictness 1.0: 331, 89.4 %).

Strictness 1.0 rows: the same results re-graded with `Scorer.rescore(result, 1.0)` (the app's strictness knob; reported, not chosen on these data).

## 3. Audio-quality gate

`bench/coach/audio_gate.py`: `pronunciada.quality.measure` on every FLEURS validation clip (3,474, normal audio), synthetic degradations of every 4th clip, and the cv-real dev half (test A: 1,992 accepted, 1,992 rejected by Common Voice reviewers). Each threshold is the 0.4 % tail of FLEURS validation (clipping: the 99.6 % point, at least 0.2 %), so their union stays under the 2 % target.

| measurement | issue code | threshold |
|---|---|---|
| clipped samples share | `clipped` | > 0.002 |
| SNR estimate (p95 − p10 frame level) | `too_noisy` | < 12.1 dB |
| speech level | `too_quiet` | < -69.1 dBFS |
| speech duration / prompt need | `too_short` | < 0.716 |
| speech share of the recording, or speech < 0.3 s | `mostly_silence` | < 0.312 |

Seconds of speech per prompt syllable (median, FLEURS validation): de 0.183, en 0.192, es 0.154, fr 0.171, it 0.173, ja 0.122, ko 0.167, pt 0.167, ru 0.158, zh 0.185.

| audio | n | flagged (any code) | expected code raised | codes |
|---|---|---|---|---|
| cv/accepted | 1992 | 5.9 % | – | clipped 0.2 %, mostly_silence 3.5 %, too_noisy 0.3 %, too_short 2.3 % |
| cv/rejected | 1992 | 13.4 % | – | clipped 0.5 %, mostly_silence 5.8 %, too_noisy 0.9 %, too_quiet 0.1 %, too_short 8.5 % |
| fleurs/att35 | 874 | 47.4 % | 41.8 % | mostly_silence 4.5 %, too_noisy 0.6 %, too_quiet 41.8 %, too_short 4.6 % |
| fleurs/att50 | 874 | 99.5 % | 86.6 % | mostly_silence 13.6 %, too_noisy 0.6 %, too_quiet 86.6 %, too_short 13.7 % |
| fleurs/clean | 3474 | 1.6 % | – | clipped 0.1 %, mostly_silence 0.4 %, too_noisy 0.4 %, too_quiet 0.4 %, too_short 0.4 % |
| fleurs/clip10 | 874 | 99.9 % | 99.9 % | clipped 99.9 %, mostly_silence 0.5 %, too_noisy 3.9 %, too_short 0.1 % |
| fleurs/clip3 | 874 | 97.3 % | 97.3 % | clipped 97.3 %, mostly_silence 0.7 %, too_noisy 0.9 %, too_short 0.5 % |
| fleurs/noise0 | 874 | 100.0 % | 100.0 % | clipped 0.1 %, mostly_silence 20.1 %, too_noisy 100.0 %, too_short 77.5 % |
| fleurs/noise10 | 874 | 33.3 % | 3.2 % | clipped 0.1 %, mostly_silence 5.3 %, too_noisy 3.2 %, too_quiet 0.1 %, too_short 28.5 % |
| fleurs/noise20 | 874 | 5.8 % | – | clipped 0.1 %, mostly_silence 1.1 %, too_noisy 0.6 %, too_quiet 0.3 %, too_short 3.8 % |
| fleurs/noise5 | 874 | 100.0 % | 100.0 % | clipped 0.1 %, mostly_silence 11.7 %, too_noisy 100.0 %, too_short 53.2 % |
| fleurs/pad4 | 874 | 100.0 % | 100.0 % | clipped 0.1 %, mostly_silence 100.0 %, too_noisy 4.7 %, too_short 0.2 % |
| fleurs/trunc40 | 874 | 99.3 % | 99.2 % | clipped 0.1 %, mostly_silence 4.6 %, too_noisy 0.8 %, too_quiet 0.3 %, too_short 99.2 % |

Noise lowers the speech-detection threshold's contrast, so noisy clips also raise `too_short` / `mostly_silence`; white noise at 10 dB SNR is mostly flagged through `too_short` (28.5 %), `too_noisy` itself needs ≲ 5–8 dB. `too_quiet` is very lenient (−69 dBFS) because FLEURS validation itself contains very quiet recordings (English, Japanese, French), which the 2 % budget has to accept. Clipping, truncation and mostly-silent recordings are caught almost always. On Common Voice dev, rejected clips are flagged 2.3× as often as accepted ones (13.4 % vs 5.9 %); most rejections are about the reading, not the audio, so the gate cannot catch them.

FLEURS validation flag rate by language: de 0.0 %, en 3.6 %, es 0.0 %, fr 0.3 %, it 0.3 %, ja 9.4 %, ko 1.8 %, pt 0.3 %, ru 0.0 %, zh 2.7 %.

## 4. Scores unchanged, latency

`bench/coach/identity_check.py`: 50 FLEURS validation clips (5 per language) scored by a copy of the package with the pre-coaching `scorer.py` / `types.py` (git HEAD) and by the current tree with `result.coaching()` and `tips()` called: every sound, word, syllable and sentence probability and score identical as repr() strings: **yes (0 of 50 clips differ)**. coaching() + tips(): median 0.01 ms, 90th percentile 0.04 ms per 10 s of audio (native clips flag few sounds); worst case measured, a 10 s clip scored against a different sentence (82 flagged sounds): 0.69 ms. The audio check (always on, `Scorer(audio_check=False)` disables it) median 0.50 ms per 10 s. Nothing is computed for coaching unless `coaching()` is called.
