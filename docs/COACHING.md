# Coaching feedback and audio-quality gate

`SentenceResult.coaching()` returns the detected errors as a list of items. Each item gives the
expected phoneme, the phoneme recognized (only when it can be named reliably), the articulatory
features that differ and a confidence. `SentenceResult.audio` reports whether the recording is
usable. The measurements behind each threshold are in
[results/coaching.md](../results/coaching.md).

```python
res = scorer.score("clip.wav", "I think this is the right path")
res.audio["issues"], res.audio["retry_recommended"]       # audio-quality gate, always computed
items = res.coaching()                                     # sensitivity="balanced"
items = res.coaching(sensitivity="careful")                # fewer, surer items ("eager": more)
from pronunciada.tips import tips
for t in tips(items, max_tips=3, result=res):              # optional default English sentences
    t["tip_key"], t["text"], t["params"]
```

```bash
python -m pronunciada score clip.wav "I think this is the right path" --lang en --coach [--sensitivity careful] [--json]
```

`coaching()` reads only evidence already cached on the result: no score changes (byte-identical
on 50 clips, 5 per language, with `PYTHONHASHSEED=0`: without it the unchanged package already
differs run to run in the last digit of a few Russian `present_prob`s) and it costs about 0.25 ms per 10 s of native speech, 1.5 ms in the
worst case measured (a clip scored against the wrong sentence). Nothing is computed unless it is
called. Since round 2 it does **not** depend on the result's strictness: strictness answers "would
a teacher accept this?", coaching answers "which are this speaker's clearest real errors?".

Shipped files (`pronunciada/data/coach/`, all read at runtime; what each was fitted on:
[docs/LICENSING.md](LICENSING.md) §2):

| file | read by | content |
|---|---|---|
| `model.json` | `coach.py`, `delivery_coach.py` | coaching logistic (global + `by_lang.en`), sensitivity thresholds, tone / length / feature-claim policies, miscue and delivery thresholds |
| `gate.json` | `coach.py` | "what it sounded like" reliability gate (heard-sound posterior bars) |
| `inventory.json` | `coach.py` | per-language reference sound inventory for naming heard sounds |
| `zh_sound.npz` | `coach_zh.py` | Mandarin consonant / vowel coaching model (section 3) |
| `attr_heads.npz` | `attr_heads.py` | articulatory attribute heads (section 3d) |
| `contrast_heads.json` | `contrast_heads.py` | contrast-evidence settings for ko / zh aspiration, ru soft / hard, ja length (section 3e) |
| `ko_tense.npz` | `contrast_heads.py` | Korean lax / tense head (section 3e) |

The audio-quality thresholds live in `pronunciada/quality.py` (fitted values from
`bench/coach/audio_gate.py`). Current output carries `schema_version` 2, `schema_minor` 6
(`pronunciada.coach.SCHEMA_VERSION` / `SCHEMA_MINOR`). The example items further down were recorded
when they were introduced and show the minor version of that time (3–5); the same kinds today
carry `"schema_minor": 6` and the `features` key.

## 1. Item schema (`schema_version` 2, `schema_minor` 6)

One flat JSON object per item; every item has every key. `coaching()` returns a list;
`pronunciada.coach.to_json(items)` wraps it as `{"schema_version": 2, "items": [...]}`.
Version 2 added `error_prob` (round 1 = version 1); minor version 1 (round 3) added the key
`schema_minor` and the word-level miscue and delivery kinds of section 3b, each with its own extra keys.
Minor version 2 (round 13) added an experimental Japanese pitch-accent kind; minor version 3
(round 15) removed it again (Japanese pitch accent is not scored). Minor version 4 (round 15)
added the kind `length` (section 3c). Consumers should ignore kinds they do not know. Minor version 5 (coaching round 4) added the key `features` (section 3d) to every item. Minor version 6 (round 17) added contrast-evidence claims (`features[].source = "contrast"`) and substitution items they create (`gate.contrast`, section 3e).

| key | type | meaning |
|---|---|---|
| `schema_version` | int | 2. Renamed / removed keys bump it. |
| `schema_minor` | int | 6. Added or removed kinds / keys bump it. |
| `kind` | str | sounds: `substitution`, `missing`, `extra`; prosody: `tone`, `stress`; timing: `length` (Japanese); words: `missing_word`, `mispronounced_word`, `extra_speech`; delivery: `long_pause`, `pause_mid_phrase`, `rushed_punctuation`, `fast` (and, with `delivery="all"`, `slow`, `flat_pitch`) |
| `word`, `word_index` | str, int | the prompt word (as written) and its index in `result.words` |
| `syllable_index` | int / null | syllable inside the word (null for `missing_word`) |
| `phone_index` | int / null | sound inside the word (`result.words[i].phones[j]`); sounds only |
| `position` | str / null | sound position in the word: `initial`, `medial`, `final`, `only` |
| `expected` | str / int | sounds: IPA; `tone`: tone 1–5; `stress`: index of the syllable that should be stressed; `missing_word`: the word |
| `heard` | str / int / null | same type as `expected`; **null = "this sound was off", without naming a replacement** |
| `expected_ipa`, `heard_ipa` | str / null | for sound items, equal to `expected` / `heard`; null otherwise |
| `diff_keys` | list[str] | articulatory features that differ (section 2), `["tone"]`, `["stress_position"]`; when `heard` is null, the features the attribute heads claim (section 3d), else empty |
| `error_prob` | float / null | coaching score: P(an expert marks this as an error). Sounds: the model of section 3 (speechocean762; English and Mandarin have their own, section 3); tones: OMPAL; missing words: 1 − `present_prob`; null for stress |
| `prob` | float | the calibrated "acceptable?" probability (`PhoneResult.prob`, `SyllableResult.tone_prob` / `stress_prob`, `WordResult.present_prob`), for reference |
| `confidence` | float / null | estimated precision of `heard` (sounds: exact or within one feature, per sound class, speechocean762 train; tones / stress: exact, planted errors); null without `heard` |
| `start`, `end` | float | seconds (the word span for stress / missing words) |
| `features` | list | coaching round 4: feature-level claims of the attribute heads on a flagged sound, strongest first: `{feature, key (articulatory diff key), expected, heard (classes), p}`; empty for other kinds and when no head is confident (section 3d) |
| `gate` | dict / null | gate inputs, for logging only |

Order: by `error_prob × (0.5 + 0.5 × confidence)` (confidence 0.5 when null); `extra_speech` ranks at
0.6 and delivery items at 0.2–0.35, so they follow the clear sound errors. What produces items:

- **Sounds**: `error_prob` ≥ the sensitivity's threshold (section 3); `kind` `substitution`;
  `heard` filled only when the heard-sound gate passes.
- **Missing sounds** (`kind: "missing"`) are **off**: no rule reached 0.7 precision on the
  experts' deletions (best 0.69). A skipped sound appears as a substitution with `heard` null.
- **Extra sounds** (`kind: "extra"`) are reserved in the schema; not produced.
- **Tones** (Mandarin): syllable tone probability ≤ the sensitivity's tone threshold (OMPAL dev);
  `heard` = the most probable tone if its posterior ≥ 0.3.
- **Stress** (en de es it pt ru): unchanged from round 1 (syllable and word stress probability
  < 0.5); `heard` = most probable stressed syllable if its share ≥ 0.3
  (re-fitted for the `stress_v3` stress model).
- **Length** (Japanese, section 3c): `PhoneResult.length_prob` ≤ the sensitivity's threshold;
  `expected` / `heard` are `"long"` / `"short"`.
- **Missing words**: `present_prob < 0.5`; the word's sounds are not listed separately.
- **Word-level miscues and delivery**: section 3b; switch off with `coaching(miscues=False, delivery=False)`.

## 2. Articulatory feature table (`pronunciada/articulatory.py`)

Plain data: two dicts (base symbols, modifiers) plus `features(phone)` and `diff(a, b)`, no
dependencies, deterministic; small enough to port to C++ / Rust as a table.

- **Consonants**: `place` (bilabial, labiodental, dental, alveolar, postalveolar, retroflex,
  alveolopalatal, palatal, labial-palatal, velar, labial-velar, uvular, pharyngeal,
  epiglottal, glottal), `manner` (stop, affricate, fricative, nasal, trill, flap, approximant,
  implosive, click), `voicing`, `lateral`, and modifiers `aspirated`, `palatalised`,
  `labialised`, `velarised`, `pharyngealised`, `ejective`, `unreleased`, `syllabic`, `long`,
  `laminal_dental`, `apical`, `nasal_release`, `rhotic`, plus `next` for rare sequences.
- **Vowels**: `height` (close, near-close, close-mid, mid, open-mid, near-open, open),
  `backness` (front, central, back), `rounded`, `nasal`, `long`, `rhotic`, `devoiced`,
  `glide`.
- **Multi-symbol sounds** as the front-ends write them: stop + fricative = one affricate with the
  fricative's place (`ts`, `tʃ`, `dʒ`, `pf`, `tɕ`, `ʈʂʰ`); a doubled consonant = long (`tt`,
  `tts`); vowel sequences = diphthong (first vowel + `glide` = last vowel: `aɪ`, `ai`, `ieu`;
  Portuguese `aj`/`ew` → glide i/u); a repeated vowel = long; vowel + `ɐ` (German) or `ɻ`
  (Mandarin erhua) = rhotic; consonant + `w`/`j` = labialised / palatalised.
- `diff(expected, heard)` lists the differing keys in a fixed order; vowel vs consonant gives
  `["type"]`. "Within one feature" in the benchmarks means at most one key and never `type`.
- Coverage is tested (`tests/test_coach.py`): every ZIPA token, every sound of every language's
  reference inventory (`pronunciada/data/coach/inventory.json`, from the front-ends on FLEURS
  train prompts) and the front-end output plus alternatives on a sample sentence per language.

## 3. The coaching operating point (round 2, fitted on learners)

**Which sounds.** `error_prob` is a logistic regression (monotone in each input; coefficients in
`pronunciada/data/coach/model.json`, ten numbers) on what the scorer already computes: the
calibrated lenient and strict sound log-odds, `llr_sub`, its native z-score, the competitor
log-ratio, log duration, nucleus flag, `llr_del`, peak posterior and blank share. Target: expert
phone score < 2 on speechocean762 **train** (English, Mandarin-L1 learners); 5-fold
speaker-grouped out-of-fold predictions.

**English** (round 12b, `by_lang.en` in `model.json`) uses its own refit of that logistic. Its
sound inputs come from the English detail model `sound_E`, which adds a projection of the acoustic
model's pooled internal layers and a native-prototype margin (`results/round12_sound.md` §12b).
Every other language uses the global model (round 2). Thresholds:

| `sensitivity` | English threshold (`by_lang.en`) | English train precision / recall (OOF) | **English, speechocean762 test precision / recall** | other languages' threshold (global model) |
|---|---|---|---|---|
| `careful` | 0.628 | 0.80 / 0.27 | 0.77 / 0.24 | 0.653 |
| `balanced` (default) | 0.489 | 0.70 / 0.40 | 0.66 / 0.36 | 0.514 |
| `eager` | 0.363 | 0.60 / 0.53 | 0.56 / 0.50 | 0.390 |

(The global model's English numbers before round 12b: train 0.80 / 0.22, 0.70 / 0.35, 0.60 / 0.49;
test 0.75 / 0.20, 0.65 / 0.33, 0.55 / 0.47.)

**When to name the heard sound.** Candidate: the language's sound of the same type with the
highest geometric-mean span posterior of its tokens. It is named only when all of these hold:
* log(P heard / P expected) ≥ 3.0;
* P heard ≥ 0.3 (per language: de 0.3, en 0.4, ja 0.375, es 0.375, fr 0.4, it 0.35, ru 0.35);
* the sound class reached ≥ 0.75 exact-or-within-one-feature on the experts' pronounced phones:
  **approximant, fricative, stop, vowel**. Never for affricate, click, flap, implosive, nasal,
  trill.

On speechocean762 test at "balanced", a named sound on a real substitution is right within one
feature 0.79 of the time (round 2: 0.77; round 1: 0.50). Only 9.2 % of the experts' substitutions
get a name, and names also land on sounds the experts accepted.

**Mandarin** (OMPAL dev):
* **Tones.** Threshold on the syllable tone probability: 0.0051 / 0.0074 / 0.0131 (careful /
  balanced / eager). On OMPAL test the tone type is right 0.55 of the time at "balanced" with
  recall 0.025 (eager: 0.55 / 0.10).
* **Consonants and vowels** (round 12b, `pronunciada/coach_zh.py`, `data/coach/zh_sound.npz`).
  A per-sound logistic on the coaching score's logit, the margin to a native FLEURS-zh prototype
  and a projection of the pooled internal layers, separate for syllable-initial consonants and
  finals.
  * Thresholds: initials 0.562 / 0.483 / 0.407, finals off / 0.522 / 0.415 (careful / balanced /
    eager; OOF precision 0.8 / 0.7 / 0.6 on OMPAL dev).
  * On OMPAL test at "balanced": consonants precision 0.71, recall 0.25; vowels 0.67 / 0.18.
  * FLEURS-zh natives get 0.14 flags per clip at "balanced".

**Yardstick: Azure Pronunciation Assessment** on the same speechocean762 test phones (cached
responses, nothing re-sent), thresholding Azure's phoneme accuracy at our recall points:

| | precision at recall 0.20 / 0.33 / 0.47 | precision–recall AUC [95 % CI] |
|---|---|---|
| ours (English, round 12b) | 0.80 / 0.68 / 0.58 | 0.570 [0.510, 0.622] |
| Azure | 0.56 / 0.46 / 0.38 | 0.382 [0.324, 0.437] |

Azure's top n-best phoneme names another sound on 78 % of the experts' substitutions, but it
matches the pronounced phone only 0.37 exactly / 0.49 within one feature. Ours names 9 %, right
0.79 within one feature.

**The other eight languages have no learner labels.** They use the English thresholds through
the calibrated probabilities (comparable across languages by construction). This is
**unverified on learners**. Checked only on FLEURS validation: at "balanced" 0.12–0.29 wrong named
sounds per native 10-s clip (cap 0.3, met in every language), and 41–90 % of planted substitutions
flagged with the named sound right within one feature 0.93–0.99.

**The item list is a candidate list, not a verdict.** Its sound bar is set for precision on
learner speech (at "balanced" about one in three candidates is wrong even there), so it lists
candidates on native speech too, in every language including English: at "balanced", per native
FLEURS validation clip (~10 s, 250 clips per language), en 3.6, ja 2.2, es 3.9, fr 3.3, de 1.2,
it 2.3, pt 0.9, ru 4.5, ko 1.0 `substitution` items (about 1–4 % of the sounds); Mandarin 0.07
(its own OMPAL thresholds). Strictness decides what counts as wrong; the coaching score ranks and
explains. So `tips()` turns a pronunciation candidate into advice only when the scorer did not
accept its sound or its word (section 4, round 19): with that gate, natives get a pronunciation tip
on 9–43 % of FLEURS test clips (en 28 %, fr 32 %, ru 43 %; before the gate 42–96 % outside Mandarin, zh 10 %), 0.10–0.55
pronunciation tips per clip (`max_tips=3`, "balanced"). The cost on learners: English utterances
with an expert-marked wrong sound that get a pronunciation tip fall from 61 % to 34 %
(speechocean762 test), while the share of those tips that are right rises from 64 % to 69 %.
An app that reads the items directly should do the same: keep a `substitution` /
`mispronounced_word` item only when its `prob` or its word's `WordResult.prob` is below 0.5
(recipe in section 6).

Overrides: `res.coaching(threshold=0.6)` (sound bar), `res.coaching(tone_threshold=0.02)`.

## 3b. Word-level reading miscues and delivery flags (round 3)

All from evidence the scorer already computed (`pronunciada/delivery_coach.py`); no score
changes. Thresholds in `pronunciada/data/coach/model.json` (`extra_speech`, `delivery`), set on
FLEURS validation natives reading correctly (`bench/coach/fit_r3.py`).

- **`missing_word`**: `present_prob` < 0.5 (unchanged).
- **`mispronounced_word`**: a word with at least one flagged sound; `error_prob` = its highest
  sound `error_prob`; extra key `n_sounds_flagged`. A word-level summary of the sound items.
- **`extra_speech`**: sounds the recognizer decoded (greedy decode) that fall outside every
  expected word's aligned span, grouped into segments; reported when a segment has at least
  `min_sounds` sounds. Keys: `heard_ipa` (best-effort decoded string, no attempt to name a word),
  `start`, `end`, `after_word` / `before_word` (−1 / n at the ends), `where` (`start`, `between`,
  `end`), `restart` (the segment starts like the next word: a restarted word), `n_sounds`.
  Bar: per language the smallest `min_sounds` with ≤ 2 % of native clips flagged. Hit rate =
  native clips scored against their prompt with one inner word removed, where a segment overlaps
  the removed word. Low for short words (an extra "is" has 2 sounds): it catches inserted
  multi-sound words, not single sounds.
- **`long_pause`**: a silent stretch ≥ the language's bar (96th percentile of the longest native
  pause per clip). Keys `duration`, `after_word`, `before_word`, `at_punctuation`.
- **`pause_mid_phrase`**: a pause not at punctuation (between words without a comma / full stop
  in the prompt, or inside a word: `in_word`) ≥ the bar (96th percentile of natives' longest such
  pause, at least 0.25 s).
- **`rushed_punctuation`**: at a sentence-internal full stop (`.!?;:`) or comma, the silence
  between the two words is below the bar; only for mark types where natives pause ≥ 0.1 s at
  ≥ 60 % of marks and ≥ 30 marks were seen; bars chosen so ≤ 3 % (stops) /
  ≤ 5 % (both) of native clips are flagged (commas fit that budget only in zh, ja, it, pt, ko;
  Japanese full stops were too rare). Keys `punctuation` (`stop` / `comma`), `gap`.
- **`fast`** / **`slow`**: articulation rate (aligned sounds per second of speech, pauses
  excluded) above the 96th / below the 4th native percentile; utterances of ≥ 8 sounds.
- **`flat_pitch`**: pitch range (95th − 5th percentile of voiced F0 in semitones over the speech
  span) below the 4th native percentile; ≥ 3 words.
- `slow` and `flat_pitch` are **off by default** (`coaching(delivery="all")` reports them): on
  speechocean762 train they fire on 67 % / 51 % of the short read learner utterances and do not go
  with lower expert fluency / prosody (table below). `flat_pitch` in French and Korean computes a
  pitch track on first use (≈ 10 ms per 10 s); elsewhere it reuses the scorer's.

| lang | extra speech `min_sounds` | natives flagged | hit (removed word) | long pause (s) | mid-phrase pause (s) | rushed bar stop / comma (s) | flat pitch (st) | slow / fast (sounds/s) | native clips with any delivery flag (all kinds) |
|---|---|---|---|---|---|---|---|---|---|
| en | 5 | 0.0 % | 27 % | 0.96 | 0.90 | 0.1 / – | 6.5 | 11.2 / 16.0 | 16 % |
| zh | 7 | 2.0 % | 14 % | 1.38 | 1.14 | 0.2 / 0.02 | 7.22 | 11.7 / 15.9 | 21 % |
| ja | 7 | 1.6 % | 5 % | 1.02 | 0.98 | – / 0.1 | 5.5 | 11.9 / 16.7 | 19 % |
| es | 4 | 2.0 % | 44 % | 1.04 | 0.90 | 0.3 / – | 6.2 | 13.1 / 16.7 | 19 % |
| fr | 4 | 1.6 % | 29 % | 0.84 | 0.76 | 0.12 / – | 5.46 | 11.9 / 16.7 | 17 % |
| de | 3 | 2.0 % | 48 % | 0.94 | 0.86 | 0.1 / – | 4.56 | 10.2 / 17.7 | 20 % |
| it | 3 | 1.2 % | 54 % | 1.06 | 0.88 | 0.3 / 0.16 | 3.14 | 10.4 / 15.2 | 22 % |
| pt | 4 | 2.0 % | 41 % | 1.28 | 1.08 | 0.3 / 0.04 | 6.69 | 11.7 / 15.1 | 20 % |
| ru | 9 | 2.0 % | 10 % | 1.08 | 0.82 | 0.18 / – | 5.45 | 12.2 / 17.3 | 20 % |
| ko | 8 | 1.8 % | 12 % | 1.16 | 0.92 | 0.3 / 0.14 | 6.1 | 13.2 / 17.6 | 22 % |

Each delivery flag fires on ≤ 5 % of native clips per language (4–4.9 %); together on 16–22 %.

Sanity check, speechocean762 train (English thresholds; experts' 0–10 sentence fluency and
prosody; flagged minus not flagged):

| flag | utterances flagged | fluency | prosody |
|---|---|---|---|
| `long_pause` | 246 (9.8 %) | -2.96 | -2.72 |
| `pause_mid_phrase` | 71 (2.8 %) | -2.92 | -2.70 |
| `rushed_punctuation` | 0 (0.0 %) | – | – |
| `flat_pitch` | 1276 (51.0 %) | +0.01 | -0.07 |
| `slow` | 1679 (67.2 %) | +0.40 | +0.39 |
| `fast` | 28 (1.1 %) | -2.36 | -2.23 |
| `any` | 2184 (87.4 %) | -0.51 | -0.49 |

`rushed_punctuation` never fires there (the prompts have no internal punctuation), so it is
unverified on learners; all thresholds are unverified on learners outside English.

### One example per new kind

**`mispronounced_word`** (en, `fleurs-en-validation-10010138729160973689`):

```json
{"schema_version": 2, "schema_minor": 3, "kind": "mispronounced_word", "word": "thousands", "word_index": 5, "syllable_index": null, "phone_index": null, "position": null, "expected": "thousands", "heard": null, "expected_ipa": null, "heard_ipa": null, "diff_keys": [], "error_prob": 0.4989, "prob": 0.9669, "confidence": null, "start": 2.59, "end": 3.03, "n_sounds_flagged": 1}
```

Tip `word.mispronounced`: Try “thousands” again; some of its sounds were off.

**`extra_speech`** (en, `fleurs-en-validation-10010138729160973689`, prompt without “someone”):

```json
{"schema_version": 2, "schema_minor": 3, "kind": "extra_speech", "word": null, "word_index": null, "syllable_index": null, "phone_index": null, "position": null, "expected": null, "heard": "səmwən", "expected_ipa": null, "heard_ipa": "səmwən", "diff_keys": [], "error_prob": null, "prob": null, "confidence": null, "start": 1.73, "end": 2.15, "after_word": 2, "before_word": 3, "where": "between", "restart": false, "n_sounds": 6}
```

Tip `speech.extra`: There was extra speech between “call” and “who”; read only the words of the sentence.

**`slow`** (en, `fleurs-en-validation-10098964113747380446`):

```json
{"schema_version": 2, "schema_minor": 3, "kind": "slow", "word": null, "word_index": null, "syllable_index": null, "phone_index": null, "position": null, "expected": null, "heard": null, "expected_ipa": null, "heard_ipa": null, "diff_keys": [], "error_prob": null, "prob": null, "confidence": null, "start": 0.41, "end": 3.71, "sounds_per_s": 10.3}
```

Tip `rate.slow`: Try reading a little faster and more smoothly.

**`long_pause`** (en, `fleurs-en-validation-10312254140358477897`):

```json
{"schema_version": 2, "schema_minor": 3, "kind": "long_pause", "word": "somehow", "word_index": 7, "syllable_index": null, "phone_index": null, "position": null, "expected": null, "heard": null, "expected_ipa": null, "heard_ipa": null, "diff_keys": [], "error_prob": null, "prob": null, "confidence": null, "start": 3.51, "end": 4.51, "duration": 1.0, "after_word": 7, "before_word": 8, "at_punctuation": false}
```

Tip `pause.long`: Try not to stop for so long (1.0 s) after “somehow”.

**`fast`** (en, `fleurs-en-validation-10408474834696432538`):

```json
{"schema_version": 2, "schema_minor": 3, "kind": "fast", "word": null, "word_index": null, "syllable_index": null, "phone_index": null, "position": null, "expected": null, "heard": null, "expected_ipa": null, "heard_ipa": null, "diff_keys": [], "error_prob": null, "prob": null, "confidence": null, "start": 0.99, "end": 6.95, "sounds_per_s": 16.61}
```

Tip `rate.fast`: Slow down a little so that every sound is clear.

**`rushed_punctuation`** (en, `fleurs-en-validation-10630374883688952860`):

```json
{"schema_version": 2, "schema_minor": 3, "kind": "rushed_punctuation", "word": "injured", "word_index": 18, "syllable_index": null, "phone_index": null, "position": null, "expected": null, "heard": null, "expected_ipa": null, "heard_ipa": null, "diff_keys": [], "error_prob": null, "prob": null, "confidence": null, "start": 5.71, "end": 5.71, "punctuation": "stop", "gap": 0.02, "after_word": 18}
```

Tip `punctuation.rushed`: Make a short pause at the full stop after “injured”.

**`flat_pitch`** (en, `fleurs-en-validation-1105184381995491301`):

```json
{"schema_version": 2, "schema_minor": 3, "kind": "flat_pitch", "word": null, "word_index": null, "syllable_index": null, "phone_index": null, "position": null, "expected": null, "heard": null, "expected_ipa": null, "heard_ipa": null, "diff_keys": [], "error_prob": null, "prob": null, "confidence": null, "start": 0.81, "end": 4.35, "pitch_range_st": 6.45}
```

Tip `pitch.flat`: Let your voice rise and fall more; it sounded quite flat.

**`pause_mid_phrase`** (en, `fleurs-en-validation-14225189458852066857`):

```json
{"schema_version": 2, "schema_minor": 3, "kind": "pause_mid_phrase", "word": "mahayana", "word_index": 15, "syllable_index": null, "phone_index": null, "position": null, "expected": null, "heard": null, "expected_ipa": null, "heard_ipa": null, "diff_keys": [], "error_prob": null, "prob": null, "confidence": null, "start": 5.73, "end": 6.65, "duration": 0.92, "after_word": 15, "before_word": 16, "in_word": false}
```

Tip `pause.mid_phrase`: Keep “mahayana” and “path” together; there was a 0.9 s pause between them.

## 3c. Length (round 15, Japanese)

`pronunciada/length.py` judges the length of Japanese vowels (long `aː` vs short `a`, including ー
and おう) and of consonants between two vowels (っ `tt` vs single `t`) from timing. The inputs are
the alignment's onset-to-onset intervals relative to the utterance, pauses, closure silence and
the acoustic model's length token. Per sound it reports `PhoneResult.length_expected`
(`"long"` / `"short"`) and `length_prob` (P(held for the right time)); `WordResult.length_prob`
is the factor multiplied into the word probability.

Item `length`:
* `expected` / `heard` are `"long"` / `"short"`; the heard value is always the other one.
* `expected_ipa` is the sound; `heard_ipa` is null.
* `diff_keys` is `["length"]`.
* `error_prob` = 1 − `length_prob`; `prob` = `length_prob`.
* `confidence` = the estimated precision of the operating point.
* `syllable_index` is null; `phone_index` is set.

The operating point per sensitivity is the loosest native quantile whose estimated precision reaches
0.8 / 0.7 / 0.6. Precision is estimated for a 5 % share of wrong-length sounds; planted swaps on
FLEURS validation stand in for learner errors, because no learner data with length labels exists.

| sensitivity | `length_prob` ≤ | est. precision | planted swaps caught | native judged sounds flagged |
|---|---|---|---|---|
| careful | 0.014 | 0.82 | 45 % | 0.5 % |
| balanced | 0.069 | 0.71 | 70 % | 1.5 % |
| eager | 0.113 | 0.66 | 75 % | 2.0 % |

The following are not judged: particles and other one-mora words, a short vowel next to another
vowel of its word (えい, いう), a single voiceless consonant before i / u (devoicing), and a vowel
before a pause. Italian double consonants and German vowel length were tried and are not shipped
([results/round15_length.md](../results/round15_length.md)).

```json
{"schema_version": 2, "schema_minor": 4, "kind": "length", "word": "その後", "word_index": 0, "syllable_index": null, "phone_index": 1, "position": "medial", "expected": "long", "heard": "short", "expected_ipa": "oː", "heard_ipa": null, "diff_keys": ["length"], "error_prob": 0.9893, "prob": 0.0107, "confidence": 0.71, "start": 1.97, "end": 2.01, "gate": {"length_prob": 0.0107, "threshold": 0.0692}}
```
→ tip `length.vowel_long`: "Hold the vowel /o/ in “その後” for two beats; it sounded short."

Round 17: the length channel also reads the attribute head `long` (a logistic on the timing
model's log-odds and the head's evidence, mapped back onto the timing model's native distribution
by quantiles, so its native thresholds keep their rates; `data/length/ja.json` "head"). FLEURS
test: "short for long" vowels caught 85 % → 96 %, "single for geminate" 79 % → 100 %.

## 3d. Feature-level claims (coaching round 4)

When a sound is flagged but no replacement sound is named, the item can still say *which feature*
was wrong ("no voice", "lips not rounded", "tongue too far forward"). Small linear heads
(`pronunciada/attr_heads.py`, 167 KB) read the acoustic model's pooled internal activations of
the sound and predict, for the sound actually produced: voicing, aspiration, manner (stop,
affricate, fricative, nasal, approximant, tap/trill), place (lips, tongue tip, behind the gum
ridge, back of the tongue, throat), palatalisation, laterality; vowel height (close / mid /
open), backness, rounding, nasality and length. Trained on native FLEURS train speech of all ten
languages; on planted swaps they report the audio's sound, not the reference's (68–95 % honest,
section "Leak check" of results/coaching.md).

A claim `{feature, key, expected, heard, p}` is made when the head is confident the expected
class is wrong (1 − P ≥ q), the coaching score is high enough (≥ t), the language's best
competing sound has the claimed class too, no accepted alternative changes that feature, and a
named `heard` sound (if any) agrees. q and t per head and sensitivity were chosen on
speechocean762 train for claim precision ≥ 0.8 / 0.75 / 0.75 and at most 0.15 / 0.3 / – wrong
claims per native FLEURS clip ("careful" / "balanced" / "eager"; `model.json` "attr"). Heads
used for claims: voicing, place, laterality, backness, rounding (and manner at "eager").
Aspiration, palatalisation, nasality and length are trained but not used for claims yet (no
learner labels to gate them; as detectors they hear Korean / Mandarin aspiration, Russian soft /
hard and Japanese length well, results/coaching.md); vowel height is off (too imprecise on
learners).

With no named sound, `diff_keys` lists the claimed features, `confidence` is the claim policy's
train precision, and the default tip is a feature tip (`feature.*` keys, section 4).

speechocean762 test (568 real substitutions): at "balanced" 13.6 % get a claim, right
0.84 of the time; with the named sounds, 15.7 % of substitutions get a right
feature (named sound alone: 7.6 %). "eager": 40.5 % get a claim at 0.72 precision, but
natives then get 0.2–2.5 wrong claims per clip. English only is verified on learners.

Example (speechocean762 test, balanced; experts: /l/ in “COLLEGE” pronounced R):

```json
{"schema_version": 2, "schema_minor": 5, "kind": "substitution", "word": "COLLEGE", "word_index": 1, "syllable_index": 1, "phone_index": 2, "position": "medial", "expected": "l", "heard": null, "expected_ipa": "l", "heard_ipa": null, "diff_keys": ["lateral"], "error_prob": 0.8257, "prob": 0.3719, "confidence": 0.868, "start": 1.17, "end": 1.19, "features": [{"feature": "lateral", "key": "lateral", "expected": "yes", "heard": "no", "p": 0.9341}]}
```

Tip `feature.lateral.add`: Let the air pass along the sides of the tongue for /l/ in “COLLEGE”.

Example 2 (experts: /u/ in “STUDENT” pronounced AH): tip `feature.rounded.add`: Round your lips for /u/ in “STUDENT”; they sounded spread.

## 3e. Contrast evidence: Korean aspiration, Russian soft / hard (round 17)

For contrasts the scorer was blind to, the attribute heads' probability of the expected value
(aspiration: head `aspirated`; Russian soft / hard: head `palatalised`) is turned into a
calibrated "realized as expected" probability `sigmoid(z)`, `z = 0.5 (logit P − s0)` with `s0`
the 1 % quantile of natives (`pronunciada/contrast_heads.py`, `data/coach/contrast_heads.json`,
FLEURS validation; `PRONUNCIADA_HEAD_CAPS=0` switches it off). It caps the shown sound
probability (ko, ru, zh) and, in Korean and Russian, feeds coaching: a member sound with
`sigmoid(z)` below the sensitivity's bar gets the claim `{feature: "aspirated" | "palatalised",
expected, heard, p, source: "contrast"}` (feature tips `feature.aspirated.*` /
`feature.palatalised.*`), and a new substitution item when it was not flagged (`gate.contrast`,
`gate.evidence`). Bars: at most 0.05 / 0.1 / 0.2 such items per native FLEURS clip
(careful / balanced / eager). On the audit's planted swaps (FLEURS validation) "balanced" catches
Korean "plain for aspirated" 55 %, "aspirated for plain" 21 %; Russian "hard for soft" 54 %,
"soft for hard" 1 %. Mandarin aspiration items stay **off**: on OMPAL dev (French learners) no
bar reaches 0.7 precision for the teachers' consonant marks (best 0.67 on 12 characters), and
adding the evidence to the Mandarin coaching score lowers its consonant AUC (0.956 → 0.935).
These coaching items are unverified on learners (no labeled Korean or Russian learner data).

Round 18 adds **Korean lax / tense** (달 / 딸): the Korean reference marks each ㄱ/ㄲ ㄷ/ㄸ ㅂ/ㅃ ㅅ/ㅆ
ㅈ/ㅉ obstruent `RefPhone.tense` (spelling plus the front-end's post-obstruent tensification; the
symbol is the same plain one), a small head (`data/coach/ko_tense.npz`) gives the evidence, and the
claim `feature: "tense"` uses the tips `feature.tense.add` ("Make /k/ in “…” tighter and stronger,
with no puff of air.") and `feature.tense.remove` ("Relax /k/ in “…”; it sounded tense."). Since
round 18 the contrast evidence also lowers the **word** probability in Mandarin and Japanese
(`WordResult.contrast_prob`, results/round18_word_contrasts.md).

## 4. Default tips (`pronunciada/tips.py`)

**Acceptance gate (round 19).** With `result` given, a tip from a pronunciation candidate
(`substitution`, including feature and contrast claims, `missing`, `mispronounced_word`) is made
only when the scorer itself did not accept the sound or the word: the item's `prob` (else the
sound's `PhoneResult.prob`) < 0.5 or the word's `WordResult.prob` < 0.5, at the result's current
strictness, so `scorer.rescore(result, strictness=...)` moves the gate with it. A
`mispronounced_word` passes when its word, or one of the sounds flagged in it, is not accepted.
`max_tips` is then filled from the items that pass. `missing_word`, `extra_speech`, delivery,
`tone`, `stress` and `length` items are never gated (they have their own calibrated bars).
`include_accepted=True` turns the gate off; without `result` there is nothing to gate on and
the behavior is the old one. No free parameter: 0.5 is the scorer's own acceptance line. The
items themselves (`res.coaching()`) are unchanged. Measured cost: section 3 and
[results/coaching.md](../results/coaching.md) "Round 19".

Separate from the data. `tips(items, max_tips=3, result=None, include_accepted=False)` returns at most `max_tips`
`{"tip_key", "text", "params", "item"}`: diagnosed items first in coaching order, then
flag-only items; at most one tip per sound and per word, so a sound never gets two tips (the
single most salient difference is chosen in a fixed priority: consonants manner → place →
voicing → aspiration → lateral → palatalisation → length; vowels glide (if it is the only
difference) → height → backness → rounding → nasality → length). `params` carries `expected`,
`heard`, `word` and the template's extra fields (`expected_place`, `heard_place`, `glide`,
`expected_tone`, `heard_tone`, `expected_ord`, `heard_ord`, `syllable`) so an app can render its
own or localized copy from `tip_key`.

| tip_key | default text |
|---|---|
| `voicing.add` | Let your voice buzz on /{expected}/ in “{word}”; it sounded like /{heard}/, without voice. |
| `voicing.remove` | Say /{expected}/ in “{word}” without voice (no buzz); it sounded like /{heard}/. |
| `aspiration.add` | Add a small puff of air after /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `aspiration.remove` | Say /{expected}/ in “{word}” without a puff of air; it sounded like /{heard}/. |
| `place` | Make /{expected}/ in “{word}” with {expected_place}; it sounded like /{heard}/, made with {heard_place}. |
| `manner.continuant` | Keep the air flowing through /{expected}/ in “{word}” instead of stopping it; it sounded like /{heard}/. |
| `manner.stop` | Close completely and release for /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `manner.affricate` | Start /{expected}/ in “{word}” with a short closure, then let the air hiss; it sounded like /{heard}/. |
| `manner.nasal` | Let the air go through your nose for /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `manner.oral` | Keep the air out of your nose for /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `manner.flap` | Make /{expected}/ in “{word}” a single quick tap of the tongue tip; it sounded like /{heard}/. |
| `manner.trill` | Let the tongue tip vibrate (roll) for /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `manner.other` | Say /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `lateral.add` | Let the air pass along the sides of the tongue for /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `lateral.remove` | Don't let the air flow around the sides of the tongue for /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `palatalisation.add` | Raise the middle of your tongue (a slight “y” color) on /{expected}/ in “{word}”; it sounded like plain /{heard}/. |
| `palatalisation.remove` | Say /{expected}/ in “{word}” without the “y” color; it sounded like /{heard}/. |
| `length.longer` | Hold /{expected}/ in “{word}” a little longer; it sounded short. |
| `length.shorter` | Make /{expected}/ in “{word}” shorter; it sounded long. |
| `vowel.open` | Open your mouth a little more for /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `vowel.close` | Raise your tongue a little (mouth less open) for /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `vowel.front` | Move your tongue forward for /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `vowel.back` | Pull your tongue back for /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `vowel.round` | Round your lips for /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `vowel.unround` | Spread your lips (no rounding) for /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `vowel.glide.add` | Glide towards /{glide}/ at the end of /{expected}/ in “{word}”; it sounded like a single /{heard}/. |
| `vowel.glide.remove` | Keep /{expected}/ in “{word}” steady without gliding; it sounded like /{heard}/. |
| `vowel.glide.change` | Glide towards /{glide}/ in /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `nasal.add` | Let some air through your nose on /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `nasal.remove` | Keep /{expected}/ in “{word}” oral (no air through the nose); it sounded like /{heard}/. |
| `rhotic.add` | Add the r-color to /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `rhotic.remove` | Say /{expected}/ in “{word}” without r-color; it sounded like /{heard}/. |
| `sound.substitution` | Say /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `sound.unclear` | Listen to /{expected}/ in “{word}” and try it again; it did not sound quite right. |
| `sound.missing` | Don't skip /{expected}/ in “{word}”. |
| `sound.extra` | There seems to be an extra sound in “{word}”; try it without it. |
| `tone.wrong` | “{syllable}” should be {expected_tone} (tone {expected}); it sounded {heard_tone} (tone {heard}). |
| `tone.unclear` | Check the tone of “{syllable}”: it should be {expected_tone} (tone {expected}). |
| `stress.position` | In “{word}”, stress the {expected_ord} syllable; it sounded stressed on the {heard_ord}. |
| `stress.unclear` | In “{word}”, put the stress on the {expected_ord} syllable. |
| `word.mispronounced` | Try “{word}” again; some of its sounds were off. |
| `speech.extra` | There was extra speech {where}; read only the words of the sentence. |
| `speech.restart` | You started “{next_word}” twice; try to say it in one go. |
| `pause.long` | Try not to stop for so long ({duration} s) after “{word}”. |
| `pause.long_punctuation` | Keep the pause after “{word}” shorter ({duration} s). |
| `pause.mid_phrase` | Keep “{word}” and “{next_word}” together; there was a {duration} s pause between them. |
| `pause.in_word` | Say “{word}” in one go, without stopping in the middle. |
| `punctuation.rushed` | Make a short pause at the {mark} after “{word}”. |
| `pitch.flat` | Let your voice rise and fall more; it sounded quite flat. |
| `rate.slow` | Try reading a little faster and more smoothly. |
| `rate.fast` | Slow down a little so that every sound is clear. |
| `word.missing` | The word “{word}” seemed to be missing; try saying every word. |
| `length.vowel_long` | Hold the vowel /{sound}/ in “{word}” for two beats; it sounded short. |
| `length.vowel_short` | Keep the vowel /{sound}/ in “{word}” short (one beat); it sounded long. |
| `length.consonant_long` | Pause briefly before /{sound}/ in “{word}” (a double consonant); it sounded single. |
| `length.consonant_short` | No pause before /{sound}/ in “{word}”; it sounded doubled. |
| `feature.voicing.add` | Let your voice buzz on /{expected}/ in “{word}”; it sounded voiceless. |
| `feature.voicing.remove` | Say /{expected}/ in “{word}” without voice (no buzz); it sounded voiced. |
| `feature.aspirated.add` | Add a small puff of air after /{expected}/ in “{word}”; it sounded unaspirated. |
| `feature.aspirated.remove` | Say /{expected}/ in “{word}” without a puff of air; it sounded aspirated. |
| `feature.manner.stop` | Close completely and release for /{expected}/ in “{word}”; the air kept flowing. |
| `feature.manner.continuant` | Keep the air flowing through /{expected}/ in “{word}”; it sounded stopped. |
| `feature.manner.affricate` | Start /{expected}/ in “{word}” with a short closure, then let the air hiss. |
| `feature.manner.nasal` | Let the air go through your nose for /{expected}/ in “{word}”; it did not sound nasal. |
| `feature.manner.oral` | Keep the air out of your nose for /{expected}/ in “{word}”; it sounded nasal. |
| `feature.manner.rhotic` | Make /{expected}/ in “{word}” a quick tap or roll of the tongue tip. |
| `feature.manner.other` | Check how you make /{expected}/ in “{word}”: the airflow sounded different. |
| `feature.place` | Make /{expected}/ in “{word}” with {expected_place}; it sounded {heard_place}. |
| `feature.palatalised.add` | Raise the middle of your tongue (a slight “y” color) on /{expected}/ in “{word}”. |
| `feature.palatalised.remove` | Say /{expected}/ in “{word}” without the “y” color. |
| `feature.lateral.add` | Let the air pass along the sides of the tongue for /{expected}/ in “{word}”. |
| `feature.lateral.remove` | Don't let the air flow around the sides of the tongue for /{expected}/ in “{word}”; it sounded like an l. |
| `feature.height.close` | Raise your tongue a little (mouth less open) for /{expected}/ in “{word}”; it sounded too open. |
| `feature.height.open` | Open your mouth a little more for /{expected}/ in “{word}”; it sounded too closed. |
| `feature.backness.front` | Move your tongue forward for /{expected}/ in “{word}”; it sounded too far back. |
| `feature.backness.back` | Pull your tongue back for /{expected}/ in “{word}”; it sounded too far forward. |
| `feature.rounded.add` | Round your lips for /{expected}/ in “{word}”; they sounded spread. |
| `feature.rounded.remove` | Spread your lips (no rounding) for /{expected}/ in “{word}”; they sounded rounded. |
| `feature.nasal.add` | Let some air through your nose on /{expected}/ in “{word}”; it sounded oral. |
| `feature.nasal.remove` | Keep /{expected}/ in “{word}” oral (no air through the nose); it sounded nasal. |
| `feature.long.add` | Hold /{expected}/ in “{word}” a little longer; it sounded short. |
| `feature.long.remove` | Make /{expected}/ in “{word}” shorter; it sounded long. |
| `feature.tense.add` | Make /{expected}/ in “{word}” tighter and stronger, with no puff of air. |
| `feature.tense.remove` | Relax /{expected}/ in “{word}”; it sounded tense. |

## 5. Audio-quality gate (`pronunciada/quality.py`, `result.audio`)

Measured from the waveform before scoring (about 0.5 ms per 10 s; `Scorer(audio_check=False)`
turns it off). Scores are returned either way; the app decides whether to ask for a retry.

`result.audio` = `{schema_version, duration_s, clipped_share, peak_dbfs, noise_db, speech_db,
snr_db, speech_s, speech_share, lead_sil_s, trail_sil_s, expected_s, speech_ratio, issues,
retry_recommended}` (20 ms frames; `expected_s` = prompt syllables × the language's median
speech seconds per syllable on FLEURS validation; Japanese counts morae).

| issue code | rule | threshold |
|---|---|---|
| `clipped` | share of samples at the rail (≥ 0.99) or in flat runs at the clip's own peak | > 0.002 |
| `too_noisy` | SNR estimate = 95th − 10th percentile frame level | < 12.1 dB |
| `too_quiet` | mean level of the speech frames | < -69.1 dBFS |
| `too_short` | speech seconds / what the prompt needs | < 0.716 |
| `mostly_silence` | speech share of the recording, or less than 0.3 s of speech | < 0.312 |

Set from data (`bench/coach/audio_gate.py`): each threshold is the 0.4 % tail of FLEURS
validation (normal audio), so together they flag 1.6 % of it (target < 2 %; Japanese 9.4 %,
English 3.6 %: those sets contain noisy and very quiet recordings). Synthetic degradations of
FLEURS validation: clipping caught 97–100 %, truncation to 40 % 99 %, a clip inside 4× silence
100 %, white noise at 0 / 5 dB SNR 100 %, at 10 dB 33 %, −50 dB attenuation 99.5 %, −35 dB 47 %.
Common Voice dev: 5.9 % of accepted and 13.4 % of rejected clips are flagged. Silence trimming
was not added: the gate reports `lead_sil_s` / `trail_sil_s` instead.


## 6. How an app should use this

1. Check `result.audio["retry_recommended"]` first. `clipped`, `too_quiet`, `mostly_silence`,
   `too_short`: ask for a new recording before showing pronunciation feedback; `too_noisy`:
   suggest a quieter place. Low scores on such audio are more often the recording than the speaker.
2. Take `result.coaching(sensitivity=...)`. "balanced" means: about two in three flagged sounds
   are real errors to an expert (speechocean762 test 0.65); "careful" three in four; "eager"
   finds 2.4× more of the errors at 0.55.
3. Show `heard` only when it is not null ("it sounded like /s/"); otherwise say "this sound was
   off, listen and try again". Do not invent a replacement sound.
4. Use `tip_key` + `params` for localized tips, or `pronunciada.tips.tips(items, 3, result)` for
   English; at most 3 per utterance. If you consume the items directly, apply the same gate as
   `tips()` so the advice never contradicts the scores:

   ```python
   def not_accepted(it, res):
       if it["kind"] not in ("substitution", "missing", "mispronounced_word"):
           return True                                   # other kinds have their own bars
       w = res.words[it["word_index"]]
       return w.prob < 0.5 or (it["prob"] is not None and it["prob"] < 0.5)
   items = [it for it in res.coaching() if not_accepted(it, res)]
   ```

   (`tips()` additionally lets a `mispronounced_word` through when one of the sounds flagged in
   its word is below 0.5.)
5. Outside English and Mandarin tones, treat coaching as unverified on learners (section 3).
   Word-level `extra_speech` / `missing_word` can drive "read only the words shown" prompts;
   `long_pause`, `pause_mid_phrase` and `fast` go with lower expert fluency on English learners,
   the others are native-derived only (section 3b).
6. Log `schema_version`, `error_prob` and `gate` with the app's events to monitor drift.

## 7. Example outputs

### English

FLEURS validation `fleurs-en-validation-10010138729160973689`, a native recording of “When you call someone who is thousands of miles away, you are using a satellite.”, scored against the prompt with planted errors: phone_sub_far in “a” (ɪ → ɔ); stress in “away” (syllable 1 → 0). So the true sounds are the *original* ones. Output of `res.coaching()` (balanced) and `tips()`:

```json
{
 "audio": {
  "schema_version": 1,
  "duration_s": 6.54,
  "clipped_share": 0.0,
  "peak_dbfs": -32.5328,
  "noise_db": -68.9703,
  "speech_db": -50.9045,
  "snr_db": 25.7277,
  "speech_s": 3.94,
  "speech_share": 0.6024,
  "lead_sil_s": 1.14,
  "trail_sil_s": 1.08,
  "expected_s": 4.032,
  "speech_ratio": 0.9772,
  "issues": [],
  "retry_recommended": false
 },
 "coaching": {
  "schema_version": 2,
  "schema_minor": 3,
  "items": [
   {
    "schema_version": 2,
    "schema_minor": 3,
    "kind": "substitution",
    "word": "a",
    "word_index": 13,
    "syllable_index": 0,
    "phone_index": 0,
    "position": "only",
    "expected": "ɔ",
    "heard": "ɪ",
    "expected_ipa": "ɔ",
    "heard_ipa": "ɪ",
    "diff_keys": [
     "height",
     "backness",
     "rounded"
    ],
    "error_prob": 0.9406,
    "prob": 0.3863,
    "confidence": 0.755,
    "start": 4.87,
    "end": 4.89
   },
   {
    "schema_version": 2,
    "schema_minor": 3,
    "kind": "mispronounced_word",
    "word": "a",
    "word_index": 13,
    "syllable_index": null,
    "phone_index": null,
    "position": null,
    "expected": "a",
    "heard": null,
    "expected_ipa": null,
    "heard_ipa": null,
    "diff_keys": [],
    "error_prob": 0.9406,
    "prob": 0.2991,
    "confidence": null,
    "start": 4.87,
    "end": 4.89,
    "n_sounds_flagged": 1
   }
  ]
 }
}
```

Default tips:

- `vowel.open`: Open your mouth a little more for /ɔ/ in “a”; it sounded like /ɪ/.

### Mandarin

FLEURS validation `fleurs-zh-validation-1366745426988339540`, a native recording of “太空中的卫星接到电话后，几乎是立刻将其反射回去。”, scored against the prompt with planted errors: phone_sub_far in “其” (i → ɑʊ); tone in “后” (4 → 1). So the true sounds are the *original* ones. Output of `res.coaching()` (balanced) and `tips()`:

```json
{
 "audio": {
  "schema_version": 1,
  "duration_s": 4.98,
  "clipped_share": 0.0,
  "peak_dbfs": -6.1392,
  "noise_db": -64.9964,
  "speech_db": -34.748,
  "snr_db": 41.7261,
  "speech_s": 3.62,
  "speech_share": 0.7269,
  "lead_sil_s": 0.6,
  "trail_sil_s": 0.76,
  "expected_s": 4.07,
  "speech_ratio": 0.8894,
  "issues": [],
  "retry_recommended": false
 },
 "coaching": {
  "schema_version": 2,
  "schema_minor": 3,
  "items": [
   {
    "schema_version": 2,
    "schema_minor": 3,
    "kind": "missing_word",
    "word": "的",
    "word_index": 2,
    "syllable_index": null,
    "phone_index": null,
    "position": null,
    "expected": "的",
    "heard": null,
    "expected_ipa": null,
    "heard_ipa": null,
    "diff_keys": [],
    "error_prob": 0.9241,
    "prob": 0.0759,
    "confidence": null,
    "start": 0.97,
    "end": 1.03
   },
   {
    "schema_version": 2,
    "schema_minor": 3,
    "kind": "tone",
    "word": "后",
    "word_index": 6,
    "syllable_index": 0,
    "phone_index": null,
    "position": null,
    "expected": 1,
    "heard": 4,
    "expected_ipa": null,
    "heard_ipa": null,
    "diff_keys": [
     "tone"
    ],
    "error_prob": 0.6338,
    "prob": 0.0069,
    "confidence": 0.959,
    "start": 1.99,
    "end": 2.15
   },
   {
    "schema_version": 2,
    "schema_minor": 3,
    "kind": "missing_word",
    "word": "中",
    "word_index": 1,
    "syllable_index": null,
    "phone_index": null,
    "position": null,
    "expected": "中",
    "heard": null,
    "expected_ipa": null,
    "heard_ipa": null,
    "diff_keys": [],
    "error_prob": 0.7706,
    "prob": 0.2294,
    "confidence": null,
    "start": 0.91,
    "end": 0.97
   }
  ]
 }
}
```

Default tips:

- `word.missing`: The word “的” seemed to be missing; try saying every word.
- `tone.wrong`: “后” should be high and level (tone 1); it sounded falling (tone 4).
- `word.missing`: The word “中” seemed to be missing; try saying every word.

### Spanish

FLEURS validation `fleurs-es-validation-10953383748233911445`, a native recording of “El cristianismo ortodoxo es la religión mayoritaria en Moldavia.”, scored against the prompt with planted errors: phone_sub_far in “religión” (l → p); phone_sub_far in “mayoritaria” (t → ɾ). So the true sounds are the *original* ones. Output of `res.coaching()` (balanced) and `tips()`:

```json
{
 "audio": {
  "schema_version": 1,
  "duration_s": 7.08,
  "clipped_share": 0.0,
  "peak_dbfs": -0.0456,
  "noise_db": -58.6569,
  "speech_db": -21.7696,
  "snr_db": 45.5184,
  "speech_s": 4.1,
  "speech_share": 0.5791,
  "lead_sil_s": 1.72,
  "trail_sil_s": 0.72,
  "expected_s": 3.542,
  "speech_ratio": 1.1575,
  "issues": [],
  "retry_recommended": false
 },
 "coaching": {
  "schema_version": 2,
  "schema_minor": 3,
  "items": [
   {
    "schema_version": 2,
    "schema_minor": 3,
    "kind": "substitution",
    "word": "religión",
    "word_index": 5,
    "syllable_index": 1,
    "phone_index": 2,
    "position": "medial",
    "expected": "p",
    "heard": "l",
    "expected_ipa": "p",
    "heard_ipa": "l",
    "diff_keys": [
     "place",
     "manner",
     "voicing",
     "lateral"
    ],
    "error_prob": 0.682,
    "prob": 0.3059,
    "confidence": 0.865,
    "start": 4.27,
    "end": 4.29
   },
   {
    "schema_version": 2,
    "schema_minor": 3,
    "kind": "mispronounced_word",
    "word": "religión",
    "word_index": 5,
    "syllable_index": null,
    "phone_index": null,
    "position": null,
    "expected": "religión",
    "heard": null,
    "expected_ipa": null,
    "heard_ipa": null,
    "diff_keys": [],
    "error_prob": 0.682,
    "prob": 0.0093,
    "confidence": null,
    "start": 4.13,
    "end": 4.59,
    "n_sounds_flagged": 1
   },
   {
    "schema_version": 2,
    "schema_minor": 3,
    "kind": "substitution",
    "word": "mayoritaria",
    "word_index": 6,
    "syllable_index": 3,
    "phone_index": 6,
    "position": "medial",
    "expected": "ɾ",
    "heard": null,
    "expected_ipa": "ɾ",
    "heard_ipa": null,
    "diff_keys": [],
    "error_prob": 0.6101,
    "prob": 0.3896,
    "confidence": null,
    "start": 4.99,
    "end": 5.01
   },
   {
    "schema_version": 2,
    "schema_minor": 3,
    "kind": "mispronounced_word",
    "word": "mayoritaria",
    "word_index": 6,
    "syllable_index": null,
    "phone_index": null,
    "position": null,
    "expected": "mayoritaria",
    "heard": null,
    "expected_ipa": null,
    "heard_ipa": null,
    "diff_keys": [],
    "error_prob": 0.6101,
    "prob": 0.0294,
    "confidence": null,
    "start": 4.59,
    "end": 5.25,
    "n_sounds_flagged": 1
   }
  ]
 }
}
```

Default tips:

- `manner.stop`: Close completely and release for /p/ in “religión”; it sounded like /l/.
- `sound.unclear`: Listen to /ɾ/ in “mayoritaria” and try it again; it did not sound quite right.

### Real learners (speechocean762 test, "balanced")

(Produced with the round-2 English coaching score; since round 12b the English score reads
`sound_E`, so individual error_prob values differ. Not re-run here: it would mean another pass
over the held-out test set. Current test numbers: `results/coaching.md`.)

- “FIRST PLACE IN THE DIVISION”: ours FIRST /t/ → off, not named (error_prob 0.588); DIVISION /v/ → /f/ (error_prob 0.561). Experts: THE ['DH', 'AH0'] scores [1.2, 2.0]; DIVISION ['D', 'IH0', 'V', 'IH1', 'ZH', 'N'] scores [2.0, 2.0, 0.4, 1.6, 0.8, 2.0], pronounced [('V', 'F')].
- “DO YOU REALLY AND TRULY SAY SO”: ours SAY /s/ → off, not named (error_prob 0.62); SAY /eɪ/ → off, not named (error_prob 0.946); SO /s/ → /n/ (error_prob 0.757). Experts: DO ['D', 'UH0'] scores [2.0, 1.8]; YOU ['Y', 'UW0'] scores [2.0, 1.8]; REALLY ['R', 'IH', 'AH1', 'L', 'IY0'] scores [2.0, 1.6, 1.6, 2.0, 2.0]; AND ['AE0', 'N', 'D'] scores [2.0, 2.0, 1.6]; TRULY ['T', 'R', 'UW1', 'L', 'IY0'] scores [2.0, 2.0, 2.0, 1.6, 2.0]; SO ['S', 'OW0'] scores [0.4, 2.0], pronounced [('S', 'N')].
