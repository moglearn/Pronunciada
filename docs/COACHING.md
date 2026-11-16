# Coaching feedback and audio-quality gate

What an app needs to turn a score into advice: for each problem, what was expected, what it
sounded like, how the two differ physically, and how sure we are; plus a check that the
recording itself is usable. Numbers behind every threshold: [results/coaching.md](../results/coaching.md).

```python
res = scorer.score("clip.wav", "I think this is the right path")
res.audio["issues"], res.audio["retry_recommended"]    # audio-quality gate, always computed
items = res.coaching()                                  # structured diagnoses, most important first
from pronunciada.tips import tips
for t in tips(items, max_tips=3, result=res):           # optional default English sentences
    t["tip_key"], t["text"], t["params"]
```

```bash
python -m pronunciada score clip.wav "I think this is the right path" --lang en --coach [--json]
```

`coaching()` reads only the evidence already cached on the result: no score changes (verified
byte-identical on 50 clips, 5 per language) and it costs about 0.01 ms per 10 s clip on native
speech, 0.7 ms in the worst case measured (a clip scored against the wrong sentence). Nothing is
computed unless it is called. It follows the result's strictness: `scorer.rescore(res, 0.9).coaching()`
reports more sounds.

## 1. Item schema (`schema_version` 1)

One flat JSON object per item; every item has every key. `coaching()` returns a list;
`pronunciada.coach.to_json(items)` wraps it as `{"schema_version": 1, "items": [...]}`.

| key | type | meaning |
|---|---|---|
| `schema_version` | int | 1. Additive changes keep the number; renamed / removed keys bump it. |
| `kind` | str | `substitution`, `missing`, `extra`, `tone`, `stress`, `missing_word` |
| `word`, `word_index` | str, int | the prompt word (as written) and its index in `result.words` |
| `syllable_index` | int / null | syllable inside the word (null for `missing_word`) |
| `phone_index` | int / null | sound inside the word (`result.words[i].phones[j]`); sounds only |
| `position` | str / null | sound position in the word: `initial`, `medial`, `final`, `only` |
| `expected` | str / int | sounds: IPA; `tone`: tone 1–5; `stress`: index of the syllable that should be stressed; `missing_word`: the word |
| `heard` | str / int / null | same type as `expected`; **null = flagged, but what it sounded like is not reliable enough to say** |
| `expected_ipa`, `heard_ipa` | str / null | for sound items, equal to `expected` / `heard`; null otherwise |
| `diff_keys` | list[str] | articulatory features that differ (section 2), `["tone"]`, `["stress_position"]`; empty when `heard` is null |
| `prob` | float | the calibrated probability that this item was acceptable: `PhoneResult.prob`, `SyllableResult.tone_prob` / `stress_prob`, `WordResult.present_prob` |
| `confidence` | float / null | estimated precision of the diagnosis (sounds: P(`heard` right within one feature); tones / stress: P(`heard` exactly right)), from FLEURS validation planted errors; null without `heard` |
| `start`, `end` | float | seconds (the word span for stress / missing words) |
| `gate` | dict / null | gate inputs, for logging only (`heard_post`, `log_ratio`, `n_frames`, `candidate`) |

Order: by `(1 − prob) × confidence` (flag-only items count with confidence 0.4), so the
clearest, best-diagnosed problems come first. Rules:

- **Sounds**: an item for every sound with `prob < flag_prob` (0.5) in a word that was said.
  `kind` is `substitution`; `heard` is filled when the reliability gate passes.
- **Missing sounds** (`kind: "missing"`) are **off by default**: on planted deletions no rule
  reached 0.85 precision (best 0.67), so a skipped sound appears as a flagged substitution.
- **Extra sounds** (`kind: "extra"`) are reserved in the schema; not produced yet.
- **Tones** (Mandarin) and **stress** (en de es it pt ru): an item when both the syllable's
  probability and the word's calibrated channel (`WordResult.tone_prob` / `stress_prob`, what the
  word score uses) are < 0.5. `heard` = the most probable tone / stressed syllable if its
  posterior is ≥ 0.3 / 0.3 and it differs from what is accepted.
- **Missing words**: `present_prob < 0.5` (the shipped omission channel); the word's sounds
  are then not listed separately.

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

## 3. Reliability gate (when `heard` is given)

The heard candidate is the language's sound of the same type (vowel / consonant) with the
highest geometric-mean span posterior of its tokens (adapted model output over the sound's
aligned span ± 1 frame); the expected sound and its accepted alternatives are excluded.
Thresholds (`pronunciada/data/coach/gate.json`, override per call: `res.coaching(min_heard_post=0.2)`):

| parameter | default | meaning |
|---|---|---|
| `flag_prob` | 0.5 | an item is reported when its probability is below this |
| `min_log_ratio` | -1.0 | log(P heard / P expected) over the span, at least |
| `min_heard_post` | 0.0 (German 0.25) | posterior of the heard sound, at least |
| `min_frames` | 0 | aligned frames, at least |
| `tone_min_post` / `stress_min_post` | 0.3 / 0.3 | posterior of the heard tone / stressed syllable |
| `missing_word_prob` | 0.5 | `missing_word` when `present_prob` is below |
| `missing_max_frames`, `missing_min_llr_del` | 0, −1e9 | "missing" rule (−1e9 = off) |
| `confidence` | edges [6.755, 7.917, 8.998] → [0.865, 0.974, 0.976, 0.976] | precision by log-ratio bin |
| `by_lang` | {"de": {"min_heard_post": 0.25}} | per-language overrides |

How they were set (`bench/coach/gather.py`, `bench/coach/fit_gate.py`): FLEURS **validation**,
250 clips per language, native audio scored against references in which one sound X was replaced
by Y (near and far substitutions), so the truth is X. Rule fixed in advance: the loosest grid
setting with pooled "within one feature" precision ≥ 0.85 and the most coverage, then
per-language tightening where a language misses 0.85 alone. Result: pooled precision 0.95
within one feature, 0.88 exact; coverage 23 % of planted substitutions, which is the share the
scorer flags at all (every flagged planted sound got a diagnosis except in German). Real learners
(report only): on L2-ARCTIC / EpaDB substitutions our `heard_ipa` matches the annotator 0.56 /
0.52 exactly and 0.66 / 0.66 within one feature, but only about 3 % of annotated substitutions
are flagged at the default strictness, and many diagnoses fall on sounds annotators marked
deleted or correct; see results/coaching.md.

**Where `heard` is unreliable**: German (vowel length a / ɑː, ə / ɛ confusions: 0.74 within one
feature before the German-only bar; with the bar only 2.6 % of planted errors get a diagnosis),
affricates in all languages (0.64), English (0.875, near the bar), and every "missing sound".
Native false alarms that are flagged usually still get a (necessarily wrong) `heard`: about 0.26
per FLEURS validation clip.

## 4. Default tips (`pronunciada/tips.py`)

Separate from the data. `tips(items, max_tips=3, result=None)` returns at most `max_tips`
`{"tip_key", "text", "params", "item"}`: diagnosed items first in coaching order, then
flag-only items; at most one tip per sound and per word, so a sound never gets two tips (the
single most salient difference is chosen in a fixed priority: consonants manner → place →
voicing → aspiration → lateral → palatalisation → length; vowels glide (if it is the only
difference) → height → backness → rounding → nasality → length). `params` carries `expected`,
`heard`, `word` and the template's extra fields (`expected_place`, `heard_place`, `glide`,
`expected_tone`, `heard_tone`, `expected_ord`, `heard_ord`, `syllable`) so an app can render its
own or localised copy from `tip_key`.

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
| `palatalisation.add` | Raise the middle of your tongue (a slight “y” colour) on /{expected}/ in “{word}”; it sounded like plain /{heard}/. |
| `palatalisation.remove` | Say /{expected}/ in “{word}” without the “y” colour; it sounded like /{heard}/. |
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
| `rhotic.add` | Add the r-colour to /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `rhotic.remove` | Say /{expected}/ in “{word}” without r-colour; it sounded like /{heard}/. |
| `sound.substitution` | Say /{expected}/ in “{word}”; it sounded like /{heard}/. |
| `sound.unclear` | Listen to /{expected}/ in “{word}” and try it again; it did not sound quite right. |
| `sound.missing` | Don't skip /{expected}/ in “{word}”. |
| `sound.extra` | There seems to be an extra sound in “{word}”; try it without it. |
| `tone.wrong` | “{syllable}” should be {expected_tone} (tone {expected}); it sounded {heard_tone} (tone {heard}). |
| `tone.unclear` | Check the tone of “{syllable}”: it should be {expected_tone} (tone {expected}). |
| `stress.position` | In “{word}”, stress the {expected_ord} syllable; it sounded stressed on the {heard_ord}. |
| `stress.unclear` | In “{word}”, put the stress on the {expected_ord} syllable. |
| `word.missing` | The word “{word}” seemed to be missing; try saying every word. |

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
   `too_short`: ask for a new recording (move back from the microphone / speak louder / say the
   whole sentence) before showing pronunciation feedback; `too_noisy`: suggest a quieter place.
   Low scores on such audio are more often the recording than the speaker.
2. Take `result.coaching()`; show the first few items. Use `heard` only when it is not null;
   otherwise say the sound needs work without claiming what was heard. `confidence` can hide
   diagnoses below the app's own bar.
3. Use `tip_key` + `params` to render localised tips, or `pronunciada.tips.tips()` for the
   default English. Keep at most 3 per utterance (default).
4. For learners who want stricter feedback, rescore with a higher strictness
   (`scorer.rescore(res, 0.9).coaching()`): more sounds are flagged; on real learners precision
   of `heard` on actual substitutions stays similar (0.73–0.75 within one feature at strictness 1.0).
5. Log `schema_version` and `gate` with the app's events to monitor drift.

## 7. Example outputs

### English

FLEURS validation `fleurs-en-validation-10098964113747380446`, a native recording of “Then, Lakkha Singh took the lead in singing the bhajans.”, scored against the prompt with planted errors: stress in “bhajans” (syllable 0 → 1); phone_sub_far in “singing” (ŋ → f). So the true sounds are the *original* ones.

```json
{
 "audio": {
  "schema_version": 1,
  "duration_s": 4.14,
  "clipped_share": 0.0,
  "peak_dbfs": -44.9021,
  "noise_db": -82.8452,
  "speech_db": -66.134,
  "snr_db": 25.7576,
  "speech_s": 3.0,
  "speech_share": 0.7246,
  "lead_sil_s": 0.38,
  "trail_sil_s": 0.34,
  "expected_s": 2.496,
  "speech_ratio": 1.2019,
  "issues": [],
  "retry_recommended": false
 },
 "coaching": {
  "schema_version": 1,
  "items": [
   {
    "schema_version": 1,
    "kind": "substitution",
    "word": "bhajans",
    "word_index": 9,
    "syllable_index": 1,
    "phone_index": 3,
    "position": "medial",
    "expected": "æ",
    "heard": "ə",
    "expected_ipa": "æ",
    "heard_ipa": "ə",
    "diff_keys": [
     "height",
     "backness"
    ],
    "prob": 0.0971,
    "confidence": 0.974,
    "start": 3.39,
    "end": 3.41
   },
   {
    "schema_version": 1,
    "kind": "substitution",
    "word": "took",
    "word_index": 3,
    "syllable_index": 0,
    "phone_index": 2,
    "position": "final",
    "expected": "k",
    "heard": "t",
    "expected_ipa": "k",
    "heard_ipa": "t",
    "diff_keys": [
     "place"
    ],
    "prob": 0.1343,
    "confidence": 0.865,
    "start": 1.43,
    "end": 1.45
   },
   {
    "schema_version": 1,
    "kind": "substitution",
    "word": "Lakkha",
    "word_index": 1,
    "syllable_index": 1,
    "phone_index": 3,
    "position": "medial",
    "expected": "k",
    "heard": "h",
    "expected_ipa": "k",
    "heard_ipa": "h",
    "diff_keys": [
     "place",
     "manner"
    ],
    "prob": 0.1795,
    "confidence": 0.865,
    "start": 0.89,
    "end": 0.91
   },
   {
    "schema_version": 1,
    "kind": "substitution",
    "word": "bhajans",
    "word_index": 9,
    "syllable_index": 1,
    "phone_index": 2,
    "position": "medial",
    "expected": "dʒ",
    "heard": "ʃ",
    "expected_ipa": "dʒ",
    "heard_ipa": "ʃ",
    "diff_keys": [
     "manner",
     "voicing"
    ],
    "prob": 0.2317,
    "confidence": 0.865,
    "start": 2.95,
    "end": 3.33
   },
   {
    "schema_version": 1,
    "kind": "substitution",
    "word": "singing",
    "word_index": 7,
    "syllable_index": 1,
    "phone_index": 2,
    "position": "medial",
    "expected": "f",
    "heard": "ŋ",
    "expected_ipa": "f",
    "heard_ipa": "ŋ",
    "diff_keys": [
     "place",
     "manner",
     "voicing"
    ],
    "prob": 0.4349,
    "confidence": 0.976,
    "start": 2.29,
    "end": 2.31
   },
   {
    "schema_version": 1,
    "kind": "substitution",
    "word": "bhajans",
    "word_index": 9,
    "syllable_index": 0,
    "phone_index": 1,
    "position": "medial",
    "expected": "ɑ",
    "heard": "u",
    "expected_ipa": "ɑ",
    "heard_ipa": "u",
    "diff_keys": [
     "height",
     "rounded"
    ],
    "prob": 0.3666,
    "confidence": 0.865,
    "start": 2.75,
    "end": 2.77
   },
   {
    "schema_version": 1,
    "kind": "substitution",
    "word": "Lakkha",
    "word_index": 1,
    "syllable_index": 0,
    "phone_index": 1,
    "position": "medial",
    "expected": "ə",
    "heard": "ɑ",
    "expected_ipa": "ə",
    "heard_ipa": "ɑ",
    "diff_keys": [
     "height",
     "backness"
    ],
    "prob": 0.4659,
    "confidence": 0.865,
    "start": 0.73,
    "end": 0.75
   },
   {
    "schema_version": 1,
    "kind": "substitution",
    "word": "Lakkha",
    "word_index": 1,
    "syllable_index": 0,
    "phone_index": 2,
    "position": "medial",
    "expected": "k",
    "heard": null,
    "expected_ipa": "k",
    "heard_ipa": null,
    "diff_keys": [],
    "prob": 0.2314,
    "confidence": null,
    "start": 0.85,
    "end": 0.87
   },
   {
    "schema_version": 1,
    "kind": "substitution",
    "word": "bhajans",
    "word_index": 9,
    "syllable_index": 1,
    "phone_index": 4,
    "position": "medial",
    "expected": "n",
    "heard": null,
    "expected_ipa": "n",
    "heard_ipa": null,
    "diff_keys": [],
    "prob": 0.4837,
    "confidence": null,
    "start": 3.51,
    "end": 3.55
   }
  ]
 }
}
```

Default tips:

- `vowel.open`: Open your mouth a little more for /æ/ in “bhajans”; it sounded like /ə/.
- `place`: Make /k/ in “took” with the back of the tongue against the soft palate; it sounded like /t/, made with the tongue tip just behind the upper teeth.
- `manner.stop`: Close completely and release for /k/ in “Lakkha”; it sounded like /h/.

### Mandarin

FLEURS validation `fleurs-zh-validation-1366745426988339540`, a native recording of “太空中的卫星接到电话后，几乎是立刻将其反射回去。”, scored against the prompt with planted errors: phone_sub_far in “其” (i → ɑʊ); tone in “后” (4 → 1). So the true sounds are the *original* ones.

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
  "schema_version": 1,
  "items": [
   {
    "schema_version": 1,
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
    "prob": 0.0069,
    "confidence": 0.959,
    "start": 1.99,
    "end": 2.15
   },
   {
    "schema_version": 1,
    "kind": "substitution",
    "word": "其",
    "word_index": 11,
    "syllable_index": 0,
    "phone_index": 1,
    "position": "final",
    "expected": "ɑʊ",
    "heard": "i",
    "expected_ipa": "ɑʊ",
    "heard_ipa": "i",
    "diff_keys": [
     "height",
     "backness",
     "glide"
    ],
    "prob": 0.4151,
    "confidence": 0.976,
    "start": 3.57,
    "end": 3.61
   },
   {
    "schema_version": 1,
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
    "prob": 0.2294,
    "confidence": 0.0,
    "start": 0.91,
    "end": 0.97
   },
   {
    "schema_version": 1,
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
    "prob": 0.0759,
    "confidence": 0.0,
    "start": 0.97,
    "end": 1.03
   }
  ]
 }
}
```

Default tips:

- `tone.wrong`: “后” should be high and level (tone 1); it sounded falling (tone 4).
- `vowel.open`: Open your mouth a little more for /ɑʊ/ in “其”; it sounded like /i/.
- `word.missing`: The word “中” seemed to be missing; try saying every word.

### Spanish

FLEURS validation `fleurs-es-validation-10953383748233911445`, a native recording of “El cristianismo ortodoxo es la religión mayoritaria en Moldavia.”, scored against the prompt with planted errors: phone_sub_far in “religión” (l → p); phone_sub_far in “mayoritaria” (t → ɾ). So the true sounds are the *original* ones.

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
  "schema_version": 1,
  "items": [
   {
    "schema_version": 1,
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
    "prob": 0.3059,
    "confidence": 0.976,
    "start": 4.27,
    "end": 4.29
   },
   {
    "schema_version": 1,
    "kind": "substitution",
    "word": "mayoritaria",
    "word_index": 6,
    "syllable_index": 3,
    "phone_index": 6,
    "position": "medial",
    "expected": "ɾ",
    "heard": "t",
    "expected_ipa": "ɾ",
    "heard_ipa": "t",
    "diff_keys": [
     "manner",
     "voicing"
    ],
    "prob": 0.3896,
    "confidence": 0.976,
    "start": 4.99,
    "end": 5.01
   }
  ]
 }
}
```

Default tips:

- `manner.stop`: Close completely and release for /p/ in “religión”; it sounded like /l/.
- `manner.flap`: Make /ɾ/ in “mayoritaria” a single quick tap of the tongue tip; it sounded like /t/.

The English example shows a typical failure: the planted stress error was not flagged, and
several native sounds of unfamiliar names (“Lakkha”, “bhajans”) were flagged and given a
`heard` that is wrong by construction.
