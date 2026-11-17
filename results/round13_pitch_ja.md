# Round 13: Japanese pitch accent, an attempt to leave "experimental"

> **Removed on 2026-11-16 (round 15):** all Japanese pitch-accent code, models and coaching items were taken out of the package by the owner's decision. Pitch accent judged against dictionary labels could not be separated from natives' own variation, and a reference-recording comparison could not say a learner was *wrong*. This file is kept as the record.

**Outcome: it stays experimental.** No candidate meets the bars. The decisive failure is bar 2:
on held-out wrong-accent words from unseen voices, the best model catches **15 %** at the
default setting, against a **50 %** bar. The default (`pitch_ja`) and `ja.json` are unchanged.
Nothing held-out was run, as briefed.

What did ship, all opt-in:
* the round-13 machinery: accent classes in the Japanese reference, a word-level judgment on
  dependable words, a `pitch_accent` coaching kind and tips;
* one selectable candidate, `PRONUNCIADA_PITCH_MODEL=pitch_ja_v5`, with its own calibrator
  `ja.pitch_ja_v5.json` and a `judge` block, so the coaching items can be tried end to end.

**Unverified on real learners.** No recordings with marked Japanese pitch-accent mistakes were
available. Every "wrong accent" below is synthetic.

## 1. Data and licenses

| source | used for | license / terms |
|---|---|---|
| FLEURS-ja (train / validation) | natural training (train); dev and "natives" (validation) | CC BY 4.0 |
| Open JTalk front-end + HTS engine via `pyopenjtalk-plus` 0.4.1.post10 (installed) | system 1: accent-edited synthesis | MIT (pyopenjtalk-plus); Open JTalk / hts_engine BSD-3 |
| HTS voice "Mei" (`mei_normal.htsvoice`, MMDAgent project, Nagoya Institute of Technology), bundled | system 1 voice | **CC BY 3.0**: commercial use and derivatives allowed with attribution. Attribution belongs in NOTICE if any model trained on it ships (none ships by default now; `pitch_ja_v5` did not use it) |
| Praat PSOLA via `parselmouth` (offline) on natural FLEURS-ja audio | system 2: natural voices re-pitched | GPL tool used offline only, nothing linked into the package; the audio remains FLEURS (CC BY 4.0) |

Considered, **not installed** (no approval was asked for, and each is more than a small addition):

| candidate | license (from the model card) | why not now |
|---|---|---|
| Kokoro-82M (hexgrad) | Apache-2.0 weights | its Japanese pitch-accent control via the misaki G2P is unverified; ~330 MB + deps |
| MeloTTS-Japanese (myshell-ai) | MIT | needs a Japanese BERT + UniDic (>1 GB with dependencies); the card does not state accent control |

Rejected outright:
* VOICEVOX: per-character terms, not a clear license for training on its output;
* JSUT / JVS-based models: corpus terms restrict commercial use;
* Coqui XTTS: non-commercial license.

A second permissive HTS voice would give a cleaner held-out synthesis voice. The Open JTalk
`nitech_jp_atr503_m001` voice (CC BY 3.0, ~2 MB) is the obvious one; it needs the owner's direct
go-ahead to download.

**How the pairs are made** (`bench/prosody/synth_ja.py`). For each sentence:
* one content-word accent phrase (≥ 2 morae) gets its nucleus moved, removed or added (the NJD
  `acc` of the phrase head edited before label making);
* labels come from the scorer's own front-end fed the edited NJD, so the mora high/low labels
  are exactly those of the audio.

The sets:
* **System 1, HTS**:
  * train: FLEURS-ja train texts, speed 0.9–1.1, half-tone −3…+3 → 1 746 clips (873 edited);
  * held-out: validation texts, speed {0.82, 1.22}, half-tone {−6, +5}, a setting range never
    seen in training, but the same voice → 458 clips.
* **System 2, PSOLA**: the edited phrase's pitch tier is set to the speaker's local median
  +2.5 / −2 semitones following the new high/low pattern. The "correct" twin is resynthesised by
  the same PSOLA with its own contour, so artefacts do not give the answer away. Measured shift on
  the changed morae: median 2.4 semitones toward the new label, 82 % moved by more than 1
  semitone.
  * held-out: FLEURS-ja validation, unseen voices → 460 clips (230 edited);
  * train (round 13b): FLEURS-ja train → 1 752 clips.

## 2. Models

All are pitch-only (pitch + local register + neighbor context, the v3r recipe; no encoder
layers), with contour-transplant augmentation 0.6 and 4 000 steps. The final weights are kept,
with no selection on validation.

| name | training data |
|---|---|
| `pitch_ja` (shipped) | round 1 local model, FLEURS-ja train |
| `pitch_ja_v3r` (round 8) | FLEURS-ja train |
| `pitch_ja_v4` | FLEURS-ja train + HTS pairs |
| `pitch_ja_v4pt` | HTS pairs only (pre-training) |
| `pitch_ja_v4ft` | `v4pt` fine-tuned on FLEURS-ja train (2 000 steps) |
| `pitch_ja_v5` | FLEURS-ja train + PSOLA pairs of train speakers (no HTS), so HTS is an **unseen system and voice** for it |
| heuristic | per word: the largest declination-corrected fall between adjacent morae vs the fall at the expected nucleus |

## 3. Dependable words (`pronunciada/prosody/pitch_judge.py`, `bench/prosody/pitch_ja_eval.py`)

The Japanese reference now carries `RefSyllable.pitch_class`: part of speech, head or chained in
its accent phrase, single or compound phrase, and accent type (heiban / atamadaka / odaka /
nakadaka). FLEURS-ja validation was split by sentence id: half A chooses, half B reports.

A class is dependable if all of these hold:
* it is a content word (noun, verb, adjective, adverb or adnominal; particles and auxiliaries
  never);
* it has ≥ 15 words on half A;
* natives' words of that class are flagged no more often than 5 % at a provisional threshold.

Words need ≥ 2 morae. Word score = mean −log P(dictionary high / low) over the word's morae. On
the HTS training pairs with `v3r`, mean beat max and sum (AUC 0.823 vs 0.812 / 0.814).
Thresholds flag 1 / 2.5 / 5 % of half-A natives' judged words (careful / balanced / eager).

For `pitch_ja_v5`, 9 classes are dependable: heiban and nakadaka single nouns, compound nouns
(heiban / nakadaka / odaka), heiban verbs, and others. **Coverage: 31.5 % of all words of running
text** on half B (1 168 judged words).

## 4. The four bars (none of this data was used for training)

Thresholds are at "balanced" (2.5 % on half A). "Caught" = edited words flagged; twin = the same
words in the unedited version.

| | bar 1: natives flagged, val B (≤ 3 %) | bar 2a: PSOLA held-out, unseen natural voices, caught (≥ 50 %) / AUC / twin | bar 2b: HTS held-out (unseen settings; unseen system only for v5) caught / AUC | bar 3: transplant follow, `prosody3` swap (≥ v3r) | coverage |
|---|---|---|---|---|---|
| heuristic | 2.7 % | 4.4 % / 0.686 / 4.4 % | 6.0 % / 0.647 | – | 29 % |
| `pitch_ja_v3r` | 3.3 % | 5.3 % / 0.674 / 2.1 % | 10.4 % / 0.828 | 0.509 | 30 % |
| `pitch_ja_v4` | 4.5 % | 8.3 % / 0.722 / 3.6 % | **69.6 % / 0.986** (same voice) | 0.523 | 28 % |
| `pitch_ja_v4pt` | **2.9 %** | 2.5 % / 0.637 / 3.4 % | 38.8 % / 0.990 (same voice) | 0.586 | 29 % |
| `pitch_ja_v4ft` | 4.4 % | 11.2 % / 0.701 / 2.0 % | 12.5 % / 0.857 | 0.525 | 29 % |
| **`pitch_ja_v5`** | 5.2 % | **14.6 % / 0.778** / 9.4 % (PSOLA seen in training, voices unseen) | 14.5 % / 0.834 (**unseen system and voice**) | 0.529 | 32 % |

Bar 4 (full-scorer native word acceptance drop ≤ 0.5 points) was not reached, since nothing
passes bar 2. For reference, the fitted word channel of `pitch_ja_v5` (`fit_prosody`,
`ja.pitch_ja_v5.json`) costs 0.9 points (.987 → .978) with product AUC 0.771, against 0.752
for `pitch_ja`.

**Which bars fail, by how much:**
* **Bar 2 fails for every candidate.** On unseen natural voices the best is 14.6 % (`v5`, 35
  points short). On an unseen system it is 14.5 % (`v5`). `v4` reaches 70 % on held-out HTS
  only because that is the same voice it was trained on: it learns that voice's realization of
  accent, not accent in general (8 % on PSOLA).
* **Bar 1** holds only for `v4pt` (2.9 %), which is useless on natural voices (2.5 % caught).
  The other candidates flag 3.3–5.2 % of half-B natives at a threshold set to 2.5 % on half A.
  Natives' variation between the two halves is larger than the margin.
* **Bar 3 holds for every candidate** (0.52–0.59 vs v3r 0.509): none of these pitch-only models
  leaks word identity.

Reading: a 2–4.5 semitone change in one word of natural connected speech is hard for these
models to tell from natives' own variation (regional accent, phrasing, declination). With
dictionary labels on natural speech as the main supervision, the signal-to-noise ratio is too
low for a word-level verdict at ≤ 3 % false alarms. Synthetic pairs from one HTS voice do not
transfer to real voices.

## 5. What would move it

* More voices for the synthetic pairs: a second permissive HTS voice (nitech m001, CC BY 3.0,
  ~2 MB) and a neural TTS whose accent can be set (Kokoro / MeloTTS: licenses fine, sizes and
  accent control to verify). Hold one voice and one system out.
* Accent-phrase-level rather than word-level judgment (the fall often lands on the particle),
  and a contrastive score (P(dictionary pattern) vs P(the best alternative pattern)) instead of
  the mean NLL.
* Real learner data with marked accent errors (none public). Even a few hundred teacher-marked
  words would replace the synthetic bar.

## 6. What changed in the package (no default behavior changed)

* `pronunciada/types.py`: `RefSyllable.pitch_class` (None outside Japanese).
* `pronunciada/text/ja.py`: sets it per mora.
* `pronunciada/prosody/pitch_judge.py` (new): class keys, dependable-word judgment, fall
  positions.
* `pronunciada/prosody/__init__.py`: a pitch model with a `"judge"` block adds `pitch_word`
  (score, per-sensitivity flags, expected / heard fall) to the morae of judged words.
* `pronunciada/coach.py`: kind `pitch_accent`, `schema_minor` 2, `pitch_threshold` override.
* `pronunciada/tips.py`: five `pitch_accent.*` keys.
* `docs/COACHING.md` §3c.
* `pronunciada/data/prosody/pitch_ja_v5.{onnx,json}`: 628 KB. The model folder is 3.09 MB
  (limit 5 MB).

Latency: `pitch_ja_v5` takes 14.6 ms (min, densest 10 s FLEURS-ja clip, 2 threads) vs 14.5 ms for
`pitch_ja`. The judgment adds no measurable time.

Checks:
* Byte identity on 5 FLEURS validation clips each for zh, ja (default model), fr and ko: all
  identical to the round-11 dump (`bench/prosody/identity_check.py`).
* Tests: 497 passed, 1 skipped. New: `tests/test_pitch_ja_r13.py`, plus a `pitch_ja_v5`
  load-and-score case that also exercises the coaching item.

Bench code:
* `bench/prosody/synth_ja.py` (both systems);
* `bench/prosody/pitch_ja_eval.py` (dependable filter, thresholds, bars);
* `bench/prosody/prosody3.py` (`--data synth|both|psola`, `--init`, `--last`).

Caches are under `~/.cache/pronunciada/prosody/` (`synja_*.pkl`) and
`~/.cache/pronunciada/data/synja/` (held-out audio).
