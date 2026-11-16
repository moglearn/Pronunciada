# pronunciada design

On-device pronunciation scoring for read-aloud prompts, built on the ZIPA universal phone
recogniser (ONNX). Input: audio + the text the learner was asked to say + language. Output:
calibrated 0–1 acceptance probabilities per sound, per word and for the whole sentence, plus
tone / stress results, with one `strictness` knob.

## Verified facts about the model (measured on this machine)

- Default model: `anyspeech/zipa-small-crctc-500k` (64 M params, **CC-BY-4.0**). `model.onnx` 248 MB,
  `model.fp16.onnx` 126 MB, `model.int8.onnx` 67 MB. Cached at `~/.cache/pronunciada/models/<name>/`.
  The `-ns-` variants are CC-BY-**NC**-SA: research comparison only, never the default.
- Inputs `x` float32 `[N, T, 80]`, `x_lens` int64 `[N]`. Outputs `log_probs` `[N, T', 127]`
  (already log-softmax) and `log_probs_len`. Output rate ≈ 49 frames/s (20 ms).
- Features: Kaldi fbank, 16 kHz, 80 mel bins, 25 ms / 10 ms, `dither=0`, `snip_edges=False`,
  waveform in [-1, 1] float (lhotse `Fbank` defaults). `kaldi-native-fbank` reproduces it without torch.
- Vocabulary: 127 tokens = `<blk>`(0) `<sos/eos>`(1) `<unk>`(2) `▁`(3) + single IPA characters and
  modifiers (`ʰ ʲ ʷ ʼ ː ˞ ˠ ˤ` and combining nasal/unreleased/voiceless/syllabic/dental/velarised/apical).
  **No tone and no stress symbols** – tones and stress need a separate pitch/energy path.
  Greedy output starts with `▁`; treat `▁` as blank-equivalent.
- Speed on an M3 Pro CPU (4 threads): 3.4 s clip → 35 ms fp32, 22 ms int8.
- Output is not very spiky (blank on ~35–40 % of frames), so frame timings are usable.

## Pipeline

```
text ──► text front-end (per language) ──► Reference: words › syllables › target sounds
                                             (+ expected tone / stress / pitch, + allowed variants)
audio ─► fbank ─► ZIPA ONNX ─► frame log-posteriors lp[T,127]
                                   │
        Reference + lp ──► lattice aligner ──► per-sound evidence + timings
        audio + timings ──► prosody path (pitch, energy, duration) ──► tone / stress evidence
        evidence ──► calibrators (sound → word → sentence) ──► probabilities @ strictness
```

Package layout (owner in brackets):

```
pronunciada/types.py            data contract (architect; frozen)
pronunciada/text/ipa.py, base.py  token conventions, registry (architect; frozen)
pronunciada/text/<lang>.py      one front-end per language, exposes build() -> G2P
pronunciada/audio.py            load / resample / fbank
pronunciada/acoustic.py         ZIPA ONNX wrapper
pronunciada/lattice.py          Reference -> token lattice
pronunciada/align.py            lattice CTC forward / Viterbi, per-sound evidence
pronunciada/variants.py         native-variant tables (learned), applied to a Reference
pronunciada/calibrate.py        calibrators + strictness
pronunciada/prosody/            f0, tone, stress
pronunciada/scorer.py           public API
bench/                      datasets, corruption protocol, metrics, Azure runner, experiment scripts
```

## 1. Text front-end

`get_g2p(lang)(text) -> Reference` (see `pronunciada/types.py`). Rules:

- Target sounds are written in **ZIPA's own conventions**, i.e. what the model emits for native
  speakers of that language, not textbook IPA. The objective check for a front-end is the phone
  error rate between its reference and ZIPA's greedy output on native speech (FLEURS); where ZIPA
  is systematically different from the dictionary, the reference follows ZIPA.
- Affricates and diphthongs are one `RefPhone` with several tokens when the language treats them
  as one sound.
- Every syllable has exactly one nucleus. Stress / tone / pitch live on the syllable.
- Natural connected speech must never be punished. Front-ends add rule-based `alts` for the
  well-known optional processes of their language (weak forms, linking, liaison, elision,
  flapping, assimilation across word boundaries, vowel devoicing, sandhi …). A second, automatic
  layer (`pronunciada.variants`) adds every realisation that native speakers produce often enough.
- No GPL code or data in the runtime path (espeak-ng may be used as an offline dev cross-check only).

## 2. Acoustic evidence (`lattice.py`, `align.py`)

The Reference becomes a **lattice** ("sausage"): slot *i* = target sound *i* with its alternatives
(canonical first; an empty alternative = the sound may be dropped). Each alternative is a chain of
token states; standard CTC topology applies along every path (optional blank between tokens,
blank mandatory between identical consecutive tokens, optional blank at both ends). Blank emission
= `logaddexp(lp[:,0], lp[:,3])`. Alternative priors are added as arc weights.

Per slot *i* we compute, with full-sum (forward) scores:

| name | definition | meaning |
|---|---|---|
| `L_can` | log P(audio \| lattice) | — |
| `L_wild_i` | same lattice, but slot *i*'s token states emit `log(1 − p_blank(t))` (any speech sound) and slot *i* may also be skipped | "anything or nothing was said here" |
| `llr_sub_i` | `L_can − L_wild_i` (≤ 0) | main evidence: how much worse the allowed realisations explain the audio than "anything" |
| `L_del_i` | lattice with slot *i* removed | — |
| `llr_del_i` | `L_can − L_del_i` | > 0: the sound is present; < 0: it was probably skipped |

Plus from the Viterbi path of the canonical lattice: start/end frame, which alternative was used,
duration, peak posterior of the target inside the span, and `heard` = the collapsed greedy tokens
overlapping the span (for feedback: "sounded like X").

This is alignment-free scoring: it does not trust a single hard alignment, which matters with CTC
models. Exactness first (a batched reference implementation checked against brute force on tiny
inputs), then a windowed/banded version (reuse global forward α and backward β, recompute only a
local window around the slot) validated against the exact one.

Budget: evidence for a 10 s / 100-sound utterance in ≤ 30 ms on a laptop CPU; pure numpy.

## 3. Native variants (`variants.py`)

Per language, run the aligner over a few hours of native read speech with the wildcard trick, and
collect, for each target sound (with coarse context: word-initial / final / medial, neighbouring
class), what natives actually produced. Any substitution or deletion with a native rate above a
threshold becomes an allowed alternative with prior = log(rate). This single mechanism absorbs
dictionary-vs-model convention gaps, allophones, and connected-speech reductions for every
language, and is what keeps the architecture language-general. Tables ship as small JSON files.

Also stored per (language, sound): mean / std of `llr_sub`, `llr_del`, duration on native speech,
used to express a learner's evidence relative to natives.

## 4. Calibration and strictness (`calibrate.py`)

- **Sound level.** Features: `llr_sub` (raw, per-frame, native-z-scored), `llr_del`, duration z,
  peak posterior, sound class, position, utterance-level mean evidence. Model: logistic regression
  first; a small boosted-tree model only if it clearly wins on held-out data.
- Two calibrated targets are learned from human-rated data: `P(lenient ok)` = a rater would call it
  at least understandable, and `P(strict ok)` = a rater would call it fully correct.
  `strictness s ∈ [0,1]` interpolates in log-odds: `logit P_s = (1−s)·logit P_lenient + s·logit P_strict`.
  Default 0.5. So the knob has a concrete meaning at both ends and stays calibrated in between.
- **Transfer to languages without human ratings.** Features are native-relative, so the English-
  trained calibrator transfers; a per-language bias is then fitted so that native speech is accepted
  at the same target rate in every language, and checked with the synthetic-error protocol below.
- **Word level.** Inputs: its sounds' log-odds (mean, min, share below 0.5, count), stress / tone
  probabilities, omission evidence → `P(word ok)`. **Sentence level**: word log-odds aggregates,
  completeness, pacing → `P(sentence ok)`. Each level has its own calibrator, trained on the
  human word / sentence ratings of speechocean762 and checked with reliability curves (ECE).

## 5. Tones, stress, pitch accent (`prosody/`)

ZIPA has no tone or stress symbols, so these use a small separate model fed by the audio and the
syllable timings from the aligner:

- Frame features at 10 ms: log-F0 (speaker-normalised), voicing, log-energy, plus the fbank frames.
- One small network family ("ProsodyNet", conv + GRU, well under 1 M params, exported to ONNX) that
  looks at a syllable with its neighbours and predicts the syllable's class:
  Mandarin tone (1-4, neutral), stress level (en, de, ru, it, es, pt), Japanese high/low.
- Trained on native speech with labels from the text front-end; scored as
  `P(expected class | audio)` then calibrated like sounds. Mandarin tone sandhi is applied in the
  reference (`tone` = surface tone) so correct sandhi is never punished.
- French and Korean have no word-level stress or tone: nothing is scored there.

## 6. Benchmarks

| purpose | data | metric |
|---|---|---|
| human agreement (English) | speechocean762 test | correlation with human scores at sound / word / sentence level; ECE |
| real learner errors (English) | L2-ARCTIC (if obtainable), EpaDB | detection AUC / F1 |
| all 10 languages | FLEURS test (native) + **controlled errors**: the same audio scored against a reference with a known wrong sound / word / tone / stress | native acceptance rate; error-detection AUC; ECE |
| Traditional script | zh-TW speech if obtainable; otherwise the same audio with Traditional text | identical reference and scores |
| tones | held-out native Mandarin speakers | tone accuracy; wrong-tone detection AUC |
| speed | all of the above | ms per utterance, real-time factor, model size |

The same clips and the same protocol are sent to Azure Pronunciation Assessment when
`AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION` are set; otherwise we compare against published Azure
numbers and say so. Controlled errors are a proxy for real learner errors and are reported as such.

## Conventions for contributors

- Python ≥ 3.10; runtime deps limited to numpy, onnxruntime, kaldi-native-fbank, soundfile, scipy
  plus small permissively-licensed per-language text packages. Training-only deps (torch, sklearn,
  datasets …) stay out of the runtime path.
- Large data and model files go under `~/.cache/pronunciada/` (override with `PRONUNCIADA_CACHE`), never in the repo.
- Tests under `tests/`, runnable offline except those marked `@pytest.mark.network`.

## 7. What shipped, where it differs from the plan above

Sections 1–6 are the plan the code was built against. Measurements changed four things:

- **Per-language output adapter (`adapter.py`), added.** A tiny correction on top of the frozen
  model's frame scores (`lp' = log_softmax(lp + f(lp))`, linear or small convolution, trained with
  CTC on native FLEURS train, run in numpy). It closes most of the gap between the model's habits
  and each language's dictionary conventions and raised controlled-error detection in every
  language (e.g. Japanese 0.956 → 0.986, German 0.946 → 0.987).
- **Learned native variants (§3), off.** With the adapter in place they made detection slightly
  worse, so only the rule-based alternatives from the text front-ends are used. The per-sound
  native statistics are still shipped and used as features.
- **Calibration (§4) uses small boosted trees**, exported to `.npz` and walked in numpy (no
  sklearn at runtime). Trained on speechocean762 train only; a wider training mix did not win
  leave-one-corpus-out. English uses sound identity as a feature, the other languages a
  universal model plus a per-language bias.
- **The word probability is a product of three separately calibrated channels:**

      WordResult.prob = sounds_prob × P_prosody × present_prob

  - `sounds_prob`: every sound of the word acceptable (the word model of §4).
  - `P_prosody` (`stress_prob` / `tone_prob`): tone (zh) or stress (en de es it pt ru) evidence of
    the word's syllables through a per-language word-level calibrator
    (`bench/calib/fit_prosody.py`). Its leniency is set by a fixed rule on development data: the
    strictest setting that costs at most one point of native word acceptance at the default
    strictness. Japanese pitch accent is reported per syllable but only enters the product with
    `Scorer(..., experimental=True)`.
  - `present_prob`: the word was said at all (`bench/calib/fit_omission.py`, one model for all
    languages, fitted on planted extra words). `omitted` is `present_prob < 0.5`. This replaced a
    hard threshold that flagged 3–6 % of native one-sound words ("é", "à", "を") as skipped.
- **Tones use pitch only.** A model that also sees the spectrum or the phone posteriors scores
  higher on native speech but guesses the tone from the syllable's sounds (it keeps "hearing" the
  dictionary tone when the pitch is flattened), which is useless for grading learners.
- An "explain-away" second pass (re-scoring neighbours of a badly wrong word) is implemented
  (`Scorer(..., explain_away=True)`) but off: it did not help on held-out data.

Held-out discipline: nothing is fitted or selected on speechocean762 test, L2-ARCTIC test, EpaDB
test or FLEURS test. The word redesign (round 2), the prosody channel (round 3) and the omission
channel and prosody leniency rule (round 4) were each designed after looking at an earlier
held-out evaluation and then fitted on train / validation data only; the results files say so.
