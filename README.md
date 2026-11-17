# Pronunciada

On-device pronunciation checking for ten languages. Give it a recording and the text the speaker
was meant to say; it returns how likely each **sound**, each **word** and the whole **sentence**
was pronounced acceptably, plus coaching feedback an app can show. It runs on a laptop CPU with
numpy and ONNX Runtime; no audio leaves the device.

Languages: English, Mandarin Chinese (Simplified and Traditional text), Japanese, Spanish (Latin
American), French, German, Italian, Brazilian Portuguese, Russian, Korean.

**Results against Microsoft Azure Pronunciation Assessment: [results/SCOREBOARD.md](results/SCOREBOARD.md)**
(one page, generated from the benchmark tables, with confidence intervals, p-values and a tally).

## What it checks

- **Sounds, words, sentences**: a calibrated 0–1 probability at each level, plus 0–1 sentence
  scores for accuracy, fluency, intonation ("prosodic") and overall, and a 0–10 word score.
  One `strictness` knob (0 = "understandable", 1 = "native-like", default 0.5) moves every
  probability; `rescore()` re-grades without re-running the model.
- **Mandarin tones** per syllable (tone sandhi applied, so correct sandhi is never punished).
- **Word stress** in English, German, Spanish, Italian, Portuguese and Russian.
- **Japanese vowel length and っ** (long / short vowels, doubled consonants).
- **Contrast evidence** for sound pairs the base model hears poorly: Korean aspiration and lax /
  tense stops, Mandarin aspiration, Russian soft / hard consonants (attribute heads), and English
  θ / ð, German and Japanese voicing (contrast caps). In Mandarin and Japanese the evidence also
  lowers the word probability.
- **Words said at all**: skipped words get `omitted = True`.
- **Coaching** (`res.coaching()`): the speaker's clearest errors, most important first, each
  with what was expected, what it sounded like (named only when that is reliable), which
  articulatory features differ (also as feature-level claims when the replacement cannot be
  named, e.g. "no voice", "lips not rounded"), and how sure it is; reading miscues
  (`missing_word`, `mispronounced_word`, `extra_speech`) and delivery flags (`long_pause`,
  `pause_mid_phrase`, `rushed_punctuation`, `fast`; `slow`, `flat_pitch` on request), three
  sensitivities, and default English tips with stable `tip_key`s (`pronunciada.tips`). Schema:
  [docs/COACHING.md](docs/COACHING.md).
- **Audio-quality gate** (`res.audio`): `clipped`, `too_noisy`, `too_quiet`, `too_short`,
  `mostly_silence`, and `retry_recommended`.

## Install

```bash
pip install .                 # core: English and the European languages, Korean
pip install ".[zh]"           # + Mandarin text front-end (pypinyin, jieba, OpenCC)
pip install ".[ja]"           # + Japanese text front-end (pyopenjtalk-plus, ~300 MB installed)
pip install ".[numbers]"      # + digits read out as words (num2words, LGPL)
```

Runtime dependencies are numpy, scipy, onnxruntime, kaldi-native-fbank and soundfile; no torch.
The acoustic model ([anyspeech/zipa-small-crctc-500k](https://huggingface.co/anyspeech/zipa-small-crctc-500k),
64 M parameters, CC BY 4.0) is downloaded on first use into `~/.cache/pronunciada`
(`PRONUNCIADA_CACHE` overrides): a 248 MB download (67 MB with `precision="int8"`); a derived
copy that exposes internal layers (`model.emb.onnx`, 260 MB) is written next to it, so the cache
takes about 0.5 GB. A clean install with the Mandarin extra is about 240 MB. Everything this
project trained ships in the package (about 10 MB) with the pronunciation dictionaries (about
10 MB).

## Usage

```python
import pronunciada

scorer = pronunciada.Scorer("es")                    # loads the model once
res = scorer.score("clip.wav", "Se recomienda a los viajeros que se informen.")

res.prob, res.scores                                 # sentence probability; accuracy / fluency / prosodic / total
for w in res.words:
    print(w.text, round(w.prob, 2), w.sounds_prob, w.stress_prob, w.present_prob, w.omitted)
    for p in w.phones:
        print("   ", p.ipa, round(p.prob, 2), p.heard)   # each sound, and what it sounded like

strict = scorer.rescore(res, strictness=0.9)         # re-grade without re-running the model
```

```python
from pronunciada.tips import tips

res = pronunciada.Scorer("zh").score("clip.wav", "我们明天去北京")
items = res.coaching(sensitivity="balanced")          # "careful": fewer, surer; "eager": more
for t in tips(items, max_tips=3, result=res):
    print(t["tip_key"], t["text"])
print(res.audio["issues"], res.audio["retry_recommended"])
```

```bash
python -m pronunciada score clip.wav "The quick brown fox" --lang en --coach --sensitivity balanced
python -m pronunciada score clip.wav "The quick brown fox" --lang en --json      # machine-readable
```

## Results in short

All numbers are held-out and come from one frozen run of the current code (2026-11-17); Azure
was called on the same recordings and prompts (cached responses, November 2026). Of **84
comparisons with Azure**, Pronunciada is ahead in 59 (39 significant), behind in 25 (4
significant), each by its own table's paired test ([tally](results/SCOREBOARD.md#tally-against-azure)).

- **English learners (speechocean762 test)**: correlation with expert raters 0.685 vs 0.538 for
  sounds and 0.761 vs 0.673 for sentence accuracy (both significant); words 0.609 vs 0.589 and
  fluency 0.792 vs 0.708 (not significant); intonation 0.782 vs 0.827 and overall 0.773 vs 0.796
  behind (not significant).
- **Real English learner errors**: AUC 0.712 vs 0.631 (L2-ARCTIC) and 0.721 vs 0.587 (EpaDB).
  Coaching: at "balanced" 66 % of flagged sounds are expert-marked errors at 36 % recall, against
  44 % for Azure at the same recall.
- **Mandarin learners (OMPAL test)**: word AUC 0.794 vs 0.704 for any error and 0.782 vs 0.679
  for tone errors; sentence accuracy correlation 0.635 vs 0.363. The current Mandarin defaults were
  tuned on OMPAL dev learners; before that tuning: 0.797 / 0.792 / 0.388.
- **Ten languages, one word of the prompt changed on native audio**: near-miss words caught at 5 %
  false alarms ahead in 7 languages (significant), behind in Mandarin and Italian (not
  significant). Tone-only swaps: level with Azure (AUC 0.844 vs 0.846).
- **Common Voice reviewer verdicts (cv-real test)**: ahead in 7 of 11 locales, significant only in
  French; behind (not significant) in Japanese, Portuguese, Russian, Korean; native vs non-native
  English speakers behind (significant).
- **Natives**: 98.7–99.9 % of sounds, 96.8–98.5 % of words, 99.0–99.8 % of sentences accepted.
- **Speed**: 51 ms for a 3.5 s clip (40 ms int8) on a laptop CPU with 4 threads; 0.3–0.45 s for a
  9–14 s clip against Azure's 1.8–3.0 s round trip.

## Known gaps

- **Real-learner proof exists only for English and Mandarin.** The other eight languages are
  measured on native recordings with altered prompts and on Common Voice reviewer votes, not on
  learners marked by experts. Real mistakes are much harder to detect than planted ones (English:
  AUC about 0.71–0.72 on learner errors against 0.98 on near-word swaps).
- **Contrasts still missed or only partly heard**: Italian double consonants (fato / fatto) and
  mid vowels, Portuguese nasal vowels; at **word** level, Korean, Russian, German and English
  contrast errors (aspiration, softness, voicing, tenseness, θ / ð) can still leave the word score
  high, because the word factor was adopted only for Mandarin and Japanese. Per-contrast rows:
  [SCOREBOARD](results/SCOREBOARD.md), [results/contrast_audit.md](results/contrast_audit.md).
- **English sentence intonation** ("prosodic") trails Azure, and the overall sentence score with it.
- **Japanese pitch accent is not scored** (tried in rounds 1, 8 and 13; [results/round13_pitch_ja.md](results/round13_pitch_ja.md)).
- **No phone build.** Everything is ONNX plus numpy, but no iOS / Android bindings exist and
  nothing has been timed on a phone.
- Text front-ends handle ordinary prose; numbers (without `num2words`), abbreviations and names
  outside the dictionaries fall back to spelling rules and are listed in `result.warnings`.
- **Licensing**: one shipped calibrator was trained partly on non-commercial data, and three
  dictionaries are CC BY-SA; see [docs/LICENSING.md](docs/LICENSING.md) before any commercial use.

## Reproducing the benchmarks

Benchmark-only packages: `pip install ".[bench]"` (or `requirements-bench.txt`; torch for
training: `requirements-prosody.txt`). Data download and layout: [docs/benchmarks_data.md](docs/benchmarks_data.md).
Azure responses are cached under `~/.cache/pronunciada/azure/` and never re-sent; the commands
below read only the cache (`--steps azure` / `run-azure` send requests and need
`AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION`).

| table | command | writes |
|---|---|---|
| English vs Azure (speechocean762, L2-ARCTIC, EpaDB; speed, int8) | `python -m bench.eval_english --workers 3` | `results/english_vs_azure.md`, `tables/english_vs_azure.json` |
| ten languages, native acceptance, planted errors, swap benchmark vs Azure | `python -m bench.eval_languages --workers 3` | `results/languages.md`, `tables/languages.json`, `tables/swap_ours_vs_azure.json` |
| cv-real (Common Voice, test half) | `python -m bench.eval_cv --steps ours,report` | `results/cv_real.md`, `tables/cv_real.json` |
| OMPAL (Mandarin learners) | `python -m bench.ompal_build --clone` once, then `python -m bench.eval_ompal --steps ours,report` | `results/ompal.md`, `tables/ompal.json` |
| coaching vs Azure (speechocean762 test) | `python -m bench.coach.dump --sets so762-test`, `python -m bench.coach.azure_yardstick`, `python -m bench.coach.heldout_12b` | `tables/coaching_azure_yardstick.json`, `tables/coaching_12b_heldout.json` |
| coaching feature claims | `python -m bench.coach.dump_r4 --sets so762-test,l2arctic-test,epadb-test,ompal-test`, `python -m bench.coach.fit_r4 eval` | `tables/coaching_r4_heldout.json` |
| contrast audit on FLEURS test | `python -m bench.audit.contrast_audit run --split test` then `report --split test` | `tables/contrast_audit_test.json` |
| Japanese length (shipped runtime) | `python -m bench.length.heldout_runtime --lang ja` | `tables/length_runtime_ja.json` |
| acoustic-core latency | `python scripts/bench_latency.py --threads 4` | stdout |
| the scoreboard | `python -m bench.scoreboard` | `results/SCOREBOARD.md` |

Our outputs are cached per clip; delete or move `~/.cache/pronunciada/calib/api/` (and the
`dump-*-test.pkl` files under `~/.cache/pronunciada/bench/coach/`) before re-running after a code
change: those two caches are keyed by name, not by code version. cv-real and OMPAL caches are
keyed by a hash of the package and refresh themselves. With 3 workers the whole set takes about
an hour on an M3 Pro laptop.

## How it works, briefly

1. The prompt is converted to the sounds a native speaker would produce (dictionaries plus rules
   per language, including tone changes in Mandarin and liaison in French).
2. A multilingual phone-recognition model (ZIPA) listens to the recording; a small per-language
   adapter corrects its habits for that language.
3. For every expected sound, the scorer compares "the speaker said this" with "the speaker said
   something else" and "the speaker skipped it", without forcing a rigid alignment.
4. Small calibration models trained on human-rated speech turn that evidence into probabilities
   for sounds, words and the sentence; separate small models judge tone, stress and length.

Design and what ships: [docs/DESIGN.md](docs/DESIGN.md). Per-round write-ups: [results/](results/).

## Licence

Code: Apache-2.0. The acoustic model is CC BY 4.0. The Italian, Portuguese and Russian
pronunciation files are derived from Wiktionary and stay under CC BY-SA
(`pronunciada/data/lex/LICENSE.txt`). Full audit and open questions:
[docs/LICENSING.md](docs/LICENSING.md); attribution: [NOTICE](NOTICE).
