# Pronunciada

[![PyPI](https://img.shields.io/pypi/v/pronunciada)](https://pypi.org/project/pronunciada/)
[![CI](https://github.com/moglearn/pronunciada/actions/workflows/ci.yml/badge.svg)](https://github.com/moglearn/pronunciada/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Pronunciada is a Python library for pronunciation assessment. It takes a recording and the text the speaker was reading, and returns the probability (0 to 1) that a human rater would accept each phoneme, each word and the sentence as a whole.

It covers English, Mandarin, Japanese, Spanish, French, German, Italian, Portuguese, Russian and Korean. Inference runs on CPU with ONNX Runtime and has no PyTorch dependency. The acoustic model is [ZIPA](https://github.com/lingjzhu/zipa), a multilingual phone recognizer with 64M parameters (248 MB, or 67 MB quantized). Scoring a 3.5-second recording takes about 50 ms on an Apple M3 Pro.

On the public learner corpora with expert annotation (English and Mandarin), its scores agree with the annotators more closely than Microsoft Azure Pronunciation Assessment for phonemes, sentence accuracy, Mandarin tones and mispronunciation detection, and less closely for English sentence prosody. The numbers are under [Benchmarks](#benchmarks).

## Quick start

```bash
pip install pronunciada
pronunciada check clip.wav "The ball is in their court now" --lang en
```

```text
✗ Try again  sentence 48% · accuracy 6.7/10 · fluency 7.4/10   en · strictness 0.5

  The ball is in their court now
      ^^^^             ^^^^^ ^^^

  ball   35%  b 84%  [ɔ 38%]  [l̴ 36%]
  court  17%  k 90%  [ɔ 60%]  [ɹ→ŋ 40%]  [t 26%]
  now    35%  [n 26%]  aʊ 66%

  → Don't let the air flow around the sides of the tongue for /n/ in “now”; it sounded like an l.
  → Keep the air out of your nose for /ɹ/ in “court”; it sounded like /ŋ/.
  → Listen to /l̴/ in “ball” and try it again; it did not sound quite right.
```

This is the output for [`site/samples/en-learner-1.mp3`](site/samples/en-learner-1.mp3), a recording from the speechocean762 test set. The corpus annotators marked the vowels of "ball" and "court" as mispronounced. `ɹ→ŋ` means /ɹ/ was expected and /ŋ/ was recognized.

The same call from Python:

```python
import pronunciada

result = pronunciada.score("clip.wav", "The ball is in their court now", lang="en")

print(f"{result.prob:.2f}")
# 0.48
for word in result.words:
    print(word.text, f"{word.prob:.2f}")
# The 0.88
# ball 0.35
# is 0.92
# in 0.96
# their 0.93
# court 0.17
# now 0.35
for phone in result.words[5].phones:
    print(phone.ipa, f"{phone.prob:.2f}", phone.heard)
# k 0.90 None
# ɔ 0.60 None
# ɹ 0.40 ŋ
# t 0.26 t
```

The model is downloaded to `~/.cache/pronunciada` on first use.

## Benchmarks

All comparisons are with the Azure Pronunciation Assessment REST API (region `westus`, November 2026, one request per recording). Both systems received the same audio and reference text. Bold marks the better system where the difference is significant (paired bootstrap over speakers, p < 0.05). Confidence intervals and every other table are in [results/SCOREBOARD.md](results/SCOREBOARD.md).

### English: agreement with expert scores

speechocean762 test set: 2,500 recordings from 125 Mandarin-speaking learners, each scored by five experts. Pearson correlation between system score and expert score.

| Level | Pronunciada | Azure |
| --- | --- | --- |
| Phoneme | **0.685** | 0.538 |
| Word | 0.609 | 0.589 |
| Sentence accuracy | **0.748** | 0.673 |
| Sentence fluency | 0.779 | 0.708 |
| Sentence prosody | 0.772 | **0.827** |
| Sentence total | 0.762 | 0.796 |

### English: mispronunciation detection

Area under the ROC curve for separating phonemes the annotators marked as mispronounced from the rest. Neither corpus was used to train or tune any part of Pronunciada.

| Corpus | Pronunciada | Azure |
| --- | --- | --- |
| L2-ARCTIC test (speakers of six first languages) | **0.712** | 0.631 |
| EpaDB test (Spanish-speaking learners) | **0.721** | 0.587 |

On speechocean762 test (47,041 phonemes, 18.6% marked as errors), with both systems thresholded to flag the same 36% of the marked phonemes, 65.9% of Pronunciada's flags are phonemes the experts marked, against 44.0% of Azure's.

### Mandarin

OMPAL test set: 21 French-speaking learners of Taiwan Mandarin, every character marked by teachers. Word rows are the area under the ROC curve for characters marked wrong. Sentence rows are Pearson correlation with the mean teacher score.

| Measure | Pronunciada | Azure |
| --- | --- | --- |
| Word, any error | **0.792** | 0.704 |
| Word, tone error only | **0.778** | 0.679 |
| Word, consonant or vowel error | 0.856 | 0.822 |
| Sentence accuracy | **0.634** | 0.363 |
| Sentence fluency | **0.590** | 0.343 |

### All ten languages

We found no public learner corpus with expert annotation for the other eight languages. The table below is a proxy test on native speech: 150 FLEURS test recordings per language, scored against a reference in which one word has been replaced by a word that differs by a single phoneme. The figure is the area under the ROC curve for separating the replaced word from the unchanged words.

| Language | Pronunciada | Azure |
| --- | --- | --- |
| English | **0.981** | 0.963 |
| Mandarin | 0.975 | 0.966 |
| Japanese | **0.968** | 0.919 |
| Spanish | **0.990** | 0.942 |
| French | **0.986** | 0.937 |
| German | **0.960** | 0.887 |
| Italian | 0.979 | 0.970 |
| Portuguese | **0.963** | 0.938 |
| Russian | 0.900 | 0.909 |
| Korean | 0.944 | 0.937 |

This test measures whether a system detects a mismatch between audio and text. It does not measure accuracy on learner speech.

Across all 84 comparisons in the scoreboard, Pronunciada is ahead in 60 (40 significant) and behind in 24 (7 significant). The seven are English sentence prosody, the native false-alarm rate in English, Japanese, Spanish and Portuguese, and two tests on Common Voice recordings (English native versus non-native speakers, Japanese accepted versus rejected clips).

Azure was not tuned on any of this data. Pronunciada's calibration models were fitted on speechocean762 train, FLEURS train and validation, and the OMPAL development speakers. Every figure above is from a held-out split.

### Latency

Median time for one `Scorer.score` call on 100 speechocean762 recordings (mean length 3.5 s), Apple M3 Pro, CPU only.

| Model | 4 threads | 1 thread |
| --- | --- | --- |
| fp32 (248 MB) | 50 ms | 100 ms |
| int8 (67 MB) | 40 ms | 65 ms |

## Supported languages

| Language | Code | Install | Also scored |
| --- | --- | --- | --- |
| English | `en` | `pronunciada` | word stress |
| Mandarin, Simplified and Traditional | `zh` | `pronunciada[zh]` | tone of each syllable, aspiration |
| Japanese | `ja` | `pronunciada[ja]` | vowel and consonant length |
| Spanish | `es` | `pronunciada` | word stress |
| French | `fr` | `pronunciada` | |
| German | `de` | `pronunciada` | word stress |
| Italian | `it` | `pronunciada` | word stress |
| Portuguese (Brazil) | `pt` | `pronunciada` | word stress |
| Russian | `ru` | `pronunciada` | word stress, palatalization |
| Korean | `ko` | `pronunciada` | aspirated and tense consonants |

`pip install "pronunciada[all]"` installs every language. The Japanese extra adds about 300 MB of dictionaries.

## Usage

### Python

Create a `Scorer` once when scoring more than one recording:

```python
from pronunciada import Scorer

scorer = Scorer("zh")                    # or Scorer("en", precision="int8")
result = scorer.score("clip.wav", "我家有四口人")

for word in result.words:
    for syllable in word.syllables:
        print(syllable.text, syllable.tone_expected, syllable.tone_heard, syllable.tone_prob)
```

`strictness` (0 to 1, default 0.5) moves every probability between a lenient and a strict calibration. `scorer.rescore(result, strictness=0.8)` applies a new value without running the acoustic model again.

`result.coaching()` returns the detected errors as dictionaries, ordered by severity. Each has the word, the expected phoneme, the phoneme recognized when one can be named, and the articulatory features that differ:

```python
for item in result.coaching():
    print(item["kind"], item["word"], item["expected_ipa"], item["heard_ipa"], item["diff_keys"])
# substitution now n None ['lateral']
# substitution court ɹ ŋ ['place', 'manner']
# ...
```

`pronunciada.tips.tips()` turns these into the sentences shown in the command-line report. Each tip has a stable key so an application can substitute its own wording or translation. The schema is in [docs/COACHING.md](docs/COACHING.md).

The result also carries timestamps for every phoneme and word, fluency and completeness scores, omitted and inserted words, and warnings when the recording is clipped, noisy, very quiet or mostly silence.

### Command line

```bash
pronunciada check clip.wav "我家有四口人" --lang zh
pronunciada check clip.wav "Buenos días" --lang es --json
pronunciada langs
```

Options are listed in [docs/CLI.md](docs/CLI.md).

### HTTP server

```bash
pronunciada serve --port 8765
curl -s -F audio=@clip.wav --form-string "text=Good morning" --form-string lang=en http://127.0.0.1:8765/v1/score
```

The server is for local use. It has no authentication. See [docs/SERVER.md](docs/SERVER.md).

### Azure-compatible output

`to_azure` converts a result to the JSON layout returned by Azure Pronunciation Assessment. `pronunciada serve --azure-compat` accepts Azure's short-audio REST request and answers in the same layout.

```python
from pronunciada.compat.azure import to_azure

payload = to_azure(pronunciada.score("clip.wav", "Good morning", lang="en"))
payload["NBest"][0]["PronunciationAssessment"]["AccuracyScore"]
```

The field names match. The score distributions do not, so thresholds chosen for Azure need to be chosen again. Field-by-field notes are in [docs/AZURE_COMPAT.md](docs/AZURE_COMPAT.md).

### MCP server

```bash
pip install "pronunciada[mcp]"
pronunciada mcp
```

This starts a Model Context Protocol server on stdio with one tool, `assess_pronunciation(audio_path, reference_text, language)`. Client configuration is in [docs/MCP.md](docs/MCP.md).

## How it works

1. The reference text is converted to a phoneme sequence with per-language dictionaries and rules. This includes Mandarin tone sandhi and French liaison.
2. ZIPA produces frame-level posteriors over IPA phones from the audio alone. A small per-language adapter corrects its systematic confusions.
3. For each expected phoneme, the scorer compares the likelihood that it was produced with the likelihood of a substitution or a deletion. Each likelihood is a CTC forward score summed over all alignments, so the result does not depend on a single forced alignment.
4. Gradient-boosted calibrators fitted on expert-scored speech map this evidence to acceptance probabilities for phonemes, words and sentences.
5. Tone, stress, length and a few consonant contrasts are not recoverable from the phone posteriors. Separate small classifiers read ZIPA's intermediate layers for these.

All models trained for this project together add about 10 MB to the package. The design is described in [docs/DESIGN.md](docs/DESIGN.md).

## Limitations

- Accuracy on learner speech has been measured for English and Mandarin only. The other eight languages are evaluated on native speech with altered reference text and on Common Voice recordings accepted or rejected by volunteer validators.
- English sentence prosody correlates less well with expert scores than Azure's (0.772 against 0.827).
- At the default strictness, about 3% of a native speaker's words score below 0.6, against about 2% for Azure, in English, Japanese, Spanish and Portuguese.
- In Mandarin, a low tone probability on one syllable does not always lower the word probability. Applications that grade tones should read `syllable.tone_prob` directly.
- Coaching tips were tuned on English and Mandarin learners. On native FLEURS recordings, between 9% and 43% of sentences (depending on language) still receive at least one pronunciation tip.
- Italian geminate consonants and Portuguese nasal vowels are detected poorly. Japanese pitch accent is not scored. Details are in [results/contrast_audit.md](results/contrast_audit.md).
- The reference text is required. For spontaneous speech, transcribe first and pass the transcript.
- There are no iOS, Android or WebAssembly builds.

## Reproducing the benchmarks

```bash
pip install -e ".[all,bench]"
python -m bench.azure_pa run speechocean762 --split test   # needs AZURE_SPEECH_KEY and AZURE_SPEECH_REGION
python -m bench.eval_english      # speechocean762, L2-ARCTIC, EpaDB
python -m bench.eval_languages    # FLEURS tests in ten languages
python -m bench.scoreboard        # regenerates results/SCOREBOARD.md
```

The corpora are public. Azure's responses are not redistributed, so the comparison needs an Azure Speech resource. `bench.azure_pa` sends each recording once and caches the response under `~/.cache/pronunciada/azure`. Download instructions, the remaining commands and run times are in [docs/benchmarks_data.md](docs/benchmarks_data.md).

## Development

Pronunciada was written at [Moglearn](https://moglearn.com) for a language-learning application. No large model was trained. The system combines an existing phone recognizer with lattice-based scoring, per-language phonology and small calibration models.

Development ran in rounds. Each round set a target in advance, selected changes on development data and was evaluated once on held-out data. Most of the implementation was done by an AI coding agent under human direction. The write-ups for all rounds are in [results/](results/), including approaches that were tried and dropped: Japanese pitch accent, a second acoustic model, and training on synthetic speech.

## License

The code is licensed under Apache-2.0. ZIPA is licensed under CC BY 4.0 and is downloaded at first use. The Italian, Portuguese and Russian pronunciation dictionaries are derived from Wiktionary and remain under CC BY-SA. [docs/LICENSING.md](docs/LICENSING.md) and [NOTICE](NOTICE) list every data file with the corpus it was fitted on and its license.

Azure is a trademark of Microsoft. This project is not affiliated with or endorsed by Microsoft.

## Citation

See [CITATION.cff](CITATION.cff).

## Acknowledgements

This project depends on [ZIPA](https://github.com/lingjzhu/zipa) (Zhu et al., ACL 2025) and was evaluated on speechocean762, OMPAL, FLEURS, Common Voice, L2-ARCTIC, EpaDB and AISHELL.
