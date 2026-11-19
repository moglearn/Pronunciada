---
license: other
license_name: apache-2.0-and-cc-by-sa-4.0
license_link: https://github.com/moglearn/pronunciada/blob/main/docs/LICENSING.md
library_name: pronunciada
base_model: anyspeech/zipa-small-crctc-500k
base_model_relation: adapter
language:
  - en
  - zh
  - ja
  - es
  - fr
  - de
  - it
  - pt
  - ru
  - ko
tags:
  - pronunciation-assessment
  - mispronunciation-detection
  - speech
  - language-learning
  - onnx
  - on-device
---

# Pronunciada

On-device pronunciation assessment with cloud-level accuracy.

Pronunciada is an open-source Python library. It takes a recording and the text the speaker was reading, and returns the probability (0 to 1) that a human rater would accept each phoneme, each word and the sentence as a whole. It covers English, Mandarin, Japanese, Spanish, French, German, Italian, Portuguese, Russian and Korean.

[Website](https://pronunciada.dev) · [GitHub](https://github.com/moglearn/pronunciada) · [PyPI](https://pypi.org/project/pronunciada/) · [Benchmarks](https://github.com/moglearn/pronunciada/blob/main/results/SCOREBOARD.md)

## What is in this repository

Pronunciada uses [ZIPA](https://huggingface.co/anyspeech/zipa-small-crctc-500k) unchanged as its phone recognizer. This repository holds the small models that Pronunciada adds on top of it, about 10 MB in total:

| Folder | Contents |
| --- | --- |
| `adapters/` | One output adapter per language. It corrects ZIPA's systematic confusions for that language. |
| `calib/` | Gradient-boosted calibrators that turn the acoustic evidence into acceptance probabilities for phonemes, words and sentences. |
| `prosody/` | Mandarin tone and word stress models (ONNX). |
| `length/` | Japanese vowel length. |
| `coach/` | Small classifiers on ZIPA's intermediate layers for consonant contrasts, such as Korean tense consonants. They feed the coaching tips. |
| `native/` | Accepted native variants per language. |

The same files ship inside the pip package, so there is nothing to download from here. This copy matches version 1.0.1. ZIPA itself is downloaded from the anyspeech repository at first use.

## Use

```bash
pip install pronunciada
pronunciada check clip.wav "It was a beautiful day for a float" --lang en
```

```python
from pronunciada import Scorer

scorer = Scorer("en")
result = scorer.score("clip.wav", "It was a beautiful day for a float")
for word in result.words:
    print(word.text, round(word.prob, 2))
```

## Results

Compared with the Azure Pronunciation Assessment REST API (region `westus`, November 2026) on the same audio and reference text. Over 84 comparisons on held-out public test sets, Pronunciada is significantly better in 40, not significantly different in 37 and significantly worse in 7.

speechocean762 test set (2,500 recordings of English learners, each scored by five experts), Pearson correlation with the expert scores:

| Level | Pronunciada | Azure | ZIPA alone |
| --- | --- | --- | --- |
| Phoneme | **0.685** | 0.538 | 0.434 |
| Word | 0.609 | 0.589 | 0.394 |
| Sentence accuracy | **0.748** | 0.673 | 0.583 |
| Sentence prosody | 0.772 | **0.827** | |

Bold marks the better of Pronunciada and Azure where the difference is significant (paired bootstrap over speakers, p < 0.05). "ZIPA alone" is the unmodified model with the standard goodness-of-pronunciation score.

Scoring a 3.5-second recording takes about 50 ms on an Apple M3 Pro CPU. With the quantized ZIPA model (67 MB) the Python process uses about 0.3 GB of memory.

Every table, the confidence intervals and the tests where Azure is ahead are in the [scoreboard](https://github.com/moglearn/pronunciada/blob/main/results/SCOREBOARD.md).

## Limitations

- Accuracy on learner speech has been measured for English and Mandarin only. The other eight languages are evaluated on native speech with altered reference text and on Common Voice recordings.
- English sentence prosody correlates less well with expert scores than Azure's.
- The reference text is required.
- There are no iOS, Android or WebAssembly builds.

The full list is in the [README](https://github.com/moglearn/pronunciada#limitations).

## License

The code and most of these files are Apache-2.0. The Italian, Portuguese and Russian adapters (`adapters/it.npz`, `pt.npz`, `ru.npz`) were trained towards pronunciation dictionaries derived from Wiktionary and are CC BY-SA 4.0. ZIPA is CC BY 4.0 (Zhu et al., ACL 2025). The corpora each file was fitted on are listed in [docs/LICENSING.md](https://github.com/moglearn/pronunciada/blob/main/docs/LICENSING.md).

Azure is a trademark of Microsoft. This project is not affiliated with or endorsed by Microsoft.

## Citation

See [CITATION.cff](https://github.com/moglearn/pronunciada/blob/main/CITATION.cff).
