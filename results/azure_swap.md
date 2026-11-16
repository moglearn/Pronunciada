# Azure Pronunciation Assessment on fleurs-swap (text-level planted errors, 10 languages)

Run: 2026-11-16, Azure region westus, REST short-audio API, settings from `bench/azure_pa.py`
(`pa_params`: HundredMark, Phoneme granularity, Comprehensive, EnableMiscue=True; en-US also gets
prosody and IPA). The numbers below are measured. Nothing is extrapolated.

## What was tested

Native FLEURS test audio is scored twice: once against its own transcript (`orig`) and once
against a transcript with ONE word replaced. Azure takes only a reference text, so this is the
only way to plant an error. A good scorer gives the replaced word a low score and leaves the
other words alone. Generator: `bench/textswap.py` + `bench/data/fleurs_swap.py`. Evaluation:
`bench/eval_swap.py`. Tests: `tests/test_bench_textswap.py` (18 pass).

* Sources: per language, the first 150 FLEURS test utterances sorted by id with dur ≤ 20 s and
  only OOV-type front-end warnings (zh/ja/ko: no digits or Latin letters). 150 × 10 recordings,
  4.75 h of distinct audio. FLEURS has several speakers per sentence, so the 150 recordings
  cover 119–130 distinct sentences per language. Each recording gets its own seeded swap.
* `swap_far`: a FLEURS train+test vocabulary word (frequency ≥ 2, OOV words and mostly-
  capitalised words excluded), phone count within ±2, phone edit distance ≥ 60 % of the
  longer word. zh: one hanzi of a word is replaced.
* `swap_near`: a word exactly one phone substitution away (same phone count) according to our
  front-end. FLEURS vocabulary is tried first at every position, then the shipped lexicons
  (ko: a one-jamo change). zh: a hanzi with the same tone whose syllable differs by one phone.
* `tone` (zh): a hanzi with the same syllable and a different tone. The original and new
  positions have surface tone = lexical tone (no sandhi) and are not neutral tone. 一 and 不
  are never used.
* Every swap is verified by running the front-end on the whole new sentence. All other units
  keep their text, phones and (zh) tones, and the replaced unit has the intended phones and
  tone. The first and last units are never replaced.
* Units: whitespace tokens with punctuation stripped (en es fr de it pt ru ko); RefWords
  (zh ja). For zh the scored unit is the RefWord that contains the replaced hanzi.
* Manifest: `~/.cache/pronunciada/data/fleurs-swap/manifest.jsonl`, 4,650 records, 6.4 MB.
  Rebuilding it gives a byte-identical file (same md5).

## Coverage and requests

| lang | orig | swap_far | swap_near | tone | near from FLEURS / lexicon / jamo | median phone dist far / near | requests |
|---|---|---|---|---|---|---|---|
| en | 150 | 150 | 150 | – | 150 / 0 / – | 5 / 1 | 450 |
| zh | 150 | 150 | 150 | 150 | 150 / – / – | 3 / 1 (syllable) | 600 |
| ja | 150 | 150 | 150 | – | 150 / – / – | 5 / 1 | 450 |
| es | 150 | 150 | 150 | – | 146 / 4 / – | 6 / 1 | 450 |
| fr | 150 | 150 | 150 | – | 147 / 3 / – | 5 / 1 | 450 |
| de | 150 | 150 | 150 | – | 149 / 1 / – | 6 / 1 | 450 |
| it | 150 | 150 | 150 | – | 150 / 0 / – | 6 / 1 | 450 |
| pt | 150 | 150 | 150 | – | 148 / 2 / – | 5 / 1 | 450 |
| ru | 150 | 150 | 150 | – | 147 / 3 / – | 6 / 1 | 450 |
| ko | 150 | 150 | 150 | – | 144 / – / 6 | 6 / 1 | 450 |

Every source utterance got a near swap, so near-swap coverage is 100 % in all 10 languages.
Lexicon and jamo fallbacks were used only in these cases: es luchadores→luchadoras,
rindieron→rindieren, trasladaron→trasladaran, cesio→cesé; fr réputation→réfutation,
allumé→assumé, autorisera→autoriserait; de Antwort→endwort; pt lado→lago, rifle→refle;
ru ионами→гонами, потушили→потушила, стили→стеги; ko (jamo) 소문을→조문을, 콘서트의→곤서트의,
독려하고→톡려하고, 피우고→비우고, 파티마→바티마, 남아있는→맘아있는.

Requests: the dry run (`python -m bench.azure_pa run fleurs-swap --split test --dry-run`)
showed 4,650 requests and 14.70 audio-hours (52,916 s). All 4,650 were sent: 16 in a smoke
test and 4,634 in the main run (`python -m bench.eval_swap run-azure --workers 4`, 2,673 s).
There were 0 failures and 0 retries, and every response had RecognitionStatus Success. The
raw responses are cached in `~/.cache/pronunciada/azure/fleurs-swap/` (87 MB).

## Results (Azure)

Values are point estimates followed by [95 % CI]. The CIs come from a 1,000-resample bootstrap
clustered by source recording, so a recording's orig and swapped versions are resampled
together. Unit score = Azure word AccuracyScore/100, and Omission = 0. "hit@x%FA" uses a
threshold set on the orig records so that exactly x % of native units are flagged (Azure's
integer scores are tied a lot, so ties at the threshold are split fractionally). "right word"
is the share of swapped records where the replaced unit has the lowest score (a tie for the
lowest shares the credit). "sent AUC" compares the swapped records with their own orig
records using the sentence AccuracyScore or PronScore. "excluded" counts swapped records where
Azure's text normaliser rewrote the planted word (so it was never scored) and records Azure
failed to assess.

### azure: swap_far

| lang | n | AUC | hit@5%FA % | hit@2%FA % | hit<60 % | hit<80 % | right word % | sent AUC (Acc) | sent AUC (Pron) | excluded (rewritten / failed) |
|---|---|---|---|---|---|---|---|---|---|---|
| de | 150 | 0.982 [0.966, 0.993] | 95.3 [91.9, 98.4] | 93.3 [88.7, 96.7] | 95.3 | 97.3 | 92.1 [87.7, 95.6] | 0.820 [0.789, 0.853] | 0.838 [0.805, 0.872] | 0 / 0 |
| en | 150 | 0.994 [0.991, 0.996] | 100.0 [100.0, 100.0] | 98.7 [95.5, 100.0] | 98.7 | 99.3 | 89.8 [85.6, 93.6] | 0.884 [0.854, 0.914] | 0.842 [0.809, 0.873] | 0 / 0 |
| es | 150 | 0.993 [0.986, 0.997] | 98.7 [96.7, 100.0] | 95.3 [91.3, 98.6] | 95.3 | 99.3 | 91.0 [86.7, 94.7] | 0.891 [0.861, 0.920] | 0.902 [0.868, 0.931] | 0 / 0 |
| fr | 150 | 0.986 [0.975, 0.993] | 98.0 [95.3, 100.0] | 91.3 [86.6, 96.7] | 91.3 | 98.0 | 82.3 [76.8, 87.6] | 0.833 [0.798, 0.871] | 0.861 [0.826, 0.896] | 0 / 0 |
| it | 150 | 0.990 [0.981, 0.996] | 96.0 [92.7, 98.7] | 95.5 [92.0, 98.7] | 96.0 | 96.7 | 92.9 [88.9, 96.3] | 0.838 [0.811, 0.873] | 0.856 [0.828, 0.890] | 0 / 0 |
| ja | 149 | 0.979 [0.962, 0.992] | 95.3 [91.3, 98.7] | 93.3 [88.6, 97.3] | 93.3 | 95.3 | 87.5 [82.9, 92.0] | 0.839 [0.808, 0.873] | 0.863 [0.834, 0.892] | 1 / 0 |
| ko | 150 | 0.995 [0.990, 0.998] | 99.3 [98.0, 100.0] | 99.3 [97.3, 100.0] | 99.3 | 100.0 | 97.3 [94.9, 99.2] | 0.873 [0.843, 0.906] | 0.879 [0.845, 0.914] | 0 / 0 |
| pt | 150 | 0.991 [0.986, 0.996] | 94.7 [91.2, 98.0] | 93.4 [88.7, 97.3] | 94.0 | 98.7 | 90.6 [86.0, 94.3] | 0.875 [0.841, 0.908] | 0.878 [0.844, 0.909] | 0 / 0 |
| ru | 150 | 0.982 [0.978, 0.987] | 100.0 [100.0, 100.0] | 77.0 [59.1, 98.0] | 100.0 | 100.0 | 76.1 [71.2, 80.8] | 0.845 [0.813, 0.877] | 0.854 [0.822, 0.886] | 0 / 0 |
| zh | 150 | 0.989 [0.977, 0.996] | 98.0 [96.0, 100.0] | 96.0 [93.3, 99.3] | 98.0 | 99.3 | 89.9 [85.4, 93.7] | 0.831 [0.802, 0.864] | 0.835 [0.804, 0.868] | 0 / 0 |

### azure: swap_near

| lang | n | AUC | hit@5%FA % | hit@2%FA % | hit<60 % | hit<80 % | right word % | sent AUC (Acc) | sent AUC (Pron) | excluded (rewritten / failed) |
|---|---|---|---|---|---|---|---|---|---|---|
| de | 150 | 0.887 [0.855, 0.916] | 51.7 [43.3, 59.6] | 44.6 [35.9, 51.8] | 51.3 | 73.3 | 46.0 [38.0, 53.7] | 0.678 [0.651, 0.709] | 0.675 [0.649, 0.706] | 0 / 0 |
| en | 150 | 0.963 [0.952, 0.972] | 79.3 [69.7, 87.3] | 49.3 [38.1, 61.3] | 50.0 | 64.0 | 68.6 [61.0, 75.1] | 0.748 [0.716, 0.784] | 0.673 [0.649, 0.704] | 0 / 0 |
| es | 150 | 0.942 [0.927, 0.954] | 67.9 [60.6, 75.5] | 50.2 [39.7, 60.0] | 46.0 | 74.7 | 56.3 [48.7, 64.0] | 0.703 [0.677, 0.735] | 0.689 [0.660, 0.723] | 0 / 0 |
| fr | 150 | 0.937 [0.919, 0.952] | 64.7 [56.3, 76.0] | 40.4 [31.8, 53.1] | 40.7 | 74.0 | 52.3 [45.0, 59.7] | 0.673 [0.641, 0.709] | 0.677 [0.647, 0.712] | 0 / 0 |
| it | 150 | 0.970 [0.954, 0.983] | 91.3 [86.7, 95.3] | 76.3 [68.3, 86.4] | 88.7 | 94.0 | 74.0 [66.7, 80.7] | 0.729 [0.701, 0.764] | 0.728 [0.699, 0.766] | 0 / 0 |
| ja | 150 | 0.919 [0.897, 0.939] | 68.0 [59.4, 75.2] | 57.7 [47.8, 66.7] | 58.0 | 84.0 | 53.9 [46.3, 61.3] | 0.708 [0.674, 0.746] | 0.710 [0.677, 0.746] | 0 / 0 |
| ko | 150 | 0.937 [0.916, 0.956] | 70.8 [60.3, 81.9] | 45.1 [33.7, 59.2] | 67.3 | 83.3 | 69.9 [62.6, 77.0] | 0.725 [0.693, 0.762] | 0.729 [0.694, 0.767] | 0 / 0 |
| pt | 150 | 0.938 [0.920, 0.954] | 72.4 [64.4, 79.5] | 49.7 [38.9, 58.9] | 50.0 | 84.0 | 59.0 [51.0, 66.3] | 0.711 [0.680, 0.746] | 0.693 [0.668, 0.725] | 0 / 0 |
| ru | 150 | 0.909 [0.892, 0.925] | 34.2 [23.3, 45.8] | 12.1 [7.3, 18.4] | 38.7 | 70.0 | 37.3 [30.2, 44.7] | 0.649 [0.626, 0.679] | 0.644 [0.618, 0.677] | 0 / 0 |
| zh | 150 | 0.966 [0.949, 0.980] | 89.6 [83.3, 94.7] | 76.7 [67.0, 84.0] | 82.7 | 95.3 | 74.7 [67.7, 81.0] | 0.774 [0.745, 0.807] | 0.778 [0.745, 0.814] | 0 / 0 |

### azure: tone

| lang | n | AUC | hit@5%FA % | hit@2%FA % | hit<60 % | hit<80 % | right word % | sent AUC (Acc) | sent AUC (Pron) | excluded (rewritten / failed) |
|---|---|---|---|---|---|---|---|---|---|---|
| zh | 150 | 0.846 [0.815, 0.875] | 40.2 [32.7, 51.4] | 21.3 [13.3, 29.5] | 35.3 | 60.7 | 38.7 [31.3, 46.0] | 0.598 [0.571, 0.626] | 0.604 [0.575, 0.633] | 0 / 0 |

### azure: native (orig) units

| lang | orig records used | units | FA <60 % | FA <80 % | mean | median | unmapped units % | failed |
|---|---|---|---|---|---|---|---|---|
| de | 150 | 3125 | 4.9 [4.1, 5.8] | 14.5 [13.2, 16.0] | 0.904 | 0.940 | 0.00 | 0 |
| en | 150 | 3100 | 2.2 [1.6, 2.9] | 3.0 [2.4, 3.7] | 0.967 | 1.000 | 0.00 | 0 |
| es | 150 | 3693 | 1.8 [1.4, 2.2] | 6.6 [5.8, 7.4] | 0.919 | 0.940 | 0.00 | 0 |
| fr | 150 | 3764 | 2.0 [1.5, 2.6] | 5.9 [5.1, 6.8] | 0.934 | 0.970 | 0.00 | 0 |
| it | 150 | 3413 | 3.7 [2.9, 4.5] | 15.0 [13.7, 16.4] | 0.894 | 0.940 | 0.00 | 0 |
| ja | 150 | 3823 | 2.0 [1.6, 2.5] | 15.4 [14.2, 16.6] | 0.881 | 0.910 | 0.00 | 0 |
| ko | 150 | 2050 | 4.5 [3.4, 5.8] | 8.3 [6.9, 9.9] | 0.941 | 0.980 | 0.00 | 0 |
| pt | 150 | 3379 | 2.0 [1.5, 2.6] | 9.7 [8.8, 10.8] | 0.911 | 0.940 | 0.03 | 0 |
| ru | 150 | 2751 | 5.4 [4.4, 6.5] | 9.2 [7.9, 10.6] | 0.927 | 1.000 | 0.00 | 0 |
| zh | 150 | 2713 | 3.6 [2.9, 4.5] | 12.8 [11.4, 14.3] | 0.927 | 0.970 | 0.00 | 0 |

## Latency (for the later speed comparison)

These are wall-clock times of the successful POST, measured from this Mac to westus, so they
include upload and network. Four requests ran in parallel. Each request is one whole clip
(mean audio 9.4–13.3 s).

| locale | n | median s | p90 s | mean s | median latency / audio duration |
|---|---|---|---|---|---|
| de-DE | 450 | 2.49 | 3.55 | 2.62 | 0.22 |
| en-US | 450 | 2.34 | 3.58 | 2.43 | 0.25 |
| es-MX | 450 | 1.94 | 2.99 | 2.06 | 0.18 |
| fr-FR | 450 | 2.18 | 3.22 | 2.23 | 0.22 |
| it-IT | 450 | 2.95 | 4.08 | 3.03 | 0.22 |
| ja-JP | 450 | 2.22 | 3.01 | 2.24 | 0.19 |
| ko-KR | 450 | 1.82 | 2.99 | 1.99 | 0.16 |
| pt-BR | 450 | 2.17 | 3.16 | 2.23 | 0.18 |
| ru-RU | 450 | 2.25 | 3.26 | 2.37 | 0.21 |
| zh-CN | 600 | 1.82 | 2.86 | 1.94 | 0.18 |

## Caveats and observations

* **Normaliser rewrites.** One swapped record was excluded: ja swap_far 吐き出す→国々. Azure never
  returns 々, so the planted word could not be placed on the text. The same 々 issue leaves
  that character unmapped in 7 other ja sentences, and pt "Platô" is unmapped in one sentence.
  Unmapped units are below 0.03 % in every language. No other swap was rewritten. Far swaps
  are often ungrammatical (e.g. en "could achieved behind"). That is intended, and Azure
  scored those words as written.
* **Azure "hears" the original word.** This is the intended test. In ko and ja, Azure often
  reports the reference word as Omission (scored 0) and inserts the word it heard, e.g. ko
  보이지 → Omission + Insertion 이란의. Insertions are ignored for unit scores.
* **ru native zeros.** ru-RU gives accuracy 0 (as Mispronunciation) to many short function
  words in native speech (и, в, с, года …). As a result 2.5 % of native ru units score 0 (en
  0.7 %, others 0.2–0.9 %), the 2 %-FA threshold falls inside the tie at 0, and ru
  hit@2%FA for swap_far has a wide CI (77 % [59, 98]). The ru sources also contain Latin
  tokens ("ASUS Eee PC", "pH"), because the digit/Latin filter applies only to zh/ja/ko as
  the protocol specifies. Azure scores those tokens 0 or omits them.
* **Word granularity in zh/ja.** Azure segments zh/ja with its own word list. Our unit takes
  the minimum over every Azure word that overlaps it, so a low score on an Azure word that
  spans two of our units (e.g. ja 敵対的 vs our 敵対 + 的) lands on both. This lowers
  "right word" and AUC slightly for zh/ja. Our system scores our own units, so it does not
  pay this cost.
* **Near swaps.** "One phone" means one substitution in our front-end's phone string. Some
  pairs look bigger in the spelling (ko 경우에는→경제는, ru vowel reduction каждая→каждого).
  Many are inflectional neighbours (it ponte→ponti, pt muitas→muitos), which a native speaker
  could plausibly have said, so swap_near is the hard condition. de/fr/ru also have high
  native FA at < 80 (fr 5.9 %, de 14.5 %, ru 9.2 %), which hurts their near-swap hit rates at
  a fixed FA.
* **ru near swaps (round 8).** The tables above are Azure's and do not change. Pronunciada's
  ru near-swap AUC went 0.872 → 0.887 [0.860, 0.914] against Azure's 0.909 [0.892, 0.925]. Its
  hit@5 % FA went 0.580 → 0.607 (Azure 0.342). Replaced words scoring > 0.9: 36 → 32 of 150
  (Azure 9). The change: measured vowel targets, a retrained adapter and sound bias +0.60.
  Round 10 added an offline stress lexicon and set the bias to 0.0. Near-swap AUC is now 0.901
  [0.874, 0.926], with 24 of 150 replaced words scoring > 0.9 and right word 0.540
  ([round8_ru.md](round8_ru.md)).
* **Word stress model `stress_v3` (round 11, en de es it pt ru).** Azure's tables above do not
  change. Pronunciada's near-swap AUC moved by +0.000 to +0.005 (de 0.956, es 0.989, it 0.976,
  pt 0.960, ru 0.902, en 0.981); no language got worse ([round11_stress.md](round11_stress.md)).
* **Tone (zh).** Same syllable with a different tone: AUC 0.846, and only 40 % of these swaps
  are caught at 5 % FA. Azure zh-CN is weak on tone-only errors, which is where we can win.
* **Statistics.** The bootstrap clusters are recordings, not sentences. 119–130 distinct
  sentences per language are read by 150 recordings, so the CIs are slightly optimistic.
  Every metric is computed on the same records for every system, so paired comparisons are
  valid.

## How to evaluate another system (pronunciada)

`bench/eval_swap.py` takes, for each record id, `{"status": "Success", "words": [(char_start,
char_end, score 0..1), ...], "sentence": {"prob": ...}}` over the record's `text`. Unit
score = min over overlapping system words. A unit with no overlapping word scores 0
(omission). Optional `"unmapped": [(a, b)]` marks regions as missing rather than wrong.

```python
from bench.data import load
from bench.eval_swap import evaluate, sentence_result_output, markdown
recs = load("fleurs-swap", "test")
outs = {}
for r in recs:
    res = scorer.score(r["audio"], r["text"], lang=r["lang"])     # pronunciada SentenceResult
    spans = [w.char_span for w in get_g2p(r["lang"])(r["text"]).words]   # one per res.words
    outs[r["id"]] = sentence_result_output(res, spans)
print(markdown(evaluate(recs, outs), "pronunciada"))
```

Alternatively, write one `{"id", "words": [[a, b, s], ...], "sentence": {...}}` per line and
run `python -m bench.eval_swap jsonl ours.jsonl --name pronunciada --tables results/tables`.
The audio is the plain FLEURS file (`r["audio"]`). The three or four records of one recording
share it, so the acoustic pass can be cached per `labels.src_id`.
