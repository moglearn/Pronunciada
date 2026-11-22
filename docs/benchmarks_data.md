# Benchmark data, tools and published Azure numbers

All data lives under `~/.cache/pronunciada/data/<name>/` (`$PRONUNCIADA_CACHE` overrides `~/.cache/pronunciada`).
Each dataset has `manifest.jsonl` (one JSON per utterance) written by `python -m bench.data.<name>`
(idempotent and resumable: per-chunk partial manifests in `parts/`, a chunk is done when
`parts/<chunk>.done` exists). Audio is 16 kHz mono FLAC (PCM_16).

```python
from bench.data import load
recs = load("speechocean762", split="test")          # also lang="en"
```

Manifest record: `{"id", "audio" (absolute path), "lang" (en zh ja es fr de it pt ru ko), "text",
"split" ("train"|"test"), "speaker" (or null), "dur" (s), "labels": {...}}`.

## Datasets

| dataset | lang | train n / h | test n / h | speakers (train/test) | disk | license |
|---|---|---|---|---|---|---|
| speechocean762 | en | 2500 / 2.88 | 2500 / 2.69 | 125/125 | 365 MB | Apache-2.0 |
| fleurs | 10 langs | 11500 / 36.09 | 3982 / 13.35 | -/- | 2547 MB | CC-BY-4.0 |
| l2arctic | en | 2699 / 2.79 | 900 / 0.88 | 18/6 | 284 MB | CC BY-NC 4.0 |
| epadb | en | 1903 / 1.65 | 1263 / 1.10 | 30/20 | 168 MB | ODC-By 1.0 |
| aishell3 | zh | 6037 / 5.95 | 800 / 0.71 | 87/20 | 374 MB | Apache-2.0 |
| zhtw | zh | - | 400 / 0.46 | -/298 | 25 MB | CC0-1.0 |
| **total** | | | | | **3.76 GB** | |

FLEURS per language: test 400 (ko 382) / train 1000 (zh 2500); 49.4 h total. Disk excludes the other workstream's validation parquet files.

### speechocean762 — `bench/data/speechocean762.py`
Source: HF `mispeech/speechocean762` (parquet). Apache-2.0. English read by Mandarin L1 speakers
(children and adults), 125 speakers per split, 5 expert raters. All labels kept verbatim:

| field | meaning |
|---|---|
| `labels.sentence.{accuracy,fluency,prosodic,total}` | 0–10 int |
| `labels.sentence.completeness` | 0–10 float (share of words read ×10) |
| `labels.words[i].{text,accuracy,stress,total}` | word text (upper case), 0–10 ints |
| `labels.words[i].phones` | canonical ARPAbet phones with stress digit, e.g. `"AA0"` |
| `labels.words[i].phones_accuracy` | 0–2 per phone, average of the raters in 0.2 steps (2 = correct, 1 = accented, 0 = wrong / missing) |
| `labels.words[i].mispronunciations` | `[{index, canonical_phone, pronounced_phone}]`; `pronounced_phone` is ARPAbet, `<DEL>` (deleted) or `<unk>`; present on 1,877 of 31,816 words |
| `labels.age`, `labels.gender` | int, `"m"`/`"f"` |

`speaker` = the 4-digit speaker id; `id` = `so762-<wav stem>`.

### FLEURS — `bench/data/fleurs.py`
Source: HF `google/fleurs` parquet, configs en_us, cmn_hans_cn, ja_jp, es_419, fr_fr, de_de, it_it,
pt_br, ru_ru, ko_kr. CC-BY-4.0. Only `test` and `train` are read (the `validation` split is reserved
as a dev set and never touched). Caps: test 400 / train 1000 per language (zh train 2500), in file
order, clips > 30 s skipped (ko test has only 382 clips ≤ 30 s). Remote row groups are read only
until the cap is reached. Text = `raw_transcription`. `speaker` = null (FLEURS has no speaker ids;
the same sentence is read by several speakers: `labels.fleurs_id` groups them).
`labels = {fleurs_id, transcription (normalised), gender ("male"/"female"/"other"), config}`.
Note: `~/.cache/pronunciada/data/fleurs/<config>/validation-0.parquet` belongs to another workstream.

### L2-ARCTIC (annotated subset) — `bench/data/l2arctic.py`
Source: HF mirror `chikingsley/l2-arctic-release-v5.0` (ungated; per-speaker zips read by HTTP range
requests, only annotated members). **CC BY-NC 4.0** (research only). 24 speakers × ~150 manually
annotated utterances; split `test` = NJS, TLV, TNI, TXHC, YKWK, ZHAA, else `train`.
`labels = {l1, gender, words: [{text,start,end}], phones: [{canonical, perceived, error, start, end,
word}], phones_ipa: same from the IPA tier (most files)}`; `error` ∈ null / `s` (substitution) /
`d` (deletion, perceived `sil`) / `a` (addition, canonical `sil`) / `?` (unparsed comment, kept in `raw`).

### EpaDB — `bench/data/epadb.py`
Source: CONICET repository, handle 11336/150678 (open access, no login), **ODC-By 1.0**. English read
by Argentinian Spanish L1 speakers; the corpus' own train (30 speakers) / test (20) split.
`labels = {score (annotator 1, 1–5), words: [{text,start,end}], annotation: [{phone,start,end}]
(annotator 1, ARPAbet + extensions like Th, Kh, DX, BH, RR; "0" = deleted), phones: [{ref,
annotated, correct}] (the corpus' own labels vs the best-matching reference transcription),
annotation_2 (second annotator, subset)}`. The waveform zip is downloaded once and deleted after
extraction. (The HF mirror `KoelLabs/EpaDB` is gated and was not used.)

### AISHELL-3 subset — `bench/data/aishell3.py`
Source: HF `AISHELL/AISHELL-3` (individual wav files; Apache-2.0). Native Mandarin, per-character
pinyin with tone digits from the corpus' `content.txt`.
**Deviation from the requested split:** the HF repo holds only 11,750 of the 88,035 wavs — 100
speakers under `train/` and 100 under `test/`, of which only 7 are test-only speakers (93 also occur in
the corpus' train list). ≥ 100 train speakers plus ≥ 20 different test speakers is therefore impossible
(107 distinct speakers). Used: train ≈ 6000 utterances from 87 speakers; test 800 utterances (40 each)
from 20 speakers = the 7 test-only + 13 held out from `train/`; no speaker is in both splits.
`labels = {hanzi: [...], pinyin: ["guang3", ...] (1–5, 5 = neutral, "v" = ü), tones: [int],
age_group (A <14, B 14–25, C 26–40, D >41), gender, accent}`. The pinyin is the corpus' own annotation
(not re-checked here) and contains occasional errors.

### zh-TW (Common Voice 25.0, test) — `bench/data/zhtw.py`
Source: HF `OpenFormosa/common_voice_25_zh-TW` (mirror of Mozilla Common Voice 25.0 zh-TW; CC0-1.0).
400 utterances from the official test split, ≤ 8 per contributor (298 contributors), Traditional-script
prompts. `lang = "zh"`, `speaker = client_id`, `labels = {client_id, sentence_id, up_votes, down_votes,
age, gender, accents, variant, locale}`. Volunteer recordings with mixed microphones; accents are
self-reported.

### Built later (see their own write-ups)
- **fleurs-swap** (`bench/data/fleurs_swap.py`, `bench/textswap.py`): FLEURS test audio with one
  word of the prompt replaced (near / far / tone variants), 150 recordings per language; the swap
  benchmark of `results/azure_swap.md` and `bench/eval_swap.py`.
- **cv-real** (`bench/cv_build.py`, record lists in `bench/data/cv_real/`): Common Voice 22.0 (CC0)
  clips accepted vs rejected by reviewers (Test A) and native vs self-declared non-native speakers
  (Test B), split by speaker into a dev half (tuning) and a test half (benchmark);
  `results/cv_real.md`.
- **OMPAL** (`bench/ompal_build.py`): see below and `results/ompal.md`.

### Other learner corpora
| corpus | language | status |
|---|---|---|
| LATIC (IEEE DataPort, DOI 10.21227/mqtj-qh10) | Mandarin by 4 learners (Russian, Korean, French, Arabic L1), 2,579 utts | files behind an IEEE DataPort login — skipped |
| OMPAL (Interspeech 2025, Hsieh et al.) | Mandarin by French L1 learners, 1,768 utts, word/sentence scores | obtained later (CC BY 4.0, github.com/phantomhsieh/OMPAL-corpus): `python -m bench.ompal_build --clone`; dev half for tuning, test half for the benchmark (results/ompal.md) |
| Heroico / USMA (OpenSLR 39) | Spanish by 10 USMA learners + native speakers, same prompts | learner suite (round 19): `python -m bench.learners.heroico`; results/round19_learners.md |
| Speech Accent Archive (GMU, OSF) | English paragraph, natives + learners of 59 L1s (test) | learner suite (round 19): `python -m bench.learners.saa`; CC BY-NC-SA 4.0, evaluation only |
| HF search (“mandarin l2”, “l2 korean”, “ifcasl”, “mispronunciation” …) | — | nothing relevant and openly downloadable for the 9 non-English languages |

## Tools

### `bench/metrics.py`
Pearson, Spearman; ROC AUC (Mann–Whitney with ties), ROC curve; PR AUC (average precision, same
definition as sklearn); precision/recall/F1 at a threshold and best-F1 threshold; EER; Brier; ECE
with equal-width or equal-mass bins and `reliability_table`; `bootstrap_ci` (item- or
cluster-resampled, e.g. by utterance or speaker); `paired_bootstrap` (one-sided p that A is not
better than B, works for higher- or lower-is-better metrics). Tests: `tests/test_bench_metrics.py`
(compared against sklearn / scipy and hand-computed values).

### `bench/corrupt.py` — controlled-error protocol
`corrupt(ref, kinds, inventory, n_errors=1, seed=0) -> Corrupted` with
`Corrupted.ref` (validated deep copy), `phone_wrong`, `syllable_flags`, `word_wrong`, `word_kinds`,
`errors` (kind, word, detail). `Inventory.from_references(refs)` builds the per-language phone
inventory and word pool.

| kind | change in the reference | simulated learner error | labeled wrong |
|---|---|---|---|
| `phone_sub_near` | phone → one of the 3 closest same-class phones | subtle substitution | that phone |
| `phone_sub_far` | phone → one from the farther half of same-class phones | gross substitution | that phone |
| `phone_ins` | a consonant inserted into a syllable (onset or coda) | learner **deleted** it | the inserted phone |
| `phone_del` | a non-nucleus phone removed (never the nucleus) | learner **inserted** a sound | word only (no reference phone exists); `detail.gap_in_syllable` |
| `word_swap` | word's syllables/phones ← a different word from the pool (same syllable count preferred) | wrong word | word + the donor phones not matched by a min-edit alignment to the original |
| `tone` | syllable tone → a tone in 1–4 ≠ current and ∉ `tone_alts` | wrong tone | syllable (`tone`) |
| `stress` | primary stress swapped with another syllable of a ≥ 2-syllable word | misplaced stress | both syllables (`stress`) |
| `pitch` | Japanese word → another Tokyo accent pattern (heiban / atamadaka / nakadaka / odaka) | wrong pitch accent | morae whose H/L changed (`pitch`) |

Replacement phones never equal the original or anything in its `alts`, and carry no `alts`.
Distances: panphon weighted feature edit distance per segment, DTW over segments for multi-segment
phones (affricates, diphthongs), cached in `~/.cache/pronunciada/bench/phone_dist.json`. Vowels are
only replaced by vowels, consonants by consonants (`pronunciada.text.ipa.is_vowel`). One error per word by
default. Tests: `tests/test_bench_corrupt.py` (hand-built en / zh / ja References).

### `bench/azure_pa.py` — Azure Pronunciation Assessment runner
`python -m bench.azure_pa run <dataset> [--split test] [--lang en] [--limit N] [--dry-run] [--rps 1]`
then `python -m bench.azure_pa parse <dataset>` → `~/.cache/pronunciada/azure/<dataset>/flat.jsonl`.
REST API for short audio, `format=detailed`, `Pronunciation-Assessment` header (base64 JSON:
HundredMark, Granularity=Phoneme, Dimension=Comprehensive, EnableMiscue; + EnableProsodyAssessment,
PhonemeAlphabet=IPA, NBestPhonemeCount=5 for en-US only). Key/region only from `AZURE_SPEECH_KEY` /
`AZURE_SPEECH_REGION` (optional `AZURE_SPEECH_ENDPOINT`), never logged or cached. One cached JSON per
utterance (request parameters + raw response, never re-sent), rate limit + exponential backoff on
429/5xx (honours Retry-After), clips > 30 s skipped. The parser handles both the REST flat layout and
the SDK nested `PronunciationAssessment` layout and maps Azure words onto manifest words by
character alignment (works for CJK). Used against the live service in November 2026 (region westus) for every cached
benchmark; the parser is also unit-tested on the response examples from the Azure docs
(`tests/test_bench_azure.py`).

Locale map and feature support (Microsoft Learn, *Language support → Pronunciation assessment* and
*How to use pronunciation assessment → Supported features per locale*, checked 2026-11-15):

| our lang | locale | PA supported | prosody | phoneme names / IPA / syllables / spoken-phoneme n-best |
|---|---|---|---|---|
| en | en-US | yes | **yes (en-US only)** | yes (IPA and SAPI) |
| zh | zh-CN | yes | no | phoneme names in SAPI only; no IPA |
| ja, es (es-MX), fr, de, it, pt (pt-BR), ru, ko | ja-JP, es-MX, fr-FR, de-DE, it-IT, pt-BR, ru-RU, ko-KR | yes | no | no (per-phoneme scores without names) |
| zh-TW data | zh-TW (supported, not in the default map) | yes | no | no |

So Azure phone-level comparisons are only well defined for English; elsewhere only word and sentence
scores can be aligned. Azure caps pronunciation assessment at 30 s per REST request.

## Published Azure Pronunciation Assessment numbers

Searched ≈ 15 min. Only numbers found in a source are listed; nothing is estimated.

| # | source | data / protocol | level | Azure result | comparable to our speechocean762 test eval? |
|---|---|---|---|---|---|
| 1 | Wang et al., *Exploring the Potential of Large Multimodal Models as Effective Alternatives for Pronunciation Assessment*, arXiv 2503.11229 (2025), Table 1 (PCC) and Table 2 (SCC) — https://arxiv.org/abs/2503.11229 | speechocean762 **test** (2,500 utts, 0 unscored); Azure PA scores vs the human labels; locale/API/mapping not stated | word accuracy | PCC 0.623 / SCC 0.465 | **yes**, if we correlate Azure word AccuracyScore with `words[].accuracy` on test |
| | | | sentence accuracy | PCC 0.700 / SCC 0.677 | yes |
| | | | sentence fluency | PCC 0.715 / SCC 0.617 | yes |
| | | | sentence prosody | PCC 0.842 / SCC 0.783 | yes (Azure ProsodyScore vs `prosodic`) |
| | | | sentence completeness | PCC 0.258 / SCC 0.143 | yes |
| | | | sentence total | PCC 0.782 / SCC 0.749 | yes (presumably PronScore vs `total`) |
| | | | phone | not reported ("different phonetic sets") | — |
| 2 | Microsoft Tech Community blog, *Speech service Ignite update: new enhancement features for pronunciation assessment* (Nov 2023), Table 1 — https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/speech-service-ignite-update-new-enhancement-features-for/ba-p/3978093 | vendor's internal evaluation on speechocean762's 2,500-utterance evaluation set, PCC; competitors anonymous | accuracy / fluency / prosody | PCC 0.70 / 0.72 / 0.84 | partly: vendor-reported, level of "accuracy" not stated (likely sentence) |
| 3 | same blog, Table 2 | third-party company, public APIs, undisclosed English test set | word acc / utterance acc / fluency / prosody | PCC 0.52 / 0.60 / 0.49 / 0.60 | **no** (unknown data) |
| 4 | Microsoft Tech Community blog, *Speech service update: hierarchical transformer for pronunciation assessment* (Feb 2023) — https://techcommunity.microsoft.com/t5/azure-ai-services-blog/speech-service-update-hierarchical-transformer-for-pronunciation/ba-p/3740866 | speechocean762, 2,500 evaluation samples, PCC; score level not stated; "MSFT w/ OR" is a model description, not explicitly the public API | unspecified | 0.6562 (GOPT 0.5950; "human parity" 0.6787 = mean pairwise PCC between raters) | **no**: level unknown |
| 5 | Microsoft *Characteristics and limitations of Pronunciation Assessment* — https://learn.microsoft.com/en-us/legal/cognitive-services/speech-service/pronunciation-assessment/characteristics-and-limitations-pronunciation-assessment | vendor statement only | — | "> 0.5 Pearson correlation with human judges" | no |

Not found: any published Azure phone-level correlation, or Azure mispronunciation-detection
precision/recall/F1 on L2-ARCTIC, EpaDB or speechocean762. One paper (arXiv 2606.19910) *uses* Azure
PA scores as pseudo-labels for L2-ARCTIC; that is not an evaluation of Azure.
All numbers above were checked against the raw text of the cited pages (2026-11-15). Numbers 2–5
are vendor-reported. GOPT-style academic systems on the same test set (e.g. 3MH sentence total PCC
0.811, word accuracy 0.682, phone PCC 0.693 as listed in source 1's Table 1) give context for where
Azure stands.
