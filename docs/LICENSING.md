# Licensing audit

Status: audit of 2026-11-17, made by reading the fit scripts in `bench/` and the metadata stored in
the shipped files, not from memory. It is not legal advice. Items the owner must decide are
collected in [Open questions](#open-questions) at the end; **one blocking finding** (a shipped
calibrator fitted on non-commercial data) is in [§3](#3-non-commercial-data-l2-arctic-epadb).

Short version:

- Code (`pronunciada/`, `bench/`, `scripts/`, `tests/`): Apache-2.0 ([LICENSE](../LICENSE)).
- Acoustic model: downloaded at first use, not in the repository, CC BY 4.0.
- Bundled data: everything in `pronunciada/data/` is project-made (Apache-2.0) **except** the
  lexicons in `pronunciada/data/lex/`, which keep their source licences (BSD-2-Clause-style,
  MIT, and CC BY-SA for it / pt / ru); see `pronunciada/data/lex/LICENSE.txt`.
- One shipped file, `pronunciada/data/calib/word.npz`, was fitted on a mix that includes
  L2-ARCTIC train (CC BY-NC 4.0) and EpaDB train. That must be resolved before a public or
  commercial release.

## 1. Models

| model | where | licence | attribution | notes |
|---|---|---|---|---|
| ZIPA `anyspeech/zipa-small-crctc-500k` (64 M parameters; `model.onnx` 248 MB, `model.int8.onnx` 67 MB) | downloaded on first use from huggingface.co into `~/.cache/pronunciada/models/` | CC BY 4.0 (model card checked 2026-11-17) | Zhu et al., "ZIPA: A family of efficient models for multilingual phone recognition", github.com/lingjzhu/zipa | `pronunciada/data/zipa_tokens.txt` is the model's token list (same licence). `acoustic.add_graph_outputs` exposes internal layers as extra ONNX outputs in memory; the downloaded file is not redistributed. The `-ns-` ZIPA variants are CC BY-NC-SA and are never used. |
| Models trained for this project (everything in `pronunciada/data/` outside `lex/`) | in the package | Apache-2.0 (project), subject to §2–§3 | — | trained on the data listed in §2 |

## 2. Shipped data files and what they were fitted on

"Use": **training** = weights or tree splits learned from it; **calibration** = a probability map
or bias fitted on it; **thresholds** = operating points / guards chosen on it; **selection** =
used only to pick between candidates. Datasets that only appear in benchmark tables are in §4.

| shipped file(s) | fitted by | data and use |
|---|---|---|
| `data/adapters/<lang>.npz` (4.4 MB) | `bench/pipeline/train_adapter.py` | FLEURS train (training, CTC), FLEURS validation (early stopping, selection). Targets come from the text front-ends, i.e. partly from the lexicons of §5. |
| `data/native/<lang>.json` | `bench/pipeline/native_stats.py` | FLEURS train / validation natives (statistics). |
| `data/calib/sound_B.npz`, `sound_E.npz` | `bench/calib/train_sound.py`, `train_sound_emb.py` | speechocean762 train (training + calibration). |
| `data/calib/sound_E_emb.npz` | `bench/calib/sound_emb.py` | FLEURS-en train (native prototypes). |
| `data/calib/sound_U.npz` (universal sound model) | `bench/calib/train_wide.py` | speechocean762 train (training, calibration; metadata `trained_on: so762`). **Selection** between candidates used leave-one-corpus-out AUC on L2-ARCTIC train and EpaDB train; the chosen candidate does not contain them. |
| `data/calib/word.npz` (word acceptance trees, 115 KB) | `bench/calib/train_word2.py` (round 2), maps re-fitted by `recal_word.py` | **speechocean762 train + L2-ARCTIC train + EpaDB train + FLEURS validation planted errors** (training; metadata `trained_on: so762+l2arctic+epadb+fleurs`, candidate `full/s3` in `results/tables/calib_round2.json`). Calibration maps: speechocean762 train. See §3. |
| `data/calib/word_reg.npz` | `bench/calib/train_word2.py` | speechocean762 train (the regression target is missing for the other corpora, so only speechocean762 rows enter the regression). |
| `data/calib/word_reg_r6.npz` | `bench/calib/train_word_emb.py` | speechocean762 train. |
| `data/calib/sentence.npz`, `sentence_r4.npz` | `bench/calib/train_delivery.py`, `train_word2.py` / `train_levels.py` | speechocean762 train. |
| `data/calib/config.json` (per-language biases, model choices) | `bench/calib/finalize.py`, `bias_grid.py` | FLEURS train / validation natives (bias), FLEURS validation (Russian bias rule), cv-real dev half = Common Voice (guards). |
| `data/calib/omission.json` | `bench/calib/fit_omission.py` | FLEURS validation planted extra words. |
| `data/calib/phone_dist.npz` | `bench/calib/phone_dist.py` | panphon feature table (MIT), computed offline. |
| `data/calib/prosody/<lang>.json`, `*.stress.json`, `zh.tone_zh_v2.json`, `zh.tone_zh_v3.json` | `bench/calib/fit_prosody.py` | FLEURS validation (calibration, leniency rule). `zh.tone_zh_v3.json` belongs to a retired model that is not in the repository. |
| `data/calib/sentence_zh_recal.json` | `bench/calib/fit_sentence_zh.py` | OMPAL dev learners (calibration). |
| `data/calib/contrast_caps.json` | `bench/calib/contrast_fix.py` | FLEURS validation (thresholds). |
| `data/calib/contrast_word/{zh,ja}.json` | `bench/round18/fit.py` | FLEURS validation, speechocean762 train, OMPAL dev (thresholds, guards). |
| `data/prosody/stress.onnx`, `stress_v3.onnx` (+ `.json`, `.zemb.npz`) | `bench/prosody/stress_train.py`, `bench/prosody/*` | FLEURS train (training), FLEURS validation (selection). |
| `data/prosody/tone_zh_v2.onnx` | `bench/prosody/tone2.py` | AISHELL-3, AISHELL-1, Common Voice zh-TW train, FLEURS zh train (training). |
| `data/prosody/tone_zh_v3z.onnx` (+ `.json`, `.zemb.npz`) | `bench/prosody/tone3.py` | as v2 plus the rest of FLEURS zh train (training); AISHELL-3 test and the zh-TW test set (Common Voice) as dev sets; chosen as the default on OMPAL dev (selection). |
| `data/length/ja.json` | `bench/length/fit_length.py`, `bench/round17/fit.py` | FLEURS ja train / validation (training, thresholds). |
| `data/coach/model.json` | `bench/coach/fit_v2.py`, `fit_r3.py`, `fit_r4.py` | speechocean762 train, OMPAL dev, FLEURS validation (thresholds, coaching logistic). |
| `data/coach/gate.json` | `bench/coach/fit_gate.py` | FLEURS validation planted errors (thresholds of the heard-sound gate). |
| audio-quality thresholds (constants in `pronunciada/quality.py`) | `bench/coach/audio_gate.py` | FLEURS validation + synthetic degradations, cv-real dev half (Common Voice) (thresholds). |
| `data/coach/attr_heads.npz` | `bench/coach/fit_r4.py heads` | FLEURS train (training), FLEURS validation (leak check). |
| `data/coach/contrast_heads.json`, `ko_tense.npz` | `bench/round17/fit.py`, `bench/round18/tense.py` | FLEURS train (training), FLEURS validation (thresholds). |
| `data/coach/zh_sound.npz` | round 12b (`bench/calib/ompal_sound_emb.py`) | OMPAL dev learners (training). |
| `data/coach/inventory.json` | front-ends over FLEURS text | FLEURS (text). |
| `data/cjk/zh_neutral.txt`, `zh_tw_tones.txt` | hand-written | project (Apache-2.0). |
| `data/lex/*` | `pronunciada/text/_eu_build.py`, `bench/calib/ru_stress_lex.py`, `ru_round8.py` | see §5. |

Everything except `word.npz` was checked against the non-commercial corpora: no other fit script
that writes into `pronunciada/data/` reads L2-ARCTIC or EpaDB (searched every script under
`bench/` that writes there; `fit_r4.py` reads them only in its `eval` step, which writes to
`results/`).

## 3. Non-commercial data (L2-ARCTIC, EpaDB)

- **L2-ARCTIC v5.0** is CC BY-NC 4.0. It is meant to be benchmark / report-only here.
- **EpaDB**: the repository docs say ODC-By 1.0 (CONICET repository, handle 11336/150678). The
  downloaded `readme.md` states no licence and the repository page could not be read
  automatically, so this is **not verified** (open question 2).

Finding: `pronunciada/data/calib/word.npz` (the word "every sound acceptable" model behind
`WordResult.sounds_prob`, hence every word and sentence probability) was trained in round 2 on
L2-ARCTIC train and EpaDB train together with speechocean762 train and FLEURS. Its calibration
maps were later re-fitted on speechocean762 only, but the trees themselves include the
non-commercial rows. The NOTICE previously said L2-ARCTIC was "never trained on for shipped
models"; that sentence was wrong and has been corrected.

Not fixed here (the task forbids changing models): the clean fix is to re-fit the word model on
the `so762` candidate (or `so762` + FLEURS), which `bench/calib/train_word2.py` already supports,
and re-run the held-out benchmarks. Until then the package should not be released for commercial
use.

Also disclosed: L2-ARCTIC train and EpaDB train were used for **model selection** in
`train_wide.py` (sound_U) and `train_word2.py`, and the English F1 thresholds in
`bench/eval_english.py` are chosen on their train splits (benchmark tables only).

## 4. Datasets

| dataset | licence (as checked) | used for |
|---|---|---|
| FLEURS (google/fleurs) | CC BY 4.0 | training (adapters, stress, tone, heads, length), calibration and thresholds (validation), benchmarks (test) |
| speechocean762 | CC BY 4.0 on OpenSLR (SLR101); the HF card `mispeech/speechocean762` says Apache-2.0 (both checked 2026-11-17) | training / calibration (train), benchmark (test) |
| OMPAL (Hsieh et al., Interspeech 2025, doi:10.21437/Interspeech.2025-983) | CC BY 4.0 (LICENSE in the corpus repository) | dev: calibration and selection of Mandarin defaults; test: benchmark |
| Common Voice (cv-real 22.0 locales, zh-TW 25.0) | CC0-1.0 (as stated by the mirror cards) | dev half: guards and audio-gate thresholds; zh-TW train: tone training; test half: benchmark; ru sentence text: stress lexicon input |
| AISHELL-3 | Apache-2.0 (HF card `AISHELL/AISHELL-3`; OpenSLR page not checked) | tone model training |
| AISHELL-1 | Apache-2.0 on OpenSLR (SLR33), which also says "free for academic use"; 200 of 300 speakers came from a community mirror whose card states no licence | tone model training (open question 3) |
| L2-ARCTIC v5.0 | CC BY-NC 4.0 | benchmark; **training of word.npz** (§3); selection |
| EpaDB | ODC-By 1.0 per earlier notes, unverified | benchmark; **training of word.npz** (§3); selection |
| Azure Pronunciation Assessment responses | Microsoft service terms | benchmark only; cached outside the repository, never committed |

## 5. Lexicons and share-alike data

| shipped file | source | licence |
|---|---|---|
| `lex/en.tsv.gz`, `en_lts.tsv.gz` | CMUdict | BSD-2-Clause-style, (C) Carnegie Mellon University |
| `lex/de.tsv.gz`, `de_lts.tsv.gz`, `fr.tsv.gz`, `fr_lts.tsv.gz`, `es_words.txt.gz` | ipa-dict | MIT |
| `lex/it.tsv.gz`, `it_lts.tsv.gz` | WikiPron `ita_latn_broad` (Wiktionary) | **CC BY-SA** |
| `lex/pt.tsv.gz` | WikiPron `por_latn_bz_broad` | **CC BY-SA** |
| `lex/ru.tsv.gz`, `ru_stress.tsv.gz` | WikiPron `rus_cyrl_narrow` (`ru_stress` is a suffix table learned from it, an adaptation) | **CC BY-SA** |
| `lex/ru_stress_extra.tsv.gz` | silero-stress 1.5 (MIT) predictions, offline, on FLEURS-ru (CC BY 4.0) and Common Voice ru (CC0) text | project; silero-stress is not a runtime dependency |
| `lex/ru_vowels.json` | measured on FLEURS-ru with the acoustic model | project |

What share-alike implies: the six it / pt / ru files may be redistributed inside the
Apache-2.0 repository, but they stay CC BY-SA. Anyone who redistributes them, or a modified
version, must keep the CC BY-SA licence and the "Wiktionary contributors" attribution. The
share-alike clause does not reach the code that reads them (separate works). Whether models
trained with references produced from these lexicons (it / pt / ru adapters and calibration)
are adaptations is unsettled (open question 4).

Packaging done now (no code change): `pronunciada/data/lex/LICENSE.txt` lists each file's
licence and attribution, `pyproject.toml` declares the combined licence expression
(`Apache-2.0 AND CC-BY-SA-4.0 AND BSD-2-Clause AND MIT`) and ships that file as a licence file,
and `NOTICE` points to it.

Cleaner option, not done because it changes load paths: move the six files to
`pronunciada/data/lex/cc-by-sa/` with their own `LICENSE`, or publish them as a separate data
package (`pronunciada-lex-sa`) that the front-ends load when installed. Without them Italian,
Portuguese and Russian fall back to rules, so scores would change and the benchmarks would need
a re-run.

## 6. Python dependencies

Runtime (core, `pip install pronunciada`):

| package | licence | note |
|---|---|---|
| numpy | BSD-3-Clause | |
| scipy | BSD-3-Clause | |
| onnxruntime | MIT | |
| kaldi-native-fbank | Apache-2.0 | |
| soundfile | BSD-3-Clause | wheels bundle libsndfile (LGPL-2.1), dynamically loaded |

Runtime extras: `[zh]` pypinyin (MIT), jieba (MIT), opencc-python-reimplemented (Apache-2.0);
`[ja]` pyopenjtalk-plus (MIT; bundles Open JTalk Modified-BSD, NAIST-jdic BSD-3, UniDic-derived
entries BSD-3 option, Unihan Unicode-3.0, an unused HTS voice CC BY 3.0) with sudachipy and
sudachidict-core (Apache-2.0), pydantic family (MIT); `[numbers]` num2words (LGPL-2.1-or-later,
optional, imported unmodified). Korean needs nothing extra.

Benchmark / training only (never imported by `pronunciada/`): torch (BSD-3), scikit-learn
(BSD-3), pandas (BSD-3), pyarrow (Apache-2.0), fsspec (BSD-3), aiohttp (Apache-2.0), requests
(Apache-2.0), panphon (MIT), huggingface_hub (Apache-2.0), onnx / onnxscript (Apache-2.0 / MIT),
praat-parselmouth (GPL-3.0: offline pitch cross-check only), silero-stress (MIT, offline),
ffmpeg (system binary, offline decoding).

## Attribution text (for an "About" screen or documentation)

> Pronunciada uses the ZIPA phone recognition model by Zhu et al. (CC BY 4.0). Trained and
> calibrated with FLEURS (Google, CC BY 4.0), speechocean762 (CC BY 4.0), OMPAL (Hsieh et al.,
> CC BY 4.0), Common Voice (Mozilla, CC0), AISHELL-1 and AISHELL-3 (Apache-2.0). Pronunciation
> dictionaries: CMUdict (Carnegie Mellon University), ipa-dict (MIT), and Wiktionary data via
> WikiPron (CC BY-SA; Wiktionary contributors).

## Open questions

1. **word.npz and L2-ARCTIC / EpaDB** (blocking): re-fit without them (recommended), or obtain
   permission. Decide also whether selection-only use (sound_U, word candidates) is acceptable.
2. **EpaDB licence**: confirm ODC-By 1.0 on the CONICET record; the downloaded files do not say.
3. **AISHELL-1 / AISHELL-3**: OpenSLR lists AISHELL-1 as Apache-2.0 but adds "free for academic use";
   200 AISHELL-1 speakers came from a mirror without a licence statement. Confirm commercial
   use of the tone models trained on them, or re-train the default tone model without AISHELL.
4. **Models trained with CC BY-SA lexicon references** (it / pt / ru adapters, calibration): owner
   decision whether to treat them as adaptations.
5. **speechocean762**: OpenSLR says CC BY 4.0, the HF card Apache-2.0; both permit commercial
   use, but cite it either way.
6. **Common Voice metadata in git**: `bench/data/cv_real/*.jsonl` carry Common Voice contributor
   hashes (`client_id`), votes and self-reported accents. The data is CC0, but Common Voice asks
   users not to try to identify speakers; consider dropping `client_id` / `accents` from the
   public copy (the speaker bootstrap needs only a grouping key).
7. **Azure output in results**: tables contain aggregate statistics of Azure responses, no raw
   responses. Check Microsoft's terms on publishing benchmark results of the service.
