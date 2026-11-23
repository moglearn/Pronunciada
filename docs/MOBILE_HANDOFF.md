## Mobile & sherpa-onnx Handoff

State of the on-device packaging on branch `opt-phase2` (2026-11-23). Every number below was
measured on one arm64 laptop CPU with practice clips only (FLEURS train, the committed list in
`bench/opt/clips.json`, and speechocean762 train for English); they are indicative, not phone
measurements. Existing outputs are unchanged: `bench/opt/parity.py` against 7834007 gives max
|diff| = 0 on every score and every posterior, 200 practice clips, fp32 and int8.

### 1. Exported artifacts

Built by `python scripts/export_sherpa_onnx.py --fp32 --scoring --verify` into
`~/.cache/pronunciada/opt/mobile/sherpa-onnx-zipa-small/` (outside the repository; nothing large is
committed):

| file | bytes | use |
|---|---|---|
| `model.int8.onnx` | 70,678,128 | sherpa-onnx (int8) |
| `model.onnx` | 260,268,328 | sherpa-onnx (fp32, the product default precision) |
| `model.int8.scoring.onnx` | 70,679,343 | direct ONNX Runtime use by the scorer (int8) |
| `model.scoring.onnx` | 260,269,543 | direct ONNX Runtime use by the scorer (fp32) |
| `tokens.txt` | 769 | 127 lines `<symbol> <id>` |
| `MODEL_CARD.md`, `manifest.json` | small | description; bytes, sha256, licence, source, metadata, checks |

The model files are the downloaded ZIPA files with metadata entries appended at the end of the
file (keys added: `model_name`, `url`, `license`, `language`, `vocab_size`, `subsampling_factor`,
`frame_shift_ms`, `feature`, `packaged_by`; kept: `model_type=zipformer2_ctc`, `version`,
`model_author`, `comment`, `onnx.infer`). The original file is an exact byte prefix of the exported
one (456 bytes appended), and ONNX Runtime outputs are bit-identical on three inputs. The scoring
variants add the encoder activations of `pronunciada.acoustic.ALL_LAYER_OUTPUTS` as graph outputs;
their log-probabilities are bit-identical to the plain files.

Per-language data bundles: `python scripts/build_language_bundles.py --check` writes
`~/.cache/pronunciada/opt/mobile/bundles/<lang>/` (the files under `pronunciada/data/` that
`score_phones` reads for that language, plus `manifest.json`). Sizes: section 7.

### 2. What a phone app needs

API (Python reference, `pronunciada/core.py`): `score_phones(samples, 16000, target, lang)` returns
the same `SentenceResult` as `Scorer(lang).score(audio, text)`. `target` is the JSON produced by
`phones_for(text, lang)` (format version 1, described in `pronunciada/core.py`; symbols, marks and
normalisation per language in `docs/PHONES.md` and `docs/phones.json`). `score_phones` imports no
text front-end, num2words, soundfile, pyopenjtalk, sudachipy, jieba, pypinyin or opencc and opens
no lexicon (checked in a fresh process per language, `tests/test_core.py`).

Steps a port performs, in order:

1. Audio: mono float32 in [-1, 1] at 16 kHz. Decoding and resampling are the app's job (the repo
   uses a Kaiser-window polyphase filter, `pronunciada/_resample.py`; sherpa-onnx's own resampler
   is a different filter and gives different features for 44.1 / 48 kHz input).
2. Features: kaldi-native-fbank `OnlineFbank`, options exactly as `pronunciada/audio.py`
   `_fbank_options`: `samp_freq=16000`, `frame_shift_ms=10`, `frame_length_ms=25`, `dither=0`,
   `preemph_coeff=0.97`, `remove_dc_offset=true`, `window_type="povey"`,
   `round_to_power_of_two=true`, `snip_edges=false`, `num_bins=80`, `low_freq=20`,
   `high_freq=-400`, `use_energy=false`, `use_power=true`, `use_log_fbank=true`; everything else at
   the library defaults. These are also sherpa-onnx's defaults: knf 1.23.3 with sherpa-onnx's
   options gives bit-identical features on 8 practice clips. T = (samples + 80) // 160 frames.
   Input with fewer than 9 frames is padded to 9 with log(float32 eps) = -15.942385.
3. Backbone (ONNX Runtime, CPU): inputs `x` float32 `[1, T, 80]`, `x_lens` int64 `[1]`; outputs
   `log_probs` float32 `[1, T', 127]`, `log_probs_len` int64 `[1]`, T' = (T - 7) // 2, plus the
   encoder activations the language reads: `/encoder/1/out_combiner/Add_output_0` (enc1, 256-d),
   `/encoder/3/out_combiner/Add_output_0` (enc3, 512-d), `/encoder/5/out_combiner/Add_output_0`
   (enc5, 256-d), `/Transpose_1_output_0` (final, 512-d) for every language; English also reads
   enc0 (`/encoder/0/layers.1/bypass/Add_output_0`), enc2, enc4 (`/encoder/{2,4}/out_combiner/Add_output_0`).
   Stack outputs have shape `[T', 1, C]`, final `[1, T', C]`. Output frame j starts at 0.02 j + 0.03 s.
4. Adapter (`pronunciada/adapter.py`, weights `adapters/<lang>.npz`): lp' = log_softmax(lp + f(x)),
   x = max(lp, floor) / scale, f linear / context / two-layer convolution by kind.
5. Alignment, evidence, features, universal sound model and word model: the dependency-free
   reference `pronunciada/reference/` (standard library only, explicit loops, binary64), checked
   against production on all 200 practice clips: every alignment and evidence quantity and the
   universal sound and word log-odds identical; `peak_post` within one float32 unit (5.96e-8)
   because production evaluates `exp` in float32. Porting notes, summation orders, constants and
   file formats: `docs/REFERENCE_PORTING.md`. Golden vectors (two practice clips per language,
   posteriors stored as int16 at 1/256 nat, expected outputs computed on the stored values):
   `tests/data/reference_golden/<lang>.json.gz`, 1.40 MB in total.
6. The rest of the scoring chain (not in the reference module yet): language sound models,
   contrast caps, word regression, the per-language word factors (length, contrast, identity,
   stress reliability, Korean identity, Japanese word step), sentence models, delivery and F0
   features, and three small ONNX networks (`prosody/stress_v3.onnx` for en de es it pt ru,
   `prosody/tone_zh_v3z.onnx` for zh, `calib/seqhead_en.onnx` for en). A port either translates
   these from the Python source (`pronunciada/scorer.py` `_calibrate`, `_assemble` and the modules
   they call) or runs the Python scorer.

### 3. What cannot run on a phone today, and what replaces it

| part | why not | options |
|---|---|---|
| Japanese text front-end | pyopenjtalk-plus (C extension + Python post-processing), sudachipy (Rust extension), sudachidict_core (193 MB), pydantic; about 300 MB of dictionaries | produce targets off device (server or build step) for fixed lesson content and ship them as JSON; or port Open JTalk (C) and re-implement the post-processing |
| Chinese text front-end | jieba, pypinyin, opencc-python-reimplemented are Python packages | targets produced off device; or C++ ports (OpenCC is C++ upstream; C++ jieba ports exist; not evaluated) |
| Number expansion (en de es fr it pt ru) | num2words is LGPL-2.1+ and Python-only; Spanish round-20 number readings also use it | targets produced off device (the readings travel in `words[i].alternatives`); or an own number-reading module per language (Korean and Chinese already have one in the repo) |
| Other EU front-ends | Python code, gzip TSV lexicons (de 4.2 MB, the largest) | targets off device; or port `pronunciada/text/_eu_*.py` with the lexicons |
| Audio decoding | soundfile bundles libsndfile (LGPL-2.1) | platform decoders (AVFoundation, MediaCodec) |
| Scoring chain after the word model | Python + NumPy | port per step 6, or run Python |

A target made off device is exact: `score_phones(samples, 16000, phones_for(text, lang), lang)`
equals `Scorer(lang).score(audio, text)` field for field (section 7). A target without
`alternatives` gives a word no round-20 second pass (Spanish numbers); every other field the
scorer reads is part of the target.

int8: the int8 backbone uses dynamic quantisation, and feature differences of 3e-4 move its
log-posteriors by up to 1-2 nats and scores by up to 0.14 (sound), 0.08 (word), 0.02 (sentence).
A phone build of int8 matches desktop numbers only if the features and the integer kernels are
bit-identical; ORT's mobile builds and execution providers (XNNPACK, NNAPI, CoreML) are not
guaranteed to be. Parity is therefore defined against fp32; fp32 is not affected by such
differences (3e-5 nats).

### 4. Offering the backbone to the sherpa-onnx project (checklist for the owner)

Nobody has been contacted. Steps, if the owner decides to offer it:

- [ ] Re-read the ZIPA licence and attribution requirements (model card: CC BY 4.0) and decide
      whether to ask the ZIPA authors first, since the model is theirs.
- [ ] Archive: `sherpa-onnx-zipa-small-crctc-500k-<date>.tar.bz2` holding `model.int8.onnx`,
      `model.onnx` (optional), `tokens.txt`, a model card (the content of `MODEL_CARD.md`; the
      upstream archives call it `README.md`), and `test_wavs/` with a few clips whose licence allows
      redistribution (FLEURS is CC BY 4.0).
- [ ] What they host and where: upstream attaches model archives to the `asr-models` release of
      `github.com/k2-fsa/sherpa-onnx` and lists each model on a docs page under
      `k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/` (Zipformer CTC: `icefall/zipformer.html`)
      with download command, file sizes and a `sherpa-onnx-offline --zipformer-ctc-model=... --tokens=...`
      example.
- [ ] Model card contents: what the model outputs (IPA phones, not words), the 127 tokens, the
      feature options (sherpa-onnx defaults), the licence and authors, file sizes and sha256
      (`manifest.json`), the decoding check (section 5), the known gaps below.
- [ ] Tell them about the timestamps: `OfflineZipformerCtcModel::Impl::SubsamplingFactor()` returns 4
      (`sherpa-onnx/csrc/offline-zipformer-ctc-model.cc`, line 62 at commit 3bf65c5), ZIPA subsamples
      by 2, so sherpa-onnx timestamps are exactly twice the true time (measured ratio 2.000 on 40 decodes).
      A fix upstream would read `subsampling_factor` from the model metadata (now present in the
      exported files) or detect it.
- [ ] Other gaps to state: input under 9 fbank frames (about 90 ms) gives no output; sherpa-onnx's
      resampler differs from the repo's; the recogniser API returns tokens and timestamps only, not
      frame posteriors or encoder activations, so it can host phone recognition but not
      Pronunciada's scoring.

### 5. sherpa-onnx check

`scripts/verify_sherpa_onnx.py` (test `tests/test_sherpa_export.py`, skipped without sherpa-onnx):
sherpa-onnx 2.0.3 loads the exported `model.int8.onnx` and `model.onnx` with
`OfflineRecognizer.from_zipformer_ctc`; on 2 practice clips in each of the ten languages (40
decodes) its token sequence equals the repo's collapsed greedy decode token for token (the
word-boundary token printed as a space); timestamp ratio 2.000 on every clip.

### 6. Licence status of shipped files

Backbone files and `tokens.txt`: CC BY 4.0 (ZIPA; Hugging Face model card metadata re-read on
2026-11-23). Every file of every language bundle carries its licence and source in its
`manifest.json`, taken from `pronunciada/data/LICENSES_SHIPPED.json`; no file is unlisted. All bundle
files are declared Apache-2.0. `docs/LICENSING.md` and `NOTICE` were corrected where the files and
the history prove it (round-19 files added to the table; five share-alike lexicon files, not six;
files trained on labels from the it / pt / ru references listed as open question 4). Open for the
owner: whether `coach/identity/{pt,ru}.npz`, `prosody/stress_v3.onnx` (+ `.json`, shared by en de
es it pt ru) and `coach/attr_heads.npz` (read for ja ko ru zh) are treated as share-alike like the
it / pt / ru adapters; less direct cases `native/{it,pt,ru}.json`, `calib/identity/{pt,ru}.json`,
`calib/prosody/{it,pt,ru}.json`. If they are, CC BY-SA files enter the bundles of en de es it pt ru
ja ko zh. No GPL or LGPL code is in the `score_phones` path.

### 7. Parity and size numbers

Reproduce: `python scripts/build_language_bundles.py --check --table results/tables/mobile_bundles.json`, then
`python scripts/mobile_parity.py` (writes `results/tables/mobile_parity.json`, `mobile_readiness.json`).
Existing pipeline = `Scorer(lang).score(audio file, text)`, full installation. Mobile path =
`phones_for` -> `score_phones` in a fresh process with only the exported backbone and the language bundle.
Compared: the SentenceResult (every field except timing), rescore at strictness 0 and 1, `details()`.

| lang | clips | fp32 identical clips | fp32 values compared | int8 identical clips | int8 values compared |
|---|---|---|---|---|---|
| en (practice) | 20 | 20 | 121,977 (max abs diff 0) | 20 | 121,977 (max abs diff 0) |
| en (speechocean762-train) | 100 | 100 | 136,062 (max abs diff 0) | 100 | 136,062 (max abs diff 0) |
| de (practice) | 20 | 20 | 147,500 (max abs diff 0) | 20 | 147,500 (max abs diff 0) |
| es (practice) | 20 | 20 | 140,688 (max abs diff 0) | 20 | 140,688 (max abs diff 0) |
| fr (practice) | 20 | 20 | 119,434 (max abs diff 0) | 20 | 119,434 (max abs diff 0) |
| it (practice) | 20 | 20 | 154,521 (max abs diff 0) | 20 | 154,521 (max abs diff 0) |
| pt (practice) | 20 | 20 | 142,533 (max abs diff 0) | 20 | 142,533 (max abs diff 0) |
| ru (practice) | 20 | 20 | 131,512 (max abs diff 0) | 20 | 131,512 (max abs diff 0) |
| ja (practice) | 20 | 20 | 145,259 (max abs diff 0) | 20 | 145,259 (max abs diff 0) |
| ko (practice) | 20 | 20 | 136,409 (max abs diff 0) | 20 | 136,409 (max abs diff 0) |
| zh (practice) | 20 | 20 | 112,579 (max abs diff 0) | 20 | 112,579 (max abs diff 0) |

All 300 clips identical in fp32 (300 of 300; 1,488,474 values) and int8 (300 of 300; 1,488,474 values). The int8 result holds on this machine, where features and ONNX Runtime kernels are the same in both paths (section 3).

Size (raw bytes, MB = 1e6 bytes; backbone = model file + tokens.txt) and readiness (fresh process scoring 5 practice clips; peak RSS of that process; median latency per clip after the first, end to end in `score_phones`; this laptop):

| lang | bundle files | bundle MB | int8 backbone + bundle MB | fp32 backbone + bundle MB | fp32 latency ms (1 / 2 / 4 threads) | fp32 peak RSS MB | int8 latency ms (1 / 2 / 4) | int8 peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| en | 22 | 4.65 | 75.3 | 264.9 | 295 / 189 / 140 | 675 | 199 / 139 / 108 | 493 |
| de | 21 | 3.63 | 74.3 | 263.9 | 435 / 285 / 207 | 754 | 301 / 213 / 170 | 592 |
| es | 19 | 3.93 | 74.6 | 264.2 | 288 / 182 / 137 | 522 | 190 / 136 / 118 | 339 |
| fr | 13 | 1.10 | 71.8 | 261.4 | 245 / 150 / 103 | 482 | 155 / 104 / 77 | 292 |
| it | 17 | 1.94 | 72.6 | 262.2 | 308 / 200 / 159 | 639 | 210 / 143 / 126 | 463 |
| pt | 19 | 3.55 | 74.2 | 263.8 | 316 / 200 / 144 | 749 | 207 / 146 / 116 | 563 |
| ru | 20 | 4.14 | 74.8 | 264.4 | 385 / 247 / 176 | 608 | 257 / 182 / 143 | 428 |
| ja | 19 | 2.41 | 73.1 | 262.7 | 400 / 251 / 174 | 555 | 264 / 180 / 136 | 374 |
| ko | 17 | 3.62 | 74.3 | 263.9 | 358 / 221 / 157 | 584 | 235 / 159 / 133 | 408 |
| zh | 23 | 4.19 | 74.9 | 264.5 | 428 / 269 / 187 | 676 | 283 / 193 / 149 | 492 |

Clip lengths in the readiness runs: 5.2 to 23.5 s.
Bundle contents per language, with licence and why scoring reads each file: the bundles' `manifest.json`. Shared by all ten bundles: `calib/config.json`, `calib/contrast_caps.json`, `calib/omission.json`, `calib/phone_dist.npz`, `calib/sound_U.npz`, `calib/word.npz`, `calib/word_reg.npz`, `calib/word_reg_r6.npz`, `coach/contrast_heads.json`, `zipa_tokens.txt`.
Not in any bundle: lexicons and other front-end data (`data/lex/`, `data/cjk/`) and coaching-only files. The Japanese and Chinese third-party front-end dictionaries (about 300 MB and 9 MB) are not needed when targets come from elsewhere.
Peak RSS is measured on desktop Python with NumPy and the full ONNX Runtime; it is an upper reference, not a phone figure.
