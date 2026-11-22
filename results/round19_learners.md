# Round 19: a learner-speech evaluation suite

Accuracy on real learner speech had been measured for English and Mandarin only. This round
looked for every openly downloadable learner corpus in our ten languages, built a fixed,
speaker-disjoint test selection from each one, defined one multilingual summary number, and ran
Pronunciada 1.0.2 (unchanged) and Azure Pronunciation Assessment on it. Nothing was fitted or
tuned. All metric choices were written into `bench/learners/evaluate.py` and committed (d231346)
before either system was run on the new data.

**Short version.** Open learner data exists for only three of the ten languages: English, Mandarin
and Spanish. Spanish (Heroico / USMA) and a second English test (Speech Accent Archive) are new.
For ja, ko, de, fr, it, pt and ru nothing usable can be downloaded without an account or a
licence request (list below). On the three-language summary, Pronunciada scores 0.714 against
Azure's 0.670 (difference +0.045 [+0.011, +0.080], p = 0.008). The lead comes from Mandarin and
Spanish. On English expert scores Azure is ahead in concordance. Spanish rests on 18 speakers
and only has native-vs-learner labels, so it is weak evidence.

## Corpora

| corpus | lang | test speakers | test items | test hours | labels | licence | may be used for |
|---|---|---|---|---|---|---|---|
| **Heroico / USMA** (OpenSLR 39 = LDC2006S37), new | es | 18 USMA (10 learners, 8 natives) + 30 HEROICO natives | 600 (250 learner, 350 native), 60 prompts | 0.37 | native / learner (corpus folder names); no proficiency, no expert marks | OpenSLR says Apache-2.0; LDC catalogue says LDC User Agreement for Non-Members | evaluation only (the two licence statements disagree) |
| **Speech Accent Archive** (GMU, official OSF FLAC), new | en | 200 (100 native, 100 learners, 59 L1s) | 200 (one paragraph each) | 1.51 (1.01 within Azure's 30 s limit) | native / learner, L1, English onset age, years of residence (self-reported) | CC BY-NC-SA 4.0 | evaluation only |
| speechocean762 test (existing, referenced in place) | en | 125 learners (Mandarin L1) | 2,500 | 2.69 | 5 experts: sentence/word/phone scores | Apache-2.0 | any use (the test split is held out) |
| OMPAL test half (existing) | zh | 18 learners (French L1) + 3 natives | 745 (663 learner) | 1.15 | teachers: sentence accuracy/fluency/prosody, per-syllable errors | CC BY 4.0 | any use (the test half is held out) |

Manifests (outside the repo): `~/.cache/pronunciada/data/learners/<corpus>/{manifest,test,dev}.jsonl`,
`build.json` (counts, exclusions, checksums, licence). The whole suite is in one file,
`~/.cache/pronunciada/data/learners/suite_test.jsonl`, with 4,045 lines. Each line follows the benchmark manifest layout:
`id, audio` (absolute path, 16 kHz mono), `lang, text` (the exact prompt), `speaker, dur, labels`
(`labels.corpus`, `labels.native`, `labels.l1`, `labels.human_score` where experts scored).
**Any runner that reads {audio, text, lang} can be pointed at it. This includes the OpenPronounce runner (`rivals`).**

Build commands: `python -m bench.learners.heroico`, `python -m bench.learners.saa`,
`python -m bench.learners.suite`. Both fetchers are resumable. The Heroico archive is checked against its
size and sha256, and every SAA FLAC against the sha256 that OSF publishes. Downloads are kept in
`~/.cache/pronunciada/round19/evalsuite/downloads/` (3.8 GB) and the built audio takes 0.3 GB.

### Heroico / USMA: what was kept and why
- Kept: USMA head-mounted-microphone recordings (`s<N>.wav`; all 18 speakers read the same 206
  sentences), and HEROICO cadets' recordings of those same 206 sentences (48 speakers, all
  native). The pool is 5,323 items from 66 speakers (3.29 h).
- Excluded: 2,439 USMA throat-microphone files; 3,892 HEROICO read sentences that no learner reads;
  7,267 free answers (`Answers_Spanish`, not read speech, never extracted); 100 HEROICO files in
  two stray nested folders.
- **WAV header problem.** In 14 of the 18 USMA folders the WAV headers give 20050 Hz; the other folders, and every HEROICO file, give 22050 Hz.
  The Kaldi recipe for this corpus forces `sox -r 22050` on every file, and so do we.
  Otherwise those recordings would be slowed down by 9 %. The same 14 folders hold
  fixed-length recording windows (identical file sizes across speakers for a prompt), so their
  clips carry more silence. Both native and learner speakers are in that group (6 of 8 natives, 6 of 10 learners),
  so it does not split along the native / learner line, but it is a recording-condition difference.
- Labels: the folder names say `native` or `nonnative`. Three "nonnative" folders carry place names
  (Castillian, Puerto.Rico, Argentina.France). We took this to be where they learned the language and kept the corpus label. The numeric
  fields in the folder names are not documented and are not used.
- Test selection (seed 19): 60 prompts drawn from those that HEROICO natives also read. Each USMA speaker
  gives 25 items, rotating through the 60 prompts, so every prompt is read by several learners and
  several natives. 30 HEROICO speakers give 5 items each. Dev: the 471 items of the 18 HEROICO
  speakers not drawn, which are speaker-disjoint from the test. Every USMA speaker is in the test, because 10 learners are already
  too few to split. Unused items of test speakers are in neither file.

### Speech Accent Archive: what was kept and why
- From 3,031 speakers in `speaker_information.xlsx`: 52 have no FLAC on OSF and are dropped.
  "Native" means first language English with no other native language (655 speakers).
  "Learner" means another first language and an English onset age of 5 or more (2,119). 205 early bilinguals (onset 0–4) are excluded as
  ambiguous.
- Test (seed 19): 100 natives drawn at random, and 100 learners drawn at random with at most 4 per L1 (59 L1s).
  The other eligible speakers are listed in `dev.jsonl` (metadata only, no audio downloaded).
- Azure takes at most 30 s per request. 41 learner and 6 native paragraphs are longer, so
  both systems are compared on the 153 clips of 30 s or less (94 natives, 59 learners). Pronunciada is also
  reported on all 200. The excluded long clips are mostly slow learners, so the 30-s set is
  harder for every system.

## Metric definitions (exact)

- **Sentence score.** Pronunciada: `SentenceResult.scores["total"]`, falling back to `prob` (it was never needed: every clip
  returned `total`). Azure: `NBest[0].PronScore`. Both run at default settings. Azure used the same request
  settings as our other Azure runs (`bench.azure_pa.pa_params`) with locales en-US, es-MX and zh-TW (the OMPAL primary).
  A failure counts as score 0. There were none.
- **Concordance C.** Over all pairs of recordings whose human labels differ, C is the share that the
  system orders the same way as the label (system ties count ½). With a binary label (native better
  than learner) C is exactly the ROC AUC of `bench.metrics.roc_auc`. With an expert score it is
  Harrell's C. *Prompt-matched* means that only pairs reading the same prompt are counted.
  Unit test: `tests/test_learners_metrics.py`. Pearson r is from `bench.metrics.pearson`.
- **Primary learner metric per language** (one per language, using the strongest labels available):
  en = C against the speechocean762 expert sentence `total`; zh = C against the OMPAL teachers' mean
  sentence `accuracy` (learners only); es = prompt-matched native-vs-learner C on USMA.
  Native vs learner (kind b) is **weaker evidence than expert scores**. Learners are not wrong
  on every sentence, and natives differ from learners in more than pronunciation: speaking rate,
  recording session.
- **Multilingual summary LS** = the unweighted mean over languages of the primary C. Its 95 % interval
  comes from 2,000 bootstrap replicates, each resampling speakers with replacement independently within each
  language, with both systems on the same replicates. The paired difference ours − Azure gets a two-sided
  p = 2·min(P(d ≤ 0), P(d ≥ 0)). LS is always reported with the number of languages, speakers and items it rests on.
- **LS-NL** (second summary) = the mean native-vs-learner AUC over en (Speech Accent Archive, ≤ 30 s),
  es (USMA, prompt-matched) and zh (OMPAL test, prompt-matched, 3 native speakers).
- Secondary: a duration-only baseline (score = −duration) for every native-vs-learner test;
  speaker-level AUC (each speaker's mean score); Heroico with the 30 HEROICO natives added; Speech
  Accent Archive learners: Spearman correlation of the score with English onset age (expected to be negative) and with
  log(1 + years of residence) (expected to be positive).
- All intervals are 95 % percentile bootstraps over speakers with 2,000 replicates.

## Results

Generated tables: `results/tables/learners_suite.json` and `learners_suite.md`
(`python -m bench.learners.evaluate --steps report`; Pronunciada fingerprint d9f6bc52048577f4).

### Multilingual summary

| summary | languages | speakers | items | Pronunciada | Azure | ours − Azure | p |
|---|---|---|---|---|---|---|---|
| **LS** (primary metric per language) | en, es, zh (3 of 10) | 161 | 3,613 | **0.714** [0.651, 0.763] | 0.670 [0.615, 0.716] | **+0.045** [+0.011, +0.080] | 0.008 |
| LS-NL (native vs learner only) | en, es, zh | 192 | 1,348 | 0.771 [0.708, 0.830] | 0.761 [0.703, 0.813] | +0.010 [−0.039, +0.058] | 0.69 |

### Per language (primary metric first)

| lang | test | evidence | speakers | items | metric | Pronunciada | Azure | ours − Azure | p |
|---|---|---|---|---|---|---|---|---|---|
| en | speechocean762 test, sentence total | experts | 125 | 2,500 | **C** | 0.836 [0.813, 0.854] | 0.855 [0.832, 0.872] | −0.019 [−0.029, −0.009] | 0.002 |
| en | | | | | Pearson | 0.762 [0.699, 0.812] | 0.796 [0.743, 0.835] | −0.034 [−0.092, +0.008] | 0.19 |
| zh | OMPAL test learners, teacher accuracy | experts | 18 | 663 | **C** | 0.639 [0.566, 0.689] | 0.573 [0.506, 0.623] | +0.066 [+0.039, +0.085] | < 0.001 |
| zh | | | | | Pearson | 0.389 [0.179, 0.493] | 0.205 [0.021, 0.308] | +0.185 [+0.116, +0.231] | < 0.001 |
| es | Heroico USMA, prompt-matched | native vs learner | 18 (8 / 10) | 450 | **C (AUC)** | 0.669 [0.506, 0.818] | 0.581 [0.432, 0.717] | +0.088 [−0.001, +0.200] | 0.057 |
| es | | | | | speaker-level AUC | 0.850 [0.600, 1.000] | 0.575 [0.284, 0.846] | +0.275 [+0.039, +0.538] | 0.024 |
| es | | | | | −duration baseline | 0.386 [0.153, 0.665] | | | |
| es | + 30 HEROICO natives (other site) | native vs learner | 48 | 600 | C | 0.618 [0.495, 0.740] | 0.542 [0.432, 0.645] | +0.076 [+0.001, +0.155] | 0.051 |
| es | | | | | speaker-level AUC | 0.629 [0.472, 0.778] | 0.437 [0.258, 0.627] | +0.192 [+0.042, +0.350] | 0.014 |
| en | Speech Accent Archive, ≤ 30 s | native vs learner | 153 (94 / 59) | 153 | C (AUC) | 0.678 [0.585, 0.764] | 0.750 [0.663, 0.822] | −0.071 [−0.148, +0.012] | 0.094 |
| en | | | | | −duration baseline | 0.698 [0.612, 0.774] | | | |
| en | Speech Accent Archive, all 200 (ours only) | native vs learner | 200 | 200 | C (AUC) | 0.763 [0.695, 0.828] | – | – | – |
| en | | | | | −duration baseline | 0.786 [0.725, 0.846] | | | |
| en | SAA learners: score vs onset age | self-report | 59 | 59 | Spearman | −0.115 [−0.365, +0.157] | −0.160 [−0.403, +0.110] | | 0.71 |
| en | SAA learners: score vs log residence | self-report | 59 | 59 | Spearman | −0.007 [−0.237, +0.245] | +0.084 [−0.149, +0.311] | | 0.40 |
| zh | OMPAL test, native vs learner | native vs learner | 21 (3 / 18) | 745 | C (AUC) | 0.967 [0.921, 0.996] | 0.953 [0.903, 0.989] | +0.014 [−0.059, +0.073] | 0.57 |
| zh | | | | | −duration baseline | 0.970 [0.930, 0.998] | | | |

What the numbers say:
- **English, expert scores.** The Pearson values reproduce the scoreboard exactly (0.762 vs 0.796, n.s.). By
  concordance, Azure orders the 2,500 speechocean762 sentences slightly but significantly better
  (−0.019). On the measure the summary uses, English counts against us.
- **Mandarin, expert scores.** Pronunciada is clearly ahead (C +0.066, r +0.185). This is the same
  direction as `ompal.md`. Its primary row used the `accuracy` sub-score (r 0.634), not `total`. Here
  `total` was fixed in advance for every language, and it correlates less well with teacher
  accuracy (r 0.389).
- **Spanish, native vs learner (new).** Pronunciada separates USMA learners from natives
  better than Azure: +0.088 at item level, just short of significance (p 0.057), and +0.275 at
  speaker level (p 0.024). Azure's es-MX scores barely separate the groups (0.58; 0.54 with the
  HEROICO natives). With 10 learners and 8 natives the intervals are wide. Duration does not
  explain it: the duration baseline is below 0.5 because of the fixed recording windows.
- **Speech Accent Archive (new).** Both systems separate natives from learners, Azure somewhat better
  (−0.071, n.s.). **Neither beats clip duration alone** (0.698 on the same clips). Learners read the
  paragraph more slowly, so this test mostly measures speaking rate. Neither system's score
  follows the self-reported background of the learners (onset age, residence). These are weak proxies, but
  there is no sign of a proficiency signal at this sample size.
- **OMPAL natives vs learners** is near ceiling for both systems and also for duration alone (0.97). It
  rests on 3 native speakers and says little.

## Azure usage

New Azure audio sent: **1.373 h**, under the 6 h cap. It was 600 Heroico clips (0.365 h, es-MX) and 153
Speech Accent Archive clips (1.008 h, en-US). The estimate was written to STATUS.md before the run, every
response is cached under `~/.cache/pronunciada/azure/learners-{heroico,saa}/`, there were 0 failures and
nothing was re-sent. speechocean762 and OMPAL used cached responses only (`azure/speechocean762`,
`azure/ompal-zh-TW`). The 47 Speech Accent Archive clips longer than 30 s were never sent.

## Limits

- **Coverage is 3 of 10 languages.** No openly downloadable learner speech was found for ja, ko, de,
  fr, it, pt or ru (list below). The summary says nothing about them.
- Speaker counts are small: Spanish has 10 learners and 8 natives, OMPAL 18 learners and 3 natives. Intervals are
  wide, and one Spanish learner can move the AUC by several points.
- Read speech only, short prompts (Heroico), or one paragraph (Speech Accent Archive).
- Native-vs-learner AUC is **not** expert scoring. It rewards anything that differs between the
  groups (speaking rate, recording session) and it treats every learner sentence as worse than every
  native one. Duration baselines are reported so the reader can see how much of this a trivial cue
  explains. On Speech Accent Archive it explains all of it.
- Recording conditions differ within Heroico (two header rates, fixed vs trimmed windows), and the
  HEROICO natives were recorded at another site and with other equipment. This is why they are a secondary row.
- The Speech Accent Archive comparison uses only clips of 30 s or less (Azure's limit). That drops the slowest
  learners and lowers the AUC for both systems.
- The summary metric differs from the scoreboard's English headline. The scoreboard reports Pearson, where
  English is n.s. behind. LS uses C, where English is significantly behind. Both are shown.
- Licences: Speech Accent Archive is non-commercial, and Heroico's two licence statements conflict. Both are
  evaluation only and the dev parts are not fit material.

## Corpora that need the user (registration, request or click-through) — not fetched

| corpus | language(s) | what it is | what the user would have to do |
|---|---|---|---|
| LATIC | zh | 4 learners (Russian, Korean, French, Arabic L1), 2,579 read utterances, expert mispronunciation labels | IEEE DataPort account (free), then download: https://ieee-dataport.org/node/4319 (DOI 10.21227/mqtj-qh10). Speaker count is tiny. |
| L2FOC | fr, it | 65 speakers, natives and learners, ~10 h, three tasks, aligned TextGrids | No data repository was found (Zenodo, GitHub, HF, Crossref links). The article https://doi.org/10.36253/oar-3497 (CC BY 4.0) is behind a bot check that we did not try to pass. The user can read its data statement or ask the authors (Università di Torino). |
| IFCASL | fr, de | French and German learners and natives, read sentences, phonetic annotation | CLARIN academic licence: http://www.ifcasl.org |
| LeaP | de, en | learner German and English, read and free speech, prosodic annotation | Request through the Hamburg Centre for Language Corpora (HZSK / CLARIN) |
| BAS Strange Corpus SC1 / SC10 | de | non-native German read speech (SC1: 72 non-native of 88; SC10: 67 non-native of 70, 17 L1s) | BAS CLARIN repository, academic licence: https://www.bas.uni-muenchen.de/Bas/BasKorporaeng.html |
| TalkBank SLABank: FLLOC, PAROLE, LANGSNAP (fr); SPLLOC, Nebrija-CORELE (es); German and Spanish packs | fr, es, de | learner corpora, mostly spontaneous, some reading tasks | Free TalkBank registration: https://slabank.talkbank.org. An ungated Hugging Face copy of LANGSNAP exists (`chatquironronne/langsnap-l2-french`). It was **not** used because it would bypass TalkBank's registration. |
| Speak & Improve Corpus 2025 | en | L2 English, CEFR-scored by examiners | Licence agreement with Cambridge English / ELiT (https://speakandimprove.com) |
| MRPL2 | en | 23 learners, 14 L1s, 4,500 utts, phone-level labels | Zenodo restricted record, access request: https://zenodo.org/records/22690685 |
| JASMIN-CGN | nl | includes non-native Dutch read speech | Free but needs a signed INT licence: https://taalmaterialen.ivdnt.org/?p=1334 |
| Korean AI Hub learner corpus | ko | Korean read by foreigners | Restricted to Korean nationals; do not use |

The search pass (about 25 minutes: OpenSLR resource list, Zenodo API, Hugging Face dataset search, web
search) found nothing else truly open with learner speech in ja, ko, de, pt, ru or Dutch. Also
checked and rejected:
- OpenSLR 57 "African Accented French": French L1/L2 varieties of African speakers, not learners.
- Samrómur L2: Icelandic.
- J-AESOP: English read by Japanese speakers.
- jml2026/multilingual-accent-speech: vendor sample, natives except English.
- Accented-Speech-Fragments: 0.5-s clips without text.

**Dutch.** No open *learner* Dutch data exists. JASMIN-CGN's non-native part needs a signed licence.
Open *native* read Dutch suitable for a native-guard test does exist: Common Voice nl (CC0), FLEURS nl
(CC BY 4.0) and Multilingual LibriSpeech Dutch (CC BY 4.0). Pronunciada has no Dutch model.

## Files

- `bench/learners/__init__.py`: shared paths and the resumable, checksum-verified fetch.
- `bench/learners/heroico.py`, `bench/learners/saa.py`: fetch, build, the seeded test selection, and the licence notes.
- `bench/learners/suite.py`: builds `suite_test.jsonl`.
- `bench/learners/evaluate.py`: the pre-specified metrics, Pronunciada runner (cached per package fingerprint;
  run with `python -m bench.heavy --lane cpu -- python -m bench.learners.evaluate --steps ours`), Azure
  runner (job cap of 6 h, never re-sends) and report.
- `tests/test_learners_metrics.py`
- `results/tables/learners_suite.{json,md}`
