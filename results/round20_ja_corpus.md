# Round 20: a mora-aligned Japanese length corpus

A training corpus for a Japanese vowel / consonant length model: 188.2 hours of real Japanese speech
under CC0, CC BY 4.0 or public-domain terms, aligned so that every short vowel, long vowel, geminate
closure, geminate release and moraic nasal has its own interval. Nothing in the package changes.
Scripts: `bench/round20_ja_corpus/`. Tables: `results/tables/round20_ja_corpus_*.json`. The corpus
(manifests, audio, README) stays in the cache folder `round20/jacorpus/`; no audio or manifest is in
the repository.

## Sources and licences

| source | URL / version | licence (where read) | hours used | speakers | how obtained |
|---|---|---|---|---|---|
| Mozilla Common Voice 22.0, Japanese | Hugging Face mirror `fsicoli/common_voice_22_0`, revision ae911de2 (the one `bench/cv_build.py` already uses) | `cc0-1.0` (dataset card metadata, read through the Hub API; not gated) | 145.28 | 4,188 | TSVs + tar shards streamed over HTTPS; only selected clips kept, as the original mp3 |
| same, clips only present in older releases' split shards | `fsicoli/common_voice_{15_0,17_0,19_0,21_0}` (pinned revisions in `download.py`) | `cc0-1.0` on 17/19/21; `cc` on 15 (same Mozilla release licence, CC0) | included above (9.4 h before QC) | (same speakers) | same; 7,578 clips that are validated in 22.0 |
| FLEURS ja_jp train | `google/fleurs`, revision 70bb2e84 | `cc-by-4.0` (dataset card) | 6.16 | unknown (no ids) | train.tsv + train.tar.gz |
| Kokoro Speech Dataset v1.3 | github.com/kaiidams/Kokoro-Speech-Dataset (release 1.3 metadata); audio: LibriVox recordings by ekzemplaro on archive.org (64 kb/s mp3) | README: "in the public domain in the USA ... no restrictions on its use" (LibriVox public-domain policy; texts from Aozora Bunko) | 36.75 | 1 | metadata zip from the GitHub release; 543 chapter mp3s from archive.org; clips cut at the dataset's own sample offsets |

Not used:
- **Common Voice validated clips outside train/dev/test** (192 h in 22.0): the mirror has no audio for
  them; Mozilla's own distribution needs an account. 9.4 h of them were recovered from older releases.
- **CSS10 Japanese**: only on Kaggle (account needed); same reader and book (明暗) as Kokoro.
- **MFA's pretrained Japanese acoustic model** (`japanese_mfa` 3.0.0): the card says CC BY 4.0, but it
  was trained on GlobalPhone (commercial licence), JVS and TEDxJP; not used. The aligner was trained
  on this corpus instead.
- ReazonSpeech, JSUT, JVS, YODAS (CC BY 3.0) and anything share-alike or non-commercial: excluded by rule.

## Exclusions (`round20_ja_corpus_exclusions.json`)

Removed by clip, by speaker (CV client id) and by prompt sentence (NFKC, letters and digits only),
before anything was downloaded:

| set | speakers | sentences | CV clips hit by speaker/clip | by sentence |
|---|---|---|---|---|
| Lingolix Japanese test (manifest `cv_clip` / `prompt` / `target` only, v2026.10 and main) | 184 | 300 | 26,293 | 1,645 |
| cv-real ja test half | 370 | 390 | 20,951 | 4,491 |
| cv-real ja dev half (round-19 dev guard) | 371 | 388 | 21,775 | 4,512 |
| round-19 CV dev pool (recomputed with `cv_pool.select`) | 368 | 680 | 22,896 | 8,189 |
| FLEURS validation / test sentences | – | 135 / 321 | – | 0 |

Union: 1,194 speakers, 2,182 sentences; 97,650 of 308,353 candidate CV clips removed. FLEURS train
and Kokoro were checked against the same sentence list (1 Kokoro clip removed). No audio, label or
score of the Lingolix test was opened. Other filters: down_votes > 0 (25,369), digits or Latin
letters in the prompt (3,446), duration outside 1-15 s (338).

Speaker split: salted hash of the client id, 85 / 7.5 / 7.5. FLEURS and Kokoro are wholly in train.
Speakers whose cv-real hash half is "test" but who have no clip in the cv-real records are kept (2,239)
and flagged (`cv_real_hash_half`), so a user can apply round 19's stricter rule.

## Alignment

- Aligner: Montreal Forced Aligner 3.3.10 (MIT; Kaldi Apache-2.0, OpenFst Apache-2.0) from conda-forge
  in a private environment under the work folder. Nothing was added to the package dependencies.
- Acoustic model: trained from scratch (MFA's standard monophone -> triphone -> LDA -> SAT recipe) on
  the CV-validated + FLEURS part, 16,893 utterances, 90 minutes on 4 jobs.
- Phones: a i u e o; aa ii uu ee oo (a long vowel is one phone); N; cl (closure); consonants, with
  palatalised C+j as one phone. Minimum one 10 ms state per phone.
- Pronunciations: Pronunciada's own Japanese front-end. Each front-end unit maps to phones as
  [cl] C V/N [cl]: a geminate mora becomes closure + release, ー and repeated-vowel merges become one
  long vowel, ん is its own N.
- Final alignment: every batch was re-aligned with a dictionary holding, for each word, the canonical
  form (probability 1.0) and variants: each long vowel shortened (0.3), each geminate closure dropped
  (0.3), the ou / ei spelling form (0.5). The output is turned into the manifest format by `build.py`
  (pauses >= 150 ms split phrases; phrase-final, pre-pausal, neighbour fields).

## Quality control

**(a) Failures and disagreements** (`round20_ja_corpus_alignqc.json`)

| step | CV validated + FLEURS | Kokoro | CV other |
|---|---|---|---|
| front-end could not build units (Latin text, label errors) | 217 CV + 229 FLEURS | 445 | (in CV) |
| Kokoro's own reading (MeCab) disagrees on the number of long vowels / geminates / N | – | 9,975 of 44,342 (long 8,010, N 1,834, geminate 869) | – |
| aligner preferred a shortened long vowel | 1,840 | 4,053 | 35,928 |
| aligner preferred a dropped geminate | 505 | 1,401 | 7,992 |
| spelling form preferred | 0 | 0 | 5 |
| no alignment / inconsistent tiers | 29 | 0 | 1,721 |
| **kept** | **22,095 of 24,469** | **28,913 of 34,367** | **113,281 of 158,927** |

The not-yet-validated CV clips lose 28 % against 10 % for validated ones, which is what a quality
filter on unreviewed recordings should do. The cost: the kept corpus under-represents long vowels that
natives really shorten; `jamora` should treat the label as "canonical length, said recognisably".

**(b) Duration alone as a baseline** (`round20_ja_corpus_durations.json`; vowels not devoiced and not
phrase-final; consonants word-medial after a vowel or N; geminate = closure + release)

| contrast | short / single median | long / geminate median | AUC (duration) | AUC (duration x speech rate) |
|---|---|---|---|---|
| vowels, all | 70 ms (n 2.55 M) | 120 ms (n 152 k) | 0.802 (overlap 0.51) | 0.821 |
| a / i / u / e / o | | | 0.883 / 0.832 / 0.813 / 0.784 / 0.805 | |
| voiceless obstruents, all | 60 ms (n 1.06 M) | 140 ms (n 67.9 k) | 0.937 (overlap 0.26) | 0.949 |
| t / k / p / s / ʃ / tʃ | | | 0.988 / 0.979 / 0.998 / 0.928 / 0.897 / 0.931 | |

By speaking rate (morae per second of speech: < 6, 6-7, 7-8, 8-9, >= 9) the vowel AUC is 0.84, 0.87,
0.84, 0.82, 0.77 and the geminate AUC 0.91, 0.94, 0.95, 0.95, 0.94. Fast speech is where vowel length
is hardest: medians 60 vs 100 ms at >= 9 morae/s. Duration alone is a weak baseline for vowels; any
length head has to beat 0.80 (0.82 rate-normalised).

**(c) Against the runtime aligner** (`round20_ja_corpus_anchors.json`): 412 utterances (CV validation
and held-back speakers, FLEURS, Kokoro) scored with Pronunciada's `Scorer`; nucleus phones paired in
order. Difference = runtime (ZIPA) minus corpus, ms:

| | n | median | p10 / p90 | median abs | within 20 ms | within 40 ms |
|---|---|---|---|---|---|---|
| short vowel midpoint | 8,602 | 0 | -35 / +20 | 10 | 68 % | 88 % |
| long vowel midpoint | 620 | -10 | -55 / +15 | 20 | 50 % | 75 % |
| N midpoint | 503 | -25 | -65 / +35 | 25 | 33 % | 62 % |
| short vowel start / end | | +30 / -30 | | | | |
| long vowel start / end | | +30 / -50 | | | | |

The runtime spans are narrower than the corpus intervals (start ~30 ms later, end 30-50 ms earlier,
the usual CTC peak), but the midpoints agree: an anchor-centred window at runtime sits within 20 ms
of the corpus vowel centre two times in three for short vowels and half the time for long ones, with
heavier tails (sd 68 / 120 ms). 251 nuclei got a zero-length runtime span and were skipped.

**(d) Independent cross-check** (`round20_ja_corpus_xcheck.json`, 600 tokens each, 5 ms frames)

| | measured | Pearson r | AUC, aligner duration | AUC, independent measure | medians, independent |
|---|---|---|---|---|---|
| vowels between voiceless consonants (voiced run: energy + autocorrelation) | 477 | 0.44 | 0.893 | 0.822 | long 120 / short 70 ms |
| k / t / p after a vowel (low-energy closure run) | 468 | 0.82 | 0.972 | 0.908 | geminate 95 / single 25 ms |

An aligner-free measure separates the labelled long / short and geminate / single tokens nearly as
well as the aligner's own durations do, so the labels and boundaries are not an artefact of the
aligner.

## Totals that decide the next step (`round20_ja_corpus_counts.json`)

- Clean hours: **188.2** = CC0 145.3 (CV validated 24.4, CV other 120.9) + CC BY 4.0 6.2 (FLEURS) +
  public domain 36.8 (Kokoro, one reader). 164,289 utterances; 4,188 CV speakers + FLEURS + 1.
- Split: train 165.5 h, validation 10.8 h (329 speakers), held-back 11.9 h (312 speakers); train has 3,547 CV speakers.
- Speaker concentration in CV train: 3,547 speakers, top speaker 3.9 h, top 10 6.5 %, top 50 10.4 %
  of the hours; median 1.8 min per speaker.
- Long vowels: aː 4,760, iː 14,524, uː 20,620, eː 24,013, oː 98,097 (phrase-final 530 / 1,987 / 694 /
  612 / 5,707). Short vowels 0.36-0.95 M each.
- Geminates: 68,293 closures (phrase-final 16,307): t 50,872, k 7,964, p 3,497, ʃ 2,398, s 1,200,
  tʃ 1,036, kj 475, ts 269, d 225; voiced geminates are rare (< 300 in all). N 158,219 (10,298 final).
- aː is the scarcest long vowel (4.8 k); phrase-final long vowels other than oː and iː number in the
  hundreds.

## How much more could be added

- Common Voice: nothing more from the open mirrors (older releases were mined: +9.4 h). The 192 h of
  validated clips outside the official splits need Mozilla's own download (account); after the same
  exclusions roughly half would survive.
- LibriVox Japanese on archive.org: 37 items, 97.7 h listed; Kokoro covers 14 of them (58.7 h listed).
  The other 23 items (about 39 h, a few readers) are public domain with Aozora Bunko texts, but need a
  text-to-audio segmentation step first (Kokoro-Align or the aligner here); realistic yield 25-30 h.
- The 9,975 Kokoro clips that failed the reading check (about 13 h) could be recovered with a
  per-word reading reconciliation.
- Larger gains need data we cannot use under these rules (ReazonSpeech, JSUT/JVS, YODAS).

## Reproduce

```
python -m bench.round20_ja_corpus.exclude          # exclusions + CV selection
python -m bench.round20_ja_corpus.download cv      # CV 22.0 + older shards (original mp3)
python -m bench.round20_ja_corpus.download fleurs
python -m bench.round20_ja_corpus.download kokoro-list && bash bench/round20_ja_corpus/fetch_kokoro.sh
python -m bench.round20_ja_corpus.download kokoro
python -m bench.round20_ja_corpus.prep records
python -m bench.round20_ja_corpus.prep batch train_clean --where "r['tier']=='validated' and r['source']!='kokoro'"
python -m bench.heavy --lane cpu --need 25 --floor 12 -- bash bench/round20_ja_corpus/mfa.sh train train_clean
bash bench/round20_ja_corpus/run_batches.sh        # variant-dictionary alignment, resumable
python -m bench.round20_ja_corpus.build manifest --batches valid,kokoro_0,kokoro_1,other_0,...,other_7
python -m bench.round20_ja_corpus.qc {durations,counts,xcheck,anchors} --table
python -m bench.round20_ja_corpus.tables
```

The MFA environment: `micromamba create -p <work>/tools/mfa -c conda-forge montreal-forced-aligner=3.3 python=3.11`.
Disk: 16 GB under the work folder (audio 7.6 GB, manifests 1.1 GB, aligner environment 2.4 GB).
