## Notes (factual)

- **Reproduce**: `python -m bench.pipeline.run --lang all` (warm → train_adapter → native_stats →
  features → analyze). Every stage is idempotent, keyed on a hash of the front-end sources
  (`bench.pipeline.common.g2p_version`), the shipped adapter and the native table.
- **FLEURS validation ids**: the parquet `id` column is a sentence id shared by several speakers, so
  utterance ids are `fleurs-<lang>-validation-<wav stem>`; `labels.fleurs_id` keeps the FLEURS id.
  Clips > 30 s are skipped, as for train/test.
- **Data problem: 177 of the 1000 FLEURS es_419 train clips are digital silence** (peak |x| < 1e-4;
  greedy decode empty). Clips with < 1 greedy token per second are excluded from adapter training
  and native statistics (only es train is affected; no FLEURS validation/test clip is).
- **Data problem: l2arctic**: `bench/data/l2arctic.py` keeps only the last ARPAbet-looking TextGrid
  tier as `labels.phones`; for 429 / 3599 files (314 train, 115 test) that is a 1–3-entry secondary
  tier and the full phone tier is lost. Those utterances are excluded from the l2arctic feature
  table (counted in the coverage table).
- **Race in bench/corrupt.py**: `PhoneDistance.save` writes a fixed `phone_dist.tmp`; with several
  worker processes this raised FileNotFoundError. The pipeline patches the method at runtime
  (per-process temp name + merge) in `bench/pipeline/evalctl.py`; bench/corrupt.py is unchanged.
- **Posteriors for aishell3 and zhtw** were already in the cache (written by another workstream);
  3 + 3 files were recomputed and matched within float16 rounding (max |Δ| 0.016).
- Native statistics are computed on FLEURS **train**, i.e. on the data the adapter was fitted on.
- **Learned variants are off for all languages** (architect decision after en/zh/es, where turning
  them on lowered the controlled-error AUC; see section 3). The JSONs still contain the learned
  tables (`variants_enabled: false`). Almost all learned alternatives are deletions (∅).
- Adapter selection: `linear` was within noise of the best only for en. Elsewhere it was
  significantly (by 0.0001–0.004 mean AUC) worse than the best, so `ctx` (zh, ja, fr, de, pt) or
  `conv` (es, it, ko, ru) shipped. Every language's chosen adapter beats raw.
- Feature columns added in feature_version 2 (all over the Viterbi span ± 1 frame, adapted
  posteriors): can_mean/can_max, comp_tok/comp_mean/comp_max/comp_cls/comp_dist (panphon distance
  via bench.corrupt.PhoneDistance; NaN when panphon cannot parse, e.g. modifier tokens),
  log_ratio_comp, ent_nonblank, blank_mean, span_frames_padded, last_* for phones with ≥ 2 tokens,
  prev/next_llr_sub, word_mean/min_llr_sub. Competitors exclude the phone's own tokens; a modifier
  token (e.g. ʰ) can be the competitor (comp_cls = "modifier").
- In the speechocean762 GBM the top permutation feature is `utt_dur` (utterance duration).
