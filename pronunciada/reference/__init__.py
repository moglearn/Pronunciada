"""Dependency-free reference of the scoring post-processing, for porting to C++ / Swift /
Kotlin / Rust.

Standard library only (no NumPy): plain lists, explicit loops, Python floats = IEEE-754
binary64. Every function mirrors one production function and gives the same numbers on the same
inputs (tests/test_reference_module.py; bounds in docs/REFERENCE_PORTING.md).

Modules
  numeric    scalar helpers with IEEE semantics; logaddexp, log-sum-exp, the three summation
             orders production uses
  tokens     the 127-token file; phone symbol check
  lattice    target sounds -> CTC lattice (pronunciada/lattice.py)
  align      emission table, scaled forward-backward with stitching (L_wild, L_del, L_ins),
             exact log-domain pass, Viterbi, greedy decode, per-sound evidence
             (pronunciada/align.py: emission_table, score_slots, score_reference)
  calibrate  tree ensembles, calibration maps, log-odds, strictness blend
             (pronunciada/calibrate.py: TreeEnsemble, CalibMap, Head, LevelModel), and the
             omission and prosody-channel word formulas (pronunciada/scorer.py: _present_prob,
             _pros_prob)
  npz        reader for the .npz model files

Input: adapted frame log-posteriors [T][127] (backbone output after the language's adapter),
the target phones [{"ipa", "alts": [[ipa, log_prior], ...]}], and the two switches the scorer
sets: insertion (extra-sound evidence: Korean always; Japanese with its word step; languages
whose identity rule reads it) and wild_slots (the explain-away pass, German).

Where it stops, and why
  * Before: audio decoding, resampling, fbank features (kaldi-native-fbank, C++), the backbone
    (ONNX Runtime) and the adapter (a small float32 matrix product; its last bits depend on the
    BLAS used, so it is not reproduced here; pronunciada/adapter.py is 60 lines).
  * After the evidence: the per-sound feature table (pronunciada/features.phone_table: native
    statistics z-scores, NumPy-compatible quantiles, word and utterance aggregates) is not
    ported; the calibrators here take its rows as input. The golden vectors include such rows.
  * Not ported: every channel that reads encoder activations (sound-identity heads, contrast
    heads, word-score projection, sound_E embeddings, sentence delivery projections), the three
    extra ONNX models (prosody/stress_v3.onnx, prosody/tone_zh_v3z.onnx, calib/seqhead_en.onnx),
    the F0 tracker and delivery features, the length channel, and the per-language word factors
    of pronunciada/scorer.py. They need the extra-output backbone (ONNX Runtime on the device)
    or more porting work.

What a port must load
  zipa_tokens.txt (127 lines, 0.8 KB); per level model one .npz file (zip of .npy arrays, format
  in npz.py and calibrate.py): the universal sound model data/calib/sound_U.npz (config key
  "universal_sound_model", 291 KB), data/calib/word.npz (119 KB), and for the omission formula
  data/calib/omission.json (8.3 KB of JSON coefficients).
"""
