# Contrast audit: which sound contrasts the scorer cannot hear (dev only)

**Question.** For which specific sound contrasts is the scorer effectively blind, per language?
The answer decides which small dedicated detectors to build first. No detector is built here, and
no package code was changed.

**Short answer.** Most segmental contrasts are seen: the sound is flagged in ≥ 60 % of cases at a
5 % native false-alarm rate. The blind spots fall into three groups.

1. **Duration (length and doubling).** Japanese vowel length and っ are blind in both directions.
   Italian doubling is blind when a single consonant is doubled. Where the error is a shortened
   long vowel or an undoubled consonant, the front-end accepts it anyway: the short form is listed
   as an alternative pronunciation.
2. **Laryngeal contrasts.** These are aspiration in Korean (blind both ways, and lax / tense cannot
   be written at all), aspiration in Mandarin (blind in one direction) and Russian hard / soft
   consonants (blind, or accepted as an alternative).
3. **One-directional vowel and nasality errors.** Portuguese nasal vs oral (said nasal where oral
   was expected: 12 %), Portuguese open vs close mid vowels (said open where close was expected:
   14 %) and German final devoicing.

Where the raw evidence already separates the pair but the shown probability does not, the
calibration hides it. This holds for Mandarin and Korean aspiration, English θ→s and ð→z, and
Russian soft for hard. There a recalibration helps more than a new detector. Japanese and Italian
length / doubling are different: even the raw evidence is near chance (AUC 0.45–0.77), because
CTC frames collapse repeated sounds. The alignment's own onset-to-onset timing does separate the
Japanese pairs (AUC 0.85 vowels, 0.90 っ), so a small duration detector is feasible with what we
already compute.

## Protocol

* **Data and settings.** FLEURS **validation** only (dev). 250 clips per language, or all 226
  for ko, chosen with `numpy.default_rng(0)`. Native audio throughout. The public
  `pronunciada.Scorer(lang)` runs at its defaults (prosody on, shipped calibrators).
* **Swaps** (`bench/audit/contrasts.py`). Every eligible position of each contrast is listed.
  240 positions per (language, contrast) are drawn: half per direction, at most 2 per clip and
  direction, seed `crc32(lang/contrast)`. Fewer are drawn where fewer exist: de final devoicing
  154, ru 161.
  * The reference is edited at that one position. The audio keeps member A, and the reference
    now demands member B with its alternatives cleared. This simulates a learner who said A where
    B was expected (the `bench/corrupt.py` protocol).
  * Italian doubling and Portuguese nasality are written as two sounds by the front-ends. For
    them the swap deletes or inserts the extra copy (a geminate's first copy, or the coda nasal).
    The flagged sound is then the remaining sound, or the lower of the pair.
  * Mandarin tone is the reference point: one syllable's tone is changed and its `tone_prob` is
    read.
* **Metrics.** The hit rate is the share of edited sounds (or words) at or below the threshold
  that flags 5 % (10 %) of all unedited native sounds (words) of the language. AUC is computed
  against all native sounds. The raw-evidence AUC uses `−llr_sub` alone. The verdict is taken on
  sound hit@5 %: *seen* ≥ 60 %, *weak* 30–60 %, *blind* < 30 %.
* **Front-end accepts the error.** This is the share of positions where the reference for the
  *expected* sound already lists the learner's sound as an alternative. In that case the error
  passes by construction today, whatever the acoustics. When this share is ≥ 50 %, the verdict is
  **not representable (accepted as alternative)**. In such rows the hit rate measures a stricter
  reference than the one shipped: what would happen if the alternative were removed.
* **Natives heard as B** (a ceiling). This is the share of the edited positions where ZIPA's greedy
  decode of the *native* audio already gives member B. These natives do not make the contrast as
  far as the model hears, so no detector can flag the edit there.
* **Caveat on direction.** Native audio has only native values. "Said A where B was expected" is
  therefore simulated by the mirror image: a native A scored against a reference that demands B.
  For length and nasality this is the realistic error in one direction only. Both directions are
  reported separately.
* **Not representable.** Here the reference symbols are identical for both members, so no swap is
  possible. The front-end output for the minimal pair is given (see the section "What the
  front-end emits").

Command (resumable, cached per language and per chunk of 25 clips under
`$PRONUNCIADA_CACHE/audit/contrast_v1/`):

    python -m bench.audit.contrast_audit run --lang all --clips 250 --workers 2
    python -m bench.audit.contrast_audit asp --lang zh,ko        # aspiration measures (natives)
    python -m bench.audit.contrast_audit report                  # -> results/tables/contrast_audit.json

The run took about 70 min with 2 workers × 2 threads. There were 0 edit failures. Unit tests for
the swap generators are in `tests/test_audit_contrasts.py`.

## Per-language results

Each language has two tables. The first pools both directions. The second has one row per
direction, in the form "said A, expected B" (what the learner said, then what the reference
expected).

### en (250 clips; flag thresholds at 5 % native FA: sound P ≤ 0.919, word P ≤ 0.744)

| contrast | n | sound hit@5 % | @10 % | word hit@5 % | @10 % | AUC | raw-evidence AUC | P > 0.9 | front-end accepts the error | natives heard as B | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| voicing (b/p d/t g/k v/f z/s ʒ/ʃ dʒ/tʃ ð/θ) | 240 | 84 % | 95 % | 65 % | 86 % | 0.964 | 0.979 | 23 % | 4 % | 2 % | seen |
| tense / lax vowel (i/ɪ, u/ʊ) | 240 | 75 % | 95 % | 72 % | 89 % | 0.959 | 0.986 | 33 % | 3 % | 1 % | seen |
| θ / s | 240 | 57 % | 88 % | 85 % | 97 % | 0.948 | 0.989 | 47 % | 0 % | 1 % | weak |
| ð / z | 240 | 59 % | 85 % | 76 % | 92 % | 0.950 | 0.994 | 45 % | 0 % | 0 % | weak |
| ð / d | 240 | 90 % | 95 % | 80 % | 92 % | 0.967 | 0.989 | 13 % | 0 % | 1 % | seen |
| v / w | 240 | 98 % | 100 % | 96 % | 100 % | 0.979 | 0.996 | 8 % | 0 % | 0 % | seen |
| v / b | 240 | 95 % | 98 % | 90 % | 96 % | 0.985 | 0.992 | 5 % | 0 % | 0 % | seen |
| æ / ɛ | 240 | 95 % | 98 % | 78 % | 92 % | 0.977 | 0.979 | 7 % | 0 % | 2 % | seen |
| r / l (ɹ / l) | 240 | 100 % | 100 % | 95 % | 100 % | 0.982 | 0.995 | 2 % | 0 % | 0 % | seen |
| s / ʃ | 240 | 97 % | 99 % | 91 % | 98 % | 0.984 | 0.988 | 3 % | 0 % | 1 % | seen |
| nasal place m / n (before a vowel) | 240 | 98 % | 100 % | 95 % | 99 % | 0.988 | 0.994 | 2 % | 0 % | 0 % | seen |
| final n / ŋ | 240 | 86 % | 95 % | 83 % | 93 % | 0.962 | 0.986 | 20 % | 35 % | 1 % | seen; not representable: said n, expected ŋ |
| aspiration (allophonic in English) | – | | | | | | | | | | not representable (not phonemic) |

| direction (said / expected) | n | sound hit@5 % | word hit@5 % | AUC | raw AUC | front-end accepts this error | verdict |
|---|---|---|---|---|---|---|---|
| voicing: said voiced, expected voiceless | 120 | 85 % | 56 % | 0.964 | 0.979 | 8 % | seen |
| voicing: said voiceless, expected voiced | 120 | 83 % | 75 % | 0.965 | 0.978 | 0 % | seen |
| tense_lax: said tense, expected lax | 120 | 53 % | 57 % | 0.947 | 0.985 | 2 % | weak |
| tense_lax: said lax, expected tense | 120 | 96 % | 86 % | 0.971 | 0.988 | 4 % | seen |
| th_s: said θ, expected s | 120 | 95 % | 77 % | 0.975 | 0.983 | 0 % | seen |
| th_s: said s, expected θ | 120 | 18 % | 92 % | 0.921 | 0.994 | 0 % | blind |
| dh_z: said ð, expected z | 120 | 95 % | 65 % | 0.980 | 0.993 | 0 % | seen |
| dh_z: said z, expected ð | 120 | 22 % | 88 % | 0.921 | 0.995 | 0 % | blind |
| dh_d: said ð, expected d | 120 | 82 % | 64 % | 0.957 | 0.984 | 0 % | seen |
| dh_d: said d, expected ð | 120 | 97 % | 96 % | 0.976 | 0.995 | 0 % | seen |
| v_w: said v, expected w | 120 | 99 % | 93 % | 0.990 | 0.996 | 0 % | seen |
| v_w: said w, expected v | 120 | 96 % | 99 % | 0.968 | 0.996 | 0 % | seen |
| v_b: said v, expected b | 120 | 94 % | 92 % | 0.982 | 0.993 | 0 % | seen |
| v_b: said b, expected v | 120 | 97 % | 87 % | 0.989 | 0.991 | 0 % | seen |
| ae_e: said æ, expected ɛ | 120 | 91 % | 66 % | 0.968 | 0.970 | 0 % | seen |
| ae_e: said ɛ, expected æ | 120 | 98 % | 90 % | 0.985 | 0.988 | 0 % | seen |
| r_l: said ɹ, expected l | 120 | 100 % | 93 % | 0.985 | 0.994 | 0 % | seen |
| r_l: said l, expected ɹ | 120 | 99 % | 97 % | 0.979 | 0.995 | 0 % | seen |
| s_sh: said s, expected ʃ | 120 | 100 % | 98 % | 0.990 | 0.995 | 0 % | seen |
| s_sh: said ʃ, expected s | 120 | 94 % | 84 % | 0.977 | 0.981 | 1 % | seen |
| m_n: said m, expected n | 120 | 97 % | 92 % | 0.987 | 0.993 | 0 % | seen |
| m_n: said n, expected m | 120 | 100 % | 98 % | 0.989 | 0.995 | 0 % | seen |
| n_ng: said n, expected ŋ | 120 | 92 % | 94 % | 0.968 | 0.995 | 69 % | not representable (accepted as alternative) |
| n_ng: said ŋ, expected n | 120 | 79 % | 72 % | 0.956 | 0.977 | 0 % | seen |

### zh (250 clips; flag thresholds at 5 % native FA: sound P ≤ 0.967, word P ≤ 0.729)

| contrast | n | sound hit@5 % | @10 % | word hit@5 % | @10 % | AUC | raw-evidence AUC | P > 0.9 | front-end accepts the error | natives heard as B | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aspiration (p/pʰ t/tʰ k/kʰ ts/tsʰ ʈʂ/ʈʂʰ tɕ/tɕʰ) | 240 | 45 % | 68 % | 42 % | 60 % | 0.886 | 0.974 | 94 % | 0 % | 0 % | weak |
| retroflex / dental (zh ch sh / z c s) | 240 | 95 % | 98 % | 65 % | 72 % | 0.982 | 0.989 | 24 % | 0 % | 1 % | seen |
| final n / ng | 240 | 93 % | 98 % | 45 % | 59 % | 0.976 | 0.974 | 26 % | 14 % | 3 % | seen |
| ü / u | 240 | 100 % | 100 % | 98 % | 99 % | 0.992 | 0.996 | 0 % | 0 % | 0 % | seen |
| r / l (ʐ / l) | 240 | 99 % | 100 % | 77 % | 85 % | 0.993 | 0.989 | 5 % | 0 % | 1 % | seen |
| tone (reference point) | 240 | 67 % | 82 % | 49 % | 66 % | 0.944 | – | 0 % | 0 % | – | seen |
| nasal place m / n (before a vowel) | 240 | 100 % | 100 % | 72 % | 81 % | 0.991 | 0.991 | 9 % | 0 % | 0 % | seen |

| direction (said / expected) | n | sound hit@5 % | word hit@5 % | AUC | raw AUC | front-end accepts this error | verdict |
|---|---|---|---|---|---|---|---|
| aspiration: said aspirated, expected plain | 120 | 18 % | 40 % | 0.821 | 0.960 | 0 % | blind |
| aspiration: said plain, expected aspirated | 120 | 72 % | 43 % | 0.951 | 0.989 | 0 % | seen |
| retroflex: said retroflex, expected dental | 120 | 95 % | 61 % | 0.980 | 0.990 | 0 % | seen |
| retroflex: said dental, expected retroflex | 120 | 95 % | 68 % | 0.984 | 0.988 | 0 % | seen |
| n_ng: said n, expected ŋ | 120 | 93 % | 53 % | 0.974 | 0.973 | 0 % | seen |
| n_ng: said ŋ, expected n | 120 | 92 % | 37 % | 0.979 | 0.976 | 28 % | seen |
| y_u: said y, expected u | 120 | 100 % | 97 % | 0.990 | 0.996 | 0 % | seen |
| y_u: said u, expected y | 120 | 100 % | 98 % | 0.994 | 0.996 | 0 % | seen |
| r_l: said ʐ, expected l | 120 | 98 % | 91 % | 0.994 | 0.989 | 0 % | seen |
| r_l: said l, expected ʐ | 120 | 100 % | 62 % | 0.993 | 0.988 | 0 % | seen |
| tone: said dictionary tone, expected other tone | 240 | 67 % | 49 % | 0.944 | – | 0 % | seen |
| m_n: said m, expected n | 120 | 99 % | 45 % | 0.986 | 0.987 | 0 % | seen |
| m_n: said n, expected m | 120 | 100 % | 100 % | 0.997 | 0.996 | 0 % | seen |

### ja (250 clips; flag thresholds at 5 % native FA: sound P ≤ 0.914, word P ≤ 0.856)

| contrast | n | sound hit@5 % | @10 % | word hit@5 % | @10 % | AUC | raw-evidence AUC | P > 0.9 | front-end accepts the error | natives heard as B | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vowel length (incl. ー, おう / えい) | 240 | 21 % | 43 % | 18 % | 44 % | 0.755 | 0.865 | 86 % | 50 % | 13 % | blind; not representable: said short, expected long |
| consonant doubling (っ) | 240 | 33 % | 47 % | 27 % | 50 % | 0.699 | 0.733 | 71 % | 50 % | 16 % | weak; not representable: said single, expected geminate |
| voicing (g/k d/t z/s b/p dʒ/ʃ) | 240 | 62 % | 77 % | 43 % | 64 % | 0.916 | 0.953 | 44 % | 0 % | 5 % | seen |
| s / ʃ | 240 | 68 % | 92 % | 30 % | 68 % | 0.955 | 0.966 | 40 % | 0 % | 1 % | seen |
| nasal place m / n (before a vowel) | 240 | 93 % | 97 % | 84 % | 92 % | 0.985 | 0.986 | 8 % | 0 % | 1 % | seen |
| r / l (not phonemic) | – | | | | | | | | | | not representable (not phonemic) |

| direction (said / expected) | n | sound hit@5 % | word hit@5 % | AUC | raw AUC | front-end accepts this error | verdict |
|---|---|---|---|---|---|---|---|
| vowel_length: said long, expected short | 120 | 5 % | 7 % | 0.608 | 0.766 | 0 % | blind |
| vowel_length: said short, expected long | 120 | 37 % | 29 % | 0.902 | 0.964 | 100 % | not representable (accepted as alternative) |
| gemination: said geminate, expected single | 120 | 6 % | 6 % | 0.454 | 0.483 | 0 % | blind |
| gemination: said single, expected geminate | 120 | 61 % | 48 % | 0.944 | 0.984 | 100 % | not representable (accepted as alternative) |
| voicing: said voiced, expected voiceless | 120 | 48 % | 32 % | 0.880 | 0.931 | 0 % | weak |
| voicing: said voiceless, expected voiced | 120 | 75 % | 54 % | 0.953 | 0.975 | 0 % | seen |
| s_sh: said s, expected ʃ | 120 | 73 % | 22 % | 0.960 | 0.963 | 0 % | seen |
| s_sh: said ʃ, expected s | 120 | 62 % | 37 % | 0.949 | 0.970 | 0 % | seen |
| m_n: said m, expected n | 120 | 92 % | 74 % | 0.984 | 0.982 | 0 % | seen |
| m_n: said n, expected m | 120 | 95 % | 93 % | 0.986 | 0.989 | 0 % | seen |

### es (250 clips; flag thresholds at 5 % native FA: sound P ≤ 0.960, word P ≤ 0.859)

| contrast | n | sound hit@5 % | @10 % | word hit@5 % | @10 % | AUC | raw-evidence AUC | P > 0.9 | front-end accepts the error | natives heard as B | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tap / trill (pero / perro) | 240 | 85 % | 97 % | 49 % | 78 % | 0.970 | 0.976 | 33 % | 0 % | 5 % | seen |
| voicing (b/p d/t g/k) | 240 | 100 % | 100 % | 93 % | 100 % | 0.987 | 0.998 | 2 % | 0 % | 0 % | seen |
| nasal place m / n (before a vowel) | 240 | 100 % | 100 % | 98 % | 100 % | 0.997 | 0.998 | 0 % | 0 % | 0 % | seen |
| b / v (not phonemic) | – | | | | | | | | | | not representable (not phonemic) |

| direction (said / expected) | n | sound hit@5 % | word hit@5 % | AUC | raw AUC | front-end accepts this error | verdict |
|---|---|---|---|---|---|---|---|
| tap_trill: said trill, expected tap | 120 | 76 % | 50 % | 0.959 | 0.963 | 0 % | seen |
| tap_trill: said tap, expected trill | 120 | 95 % | 48 % | 0.981 | 0.990 | 0 % | seen |
| voicing: said voiced, expected voiceless | 120 | 100 % | 89 % | 0.985 | 0.998 | 0 % | seen |
| voicing: said voiceless, expected voiced | 120 | 100 % | 98 % | 0.990 | 0.998 | 0 % | seen |
| m_n: said m, expected n | 120 | 99 % | 98 % | 0.996 | 0.997 | 0 % | seen |
| m_n: said n, expected m | 120 | 100 % | 99 % | 0.998 | 0.998 | 0 % | seen |

### fr (250 clips; flag thresholds at 5 % native FA: sound P ≤ 0.896, word P ≤ 0.867)

| contrast | n | sound hit@5 % | @10 % | word hit@5 % | @10 % | AUC | raw-evidence AUC | P > 0.9 | front-end accepts the error | natives heard as B | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nasal / oral vowel (ɑ̃/a ɔ̃/ɔ ɛ̃/ɛ œ̃/œ) | 240 | 75 % | 83 % | 82 % | 92 % | 0.917 | 0.987 | 24 % | 1 % | 0 % | seen |
| front rounded / unrounded (y/i ø/e œ/ɛ) | 240 | 100 % | 100 % | 96 % | 99 % | 0.989 | 0.994 | 0 % | 0 % | 0 % | seen |
| front / back rounded (y/u ø/o œ/ɔ) | 240 | 99 % | 99 % | 93 % | 100 % | 0.981 | 0.993 | 1 % | 0 % | 0 % | seen |
| close / open mid (e/ɛ o/ɔ ø/œ) | 240 | 84 % | 96 % | 62 % | 84 % | 0.969 | 0.974 | 15 % | 100 % | 2 % | not representable (accepted as alternative) |
| voicing | 240 | 94 % | 98 % | 86 % | 95 % | 0.976 | 0.987 | 6 % | 0 % | 1 % | seen |
| s / ʃ | 240 | 98 % | 99 % | 95 % | 98 % | 0.985 | 0.989 | 2 % | 0 % | 1 % | seen |
| nasal place m / n (before a vowel) | 240 | 100 % | 100 % | 98 % | 100 % | 0.990 | 0.995 | 0 % | 0 % | 0 % | seen |

| direction (said / expected) | n | sound hit@5 % | word hit@5 % | AUC | raw AUC | front-end accepts this error | verdict |
|---|---|---|---|---|---|---|---|
| nasal: said nasal, expected oral | 120 | 62 % | 71 % | 0.860 | 0.978 | 0 % | seen |
| nasal: said oral, expected nasal | 120 | 88 % | 94 % | 0.974 | 0.996 | 2 % | seen |
| round_front: said rounded, expected unrounded | 120 | 99 % | 94 % | 0.988 | 0.992 | 0 % | seen |
| round_front: said unrounded, expected rounded | 120 | 100 % | 98 % | 0.990 | 0.995 | 0 % | seen |
| front_back: said front, expected back | 120 | 98 % | 92 % | 0.981 | 0.992 | 0 % | seen |
| front_back: said back, expected front | 120 | 100 % | 94 % | 0.982 | 0.994 | 0 % | seen |
| mid_vowels: said close, expected open | 120 | 82 % | 52 % | 0.968 | 0.975 | 100 % | not representable (accepted as alternative) |
| mid_vowels: said open, expected close | 120 | 87 % | 72 % | 0.970 | 0.972 | 100 % | not representable (accepted as alternative) |
| voicing: said voiced, expected voiceless | 120 | 94 % | 89 % | 0.977 | 0.991 | 0 % | seen |
| voicing: said voiceless, expected voiced | 120 | 93 % | 83 % | 0.976 | 0.984 | 0 % | seen |
| s_sh: said s, expected ʃ | 154 | 99 % | 99 % | 0.988 | 0.995 | 0 % | seen |
| s_sh: said ʃ, expected s | 86 | 97 % | 87 % | 0.981 | 0.980 | 0 % | seen |
| m_n: said m, expected n | 120 | 100 % | 100 % | 0.991 | 0.996 | 0 % | seen |
| m_n: said n, expected m | 120 | 100 % | 95 % | 0.990 | 0.994 | 0 % | seen |

### de (250 clips; flag thresholds at 5 % native FA: sound P ≤ 0.949, word P ≤ 0.722)

| contrast | n | sound hit@5 % | @10 % | word hit@5 % | @10 % | AUC | raw-evidence AUC | P > 0.9 | front-end accepts the error | natives heard as B | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| long / short vowel, stressed (iː/ɪ yː/ʏ uː/ʊ eː/ɛ øː/œ oː/ɔ ɑː/a) | 240 | 61 % | 73 % | 30 % | 48 % | 0.898 | 0.906 | 70 % | 2 % | 8 % | seen |
| front rounded / unrounded (yː/iː ʏ/ɪ øː/eː œ/ɛ) | 240 | 95 % | 98 % | 81 % | 92 % | 0.987 | 0.985 | 12 % | 0 % | 1 % | seen |
| front / back rounded (yː/uː ʏ/ʊ øː/oː œ/ɔ) | 240 | 97 % | 99 % | 78 % | 93 % | 0.990 | 0.990 | 4 % | 0 % | 0 % | seen |
| voicing, not word-final | 240 | 57 % | 79 % | 29 % | 52 % | 0.917 | 0.943 | 70 % | 11 % | 5 % | weak |
| final devoicing (Rad = [t]) | 154 | 13 % | 36 % | 6 % | 14 % | 0.721 | 0.744 | 96 % | 58 % | 38 % | not representable (accepted as alternative) |
| s / ʃ | 240 | 82 % | 89 % | 34 % | 62 % | 0.950 | 0.932 | 45 % | 17 % | 11 % | seen |
| ç / ʃ (ich / isch) | 240 | 98 % | 99 % | 86 % | 95 % | 0.990 | 0.990 | 6 % | 0 % | 0 % | seen |
| nasal place m / n (before a vowel) | 240 | 98 % | 100 % | 68 % | 87 % | 0.989 | 0.990 | 11 % | 0 % | 0 % | seen |
| final n / ŋ | 240 | 81 % | 95 % | 54 % | 65 % | 0.966 | 0.972 | 42 % | 0 % | 2 % | seen |

| direction (said / expected) | n | sound hit@5 % | word hit@5 % | AUC | raw AUC | front-end accepts this error | verdict |
|---|---|---|---|---|---|---|---|
| vowel_length: said long, expected short | 120 | 44 % | 19 % | 0.850 | 0.842 | 1 % | weak |
| vowel_length: said short, expected long | 120 | 78 % | 41 % | 0.946 | 0.971 | 3 % | seen |
| round_front: said rounded, expected unrounded | 120 | 90 % | 64 % | 0.977 | 0.974 | 0 % | seen |
| round_front: said unrounded, expected rounded | 120 | 100 % | 98 % | 0.996 | 0.997 | 0 % | seen |
| front_back: said front, expected back | 120 | 96 % | 62 % | 0.989 | 0.986 | 0 % | seen |
| front_back: said back, expected front | 120 | 98 % | 93 % | 0.991 | 0.994 | 0 % | seen |
| voicing: said voiced, expected voiceless | 120 | 52 % | 30 % | 0.907 | 0.935 | 0 % | weak |
| voicing: said voiceless, expected voiced | 120 | 62 % | 28 % | 0.927 | 0.950 | 22 % | seen |
| final_devoicing: said devoiced, expected voiced | 154 | 13 % | 6 % | 0.721 | 0.744 | 58 % | not representable (accepted as alternative) |
| s_sh: said s, expected ʃ | 120 | 90 % | 23 % | 0.962 | 0.968 | 32 % | seen |
| s_sh: said ʃ, expected s | 120 | 74 % | 44 % | 0.937 | 0.896 | 1 % | seen |
| ich_isch: said ç, expected ʃ | 120 | 99 % | 95 % | 0.994 | 0.995 | 0 % | seen |
| ich_isch: said ʃ, expected ç | 120 | 98 % | 78 % | 0.986 | 0.985 | 0 % | seen |
| m_n: said m, expected n | 120 | 97 % | 60 % | 0.985 | 0.988 | 0 % | seen |
| m_n: said n, expected m | 120 | 100 % | 76 % | 0.992 | 0.992 | 0 % | seen |
| n_ng: said n, expected ŋ | 120 | 98 % | 98 % | 0.990 | 0.988 | 0 % | seen |
| n_ng: said ŋ, expected n | 120 | 64 % | 10 % | 0.942 | 0.955 | 0 % | seen |

### it (250 clips; flag thresholds at 5 % native FA: sound P ≤ 0.964, word P ≤ 0.855)

| contrast | n | sound hit@5 % | @10 % | word hit@5 % | @10 % | AUC | raw-evidence AUC | P > 0.9 | front-end accepts the error | natives heard as B | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| consonant doubling (fato / fatto) | 240 | 50 % | 52 % | 32 % | 44 % | 0.743 | 0.716 | 72 % | 50 % | – | weak; not representable: said single, expected geminate |
| close / open mid (e/ɛ o/ɔ) | 240 | 66 % | 81 % | 45 % | 57 % | 0.946 | 0.952 | 56 % | 80 % | 15 % | not representable (accepted as alternative) |
| voicing (b/p d/t g/k v/f dʒ/tʃ dz/ts) | 240 | 95 % | 98 % | 87 % | 95 % | 0.984 | 0.996 | 27 % | 3 % | 1 % | seen |
| s / ʃ | 240 | 96 % | 99 % | 85 % | 93 % | 0.988 | 0.993 | 14 % | 0 % | 1 % | seen |
| nasal place m / n (before a vowel) | 240 | 99 % | 100 % | 95 % | 98 % | 0.995 | 0.998 | 4 % | 0 % | 0 % | seen |

| direction (said / expected) | n | sound hit@5 % | word hit@5 % | AUC | raw AUC | front-end accepts this error | verdict |
|---|---|---|---|---|---|---|---|
| gemination: said geminate, expected single | 120 | 2 % | 17 % | 0.502 | 0.437 | 0 % | blind |
| gemination: said single, expected geminate | 120 | 98 % | 48 % | 0.983 | 0.995 | 100 % | not representable (accepted as alternative) |
| mid_vowels: said close, expected open | 120 | 94 % | 75 % | 0.982 | 0.992 | 100 % | not representable (accepted as alternative) |
| mid_vowels: said open, expected close | 120 | 38 % | 16 % | 0.909 | 0.911 | 60 % | not representable (accepted as alternative) |
| voicing: said voiced, expected voiceless | 120 | 92 % | 79 % | 0.978 | 0.997 | 2 % | seen |
| voicing: said voiceless, expected voiced | 120 | 98 % | 95 % | 0.990 | 0.996 | 4 % | seen |
| s_sh: said s, expected ʃ | 198 | 100 % | 94 % | 0.993 | 0.998 | 0 % | seen |
| s_sh: said ʃ, expected s | 42 | 79 % | 43 % | 0.966 | 0.968 | 0 % | seen |
| m_n: said m, expected n | 120 | 98 % | 91 % | 0.993 | 0.998 | 0 % | seen |
| m_n: said n, expected m | 120 | 100 % | 100 % | 0.997 | 0.999 | 0 % | seen |

### pt (250 clips; flag thresholds at 5 % native FA: sound P ≤ 0.945, word P ≤ 0.788)

| contrast | n | sound hit@5 % | @10 % | word hit@5 % | @10 % | AUC | raw-evidence AUC | P > 0.9 | front-end accepts the error | natives heard as B | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nasal / oral vowel (lã / lá) | 240 | 53 % | 61 % | 50 % | 67 % | 0.863 | 0.934 | 53 % | 50 % | – | weak; not representable: said oral, expected nasal |
| close / open mid (e/ɛ o/ɔ: avô / avó) | 240 | 45 % | 62 % | 31 % | 43 % | 0.880 | 0.890 | 67 % | 1 % | 21 % | weak |
| tap / strong r between vowels (caro / carro) | 240 | 88 % | 95 % | 79 % | 81 % | 0.977 | 0.987 | 20 % | 77 % | 1 % | seen; not representable: said tap, expected strong r |
| voicing | 240 | 87 % | 95 % | 72 % | 85 % | 0.962 | 0.983 | 37 % | 0 % | 2 % | seen |
| s / ʃ, before a vowel | 240 | 85 % | 90 % | 75 % | 85 % | 0.945 | 0.960 | 22 % | 0 % | 6 % | seen |
| nasal place m / n (before a vowel) | 240 | 95 % | 97 % | 78 % | 94 % | 0.978 | 0.991 | 12 % | 0 % | 0 % | seen |

| direction (said / expected) | n | sound hit@5 % | word hit@5 % | AUC | raw AUC | front-end accepts this error | verdict |
|---|---|---|---|---|---|---|---|
| nasal: said nasal, expected oral | 120 | 12 % | 26 % | 0.744 | 0.880 | 0 % | blind |
| nasal: said oral, expected nasal | 120 | 95 % | 74 % | 0.982 | 0.987 | 100 % | not representable (accepted as alternative) |
| mid_vowels: said close, expected open | 120 | 77 % | 52 % | 0.953 | 0.972 | 0 % | seen |
| mid_vowels: said open, expected close | 120 | 14 % | 9 % | 0.808 | 0.808 | 2 % | blind |
| tap_trill: said strong r, expected tap | 56 | 46 % | 9 % | 0.928 | 0.946 | 0 % | weak |
| tap_trill: said tap, expected strong r | 184 | 100 % | 100 % | 0.991 | 0.999 | 100 % | not representable (accepted as alternative) |
| voicing: said voiced, expected voiceless | 120 | 82 % | 63 % | 0.952 | 0.974 | 0 % | seen |
| voicing: said voiceless, expected voiced | 120 | 91 % | 82 % | 0.972 | 0.993 | 0 % | seen |
| s_sh: said s, expected ʃ | 201 | 98 % | 88 % | 0.979 | 0.991 | 0 % | seen |
| s_sh: said ʃ, expected s | 39 | 23 % | 8 % | 0.770 | 0.801 | 0 % | blind |
| m_n: said m, expected n | 120 | 91 % | 80 % | 0.973 | 0.990 | 0 % | seen |
| m_n: said n, expected m | 120 | 98 % | 77 % | 0.983 | 0.992 | 0 % | seen |

### ru (250 clips; flag thresholds at 5 % native FA: sound P ≤ 0.884, word P ≤ 0.679)

| contrast | n | sound hit@5 % | @10 % | word hit@5 % | @10 % | AUC | raw-evidence AUC | P > 0.9 | front-end accepts the error | natives heard as B | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hard / soft consonant (Cʲ written in the reference) | 240 | 20 % | 40 % | 19 % | 38 % | 0.717 | 0.888 | 76 % | 60 % | 8 % | blind; not representable: said hard, expected soft |
| unstressed vowel reduction (о / е read as written) | 240 | 72 % | 82 % | 33 % | 51 % | 0.940 | 0.935 | 25 % | 16 % | 8 % | seen |
| voicing, not word-final | 240 | 86 % | 95 % | 50 % | 71 % | 0.969 | 0.982 | 9 % | 0 % | 2 % | seen |
| final devoicing (город = [t]) | 161 | 43 % | 60 % | 12 % | 29 % | 0.857 | 0.854 | 53 % | 12 % | 21 % | weak |
| s / ʂ | 240 | 94 % | 98 % | 60 % | 87 % | 0.976 | 0.994 | 3 % | 0 % | 0 % | seen |
| nasal place m / n (before a vowel) | 240 | 97 % | 100 % | 84 % | 95 % | 0.991 | 0.994 | 3 % | 0 % | 0 % | seen |

| direction (said / expected) | n | sound hit@5 % | word hit@5 % | AUC | raw AUC | front-end accepts this error | verdict |
|---|---|---|---|---|---|---|---|
| hard_soft: said soft, expected hard | 120 | 2 % | 10 % | 0.526 | 0.806 | 20 % | blind |
| hard_soft: said hard, expected soft | 120 | 38 % | 28 % | 0.908 | 0.971 | 100 % | not representable (accepted as alternative) |
| unstressed_vowel: said reduced, expected full | 240 | 72 % | 33 % | 0.940 | 0.935 | 16 % | seen |
| voicing: said voiced, expected voiceless | 120 | 79 % | 44 % | 0.960 | 0.970 | 0 % | seen |
| voicing: said voiceless, expected voiced | 120 | 92 % | 55 % | 0.978 | 0.993 | 0 % | seen |
| final_devoicing: said devoiced, expected voiced | 161 | 43 % | 12 % | 0.857 | 0.854 | 12 % | weak |
| s_sh: said s, expected ʂ | 120 | 96 % | 60 % | 0.980 | 0.995 | 0 % | seen |
| s_sh: said ʂ, expected s | 120 | 92 % | 59 % | 0.972 | 0.993 | 0 % | seen |
| m_n: said m, expected n | 120 | 95 % | 74 % | 0.988 | 0.992 | 0 % | seen |
| m_n: said n, expected m | 120 | 99 % | 94 % | 0.994 | 0.996 | 0 % | seen |

### ko (226 clips; flag thresholds at 5 % native FA: sound P ≤ 0.953, word P ≤ 0.863)

| contrast | n | sound hit@5 % | @10 % | word hit@5 % | @10 % | AUC | raw-evidence AUC | P > 0.9 | front-end accepts the error | natives heard as B | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aspirated / unaspirated (ㅋㅌㅍㅊ vs ㄱㄷㅂㅈ, ㄲㄸㅃㅉ) | 240 | 12 % | 25 % | 18 % | 35 % | 0.640 | 0.877 | 96 % | 50 % | 19 % | blind; not representable: said plain, expected aspirated |
| ㅓ / ㅗ (ɔ / o) | 240 | 82 % | 91 % | 50 % | 74 % | 0.965 | 0.972 | 44 % | 50 % | 2 % | seen; not representable: said o, expected ɔ |
| nasal place m / n (before a vowel) | 240 | 94 % | 98 % | 75 % | 88 % | 0.983 | 0.991 | 30 % | 0 % | 0 % | seen |
| final n / ŋ | 240 | 75 % | 88 % | 36 % | 60 % | 0.955 | 0.955 | 48 % | 0 % | 4 % | seen |
| lax / tense stops (ㄷ / ㄸ) | – | | | | | | | | | | not representable |
| lax / tense stops (ㄱ / ㄲ) | – | | | | | | | | | | not representable |
| lax / tense s (ㅅ / ㅆ) | – | | | | | | | | | | not representable |
| ㅐ / ㅔ (merged in Seoul speech) | – | | | | | | | | | | not representable (not phonemic) |
| r / l (allophones of ㄹ) | – | | | | | | | | | | not representable (not phonemic) |

| direction (said / expected) | n | sound hit@5 % | word hit@5 % | AUC | raw AUC | front-end accepts this error | verdict |
|---|---|---|---|---|---|---|---|
| aspiration: said aspirated, expected plain | 120 | 2 % | 5 % | 0.502 | 0.808 | 0 % | blind |
| aspiration: said plain, expected aspirated | 120 | 22 % | 30 % | 0.777 | 0.947 | 100 % | not representable (accepted as alternative) |
| eo_o: said ɔ, expected o | 120 | 82 % | 55 % | 0.971 | 0.975 | 0 % | seen |
| eo_o: said o, expected ɔ | 120 | 82 % | 46 % | 0.960 | 0.970 | 100 % | not representable (accepted as alternative) |
| m_n: said m, expected n | 120 | 91 % | 62 % | 0.974 | 0.988 | 0 % | seen |
| m_n: said n, expected m | 120 | 98 % | 88 % | 0.991 | 0.995 | 0 % | seen |
| n_ng: said n, expected ŋ | 120 | 82 % | 44 % | 0.967 | 0.965 | 0 % | seen |
| n_ng: said ŋ, expected n | 120 | 68 % | 28 % | 0.943 | 0.946 | 0 % | seen |

## Blind and weak contrasts, ranked by what matters for learners

The ranking weighs how common and how meaning-bearing the error is for learners against how blind
we are today. Hit rates are sound hit@5 % unless stated. The last column is a ceiling: the share of
natives the model already hears as the other member, or that the alignment already maps to the
short or plain alternative.

| # | language: contrast (learner error) | today | raw evidence | ceiling (natives) | what would fix it |
|---|---|---|---|---|---|
| 1 | **ja vowel length**: short for long (おばさん for おばあさん) | not representable (short listed as alternative, 100 %) | AUC 0.96 against a strict long reference | the alignment already maps native long vowels to the short alternative 17–56 % of the time (aː 46 %, iː 56 %, oː 17 %) | a duration detector: native AUC 0.85, see the feasibility section |
| 1 | **ja vowel length**: long for short | blind (5 %, word 7 %) | 0.77 | 13 % heard as long | the same detector |
| 2 | **ja っ**: single for double (きて for きって) | not representable (100 %) | 0.98 against a strict reference | the alignment already maps native geminates to the single alternative 13–94 % of the time (tt 13 %, kk 30 %, ss 94 %) | duration detector, native AUC 0.90 |
| 2 | **ja っ**: double for single | blind (6 %) | 0.48 (no evidence) | 16 % | duration detector |
| 3 | **ko aspiration**: plain for aspirated (달 for 탈) | not representable (tʰ{t}: plain listed as alternative, 100 %); strict: 22 % | 0.95 | 14–23 % of native aspirated stops aligned to the plain alternative | detector on vowel-onset f0 + VOT (AUC 0.91), see below |
| 3 | **ko aspiration**: aspirated for plain | blind (2 %, word 5 %) | 0.81 | 19 % heard as aspirated | the same detector |
| 4 | **ko lax / tense** (달 / 딸, 살 / 쌀) | not representable: both are written `t` / `s` | – | – | simple measures do not separate it (best AUC 0.71 with VOT, 0.64 combined); not a quick win |
| 5 | **zh aspiration**: aspirated for plain (怕 for 爸) | blind (18 %, word 40 %) | **0.96**: calibration hides it | 0 % | recalibration (a feature for aspiration pairs in the sound model), not a new detector; plain for aspirated is seen (72 %) |
| 6 | **ru hard / soft**: soft for hard | blind (2 %) | 0.81 | 8 % | needs work on soft-consonant evidence; weak raw signal |
| 6 | **ru hard / soft**: hard for soft | not representable (hard listed as alternative, 100 %); strict: 38 % | 0.97 | 5–25 % of natives' soft consonants aligned to the hard alternative | first a front-end decision: the alternatives were added because natives vary |
| 7 | **it doubling**: single for double (fato for fatto) | not representable (first copy deletable, 100 %) | 0.995 against a strict reference | 12 % of native geminates align with the first copy dropped | duration detector, native AUC only 0.74 |
| 7 | **it doubling**: double for single | blind (2 %) | 0.44 (no evidence) | – | same |
| 8 | **pt nasal vowels**: nasal for oral | blind (12 %) | 0.88 | – | calibration / evidence |
| 8 | **pt nasal vowels**: oral for nasal (lá for lã) | not representable (coda nasal deletable, 100 %) | 0.99 against a strict reference | – | front-end: a nasal vowel is written as V + optional n |
| 9 | **pt open / close mid**: open for close (avó for avô) | blind (14 %) | 0.81 | 21 % of natives heard as the partner | weak acoustic evidence; regional variation limits it |
| 10 | **de long / short vowels**: long for short | weak (44 %) | 0.84 | 8 % | duration helps a little (native AUC 0.73) |
| 11 | **de final devoicing** (Rad said with [d]) | not representable (t{d}, 58 %); strict 13 % | 0.74 | 38 % of natives' final stops heard as voiced | low value: an accent marker that rarely changes meaning |
| 12 | **ja voicing**: voiced for voiceless | weak (48 %) | 0.93 | 5 % | calibration |
| 13 | **en θ→s, ð→z**: s for θ, z for ð | sound blind (18 % / 22 %) **but word hit 92 % / 88 %** | 0.99 | 0–1 % | the word already flags it. The shown sound probability discounts it (English sound model). The coaching item should come from the word plus the raw evidence. |
| 14 | **it / fr / pt mid vowels** (e/ɛ, o/ɔ) | not representable in it and fr by design (alternatives); pt seen in one direction (77 %), blind in the other | 0.91–0.99 | 2–21 % | leave as is: regional variation |
| 15 | **ru final devoicing**, **de non-final voicing** | weak (43 %, 57 %) | 0.85, 0.94 | 21 %, 5 % | calibration |

**Not phonemic, reported only.** These are English aspiration (pin / spin are both written `p`),
Korean r / l and ㅐ / ㅔ, Japanese r / l, and Spanish b / v. Spanish tap / trill is seen (85 %).
The es front-end writes them with ZIPA's swapped symbols: *pero* `r`, *perro* `ɾ`. That is
deliberate (`es.py::rewrite`), and the contrast is still representable.

## Can a small detector use what we already compute?

### Length and doubling: onset-to-onset intervals from the canonical alignment (natives only)

The measure is the interval from a sound's aligned onset to the next sound's onset. Frames are
20 ms, and dropped slots are skipped. For vowels the interval is divided by the utterance's median
for vowels. For doubling, the measure is the span from the preceding vowel's onset to the
following vowel's onset, divided by the utterance's median for single intervocalic consonants. The
labels (long or short, double or single) are the text's own.

| language: contrast | n long / short | AUC (rate-normalised) | AUC (raw frames) | median long / short | short items flagged at 5 % / 10 % FA on long natives | long items flagged at 5 % / 10 % FA on short natives |
|---|---|---|---|---|---|---|
| **ja vowel length** | 1 338 / 10 865 | **0.846** | 0.846 | 1.80 / 1.00 | 38 % / 70 % | 4 % / 28 % |
| **ja っ** | 140 / 2 048 | **0.898** | 0.893 | 1.50 / 1.00 | **79 % / 81 %** | 26 % / 59 % |
| it doubling | 896 / 4 124 | 0.740 | 0.709 | 1.29 / 1.00 | 23 % / 36 % | 17 % / 27 % |
| de vowel length (stressed) | 1 137 / 1 365 | 0.734 | 0.718 | 1.33 / 1.00 | 20 % / 35 % | 19 % / 33 % |
| en tense / lax (stressed, non-phonemic length) | 450 / 427 | 0.748 | 0.735 | 1.26 / 1.00 | 17 % / 23 % | 23 % / 36 % |

The Japanese per-vowel AUCs run from 0.835 (aː) to 0.891 (iː).

* **Japanese is feasible.** A one-threshold "too short" check on っ would flag 79 % of
  native singletons while flagging only 5 % of native geminates. Learners who skip っ entirely
  should look at least as short as native singletons. For vowels, the same check catches 38 % at
  5 % FA and 70 % at 10 %. Long vowels before a pause or at the end of an utterance are the main
  source of overlap, and a rule for position in the phrase is the obvious first refinement.
* **Italian and German are weak with this measure alone.** Their AUCs are 0.73–0.74. Italian
  geminates are often shortened in fast read speech, and the 20 ms grid is coarse against their
  ~1.3× ratio.

### Aspiration: burst and voicing measures at the aligned stop (natives only)

The stop is the CTC-aligned syllable-initial consonant: every zh stop or affricate, and ko
word-initial p / t / k labelled from the hangul (lax / tense / aspirated). Voicing comes from
`pronunciada/prosody/f0.py`. Five measures were tried:
* `gap`: the voiceless stretch around the stop, i.e. closure plus release;
* `vot`: voicing onset minus the burst, where the burst is the steepest rise of 2.5–8 kHz energy
  inside the voiceless stretch;
* `hf`: high-frequency energy, both at the burst and over the release;
* the stop's onset-to-onset interval, normalised;
* f0 at the vowel onset, in semitones relative to the utterance median.

| language: pair | n | gap | vot | HF burst | HF ratio | interval | f0 onset | combined (5-fold, grouped by clip) |
|---|---|---|---|---|---|---|---|---|
| zh aspirated vs plain (all) | 1 212 / 3 263 | 0.62 | 0.68 | 0.54 | 0.57 | **0.82** | 0.52 | 0.83 |
| zh aspirated vs plain (p t k only) | 605 / 1 861 | 0.62 | 0.69 | 0.50 | 0.61 | **0.88** | 0.58 | **0.89** |
| ko aspirated vs lax | 197 / 936 | 0.61 | 0.60 | 0.53 | 0.52 | 0.62 | **0.91** | **0.91** |
| ko aspirated vs tense | 197 / 83 | 0.59 | **0.83** | 0.54 | 0.48 | 0.75 | 0.79 | 0.85 |
| ko lax vs tense | 936 / 83 | 0.46 | 0.71 | 0.51 | 0.46 | 0.59 | 0.44 | 0.64 |

* **Korean aspirated vs lax** separates on **f0 at the vowel onset** (AUC 0.91; median +4.3 vs
  −0.6 semitones). This is the known Seoul tonogenesis: in today's speech, aspirated and lax stops
  differ more in pitch than in VOT. Adding VOT takes aspirated vs tense to 0.85.
  * Lax vs tense is not separable with these simple cues (0.64). The textbook cues are tense
    voice quality and a short VOT; the VOT is visible (median 37 ms vs 12 ms, AUC 0.71) but too
    noisy alone.
* **Mandarin** is best served by the alignment's own interval (0.88 on p / t / k).
  * The burst / VOT estimates are poor here (0.68): a 10 ms pitch tracker and a 5 ms energy
    track do not resolve VOTs of 15–70 ms reliably.
  * The raw `llr_sub` already reaches 0.96–0.99 on the swaps. For zh, the missing piece is
    calibration rather than a detector.

## What the front-end emits for minimal pairs

| language | pair → reference sounds (alternatives in braces) |
|---|---|
| ja | おばさん `o b a s a ŋ` / おばあさん `o b aː{a} s a ŋ`; きて `k i t e` / きって `k i tt{t} e`. The long and double forms are representable, but the short and single forms are accepted as alternatives. |
| it | fato `f a t o` / fatto `f a t{∅} t o`. The first copy is deletable. |
| ko | 달 `t{d} a l` / 딸 `t a l` / 탈 `tʰ{t} a l`; 살 `s a l` / 쌀 `s a l`. Lax and tense differ only by the voiced alternative, and ㅅ / ㅆ are identical. Aspirated accepts plain. |
| zh | 怕 `pʰ a` / 爸 `p a`; 山 `ʂ a n` / 三 `s a n`. Fully representable. |
| ru | рад `r ɑ t{ts}` / ряд `rʲ{r} a t{ts}`; мол `m o ɭ` / мёл `mʲ{m} ɛ ɭ`; брат `… t{ts}` / брать `… ts`. A soft consonant accepts its hard partner, and a final hard t accepts ts. |
| pt | lã `l a{ɐ̃} n{∅,ŋ}` / lá `l a`; avô `a v o` / avó `a v ɔ`; caro `k a ɾ{r} o` / carro `k a ʁ{x,h,r,ɾ} o`. Oral is accepted for nasal, and tap for strong r. |
| de | Rad `ʀ a{ɑː} t` / Rat `ʀ ɑː{aː} t`; Miete `m iː t …` / Mitte `m ɪ t …`. |
| en | pin `p ɪ n` / spin `s p ɪ n`. Aspiration is not written. |
| es | pero `p e r o` / perro `p e ɾ o` (ZIPA's swapped convention); baca = vaca `b a k a`. |

## Recommendation (for the detector work, not done here)

1. **Japanese mora-timing detector (vowel length + っ).** This is the largest blind spot that
   carries meaning.
   * The front-end accepts the error by construction, and the acoustic model cannot see it.
   * The timing we already compute separates natives well: AUC 0.85 for vowels and 0.90 for っ.
   * It also gives a clean coaching tip: "hold the vowel for two beats" or "pause before the
     consonant (っ)".
2. **Korean aspiration detector from the vowel-onset pitch (+ VOT).** It reaches AUC 0.91 on
   aspirated vs lax and 0.85 on aspirated vs tense. The coaching tip is "start the vowel higher and
   with a puff of air (ㅋㅌㅍㅊ)". Lax vs tense is not feasible with simple cues; leave it.

Mandarin aspiration and English θ / ð should **not** get detectors. Their raw evidence is already
0.96–0.99, so a calibration change, or a coaching item read off `llr_sub`, is the cheaper fix.

Data: FLEURS validation only. No held-out set was touched, and no package file was changed. The
full numbers, including per direction, per symbol pair and native alternative usage, are in
`results/tables/contrast_audit.json`.
