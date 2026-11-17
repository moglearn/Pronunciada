# Coaching feedback and audio-quality gate

Design, schema and app guidance: [docs/COACHING.md](../docs/COACHING.md).

## Round 19: tips only where the scorer itself did not accept the sound or word

**What was wrong.** `tips()` turned every coaching candidate into advice, even when the scorer
accepted the word and the sound: "their" at 93 % with /ɛ/ at 90 % still got "listen to /ɛ/ and
try it again". The candidate list is tuned for precision on learner speech (about one in three
"balanced" candidates is wrong even there), so on native speech it lists about 1–4 % of the
sounds in every language, English included. Per native FLEURS validation clip at "balanced":
en 3.6, ja 2.2, es 3.9, fr 3.3, de 1.2, it 2.3, pt 0.9, ru 4.5, ko 1.0, zh 0.07 candidates. A
first plan, tuning per-language thresholds down to the English native rate, was dropped because
English itself is at 3.6 per clip.

**The change.** `pronunciada/tips.py`: with `result` given, a `substitution` (including
feature and contrast claims), `missing` or `mispronounced_word` item gives a tip only when its
sound's prob or its word's prob is below 0.5 at the result's strictness (`include_accepted=True`
switches the gate off). Nothing is fitted: 0.5 is the scorer's own acceptance line. Coaching
items, model.json and every threshold are unchanged (`res.coaching()` output byte-identical on
200 speechocean762 test and 100 OMPAL test utterances, and on 5 FLEURS clips per language at all
three sensitivities). Other tip kinds are not gated.

**Before → after** (test splits, strictness 0.5, "balanced", `max_tips=3`; `bench/coach/tips_r19.py`
→ `tables/coaching_tips_r19.json`). Natives, FLEURS test, 250 clips per language (every
pronunciation tip is a false alarm):

| lang | clips with a pronunciation tip | pronunciation tips per clip |
|---|---|---|
| en | 85 % → 28 % | 1.66 → 0.32 |
| zh | 10 % → 9 % | 0.11 → 0.10 |
| ja | 70 % → 28 % | 1.12 → 0.32 |
| es | 74 % → 19 % | 1.35 → 0.24 |
| fr | 89 % → 32 % | 1.96 → 0.42 |
| de | 49 % → 22 % | 0.75 → 0.27 |
| it | 76 % → 15 % | 1.38 → 0.16 |
| pt | 42 % → 14 % | 0.58 → 0.16 |
| ru | 96 % → 43 % | 2.18 → 0.55 |
| ko | 52 % → 23 % | 0.76 → 0.27 |

Learners (English: speechocean762 test, wrong = expert phone accuracy < 2 on a phone aligned
to ours; Mandarin: OMPAL test learners, teachers' consonant / vowel marks):

| | speechocean762 (2,500 utt.) | OMPAL (663 utt.) |
|---|---|---|
| (a) utterances with a wrong sound that get a pronunciation tip | 61.4 % → **33.6 %** | 23.1 % → 21.3 % |
| (b) pronunciation tips that are right (sound-level: on a wrong sound; word-level: the word has one) | 63.6 % → 69.1 % | 70.0 % → 71.7 % |
| (c) utterances with no wrong sound that get a pronunciation tip | 21.2 % → 6.9 % | 2.0 % → 1.8 % |
| pronunciation tips in all | 2,450 → 1,233 | 50 → 46 |

**What it costs.** On English learners the gate halves the tips: 45 % fewer utterances with a
real error get a pronunciation tip (more than a third), because many of the experts' errors sit
in sounds and words the scorer accepts at strictness 0.5. The tips that remain are right more
often, and clean utterances get a third as many. A stricter `rescore` (strictness > 0.5) lowers the
probabilities and so lets more through (not measured here). Mandarin barely moves (its candidates are already few and mostly unaccepted).
Native pronunciation tips per clip fall by a factor of 2.7–8.6 outside Mandarin, but 14–43 % of native clips (Mandarin 9 %)
still get one pronunciation tip.

## Coaching round 4: feature-level claims from attribute heads

**Question.** Most flagged sounds are reported as "this sound was off" because naming the exact replacement is fragile (round 2: 8 % of real English substitutions named). Can small heads on the acoustic model's internal layers say *which feature* was wrong instead?

**Short answer.** The heads hear the audio, not the reference (no leak: on planted swaps they report the audio's sound 68–95 % of the time, length weakest). On real learners they double the share of substitutions that get a correct feature-level diagnosis at "balanced" (7.6 % → 15.7 % on speechocean762 test, claims right 84 %), but the 40 % target is reached only at "eager" (40.5 % with a claim, 72 % right, 30 % with a right claim), where native speakers get 0.2–2.5 wrong claims per clip. The native cap (≤ 0.3 per clip at "balanced") is what limits coverage. Used as stand-alone detectors, the heads hear contrasts the scorer is blind to: Korean aspiration AUC 0.97 (scorer 0.64), Russian soft / hard 0.95 (0.72), Japanese vowel length 0.95 (0.75).

### Heads (`pronunciada/attr_heads.py`, `data/coach/attr_heads.npz`, 167 KB)

Per feature, a multinomial logistic regression on the round-12b pooled vector of the sound (enc1 | enc3 | enc5 mean over its aligned frames ± 1) and optionally the mean of its two neighbours' vectors. Trained on FLEURS **train** natives, all ten languages pooled (207,341 sounds, ≤ 2,500 per language and head), labels from the reference sounds (`articulatory.py`, grouped: place in 5 classes, vowel height in 3). Six variants per head (layers × context); the kept one has the best honest accuracy on the leak check. Numpy at runtime.

### Leak check first (FLEURS validation, contrast-audit swaps)

The audio has sound X, the reference demands Y; the vector is pooled under the *edited* reference's alignment. Honest = the head reports X's class; follows = it reports Y's. Natives = unedited sounds of the same clips.

| head | kept variant | natives | swaps | honest | follows reference | honest by language |
|---|---|---|---|---|---|---|
| voicing | e135+ctx | 0.968 | 2235 | 0.916 | 0.084 | de 0.85, en 0.92, es 0.99, fr 0.95, it 0.98, ja 0.90, pt 0.95, ru 0.86 |
| aspirated | e13+ctx | 0.994 | 480 | 0.838 | 0.163 | ko 0.81, zh 0.86 |
| manner | e135+ctx | 0.956 | 1230 | 0.894 | 0.066 | en 0.92, ja 0.93, pt 0.85, zh 0.86 |
| place | e135 | 0.957 | 6000 | 0.948 | 0.042 | de 0.96, en 0.96, es 0.99, fr 0.95, it 0.99, ja 0.93, ko 0.92, pt 0.90, ru 0.96, zh 0.94 |
| palatalised | e135+ctx | 0.994 | 240 | 0.758 | 0.242 | ru 0.76 |
| lateral | e135+ctx | 0.992 | 480 | 0.944 | 0.056 | en 0.94, zh 0.95 |
| height | e135 | 0.923 | 480 | 0.890 | 0.085 | en 0.92, ru 0.86 |
| backness | e135 | 0.941 | 1104 | 0.896 | 0.092 | de 0.91, fr 0.93, ru 0.79, zh 0.90 |
| rounded | e135 | 0.967 | 675 | 0.944 | 0.056 | de 0.95, fr 0.95, ru 0.93 |
| nasal | e135+ctx | 0.999 | 240 | 0.942 | 0.058 | fr 0.94 |
| long | e13+ctx | 0.989 | 720 | 0.678 | 0.322 | de 0.78, ja 0.63 |

Earlier layers alone were *less* honest (enc1: voicing 0.88, aspiration 0.59, palatalisation 0.55, length 0.52), so the leak, where present, is mostly the head not hearing the difference rather than reading the word: the activations do not see the reference, only the pooling span does. All heads were kept; length (0.68 honest; Japanese 0.63) and palatalisation (0.76) are the weakest.

### Claim policy (speechocean762 train) and the native cap

A head claims "feature f differs, heard class c" on a flagged sound when 1 − P(expected class) ≥ q_f, the sound's coaching score ≥ t_f, the language's best competing sound (round-1 heard candidate) also has class c, no accepted alternative changes f, and a named heard sound (if any) agrees. Per sensitivity, a greedy choice of (q_f, t_f) per head maximises right coverage of real substitutions (train, out-of-fold flags) with claim precision ≥ 0.8 / 0.75 / 0.75 and at most 0.15 / 0.3 / – wrong claims per English FLEURS native clip; other languages get a coaching-score floor where their natives exceed the cap. Rules fixed before the held-out run.

| sensitivity | policy (head: q, t) | train: with claim / with right claim / precision |
|---|---|---|
| careful | voicing 0.999, 0.7; lateral 0.9, 0.0; backness 0.99, 0.8; rounded 0.999, 0.0 | 12.0 % / 10.5 % / 0.860 |
| balanced | voicing 0.999, 0.8; place 0.999, 0.6; lateral 0.5, 0.0; backness 0.99, 0.8; rounded 0.999, 0.0 | 17.9 % / 15.7 % / 0.868 |
| eager | voicing 0.5, 0.4; manner 0.5, 0.6; place 0.5, 0.0; lateral 0.5, 0.0; backness 0.5, 0.0; rounded 0.99, 0.0 | 41.8 % / 33.6 % / 0.761 |

Heads never enabled for claims: aspiration, palatalisation, nasality and length (English learners give no labelled examples, so no gate can be fitted) and vowel height (best train precision 0.65).

Wrong feature claims per FLEURS validation native clip (~10 s; every claim on a native is wrong):

| lang | flags per clip (balanced) | claims per clip careful / balanced / eager | balanced floor |
|---|---|---|---|
| en | 3.96 | 0.15 / 0.30 / 2.46 | 0.0 |
| zh | 0.14 | 0.02 / 0.04 / 0.22 | 0.0 |
| ja | 2.48 | 0.02 / 0.10 / 1.76 | 0.0 |
| es | 3.97 | 0.12 / 0.18 / 1.34 | 0.0 |
| fr | 4.01 | 0.12 / 0.26 / 2.31 | 0.6 |
| de | 1.67 | 0.04 / 0.10 / 1.36 | 0.0 |
| it | 2.37 | 0.01 / 0.06 / 0.89 | 0.0 |
| pt | 1.32 | 0.01 / 0.05 / 0.93 | 0.0 |
| ru | 4.43 | 0.06 / 0.16 / 2.21 | 0.0 |
| ko | 0.92 | 0.01 / 0.13 / 1.49 | 0.0 |

### Held-out, once (speechocean762 test, 568 real substitutions with the pronounced phone)

| sensitivity | substitutions with a feature claim | with a right claim | claim precision | with any right feature (claims or the named sound) | today (named sound only) | claims on sounds the experts scored 2 |
|---|---|---|---|---|---|---|
| careful | 8.8 % | 7.7 % | 0.873 | 13.2 % | 7.0 % | 76 |
| balanced | 13.6 % | 11.6 % | 0.843 | 15.7 % | 7.6 % | 156 |
| eager | 40.5 % | 29.9 % | 0.716 | 31.5 % | 8.3 % | 1051 |

Per head at "balanced" (test): backness 0.821 (n 28), lateral 0.875 (n 16), place 0.800 (n 25), rounded 0.875 (n 8), voicing 1.000 (n 6). At "eager": backness 0.680 (n 97), lateral 0.875 (n 16), manner 0.712 (n 59), place 0.762 (n 63), rounded 0.818 (n 22), voicing 0.619 (n 42).

Curve without the native cap (balanced flags, every head at the same q, t; test):

| q | t | with claim | precision |
|---|---|---|---|
| 0.5 | 0.0 | 56.0 % | 0.691 |
| 0.5 | 0.6 | 47.4 % | 0.683 |
| 0.5 | 0.8 | 28.2 % | 0.703 |
| 0.9 | 0.0 | 50.5 % | 0.697 |
| 0.9 | 0.6 | 43.3 % | 0.698 |
| 0.9 | 0.8 | 25.9 % | 0.726 |
| 0.99 | 0.0 | 37.0 % | 0.727 |
| 0.99 | 0.6 | 32.2 % | 0.721 |
| 0.99 | 0.8 | 20.6 % | 0.765 |
| 0.999 | 0.0 | 20.1 % | 0.778 |
| 0.999 | 0.6 | 18.3 % | 0.782 |
| 0.999 | 0.8 | 12.7 % | 0.824 |

**Where it falls short.** ≥ 40 % coverage at ≥ 0.75 precision is not reached on test: the closest points are 40.5 % at 0.72 (eager) and 20 % at 0.78 (q 0.999). At "balanced" the native cap keeps it to 13.6 %. The offline evaluation does not drop sounds of words the omission channel marks missing (the runtime does), a small difference.

L2-ARCTIC / EpaDB test (report only, balanced): l2arctic 2.1 % of annotated substitutions get a claim, precision 0.800, any right feature 2.9 %; epadb 0.6 % of annotated substitutions get a claim, precision 0.682, any right feature 1.7 %. These corpora's annotated substitutions are mostly not flagged at all (round 2), so the heads add little there.

OMPAL (Mandarin learners, balanced; teachers mark the type only, a claim on an initial counts right when the consonant is marked wrong): dev: initials 15 characters with a claim, precision 0.800; finals 9, 1.000; test: initials 4 characters with a claim, precision 1.000; finals 4, 0.750. Coverage is tiny: Mandarin flags are rare and the aspiration and retroflex heads are not enabled for claims (no learner labels to gate them).

### Side result: the heads as stand-alone detectors on blind / weak contrasts

Score = 1 − P(the reference sound's class) of the matching head; positives = the audit's swaps, negatives = native sounds of the same members (the audit used all native sounds). Report only.

| contrast | head | swaps | heads AUC | heads hit@5 % | audit AUC | audit hit@5 % | heads AUC per direction (A>B / B>A) |
|---|---|---|---|---|---|---|---|
| ko/aspiration | aspirated | 240 | 0.971 | 84.6 % | 0.640 | 12.5 % | 0.962 / 0.979 |
| zh/aspiration | aspirated | 240 | 0.988 | 96.2 % | 0.886 | 45.0 % | 0.980 / 0.996 |
| ru/hard_soft | palatalised | 240 | 0.954 | 74.6 % | 0.717 | 20.0 % | 0.922 / 0.986 |
| pt/nasal | nasal | 240 | 0.554 | 10.0 % | 0.863 | 53.3 % | 0.615 / 0.492 |
| it/gemination | long | 240 | 0.633 | 12.1 % | 0.743 | 50.0 % | 0.814 / 0.451 |
| de/vowel_length | long | 240 | 0.864 | 46.7 % | 0.898 | 61.3 % | 0.806 / 0.923 |
| ja/vowel_length | long | 240 | 0.954 | 73.8 % | 0.755 | 20.8 % | 0.914 / 0.994 |
| ja/gemination | long | 240 | 0.881 | 53.8 % | 0.699 | 33.3 % | 0.771 / 0.992 |

A>B = the audio has the contrast's first member (e.g. aspirated, soft, long, geminate, nasal) and the reference demands the second. Korean lax / tense cannot be written in the reference, so it is not testable. The heads separate Korean and Mandarin aspiration, Russian soft / hard and Japanese length well in both directions, where the shipped scorer is blind in at least one; separate detectors are still needed for Italian doubling and Portuguese nasal vowels (the front-ends write those as two sounds, so the per-sound heads do not see them: AUC 0.63 / 0.55).

### Examples (speechocean762 test, balanced; a feature tip without a named sound)

- “IN COLLEGE YOU COULD GET AWAY WITH THINGS”, /l/ in “COLLEGE”: claim lateral yes → no (p 0.9341); tip: “Let the air pass along the sides of the tongue for /l/ in “COLLEGE”.” Experts: COLLEGE ['K', 'AH1', 'L', 'IH0', 'JH'] scores [2.0, 2.0, 0.0, 1.2, 1.6], pronounced L→R.
- “TIM IS NOT A REAL STUDENT”, /u/ in “STUDENT”: claim rounded yes → no (p 0.9998); tip: “Round your lips for /u/ in “STUDENT”; they sounded spread.” Experts: STUDENT ['S', 'T', 'UW1', 'D', 'N', 'T'] scores [1.6, 1.2, 0.0, 0.0, 0.0, 0.0], pronounced UW→AH, D→D*, N→<DEL>, T→<unk>.

No score changed: 50 clips (5 per language, `PYTHONHASHSEED=0`) identical to the previous commit (the scorer keeps the activations it already computes in the result's cache). coaching() with the heads: median 1.1 ms, 90th percentile 1.7 ms per 10 s.

## Round 12b: coaching scores that read the acoustic model's internal layers

Full write-up: `results/round12_sound.md` §12b.

**English.** The coaching score's sound inputs now come from `sound_E`: the English sound heads,
plus a projection of the ZIPA enc1/enc3/enc5 activations pooled over each sound, plus the margin
to a native FLEURS-en prototype of the expected sound. The English coaching logistic was refitted
by `bench/coach/fit_v2.py` on sound_E out-of-fold inputs (speechocean762 train) and is stored as
`by_lang.en` in `pronunciada/data/coach/model.json`. Other languages keep the global model: refitting
it on English-only inputs raised their native flags by 10–20 %.

Thresholds (careful / balanced / eager): 0.628 / 0.489 / 0.363, formerly 0.653 / 0.514 / 0.390.

Train out-of-fold (5-fold, speaker-grouped), precision at recall 0.20 / 0.33 / 0.47:

| | precision at recall 0.20 / 0.33 / 0.47 | recall at precision 0.8 / 0.7 / 0.6 |
|---|---|---|
| round 11 | 0.811 / 0.711 / 0.613 | 0.22 / 0.35 / 0.49 |
| round 12b | **0.850 / 0.756 / 0.648** | 0.27 / 0.40 / 0.53 |

**Held-out, speechocean762 test, run once.** 47,041 phones; Azure is its cached phoneme accuracy.
CIs are 1,000 speaker bootstraps.

| | PR-AUC | precision at recall 0.20 | 0.33 | 0.47 |
|---|---|---|---|---|
| round 11 | 0.546 [0.485, 0.598] | 0.743 [0.656, 0.813] | 0.643 [0.562, 0.710] | 0.551 [0.483, 0.611] |
| **round 12b** | **0.570 [0.510, 0.622]** | **0.799 [0.711, 0.856]** | **0.684 [0.608, 0.753]** | **0.577 [0.511, 0.646]** |
| 12b − 11 (paired) | +0.024 [+0.019, +0.029] | +0.056 [+0.035, +0.075] | +0.041 [+0.026, +0.056] | +0.026 [+0.018, +0.043] |
| Azure | 0.382 [0.324, 0.437] | 0.559 | 0.464 | 0.384 |

**At the shipped operating points** (`bench/coach/eval_v2.py`, `azure_yardstick.py`):

| | round 11 | round 12b | Azure at our recall |
|---|---|---|---|
| careful: precision / recall | 0.75 / 0.20 | **0.77 / 0.24** | 0.53 |
| balanced: precision / recall | 0.65 / 0.33 | **0.66 / 0.36** | 0.44 |
| eager: precision / recall | 0.55 / 0.47 | **0.56 / 0.50** | 0.37 |
| named sound right within one feature (balanced) | 0.77 | 0.79 | |
| share of substitutions named (balanced) | 7.7 % | 9.2 % | |

**Costs on English natives** (FLEURS-en validation, 393 clips): flags per clip at balanced rose
from 3.0 to 4.2. Named sounds per clip rose from 0.27 to 0.30, at the 0.3 cap. Word and sound
probabilities are unchanged.

**Mandarin consonants / vowels** (`pronunciada/coach_zh.py`, `bench/calib/ompal_sound_emb.py`).
The model is a per-sound logistic on [logit of the coaching score, margin to the native FLEURS-zh
prototype, ridge projection of the pooled activations], separate for initials and finals. It is
fitted on OMPAL **dev**, 5-fold speaker-grouped.

CV AUC, OMPAL dev:

| | before | after | gain [95 % CI] |
|---|---|---|---|
| initials | 0.816 | 0.925 | +0.109 [+0.062, +0.168] |
| finals | 0.890 | 0.904 | +0.013 [+0.003, +0.025] |

Held-out, OMPAL **test**, run once:

| | round 11 | round 12b |
|---|---|---|
| consonant, careful | – | **0.71 / 0.21** |
| consonant, balanced | 0.27 / 0.06 | **0.71 / 0.25** |
| consonant, eager | 0.33 / 0.09 | **0.59 / 0.28** |
| vowel, balanced | – | **0.67 / 0.18** |
| vowel, eager | 0.46 / 0.12 | **0.60 / 0.23** |

Cost on FLEURS-zh natives: flags per clip at balanced rose from 0.02 to 0.14.

The first evaluation call scored OMPAL units with the general score instead of the Mandarin one.
`fit_v2.ompal_units` did not pass `lang`, which gave 0.26 precision. That was a wiring bug, fixed
and re-run; no thresholds were changed.

## Yardstick: Azure Pronunciation Assessment, speechocean762 test (round 11 model; round 12b numbers above)

Which sounds to flag, on the same 47,041 expert-scored phones (error = expert score < 2, 18.6 % of phones). Ours = the coaching score; Azure = 100 − its phoneme accuracy from the cached responses (no new requests), aligned to the corpus phones as in `bench/eval_azure_so762.py`. Azure is thresholded to reach our recall. 95 % CIs: 1,000 speaker bootstraps (`bench/coach/azure_yardstick.py`). This is a third, report-only look at speechocean762 test; nothing was changed after it.

| operating point | recall | precision ours [CI] | precision Azure [CI] | ours − Azure [CI] |
|---|---|---|---|---|
| careful | 0.196 | 0.746 [0.689, 0.793] | 0.561 [0.496, 0.621] | [0.144, 0.228] |
| balanced | 0.326 | 0.647 [0.597, 0.693] | 0.464 [0.416, 0.513] | [0.157, 0.209] |
| eager | 0.465 | 0.554 [0.510, 0.594] | 0.386 [0.347, 0.424] | [0.148, 0.190] |
| precision–recall AUC | | 0.546 [0.485, 0.598] | 0.382 [0.324, 0.437] | [0.144, 0.180] |

What it heard instead, on the experts' 568 real substitutions (pronounced phone known). The responses were requested with 5 n-best phonemes per phone, so Azure's candidates exist:

| | substitutions named | exact | within one feature |
|---|---|---|---|
| ours (balanced) | 44 (8 %) | 0.636 | 0.773 |
| Azure: top candidate ≠ expected | 442 (78 %) | 0.369 | 0.493 |
| Azure: same, its lowest-scored 44 only (our coverage) | 44 | 0.273 | 0.455 |
| Azure: best candidate other than expected | 475 (84 %) | 0.358 | 0.476 |

OMPAL (Mandarin): not comparable by error type: the cached zh-TW responses carry no phoneme labels (empty strings) and zh-CN gives one score per syllable ('ta 1'), not separate initial / final / tone scores; Azure has no tone score. Only word-level any-error detection is comparable (results/ompal.md).

## Round 3: word-level reading miscues and delivery flags

Scripts `bench/coach/dump_r3.py`, `bench/coach/fit_r3.py`; definitions in docs/COACHING.md §3b. Thresholds from FLEURS validation natives (250 clips per language, Korean 226), each flag ≤ 5 % of native clips (extra speech ≤ 2 %).

| lang | extra speech bar (sounds) | natives flagged | removed word caught | long pause | mid-phrase pause | rushed punctuation | flat pitch | slow | fast | any delivery flag |
|---|---|---|---|---|---|---|---|---|---|---|
| en | 5 | 0.0 % | 27 % (250) | 4.0 % | 0.4 % | 2.4 % | 4.0 % | 4.0 % | 4.0 % | 16.0 % |
| zh | 7 | 2.0 % | 14 % (250) | 4.4 % | 2.8 % | 3.2 % | 4.0 % | 4.4 % | 4.0 % | 21.2 % |
| ja | 7 | 1.6 % | 5 % (250) | 4.0 % | 1.2 % | 4.8 % | 4.0 % | 4.0 % | 4.0 % | 18.8 % |
| es | 4 | 2.0 % | 44 % (250) | 4.0 % | 2.0 % | 2.8 % | 4.0 % | 4.0 % | 4.0 % | 18.8 % |
| fr | 4 | 1.6 % | 29 % (250) | 4.0 % | 1.6 % | 2.8 % | 4.0 % | 4.0 % | 4.0 % | 17.2 % |
| de | 3 | 2.0 % | 48 % (250) | 4.4 % | 3.2 % | 2.8 % | 4.0 % | 4.0 % | 4.0 % | 20.0 % |
| it | 3 | 1.2 % | 54 % (250) | 4.8 % | 3.2 % | 4.8 % | 4.0 % | 4.0 % | 4.0 % | 22.0 % |
| pt | 4 | 2.0 % | 41 % (250) | 4.0 % | 2.8 % | 2.8 % | 4.0 % | 4.0 % | 4.0 % | 20.0 % |
| ru | 9 | 2.0 % | 10 % (250) | 4.0 % | 2.0 % | 2.8 % | 4.4 % | 4.0 % | 4.0 % | 20.0 % |
| ko | 8 | 1.8 % | 12 % (226) | 4.4 % | 1.8 % | 4.9 % | 4.4 % | 4.4 % | 4.0 % | 22.1 % |

Extra speech catches a removed inner word in 5–54 % of clips: well for multi-sound words in Romance languages and German, poorly in Japanese, Russian, Korean and Mandarin, where natives' decodes already leave several sounds outside the expected words, so the 2 % bar is high (7–9 sounds).

Hit rate by bar (pooled curve per language in `results/tables/coaching_r3.json`).

Sanity check on speechocean762 train (English thresholds; mean expert 0–10 sentence score, flagged minus not flagged):

| flag | flagged | fluency | prosody |
|---|---|---|---|
| long_pause | 246 (9.8 %) | -2.96 | -2.72 |
| pause_mid_phrase | 71 (2.8 %) | -2.92 | -2.70 |
| rushed_punctuation | 0 (0.0 %) | – | – |
| flat_pitch | 1276 (51.0 %) | +0.01 | -0.07 |
| slow | 1679 (67.2 %) | +0.40 | +0.39 |
| fast | 28 (1.1 %) | -2.36 | -2.23 |
| any | 2184 (87.4 %) | -0.51 | -0.49 |

`long_pause`, `pause_mid_phrase` and `fast` go with ~2.2–3 points lower expert fluency and prosody. `slow` and `flat_pitch` do not (they fire on most of these short, carefully read learner prompts and the flagged ones score no lower), so they are off by default (`coaching(delivery="all")`). `rushed_punctuation` cannot be checked there (no internal punctuation in the prompts).

Russian `present_prob` nondeterminism: `pronunciada/text/ru.py` built some alternatives by iterating a Python `set` of strings, so their order (and the summation order in the lattice) depended on `PYTHONHASHSEED`; now an ordered tuple. Two hash seeds now give identical Russian outputs on 30 clips; against the old code the largest difference is 9e-16 (the old code differed from itself by 4e-16 between seeds). No other language reads that file.

Scores unchanged: the scorer now keeps the pitch track it computes for the prosody models (the same `track(wav)` ProsodyScorer would compute) and the waveform in the result's cache; 50 clips (5 per language, `PYTHONHASHSEED=0`) are identical to the previous commit as repr() strings. coaching() with miscues and default delivery flags: median 0.5 ms per 10 s; `delivery="all"` in French / Korean adds a pitch track on first use (~10 ms).

## Round 2: a coaching operating point fitted on learner data

Data. Fitted on speechocean762 **train** (2,500 utterances of Mandarin-L1 children and adults reading English; per-phone expert scores 0–2; 864 phones with the pronounced phone, 443 marked deleted) and OMPAL **dev** (Mandarin learners, per-character tone / consonant / vowel marks). FLEURS validation (native) sets the per-language caps. Held out: speechocean762 test and OMPAL test; L2-ARCTIC test and EpaDB test are report-only (non-commercial licences). Scripts: `bench/coach/dump.py` (public Scorer outputs), `bench/coach/fit_v2.py`, `bench/coach/eval_v2.py`.

**Held-out discipline, plainly.** The held-out sets were evaluated twice. The first fit used the English sound calibrator's logits on speechocean762 train, where that calibrator was itself fitted (in-sample), so the coaching model over-trusted them: out-of-fold precision 0.70 on train but 0.512 on test at "balanced". The fix (the calibrator's out-of-fold logits on train, `bench/calib/train_sound.py`) was found from that first test run; everything else (rules, targets, grid) is unchanged. Both runs are reported (`results/tables/coaching_v2_heldout_leaky.json`, `coaching_v2_heldout.json`).

### 1. Which sounds to flag (speechocean762)

Coaching score = logistic regression (monotone in each input) on the calibrated lenient and strict log-odds, `llr_sub`, `z_llr_sub`, competitor log-ratio, log duration, nucleus, `llr_del`, peak posterior and blank share; target expert phone score < 2. Thresholds = lowest score with out-of-fold (5-fold, speaker-grouped) precision ≥ 0.8 / 0.7 / 0.6. OOF AUC 0.826 (the strict sound probability alone: 0.821). Coefficients (standardised): zl +0.38, zs -1.65, llr_sub -0.10, z_llr_sub -0.06, log_ratio_comp +0.00, log_frames -0.09, nucleus +0.14, llr_del -0.11, peak_post +0.12, blank_mean +0.04.

| setting | threshold | train OOF precision | train OOF recall | **test precision** | **test recall** | test precision / recall, error = score < 1.5 | flags per FLEURS native clip (en) |
|---|---|---|---|---|---|---|---|
| round 1 (prob < 0.5) | – | 0.825 | 0.193 | 0.773 | 0.175 | 0.641 / 0.398 | – |
| careful | 0.653 | 0.800 | 0.221 | 0.746 | 0.196 | 0.574 / 0.414 | 1.42 |
| balanced | 0.514 | 0.700 | 0.351 | 0.647 | 0.326 | 0.431 / 0.596 | 2.90 |
| eager | 0.390 | 0.600 | 0.487 | 0.554 | 0.465 | 0.323 / 0.744 | 5.66 |

Test precision is 5 points below the train target at every setting (0.65 at "balanced"). The gain over round 1 is coverage: "balanced" finds 1.9× as many expert-marked errors (recall 0.326 vs 0.175).

### 2. When to name the heard sound (speechocean762 pronounced phones)

Candidate as in round 1. Gate fitted on the 642 "balanced"-flagged train sounds with a pronounced-phone substitution label: per sound class ON when exact-or-within-one-feature precision ≥ 0.75 on ≥ 15 items; the setting with the most correct names. Chosen: log-ratio ≥ 3.0, heard posterior ≥ 0.3; ON classes approximant, fricative, stop, vowel; OFF affricate, click, flap, implosive, nasal, trill (too few or too inaccurate). Per language the posterior bar is raised until "balanced" makes ≤ 0.3 wrong named sounds per FLEURS native clip: de 0.3, en 0.4, ja 0.375, es 0.375, fr 0.4, it 0.35, ru 0.35.

| | named sounds on test | on real substitutions | exact | within one feature | share of real substitutions named | named where experts gave 2 / marked deleted |
|---|---|---|---|---|---|---|
| round 1 | 1980 | 306 | 0.330 | 0.500 | 53.9 % | 450 / 297 |
| careful | 252 | 36 | 0.611 | 0.778 | 6.3 % | 71 / 23 |
| balanced | 455 | 44 | 0.636 | 0.773 | 7.7 % | 172 / 26 |
| eager | 648 | 55 | 0.655 | 0.800 | 9.7 % | 267 / 28 |

By class at "balanced" (test, within one feature): approximant 0.800 (n 5), fricative 0.947 (n 19), stop 0.583 (n 12), vowel 0.625 (n 8). Train (OOF flags): affricate 0.667 (n 3), approximant 0.818 (n 22), fricative 0.840 (n 50), nasal 0.750 (n 12), stop 0.865 (n 37), vowel 0.755 (n 49).

When a sound is named on a real substitution it is right within one feature 77 % of the time (round 1: 50 %), but only 8 % of the experts' substitutions get a name (round 1: 54 %, mostly wrong). Many named sounds still fall on phones the experts scored 2: the experts and the recogniser disagree about mild accents.

### 3. Missing sounds

Logistic on flagged train sounds, label `<DEL>` (393 of 4620): OOF AUC 0.864, best precision on ≥ 20 items 0.690 < 0.7, so `kind: "missing"` stays **off**; a skipped sound is reported as "this sound was off".

### 4. Mandarin (OMPAL dev → test)

Per character, learners only (12518 dev characters; teachers marked 1454 tone, 202 consonant, 205 vowel errors). Tone: threshold on the syllable tone probability; consonant / vowel: the English coaching score of the syllable's initial / other sounds. Type precision target 0.8 / 0.7 / 0.6.

Tone curve on dev (flag when the syllable tone probability ≤ threshold):

| threshold | flagged | precision | recall |
|---|---|---|---|
| 0.7272 | 6259 | 0.216 | 0.932 |
| 0.2367 | 3756 | 0.326 | 0.842 |
| 0.0872 | 2504 | 0.408 | 0.703 |
| 0.0312 | 1252 | 0.503 | 0.433 |
| 0.0177 | 626 | 0.553 | 0.238 |
| 0.0131 | 376 | 0.596 | 0.154 |
| 0.0110 | 251 | 0.594 | 0.102 |
| 0.0083 | 126 | 0.643 | 0.056 |
| 0.0065 | 63 | 0.714 | 0.031 |

| setting | type | dev threshold | dev precision / recall | test flagged | **test precision** | **test recall** |
|---|---|---|---|---|---|---|
| careful | tone | 0.0051 | 0.800 / 0.017 | 10 | 0.600 | 0.005 |
| careful | consonant | off | – | 0 | – | 0.000 |
| careful | vowel | off | – | 0 | – | 0.000 |
| balanced | tone | 0.0074 | 0.703 / 0.044 | 51 | 0.549 | 0.025 |
| balanced | consonant | 0.8492 | 0.720 / 0.089 | 30 | 0.267 | 0.056 |
| balanced | vowel | off | – | 0 | – | 0.000 |
| eager | tone | 0.0131 | 0.601 / 0.153 | 204 | 0.549 | 0.101 |
| eager | consonant | 0.8271 | 0.613 / 0.094 | 40 | 0.325 | 0.091 |
| eager | vowel | 0.8359 | 0.607 / 0.166 | 37 | 0.459 | 0.120 |

The expected gain on tones did not materialise: the tone probability ranks tone errors well (dev AUC 0.86) but OMPAL teachers mark tone errors on 12 % of characters and the top of the ranking is noisy, so precision 0.7 is reached only for the ~3 % most certain tone errors; adding the tone posteriors, the word channel or the tone identity did not help (dev CV, 0.86 → 0.88 AUC, no gain at precision 0.7). Consonant / vowel thresholds were set on 25–56 dev items and do not hold on test (0.27–0.46 precision): **Mandarin sound coaching is not reliable**.

### 5. Other languages (no learner labels; unverified on learners)

The English thresholds are used as they are (the calibrated probabilities are comparable across languages by construction); German keeps an extra heard-sound bar. FLEURS validation native clips (2,476; every flag is a false alarm) and planted substitutions (round-1 data):

| lang | flags / native clip (careful, balanced, eager) | wrong named sounds / native clip (careful, balanced, eager) | planted substitutions flagged (balanced) | named | named within one feature |
|---|---|---|---|---|---|
| en | 1.42 / 2.90 / 5.66 | 0.15 / 0.26 / 0.36 | 83.2 % | 35.1 % | 0.966 |
| zh | 0.00 / 0.02 / 0.11 | 0.00 / 0.00 / 0.02 | 0.0 % | 0.0 % | – |
| ja | 0.87 / 2.48 / 5.30 | 0.12 / 0.28 / 0.49 | 69.4 % | 12.8 % | 0.977 |
| es | 1.94 / 3.97 / 6.93 | 0.22 / 0.29 / 0.32 | 86.8 % | 41.5 % | 0.988 |
| fr | 1.78 / 4.01 / 8.33 | 0.11 / 0.24 / 0.32 | 90.4 % | 37.2 % | 0.987 |
| de | 0.76 / 1.67 / 4.02 | 0.04 / 0.12 / 0.34 | 41.2 % | 11.2 % | 0.929 |
| it | 0.38 / 2.37 / 6.26 | 0.05 / 0.29 / 0.45 | 66.1 % | 33.5 % | 0.985 |
| pt | 0.43 / 1.16 / 3.20 | 0.05 / 0.23 / 0.53 | 43.7 % | 18.2 % | 0.962 |
| ru | 1.26 / 4.43 / 11.76 | 0.12 / 0.22 / 0.39 | 77.9 % | 22.6 % | 0.951 |
| ko | 0.23 / 0.92 / 2.84 | 0.04 / 0.21 / 0.45 | 45.7 % | 22.3 % | 0.911 |
| pooled | | | 60.6 % | 23.5 % | 0.968 |

"Balanced" stays ≤ 0.3 wrong named sounds per native clip in every language (the cap; FLEURS clips are ~10 s). Flags without a name are far more frequent on native speech (1–4.4 per clip at "balanced", up to 11.8 at "eager" in Russian): an app should say "listen and try again" for those, not "you said X". Mandarin sounds are almost never flagged (thresholds from OMPAL). Planted substitutions are now flagged far more often than in round 1 (round 1: 23 %), with the named sound still right within one feature 97 % of the time.

### 6. Real learners, report only (L2-ARCTIC test, EpaDB test)

| corpus | system | annotated substitutions flagged | named (coverage) | precision of all names: exact / within 1 | on substitutions: exact / within 1 | names on phones marked deleted / correct |
|---|---|---|---|---|---|---|
| l2arctic | round 1 | 2.8 % | 2.8 % | 0.183 / 0.270 | 0.565 / 0.659 | 116 / 62 |
| l2arctic | careful | 5.9 % | 1.1 % | 0.384 / 0.466 | 0.800 / 0.857 | 27 / 11 |
| l2arctic | balanced | 14.1 % | 2.0 % | 0.370 / 0.533 | 0.781 / 0.859 | 42 / 29 |
| l2arctic | eager | 23.3 % | 2.7 % | 0.341 / 0.535 | 0.750 / 0.845 | 52 / 49 |
| epadb | round 1 | 2.9 % | 2.9 % | 0.242 / 0.389 | 0.522 / 0.663 | 43 / 63 |
| epadb | careful | 7.0 % | 1.0 % | 0.471 / 0.667 | 0.649 / 0.811 | 5 / 9 |
| epadb | balanced | 14.1 % | 1.5 % | 0.526 / 0.705 | 0.719 / 0.842 | 6 / 15 |
| epadb | eager | 23.1 % | 1.8 % | 0.466 / 0.660 | 0.716 / 0.836 | 7 / 29 |

Round 2 flags 5× more of the annotated substitutions and the sounds it names on them are right 84–86 % of the time within one feature (round 1: 66 %), but it names fewer of them (1.5–2 %) and many names still land on phones annotated as deleted.

### 7. Word stress gate

The stress-model switch landed (`STRESS_MODEL = "stress_v3"`), so the heard-syllable bar was re-fitted on FLEURS validation planted stress moves with the new model (round-1 rule):

| bar | planted flagged | diagnosed | precision | coverage of planted | native items with a diagnosis |
|---|---|---|---|---|---|
| 0.3 | 1235 / 2995 | 1233 | 0.942 | 41.2 % | 169 of 18717 |
| 0.4 | 1235 / 2995 | 1231 | 0.943 | 41.1 % | 165 of 18717 |
| 0.5 | 1235 / 2995 | 1214 | 0.950 | 40.5 % | 159 of 18717 |
| 0.6 | 1235 / 2995 | 1170 | 0.963 | 39.1 % | 152 of 18717 |
| 0.7 | 1235 / 2995 | 1118 | 0.974 | 37.3 % | 141 of 18717 |
| 0.8 | 1235 / 2995 | 1040 | 0.982 | 34.7 % | 128 of 18717 |
| 0.9 | 1235 / 2995 | 901 | 0.990 | 30.1 % | 112 of 18717 |

By language at the chosen bar: en 0.979 / 29.1 %; es 0.953 / 50.6 %; de 0.920 / 45.2 %; it 0.905 / 44.4 %; pt 0.950 / 51.6 %; ru 0.962 / 26.0 %.

### Examples (speechocean762 test, "balanced")

- “FIRST PLACE IN THE DIVISION”. Ours: FIRST /t/ → off (no name) (error_prob 0.588); DIVISION /v/ → /f/ (error_prob 0.561). Experts: THE ['DH', 'AH0'] scores [1.2, 2.0]; DIVISION ['D', 'IH0', 'V', 'IH1', 'ZH', 'N'] scores [2.0, 2.0, 0.4, 1.6, 0.8, 2.0], pronounced [('V', 'F')].
- “DO YOU REALLY AND TRULY SAY SO”. Ours: SAY /s/ → off (no name) (error_prob 0.62); SAY /eɪ/ → off (no name) (error_prob 0.946); SO /s/ → /n/ (error_prob 0.757). Experts: DO ['D', 'UH0'] scores [2.0, 1.8]; YOU ['Y', 'UW0'] scores [2.0, 1.8]; REALLY ['R', 'IH', 'AH1', 'L', 'IY0'] scores [2.0, 1.6, 1.6, 2.0, 2.0]; AND ['AE0', 'N', 'D'] scores [2.0, 2.0, 1.6]; TRULY ['T', 'R', 'UW1', 'L', 'IY0'] scores [2.0, 2.0, 2.0, 1.6, 2.0]; SO ['S', 'OW0'] scores [0.4, 2.0], pronounced [('S', 'N')].

---


Design and schema: [docs/COACHING.md](../docs/COACHING.md). Code: `pronunciada/coach.py`, `pronunciada/tips.py`, `pronunciada/articulatory.py`, `pronunciada/quality.py`; benchmarks `bench/coach/`. Everything was fitted or chosen on FLEURS **validation** (planted errors and synthetic degradations) and Common Voice cv-real **dev**; L2-ARCTIC test, EpaDB test and OMPAL test are reported only (nothing was tuned on them). No cv-real test clip was read.

## Round 1 (planted errors; the sound operating point is superseded)

### 1. Reliability gate for "what it sounded like" (FLEURS validation, planted substitutions)

Protocol (`bench/coach/gather.py`): 250 validation clips per language (Korean: 226), the native audio scored against references in which `bench.corrupt` replaced a sound X by Y (`phone_sub_near` and `phone_sub_far`, 2 per reference), so the true "heard" sound is X. A planted sound is *flagged* when its probability is < 0.5 (default strictness); a flagged sound gets a *diagnosis* (`heard_ipa`) when it passes the gate. The heard candidate is the language's sound of the same type (vowel / consonant) with the highest geometric-mean span posterior of its tokens (adapted model output, the sound's aligned span ± 1 frame), never the expected sound or one of its accepted alternatives.

Rule, fixed before the numbers were seen: the grid setting (log-ratio heard/expected ∈ {-1 … 3}, heard posterior ∈ {0 … 0.3}, frames ∈ {0, 1, 2}) with pooled "within one articulatory feature" precision ≥ 0.85 and the highest pooled coverage; then, per language whose own precision is below 0.85, the loosest heard-posterior bar reaching 0.85 with ≥ 20 diagnoses. "Within one feature" = `articulatory.diff(truth, heard)` has at most one entry (a vowel/consonant change never counts).

Chosen: `min_log_ratio` -1.0, `min_heard_post` 0.0, `min_frames` 0, per-language overrides {"de": {"min_heard_post": 0.25}}. The pooled target is met without any gating: once a planted sound is flagged, the best competing sound is nearly always the true one. The gate binds only in German.

| lang | planted | flagged | diagnosed | coverage (of planted) | coverage (of flagged) | exact | within 1 feature | near-miss: cov / within 1 | far: cov / within 1 | native flagged sounds with a (wrong) diagnosis |
|---|---|---|---|---|---|---|---|---|---|---|
| en | 1000 | 192 | 192 | 19.2 % | 100.0 % | 0.823 | 0.875 | 14.6 % / 0.863 | 23.8 % / 0.882 | 147 of 161 |
| zh | 1000 | 177 | 177 | 17.7 % | 100.0 % | 0.853 | 0.921 | 11.6 % / 0.897 | 23.8 % / 0.933 | 32 of 32 |
| ja | 1000 | 192 | 192 | 19.2 % | 100.0 % | 0.807 | 0.958 | 15.8 % / 0.987 | 22.6 % / 0.938 | 55 of 55 |
| es | 1000 | 491 | 491 | 49.1 % | 100.0 % | 0.945 | 0.980 | 35.6 % / 0.978 | 62.6 % / 0.981 | 68 of 76 |
| fr | 1000 | 522 | 522 | 52.2 % | 100.0 % | 0.912 | 0.954 | 46.4 % / 0.922 | 58.0 % / 0.979 | 187 of 207 |
| de | 1000 | 70 | 26 | 2.6 % | 37.1 % | 0.846 | 0.885 | 1.6 % / 0.750 | 3.6 % / 0.944 | 11 of 79 |
| it | 1000 | 164 | 164 | 16.4 % | 100.0 % | 0.951 | 1.000 | 11.2 % / 1.000 | 21.6 % / 1.000 | 6 of 6 |
| pt | 1000 | 70 | 70 | 7.0 % | 100.0 % | 0.929 | 0.986 | 10.2 % / 0.980 | 3.8 % / 1.000 | 43 of 48 |
| ru | 1000 | 349 | 349 | 34.9 % | 100.0 % | 0.785 | 0.943 | 23.6 % / 0.898 | 46.2 % / 0.965 | 80 of 91 |
| ko | 904 | 112 | 112 | 12.4 % | 100.0 % | 0.929 | 0.955 | 11.7 % / 0.981 | 13.1 % / 0.932 | 9 of 10 |
| pooled | 9904 | 2339 | 2295 | 23.2 % | 98.1 % | 0.882 | 0.953 | 18.3 % / 0.939 | 28.0 % / 0.961 | 638 of 765 |

Coverage is limited by flagging, not by the gate: only 7–52 % of the planted substitutions are flagged at all (German and Portuguese 7 %). The last column counts sounds of the *unaltered* native clips that were flagged (false alarms) and still received a heard sound, which is then necessarily wrong; per clip that is about 0.26 wrong diagnoses (2,476 clips).

Precision / coverage as the log-ratio bar rises (pooled and per language):

| lang | lr ≥ -1 | lr ≥ 0 | lr ≥ 1 | lr ≥ 2 | lr ≥ 3 |
|---|---|---|---|---|---|
| en | 19.2 % / 0.875 | 19.2 % / 0.875 | 19.0 % / 0.884 | 18.8 % / 0.888 | 18.2 % / 0.896 |
| zh | 17.7 % / 0.921 | 17.7 % / 0.921 | 17.7 % / 0.921 | 17.7 % / 0.921 | 17.7 % / 0.921 |
| ja | 19.2 % / 0.958 | 19.2 % / 0.958 | 19.2 % / 0.958 | 19.2 % / 0.958 | 18.9 % / 0.963 |
| es | 49.1 % / 0.980 | 49.1 % / 0.980 | 49.0 % / 0.980 | 49.0 % / 0.980 | 48.2 % / 0.979 |
| fr | 52.2 % / 0.954 | 52.2 % / 0.954 | 52.2 % / 0.954 | 52.0 % / 0.958 | 51.6 % / 0.961 |
| de | 2.6 % / 0.885 | 2.6 % / 0.885 | 2.6 % / 0.885 | 2.6 % / 0.885 | 2.6 % / 0.885 |
| it | 16.4 % / 1.000 | 16.4 % / 1.000 | 16.4 % / 1.000 | 16.4 % / 1.000 | 16.3 % / 1.000 |
| pt | 7.0 % / 0.986 | 7.0 % / 0.986 | 6.8 % / 0.985 | 6.8 % / 0.985 | 6.8 % / 0.985 |
| ru | 34.9 % / 0.943 | 34.9 % / 0.943 | 34.9 % / 0.943 | 34.8 % / 0.943 | 34.7 % / 0.942 |
| ko | 12.4 % / 0.955 | 12.4 % / 0.955 | 12.4 % / 0.955 | 12.4 % / 0.955 | 12.4 % / 0.955 |
| pooled | 23.2 % / 0.953 | 23.2 % / 0.953 | 23.1 % / 0.953 | 23.1 % / 0.954 | 22.8 % / 0.956 |

By sound class of the true sound (pooled, diagnosed items):

| class | n | within 1 feature |
|---|---|---|
| affricate | 22 | 0.636 |
| approximant | 174 | 0.931 |
| flap | 2 | 1.000 |
| fricative | 258 | 0.965 |
| nasal | 239 | 0.967 |
| stop | 318 | 0.978 |
| trill | 68 | 0.971 |
| vowel | 1214 | 0.948 |

Confidence (`item.confidence`): within-one-feature precision by log-ratio quartile of the diagnosed rows, made monotone: edges [6.755, 7.917, 8.998], values [0.865, 0.974, 0.976, 0.976].

**Missing sounds.** Planted extra reference consonants (`phone_ins`: the audio lacks the sound) cannot be told apart from substitutions and false alarms: the best rule over frames ≤ 1–3, nothing decoded in the span, `llr_del` ≤ −4…−15 and blank share reached 0.67 precision on 33 items. Below the 0.85 bar, so `kind: "missing"` is **off** by default (`missing_min_llr_del` = −1e9); a skipped sound is reported as a flagged substitution, usually without `heard_ipa`. Missing *words* use the shipped omission channel (`present_prob` < 0.5). `kind: "extra"` is in the schema but not produced yet.

**Tones (Mandarin).** Items need the syllable probability and the word's calibrated channel (`WordResult.tone_prob`) both < 0.5; heard = the most probable tone / stressed syllable. Bar on its posterior (planted: precision = heard equals the original):

| bar | planted flagged | diagnosed | precision | coverage of planted | native items with a diagnosis |
|---|---|---|---|---|---|
| 0.3 | 73 / 500 | 73 | 0.959 | 14.6 % | 34 of 8407 (0.4 %) |
| 0.4 | 73 / 500 | 73 | 0.959 | 14.6 % | 34 of 8407 (0.4 %) |
| 0.5 | 73 / 500 | 73 | 0.959 | 14.6 % | 33 of 8407 (0.4 %) |
| 0.6 | 73 / 500 | 69 | 0.971 | 13.8 % | 31 of 8407 (0.4 %) |
| 0.7 | 73 / 500 | 68 | 0.971 | 13.6 % | 28 of 8407 (0.3 %) |
| 0.8 | 73 / 500 | 64 | 0.969 | 12.8 % | 26 of 8407 (0.3 %) |
| 0.9 | 73 / 500 | 57 | 0.965 | 11.4 % | 22 of 8407 (0.3 %) |

By language at the chosen bar: zh 0.959 precision, 14.6 % coverage.

**Word stress (en de es it pt ru).** Items need the syllable probability and the word's calibrated channel (`WordResult.stress_prob`) both < 0.5; heard = the most probable tone / stressed syllable. Bar on its posterior (planted: precision = heard equals the original):

| bar | planted flagged | diagnosed | precision | coverage of planted | native items with a diagnosis |
|---|---|---|---|---|---|
| 0.3 | 993 / 2995 | 991 | 0.920 | 33.1 % | 180 of 18717 (1.0 %) |
| 0.4 | 993 / 2995 | 986 | 0.923 | 32.9 % | 177 of 18717 (0.9 %) |
| 0.5 | 993 / 2995 | 972 | 0.932 | 32.5 % | 168 of 18717 (0.9 %) |
| 0.6 | 993 / 2995 | 938 | 0.948 | 31.3 % | 160 of 18717 (0.9 %) |
| 0.7 | 993 / 2995 | 902 | 0.959 | 30.1 % | 154 of 18717 (0.8 %) |
| 0.8 | 993 / 2995 | 872 | 0.967 | 29.1 % | 144 of 18717 (0.8 %) |
| 0.9 | 993 / 2995 | 778 | 0.979 | 26.0 % | 121 of 18717 (0.6 %) |

By language at the chosen bar: en 0.901 precision, 24.4 % coverage; es 0.948 precision, 42.4 % coverage; de 0.895 precision, 34.2 % coverage; it 0.891 precision, 46.0 % coverage; pt 0.949 precision, 35.6 % coverage; ru 0.949 precision, 15.8 % coverage.

### 2. Real learners, round 1 (report only)

`bench/coach/eval_learners.py`: the shipped gate on L2-ARCTIC test (900 utterances, 24 speakers of 6 L1s) and EpaDB test (1,263, Spanish L1), phones aligned to the annotators' as in `bench/eval_english.py`. Precision is over every diagnosis we give on a matched phone: right when it equals what the annotator heard (canonical sound when marked correct, nothing when marked deleted). English vowel labels folded ʌ→ə, ə˞→ɜ˞.

| corpus | annotated substitutions | flagged | with heard_ipa (coverage) | our diagnoses on matched phones | precision exact | within 1 feature | on substitutions: exact / within 1 | diagnoses on phones marked correct / deleted |
|---|---|---|---|---|---|---|---|---|
| l2arctic | 3168 | 2.8 % | 2.8 % | 263 | 0.183 | 0.270 | 0.565 / 0.659 | 62 / 116 |
| epadb | 3848 | 2.9 % | 2.9 % | 198 | 0.242 | 0.389 | 0.522 / 0.663 | 63 / 43 |
| l2arctic, strictness 1.0 | 3168 | 8.1 % | 7.0 % | 591 | 0.218 | 0.357 | 0.606 / 0.732 | 187 / 191 |
| epadb, strictness 1.0 | 3848 | 9.3 % | 7.8 % | 472 | 0.356 | 0.496 | 0.656 / 0.750 | 157 / 59 |

Coverage is limited by flagging: at the default strictness only ~3 % of the annotated substitutions get a sound probability below 0.5 (the scores are calibrated on speechocean762 raters, who accept most accented sounds). Overall precision is low mainly because many of our diagnoses fall on sounds the annotators marked *deleted* (we cannot say "missing" reliably) or *correct*.

Most frequent diagnosed substitutions (expected, annotator, ours, count):

- l2arctic: /ʃ/ → s vs s (6), /oʊ/ → ɔ vs ɑ (2), /p/ → t vs t (2), /oʊ/ → aʊ vs ʊ (2), /s/ → t vs t (2), /d/ → n vs n (2), /p/ → f vs f (2), /ʊ/ → ɔ vs ɑ (1), /m/ → b vs b (1), /ɑ/ → ɛ vs ɛ (1)
- epadb: /eɪ/ → aɪ vs aɪ (7), /eɪ/ → i vs i (4), /ə/ → i vs i (4), /ə/ → ɛ vs ɛ (3), /z/ → s vs s (3), /dʒ/ → ʃ vs ʃ (3), /aɪ/ → ɔ vs ɔ (3), /ɛ/ → i vs i (3), /ə/ → aɪ vs i (3), /eɪ/ → ɛ vs ə (2)

OMPAL test, learners only (7257 characters, 1220 marked wrong by the teachers): does the *type* of our item match the teachers' mark? Tone item → tone; syllable-initial consonant → consonant; any other sound → vowel (final).

| type | our items (characters) | teachers' marks | P(teacher marks this type \| our item) | P(teacher marks any error \| our item) | recall |
|---|---|---|---|---|---|
| tone | 210 | 1113 | 0.476 | 0.490 | 0.090 |
| consonant | 81 | 143 | 0.407 | 0.605 | 0.231 |
| vowel | 102 | 142 | 0.324 | 0.471 | 0.232 |
| tone, strictness 1.0 | 521 | 1113 | 0.451 | 0.463 | 0.211 |
| consonant, strictness 1.0 | 143 | 143 | 0.287 | 0.441 | 0.287 |
| vowel, strictness 1.0 | 189 | 142 | 0.243 | 0.392 | 0.324 |

Where both we and the teachers flag a character (174), our type matches one of theirs in 86.8 % (strictness 1.0: 331, 89.4 %).

Strictness 1.0 rows: the same results re-graded with `Scorer.rescore(result, 1.0)` (the app's strictness knob; reported, not chosen on these data).

## 3. Audio-quality gate

`bench/coach/audio_gate.py`: `pronunciada.quality.measure` on every FLEURS validation clip (3,474, normal audio), synthetic degradations of every 4th clip, and the cv-real dev half (test A: 1,992 accepted, 1,992 rejected by Common Voice reviewers). Each threshold is the 0.4 % tail of FLEURS validation (clipping: the 99.6 % point, at least 0.2 %), so their union stays under the 2 % target.

| measurement | issue code | threshold |
|---|---|---|
| clipped samples share | `clipped` | > 0.002 |
| SNR estimate (p95 − p10 frame level) | `too_noisy` | < 12.1 dB |
| speech level | `too_quiet` | < -69.1 dBFS |
| speech duration / prompt need | `too_short` | < 0.716 |
| speech share of the recording, or speech < 0.3 s | `mostly_silence` | < 0.312 |

Seconds of speech per prompt syllable (median, FLEURS validation): de 0.183, en 0.192, es 0.154, fr 0.171, it 0.173, ja 0.122, ko 0.167, pt 0.167, ru 0.158, zh 0.185.

| audio | n | flagged (any code) | expected code raised | codes |
|---|---|---|---|---|
| cv/accepted | 1992 | 5.9 % | – | clipped 0.2 %, mostly_silence 3.5 %, too_noisy 0.3 %, too_short 2.3 % |
| cv/rejected | 1992 | 13.4 % | – | clipped 0.5 %, mostly_silence 5.8 %, too_noisy 0.9 %, too_quiet 0.1 %, too_short 8.5 % |
| fleurs/att35 | 874 | 47.4 % | 41.8 % | mostly_silence 4.5 %, too_noisy 0.6 %, too_quiet 41.8 %, too_short 4.6 % |
| fleurs/att50 | 874 | 99.5 % | 86.6 % | mostly_silence 13.6 %, too_noisy 0.6 %, too_quiet 86.6 %, too_short 13.7 % |
| fleurs/clean | 3474 | 1.6 % | – | clipped 0.1 %, mostly_silence 0.4 %, too_noisy 0.4 %, too_quiet 0.4 %, too_short 0.4 % |
| fleurs/clip10 | 874 | 99.9 % | 99.9 % | clipped 99.9 %, mostly_silence 0.5 %, too_noisy 3.9 %, too_short 0.1 % |
| fleurs/clip3 | 874 | 97.3 % | 97.3 % | clipped 97.3 %, mostly_silence 0.7 %, too_noisy 0.9 %, too_short 0.5 % |
| fleurs/noise0 | 874 | 100.0 % | 100.0 % | clipped 0.1 %, mostly_silence 20.1 %, too_noisy 100.0 %, too_short 77.5 % |
| fleurs/noise10 | 874 | 33.3 % | 3.2 % | clipped 0.1 %, mostly_silence 5.3 %, too_noisy 3.2 %, too_quiet 0.1 %, too_short 28.5 % |
| fleurs/noise20 | 874 | 5.8 % | – | clipped 0.1 %, mostly_silence 1.1 %, too_noisy 0.6 %, too_quiet 0.3 %, too_short 3.8 % |
| fleurs/noise5 | 874 | 100.0 % | 100.0 % | clipped 0.1 %, mostly_silence 11.7 %, too_noisy 100.0 %, too_short 53.2 % |
| fleurs/pad4 | 874 | 100.0 % | 100.0 % | clipped 0.1 %, mostly_silence 100.0 %, too_noisy 4.7 %, too_short 0.2 % |
| fleurs/trunc40 | 874 | 99.3 % | 99.2 % | clipped 0.1 %, mostly_silence 4.6 %, too_noisy 0.8 %, too_quiet 0.3 %, too_short 99.2 % |

Noise lowers the speech-detection threshold's contrast, so noisy clips also raise `too_short` / `mostly_silence`; white noise at 10 dB SNR is mostly flagged through `too_short` (28.5 %), `too_noisy` itself needs ≲ 5–8 dB. `too_quiet` is very lenient (−69 dBFS) because FLEURS validation itself contains very quiet recordings (English, Japanese, French), which the 2 % budget has to accept. Clipping, truncation and mostly-silent recordings are caught almost always. On Common Voice dev, rejected clips are flagged 2.3× as often as accepted ones (13.4 % vs 5.9 %); most rejections are about the reading, not the audio, so the gate cannot catch them.

FLEURS validation flag rate by language: de 0.0 %, en 3.6 %, es 0.0 %, fr 0.3 %, it 0.3 %, ja 9.4 %, ko 1.8 %, pt 0.3 %, ru 0.0 %, zh 2.7 %.

## 4. Scores unchanged, latency

`bench/coach/identity_check.py`: 50 FLEURS validation clips (5 per language) scored by a copy of the package with the pre-coaching `scorer.py` / `types.py` (commit 773347f) and by the current tree with `result.coaching()` and `tips()` called: every sound, word, syllable and sentence probability and score identical as repr() strings: **yes (0 of 50 clips differ, PYTHONHASHSEED=0 in both runs)**. coaching() + tips(): median 0.25 ms, 90th percentile 0.35 ms per 10 s of audio (native clips flag few sounds); worst case measured, a 10 s clip scored against a different sentence (82 flagged sounds): 1.48 ms. The audio check (always on, `Scorer(audio_check=False)` disables it) median 0.50 ms per 10 s. Nothing is computed for coaching unless `coaching()` is called. Without a fixed PYTHONHASHSEED, two runs of the UNCHANGED package already differ in the last digit of WordResult.present_prob on 1-3 Russian clips (hash-ordered sets in the Russian path); with PYTHONHASHSEED=0 old and new are identical.
