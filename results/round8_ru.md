# Round 8: Russian reduction-aware references (not adopted)

**Outcome.** The change did not pass the dev bar, so nothing shipped changed. The default Russian
references, adapter, sound bias and stress channel are still the round-7 ones. The held-out sets
(FLEURS-ru test swap and ten-language rows, cv-real ru) were **not run**, so the one held-out look
is still unused. All Russian held-out numbers in README.md, languages.md, azure_swap.md and
cv_real.md stand unchanged. The new references are in the code, but off by default
(`get_g2p("ru").measured_targets = True` turns them on).

Everything below is FLEURS-ru **validation** (dev) and FLEURS-ru **train** (adapter, bias). No
Azure requests were made.

## Diagnosis (the round-7 problem)

On the swap test, Russian near-swap AUC is 0.872 for us and 0.909 for Azure. The loss is a tail of
total misses: 36/150 replaced words score > 0.9 with us, 9/150 with Azure. Most of them are
unstressed inflectional endings. The round-7 front-end gives every unstressed vowel broad
hand-written alternatives, e.g. unstressed `ɨ` accepts `(ɪ, a)` and unstressed `a` accepts
`(ɐ, ə, ɑ, ʌ)`. So pairs like годы/года are invisible by construction.

## What was built

1. **Audit** (`python -m bench.calib.ru_round8 audit`, table in
   `results/tables/ru_round8_audit_raw.json`):
   * Covers every vowel, and every t / tʲ / d / dʲ slot, of the 355 validation references.
   * Only words whose stress is known are used: lexicon, accent mark, ё, one vowel, or an
     unstressed clitic.
   * Each slot is offered 13 vowel candidates (or the t/d series), all with equal weight. The
     Viterbi path through the **raw** ZIPA posteriors picks one.
   * Counts are kept per (letter, position S/P1/P2/T/F/C, left context V/H/C) and per (t/d
     origin, next sound).
   * Findings:
     * ZIPA practically never emits ɐ / ə / ʊ for Russian. Reduced а/о come out as `a` (or `ɑ`).
     * Final -ы and final -а are well separated: ы → ɨ 88 %, a 8 %; а → a 87 %, ɨ 4 %.
     * Final -е after a soft consonant is genuinely mixed: ɛ 48 %, i 39 %. After a vowel
       (-ие): ɛ 46 %, a 17 %, u 15 %.
     * Final -ть is heard as plain t 13 % of the time. Final -т is heard as ts 25 % of the time.
2. **New references** (`pronunciada/text/ru.py`, data `pronunciada/data/lex/ru_vowels.json`):
   * Applies only to known-stress words, and only to classes with n ≥ 30.
   * The primary symbol is the most frequent realisation.
   * An alternative is kept only if natives produce it in at least τ of the cases. Its log prior
     is log(share / primary share).
   * Words with predicted stress keep the round-7 broad alternatives. That is 19 % of dev words:
     12.9 % stem-predicted, 6.2 % fully out of vocabulary.
   * With the switch off, the output is bit-identical to round 7 on all 1 205 FLEURS-ru
     validation, test and swap texts (checked).
3. **Adapter retrained** with the existing recipe (`bench.pipeline.train_adapter`, FLEURS-ru
   train, conv chosen). It is not shipped; `results/tables/adapter_ru.json` was restored to round 7.
4. **Bias refitted** by the finalize rule (`ru_round8 bias`): +1.108, against +1.075 shipped.

## Dev table (FLEURS-ru validation, 355 clips; dev swap set 349 near and 349 far items)

The dev swap set is built from validation sentences by the fleurs-swap recipe and was frozen
before any change. "Old" = round-7 adapter; "re" = retrained adapter. Native words is the share
of native words accepted at default strictness.

| | base (round 7) | A τ=.10, old | A τ=.15, old | A+B τ=.10 | A+B τ=.15 | A+B+C τ=.15 (candidate) |
|---|---|---|---|---|---|---|
| near AUC | 0.871 [0.850, 0.892] | 0.876 | 0.877 | 0.881 | 0.883 | **0.883 [0.861, 0.903]** |
| replaced > 0.9 | 24.9 % | 22.9 % | 22.1 % | 20.6 % | 20.3 % | 20.3 % |
| median replaced score | 0.52 | 0.46 | 0.37 | 0.38 | 0.38 | 0.38 |
| near hit@5 % FA | 0.559 | 0.564 | 0.587 | 0.599 | 0.616 | 0.616 [0.564, 0.669] |
| near right word | 0.490 | 0.504 | 0.510 | 0.521 | 0.521 | 0.516 |
| far AUC | 0.991 | 0.990 | 0.990 | 0.991 | 0.991 | 0.990 |
| native words / sounds | 0.9722 / 0.9997 | 0.9710 / 0.9996 | 0.9694 / 0.9996 | 0.9711 / 0.9997 | 0.9707 / 0.9998 | 0.9705 / 0.9998 |
| planted sound AUC (all) | 0.9811 | 0.9806 | 0.9801 | 0.9820 | 0.9816 | 0.9816 |
| sub_near / sub_far / ins / word_swap | .9679/.9868/.9834/.9750 | .9655/.9865/.9830/.9753 | .9626/.9862/.9828/.9752 | .9707/.9883/.9842/.9753 | .9681/.9883/.9841/.9752 | .9681/.9883/.9841/.9752 |
| planted word AUC | 0.9231 | 0.9209 | 0.9194 | 0.9209 | 0.9193 | 0.9195 |
| stress-shift AUC | 0.8710 | 0.8682 | 0.8662 | 0.8705 | 0.8692 | 0.8693 |

Other settings tried:
* τ = 0.05 / 0.20 / 0.25 with the old adapter: near AUC 0.871 / 0.877 / 0.874.
* τ = 0.20 with the retrained adapter: 0.883.

The stress channel was not refitted for the candidate. It is shown with the round-7 calibrator;
its AUC moved by 0.002 or less.

**Gate:**
* near AUC +0.011 (the bar is +0.02): **fail**;
* native words −0.17 points (limit 0.5): pass;
* worst planted drop is word AUC −0.0036 (limit 0.005): pass.

**Not adopted.**

Reading the table:
* **The adapter is part of the blur.** With the same references, the retrained adapter adds
  +0.005 to +0.006 near AUC over the old one. It also undoes the planted sub_near loss that
  stricter references cause with the old adapter (0.9626 → 0.9681 at τ = 0.15).
* **References alone help little** (+0.006 at best). Tightening alternatives below τ = 0.15
  starts to cost planted-error AUC and native acceptance.

## What actually limits Russian

The per-sound evidence on the remaining misses is clear. A typical case is является → являются:
* the slot for the reference ю gets llr_sub ≈ −4.5 to −5.6;
* the model heard ɛ;
* yet the sound probability is still 0.95–0.98.

The universal sound calibrator plus the Russian bias (+1.1 logits, the largest of all languages)
maps that evidence to "fine".

As a dev-only diagnostic (not a proposal I fitted), the bias was set to 0:
* round-7 system: near AUC 0.871 → 0.889;
* A+B: 0.883 → 0.896, with > 0.9 dropping to 12 %;
* cost: native word acceptance −0.3 points, and stress-shift AUC −0.012 (the stress channel was
  not refitted).

The bias comes from a fixed cross-language rule (match the English native median), so changing it
is outside this brief. It is the next thing to try, chosen on dev.

## Pairs that remain indistinguishable

| pair | why | do natives distinguish it? |
|---|---|---|
| которые / которое, самые / самое, разные / разное, первые / первое | unstressed -ые vs -ое: ɨ vs ɑ/a overlap in fast speech | barely; in fluent speech the two are close to homophones |
| открытие / открытия, создание / создания, мнение / мнения, побережье / побережья | -ие / -ия, -ье / -ья: the final vowel after j is ɛ / a / ɪ in natives (audit) | in careful speech yes; in fluent speech often no |
| является / являются, имеет / имеют, считается / считаются | unstressed е vs ю after a vowel. The evidence is there (llr ≈ −5) but the calibration forgives it | **yes**: different grammatical person |
| обеспечит / обеспечить, взял / взять (final т / ть) | final т is affricated 25 % of the time, ть is plain t 13 % | yes in careful speech; the acoustic overlap is real |
| организации / организаций, ситуации / ситуаций | final i vs j | weakly (vowel length / glide) |
| всё / все, ещё / еще, трёх / трех | spelling only (ё written as е), same pronunciation | no |
| тем / чем, часть / честь, каждый / каждой | consonant or stressed-vowel change, still > 0.9 (not inspected sound by sound) | **yes** |

## Files

* `bench/calib/ru_round8.py`: dev swap builder, audit, dev evaluation, Russian-only bias refit.
* `pronunciada/text/ru.py`: `vowel_keys`, `cons_key`, `measured_targets` (default False),
  `VOWEL_MIN = 0.15`.
* `pronunciada/data/lex/ru_vowels.json`: the audit table.
* `tests/test_text_ru_round8.py`.
* Caches: `calib/round8_ru/` (dev swap set, per-run outputs and summaries).

Note that `bench.pipeline.common.g2p_version("ru")` changed (ru.py and the new lexicon file). A
future `train_adapter --lang ru` run would therefore retrain instead of reusing the round-7 cached
selection.
