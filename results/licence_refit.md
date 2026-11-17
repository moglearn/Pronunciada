# License-clean refit (2026-11-17)

## What was contaminated, and how

The licensing audit (docs/LICENSING.md) found that the shipped **word model**
`pronunciada/data/calib/word.npz` was trained in round 2 on speechocean762 train + **L2-ARCTIC
train (CC BY-NC 4.0)** + **EpaDB train (license unverified)** + FLEURS validation planted errors
(stored metadata `trained_on: so762+l2arctic+epadb+fleurs`; the round-2 rule had picked
`full/s3`). The word model's log-odds feed `WordResult.sounds_prob` (so every word and sentence
probability) and the sentence heads' inputs. L2-ARCTIC / EpaDB train had also been used to
**select** between candidate universal sound models (`sound_U.npz`) and word models.

## What was done

Mechanically, in dependency order, each step with its original recipe, grid, selection rule and
adoption bar; no new candidates were adopted and nothing was re-designed. Collection caches
that held word-model outputs were moved aside (not deleted) and rebuilt.

| step | recipe | outcome | shipped file |
|---|---|---|---|
| 1. `sound_U` selection | `bench/calib/train_wide.py --clean`: candidates so762 / so762+FLEURS, rule = speechocean762 strict AUC (the only real corpus left), 0.002 margin | so762 0.8705 vs 0.8683: **same winner** | `sound_U.npz` unchanged |
| 2. word model | `bench/calib/train_word2.py` (round 2), clean candidates so762 / so762+FLEURS × s1–s3, same rule (native word acceptance ≥ 0.97, max mean of FLEURS word AUC and speechocean762 strict AUC) | `so762+fleurs/s3` (score 0.8974; historical `full/s3` 0.8935) | `word.npz` **changed** |
| 2b. word 0–10 regression | same script, same choice rule | `old_inputs` again (no word log-odds in its inputs) | `word_reg.npz` unchanged (refit byte-identical) |
| 2c. round-2 sentence heads | same script, SENT_NUM vs SENT2_NUM | SENT_NUM again; inputs include the word log-odds | `sentence_r4.npz` **changed** |
| 3. word calibration maps | `bench/calib/recal_word.py` (round 3) | isotonic maps re-fitted on final sound inputs | `word.npz` (maps) |
| 4. round-5 delivery heads | `bench/calib/train_delivery.py ablate` + `fit` | same groups adopted (emb on enc1 + disfluency), blend kept (it raises every head's CV) | `sentence.npz` **changed** |
| 5. round-5 native guard | `bench/calib/native_sentence_check.py --write-config` (median native fluency / prosodic / total may drop ≤ 0.3) | **fails now in every language**, including English (prosodic −0.32) and Portuguese, which passed before | `config.json` `sentence_model`: **en and pt now use `sentence_r4`** (round-5 heads un-adopted; `sentence.npz` is shipped but selected by no language) |
| 6. word-level tone / stress channels | `bench/calib/fit_prosody.py`, all 7 languages, default and selectable models (`stress`, `tone_zh_v2`), same leniency rule | the rule picks the same native-quantile pair in every case; coefficients identical (pt differs in the 3rd decimal because of the round-16 Portuguese reference change, not the word model) | all `calib/prosody/*.json` unchanged |
| 7. Mandarin sentence recalibration | `bench/calib/fit_sentence_zh.py` (OMPAL dev, +0.03 CV bar, FLEURS native guard), recalibration off while collecting | accuracy and prosodic adopted again, fluency kept | `sentence_zh_recal.json` **changed** (coefficients) |
| 8. Russian bias rule | `bench/calib/bias_grid.py run/select` (round 10: grid 0–1.1, guards vs the round-7 baseline, highest near-swap AUC) | against the round-7 baseline no value is feasible any more (later rounds' contrast caps already cost more native-sound acceptance than the guard allows), so the rule makes no choice and the shipped 0.0 stays. With today's system as the baseline instead, the rule would pick 0.5 (near-swap AUC 0.9133 vs 0.9121); not adopted, because that would change the rule | `config.json` bias unchanged |
| 9. Japanese length word factor | `bench/length/fit_length.py fit` (word-factor grid, native-drop and AUC guards), dev rows recollected | same setting (q 0.02, width 4, floor 0.4) | `length/ja.json` unchanged |
| 10. word contrast factors | `bench/round18/fit.py dev`, zh and ja rows recollected | zh and ja adopted again with the same parameters; en / ru / de / ko stay rejected | `contrast_word/*.json` unchanged |

Coaching models read sound-level inputs only (`coach/model.json` features), the omission channel
reads deletion evidence only, and `word_reg_r6.npz` reads the word regression inputs, so none of
them depends on the word model; they were not refitted and are byte-identical.

**Shipped files that changed:** `pronunciada/data/calib/word.npz`, `sentence_r4.npz`,
`sentence.npz`, `sentence_zh_recal.json`, `config.json` (`sentence_model` gains en and pt).
Everything else in `pronunciada/data/` is byte-identical to the previous commit.

**Reproducibility note.** Re-running the round-2 recipe today does not reproduce the historical
CV scores exactly (e.g. `so762/s2` 0.8677 now vs 0.8681 then): the cached feature tables were
rebuilt in later rounds. The clean model was chosen and compared within one run.

## Before / after (held-out, both frozen runs of 2026-11-17)

"Before" = the first frozen run (previous commit), "after" = the frozen run after the refit; same
code otherwise, Azure unchanged.

| group | measure | before (frozen run 1) | after (clean refit) | Δ |
|---|---|---|---|---|
| speechocean762 test (Pearson) | sounds | 0.685 | 0.685 | +0.000 |
| speechocean762 test (Pearson) | words | 0.609 | 0.609 | +0.000 |
| speechocean762 test (Pearson) | sentence accuracy | 0.761 | 0.748 | -0.013 |
| speechocean762 test (Pearson) | sentence fluency | 0.792 | 0.779 | -0.014 |
| speechocean762 test (Pearson) | sentence prosodic | 0.782 | 0.772 | -0.010 |
| speechocean762 test (Pearson) | sentence total | 0.773 | 0.762 | -0.011 |
| real learner errors (AUC) | l2arctic | 0.712 | 0.712 | +0.000 |
| real learner errors (AUC) | epadb | 0.721 | 0.721 | +0.000 |
| OMPAL test | word AUC, primary: any error | 0.794 | 0.792 | -0.002 |
| OMPAL test | word AUC, (a) tone-only error | 0.782 | 0.778 | -0.003 |
| OMPAL test | word AUC, (b) consonant or vowel error | 0.851 | 0.856 | +0.005 |
| OMPAL test | sentence accuracy (Pearson) | 0.635 | 0.634 | -0.001 |
| OMPAL test | sentence fluency (Pearson) | 0.594 | 0.590 | -0.004 |
| OMPAL test | sentence prosody (Pearson) | 0.532 | 0.530 | -0.002 |
| swap benchmark en | near AUC | 0.981 | 0.981 | -0.000 |
| swap benchmark en | near hit@5 % | 0.900 | 0.893 | -0.007 |
| swap benchmark en | far right word | 0.893 | 0.887 | -0.007 |
| swap benchmark en | native FA | 0.035 | 0.034 | -0.001 |
| swap benchmark zh | near AUC | 0.974 | 0.975 | +0.001 |
| swap benchmark zh | near hit@5 % | 0.893 | 0.900 | +0.007 |
| swap benchmark zh | far right word | 0.893 | 0.893 | +0.000 |
| swap benchmark zh | native FA | 0.038 | 0.040 | +0.002 |
| swap benchmark ja | near AUC | 0.969 | 0.968 | -0.001 |
| swap benchmark ja | near hit@5 % | 0.820 | 0.840 | +0.020 |
| swap benchmark ja | far right word | 0.920 | 0.927 | +0.007 |
| swap benchmark ja | native FA | 0.029 | 0.033 | +0.004 |
| swap benchmark es | near AUC | 0.989 | 0.990 | +0.000 |
| swap benchmark es | near hit@5 % | 0.953 | 0.967 | +0.013 |
| swap benchmark es | far right word | 0.947 | 0.953 | +0.007 |
| swap benchmark es | native FA | 0.024 | 0.025 | +0.001 |
| swap benchmark fr | near AUC | 0.986 | 0.986 | +0.000 |
| swap benchmark fr | near hit@5 % | 0.947 | 0.927 | -0.020 |
| swap benchmark fr | far right word | 0.920 | 0.927 | +0.007 |
| swap benchmark fr | native FA | 0.026 | 0.027 | +0.001 |
| swap benchmark de | near AUC | 0.956 | 0.960 | +0.004 |
| swap benchmark de | near hit@5 % | 0.767 | 0.767 | +0.000 |
| swap benchmark de | far right word | 0.907 | 0.913 | +0.007 |
| swap benchmark de | native FA | 0.043 | 0.046 | +0.004 |
| swap benchmark it | near AUC | 0.976 | 0.979 | +0.003 |
| swap benchmark it | near hit@5 % | 0.847 | 0.887 | +0.040 |
| swap benchmark it | far right word | 0.980 | 0.980 | +0.000 |
| swap benchmark it | native FA | 0.022 | 0.023 | +0.000 |
| swap benchmark pt | near AUC | 0.960 | 0.963 | +0.003 |
| swap benchmark pt | near hit@5 % | 0.833 | 0.853 | +0.020 |
| swap benchmark pt | far right word | 0.940 | 0.940 | +0.000 |
| swap benchmark pt | native FA | 0.025 | 0.028 | +0.004 |
| swap benchmark ru | near AUC | 0.902 | 0.900 | -0.003 |
| swap benchmark ru | near hit@5 % | 0.607 | 0.620 | +0.013 |
| swap benchmark ru | far right word | 0.920 | 0.913 | -0.007 |
| swap benchmark ru | native FA | 0.052 | 0.052 | +0.000 |
| swap benchmark ko | near AUC | 0.941 | 0.944 | +0.003 |
| swap benchmark ko | near hit@5 % | 0.767 | 0.753 | -0.013 |
| swap benchmark ko | far right word | 0.960 | 0.960 | +0.000 |
| swap benchmark ko | native FA | 0.021 | 0.023 | +0.002 |
| cv-real test (AUC) | en/A | 0.744 | 0.748 | +0.004 |
| cv-real test (AUC) | en/B | 0.522 | 0.517 | -0.005 |
| cv-real test (AUC) | zh-CN/A | 0.768 | 0.771 | +0.003 |
| cv-real test (AUC) | zh-TW/A | 0.884 | 0.885 | +0.001 |
| cv-real test (AUC) | ja/A | 0.802 | 0.798 | -0.003 |
| cv-real test (AUC) | es/A | 0.770 | 0.766 | -0.004 |
| cv-real test (AUC) | fr/A | 0.801 | 0.789 | -0.012 |
| cv-real test (AUC) | de/A | 0.827 | 0.823 | -0.004 |
| cv-real test (AUC) | de/B | 0.544 | 0.541 | -0.003 |
| cv-real test (AUC) | it/A | 0.781 | 0.767 | -0.014 |
| cv-real test (AUC) | pt/A | 0.743 | 0.729 | -0.014 |
| cv-real test (AUC) | ru/A | 0.788 | 0.788 | +0.000 |
| cv-real test (AUC) | ko/A | 0.860 | 0.802 | -0.058 |
| native acceptance en | sounds | 0.993 | 0.993 | +0.000 |
| native acceptance en | words | 0.971 | 0.972 | +0.000 |
| native acceptance en | sentences | 0.993 | 0.988 | -0.005 |
| native acceptance zh | sounds | 0.996 | 0.996 | +0.000 |
| native acceptance zh | words | 0.973 | 0.971 | -0.002 |
| native acceptance zh | sentences | 0.993 | 0.995 | +0.002 |
| native acceptance ja | sounds | 0.994 | 0.994 | +0.000 |
| native acceptance ja | words | 0.974 | 0.971 | -0.003 |
| native acceptance ja | sentences | 0.993 | 0.990 | -0.003 |
| native acceptance es | sounds | 0.997 | 0.997 | +0.000 |
| native acceptance es | words | 0.985 | 0.984 | -0.001 |
| native acceptance es | sentences | 0.995 | 0.995 | +0.000 |
| native acceptance fr | sounds | 0.993 | 0.993 | +0.000 |
| native acceptance fr | words | 0.979 | 0.978 | -0.001 |
| native acceptance fr | sentences | 0.993 | 0.990 | -0.003 |
| native acceptance de | sounds | 0.996 | 0.996 | +0.000 |
| native acceptance de | words | 0.968 | 0.965 | -0.003 |
| native acceptance de | sentences | 0.993 | 0.993 | +0.000 |
| native acceptance it | sounds | 0.999 | 0.999 | +0.000 |
| native acceptance it | words | 0.985 | 0.984 | -0.001 |
| native acceptance it | sentences | 0.998 | 0.998 | +0.000 |
| native acceptance pt | sounds | 0.998 | 0.998 | +0.000 |
| native acceptance pt | words | 0.979 | 0.976 | -0.003 |
| native acceptance pt | sentences | 0.990 | 0.990 | +0.000 |
| native acceptance ru | sounds | 0.987 | 0.987 | +0.000 |
| native acceptance ru | words | 0.968 | 0.969 | +0.001 |
| native acceptance ru | sentences | 0.990 | 0.990 | +0.000 |
| native acceptance ko | sounds | 0.993 | 0.993 | +0.000 |
| native acceptance ko | words | 0.978 | 0.977 | -0.002 |
| native acceptance ko | sentences | 0.990 | 0.987 | -0.003 |

Unchanged (identical tables): every sound-level row (speechocean762 phone correlation, L2-ARCTIC /
EpaDB detection, planted sound errors), the coaching yardstick against Azure, the coaching
feature-claim table, and every contrast-audit sound row. Contrast-audit **word** rows moved by up
to 11 points (Italian ʃ-for-s 28 → 40 %, Mandarin plain-for-aspirated 57 → 68 %); Japanese length:
sounds 95 % → 95 %, words 54 % → 54 % caught, native word acceptance on those clips 97.4 → 97.1 %.

**Cost and gain in short.** English sentence correlations fall by 0.010–0.014 because the
round-5 delivery heads no longer pass their native guard (intonation 0.782 → 0.772, now
significantly behind Azure's 0.827; overall 0.773 → 0.762). Word-level detection on planted
errors rises slightly in most languages (word AUC +0.001 to +0.005); native word acceptance drops
by 0.1–0.3 points in most languages. Swap-benchmark and cv-real rows move within their CIs
(largest: Korean cv-real Test A, 11 + 11 clips, 0.860 → 0.802; Italian near-miss hit@5 % 0.847 →
0.887). OMPAL rows move by ≤ 0.005.

