"""results/coaching.md: round 2 (learner-fitted coaching) on top, round 1 below.

    python -m bench.coach.report && python -m bench.coach.report_v2
"""
from __future__ import annotations

import json
from pathlib import Path

T = Path("results/tables")
LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")
SENS = ("careful", "balanced", "eager")


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.1f} %"


def main():
    F = json.loads((T / "coaching_v2_fit.json").read_text())
    H = json.loads((T / "coaching_v2_heldout.json").read_text())
    HL = json.loads((T / "coaching_v2_heldout_leaky.json").read_text())
    L1 = json.loads((T / "coaching_learners.json").read_text())
    ST = json.loads((T / "coaching_stress.json").read_text()) if (T / "coaching_stress.json").exists() else None
    M = json.loads(Path("pronunciada/data/coach/model.json").read_text())
    o = ["# Coaching feedback and audio-quality gate", "",
         "Design, schema and app guidance: [docs/COACHING.md](../docs/COACHING.md). Round 2 (this "
         "section) replaced the sound / Mandarin-tone operating point of round 1 (below) with one fitted "
         "on real learners.", "",
         "## Round 2: a coaching operating point fitted on learner data", "",
         "Data. Fitted on speechocean762 **train** (2,500 utterances of Mandarin-L1 children and adults "
         "reading English; per-phone expert scores 0–2; 864 phones with the pronounced phone, 443 marked "
         "deleted) and OMPAL **dev** (Mandarin learners, per-character tone / consonant / vowel marks). "
         "FLEURS validation (native) sets the per-language caps. Held out: speechocean762 test and "
         "OMPAL test; L2-ARCTIC test and EpaDB test are report-only (non-commercial licences). Scripts: "
         "`bench/coach/dump.py` (public Scorer outputs), `bench/coach/fit_v2.py`, `bench/coach/eval_v2.py`.", "",
         "**Held-out discipline, plainly.** The held-out sets were evaluated twice. The first fit used the "
         "English sound calibrator's logits on speechocean762 train, where that calibrator was itself "
         "fitted (in-sample), so the coaching model over-trusted them: out-of-fold precision 0.70 on "
         f"train but {f(HL['so762_test']['balanced']['precision'])} on test at \"balanced\". The fix "
         "(the calibrator's out-of-fold logits on train, `bench/calib/train_sound.py`) was found from that "
         "first test run; everything else (rules, targets, grid) is unchanged. Both runs are reported "
         "(`results/tables/coaching_v2_heldout_leaky.json`, `coaching_v2_heldout.json`).", "",
         "### 1. Which sounds to flag (speechocean762)", "",
         "Coaching score = logistic regression (monotone in each input) on the calibrated lenient and "
         "strict log-odds, `llr_sub`, `z_llr_sub`, competitor log-ratio, log duration, nucleus, "
         "`llr_del`, peak posterior and blank share; target expert phone score < 2. Thresholds = "
         "lowest score with out-of-fold (5-fold, speaker-grouped) precision ≥ 0.8 / 0.7 / 0.6. "
         f"OOF AUC {f(F['auc_oof'])} (the strict sound probability alone: {f(F['auc_prob_only'])}). "
         "Coefficients (standardised): " + ", ".join(f"{k} {v:+.2f}" for k, v in F["coef"].items()) + ".", "",
         "| setting | threshold | train OOF precision | train OOF recall | **test precision** | **test recall** | test precision / recall, error = score < 1.5 | flags per FLEURS native clip (en) |",
         "|---|---|---|---|---|---|---|---|"]
    so = H["so762_test"]
    v1 = so["v1"]
    o.append(f"| round 1 (prob < 0.5) | – | {f(F['v1_prob_below_0.5']['precision'])} | {f(F['v1_prob_below_0.5']['recall'])} | "
             f"{f(v1['precision'])} | {f(v1['recall'])} | {f(v1['precision_acc<1.5'])} / {f(v1['recall_acc<1.5'])} | – |")
    for s in SENS:
        op = F["operating_points"][s]
        t = so[s]
        o.append(f"| {s} | {op['threshold']:.3f} | {f(op['precision'])} | {f(op['recall'])} | {f(t['precision'])} | "
                 f"{f(t['recall'])} | {f(t['precision_acc<1.5'])} / {f(t['recall_acc<1.5'])} | "
                 f"{F['native']['en'][s]['flags_per_clip']:.2f} |")
    o += ["", "Test precision is 5 points below the train target at every setting (0.65 at \"balanced\"). The "
          "gain over round 1 is coverage: \"balanced\" finds 1.9× as many expert-marked errors (recall "
          f"{f(so['balanced']['recall'])} vs {f(v1['recall'])}).", "",
          "### 2. When to name the heard sound (speechocean762 pronounced phones)", "",
          f"Candidate as in round 1. Gate fitted on the {F['heard']['n_labelled_flagged']} \"balanced\"-flagged train "
          "sounds with a pronounced-phone substitution label: per sound class ON when exact-or-within-one-"
          "feature precision ≥ 0.75 on ≥ 15 items; the setting with the most correct names. Chosen: "
          f"log-ratio ≥ {M['heard']['min_log_ratio']}, heard posterior ≥ {M['heard']['min_heard_post']}; ON "
          f"classes {', '.join(sorted(set(['vowel','stop','fricative','approximant'])))}; OFF "
          f"{', '.join(M['heard']['off_classes'])} (too few or too inaccurate). Per language the posterior bar is "
          f"raised until \"balanced\" makes ≤ 0.3 wrong named sounds per FLEURS native clip: "
          + ", ".join(f"{l} {v['min_heard_post']}" for l, v in M["heard"]["by_lang"].items()) + ".", "",
          "| | named sounds on test | on real substitutions | exact | within one feature | share of real substitutions named | named where experts gave 2 / marked deleted |",
          "|---|---|---|---|---|---|---|"]
    for s in ("v1",) + SENS:
        t = so[s]
        o.append(f"| {'round 1' if s == 'v1' else s} | {t['named']} | {t['named_on_substitutions']} | {f(t['heard_exact'])} | "
                 f"{f(t['heard_within1'])} | {pct(t['heard_coverage_of_substitutions'])} | {t['named_on_correct']} / {t['named_on_deleted']} |")
    o += ["", "By class at \"balanced\" (test, within one feature): " + ", ".join(
        f"{c} {f(v['within1'])} (n {v['n']})" for c, v in so["balanced"]["heard_by_class"].items())
          + ". Train (OOF flags): " + ", ".join(f"{c} {f(v['precision'])} (n {v['n']})"
                                               for c, v in F["heard"]["best"]["per_class"].items()) + ".", "",
          "When a sound is named on a real substitution it is right within one feature 77 % of the time "
          "(round 1: 50 %), but only 8 % of the experts' substitutions get a name (round 1: 54 %, mostly "
          "wrong). Many named sounds still fall on phones the experts scored 2: the experts and the "
          "recogniser disagree about mild accents.", "",
          "### 3. Missing sounds", "",
          f"Logistic on flagged train sounds, label `<DEL>` ({F['missing']['n_deleted']} of {F['missing']['n_flagged']}): "
          f"OOF AUC {f(F['missing']['auc_oof'])}, best precision on ≥ 20 items {f(F['missing']['best_precision'])} < 0.7, "
          "so `kind: \"missing\"` stays **off**; a skipped sound is reported as \"this sound was off\".", "",
          "### 4. Mandarin (OMPAL dev → test)", "",
          f"Per character, learners only ({F['zh']['n_units']} dev characters; teachers marked {F['zh']['curves']['tone']['n_wrong']} tone, "
          f"{F['zh']['curves']['initial']['n_wrong']} consonant, {F['zh']['curves']['final']['n_wrong']} vowel errors). "
          "Tone: threshold on the syllable tone probability; consonant / vowel: the English coaching score "
          "of the syllable's initial / other sounds. Type precision target 0.8 / 0.7 / 0.6.", "",
          "Tone curve on dev (flag when the syllable tone probability ≤ threshold):", "",
          "| threshold | flagged | precision | recall |", "|---|---|---|---|"]
    for c in F["zh"]["curves"]["tone"]["curve"]:
        o.append(f"| {c['threshold']:.4f} | {c['flagged']} | {f(c['precision'])} | {f(c['recall'])} |")
    o += ["", "| setting | type | dev threshold | dev precision / recall | test flagged | **test precision** | **test recall** |",
          "|---|---|---|---|---|---|---|"]
    for s in SENS:
        for key, dk in (("tone", "tone"), ("consonant", "initial"), ("vowel", "final")):
            at = F["zh"]["curves"][dk]["at"].get(s)
            t = H["ompal_test"][s][key]
            o.append(f"| {s} | {key} | {('off' if at is None else f'{abs(at["threshold"]):.4f}')} | "
                     f"{('–' if at is None else f'{f(at['precision'])} / {f(at['recall'])}')} | {t['flagged']} | "
                     f"{f(t['precision'])} | {f(t['recall'])} |")
    o += ["", "The expected gain on tones did not materialise: the tone probability ranks tone errors well "
          f"(dev AUC 0.86) but OMPAL teachers mark tone errors on 12 % of characters and the top of the "
          "ranking is noisy, so precision 0.7 is reached only for the ~3 % most certain tone errors; adding "
          "the tone posteriors, the word channel or the tone identity did not help (dev CV, 0.86 → 0.88 AUC, "
          "no gain at precision 0.7). Consonant / vowel thresholds were set on 25–56 dev items and do not "
          "hold on test (0.27–0.46 precision): **Mandarin sound coaching is not reliable**.", "",
          "### 5. Other languages (no learner labels; unverified on learners)", "",
          "The English thresholds are used as they are (the calibrated probabilities are comparable across "
          "languages by construction); German keeps an extra heard-sound bar. FLEURS validation native "
          "clips (2,476; every flag is a false alarm) and planted substitutions (round-1 data):", "",
          "| lang | flags / native clip (careful, balanced, eager) | wrong named sounds / native clip (careful, balanced, eager) | planted substitutions flagged (balanced) | named | named within one feature |",
          "|---|---|---|---|---|---|"]
    for l in LANGS:
        n = F["native"][l]
        pl = F["planted"].get(l, {}).get("balanced", {})
        o.append(f"| {l} | " + " / ".join(f"{n[s]['flags_per_clip']:.2f}" for s in SENS) + " | "
                 + " / ".join(f"{n[s]['diagnoses_per_clip']:.2f}" for s in SENS)
                 + f" | {pct(pl.get('flagged'))} | {pct(pl.get('named'))} | {f(pl.get('within1'))} |")
    pp = F["planted"]["pooled"]["balanced"]
    o += [f"| pooled | | | {pct(pp['flagged'])} | {pct(pp['named'])} | {f(pp['within1'])} |", "",
          "\"Balanced\" stays ≤ 0.3 wrong named sounds per native clip in every language (the cap; FLEURS "
          "clips are ~10 s). Flags without a name are far more frequent on native speech (1–4.4 per clip "
          "at \"balanced\", up to 11.8 at \"eager\" in Russian): an app should say \"listen and try "
          "again\" for those, not \"you said X\". Mandarin sounds are almost never flagged (thresholds "
          "from OMPAL). Planted substitutions are now flagged far more often than in round 1 (round 1: "
          "23 %), with the named sound still right within one feature 97 % of the time.", "",
          "### 6. Real learners, report only (L2-ARCTIC test, EpaDB test)", "",
          "| corpus | system | annotated substitutions flagged | named (coverage) | precision of all names: exact / within 1 | on substitutions: exact / within 1 | names on phones marked deleted / correct |",
          "|---|---|---|---|---|---|---|"]
    for ds in ("l2arctic", "epadb"):
        v = L1[ds]
        o.append(f"| {ds} | round 1 | {pct(v['flagged_sub'])} | {pct(v['coverage_sub'])} | {f(v['precision_exact'])} / "
                 f"{f(v['precision_within1'])} | {f(v['sub_exact'])} / {f(v['sub_within1'])} | "
                 f"{v['diagnosed_labels']['del']} / {v['diagnosed_labels']['correct']} |")
        for s in SENS:
            v = H["learners"][ds][f"v2_{s}"]
            o.append(f"| {ds} | {s} | {pct(v['flagged_sub'])} | {pct(v['coverage_sub'])} | {f(v['precision_exact'])} / "
                     f"{f(v['precision_within1'])} | {f(v['sub_exact'])} / {f(v['sub_within1'])} | "
                     f"{v['diagnosed_labels']['del']} / {v['diagnosed_labels']['correct']} |")
    o += ["", "Round 2 flags 5× more of the annotated substitutions and the sounds it names on them are "
          "right 84–86 % of the time within one feature (round 1: 66 %), but it names fewer of them (1.5–2 %) "
          "and many names still land on phones annotated as deleted.", ""]
    if ST:
        o += ["### 7. Word stress gate", "",
              f"The stress-model switch landed (`STRESS_MODEL = \"{ST['stress_model']}\"`), so the heard-syllable bar "
              "was re-fitted on FLEURS validation planted stress moves with the new model (round-1 rule):", "",
              "| bar | planted flagged | diagnosed | precision | coverage of planted | native items with a diagnosis |",
              "|---|---|---|---|---|---|"]
        for c in ST["curve"]:
            o.append(f"| {c['bar']} | {c['planted']['flagged']} / {c['planted']['n']} | {c['planted']['diagnosed']} | "
                     f"{f(c['planted']['precision'])} | {pct(c['planted']['coverage'])} | {c['clean']['diagnosed']} of {c['clean']['n']} |")
        o += ["", "By language at the chosen bar: " + "; ".join(
            f"{l} {f(v['planted']['precision'])} / {pct(v['planted']['coverage'])}" for l, v in ST["by_lang"].items()) + ".", ""]
    o += ["### Examples (speechocean762 test, \"balanced\")", ""]
    for e in H["examples"]:
        o.append(f"- “{e['text']}”. Ours: " + "; ".join(
            f"{x['word']} /{x['expected']}/ → {('/' + x['heard'] + '/') if x['heard'] else 'off (no name)'} (error_prob {x['error_prob']})"
            for x in e["ours"]) + ". Experts: " + "; ".join(
            f"{x['word']} {x['phones']} scores {x['scores']}" + (f", pronounced {[(m['canonical_phone'], m['pronounced_phone']) for m in x['mispronunciations']]}"
                                                               if x["mispronunciations"] else "") for x in e["experts"]) + ".")
    o += ["", "---", ""]
    v1 = Path("results/coaching.md").read_text(encoding="utf-8")
    v1 = v1.split("\n", 1)[1] if v1.startswith("# ") else v1
    v1 = v1[v1.index("## Round 1"):] if "## Round 1" in v1 else v1
    v1 = v1.replace("## 1. Reliability gate", "## Round 1 (planted errors; the sound operating point is superseded)\n\n### 1. Reliability gate", 1) \
        if not v1.startswith("## Round 1") else v1
    Path("results/coaching.md").write_text("\n".join(o) + "\n" + v1, encoding="utf-8")
    print("wrote results/coaching.md")


if __name__ == "__main__":
    main()
