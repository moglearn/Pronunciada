"""results/coaching.md: Azure yardstick and round 3 on top of the round-2 / round-1 sections.

    python -m bench.coach.report && python -m bench.coach.report_v2 && python -m bench.coach.report_r3
"""
from __future__ import annotations

import json
from pathlib import Path

T = Path("results/tables")
LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def ci(v):
    return f"[{v[0]:.3f}, {v[1]:.3f}]"


def main():
    A = json.loads((T / "coaching_azure_yardstick.json").read_text())
    R = json.loads((T / "coaching_r3.json").read_text())
    md = Path("results/coaching.md").read_text(encoding="utf-8")
    rest = md[md.index("## Round 2"):]
    o = ["# Coaching feedback and audio-quality gate", "",
         "Design, schema and app guidance: [docs/COACHING.md](../docs/COACHING.md).", "",
         "## Yardstick: Azure Pronunciation Assessment, speechocean762 test", "",
         f"Which sounds to flag, on the same {A['n_phones']:,} expert-scored phones (error = expert score < 2, "
         f"{100 * A['error_rate']:.1f} % of phones). Ours = the coaching score; Azure = 100 − its phoneme "
         "accuracy from the cached responses (no new requests), aligned to the corpus phones as in "
         "`bench/eval_azure_so762.py`. Azure is thresholded to reach our recall. 95 % CIs: 1,000 "
         "speaker bootstraps (`bench/coach/azure_yardstick.py`). This is a third, report-only look at "
         "speechocean762 test; nothing was changed after it.", "",
         "| operating point | recall | precision ours [CI] | precision Azure [CI] | ours − Azure [CI] |",
         "|---|---|---|---|---|"]
    for s, v in A["at_recall"].items():
        o.append(f"| {s} | {v['recall']:.3f} | {v['precision_ours']:.3f} {ci(v['ours_ci'])} | "
                 f"{v['precision_azure']:.3f} {ci(v['azure_ci'])} | {ci(v['diff_ci'])} |")
    p = A["pr_auc"]
    o += [f"| precision–recall AUC | | {p['ours']:.3f} {ci(p['ours_ci'])} | {p['azure']:.3f} {ci(p['azure_ci'])} | {ci(A['diff_ap_ci'])} |",
          "", "What it heard instead, on the experts' 568 real substitutions (pronounced phone known). The "
          "responses were requested with 5 n-best phonemes per phone, so Azure's candidates exist:", "",
          "| | substitutions named | exact | within one feature |", "|---|---|---|---|"]
    H = A["heard"]
    for k, nm in (("ours_balanced", "ours (balanced)"), ("azure_top_names_other", "Azure: top candidate ≠ expected"),
                  ("azure_top_at_our_coverage", "Azure: same, its lowest-scored 44 only (our coverage)"),
                  ("azure_best_other_candidate", "Azure: best candidate other than expected")):
        v = H[k]
        cov = f"{v['n']} ({100 * v['coverage']:.0f} %)" if "coverage" in v else f"{v['n']}"
        o.append(f"| {nm} | {cov} | {f(v['exact'])} | {f(v['within1'])} |")
    o += ["", f"OMPAL (Mandarin): {A['ompal']}", "",
          "## Round 3: word-level reading miscues and delivery flags", "",
          "Scripts `bench/coach/dump_r3.py`, `bench/coach/fit_r3.py`; definitions in docs/COACHING.md §3b. "
          "Thresholds from FLEURS validation natives (250 clips per language, Korean 226), each flag "
          "≤ 5 % of native clips (extra speech ≤ 2 %).", "",
          "| lang | extra speech bar (sounds) | natives flagged | removed word caught | long pause | mid-phrase pause | rushed punctuation | flat pitch | slow | fast | any delivery flag |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for l in LANGS:
        e, r = R["extra_speech"][l], R["delivery_rates"][l]
        o.append(f"| {l} | {e['min_sounds']} | {100 * e['native_rate']:.1f} % | {100 * (e['hit_rate'] or 0):.0f} % ({e['n_removed']}) | "
                 + " | ".join(f"{100 * r[k]:.1f} %" for k in ("long_pause", "pause_mid_phrase", "rushed_punctuation",
                                                               "flat_pitch", "slow", "fast", "any")) + " |")
    o += ["", "Extra speech catches a removed inner word in 5–54 % of clips: well for multi-sound words in "
          "Romance languages and German, poorly in Japanese, Russian, Korean and Mandarin, where natives' "
          "decodes already leave several sounds outside the expected words, so the 2 % bar is high "
          "(7–9 sounds).", "", "Hit rate by bar (pooled curve per language in `results/tables/coaching_r3.json`).", "",
          "Sanity check on speechocean762 train (English thresholds; mean expert 0–10 sentence score, "
          "flagged minus not flagged):", "", "| flag | flagged | fluency | prosody |", "|---|---|---|---|"]
    for k, v in R["so762_train_sanity"].items():
        df = (lambda a, b: "–" if a is None else f"{a - b:+.2f}")
        o.append(f"| {k} | {v['n_flagged']} ({100 * v['share']:.1f} %) | {df(v['fluency_flagged'], v['fluency_rest'])} | "
                 f"{df(v['prosodic_flagged'], v['prosodic_rest'])} |")
    o += ["", "`long_pause`, `pause_mid_phrase` and `fast` go with ~2.2–3 points lower expert fluency and "
          "prosody. `slow` and `flat_pitch` do not (they fire on most of these short, carefully read "
          "learner prompts and the flagged ones score no lower), so they are off by default "
          "(`coaching(delivery=\"all\")`). `rushed_punctuation` cannot be checked there (no internal "
          "punctuation in the prompts).", "",
          "Russian `present_prob` nondeterminism: `pronunciada/text/ru.py` built some alternatives by "
          "iterating a Python `set` of strings, so their order (and the summation order in the lattice) "
          "depended on `PYTHONHASHSEED`; now an ordered tuple. Two hash seeds now give identical Russian "
          "outputs on 30 clips; against the old code the largest difference is 9e-16 (the old code "
          "differed from itself by 4e-16 between seeds). No other language reads that file.", "",
          "Scores unchanged: the scorer now keeps the pitch track it computes for the prosody models "
          "(the same `track(wav)` ProsodyScorer would compute) and the waveform in the result's cache; "
          "50 clips (5 per language, `PYTHONHASHSEED=0`) are identical to the previous commit as repr() "
          "strings. coaching() with miscues and default delivery flags: median 0.5 ms per 10 s; "
          "`delivery=\"all\"` in French / Korean adds a pitch track on first use (~10 ms).", ""]
    Path("results/coaching.md").write_text("\n".join(o) + "\n" + rest, encoding="utf-8")
    print("wrote results/coaching.md")


if __name__ == "__main__":
    main()
