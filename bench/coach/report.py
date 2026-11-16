"""results/coaching.md from results/tables/coaching_gate.json, coaching_learners.json,
audio_gate.json and coaching_identity.json.

    python -m bench.coach.report
"""
from __future__ import annotations

import json
from pathlib import Path

T = Path("results/tables")
LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.1f} %"


def main():
    G = json.loads((T / "coaching_gate.json").read_text())
    L = json.loads((T / "coaching_learners.json").read_text()) if (T / "coaching_learners.json").exists() else None
    AU = json.loads((T / "audio_gate.json").read_text()) if (T / "audio_gate.json").exists() else None
    ID = json.loads((T / "coaching_identity.json").read_text()) if (T / "coaching_identity.json").exists() else None
    th = G["thresholds"]
    o = ["# Coaching feedback and audio-quality gate", "",
         "Design and schema: [docs/COACHING.md](../docs/COACHING.md). Code: `pronunciada/coach.py`, "
         "`pronunciada/tips.py`, `pronunciada/articulatory.py`, `pronunciada/quality.py`; benchmarks "
         "`bench/coach/`. Everything was fitted or chosen on FLEURS **validation** (planted errors and "
         "synthetic degradations) and Common Voice cv-real **dev**; L2-ARCTIC test, EpaDB test and OMPAL "
         "test are reported only (nothing was tuned on them). No cv-real test clip was read.", ""]
    o += ["## 1. Reliability gate for \"what it sounded like\" (FLEURS validation, planted substitutions)", "",
          "Protocol (`bench/coach/gather.py`): 250 validation clips per language (Korean: 226), the "
          "native audio scored against references in which `bench.corrupt` replaced a sound X by Y "
          "(`phone_sub_near` and `phone_sub_far`, 2 per reference), so the true \"heard\" sound is X. "
          "A planted sound is *flagged* when its probability is < 0.5 (default strictness); a flagged "
          "sound gets a *diagnosis* (`heard_ipa`) when it passes the gate. The heard candidate is the "
          "language's sound of the same type (vowel / consonant) with the highest geometric-mean span "
          "posterior of its tokens (adapted model output, the sound's aligned span ± 1 frame), never "
          "the expected sound or one of its accepted alternatives.", "",
          "Rule, fixed before the numbers were seen: the grid setting (log-ratio heard/expected ∈ "
          "{-1 … 3}, heard posterior ∈ {0 … 0.3}, frames ∈ {0, 1, 2}) with pooled \"within one "
          "articulatory feature\" precision ≥ 0.85 and the highest pooled coverage; then, per "
          "language whose own precision is below 0.85, the loosest heard-posterior bar reaching "
          "0.85 with ≥ 20 diagnoses. \"Within one feature\" = `articulatory.diff(truth, heard)` "
          "has at most one entry (a vowel/consonant change never counts).", "",
          f"Chosen: `min_log_ratio` {th['min_log_ratio']}, `min_heard_post` {th['min_heard_post']}, "
          f"`min_frames` {th['min_frames']}, per-language overrides {json.dumps(th.get('by_lang', {}))}. "
          "The pooled target is met without any gating: once a planted sound is flagged, the best "
          "competing sound is nearly always the true one. The gate binds only in German.", "",
          "| lang | planted | flagged | diagnosed | coverage (of planted) | coverage (of flagged) | exact | within 1 feature | near-miss: cov / within 1 | far: cov / within 1 | native flagged sounds with a (wrong) diagnosis |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for l in list(LANGS) + ["pooled"]:
        s = G["phones"][l]
        o.append(f"| {l} | {s['planted']} | {s['flagged']} | {s['diagnosed']} | {pct(s['coverage'])} | "
                 f"{pct(s['coverage_of_flagged'])} | {f(s['precision_exact'])} | {f(s['precision_within1'])} | "
                 f"{pct(s['near']['coverage'])} / {f(s['near']['precision_within1'])} | "
                 f"{pct(s['far']['coverage'])} / {f(s['far']['precision_within1'])} | "
                 f"{s['clean_diagnosed']} of {s['clean_flagged']} |")
    o += ["", "Coverage is limited by flagging, not by the gate: only 7–52 % of the planted substitutions "
          "are flagged at all (German and Portuguese 7 %). The last column counts sounds of the "
          "*unaltered* native clips that were flagged (false alarms) and still received a heard "
          "sound, which is then necessarily wrong; per clip that is about "
          f"{G['phones']['pooled']['clean_diagnosed'] / 2476:.2f} wrong diagnoses (2,476 clips).", "",
          "Precision / coverage as the log-ratio bar rises (pooled and per language):", "",
          "| lang | " + " | ".join(f"lr ≥ {c['min_log_ratio']:g}" for c in G["curves"]["pooled"]) + " |",
          "|---|" + "---|" * len(G["curves"]["pooled"])]
    for l in list(LANGS) + ["pooled"]:
        o.append(f"| {l} | " + " | ".join(f"{pct(c['coverage'])} / {f(c['precision_within1'])}"
                                          for c in G["curves"][l]) + " |")
    o += ["", "By sound class of the true sound (pooled, diagnosed items):", "",
          "| class | n | within 1 feature |", "|---|---|---|"]
    for c, v in G["classes"].items():
        o.append(f"| {c} | {v['n']} | {f(v['precision_within1'])} |")
    o += ["", "Confidence (`item.confidence`): within-one-feature precision by log-ratio quartile of the "
          f"diagnosed rows, made monotone: edges {th['confidence']['edges']}, values "
          f"{th['confidence']['values']}.", "",
          "**Missing sounds.** Planted extra reference consonants (`phone_ins`: the audio lacks the "
          "sound) cannot be told apart from substitutions and false alarms: the best rule over "
          "frames ≤ 1–3, nothing decoded in the span, `llr_del` ≤ −4…−15 and blank share reached "
          "0.67 precision on 33 items. Below the 0.85 bar, so `kind: \"missing\"` is **off** by "
          "default (`missing_min_llr_del` = −1e9); a skipped sound is reported as a flagged "
          "substitution, usually without `heard_ipa`. Missing *words* use the shipped omission "
          "channel (`present_prob` < 0.5). `kind: \"extra\"` is in the schema but not produced yet.", ""]
    for kind, nm in (("tone", "Tones (Mandarin)"), ("stress", "Word stress (en de es it pt ru)")):
        P = G[kind]
        o += [f"**{nm}.** Items need the syllable probability and the word's calibrated channel "
              f"(`WordResult.{kind}_prob`) both < 0.5; heard = the most probable tone / stressed "
              "syllable. Bar on its posterior (planted: precision = heard equals the original):", "",
              "| bar | planted flagged | diagnosed | precision | coverage of planted | native items with a diagnosis |",
              "|---|---|---|---|---|---|"]
        for c in P["curve"]:
            o.append(f"| {c['bar']} | {c['planted']['flagged']} / {c['planted']['n']} | {c['planted']['diagnosed']} | "
                     f"{f(c['planted']['precision'])} | {pct(c['planted']['coverage'])} | "
                     f"{c['clean']['diagnosed']} of {c['clean']['n']} ({pct(c['clean']['coverage'])}) |")
        o += ["", "By language at the chosen bar: " + "; ".join(
            f"{l} {f(v['planted']['precision'])} precision, {pct(v['planted']['coverage'])} coverage"
            for l, v in P["by_lang"].items()) + ".", ""]
    if L:
        o += ["### 2. Real learners, round 1 (report only)", "",
              "`bench/coach/eval_learners.py`: the shipped gate on L2-ARCTIC test (900 utterances, 24 "
              "speakers of 6 L1s) and EpaDB test (1,263, Spanish L1), phones aligned to the annotators' "
              "as in `bench/eval_english.py`. Precision is over every diagnosis we give on a matched "
              "phone: right when it equals what the annotator heard (canonical sound when marked "
              "correct, nothing when marked deleted). English vowel labels folded ʌ→ə, ə˞→ɜ˞.", "",
              "| corpus | annotated substitutions | flagged | with heard_ipa (coverage) | our diagnoses on matched phones | precision exact | within 1 feature | on substitutions: exact / within 1 | diagnoses on phones marked correct / deleted |",
              "|---|---|---|---|---|---|---|---|---|"]
        for ds, s in [(d, L[d]) for d in ("l2arctic", "epadb")] + \
                [(d + ", strictness 1.0", L["strictness_1"][d]) for d in ("l2arctic", "epadb")]:
            dl = s["diagnosed_labels"]
            o.append(f"| {ds} | {s.get('n_sub', 0)} | {pct(s['flagged_sub'])} | {pct(s['coverage_sub'])} | "
                     f"{sum(dl.values())} | {f(s['precision_exact'])} | {f(s['precision_within1'])} | "
                     f"{f(s['sub_exact'])} / {f(s['sub_within1'])} | {dl['correct']} / {dl['del']} |")
        o += ["", "Coverage is limited by flagging: at the default strictness only ~3 % of the "
              "annotated substitutions get a sound probability below 0.5 (the scores are calibrated "
              "on speechocean762 raters, who accept most accented sounds). Overall precision is low "
              "mainly because many of our diagnoses fall on sounds the annotators marked *deleted* "
              "(we cannot say \"missing\" reliably) or *correct*.", "",
              "Most frequent diagnosed substitutions (expected, annotator, ours, count):", ""]
        for ds in ("l2arctic", "epadb"):
            o.append(f"- {ds}: " + ", ".join(f"/{e}/ → {t} vs {h} ({n})" for e, t, h, n in L[ds]["top_sub"][:10]))
        om = L["ompal"]
        o += ["", f"OMPAL test, learners only ({om['units']} characters, {om['units_any_wrong']} marked wrong "
              "by the teachers): does the *type* of our item match the teachers' mark? Tone item → "
              "tone; syllable-initial consonant → consonant; any other sound → vowel (final).", "",
              "| type | our items (characters) | teachers' marks | P(teacher marks this type \\| our item) | P(teacher marks any error \\| our item) | recall |",
              "|---|---|---|---|---|---|"]
        for t, v in [(t, om[t]) for t in ("tone", "consonant", "vowel")] + \
                [(t + ", strictness 1.0", L["strictness_1"]["ompal"][t]) for t in ("tone", "consonant", "vowel")]:
            o.append(f"| {t} | {v['ours']} | {v['teacher']} | {f(v['precision_type'])} | {f(v['precision_any'])} | {f(v['recall'])} |")
        o1 = L["strictness_1"]["ompal"]
        o += ["", f"Where both we and the teachers flag a character ({om['n_both']}), our type matches one of "
              f"theirs in {pct(om['type_agreement_when_both_flag'])} (strictness 1.0: {o1['n_both']}, "
              f"{pct(o1['type_agreement_when_both_flag'])}).", "",
              "Strictness 1.0 rows: the same results re-graded with `Scorer.rescore(result, 1.0)` "
              "(the app's strictness knob; reported, not chosen on these data).", ""]
    if AU:
        th = AU["thresholds"]
        o += ["## 3. Audio-quality gate", "",
              "`bench/coach/audio_gate.py`: `pronunciada.quality.measure` on every FLEURS validation clip "
              "(3,474, normal audio), synthetic degradations of every 4th clip, and the cv-real dev half "
              "(test A: 1,992 accepted, 1,992 rejected by Common Voice reviewers). Each threshold is the "
              "0.4 % tail of FLEURS validation (clipping: the 99.6 % point, at least 0.2 %), so their union "
              "stays under the 2 % target.", "",
              "| measurement | issue code | threshold |", "|---|---|---|",
              f"| clipped samples share | `clipped` | > {th['clipped_share']} |",
              f"| SNR estimate (p95 − p10 frame level) | `too_noisy` | < {th['snr_db']} dB |",
              f"| speech level | `too_quiet` | < {th['speech_db']} dBFS |",
              f"| speech duration / prompt need | `too_short` | < {th['speech_ratio']} |",
              f"| speech share of the recording, or speech < {th['min_speech_s']} s | `mostly_silence` | < {th['speech_share']} |",
              "", "Seconds of speech per prompt syllable (median, FLEURS validation): " +
              ", ".join(f"{k} {v}" for k, v in AU["sec_per_syl"].items()) + ".", "",
              "| audio | n | flagged (any code) | expected code raised | codes |", "|---|---|---|---|---|"]
        for k, v in AU["rates"].items():
            o.append(f"| {k} | {v['n']} | {pct(v['flagged'])} | {pct(v.get('expected_code'))} | "
                     + ", ".join(f"{c} {pct(x)}" for c, x in v["codes"].items()) + " |")
        o += ["", "Noise lowers the speech-detection threshold's contrast, so noisy clips also raise "
              "`too_short` / `mostly_silence`; white noise at 10 dB SNR is mostly flagged through "
              "`too_short` (28.5 %), `too_noisy` itself needs ≲ 5–8 dB. `too_quiet` is very lenient "
              "(−69 dBFS) because FLEURS validation itself contains very quiet recordings (English, "
              "Japanese, French), which the 2 % budget has to accept. Clipping, truncation and "
              "mostly-silent recordings are caught almost always. On Common Voice dev, rejected "
              "clips are flagged 2.3× as often as accepted ones (13.4 % vs 5.9 %); most rejections "
              "are about the reading, not the audio, so the gate cannot catch them.", "",
              "FLEURS validation flag rate by language: " + ", ".join(
            f"{l} {pct(x)}" for l, x in AU["rates"]["fleurs/clean"]["by_lang"].items()) + ".", ""]
        if "trim" in AU:
            o += [AU["trim"], ""]
    if ID:
        o += ["## 4. Scores unchanged, latency", "",
              f"`bench/coach/identity_check.py`: {ID['clips']} FLEURS validation clips ({ID['per_lang']} per "
              "language) scored by a copy of the package with the pre-coaching `scorer.py` / `types.py` "
              "(commit 773347f) and by the current tree with `result.coaching()` and `tips()` called: every "
              f"sound, word, syllable and sentence probability and score identical as repr() strings: "
              f"**{ID['identical']}**. coaching() + tips(): median {ID['coach_ms_per_10s_median']:.2f} ms, "
              f"90th percentile {ID['coach_ms_per_10s_p90']:.2f} ms per 10 s of audio (native clips flag few "
              f"sounds); worst case measured, a 10 s clip scored against a different sentence (82 "
              f"flagged sounds): {ID['worst_case_ms_10s']:.2f} ms. The audio check "
              f"(always on, `Scorer(audio_check=False)` disables it) median {ID.get('audio_ms_per_10s', 0):.2f} ms "
              "per 10 s. Nothing is computed for coaching unless `coaching()` is called. " + ID.get("note", ""), ""]
    Path("results/coaching.md").write_text("\n".join(o), encoding="utf-8")
    print("wrote results/coaching.md")


if __name__ == "__main__":
    main()
