"""Insert the coaching round-4 section (attribute heads) at the top of results/coaching.md.

    python -m bench.coach.report_r4
"""
from __future__ import annotations

import json
from pathlib import Path

T = Path("results/tables")
LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")
SENS = ("careful", "balanced", "eager")
MARK = "## Coaching round 4: feature-level claims from attribute heads"


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.1f} %"


def main():
    Hd = json.loads((T / "coaching_r4_heads.json").read_text())
    G = json.loads((T / "coaching_r4_gates.json").read_text())
    E = json.loads((T / "coaching_r4_heldout.json").read_text())
    X = json.loads((T / "coaching_r4_examples.json").read_text())
    o = [MARK, "",
         "**Question.** Most flagged sounds are reported as \"this sound was off\" because naming the exact "
         "replacement is fragile (round 2: 8 % of real English substitutions named). Can small heads on the "
         "acoustic model's internal layers say *which feature* was wrong instead?", "",
         "**Short answer.** The heads hear the audio, not the reference (no leak: on planted swaps they report "
         "the audio's sound 68–95 % of the time, length weakest). On real learners they double the share of substitutions that "
         "get a correct feature-level diagnosis at \"balanced\" (7.6 % → 15.7 % on speechocean762 test, claims "
         "right 84 %), but the 40 % target is reached only at \"eager\" (40.5 % with a claim, 72 % right, "
         "30 % with a right claim), where native speakers get 0.2–2.5 wrong claims per clip. The native cap "
         "(≤ 0.3 per clip at \"balanced\") is what limits coverage. Used as stand-alone detectors, the heads "
         "hear contrasts the scorer is blind to: Korean aspiration AUC 0.97 (scorer 0.64), Russian soft / hard "
         "0.95 (0.72), Japanese vowel length 0.95 (0.75).", "",
         "### Heads (`pronunciada/attr_heads.py`, `data/coach/attr_heads.npz`, "
         f"{Path('pronunciada/data/coach/attr_heads.npz').stat().st_size // 1024} KB)", "",
         "Per feature, a multinomial logistic regression on the round-12b pooled vector of the sound (enc1 | enc3 | "
         "enc5 mean over its aligned frames ± 1) and optionally the mean of its two neighbours' vectors. Trained on "
         f"FLEURS **train** natives, all ten languages pooled ({Hd['n_train']:,} sounds, ≤ 2,500 per language and "
         "head), labels from the reference sounds (`articulatory.py`, grouped: place in 5 classes, vowel height in 3). "
         "Six variants per head (layers × context); the kept one has the best honest accuracy on the leak check. "
         "Numpy at runtime.", "",
         "### Leak check first (FLEURS validation, contrast-audit swaps)", "",
         "The audio has sound X, the reference demands Y; the vector is pooled under the *edited* reference's "
         "alignment. Honest = the head reports X's class; follows = it reports Y's. Natives = unedited sounds of "
         "the same clips.", "",
         "| head | kept variant | natives | swaps | honest | follows reference | honest by language |",
         "|---|---|---|---|---|---|---|"]
    for h, v in Hd["heads"].items():
        c = v["variants"][v["chosen"]]
        o.append(f"| {h} | {v['chosen']} | {f(c['native_acc'])} | {c['swap_n']} | {f(c['swap_honest'])} | "
                 f"{f(c['swap_follows_ref'])} | " + ", ".join(f"{l} {x['honest']:.2f}" for l, x in c["swap_by_lang"].items()) + " |")
    o += ["", "Earlier layers alone were *less* honest (enc1: voicing 0.88, aspiration 0.59, palatalisation 0.55, length 0.52), "
          "so the leak, where present, is mostly the head not hearing the difference rather than reading the "
          "word: the activations do not see the reference, only the pooling span does. All heads were kept; "
          "length (0.68 honest; Japanese 0.63) and palatalisation (0.76) are the weakest.", "",
          "### Claim policy (speechocean762 train) and the native cap", "",
          "A head claims \"feature f differs, heard class c\" on a flagged sound when 1 − P(expected class) ≥ q_f, "
          "the sound's coaching score ≥ t_f, the language's best competing sound (round-1 heard candidate) also "
          "has class c, no accepted alternative changes f, and a named heard sound (if any) agrees. Per "
          "sensitivity, a greedy choice of (q_f, t_f) per head maximises right coverage of real substitutions "
          "(train, out-of-fold flags) with claim precision ≥ 0.8 / 0.75 / 0.75 and at most 0.15 / 0.3 / – wrong "
          "claims per English FLEURS native clip; other languages get a coaching-score floor where their natives "
          "exceed the cap. Rules fixed before the held-out run.", "",
          "| sensitivity | policy (head: q, t) | train: with claim / with right claim / precision |",
          "|---|---|---|"]
    for s in SENS:
        p = G["by_sens"][s]["policy"]
        t = G["by_sens"][s]["train"]
        o.append(f"| {s} | " + "; ".join(f"{h} {v['q']}, {v['t']}" for h, v in p.items() if v) +
                 f" | {pct(t['with_claim'])} / {pct(t['with_right_claim'])} / {f(t['precision'])} |")
    o += ["", "Heads never enabled for claims: aspiration, palatalisation, nasality and length (English learners "
          "give no labelled examples, so no gate can be fitted) and vowel height (best train precision 0.65).", "",
          "Wrong feature claims per FLEURS validation native clip (~10 s; every claim on a native is wrong):", "",
          "| lang | flags per clip (balanced) | claims per clip careful / balanced / eager | balanced floor |", "|---|---|---|---|"]
    for l in LANGS:
        o.append(f"| {l} | {G['native_flags'][l]['balanced']:.2f} | " + " / ".join(f"{G['native'][l][s]:.2f}" for s in SENS)
                 + f" | {G['by_lang_floor'][l]['balanced']} |")
    o += ["", "### Held-out, once (speechocean762 test, 568 real substitutions with the pronounced phone)", "",
          "| sensitivity | substitutions with a feature claim | with a right claim | claim precision | with any right feature (claims or the named sound) | today (named sound only) | claims on sounds the experts scored 2 |",
          "|---|---|---|---|---|---|---|"]
    for s in SENS:
        t = E["test"][s]
        o.append(f"| {s} | {pct(t['with_claim'])} | {pct(t['with_right_claim'])} | {f(t['precision'])} | "
                 f"{pct(t['any_right_feature'])} | {pct(E['today'][s]['any_right_feature'])} | {t['claims_on_expert_correct']} |")
    o += ["", "Per head at \"balanced\" (test): " + ", ".join(f"{h} {f(v['precision'])} (n {v['n']})"
                                                          for h, v in E["test"]["balanced"]["per_head"].items())
          + ". At \"eager\": " + ", ".join(f"{h} {f(v['precision'])} (n {v['n']})" for h, v in E["test"]["eager"]["per_head"].items()) + ".", "",
          "Curve without the native cap (balanced flags, every head at the same q, t; test):", "",
          "| q | t | with claim | precision |", "|---|---|---|---|"]
    for c in E["curve"]:
        if c["t"] in (0.0, 0.6, 0.8):
            o.append(f"| {c['q']} | {c['t']} | {pct(c['with_claim'])} | {f(c['precision'])} |")
    o += ["", "**Where it falls short.** ≥ 40 % coverage at ≥ 0.75 precision is not reached on test: the closest "
          "points are 40.5 % at 0.72 (eager) and 20 % at 0.78 (q 0.999). At \"balanced\" the native cap keeps it to "
          "13.6 %. The offline evaluation does not drop sounds of words the omission channel marks missing (the "
          "runtime does), a small difference.", "",
          "L2-ARCTIC / EpaDB test (report only, balanced): " + "; ".join(
              f"{ds} {pct(v['with_claim'])} of annotated substitutions get a claim, precision {f(v['precision'])}, any right "
              f"feature {pct(v['any_right_feature'])}" for ds, v in E["learners"].items()) +
          ". These corpora's annotated substitutions are mostly not flagged at all (round 2), so the heads add little there.", "",
          "OMPAL (Mandarin learners, balanced; teachers mark the type only, a claim on an initial counts right when the "
          "consonant is marked wrong): " + "; ".join(
              f"{sp}: initials {E['ompal'][sp]['initial']['with_claim']} characters with a claim, precision "
              f"{f(E['ompal'][sp]['initial']['claim_precision'])}; finals {E['ompal'][sp]['final']['with_claim']}, "
              f"{f(E['ompal'][sp]['final']['claim_precision'])}" for sp in ("dev", "test")) +
          ". Coverage is tiny: Mandarin flags are rare and the aspiration and retroflex heads are not enabled for "
          "claims (no learner labels to gate them).", "",
          "### Side result: the heads as stand-alone detectors on blind / weak contrasts", "",
          "Score = 1 − P(the reference sound's class) of the matching head; positives = the audit's swaps, negatives = "
          "native sounds of the same members (the audit used all native sounds). Report only.", "",
          "| contrast | head | swaps | heads AUC | heads hit@5 % | audit AUC | audit hit@5 % | heads AUC per direction (A>B / B>A) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in Hd["side"].items():
        if "auc" not in v:
            continue
        bd = v["by_direction"]
        o.append(f"| {k} | {v['head']} | {v['n_swaps']} | {f(v['auc'])} | {pct(v['hit5'])} | {f(v['audit_auc'])} | "
                 f"{pct(v['audit_hit5'])} | {f(bd.get('A>B', {}).get('auc'))} / {f(bd.get('B>A', {}).get('auc'))} |")
    o += ["", "A>B = the audio has the contrast's first member (e.g. aspirated, soft, long, geminate, nasal) and the "
          "reference demands the second. Korean lax / tense cannot be written in the reference, so it is not "
          "testable. The heads separate Korean and Mandarin aspiration, Russian soft / hard and Japanese length "
          "well in both directions, where the shipped scorer is blind in at least one; separate detectors are "
          "still needed for Italian doubling and Portuguese nasal vowels (the front-ends write those as two sounds, "
          "so the per-sound heads do not see them: AUC 0.63 / 0.55).", "",
          "### Examples (speechocean762 test, balanced; a feature tip without a named sound)", ""]
    for k in ("so762-011350064", "so762-014220140"):
        v = X.get(k)
        if not v:
            continue
        it = v["item"]
        e = v["experts"]
        o.append(f"- “{v['text']}”, /{it['expected']}/ in “{it['word']}”: claim {it['features'][0]['feature']} "
                 f"{it['features'][0]['expected']} → {it['features'][0]['heard']} (p {it['features'][0]['p']}); tip: "
                 f"“{v['tip']['text']}” Experts: {e['word']} {e['phones']} scores {e['scores']}, pronounced "
                 + ", ".join(f"{m['canonical_phone']}→{m['pronounced_phone']}" for m in e["mispronunciations"]) + ".")
    o += ["", "No score changed: 50 clips (5 per language, `PYTHONHASHSEED=0`) identical to the previous commit "
          "(the scorer keeps the activations it already computes in the result's cache). coaching() with the heads: "
          "median 1.1 ms, 90th percentile 1.7 ms per 10 s.", ""]
    md = Path("results/coaching.md").read_text(encoding="utf-8")
    if MARK in md:
        a = md.index(MARK)
        b = md.index("\n## ", a + 5)
        md = md[:a] + md[b + 1:]
    k = md.index("\n## ")
    md = md[:k + 1] + "\n".join(o) + "\n" + md[k + 1:]
    Path("results/coaching.md").write_text(md, encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main()
