"""Every row of results/SCOREBOARD.md as a value, read from a folder of results/tables/*.json.

Each row: {"key", "area", "lang", "measure", "value", "azure", "delta", "p", "ci", "cat", "hib",
"kind", "n"}; "cat" is the scoreboard's tally class (ours vs Azure, the table's own paired test,
bench.scoreboard.classify) for the 84 Azure rows and None for the rows without an Azure column.
"kind" says which gate rule applies (bench/opt/gate.py):
  pcc            speechocean762 Pearson rows (drop > 0.002 fails)
  detection_auc  ROC AUC of finding errors / bad clips (drop > 0.002 fails)
  native_fa      share of native speech flagged (rise > 0.1 point fails)
  native_accept  share of native speech accepted (drop > 0.1 point fails)
  rate           hit rates and precisions without a fixed tolerance (paired test only)
  other          everything else (paired test only)
The extraction mirrors bench/scoreboard.py section by section; tests/test_opt_gate.py checks that
the Azure rows and their classes equal the scoreboard's own tally on the committed tables.
"""
from __future__ import annotations

import json
from pathlib import Path

from bench import scoreboard as SB

LANGS = SB.LANGS


def _load(T: Path, name: str):
    p = Path(T) / name
    return json.loads(p.read_text()) if p.exists() else None


def _row(out, area, lang, measure, value, *, kind, hib=True, azure=None, delta=None, p=None, ci=None, n=None,
         tally=False):
    cat = SB.classify(delta, p, ci, hib) if tally else None
    key = f"{area}|{lang}|{measure}"
    out.append({"key": key, "area": area, "lang": lang, "measure": measure, "value": value, "azure": azure,
                "delta": delta, "p": p, "ci": list(ci) if ci is not None else None, "cat": cat, "hib": hib,
                "kind": kind, "n": n, "tally": tally})


def english(T, out):
    E = _load(T, "english_vs_azure.json")
    if not E:
        return
    s = E["so762"]
    rows = [("individual sounds", s["phone"]["ours_B"], s["phone"]["azure"]), ("words", s["word"]["ours"], s["word"]["azure"])]
    rows += [(f"sentence {k}" + (" (intonation)" if k == "prosodic" else ""), s["sentence"][k]["ours"],
              s["sentence"][k]["azure"]) for k in ("accuracy", "fluency", "prosodic", "total")]
    for name, o, a in rows:
        d = o["minus_azure"]
        _row(out, "speechocean762", "en", name, o["pearson"], kind="pcc", azure=a["pearson"], delta=d["delta"],
             p=d["p_value_two_sided"], tally=True)
    for ds, label in (("l2arctic", "L2-ARCTIC test"), ("epadb", "EpaDB test")):
        if ds not in E.get("errdet", {}):          # the screen runs eval_english without errdet
            continue
        a = E["errdet"][ds]["all"]
        o, z = a["ours_B"], a["azure"]
        d = o.get("minus_azure")
        if d:
            _row(out, "learner errors", "en", label, o["auc"], kind="detection_auc", azure=z["auc"], delta=d["delta"],
                 p=d["p_value_two_sided"], tally=True)
    Y = _load(T, "coaching_azure_yardstick.json")
    if Y:
        for k in ("careful", "balanced", "eager"):
            r = Y["at_recall"][k]
            _row(out, "coaching", "en", f"precision at '{k}' recall", r["precision_ours"], kind="rate",
                 azure=r["precision_azure"], delta=r["precision_ours"] - r["precision_azure"], ci=r["diff_ci"], tally=True)
        pr = Y["pr_auc"]
        _row(out, "coaching", "en", "precision-recall AUC", pr["ours"], kind="rate", azure=pr["azure"],
             delta=pr["ours"] - pr["azure"], ci=Y["diff_ap_ci"], tally=True)
        h = Y.get("heard", {})
        o = h.get("ours_balanced") or {}
        for k in ("coverage", "within1"):
            if o.get(k) is not None:
                _row(out, "coaching (heard)", "en", f"names the sound heard: {k}", o[k], kind="other")
    R4 = _load(T, "coaching_r4_heldout.json")
    if R4:
        b = R4["test"]["balanced"]
        for k, alt in (("with_claim", "share_with_claim"), ("precision", "claim_precision"), ("any_right_feature", None)):
            v = b.get(k, b.get(alt) if alt else None)
            if v is not None:
                _row(out, "coaching (features)", "en", f"balanced: {k}", v, kind="rate")


def mandarin(T, out):
    O = _load(T, "ompal.json")
    if O:
        rows = [("word: any error", "word", "primary: any error"), ("word: tone-only error", "word", "(a) tone-only error"),
                ("word: consonant / vowel error", "word", "(b) consonant or vowel error"),
                ("sentence accuracy", "sentence", "accuracy"), ("sentence fluency", "sentence", "fluency"),
                ("sentence prosody", "sentence", "prosody")]
        for name, sec, key in rows:
            x = O["test"][sec].get(key)
            if x is None:
                continue
            o = x["ours"] if sec == "word" else x["pearson"]
            _row(out, "OMPAL", "zh", name, o["value"], kind="detection_auc" if sec == "word" else "other",
                 azure=o["azure"], delta=o["delta"], p=o.get("p_speaker"), tally=True)
    S = _load(T, "swap_ours_vs_azure.json")
    P = (S or {}).get("paired_ours_minus_azure", {}).get("zh", {})
    for k, name, kind in (("tone:auc", "detection (AUC)", "detection_auc"),
                          ("tone:hit@5%FA", "caught at 5 % native false alarms", "rate"),
                          ("tone:right_word", "right word blamed", "rate")):
        x = P.get(k)
        if x:
            _row(out, "tone swaps", "zh", name, x["a"], kind=kind, azure=x["b"], delta=x["delta"], p=x["p_two_sided"],
                 tally=True)


def swaps(T, out):
    S = _load(T, "swap_ours_vs_azure.json")
    if not S:
        return
    P = S.get("paired_ours_minus_azure", {})
    keys = (("swap_far:auc", "far AUC", True, "detection_auc"), ("swap_near:auc", "near AUC", True, "detection_auc"),
            ("swap_near:hit@5%FA", "near hit@5 %", True, "rate"), ("swap_far:right_word", "far right word", True, "rate"),
            ("native:fa<60", "native false alarms", False, "native_fa"))
    for lang in LANGS:
        p = P.get(lang)
        if not p:
            continue
        for k, name, hib, kind in keys:
            x = p.get(k)
            if x:
                _row(out, "swap benchmark", lang, name, x["a"], kind=kind, hib=hib, azure=x["b"], delta=x["delta"],
                     p=x["p_two_sided"], tally=True)


def cvreal(T, out):
    C = _load(T, "cv_real.json")
    if not C:
        return
    for k, r in C["results"].items():
        if k.endswith("Aweak"):
            continue
        p = r["primary"]
        lang = k.split("/")[0].split("-")[0]
        _row(out, f"cv-real test {k.split('/')[1]}", lang, k, p["ours"], kind="detection_auc", azure=p["azure"],
             delta=p["delta"], p=p["p_two_sided"], n=r["n_pos"] + r["n_neg"], tally=True)


def native(T, out):
    G = _load(T, "languages.json")
    if not G:
        return
    for lang in LANGS:
        b = G["by_lang"].get(lang)
        if not b:
            continue
        n = b["native"]["0.5"]
        for k in ("sounds", "words", "sentences"):
            _row(out, "natives accepted", lang, f"{k} accepted", n[k], kind="native_accept")
        _row(out, "planted errors", lang, "sound errors, AUC", b["sound"]["auc"], kind="detection_auc")
        _row(out, "planted errors", lang, "word errors, AUC", b["word"]["auc"], kind="detection_auc")


def contrasts(T, out):
    A = _load(T, "contrast_audit_test.json")
    if not A:
        return
    for lang in LANGS:
        r = A["languages"].get(lang)
        if not r:
            continue
        for cn, c in r["contrasts"].items():
            if not c.get("n") or (lang == "ja" and cn in ("vowel_length", "gemination")):
                continue
            for dn, d in c["directions"].items():
                for k in ("sound_hit5", "word_hit5"):
                    if d.get(k) is not None:
                        _row(out, "contrast audit", lang, f"{cn} | {dn} | {k}", d[k], kind="rate", n=d.get("n"))


def length(T, out):
    H = _load(T, "length_runtime_ja.json")
    if not H:
        return
    for k in ("vowel_length:A>B", "vowel_length:B>A", "gemination:A>B", "gemination:B>A", "all"):
        if k in H["sound"]:
            for lvl in ("sound", "word"):
                if H[lvl][k].get("hit5") is not None:
                    _row(out, "length", "ja", f"{k} {lvl} caught", H[lvl][k]["hit5"], kind="rate", n=H[lvl][k]["n"])
    _row(out, "length", "ja", "native word acceptance", H["native_word_accept_0.5"], kind="native_accept")


def coach_tips(T, out):
    R = _load(T, "coaching_tips_r19.json")
    if not R:
        return
    for lang in LANGS:
        b = R["natives_fleurs_test"].get(lang)
        if b:
            _row(out, "coaching tips on natives", lang, "clips with a pronunciation tip",
                 b["after"]["share_clips_with_pron_tip"], kind="native_fa", hib=False)
            _row(out, "coaching tips on natives", lang, "tips per clip", b["after"]["pron_tips_per_clip"], kind="other",
                 hib=False)
    for key, lang in (("speechocean762_test", "en"), ("ompal_test", "zh")):
        E = R[key]["after"]
        for k, hib in (("a_marked_utts_with_pron_tip", True), ("b_pron_tips_right", True),
                       ("c_clean_utts_with_pron_tip", False)):
            if E.get(k) is not None:
                _row(out, f"coaching tips on learners ({key.split('_')[0]})", lang, k, E[k], kind="rate", hib=hib)


SECTIONS = (english, mandarin, swaps, cvreal, native, contrasts, length, coach_tips)


def extract(T) -> list[dict]:
    out: list[dict] = []
    for f in SECTIONS:
        f(Path(T), out)
    keys = [r["key"] for r in out]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise RuntimeError(f"duplicate row keys: {sorted(dup)[:5]}")
    return out


def scoreboard_tally(T) -> list[dict]:
    """bench.scoreboard's own tally for the tables in T (no file written)."""
    old_t, old_tally = SB.T, list(SB.TALLY)
    SB.T = Path(T)
    SB.TALLY.clear()
    try:
        body: list[str] = []
        for f in (SB.english, SB.mandarin, SB.swaps, SB.cvreal):
            f(body)
        return list(SB.TALLY)
    finally:
        SB.T = old_t
        SB.TALLY.clear()
        SB.TALLY.extend(old_tally)
