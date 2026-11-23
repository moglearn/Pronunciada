"""Native false-alarm taxonomy on the practice sets (see features.py for every test).

    python -m bench.round20_native_fa.taxonomy --lang en,de,ko,es,fr

Populations (correct words only: every word of control items, the unchanged words of changed items):
  FA     error signal >= the FA@80 threshold of that language's practice set
  low06  word score < 0.6 (the app's fixed threshold)
Exclusive buckets, fixed priority (first test that fires):
  noise > neighbour > variant (rescued) > boundary > function/short > tail > residual
Raw (overlapping) hit rates of every test are reported beside them, with the same tests on all
correct words as the base rate. Intervals: 95 % percentile, 1,000 bootstrap draws over speakers.
Tables: $CACHE/round20/nativefa/tables/taxonomy_<lang>.json (copied to results/tables by the write-up).
"""
from __future__ import annotations

import argparse
import math
import random
import re
from collections import Counter, defaultdict

import numpy as np

from bench.round20_native_fa import core as K
from bench.round20_native_fa import features as FT

ORDER = ("noise", "neighbour", "variant", "text", "boundary", "function", "tail", "residual")
LOGIT_MARGIN = 1.0
JITTER_S = 0.15


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def prompt_index(r) -> int | None:
    """Index of target token r['k'] in the clip's own prompt (None for the changed / inserted word)."""
    if not r["changed_item"]:
        return r["k"]
    if r["k"] == r["pos"]:
        return None
    if r["kind"] == "word_omission":
        return r["k"] if r["k"] < r["pos"] else r["k"] - 1
    return r["k"]


def enrich(lang: str, rows: list[dict], cf: dict | None, full_audio: set) -> None:
    """Add every test field to the rows (audio tests only for rows whose item id is in full_audio)."""
    its = {it["id"]: it for it in K.practice_items(lang)}
    dump = K.dump_rec(lang)
    A = FT.asr(lang)
    asr_cache = {}
    edge_cache = {}
    ref_cache = {}
    thr = K.threshold(rows)
    thr_score = 1 - thr
    for r in rows:
        it, rec = its[r["item"]], dump[r["item"]]
        rw = [rec["rwords"][j] for j in r["iw"]]
        q = rec.get("audio_q") or {}
        r["dur"] = (max(w["end"] for w in rw) - min(w["start"] for w in rw)) if rw else 0.0
        r["start"] = min(w["start"] for w in rw) if rw else None
        r["end"] = max(w["end"] for w in rw) if rw else None
        r["n_snd"] = sum(len(w["phones"]) for w in rw)
        r["rate"] = r["n_snd"] / r["dur"] if r["dur"] > 0 else float("nan")
        r["function"] = FT.is_function(lang, r["tok"])
        r["short"] = r["dur"] < 0.12 or r["n_snd"] <= 2
        # --- noise
        j = prompt_index(r)
        if it["clip"] not in asr_cache:
            asr_cache[it["clip"]] = FT.asr_words(it["prompt"], A.get(it["clip"]))
        aw = asr_cache[it["clip"]][j] if j is not None and j < len(asr_cache[it["clip"]]) else {"mismatch": None, "t": None}
        r["asr_mismatch"] = aw["mismatch"]
        r["asr_cer"] = (A.get(it["clip"]) or {}).get("cer")
        r["snr"] = q.get("snr_db")
        r["low_snr"] = r["snr"] is not None and r["snr"] < 15
        first, last = r["k"] == 0, r["k"] == r["n_tok"] - 1
        r["first"], r["last"] = first, last
        if it["audio"] not in edge_cache:
            try:
                edge_cache[it["audio"]] = FT.edge_db(it["audio"])
            except Exception:  # noqa: BLE001
                edge_cache[it["audio"]] = (None, None)
        e0, e1 = edge_cache[it["audio"]]
        sp = q.get("speech_db")
        r["clip_on"] = e0 is not None and sp is not None and e0 > sp - 10 and (q.get("lead_sil_s") or 0) < 0.05
        r["clip_off"] = e1 is not None and sp is not None and e1 > sp - 10 and (q.get("trail_sil_s") or 0) < 0.05
        r["truncated"] = (first and r["clip_on"]) or (last and r["clip_off"])
        # a prompt token with no letter or digit (French « » ! ? : ; written with spaces, Spanish ¿ ¡): the
        # metric counts it as a word, no system scores it, so its error signal is 1 (a metric artifact)
        r["punct"] = FT.key(r["tok"]) == ""
        # practice-set label artifact (index_check.py): the builder's word_index counts front-end units, the
        # metric counts whitespace tokens; after a punctuation-only token the true changed word is scored as
        # a correct word. Marked here: a correct-labelled token equal to the replacement / added word of a
        # misindexed item.
        lab = it.get("label") or {}
        new = lab.get("replacement") or lab.get("added")
        toks0 = it["target"].split()
        r["label_artifact"] = bool(lab) and not r["label"] and new is not None and FT.key(r["tok"]) == FT.key(new) \
            and FT.key(toks0[r["pos"]]) != FT.key(new)
        # metric tokenisation artifact: a token with letters or digits that the metric's key alignment leaves
        # without a word (e.g. "1 400" written with a no-break space: two prompt tokens, one front-end word)
        r["tok_artifact"] = bool(r.get("unaligned")) and not r["punct"]
        r["noise"] = bool(r["asr_mismatch"]) or r["low_snr"] or r["truncated"] or r["punct"] or r["label_artifact"] \
            or r["tok_artifact"]
        # --- neighbour
        # within two words of the changed word (the paired table below shows the effect by distance)
        r["dist"] = abs(r["k"] - r["pos"]) if (r["changed_item"] and r["pos"] is not None) else None
        r["neighbour"] = r["dist"] is not None and 1 <= r["dist"] <= 2 and not r["label"]
        # --- boundary
        r["jitter"] = None
        if aw["t"] is not None and r["start"] is not None:
            r["jitter"] = max(abs(aw["t"][0] - r["start"]), abs(aw["t"][1] - r["end"]))
        r["click"] = None
        if r["item"] in full_audio and r["start"] is not None:
            try:
                r["click"] = FT.click_near(it["audio"], r["start"], r["end"])
            except Exception:  # noqa: BLE001
                r["click"] = None
        r["boundary"] = first or last or r["clip_on"] and first or r["clip_off"] and last or bool(r["click"]) \
            or (r["jitter"] is not None and r["jitter"] > JITTER_S)
        # --- variant
        c = cf.get((r["item"], r["k"])) if cf else None
        vk = var_kinds(lang, it, r, rw, ref_cache)
        r["has_lex"] = "lex" in vk                   # another reading in the shipped lexicon
        r["has_brit"] = "brit" in vk                 # en: a different Britfone reading
        r["has_rule"] = any(k.startswith("rule:") for k in vk)
        r["has_alt"] = any(p["alt"] != 0 for w in rw for p in w["phones"])
        # --- text front-end risk: unknown word (letter rules), name, number / abbreviation expansion
        oov = {w.split(":", 1)[1] for w in rec.get("warnings", []) if w.startswith("oov:")}
        toks = it["target"].split()
        r["oov"] = any(FT.key(x) in oov for x in re.split(r"[-']", r["tok"].lower()) if x) or FT.key(r["tok"]) in oov
        prev = toks[r["k"] - 1] if r["k"] > 0 else ""
        cap = r["tok"][:1].isupper() and r["k"] > 0 and not prev.endswith((".", "!", "?", ":"))
        r["name"] = lang != "ko" and cap and (lang != "de" or r["oov"])
        r["number"] = any(ch.isdigit() for ch in r["tok"]) or len(r["iw"]) > 1 or bool(
            re.search(r"\w\.\w", r["tok"])) or (len(re.sub(r"\W", "", r["tok"])) >= 2 and re.sub(r"\W", "", r["tok"]).isupper())
        r["text_risk"] = r["oov"] or r["name"] or r["number"]
        # Whisper's spelling of names, unknown words and numbers is not a reliable reading check (as for digits):
        # for those tokens the transcript test does not count as corpus noise
        r["asr_noise"] = bool(r["asr_mismatch"]) and not r["text_risk"]
        r["noise"] = r["asr_noise"] or r["low_snr"] or r["truncated"] or r["punct"] or r["label_artifact"] \
            or r["tok_artifact"]
        r["cf"] = c is not None
        if c and c["var"]:
            best = max(c["var"], key=lambda v: v["p"])
            r["best_p"], r["best_kind"] = best["p"], best["kind"]
            r["best_score"] = round(min(1.0, max(0.0, best["p"])) * 100, 1) / 100
            r["kind_scores"] = {}
            for v in c["var"]:
                k0 = v["kind"].split(":")[0]
                r["kind_scores"][k0] = max(r["kind_scores"].get(k0, 0.0), round(v["p"] * 100, 1) / 100)
        else:
            r["best_p"] = r["best_kind"] = r["best_score"] = None
            r["kind_scores"] = {}
        r["rescued_fa"] = r["best_score"] is not None and r["best_score"] > thr_score
        r["rescued_06"] = r["best_score"] is not None and r["best_score"] >= K.APP_THRESHOLD
        # --- evidence
        r["driver"], r["low_snd"], r["low_kind"], r["ins_flag"] = evidence(lang, rw)
        r["tail_margin"] = logit(thr_score) - logit(max(r["score"], 1e-4))


def var_kinds(lang: str, it: dict, r: dict, rw: list[dict], cache: dict) -> set:
    """Kinds of alternative readings that exist for the word (variants.all_variants, no aligner input),
    independent of the counterfactual run, so the base rate covers every word."""
    from bench.round20_native_fa import variants as VA
    from pronunciada.text import get_g2p

    if len(r["iw"]) != 1:
        return set()
    if it["id"] not in cache:
        g2p = cache.setdefault("_g2p", get_g2p(lang))
        try:
            cache[it["id"]] = g2p(it["target"])
        except Exception:  # noqa: BLE001
            cache[it["id"]] = None
    ref = cache[it["id"]]
    if ref is None or r["iw"][0] >= len(ref.words):
        return set()
    return {k for k, _, _ in VA.all_variants(lang, cache["_g2p"], ref, r["iw"][0], None)}


def evidence(lang: str, rw: list[dict]):
    """(factor that lowers the word most, lowest sound 'expected>heard', its evidence kind, insertion flag)."""
    if not rw:
        return "missing", None, None, False
    w = min(rw, key=lambda x: x["prob"] if x["prob"] is not None else 1.0)
    lg = lambda x: math.log(max(x, 1e-6)) if x is not None else 0.0  # noqa: E731
    parts = {"sounds": lg(w["sounds"]), "length": lg(w["length"]), "present": lg(w["present"]),
             "identity": lg(w["idf"]) if w["idf"] is not None else 0.0}
    if lang == "ko":
        parts["identity"] = min(0.0, w["ko_delta"] or 0.0)
        parts["contrast"] = 0.0
    else:
        parts["contrast"] = lg(w["contrast"])
    known = sum(v for k, v in parts.items() if k != "identity" or lang != "ko")
    parts["stress"] = min(0.0, lg(w["prob"]) - known) if lang != "ko" else 0.0
    if lang in ("ko",):
        parts["stress"] = 0.0
    driver = min(parts, key=lambda k: parts[k])
    if parts[driver] > -0.05:
        driver = "none"
    ph = [p for p in w["phones"] if p["p"] is not None]
    if not ph:
        return driver, None, None, False
    s = min(ph, key=lambda p: p["p"])
    heard = s["heard"] if s["heard"] is not None else "?"
    # llr_sub = L_can - L_wild (<= 0), llr_del = L_can - L_del. The sound is "unsupported" when dropping it
    # explains the audio at least as well (llr_del <= 0); then "deletion" if another sound in its place
    # explains it hardly better than nothing (L_wild - L_del = llr_del - llr_sub < 2 nats), else "substitution".
    # Supported sounds whose wildcard is clearly better (llr_sub <= -2) are also "substitution"; the rest "weak".
    ld, ls = s["llr_del"], s["llr_sub"]
    if ld is not None and ls is not None and ld <= 0:
        kind = "deletion" if ld - ls < 2 else "substitution"
    elif ls is not None and ls <= -2:
        kind = "substitution"
    else:
        kind = "weak"
    return driver, f"{s['ipa']}>{heard or '∅'}", kind, (w["ins"] is not None and w["ins"] <= -3)


def fires(r) -> dict:
    """Which bucket tests fire for one word (the bucket is the first in ORDER that fires)."""
    return {"noise": bool(r["noise"]),
            # the neighbour effect is attributed only when the same word in the control twin (same audio,
            # the clip's own text) is NOT a false alarm: the change itself caused it
            "neighbour": bool(r["neighbour"]) and r.get("twin_fa") is False,
            "variant": bool(r.get("_rescued")),
            "text": bool(r["text_risk"]),
            "boundary": bool(r["boundary"]),
            "function": bool(r["function"] or r["short"]),
            "tail": 0 <= r.get("_margin", r["tail_margin"]) <= LOGIT_MARGIN and not r["has_lex"],
            "residual": True}


def assign(r, order=None) -> str:
    f = fires(r)
    return next(b for b in (order or ORDER) if f[b])


TESTS = ("noise", "punct", "label_artifact", "tok_artifact", "asr_mismatch", "asr_noise", "low_snr", "truncated", "neighbour", "neighbour_twin_ok", "neighbour_omission",
         "has_lex", "has_brit", "has_rule", "has_alt", "_rescued", "text_risk", "oov", "name", "number",
         "boundary", "first", "last", "clip_on", "clip_off", "click", "jitter_gt", "function", "short", "tail_zone")


def test_vals(r) -> dict:
    return {"noise": r["noise"], "punct": r["punct"], "label_artifact": r["label_artifact"], "tok_artifact": r["tok_artifact"], "asr_mismatch": bool(r["asr_mismatch"]), "asr_noise": r["asr_noise"], "low_snr": r["low_snr"],
            "truncated": r["truncated"], "neighbour": r["neighbour"],
            "neighbour_twin_ok": bool(r["neighbour"]) and r.get("twin_fa") is False,
            "neighbour_omission": bool(r["neighbour"]) and r["kind"] == "word_omission",
            "has_lex": r["has_lex"], "has_brit": r["has_brit"], "has_rule": r["has_rule"],
            "has_alt": r["has_alt"], "_rescued": bool(r.get("_rescued")), "text_risk": r["text_risk"], "oov": r["oov"],
            "name": r["name"], "number": r["number"], "boundary": r["boundary"], "first": r["first"],
            "last": r["last"], "clip_on": r["clip_on"], "clip_off": r["clip_off"], "click": bool(r["click"]),
            "jitter_gt": r["jitter"] is not None and r["jitter"] > JITTER_S, "function": r["function"],
            "short": r["short"], "tail_zone": 0 <= r.get("_margin", r["tail_margin"]) <= LOGIT_MARGIN}


def share_ci(rows, pred, n=1000, seed=1):
    g = defaultdict(list)
    for r in rows:
        g[r["speaker"]].append(1.0 if pred(r) else 0.0)
    groups = list(g.values())
    rng = random.Random(seed)
    v = []
    for _ in range(n):
        pick = [groups[rng.randrange(len(groups))] for _ in range(len(groups))]
        a = [x for gg in pick for x in gg]
        v.append(sum(a) / max(1, len(a)))
    v.sort()
    return [v[int(0.025 * n)], v[int(0.975 * n) - 1]]


def run(lang: str, seed: int = 0) -> dict:
    from bench.round20_native_fa.counterfactual import load as load_cf

    rows = K.word_rows(lang)
    try:
        cf = load_cf(lang)
    except FileNotFoundError:
        cf = {}
    thr = K.threshold(rows)
    auc, fa80 = K.metrics(rows)
    ci = K.boot(rows, K.metrics)
    neg = [r for r in rows if not r["label"]]
    fa_items = {r["item"] for r in neg if r["err"] >= thr or r["score"] < K.APP_THRESHOLD}
    rng = random.Random(seed)
    base_sample = rng.sample(neg, min(3000, len(neg)))
    full = fa_items | {r["item"] for r in base_sample}
    enrich(lang, rows, cf, full)
    twin = {(r["clip"], r["k"]): r for r in rows if not r["changed_item"]}
    for r in rows:
        if r["changed_item"] and not r["label"]:
            j = prompt_index(r)
            t = twin.get((r["clip"], j))
            r["twin_fa"] = None if t is None else bool(t["err"] >= thr)
    out = {"lang": lang, "n_items": len({r["item"] for r in rows}), "n_words": len(rows),
           "n_changed": sum(r["label"] for r in rows), "n_correct": len(neg),
           "auc": auc, "auc_ci": ci[0], "fa80": fa80, "fa80_ci": ci[1], "thr_score": 1 - thr,
           "n_speakers": len({r["speaker"] for r in rows})}
    pops = {"FA": [r for r in neg if r["err"] >= thr], "low06": [r for r in neg if r["score"] < K.APP_THRESHOLD]}
    for name, pop in pops.items():
        for r in pop:
            r["_rescued"] = r["rescued_fa"] if name == "FA" else r["rescued_06"]
            r["_margin"] = r["tail_margin"] if name == "FA" else logit(K.APP_THRESHOLD) - logit(max(r["score"], 1e-4))
            r["_bucket"] = assign(r)
            r[f"_bucket_{name}"], r[f"_rescued_{name}"] = r["_bucket"], r["_rescued"]
        cnt = Counter(r["_bucket"] for r in pop)
        res = {"n": len(pop), "buckets": {}, "overlap": {}, "base_rate": {}}
        for b in ORDER:
            res["buckets"][b] = {"n": cnt[b], "pct": 100 * cnt[b] / max(1, len(pop)),
                                 "ci": [100 * x for x in share_ci(pop, lambda r, b=b: r["_bucket"] == b)]}
        for t in TESTS:
            res["overlap"][t] = 100 * sum(test_vals(r)[t] for r in pop) / max(1, len(pop))
        bs = [r for r in base_sample]
        for r in bs:
            r.setdefault("_rescued", False)
        for t in TESTS:
            if t == "_rescued":
                continue
            res["base_rate"][t] = 100 * sum(test_vals(r)[t] for r in bs) / max(1, len(bs))
        # alternative priority orders: how much the order matters
        alt_orders = {"variant_first": ("variant", "text", "noise", "neighbour", "boundary", "function", "tail", "residual"),
                      "boundary_first": ("boundary", "noise", "neighbour", "variant", "text", "function", "tail", "residual"),
                      "function_first": ("function", "noise", "neighbour", "variant", "text", "boundary", "tail", "residual"),
                      "text_last": ("noise", "neighbour", "variant", "boundary", "function", "text", "tail", "residual")}
        res["alt_orders"] = {}
        for nm, order in alt_orders.items():
            c2 = Counter(assign(r, order) for r in pop)
            res["alt_orders"][nm] = {b: 100 * c2[b] / max(1, len(pop)) for b in ORDER}
        # evidence breakdown
        res["driver"] = dict(Counter(r["driver"] for r in pop).most_common())
        res["low_kind"] = dict(Counter(r["low_kind"] for r in pop if r["driver"] == "sounds").most_common())
        res["ins_flag"] = 100 * sum(r["ins_flag"] for r in pop) / max(1, len(pop))
        res["low_pairs"] = Counter(r["low_snd"] for r in pop if r["driver"] == "sounds").most_common(25)
        res["best_kind_rescue"] = dict(Counter(r["best_kind"].split(":")[0] if r["best_kind"] else None
                                               for r in pop if r["_rescued"]).most_common())
        res["rescue_by_rule"] = dict(Counter(r["best_kind"] for r in pop if r["_rescued"]).most_common())
        res["cf_covered"] = sum(r["cf"] for r in pop)
        out[name] = res
    # neighbour effect, paired with the same word in the control item of the same clip
    ctl = {(r["clip"], r["k"]): r for r in rows if not r["changed_item"]}
    pairs = []
    for r in rows:
        if r["changed_item"] and r["neighbour"]:
            j = prompt_index(r)
            c = ctl.get((r["clip"], j))
            if c is not None:
                pairs.append((r, c))
    out["neighbour_paired"] = {
        "n": len(pairs),
        "fa_changed": 100 * sum(a["err"] >= thr for a, _ in pairs) / max(1, len(pairs)),
        "fa_control": 100 * sum(b["err"] >= thr for _, b in pairs) / max(1, len(pairs)),
        "mean_dscore": float(np.mean([a["score"] - b["score"] for a, b in pairs])) if pairs else None,
        "fa_other_changed_item_words": 100 * sum(r["err"] >= thr for r in neg if r["changed_item"] and not r["neighbour"])
        / max(1, sum(1 for r in neg if r["changed_item"] and not r["neighbour"])),
        "fa_control_items": 100 * sum(r["err"] >= thr for r in neg if not r["changed_item"])
        / max(1, sum(1 for r in neg if not r["changed_item"])),
    }
    by_dist = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        if r["changed_item"] and not r["label"] and r["dist"] is not None:
            c = ctl.get((r["clip"], prompt_index(r)))
            if c is None:
                continue
            key = ("omission" if r["kind"] == "word_omission" else "other", min(r["dist"], 4))
            by_dist[key][0] += 1
            by_dist[key][1] += r["err"] >= thr
            by_dist[key][2] += c["err"] >= thr
    out["neighbour_by_distance"] = {f"{a}|{d}{'+' if d == 4 else ''}": {"n": v[0], "fa_changed": 100 * v[1] / v[0],
                                                                         "fa_control": 100 * v[2] / v[0]}
                                    for (a, d), v in sorted(by_dist.items())}
    nb_by_kind = defaultdict(lambda: [0, 0, 0])
    for a, b in pairs:
        if a["dist"] != 1:
            continue
        x = nb_by_kind[a["kind"]]
        x[0] += 1
        x[1] += a["err"] >= thr
        x[2] += b["err"] >= thr
    out["neighbour_by_kind"] = {k: {"n": v[0], "fa_changed": 100 * v[1] / v[0], "fa_control": 100 * v[2] / v[0]}
                                for k, v in sorted(nb_by_kind.items())}
    # FA rate by duration and by number of sounds, function vs content (correct words)
    curves = {}
    for nm, sel in (("function", lambda r: r["function"]), ("content", lambda r: not r["function"])):
        by_d, by_n, by_r = (defaultdict(lambda: [0, 0]) for _ in range(3))
        for r in neg:
            if not sel(r):
                continue
            d = min(int(r["dur"] / 0.05), 12)
            by_d[d][0] += 1
            by_d[d][1] += r["err"] >= thr
            n = min(r["n_snd"], 10)
            by_n[n][0] += 1
            by_n[n][1] += r["err"] >= thr
            if r["rate"] == r["rate"]:                       # sounds per second of aligned time, bins of 5
                q = min(int(r["rate"] / 5), 8)
                by_r[q][0] += 1
                by_r[q][1] += r["err"] >= thr
        curves[nm] = {"dur_50ms": {f"{50 * d}-{50 * d + 50}": [v[0], 100 * v[1] / v[0]] for d, v in sorted(by_d.items())},
                      "n_sounds": {str(n): [v[0], 100 * v[1] / v[0]] for n, v in sorted(by_n.items())},
                      "sounds_per_s": {f"{5 * q}-{5 * q + 5}": [v[0], 100 * v[1] / v[0]] for q, v in sorted(by_r.items())}}
    out["curves"] = curves
    out["fa_rate_function"] = 100 * sum(r["err"] >= thr for r in neg if r["function"]) / max(1, sum(r["function"] for r in neg))
    out["fa_rate_content"] = 100 * sum(r["err"] >= thr for r in neg if not r["function"]) / max(1, sum(not r["function"] for r in neg))
    out["share_function_all"] = 100 * sum(r["function"] for r in neg) / len(neg)
    # tail: mass of correct words just above / below the threshold (in logit units of the score)
    m = np.array([r["tail_margin"] for r in neg])
    out["tail_mass"] = {f"{a}..{b}": 100 * float(np.mean((m > a) & (m <= b))) for a, b in
                        ((-3, -2), (-2, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 99))}
    out["fa_if_thr_shift"] = {str(d): 100 * float(np.mean(m >= d)) for d in (-0.5, -0.25, 0.0, 0.25, 0.5)}
    # evidence base rates (every correct word) and FA rate by the lowest sound's evidence kind
    out["driver_all"] = dict(Counter(r["driver"] for r in neg).most_common())
    out["low_kind_all"] = dict(Counter(r["low_kind"] for r in neg if r["driver"] == "sounds").most_common())
    out["fa_rate_by_low_kind"] = {k: 100 * sum(r["err"] >= thr for r in neg if r["low_kind"] == k)
                                  / max(1, sum(r["low_kind"] == k for r in neg)) for k in ("deletion", "substitution", "weak")}
    out["fa_rate_by_driver"] = {k: 100 * sum(r["err"] >= thr for r in neg if r["driver"] == k)
                                / max(1, sum(r["driver"] == k for r in neg)) for k in out["driver_all"]}
    pos_rows = [r for r in rows if r["label"]]
    out["changed_driver"] = dict(Counter(r["driver"] for r in pos_rows).most_common())
    out["changed_low_kind"] = dict(Counter(r["low_kind"] for r in pos_rows if r["driver"] == "sounds").most_common())
    out["changed_low_kind_by_bucket"] = {b: dict(Counter((r["driver"], r["low_kind"]) if r["driver"] == "sounds"
                                                         else (r["driver"], None) for r in pos_rows if r["kind"] == b)
                                                 .most_common()) for b in sorted({r["kind"] for r in pos_rows})}
    out["changed_low_kind_by_bucket"] = {b: {f"{a}/{c}": n for (a, c), n in v.items()}
                                         for b, v in out["changed_low_kind_by_bucket"].items()}
    fa_rows = [r for r in neg if r["err"] >= thr]
    out["FA_driver_x"] = {
        "neighbour": dict(Counter(r["driver"] for r in fa_rows if r["neighbour"]).most_common()),
        "not_neighbour": dict(Counter(r["driver"] for r in fa_rows if not r["neighbour"]).most_common()),
        "function": dict(Counter(r["driver"] for r in fa_rows if r["function"]).most_common()),
        "content": dict(Counter(r["driver"] for r in fa_rows if not r["function"]).most_common()),
        "first_or_last": dict(Counter(r["driver"] for r in fa_rows if r["first"] or r["last"]).most_common())}
    # deletion-type evidence: FA share among correct words vs catch share among changed words
    dl = lambda r: r["driver"] == "present" or (r["driver"] == "sounds" and r["low_kind"] == "deletion")  # noqa: E731
    out["deletion_type"] = {"FA_share": 100 * sum(dl(r) for r in neg if r["err"] >= thr) / max(1, sum(r["err"] >= thr for r in neg)),
                            "caught_changed_share": 100 * sum(dl(r) for r in pos_rows if r["err"] >= thr)
                            / max(1, sum(r["err"] >= thr for r in pos_rows)),
                            "all_correct_share": 100 * sum(dl(r) for r in neg) / len(neg)}
    # extra-sound (insertion) evidence next to / inside the word: llr_ins <= -3
    out["insertion_flag"] = {"FA_share": 100 * sum(r["ins_flag"] for r in neg if r["err"] >= thr) / max(1, sum(r["err"] >= thr for r in neg)),
                             "caught_changed_share": 100 * sum(r["ins_flag"] for r in pos_rows if r["err"] >= thr)
                             / max(1, sum(r["err"] >= thr for r in pos_rows)),
                             "all_correct_share": 100 * sum(r["ins_flag"] for r in neg) / len(neg)}
    # length: FA rate (correct words) and catch rate (changed words) at the one threshold, by number of sounds
    by = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        nb = r["n_snd"] if r["n_snd"] < 4 else min(2 * (r["n_snd"] // 2), 12)
        acc = by[nb]
        if r["label"]:
            acc[2] += 1
            acc[3] += r["err"] >= thr
        else:
            acc[0] += 1
            acc[1] += r["err"] >= thr
    out["by_n_sounds"] = {str(nb): {"n_correct": v[0], "fa_pct": 100 * v[1] / max(1, v[0]), "n_changed": v[2],
                                    "catch_pct": 100 * v[3] / max(1, v[2])} for nb, v in sorted(by.items())}
    # reverse safety check: best-of-N on every word (changed words too)
    out["best_of_n"] = best_of_n(rows, thr)
    out["_rows"] = rows
    return out


def best_of_n(rows, thr) -> dict:
    """AUC / FA@80 when every word re-scored in the counterfactual run takes its best variant, per source kind."""
    res = {}
    kinds = {"all": None, "lex": ("lex",), "brit": ("brit",), "rule": ("rule",), "alt": ("alt",)}
    for nm, ks in kinds.items():
        new = []
        raised_pos = raised_pos_over = 0
        for r in rows:
            s = r["score"]
            ksc = r.get("kind_scores") or {}
            if ks is None:
                cand = list(ksc.values())
            else:
                cand = [v for k, v in ksc.items() if k in ks]
            s2 = max([s] + cand)
            if r["label"] and s2 > s:
                raised_pos += 1
                raised_pos_over += (1 - s2) < thr <= (1 - s)
            new.append({**r, "err": 1 - s2, "score": s2})
        auc, fa = K.metrics(new)
        thr2 = K.threshold(new)
        caught = [r for r in rows if r["label"] and r["err"] >= thr]
        fa_rows = [r for r in rows if not r["label"] and r["err"] >= thr]

        def best(r):
            ksc = r.get("kind_scores") or {}
            c = list(ksc.values()) if ks is None else [v for k, v in ksc.items() if k in ks]
            return max([r["score"]] + c)
        res_fa = sum(1 for r in fa_rows if 1 - best(r) < thr) / max(1, len(fa_rows))
        res_caught = sum(1 for r in caught if 1 - best(r) < thr) / max(1, len(caught))
        res[nm] = {"auc": auc, "fa80": fa, "thr_score": 1 - thr2, "changed_raised": raised_pos,
                   "rescued_share_FA": res_fa, "rescued_share_caught_changed": res_caught,
                   "changed_crossing_old_thr": raised_pos_over,
                   "mean_changed_raise": float(np.mean([max([r["score"]] + list((r.get("kind_scores") or {}).values()) if ks is None
                                                            else [r["score"]] + [v for k, v in (r.get("kind_scores") or {}).items() if k in ks])
                                                        - r["score"] for r in rows if r["label"]]))}
    return res


def main(argv=None):
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    a = ap.parse_args(argv)
    for lang in a.lang.split(","):
        out = run(lang)
        rows = out.pop("_rows")
        K.save_json(f"taxonomy_{lang}.json", out)
        keep = ("item", "clip", "speaker", "source", "k", "tok", "label", "pos", "err", "score", "kind", "n_tok",
                "dur", "n_snd", "function", "short", "asr_mismatch", "asr_cer", "snr", "low_snr", "first", "last",
                "clip_on", "clip_off", "truncated", "noise", "neighbour", "jitter", "click", "boundary", "has_lex",
                "has_brit", "has_rule", "has_alt", "best_score", "best_kind", "driver", "low_snd", "low_kind", "ins_flag",
                "tail_margin", "_bucket_FA", "_rescued_FA", "_bucket_low06", "_rescued_low06", "kind_scores",
                "oov", "name", "number", "text_risk", "twin_fa", "start", "end", "punct", "label_artifact", "tok_artifact", "dist")
        p = K.ROOT / "tables" / f"rows_{lang}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False) + "\n")
        print(lang, "AUC", round(out["auc"], 4), out["auc_ci"], "FA80", round(100 * out["fa80"], 2),
              [round(100 * x, 2) for x in out["fa80_ci"]], "n FA", out["FA"]["n"], "n<0.6", out["low06"]["n"])
        for pop in ("FA", "low06"):
            print(" ", pop, {b: round(v["pct"], 1) for b, v in out[pop]["buckets"].items()})
            print("   overlap", {t: round(v, 1) for t, v in out[pop]["overlap"].items()})
        print("  base", {t: round(v, 1) for t, v in out["FA"]["base_rate"].items()})
        print("  best-of-N", out["best_of_n"])
        print("  neighbour", out["neighbour_paired"])


if __name__ == "__main__":
    main()
