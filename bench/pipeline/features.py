"""Part 4: feature tables for calibration, one row per reference phone.

    python -m bench.pipeline.features --lang en|all [--dataset speechocean762,l2arctic,epadb,fleurs,fleurs-validation]

Output: $PRONUNCIADA_CACHE/features/<dataset>.parquet (fleurs / fleurs-validation are assembled from
per-language parts in features/parts/), plus features/speechocean762_theirs.parquet: one row per
speechocean762 phone (THEIR canonical phones) with the matching row of ours (or -1) and our
phones of the aligned word(s) for the "mean of the word" fall-back.

Per-phone features: pronunciada.features.phone_rows (shared with the runtime scorer; this module only
adds identifiers and labels). feature_version 3 adds start_frame / end_frame.

Configuration: the shipped adapter of the language (pronunciada/data/adapters/<lang>.npz, if any) and
learned variants iff variants_enabled in pronunciada/data/native/<lang>.json; recorded per row
(columns adapter, variants). Native z-scores from pronunciada.variants.NativeStats.

FLEURS (test, validation): the correct reference (ref_kind "clean", label_ok 1) and one corrupted
reference per kind in phone_sub_near, phone_sub_far, phone_del, phone_ins, word_swap (bench.corrupt,
2 errors / utterance, seed = hash(utt id, kind)); label_ok = 0 for the flagged phones,
word_wrong / word_kinds from bench.corrupt, near_gap = phone adjacent to a phone_del gap,
dropped = corrupted phone whose replacement lists the true phone as a learned variant.
"""
from __future__ import annotations

import hashlib
import json
import math
import time


from bench.pipeline.common import (LANGS, build_ref, evidence_arrays, feat_dir, g2p_version, load_lp, log,
                                   parse_langs, pmap, records, stable_seed)
from bench.pipeline.labels import align_reference, arpa_to_ours
from pronunciada.features import FEATURE_VERSION, phone_rows  # the ONE per-phone feature implementation

FLEURS_KINDS = ("phone_sub_near", "phone_sub_far", "phone_del", "phone_ins", "word_swap")
EN_DATASETS = ("speechocean762", "l2arctic", "epadb")


def config(lang: str) -> dict:
    from pronunciada.adapter import Adapter
    from pronunciada.variants import NativeStats, table_path

    p = Adapter.path(lang)
    ad = "raw" if not p.exists() else "ad" + hashlib.sha256(p.read_bytes()).hexdigest()[:8]
    ns = NativeStats.load(lang)
    var = bool(ns and ns.variants_enabled)
    tp = table_path(lang)
    nat = hashlib.sha256(tp.read_bytes()).hexdigest()[:8] if tp.exists() else "none"
    return {"adapter": ad, "variants": var, "native": nat, "g2p": g2p_version(lang),
            "feature_version": FEATURE_VERSION}


def _rows_for(rec, dataset, ref, ev, lang, ns, cfg, extra_per_phone, extra_utt):
    """Feature-table rows: dataset / utterance identifiers + pronunciada.features.phone_rows (the
    runtime feature code) + label columns."""
    utt = {"dataset": dataset, "split": rec["split"], "lang": lang, "utt": rec["id"],
           "speaker": rec.get("speaker"), "adapter": cfg["adapter"], "variants": cfg["variants"],
           **extra_utt}
    rows = []
    for i, r in enumerate(phone_rows(ev["_lp"], ref, ev, ns, float(rec["dur"]))):
        d = dict(utt)
        d.update(r)
        d.update(extra_per_phone[i] if extra_per_phone else {})
        rows.append(d)
    return rows


def _prep(rec, lang):
    from pronunciada.adapter import apply
    from pronunciada.variants import NativeStats, apply_variants

    ns = NativeStats.load(lang)
    var = bool(ns and ns.variants_enabled)
    base = build_ref(lang, rec["text"])
    if base is None:
        return None
    lp = load_lp(rec["_dataset"], rec)
    if lp is None:
        return None
    lp = apply(lp, lang)
    ref = apply_variants(base, lang) if var else base
    return base, ref, lp, ns, var


# ----------------------------------------------------------------------------- FLEURS

def fleurs_utt(args):
    rec, lang, cfg = args
    from bench.corrupt import NotApplicable, corrupt
    from bench.pipeline.evalctl import _true_phone_of_wrong, inventory
    from pronunciada.variants import apply_variants, learned_alts, strip_variants

    prep = _prep(rec, lang)
    if prep is None:
        return []
    base, ref, lp, ns, var = prep
    ds = rec["_dataset"]
    n = len(ref.phones)
    rows = _rows_for(rec, ds, ref, evidence_arrays(lp, ref), lang, ns, cfg,
                     [{"label_ok": 1, "word_wrong": False, "err_kinds": "", "near_gap": False,
                       "dropped": False} for _ in range(n)],
                     {"ref_kind": "clean", "n_errors": 0})
    inv = inventory(lang)
    for kind in FLEURS_KINDS:
        try:
            cor = corrupt(ref, [kind], inv, n_errors=2, seed=stable_seed(rec["id"], kind, "features-v1"),
                          strict=False)
        except NotApplicable:
            continue
        if not cor.errors:
            continue
        r2 = cor.ref
        if var:
            r2 = apply_variants(strip_variants(r2, lang), lang)
        near = set()
        for e in cor.errors:
            if e.kind == "phone_del":
                s0 = r2.words[e.word].syllables[0] + e.detail["syllable_local"]
                a, b = r2.syllables[s0].phones
                g = a + e.detail["gap_in_syllable"]
                near |= {k for k in (g - 1, g) if a <= k < b}
        per = []
        for k, p in enumerate(r2.phones):
            dropped = False
            if var and cor.phone_wrong[k]:
                t = _true_phone_of_wrong(cor, k)
                dropped = t is not None and t in {x for x, _ in learned_alts(lang, p.ipa)}
            per.append({"label_ok": int(not cor.phone_wrong[k]), "word_wrong": bool(cor.word_wrong[p.word]),
                        "err_kinds": ",".join(cor.word_kinds[p.word]), "near_gap": k in near,
                        "dropped": dropped})
        rows += _rows_for(rec, ds, r2, evidence_arrays(lp, r2), lang, ns, cfg, per,
                          {"ref_kind": kind, "n_errors": len(cor.errors)})
    return rows


# ----------------------------------------------------------------------------- English learners

def so762_utt(args):
    rec, lang, cfg = args
    prep = _prep(rec, lang)
    if prep is None:
        return [], [], None
    base, ref, lp, ns, var = prep
    L = rec["labels"]
    W = L["words"]
    tp = [[arpa_to_ours(p) for p in w["phones"]] for w in W]
    o2t, wpairs = align_reference(base, [w["text"] for w in W], tp)
    S = L["sentence"]
    sent = {f"sent_{k}": float(S[k]) for k in ("accuracy", "fluency", "prosodic", "completeness", "total")}
    our_word_to_their = {i: j for i, j in wpairs}
    for i, (j, _) in o2t.items():
        our_word_to_their.setdefault(base.phones[i].word, j)
    per = []
    for i, p in enumerate(ref.phones):
        j = None
        if i in o2t:
            j, k = o2t[i]
            d = {"their_word_index": j, "their_phone_index": k, "their_phone": W[j]["phones"][k],
                 "phone_score": float(W[j]["phones_accuracy"][k])}
        else:
            j = our_word_to_their.get(p.word)
            d = {"their_word_index": -1 if j is None else j, "their_phone_index": -1,
                 "their_phone": None, "phone_score": math.nan}
        if j is not None:
            d.update({"word_accuracy": float(W[j]["accuracy"]), "word_stress": float(W[j]["stress"]),
                      "word_total": float(W[j]["total"])})
        else:
            d.update({"word_accuracy": math.nan, "word_stress": math.nan, "word_total": math.nan})
        per.append(d)
    rows = _rows_for(rec, "speechocean762", ref, evidence_arrays(lp, ref), lang, ns, cfg, per,
                     {"ref_kind": "prompt", **sent})
    t2o = {v: i for i, v in o2t.items()}
    their_word_ours = {}
    for i, (j, _) in o2t.items():
        their_word_ours.setdefault(j, set()).add(base.phones[i].word)
    for i, j in wpairs:
        their_word_ours.setdefault(j, set()).add(i)
    theirs = []
    for j, w in enumerate(W):
        ow = sorted(their_word_ours.get(j, set()))
        our_ph = [k for x in ow for k in range(*ref.words[x].phones)]
        for k, ph in enumerate(w["phones"]):
            theirs.append({"utt": rec["id"], "split": rec["split"], "speaker": rec["speaker"],
                           "their_word_index": j, "their_phone_index": k, "their_phone": ph,
                           "phone_score": float(w["phones_accuracy"][k]),
                           "our_phone_index": t2o.get((j, k), -1), "fallback_phone_indices": our_ph})
    cov = {"theirs": sum(len(w["phones"]) for w in W), "theirs_matched": len(o2t),
           "ours": len(base.phones), "ours_matched": len(o2t),
           "identical": sum(base.phones[i].ipa == tp[j][k] for i, (j, k) in o2t.items())}
    return rows, theirs, cov


def _err_utt(rec, lang, cfg, their_words, their_entries):
    """their_entries: per their word, list of dicts {ph (ours|None), label ("correct"/"sub"/"del"/
    "unk"), ins (bool), raw, perceived}."""
    prep = _prep(rec, lang)
    if prep is None:
        return [], None
    base, ref, lp, ns, var = prep
    tp = [[e["ph"] for e in w] for w in their_entries]
    o2t, _ = align_reference(base, their_words, tp)
    per = []
    for i in range(len(ref.phones)):
        if i in o2t:
            j, k = o2t[i]
            e = their_entries[j][k]
            lab = e["label"]
            per.append({"their_phone": e["raw"], "perceived": e["perceived"], "err_label": lab,
                        "label_err": {"correct": 0.0, "sub": 1.0, "del": 1.0}.get(lab, math.nan),
                        "ins_flag": bool(e["ins"])})
        else:
            per.append({"their_phone": None, "perceived": None, "err_label": "unaligned",
                        "label_err": math.nan, "ins_flag": False})
    rows = _rows_for(rec, rec["_dataset"], ref, evidence_arrays(lp, ref), lang, ns, cfg, per,
                     {"ref_kind": "prompt"})
    n_their = sum(1 for w in tp for p in w)
    cov = {"theirs": n_their, "theirs_mappable": sum(1 for w in tp for p in w if p is not None),
           "theirs_matched": len(o2t), "ours": len(base.phones), "ours_matched": len(o2t),
           "identical": sum(base.phones[i].ipa == tp[j][k] for i, (j, k) in o2t.items())}
    return rows, cov


def l2arctic_utt(args):
    rec, lang, cfg = args
    L = rec["labels"]
    words = [w["text"] for w in L["words"]]
    # bench/data/l2arctic.py keeps only the LAST ARPAbet-looking tier as `phones`; for ~12 % of
    # the files that is a small secondary tier, the full phone tier is lost -> exclude them.
    base = build_ref(lang, rec["text"])
    n_can = sum(1 for p in L["phones"] if p["error"] != "a")
    if base is None or n_can < 0.6 * len(base.phones):
        return [], {"excluded_utts": 1, "excluded_their_phones": n_can}
    ents = []
    last_w = 0
    for p in L["phones"]:
        w = p["word"]
        if w is None:
            w = last_w
        last_w = w
        ents.append((w, p))
    per = [[] for _ in words]
    pending_ins = False
    flat = []
    for w, p in ents:
        if p["error"] == "a":
            if flat:
                flat[-1]["ins"] = True
            else:
                pending_ins = True
            continue
        lab = {None: "correct", "s": "sub", "d": "del"}.get(p["error"], "unk")
        e = {"ph": arpa_to_ours(p["canonical"]), "label": lab, "ins": pending_ins,
             "raw": p["canonical"], "perceived": p["perceived"]}
        pending_ins = False
        flat.append(e)
        if w < len(per):
            per[w].append(e)
    return _err_utt(rec, lang, cfg, words, per)


def epadb_utt(args):
    rec, lang, cfg = args
    flat = []
    pending = False
    for p in rec["labels"]["phones"]:
        if p["ref"] == "0":                     # annotator heard a sound with no reference phone
            if flat:
                flat[-1]["ins"] = True
            else:
                pending = True
            continue
        if p["correct"]:
            lab = "correct"
        elif p["annotated"] == "0":
            lab = "del"
        else:
            lab = "sub"
        flat.append({"ph": arpa_to_ours(p["ref"]), "label": lab, "ins": pending, "raw": p["ref"],
                     "perceived": p["annotated"]})
        pending = False
    return _err_utt(rec, lang, cfg, None, [flat])


# ----------------------------------------------------------------------------- driver

def _write(df, path):
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _sum_cov(covs):
    tot = {}
    for c in covs:
        if c is None:
            continue
        for k, v in c.items():
            tot[k] = tot.get(k, 0) + v
    out = dict(tot)
    if tot.get("theirs"):
        out["pct_theirs_matched"] = 100 * tot["theirs_matched"] / tot["theirs"]
    if tot.get("ours"):
        out["pct_ours_matched"] = 100 * tot["ours_matched"] / tot["ours"]
    if tot.get("theirs_matched"):
        out["pct_identical_of_matched"] = 100 * tot["identical"] / tot["theirs_matched"]
    return out


def build(dataset: str, langs: list[str], workers: int, force: bool):
    import pandas as pd

    from bench.pipeline.common import repo_root

    fd = feat_dir()
    (fd / "parts").mkdir(exist_ok=True)
    cov_path = repo_root() / "results" / "tables" / "alignment_coverage.json"
    covs_all = json.loads(cov_path.read_text()) if cov_path.exists() else {}
    if dataset in EN_DATASETS:
        if "en" not in langs:
            return
        cfg = config("en")
        stamp = fd / f"{dataset}.cfg.json"
        out = fd / f"{dataset}.parquet"
        if out.exists() and stamp.exists() and json.loads(stamp.read_text()) == cfg and not force:
            log(f"{dataset}: up to date")
            return
        recs = records(dataset)
        for r in recs:
            r["_dataset"] = dataset
        fn = {"speechocean762": so762_utt, "l2arctic": l2arctic_utt, "epadb": epadb_utt}[dataset]
        t0 = time.time()
        res = pmap(fn, [(r, "en", cfg) for r in recs], workers=workers, chunksize=8, desc=dataset)
        rows = [x for r in res for x in r[0]]
        df = pd.DataFrame(rows)
        _write(df, out)
        if dataset == "speechocean762":
            th = pd.DataFrame([x for r in res for x in r[1]])
            _write(th, fd / "speechocean762_theirs.parquet")
            covs = {s: _sum_cov([r[2] for r, rec in zip(res, recs) if rec["split"] == s])
                    for s in ("train", "test")}
        else:
            covs = {s: _sum_cov([r[1] for r, rec in zip(res, recs) if rec["split"] == s])
                    for s in ("train", "test")}
        covs_all[dataset] = covs
        cov_path.write_text(json.dumps(covs_all, indent=1))
        stamp.write_text(json.dumps(cfg))
        log(f"{dataset}: {len(df)} rows, {df.utt.nunique()} utts, coverage {json.dumps(covs)} "
            f"({time.time() - t0:.0f}s)")
        return
    # FLEURS (test or validation), per language parts
    split = "validation" if dataset == "fleurs-validation" else "test"
    for lang in langs:
        cfg = config(lang)
        part = fd / "parts" / f"{dataset}-{lang}.parquet"
        stamp = fd / "parts" / f"{dataset}-{lang}.cfg.json"
        if part.exists() and stamp.exists() and json.loads(stamp.read_text()) == cfg and not force:
            log(f"{dataset}/{lang}: up to date")
            continue
        recs = records(dataset, split=None if dataset == "fleurs-validation" else split, lang=lang)
        for r in recs:
            r["_dataset"] = dataset
        t0 = time.time()
        res = pmap(fleurs_utt, [(r, lang, cfg) for r in recs], workers=workers, chunksize=4,
                   desc=f"{dataset}/{lang}")
        df = pd.DataFrame([x for r in res for x in r])
        _write(df, part)
        stamp.write_text(json.dumps(cfg))
        log(f"{dataset}/{lang}: {len(df)} rows, {df.utt.nunique()} utts ({time.time() - t0:.0f}s)")
    parts = [fd / "parts" / f"{dataset}-{l}.parquet" for l in LANGS]
    parts = [p for p in parts if p.exists()]
    if parts:
        _write(pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True), fd / f"{dataset}.parquet")


def main():
    from bench.pipeline.common import base_parser

    ap = base_parser(__doc__)
    ap.add_argument("--dataset", default="speechocean762,l2arctic,epadb,fleurs,fleurs-validation")
    a = ap.parse_args()
    langs = parse_langs(a.lang)
    for d in a.dataset.split(","):
        build(d, langs, a.workers, a.force)


if __name__ == "__main__":
    main()
