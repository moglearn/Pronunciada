"""Controlled-error evaluation on native speech (FLEURS), shared by train_adapter (adapter
selection), native_stats (variants off/on) and analyze.

For every utterance: score the correct reference and one corrupted reference per error kind
(bench.corrupt, 2 errors per utterance, seed = hash(utt id, kind)). Output = one row per scored
slot with its labels, so any AUC can be computed later:

    utt, kind ("clean" for the correct reference), wrong (0/1), llr_sub, llr_sub_med
    (llr_sub minus the median llr_sub of that scored reference), llr_del, phone, cls, dropped

`dropped` (variants on only): the slot is a corrupted phone whose replacement lists the true
(audio) phone as a learned variant -> by definition not an error; excluded from AUCs.
"""
from __future__ import annotations

import pickle
from functools import lru_cache

import numpy as np

from bench.pipeline.common import (broad_class, build_ref, evidence_arrays, g2p_version, load_lp, near_silent,
                                   pipe_dir, records, stable_seed)

KINDS_AUC = ("phone_sub_near", "phone_sub_far", "word_swap")
KINDS_ALL = ("phone_sub_near", "phone_sub_far", "phone_del", "phone_ins", "word_swap")


def _patch_phone_distance_save():
    """bench.corrupt.PhoneDistance.save writes a fixed `phone_dist.tmp`, which races when several
    worker processes save at once (FileNotFoundError on rename). Within this pipeline's processes
    use a per-process temp name and merge with what is on disk (bench/corrupt.py is not edited)."""
    import json as _json
    import os as _os

    from bench import corrupt as C

    if getattr(C.PhoneDistance.save, "_pid_safe", False):
        return

    def save(self):
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged = {}
        try:
            merged = _json.loads(self.path.read_text())
        except (OSError, ValueError):
            pass
        merged.update(self.table)
        tmp = self.path.with_suffix(f".{_os.getpid()}.tmp")
        tmp.write_text(_json.dumps(merged, ensure_ascii=False, sort_keys=True))
        _os.replace(tmp, self.path)
        self._dirty = False

    save._pid_safe = True
    C.PhoneDistance.save = save


_patch_phone_distance_save()


@lru_cache(maxsize=None)
def inventory(lang: str):
    """Phone inventory + word pool from the FLEURS-train references (cached per g2p version)."""
    from bench.corrupt import Inventory

    p = pipe_dir("inventory") / f"{lang}-{g2p_version(lang)}.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    refs = [r for rec in records("fleurs", split="train", lang=lang)
            if (r := build_ref(lang, rec["text"])) is not None]
    inv = Inventory.from_references(refs, lang=lang)
    p.write_bytes(pickle.dumps(inv))
    return inv


# ----------------------------------------------------------------------------- adapters

_ADAPTERS: dict = {}


def get_adapter(spec):
    """spec: None (raw) | path to an adapter npz."""
    if spec is None:
        return None
    if spec not in _ADAPTERS:
        from pronunciada.adapter import Adapter

        _ADAPTERS[spec] = Adapter.from_file(spec)
    return _ADAPTERS[spec]


def adapted_lp(dataset, rec, adapter_spec):
    lp = load_lp(dataset, rec)
    if lp is None:
        return None
    a = get_adapter(adapter_spec)
    return lp if a is None else a(lp)


# ----------------------------------------------------------------------------- TER

def token_errors(ref_toks: list[str], hyp: list[str]) -> int:
    n, m = len(ref_toks), len(hyp)
    prev = np.arange(m + 1)
    for i in range(1, n + 1):
        cur = np.empty(m + 1, dtype=int)
        cur[0] = i
        r = ref_toks[i - 1]
        sub = prev[:-1] + np.array([r != h for h in hyp], dtype=int)
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, sub[j - 1])
        prev = cur
    return int(prev[m])


def greedy(lp) -> list[str]:
    from pronunciada.acoustic import greedy_tokens

    return [t for t, _, _ in greedy_tokens(lp)]


# ----------------------------------------------------------------------------- one utterance

def _aligned_pairs(a, b):
    """Min-edit alignment of phone sequences a (original) and b (donor): {j: a_i} for
    substituted / matched positions j of b."""
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), dtype=int)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    i, j, out = n, m, {}
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            out[j - 1] = a[i - 1]
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def _true_phone_of_wrong(cor, k_phone: int):
    """For a wrong slot of the corrupted ref, the phone the audio actually contains there (None
    for inserted phones)."""
    p = cor.ref.phones[k_phone]
    wi = p.word
    for e in cor.errors:
        if e.word != wi:
            continue
        if e.kind in ("phone_sub_near", "phone_sub_far"):
            return e.detail["from"]
        if e.kind == "word_swap":
            orig = e.detail["from"].split(" ")
            p0, p1 = cor.ref.words[wi].phones
            donor = [cor.ref.phones[i].ipa for i in range(p0, p1)]
            return _aligned_pairs(orig, donor).get(k_phone - p0)
    return None


def eval_utterance(args) -> dict:
    """args = (dataset, rec, adapter_spec, variants_lang | None, variants_on, kinds, n_errors,
    want_ter). With variants_lang set, corruptions are drawn from the variants-on reference in
    both modes (paired), and variants_on decides whether the scored references keep them."""
    dataset, rec, adapter_spec, var_lang, var_on, kinds, n_errors, want_ter = args
    from bench.corrupt import NotApplicable, corrupt

    lang = rec["lang"]
    base = build_ref(lang, rec["text"])
    if base is None:
        return {"id": rec["id"], "ok": False, "why": "g2p"}
    raw = load_lp(dataset, rec)
    if raw is None:
        return {"id": rec["id"], "ok": False, "why": "no posteriors"}
    if near_silent(raw, rec["dur"]):
        return {"id": rec["id"], "ok": False, "why": "near-silent audio"}
    a = get_adapter(adapter_spec)
    lp = raw if a is None else a(raw)
    if var_lang:
        from pronunciada.variants import apply_variants, learned_alts, strip_variants

        ref_v = apply_variants(base, var_lang)
        ref0 = ref_v if var_on else strip_variants(ref_v, var_lang)
    else:
        ref_v = base
        ref0 = base
    rows = []
    n_applied = {}
    n_dropped = {}

    def add(ref, kind, wrong, dropped):
        ev = evidence_arrays(lp, ref)
        s = ev["llr_sub"]
        med = float(np.median(s)) if len(s) else 0.0
        for i, p in enumerate(ref.phones):
            rows.append((rec["id"], kind, int(wrong[i]), float(s[i]), float(s[i] - med),
                         float(ev["llr_del"][i]), p.ipa, broad_class(p.ipa), int(dropped[i])))
        return ev

    n = len(ref0.phones)
    ev0 = add(ref0, "clean", [0] * n, [0] * n)
    inv = inventory(lang)
    for kind in kinds:
        seed = stable_seed(rec["id"], kind, "evalctl-v1")
        try:
            # the corruption is drawn from the variants-on reference so that its forbidden set
            # (original + alts) includes the learned alts; for "off" the learned alts are removed
            cor = corrupt(ref_v, [kind], inv, n_errors=n_errors, seed=seed, strict=False)
        except NotApplicable:
            continue
        if not cor.errors:
            continue
        n_applied[kind] = len(cor.errors)
        ref = cor.ref
        dropped = [0] * len(ref.phones)
        if var_lang:
            ref = strip_variants(ref, var_lang)
            if var_on:
                ref = apply_variants(ref, var_lang)
            for k, w in enumerate(cor.phone_wrong):
                if not w:
                    continue
                true = _true_phone_of_wrong(cor, k)
                if true is not None and true in {a for a, _ in learned_alts(var_lang, ref.phones[k].ipa)}:
                    dropped[k] = 1
            n_dropped[kind] = int(sum(dropped))
        add(ref, kind, cor.phone_wrong, dropped)
    out = {"id": rec["id"], "ok": True, "rows": rows, "applied": n_applied, "dropped": n_dropped,
           "n_frames": int(len(lp))}
    if want_ter:
        toks = [c for p in base.phones for c in p.ipa]
        out["ter"] = (token_errors(toks, greedy(lp)), len(toks))
    return out


ROW_COLS = ("utt", "kind", "wrong", "llr_sub", "llr_sub_med", "llr_del", "phone", "cls", "dropped")


def run(dataset, recs, adapter_spec=None, var_lang=None, var_on=False, kinds=KINDS_AUC,
        n_errors=2, want_ter=True, workers=6, desc=""):
    """Returns (pandas DataFrame of slot rows, summary dict)."""
    import pandas as pd

    from bench.pipeline.common import pmap

    res = pmap(eval_utterance, [(dataset, r, adapter_spec, var_lang, var_on, tuple(kinds), n_errors,
                                 want_ter)
                                for r in recs], workers=workers, chunksize=4, desc=desc)
    rows = [x for r in res if r["ok"] for x in r["rows"]]
    df = pd.DataFrame(rows, columns=ROW_COLS)
    summ = {"n_utts": len(recs), "n_ok": sum(r["ok"] for r in res),
            "failed": {r["id"]: r["why"] for r in res if not r["ok"]},
            "applied": {}, "dropped": {}}
    for r in res:
        if not r["ok"]:
            continue
        for k, v in r["applied"].items():
            summ["applied"][k] = summ["applied"].get(k, 0) + v
        for k, v in r["dropped"].items():
            summ["dropped"][k] = summ["dropped"].get(k, 0) + v
    if want_ter:
        e = sum(r["ter"][0] for r in res if r["ok"])
        t = sum(r["ter"][1] for r in res if r["ok"])
        summ["ter"] = e / max(t, 1)
        summ["ter_utt"] = {r["id"]: r["ter"] for r in res if r["ok"]}
    return df, summ


# ----------------------------------------------------------------------------- metrics

def auc_table(df, score="llr_sub", kinds=KINDS_AUC, by=None):
    """AUC of -score (lower llr = more likely wrong) for wrong slots of kind k vs untouched
    slots of the same corrupted references (column auc_<k>), and vs the slots of the correct
    references (auc_clean_<k>)."""
    from bench.metrics import roc_auc

    out = {}
    clean = df[df.kind == "clean"]
    for k in kinds:
        d = df[(df.kind == k) & (df.dropped == 0)]
        if d.empty or d.wrong.sum() == 0:
            continue
        out[f"auc_{k}"] = roc_auc(d.wrong.values, -d[score].values)
        pos = d[d.wrong == 1]
        y = np.r_[np.ones(len(pos)), np.zeros(len(clean))]
        s = np.r_[-pos[score].values, -clean[score].values]
        out[f"auc_clean_{k}"] = roc_auc(y, s)
    return out


def mean_auc_untouched(df, score="llr_sub", kinds=KINDS_AUC):
    t = auc_table(df, score, kinds)
    v = [t[f"auc_{k}"] for k in kinds if f"auc_{k}" in t]
    return float(np.mean(v)) if v else float("nan")


def utt_bootstrap(df_a, df_b, fn, n_boot=500, seed=0):
    """Paired cluster bootstrap over utterances of fn(df) for two systems scored on the same
    utterances/corruptions. Returns {a, b, delta, lo, hi, p (one-sided: a not better than b)}."""
    utts = np.array(sorted(set(df_a.utt) & set(df_b.utt)))
    df_a = df_a[df_a.utt.isin(utts)].reset_index(drop=True)
    df_b = df_b[df_b.utt.isin(utts)].reset_index(drop=True)
    ga, gb = df_a.groupby("utt").indices, df_b.groupby("utt").indices
    rng = np.random.default_rng(seed)
    a, b = fn(df_a), fn(df_b)
    d = []
    for _ in range(n_boot):
        pick = utts[rng.integers(0, len(utts), len(utts))]
        ia = np.concatenate([ga[u] for u in pick])
        ib = np.concatenate([gb[u] for u in pick])
        d.append(fn(df_a.iloc[ia]) - fn(df_b.iloc[ib]))
    d = np.array(d)
    d = d[np.isfinite(d)]
    return {"a": a, "b": b, "delta": a - b, "lo": float(np.quantile(d, 0.025)),
            "hi": float(np.quantile(d, 0.975)), "p": float((np.sum(d <= 0) + 1) / (len(d) + 1))}


def boot_ci(df, fn, n_boot=500, seed=0):
    df = df.reset_index(drop=True)
    g = df.groupby("utt").indices
    utts = np.array(sorted(g))
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        pick = utts[rng.integers(0, len(utts), len(utts))]
        v.append(fn(df.iloc[np.concatenate([g[u] for u in pick])]))
    v = np.array(v)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))
