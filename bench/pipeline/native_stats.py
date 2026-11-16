"""Part 3: native statistics and learned variants.

    python -m bench.pipeline.native_stats --lang en|all [--force]

1. Score native FLEURS-train speech (shipped adapter of the language applied, if any) against
   the references (rule-based alts only) -> one row per slot.
2. Learned variants: per canonical phone, slots in the worst 10 % of llr_sub for that phone;
   tabulate `heard` ("" when llr_del < 0). A realisation with count >= 20 and >= 3 % of ALL
   native tokens of the phone, valid ZIPA string and != the phone becomes an alternative with
   log-prior log(rate).
3. Variants off vs on on FLEURS validation (controlled errors, evalctl; corruptions drawn from
   the variants-on reference so both modes are paired; slots whose true phone is a learned alt of
   the corrupted target are dropped and counted). Recommendation: on iff the mean controlled-
   error AUC (corrupted vs untouched, three kinds) does not drop significantly (paired utterance
   bootstrap 95 % CI of on - off has hi >= 0) -- the native 5th percentile is reported alongside.
   Decision after en/zh: variants are shipped OFF for every language; step 3 only runs with
   --eval-variants (it was run for en, zh, es).
4. Statistics (count/mean/std/quantiles of llr_sub, llr_del, n_frames, peak_post per phone,
   phone+position, broad class, global) are computed in the configuration that will be used
   (variants on iff recommended) and saved with the variants to pronunciada/data/native/<lang>.json.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter

import numpy as np

from bench.pipeline.common import (build_ref, evidence_arrays, g2p_version, load_lp, log, near_silent,
                                   parse_langs, pipe_dir, pmap, records, repo_root)
from pronunciada.variants import FEATURES, QUANTILES, broad_class, phone_position

MIN_RATE, MIN_COUNT, WORST_Q = 0.03, 20, 0.10


def adapter_tag(lang: str) -> str:
    import hashlib

    from pronunciada.adapter import Adapter

    p = Adapter.path(lang)
    return "raw" if not p.exists() else "ad" + hashlib.sha256(p.read_bytes()).hexdigest()[:8]


def _score_train(args):
    dataset, rec, lang, var_on = args
    from pronunciada.adapter import apply

    ref = build_ref(lang, rec["text"])
    if ref is None:
        return []
    if var_on:
        from pronunciada.variants import apply_variants

        ref = apply_variants(ref, lang)
    lp = load_lp(dataset, rec)
    if lp is None or near_silent(lp, rec["dur"]):
        return []
    ev = evidence_arrays(apply(lp, lang), ref)
    out = []
    for i, p in enumerate(ref.phones):
        out.append((rec["id"], i, p.ipa, phone_position(ref, i), broad_class(p.ipa),
                    float(ev["llr_sub"][i]), float(ev["llr_del"][i]), int(ev["n_frames"][i]),
                    float(ev["peak_post"][i]), ev["heard"][i], int(ev["alt_index"][i])))
    return out


COLS = ("utt", "i", "phone", "position", "cls", "llr_sub", "llr_del", "n_frames", "peak_post",
        "heard", "alt_index")


def native_rows(lang, var_on, workers, force=False):
    import pandas as pd

    tag = f"{g2p_version(lang)}-{adapter_tag(lang)}-{'var' if var_on else 'novar'}"
    if var_on:
        from pronunciada.variants import table_path

        import hashlib

        v = json.dumps(json.loads(table_path(lang).read_text())["variants"], sort_keys=True)
        tag += "-" + hashlib.sha256(v.encode()).hexdigest()[:8]
    p = pipe_dir("native", lang) / f"train-{tag}.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)
    recs = records("fleurs", split="train", lang=lang)
    res = pmap(_score_train, [("fleurs", r, lang, var_on) for r in recs], workers=workers,
               desc=f"{lang} native train")
    df = pd.DataFrame([x for r in res for x in r], columns=COLS)
    df.to_parquet(p)
    return df


def _summ(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=float)
    q = np.percentile(v, QUANTILES) if len(v) else [math.nan] * len(QUANTILES)
    return {"count": int(len(v)), "mean": float(v.mean()) if len(v) else math.nan,
            "std": float(v.std()) if len(v) else math.nan,
            **{f"q{k:02d}": float(x) for k, x in zip(QUANTILES, q)}}


def stats_from_rows(df) -> dict:
    out = {"phone": {}, "phone_pos": {}, "class": {}, "global": {}}
    for f in FEATURES:
        out["global"][f] = _summ(df[f].values)
    for ph, g in df.groupby("phone"):
        out["phone"][ph] = {f: _summ(g[f].values) for f in FEATURES}
        out["phone_pos"][ph] = {pos: {f: _summ(h[f].values) for f in FEATURES}
                                for pos, h in g.groupby("position")}
    for c, g in df.groupby("cls"):
        out["class"][c] = {f: _summ(g[f].values) for f in FEATURES}
    return out


def learn_variants(df) -> tuple[dict, dict]:
    from pronunciada.text.ipa import is_valid

    var, diag = {}, {}
    for ph, g in df.groupby("phone"):
        n = len(g)
        thr = np.quantile(g.llr_sub.values, WORST_Q)
        w = g[g.llr_sub <= thr]
        heard = np.where(w.llr_del.values < 0, "", w.heard.values)
        cnt = Counter(heard.tolist())
        diag[ph] = {"n": n, "thr": float(thr), "worst": int(len(w)),
                    "top_heard": [[h, c, c / n] for h, c in cnt.most_common(5)]}
        alts = []
        for h, c in cnt.most_common():
            rate = c / n
            if c < MIN_COUNT or rate < MIN_RATE:
                continue
            if h == ph or (h and not is_valid(h)):
                continue
            alts.append([h, float(math.log(rate)), int(c), float(rate)])
        if alts:
            var[ph] = alts
    return var, diag


def compare_variants(lang, workers):
    from bench.pipeline import evalctl
    from pronunciada.adapter import Adapter

    spec = str(Adapter.path(lang)) if Adapter.path(lang).exists() else None
    recs = records("fleurs-validation", lang=lang)
    out, dfs = {}, {}
    for on in (False, True):
        df, s = evalctl.run("fleurs-validation", recs, spec, var_lang=lang, var_on=on,
                            want_ter=False, workers=workers, desc=f"{lang} variants {on}")
        dfs[on] = df
        t = evalctl.auc_table(df)
        t.update({f"med_{a}": b for a, b in evalctl.auc_table(df, "llr_sub_med").items()})
        clean = df[df.kind == "clean"]
        t["mean_auc"] = evalctl.mean_auc_untouched(df)
        t["mean_auc_med"] = evalctl.mean_auc_untouched(df, "llr_sub_med")
        t["native_llr_sub_p05"] = float(np.quantile(clean.llr_sub, 0.05))
        t["native_llr_sub_median"] = float(np.median(clean.llr_sub))
        t["applied"] = s["applied"]
        t["dropped"] = s["dropped"]
        t["n_wrong"] = {k: int(df[(df.kind == k)].wrong.sum()) for k in evalctl.KINDS_AUC}
        out["on" if on else "off"] = t
    b = evalctl.utt_bootstrap(dfs[True], dfs[False], evalctl.mean_auc_untouched, n_boot=500)
    out["on_minus_off"] = b
    out["recommend_on"] = bool(b["hi"] >= 0.0)
    return out, dfs


def run_lang(lang, workers=6, force=False, eval_variants=False):
    from pronunciada.variants import reload, table_path

    t0 = time.time()
    dst = table_path(lang)
    out = repo_root() / "results" / "tables" / f"native_{lang}.json"
    if out.exists() and dst.exists() and not force:
        old = json.loads(out.read_text())
        if old["meta"].get("g2p_version") == g2p_version(lang) and old["meta"].get("adapter") == adapter_tag(lang):
            log(f"[{lang}] native stats up to date")
            return old
    dst.parent.mkdir(parents=True, exist_ok=True)
    # pass 1: variants off (rule alts only)
    df0 = native_rows(lang, False, workers, force)
    var, diag = learn_variants(df0)
    meta = {"g2p_version": g2p_version(lang), "adapter": adapter_tag(lang),
            "source": "FLEURS train (native)", "n_utts": int(df0.utt.nunique()),
            "n_slots": int(len(df0)), "min_rate": MIN_RATE, "min_count": MIN_COUNT,
            "worst_q": WORST_Q}
    table = {"lang": lang, "meta": meta, "variants": var, "variants_enabled": False,
             "stats": stats_from_rows(df0), "stats_config": {"variants": False}}
    dst.write_text(json.dumps(table, ensure_ascii=False, indent=0))
    reload()
    log(f"[{lang}] {len(var)} phones with learned variants: "
        + ", ".join(f"{p}->{'/'.join(a[0] or '∅' for a in v)}" for p, v in var.items()))
    # Decision (architect, after en/zh): learned variants OFF everywhere. The off/on evaluation
    # is kept behind --eval-variants; its recommendation is reported but not applied.
    cmp = compare_variants(lang, workers)[0] if eval_variants else None
    on = False
    table["variants_enabled"] = on
    if on:
        dst.write_text(json.dumps(table, ensure_ascii=False, indent=0))
        reload()
        df1 = native_rows(lang, True, workers, force)
        table["stats"] = stats_from_rows(df1)
        table["stats_config"] = {"variants": True}
    dst.write_text(json.dumps(table, ensure_ascii=False, indent=0))
    reload()
    # worst phones (variants off, i.e. front-end + rule alts) for front-end fixes
    g = df0.groupby("phone")
    worst = []
    for ph, x in g:
        if len(x) < 30:
            continue
        h = np.where(x.llr_del.values < 0, "∅", x.heard.values)
        worst.append({"phone": ph, "n": int(len(x)), "mean": float(x.llr_sub.mean()),
                      "median": float(x.llr_sub.median()), "p05": float(x.llr_sub.quantile(0.05)),
                      "share_del": float((x.llr_del < 0).mean()),
                      "top_heard": [[a, int(c)] for a, c in Counter(h.tolist()).most_common(3)]})
    worst = sorted(worst, key=lambda r: r["mean"])[:20]
    res = {"lang": lang, "meta": meta, "variants": var, "variant_diag": diag, "compare": cmp,
           "variants_enabled": on, "worst_phones": worst, "wall_s": round(time.time() - t0, 1)}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    if cmp:
        log(f"[{lang}] variants eval: AUC off {cmp['off']['mean_auc']:.4f} on {cmp['on']['mean_auc']:.4f}; "
            f"p05 off {cmp['off']['native_llr_sub_p05']:.3f} on {cmp['on']['native_llr_sub_p05']:.3f} -> shipped off")
    else:
        log(f"[{lang}] native stats written (variants off, not evaluated)")
    return res


def main():
    from bench.pipeline.common import base_parser

    ap = base_parser(__doc__)
    ap.add_argument("--eval-variants", action="store_true", help="also run the variants off/on evaluation")
    a = ap.parse_args()
    for lang in parse_langs(a.lang):
        run_lang(lang, a.workers, a.force, a.eval_variants)


if __name__ == "__main__":
    main()
