"""Round 19 (Korean): select the identity word factor on the dev set (offline, exactly paired).

    python -m bench.round19ko.select --head PATH [--write]

The factor is a deterministic function of (word probability, the word's lowest identity evidence s,
strictness), so every candidate is evaluated on the rows collected with the factor off
(bench.round19ko.collect), like round 18.

Sets (none held out):
  dev           planted Lingolix-style edits (plant.py) on FLEURS ko validation + Common Voice dev
                pool clips that pass the Whisper read-the-prompt check (asr_check.py, CER <= 0.1)
  fleurs-val-cor  FLEURS ko validation with the controlled errors of bench.eval_languages (guard)
Guards (fixed before looking at candidates; the rules of RULES.md and round 18):
  * native word acceptance (P >= 0.5) on the FLEURS validation controls drops by <= 0.3 pt;
  * planted word-error AUC on fleurs-val-cor does not drop (tolerance 1e-4);
  * dev AUC on the FLEURS-validation part does not drop.
Choice: the guarded setting with the highest pooled dev AUC (ties: lower FA@80).
Per-class offsets: s minus the class's q_c native quantile (dev controls), q_c in {none, 0.05}.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.round19ko import collect as CO
from bench.round19ko import devmetrics as DM
from bench.round19ko import offline as OF
from bench.round19ko.cv_pool import work
from pronunciada import ko_identity as KI

QS = (0.01, 0.02, 0.05, 0.1, 0.2)
FORMS = [("mult", 0.3), ("mult", 0.5), ("mult", 0.6), ("mult", 0.7), ("logit", 0.25), ("logit", 0.5),
         ("logit", 1.0)]
WIDTHS = (4.0, 8.0)
RIVALS_ON = False


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply(p, z, knee, form, par, width, sounds=None):
    """Word probabilities after the factor (the runtime's ko_identity.word_factor, vectorised);
    the logit form lowers the sounds-based probability, as the runtime does."""
    p = np.asarray(p, np.float64)
    t = np.where(np.isfinite(z), np.clip((knee - np.nan_to_num(z, nan=1e9)) / width, 0, 1), 0.0)
    if form == "mult":
        return p * (1 - (1 - par) * t)
    sp = np.clip(np.asarray(sounds if sounds is not None else p, np.float64), 1e-6, 1 - 1e-6)
    return p * (1 / (1 + np.exp(-(lg(sp) - par * width * t)))) / sp


def load_dev(asr=True):
    R = CO.load("dev", "base")
    p = work() / "asr.jsonl"
    if asr:
        if not p.exists():
            raise SystemExit("asr.jsonl missing: run bench.round19ko.asr_check first")
        R = OF.keep_items(R, p)
    else:   # exploration only (never for the frozen choice): drop the 5 CV speakers that read other text
        bad = {"cvk_006", "cvk_018", "cvk_017", "cvk_010", "cvk_002"}
        R = {k: r for k, r in R.items() if r["meta"]["speaker"] not in bad}
    return R


def evidence_sets(rows, H, ref_cache):
    """per item: per-sound s (no offsets) and classes."""
    from pronunciada.text import get_g2p

    g = get_g2p("ko")
    out = {}
    for k, r in rows.items():
        if "error" in r:
            continue
        t = r["meta"]["target"]
        if r["meta"].get("cor_kind") not in (None, "clean"):
            ref = None
        else:
            ref = ref_cache.get(t) or g(t)
            ref_cache[t] = ref
        if ref is None or [p.ipa for p in ref.phones] != list(r["phones"]["ipa"]):
            ref = _ref_from_row(r, g)
        if ref is None:
            out[k] = (np.full(len(r["phones"]["ipa"]), np.nan), [None] * len(r["phones"]["ipa"]))
            continue
        s = KI.evidence(ref, r["phones"]["V"].astype(np.float32), H, offsets={}, rivals=RIVALS_ON)
        out[k] = (s, KI.classes(ref))
    return out


def _ref_from_row(r, g):
    """Corrupted references: rebuild with bench.corrupt from the stored meta (same seed)."""
    m = r["meta"]
    if m.get("cor_kind") in (None, "clean"):
        return None
    from bench.corrupt import corrupt
    from bench.pipeline.common import stable_seed
    from bench.pipeline.evalctl import inventory

    cor = corrupt(g(m["target"]), [m["cor_kind"]], inventory("ko"), n_errors=2,
                  seed=stable_seed(m["id"], m["cor_kind"], "features-v1"), strict=False)
    ref = cor.ref
    return ref if [p.ipa for p in ref.phones] == list(r["phones"]["ipa"]) else None


def word_s(rows, ev, offsets):
    out = {}
    for k, (s, cls) in ev.items():
        s2 = np.array([x - offsets.get(c, 0.0) if c is not None else np.nan for x, c in zip(s, cls)])
        out[k] = OF.word_min(rows[k], s2)
    return out


def class_offsets(rows, ev, qc):
    if qc is None:
        return {}
    by = defaultdict(list)
    for k, (s, cls) in ev.items():
        if rows[k]["meta"].get("label"):
            continue
        for x, c in zip(s, cls):
            if c is not None and np.isfinite(x):
                by[c].append(x)
    return {c: float(np.quantile(v, qc)) for c, v in by.items() if len(v) >= 50}


def planted_auc(cor_rows, P):
    pos, neg = [], []
    for k, r in cor_rows.items():
        if "error" in r:
            continue
        if r["meta"]["cor_kind"] == "clean":
            neg += list(P[k])
        else:
            ww = r.get("word_wrong") or []
            pos += [p for p, w in zip(P[k], ww) if w]
    from bench.third_party.lingolix import roc_auc

    return roc_auc([1] * len(pos) + [0] * len(neg), [1 - x for x in pos + neg])


def native_accept(rows, P):
    v = np.concatenate([P[k] for k, r in rows.items() if r["meta"]["src"] == "fleurs-val"
                        and not r["meta"].get("label") and "error" not in r])
    return float(np.mean(v >= 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--rivals", action="store_true", help="minimal-pair rivals only (counterfactual twins)")
    a = ap.parse_args()
    global RIVALS_ON
    RIVALS_ON = a.rivals
    z = np.load(a.head)
    H = {k: (z[k].astype(np.float32) if z[k].dtype.kind == "f" else z[k]) for k in z.files}
    H["index"] = {str(c): j for j, c in enumerate(H["classes"].tolist())}
    R = load_dev(not a.no_asr)
    COR = CO.load("fleurs-val-cor", "base")
    cache = {}
    evR = evidence_sets(R, H, cache)
    evC = evidence_sets(COR, H, cache)
    base_P = {k: r["words"]["prob"] for k, r in R.items() if "error" not in r}
    base_C = {k: r["words"]["prob"] for k, r in COR.items() if "error" not in r}
    m0 = DM.metrics(R)
    b_acc, b_pl = native_accept(R, base_P), planted_auc(COR, base_C)
    print("base", f"nat {b_acc:.4f} planted {b_pl:.4f}", DM.line(m0), flush=True)
    rows = []
    for qc in (None, 0.05):
        off = class_offsets(R, evR, qc)
        SR, SC = word_s(R, evR, off), word_s(COR, evC, off)
        nat = np.concatenate([SR[k] for k, r in R.items() if not r["meta"].get("label")])
        nat = nat[np.isfinite(nat)]
        for q, (form, par), w in itertools.product(QS, FORMS, WIDTHS):
            knee = float(np.quantile(nat, q))
            P = {k: apply(base_P[k], SR[k], knee, form, par, w, R[k]["words"]["sounds"]) for k in base_P}
            PC = {k: apply(base_C[k], SC[k], knee, form, par, w, COR[k]["words"]["sounds"]) for k in base_C}
            m = DM.metrics(R, P)
            acc, pl = native_accept(R, P), planted_auc(COR, PC)
            ok = (acc >= b_acc - 0.003 and pl >= b_pl - 1e-4 and m["fleurs-val"]["auc"] >= m0["fleurs-val"]["auc"] - 1e-4)
            rows.append({"qc": qc, "q": q, "knee": knee, "form": form, "par": par, "width": w, "ok": bool(ok),
                         "native_accept": acc, "planted_auc": pl, "auc": m["all"]["auc"], "fa80": m["all"]["fa80"],
                         "loc": m["all"]["loc"], "bucket": m["all"]["bucket"], "cv_auc": m.get("cv", {}).get("auc"),
                         "cv_fa80": m.get("cv", {}).get("fa80"), "fv_auc": m["fleurs-val"]["auc"],
                         "fv_fa80": m["fleurs-val"]["fa80"], "offsets": off})
    oks = [r for r in rows if r["ok"]]
    best = max(oks, key=lambda r: (round(r["auc"], 4), -r["fa80"])) if oks else None
    top = sorted(rows, key=lambda r: -r["auc"])[:5]
    for r in top:
        print("top", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("offsets", "bucket")})
    print("best", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in (best or {}).items() if k != "offsets"})
    out = {"head": Path(a.head).name, "base": {"native_accept": b_acc, "planted_auc": b_pl, "dev": m0},
           "n_items": len(R), "grid": [{k: v for k, v in r.items() if k != "offsets"} for r in rows], "best": best}
    Path(f"results/tables/round19_ko_select{a.tag}.json").write_text(json.dumps(out, indent=1, default=float))
    if a.write and best:
        # knee by strictness: q/2, q, 2q native quantiles (as round 18)
        off = best["offsets"]
        SR = word_s(R, evR, off)
        nat = np.concatenate([SR[k] for k, r in R.items() if not r["meta"].get("label")])
        nat = nat[np.isfinite(nat)]
        par = {"form": best["form"], "width": best["width"],
               "knee": [float(np.quantile(nat, best["q"] / 2)), best["knee"], float(np.quantile(nat, min(2 * best["q"], 0.5)))],
               "offset": off, "q": best["q"], "qc": best["qc"], "rivals": RIVALS_ON,
               "fitted_on": "round 19 Korean dev set (bench/round19ko/select.py): FLEURS ko validation + Common Voice dev pool"}
        if best["form"] == "mult":
            par["floor"] = best["par"]
        else:
            par["slope"] = best["par"]
            par["floor"] = 0.0
        Path("pronunciada/data/calib/ko_identity_word.json").write_text(json.dumps(par, indent=1, ensure_ascii=False))
        print("wrote pronunciada/data/calib/ko_identity_word.json")


if __name__ == "__main__":
    main()
