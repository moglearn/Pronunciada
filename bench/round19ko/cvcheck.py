"""Round 19 (Korean): speaker-grouped 2-fold check of the whole selection (offsets, knee, grid choice).

    python -m bench.round19ko.cvcheck --head PATH

Dev speakers (CV speaker / FLEURS sentence id) are split in two by a fixed hash. On each half the
selection of select.py runs (class offsets and knee from that half's controls, guards on that half's
FLEURS-validation controls; the planted guard uses fleurs-val-cor of that half's sentences), and the
chosen setting is applied to the other half. Out-of-fold word probabilities of both halves are pooled
and scored with the dev metrics, against the base, with speaker-bootstrap paired differences.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from bench.round19ko import collect as CO
from bench.round19ko import devmetrics as DM
from bench.round19ko import select as SE


def half(spk: str) -> int:
    return int(hashlib.sha1(f"r19ko|{spk}".encode()).hexdigest(), 16) % 2


def choose(R, COR, evR, evC):
    base_P = {k: r["words"]["prob"] for k, r in R.items()}
    base_C = {k: r["words"]["prob"] for k, r in COR.items()}
    m0 = DM.metrics(R)
    b_acc, b_pl = SE.native_accept(R, base_P), SE.planted_auc(COR, base_C)
    best = None
    for qc in (None, 0.05):
        off = SE.class_offsets(R, evR, qc)
        SR, SC = SE.word_s(R, evR, off), SE.word_s(COR, evC, off)
        nat = np.concatenate([SR[k] for k, r in R.items() if not r["meta"].get("label")])
        nat = nat[np.isfinite(nat)]
        for q, (form, par), w in itertools.product(SE.QS, SE.FORMS, SE.WIDTHS):
            knee = float(np.quantile(nat, q))
            P = {k: SE.apply(base_P[k], SR[k], knee, form, par, w, R[k]["words"]["sounds"]) for k in base_P}
            PC = {k: SE.apply(base_C[k], SC[k], knee, form, par, w, COR[k]["words"]["sounds"]) for k in base_C}
            m = DM.metrics(R, P)
            ok = (SE.native_accept(R, P) >= b_acc - 0.003 and SE.planted_auc(COR, PC) >= b_pl - 1e-4
                  and m["fleurs-val"]["auc"] >= m0["fleurs-val"]["auc"] - 1e-4)
            key = (round(m["all"]["auc"], 4), -m["all"]["fa80"])
            if ok and (best is None or key > best[0]):
                best = (key, {"off": off, "knee": knee, "form": form, "par": par, "width": w, "q": q, "qc": qc})
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    a = ap.parse_args()
    z = np.load(a.head)
    H = {k: (z[k].astype(np.float32) if z[k].dtype.kind == "f" else z[k]) for k in z.files}
    H["index"] = {str(c): j for j, c in enumerate(H["classes"].tolist())}
    R = {k: r for k, r in SE.load_dev(True).items() if "error" not in r}
    COR = {k: r for k, r in CO.load("fleurs-val-cor", "base").items() if "error" not in r}
    cache = {}
    evR, evC = SE.evidence_sets(R, H, cache), SE.evidence_sets(COR, H, cache)
    P_oof = {}
    chosen = []
    for h in (0, 1):
        tr = {k: r for k, r in R.items() if half(r["meta"]["speaker"]) != h}
        te = {k: r for k, r in R.items() if half(r["meta"]["speaker"]) == h}
        ctr = {k: r for k, r in COR.items() if half(r["meta"]["speaker"]) != h}
        c = choose(tr, ctr, {k: evR[k] for k in tr}, {k: evC[k] for k in ctr})
        chosen.append({k: v for k, v in c.items() if k != "off"})
        S = SE.word_s(te, {k: evR[k] for k in te}, c["off"])
        for k in te:
            P_oof[k] = SE.apply(te[k]["words"]["prob"], S[k], c["knee"], c["form"], c["par"], c["width"],
                                te[k]["words"]["sounds"])
    m0 = DM.metrics(R, boot=1000)
    m1 = DM.metrics(R, P_oof, boot=1000)
    pd = DM.paired(R, None, P_oof)
    out = {"chosen_per_fold": chosen, "base": m0, "oof": m1, "paired": pd,
           "native_accept_fleurs_val": [SE.native_accept(R, {k: r["words"]["prob"] for k, r in R.items()}),
                                        SE.native_accept(R, P_oof)]}
    Path("results/tables/round19_ko_cvcheck.json").write_text(json.dumps(out, indent=1, default=float))
    print("base", DM.line(m0))
    print("oof ", DM.line(m1))
    print("paired", pd, "chosen", chosen, "native", out["native_accept_fleurs_val"])


if __name__ == "__main__":
    main()
