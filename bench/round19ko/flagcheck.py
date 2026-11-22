"""Round 19 (Korean, part 2): runtime check of the delta parameters, and the error-flag decision.

    python -m bench.round19ko.flagcheck

Scores the Whisper-filtered dev items with the public Scorer (shipped parameters), and records per
word: the probability, the identity delta, the Azure-format ErrorType of pronunciada.compat.azure
(the system's own flag, which today comes from flagged sounds), and the changed-word label.
1) runtime == offline: word probabilities against bench.round19ko.fit2's offline formula on the
   factor-off rows (max |dP|);
2) flag rule candidates: also flag a word whose delta <= -D (D grid). Adopted only if the flag's
   precision on dev (changed words among flagged words) does not fall; the D with the highest
   recall among those is taken.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from bench.round19ko import fit2 as F2
from bench.round19ko.cv_pool import work


def main():
    from pronunciada import Scorer
    from pronunciada import ko_identity as KI
    from pronunciada.compat.azure import to_azure

    dev, _, _ = F2.data()
    cal = KI._word_params()
    items = {json.loads(x)["item"]: json.loads(x) for x in (work() / "dev_items.jsonl").read_text(encoding="utf-8").splitlines()}
    sc = Scorer("ko", threads=2)
    rows, maxd = [], 0.0
    from bench.pipeline.common import iter_parquet_audio

    def one(k, audio):
        nonlocal maxd
        r = dev[k]
        res = sc.score(audio, items[k]["target"])
        kid = res._pronunciada["ko_identity_s"]  # noqa: SLF001
        off = F2.apply(r, F2.feats(r, cal["thr"], cal.get("ins_c", 2.0)), np.asarray(cal["coef"]), cal["cap_up"])
        p = np.array([w.prob for w in res.words])
        maxd = max(maxd, float(np.max(np.abs(p - off))))
        az = to_azure(res, granularity="Word", enable_miscue=True, layout="rest")
        et = [w.get("ErrorType") for w in az["NBest"][0]["Words"] if w.get("ErrorType") != "Insertion"]
        for j, w in enumerate(res.words):
            rows.append({"item": k, "j": j, "prob": float(w.prob), "delta": float(kid["delta"][j]),
                         "flag": et[j] not in (None, "None") if j < len(et) else False,
                         "changed": j in r["changed"], "src": r["src"]})

    pq = {k: items[k] for k in dev if str(items[k]["audio"]).startswith("parquet:")}
    if pq:
        path = next(iter(pq.values()))["audio"].split(":")[1]
        by = {}
        for k, it in pq.items():
            by.setdefault(int(it["audio"].split(":")[2]), []).append(k)
        for i, wav in iter_parquet_audio(path, set(by)):
            for k in by[i]:
                one(k, (wav, 16000))
    for k in dev:
        if k not in pq:
            one(k, items[k]["audio"])
    y = np.array([r["changed"] for r in rows])
    fl = np.array([r["flag"] for r in rows])
    d = np.array([r["delta"] for r in rows])
    base = {"precision": float(y[fl].mean()), "recall": float(fl[y].mean()), "fa": float(fl[~y].mean())}
    cand = []
    for D in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        f2 = fl | (d <= -D)
        cand.append({"D": D, "precision": float(y[f2].mean()), "recall": float(f2[y].mean()), "fa": float(f2[~y].mean())})
    ok = [c for c in cand if c["precision"] >= base["precision"]]
    best = max(ok, key=lambda c: c["recall"]) if ok else None
    out = {"runtime_vs_offline_max_abs": maxd, "n_words": len(rows), "base_flag": base, "candidates": cand, "adopted": best}
    Path("results/tables/round19_ko2_flag.json").write_text(json.dumps(out, indent=1))
    (work() / "flagcheck_rows.pkl").write_bytes(pickle.dumps(rows))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
