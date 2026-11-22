"""Runtime parity of the round-20 switch on the dev set.

    python -m bench.round20_ja_length.parity [--n 60] [--precision fp32|int8]

* switch unset and PRONUNCIADA_JA_LENGTH20=0: every word probability, flag and sound length
  probability equals the devrun "main" record (current main; the devrun main records equal the
  round-19 final dumps), max |diff| reported;
* PRONUNCIADA_JA_LENGTH20=1 (packaged head and parameters): equals the devrun record of the same
  setting (a = 1, both classes at the 1 % native quantile, tag a1_q1 of devrun g1).
For int8, unset vs =0 vs =1 are compared with each other only (the devrun records are fp32).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from bench.round20_ja_length.common import OUT


def outputs(rec) -> list[float]:
    v = []
    for w in rec["words"]:
        v.append(w["prob"])
        v.append(float(bool(w["flag"])))
        v += [p[3] if p[3] is not None else -1.0 for p in w["ph"]]
    return v


def maxdiff(a, b) -> float:
    xa, xb = outputs(a), outputs(b)
    assert len(xa) == len(xb)
    return max((abs(x - y) for x, y in zip(xa, xb)), default=0.0)


def main(argv=None) -> None:
    import warnings

    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--precision", default="fp32")
    a = ap.parse_args(argv)
    from bench.round19_ja.dump import _wav, record
    from pronunciada.scorer import Scorer

    def load(tag):
        f = glob.glob(str(OUT / "devrun" / "g1" / f"dev_{tag}.s*.jsonl"))[0]
        return {r["id"]: r for r in map(json.loads, open(f, encoding="utf-8")) if not r.get("error")}

    main_r, on_r = load("main"), load("a1_q1")
    ids = sorted(main_r)[:: max(len(main_r) // a.n, 1)][: a.n]
    sc = Scorer("ja", precision=a.precision, threads=2)
    res = {"n": len(ids), "precision": a.precision}
    d_unset = d_off = d_on = d_off_on_int8 = 0.0
    changed = 0
    for i in ids:
        r = main_r[i]
        wav = _wav(r["audio"])
        out = {}
        for val in (None, "0", "1"):
            if val is None:
                os.environ.pop("PRONUNCIADA_JA_LENGTH20", None)
            else:
                os.environ["PRONUNCIADA_JA_LENGTH20"] = val
            out[val] = record(sc.score((wav, 16000), r["text"]))
        os.environ.pop("PRONUNCIADA_JA_LENGTH20", None)
        if a.precision == "fp32":
            d_unset = max(d_unset, maxdiff(out[None], r))
            d_off = max(d_off, maxdiff(out["0"], r))
            d_on = max(d_on, maxdiff(out["1"], on_r[i]))
        else:
            d_off = max(d_off, maxdiff(out[None], out["0"]))
        changed += maxdiff(out["1"], out["0"]) > 0
    res.update(max_diff_unset_vs_main=d_unset, max_diff_off_vs_main=d_off, max_diff_on_vs_devrun=d_on,
               clips_changed_when_on=changed)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
