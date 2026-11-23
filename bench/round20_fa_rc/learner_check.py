"""Learner practice check of the false-alarm release candidate: TRAIN splits only.

    python -m bench.round20_fa_rc.learner_check OUT.json

speechocean762, L2-ARCTIC and EpaDB TRAIN clips (English learners), each scored with the release defaults and
with every round-20 switch set to 0 (= main, proven by bench/round20_fa_rc/run_equiv.sh), same Scorer, the
user call Scorer.score(audio, text). Counts the clips whose output (SentenceResult.to_dict() without timing)
changes at all. For speechocean762 also the word correlation with the human word accuracy
(WordResult.prob, prompt = the scored words joined by spaces, as bench/round20_native_fa/so762_wordprob.py),
with a paired speaker-clustered bootstrap (2,000 resamples) of the change.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

OFF = {"PRONUNCIADA_R20_NUMALT": "0", "PRONUNCIADA_R20_DE_LEXALT": "0", "PRONUNCIADA_R20_ES_GLIDE": "0",
       "PRONUNCIADA_R20_KO_NINS": "0"}


def _d(res):
    d = res.to_dict()
    d.pop("timing_ms", None)
    return json.dumps(d, sort_keys=True, default=str)


def main() -> None:
    from bench.data import load
    from pronunciada import Scorer

    out_path = sys.argv[1]
    for k in OFF:
        os.environ.pop(k, None)
    sc = Scorer("en", threads=4)
    out = {}
    so_rows = []
    for ds in ("speechocean762", "l2arctic", "epadb"):
        recs = load(ds, split="train")
        n = changed = errors = 0
        examples = []
        for rec in recs:
            if ds == "speechocean762":
                hw = rec["labels"]["words"]
                text = " ".join(w["text"] for w in hw)
            else:
                text = rec["text"]
            try:
                r1 = sc.score(rec["audio"], text)
                os.environ.update(OFF)
                r0 = sc.score(rec["audio"], text)
            except ValueError:
                errors += 1
                continue
            finally:
                for k in OFF:
                    os.environ.pop(k, None)
            n += 1
            if _d(r0) != _d(r1):
                changed += 1
                if len(examples) < 10:
                    examples.append(rec["id"])
            if ds == "speechocean762" and len(r0.words) == len(hw):
                so_rows.append({"spk": rec["speaker"], "h": [w["accuracy"] for w in hw],
                                "p0": [w.prob for w in r0.words], "p1": [w.prob for w in r1.words]})
        out[ds] = {"clips": n, "clips_changed": changed, "errors": errors, "changed_ids": examples}
        print(ds, out[ds], flush=True)
    h = np.concatenate([r["h"] for r in so_rows])
    p0 = np.concatenate([r["p0"] for r in so_rows])
    p1 = np.concatenate([r["p1"] for r in so_rows])
    spk = np.concatenate([[r["spk"]] * len(r["h"]) for r in so_rows])

    def c(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    res = {"clips": len(so_rows), "words": int(len(h)), "words_changed": int((p0 != p1).sum()),
           "word_pcc_off": c(p0, h), "word_pcc_rc": c(p1, h)}
    res["delta"] = res["word_pcc_rc"] - res["word_pcc_off"]
    if res["words_changed"]:
        rng = np.random.default_rng(0)
        us = np.unique(spk)
        idx = {s: np.where(spk == s)[0] for s in us}
        d = []
        for _ in range(2000):
            ii = np.concatenate([idx[s] for s in rng.choice(us, len(us))])
            d.append(c(p1[ii], h[ii]) - c(p0[ii], h[ii]))
        res["delta_ci95"] = [float(np.quantile(d, .025)), float(np.quantile(d, .975))]
    out["speechocean762"]["word_correlation"] = res
    print(json.dumps(out, indent=1))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
