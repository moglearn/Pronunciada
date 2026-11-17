"""Round 12b: Mandarin coaching score parity (runtime vs OMPAL-dev training arrays, 40 utterances)
and FLEURS-zh validation natives: flags per clip before / after at the three sensitivities.

    python -m bench.calib.check_zh_coach
"""
from __future__ import annotations

import copy
import json
import os
import pickle

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")


def main():
    from bench.calib.common import save_json
    from bench.calib.ompal_sound_emb import _path
    from bench.data import load
    from bench.pipeline.common import iter_parquet_audio, records
    from pronunciada import coach as Co
    from pronunciada import coach_zh as CZ
    from pronunciada.scorer import Scorer

    sc = Scorer("zh", threads=2, explain_away=False)
    D = pickle.loads(_path().read_bytes())
    recs = [r for r in load("ompal", split="dev")][:40]
    mx = 0.0
    for r in recs:
        res = sc.score(r["audio"], r["text"])
        X = Co.sound_inputs(res)
        rt = Co.error_prob(X, Co.model(), "zh")
        o = D[r["id"]]
        Xo = {"phone_vec": o["V"].astype(np.float32), "expected": X["expected"], "alts": X["alts"],
              "initial": o["initial"]}
        tr = CZ.error_prob(Xo, o["ep"])
        mx = max(mx, float(np.max(np.abs(rt - tr))))
    m_after = Co.model()
    m_before = copy.deepcopy(m_after)
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "tables", "coaching_v2_fit.json")) as f:
        old = json.load(f)
    m_before["by_lang"]["zh"] = {"initial": {s: (old["zh"]["thresholds"]["initial"][s] or 1.01) for s in ("careful", "balanced", "eager")},
                                 "final": {s: (old["zh"]["thresholds"]["final"][s] or 1.01) for s in ("careful", "balanced", "eager")}}
    vr = records("fleurs-validation", lang="zh")[:250]
    by = {int(r["audio"].split(":")[2]): r for r in vr}
    path = vr[0]["audio"].split(":")[1]
    cnt = {k: {s: 0 for s in ("careful", "balanced", "eager")} for k in ("before", "after")}
    n = 0
    for i, wav in iter_parquet_audio(path, set(by)):
        try:
            res = sc.score((wav, 16000), by[i]["text"])
        except ValueError:
            continue
        n += 1
        X = Co.sound_inputs(res, heard=True)
        for k, m in (("before", m_before), ("after", m_after)):
            for s in cnt[k]:
                cnt[k][s] += int(np.sum(Co.decide(X, "zh", s, m)["flag"]))
    out = {"parity_max_abs_score": mx, "fleurs_zh_clips": n,
           "native_flags_per_clip": {k: {s: v / n for s, v in d.items()} for k, d in cnt.items()}}
    save_json("zh_coach_12b_check.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
