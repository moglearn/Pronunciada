"""Round 12b: FLEURS-en VALIDATION natives, sound_B vs sound_E (shown sound / word acceptance,
coaching flags per clip at the three sensitivities).

    python -m bench.calib.native_sound_emb
"""
from __future__ import annotations

import json
import os

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")


def main(tag=""):
    from bench.calib.common import save_json
    from bench.pipeline.common import iter_parquet_audio, records
    from pronunciada.scorer import Scorer

    recs = records("fleurs-validation", lang="en")
    by = {int(r["audio"].split(":")[2]): r for r in recs}
    path = recs[0]["audio"].split(":")[1]
    # before: round-11 behaviour (sound_B everywhere, global coaching model); after: shipped default
    import copy

    from pronunciada import coach as Co

    S = {"before": Scorer("en", threads=2, sound_model="sound_B", explain_away=False),
         "after": Scorer("en", threads=2, explain_away=False)}
    m_after = Co.model()
    m_before = copy.deepcopy(m_after)
    m_before["by_lang"].pop("en", None)
    acc = {n: {"s": [], "w": [], "flags": {k: 0 for k in ("careful", "balanced", "eager")}} for n in S}
    nclip = 0
    for i, wav in iter_parquet_audio(path, set(by)):
        rec = by[i]
        nclip += 1
        for n, sc in S.items():
            try:
                res = sc.score((wav, 16000), rec["text"])
            except ValueError:
                continue
            acc[n]["s"] += [p.prob for w in res.words for p in w.phones]
            acc[n]["w"] += [w.prob for w in res.words]
            for k in acc[n]["flags"]:
                X = Co.sound_inputs(res, heard=True)
                d = Co.decide(X, "en", k, m_before if n == "before" else m_after)
                acc[n]["flags"][k] += int(np.sum(d["flag"]))
                acc[n].setdefault("named", {}).setdefault(k, 0)
                acc[n]["named"][k] += sum(1 for i in np.flatnonzero(d["flag"]) if d["heard"][i])
    out = {"clips": nclip}
    for n, a in acc.items():
        s, w = np.array(a["s"]), np.array(a["w"])
        out[n] = {"native_sounds": float(np.mean(s >= 0.5)), "native_words": float(np.mean(w >= 0.5)),
                  "coach_flags_per_clip": {k: v / nclip for k, v in a["flags"].items()},
                  "coach_named_per_clip": {k: v / nclip for k, v in a.get("named", {}).items()}}
    save_json(f"native_sound_emb_en{tag}.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "")
