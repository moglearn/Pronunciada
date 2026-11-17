"""Round 12b: training-vs-runtime parity of sound_E on 200 speechocean762 TRAIN clips, latency, size.

    python -m bench.calib.parity_sound_emb

Training side: the shipped sound_E applied to the training feature table (bench pipeline features
+ the projection fitted on all rows + prototype features from bench/calib/sound_emb.py vectors).
Runtime side: pronunciada.Scorer("en").score(audio, text) (activations, pooling, projection,
prototypes and calibrator computed live). Compared per sound: P lenient, P strict, 0-2 score.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")


def main():
    from bench.calib.common import save_json
    from bench.calib.train_sound_emb import load
    from bench.data import load as load_ds
    from pronunciada import calibrate as C
    from pronunciada import features as F
    from pronunciada.scorer import Scorer
    from pronunciada.sound_emb import load as load_aux

    df, t, lab, groups, vocab, V = load()
    sc = Scorer("en", threads=2, sound_model="sound_E", explain_away=False)
    assert sc.sound_aux is not None and sc.am.embeddings
    aux = load_aux("sound_E")
    t = dict(t)
    ok = np.isfinite(V[:, 0])
    t["emb_proj"] = np.where(ok, ((V - aux.pmu) / aux.psd) @ aux.pW + aux.pb, np.nan)
    m = sc.sound
    X = m.matrix(t)
    out = m.predict(X=X) if hasattr(m, "predict") else None
    lg = m.logits(X=X)
    recs = [r for r in load_ds("speechocean762", split="train")]
    rng = np.random.default_rng(0)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), 200, replace=False))]
    rows = {}
    for i, (u, k) in enumerate(zip(df.utt.values, df.phone_index.values.astype(int))):
        rows[(u, k)] = i
    dl, ds, dr, n = [], [], [], 0
    for r in recs:
        res = sc.score(r["audio"], r["text"])
        d = sc.details(res)
        for k in range(len(res._pronunciada["ref"].phones)):  # noqa: SLF001
            i = rows.get((r["id"], k))
            if i is None:
                continue
            n += 1
            dl.append(abs(C.sigmoid(d["phone_z_len"][k]) - C.sigmoid(lg["z_len"][i])))
            ds.append(abs(C.sigmoid(d["phone_z_str"][k]) - C.sigmoid(min(lg["z_str"][i], lg["z_len"][i]))))
            dr.append(abs(d["phone_score"][k] - out["reg"][i]))
    # latency on a ~3.5 s clip
    durs = np.array([r["dur"] for r in recs])
    r0 = recs[int(np.argmin(np.abs(durs - 3.5)))]
    lat = {}
    for name in ("sound_B", "sound_E"):
        s2 = Scorer("en", threads=2, sound_model=name, explain_away=False)
        s2.score(r0["audio"], r0["text"])
        ts = []
        for _ in range(15):
            s2._lp_cache.clear()  # noqa: SLF001
            t0 = time.perf_counter()
            s2.score(r0["audio"], r0["text"])
            ts.append((time.perf_counter() - t0) * 1e3)
        lat[name] = float(np.median(ts))
    res = {"n_sounds": n, "max_abs_p_len": float(np.max(dl)), "max_abs_p_str": float(np.max(ds)),
           "p99_abs_p_len": float(np.quantile(dl, 0.99)), "max_abs_reg": float(np.max(dr)),
           "latency_ms_median": lat, "clip_s": float(r0["dur"]),
           "kb": {p: round(os.path.getsize(os.path.join(C.data_dir(), p)) / 1024, 1)
                  for p in ("sound_E.npz", "sound_E_emb.npz", "sound_B.npz")}}
    save_json("parity_sound_emb.json", res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
