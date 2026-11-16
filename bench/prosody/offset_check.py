"""Is there a constant time offset between the aligner's spans and the pitch contour?
Shift every syllable window by delta frames, train a quick gradient-boosted classifier on the
contour-vector baseline features (aishell3 train, speakers split in two halves), report accuracy.
Train data only."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from bench.prosody.common import load_prepped, to_track, tone_targets  # noqa: E402
from pronunciada.prosody.features import (SylWindow, contour_vector, frame_features,  # noqa: E402
                                      syllable_windows)


def run(deltas=(-8, -6, -4, -2, 0, 2, 4, 6, 8), n_utt=2500):
    utts = load_prepped("aishell3", "zh", "train")[::2][:n_utt]
    spk = sorted({u["speaker"] for u in utts})
    half = set(spk[::2])
    base = []
    for u in utts:
        f = frame_features(to_track(u["track"]))
        w = syllable_windows(u["ref"], u["spans"], len(f))
        y, _, _ = tone_targets(u)
        base.append((f, w, y, u["speaker"] in half))
    res = {}
    for d in deltas:
        X, Y, G = [], [], []
        for f, w, y, g in base:
            T = len(f)
            ws = [SylWindow(*(int(np.clip(v + d, 0, T - 1)) for v in (x.start, x.end, x.nuc_start, x.nuc_end)),
                            x.tok_start, x.tok_end) for x in w]
            ws = [x if x.end > x.start else SylWindow(x.start, x.start + 1, x.start, x.start + 1, 0, 0) for x in ws]
            for i in range(len(ws)):
                if y[i] >= 0:
                    X.append(contour_vector(f, ws, i))
                    Y.append(y[i])
                    G.append(g)
        X, Y, G = np.array(X), np.array(Y), np.array(G)
        c = HistGradientBoostingClassifier(max_iter=150).fit(X[G], Y[G])
        res[d] = float((c.predict(X[~G]) == Y[~G]).mean())
        print(d, round(res[d], 4), flush=True)
    return res


if __name__ == "__main__":
    run()
