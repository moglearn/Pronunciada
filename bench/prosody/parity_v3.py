"""Training-vs-runtime parity for the v3 tone models (dev clips only: FLEURS zh validation).

    .venv/bin/python -m bench.prosody.parity_v3 tone_zh_v3z [n=150]

Training side: bench.prosody.tone3 (prepared pitch track stored as float16, cached low-mel and
pooled-ZIPA features, torch checkpoint). Runtime side: pronunciada.prosody.ProsodyScorer
(onnxruntime) on the waveform, with the same phone spans and ZIPA activations computed live by
pronunciada.acoustic.AcousticModel(embeddings=True). Reports max / 99th pct |dP| over syllables,
argmax agreement and accepted-probability differences.
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")


def main(stem, n=150):
    os.environ["PRONUNCIADA_TONE_MODEL"] = stem
    from bench.prosody import tone2, tone3
    from bench.prosody.common import load_prepped
    from bench.prosody.prep import fleurs_validation
    from pronunciada.acoustic import AcousticModel
    from pronunciada.prosody import ProsodyScorer

    sc = ProsodyScorer("zh")
    net, kind, _ = tone3.load3(sc.meta["model"])
    T = sc.meta["temperature"]
    utts = load_prepped("fleurs_val", "zh", "dev")
    utts = [u for u in utts if u["ref"].syllables][::max(1, len(utts) // n)][:n]
    wavs = {u["id"]: None for u in utts}
    for r in fleurs_validation("zh"):
        if r["id"] in wavs:
            wavs[r["id"]] = r["wav"]
    am = AcousticModel(threads=2, embeddings=True) if sc.inputs.get("zemb") else None
    d, agree, dacc = [], [], []
    for u in utts:
        X, _, _, _ = tone3.utt_example(u)
        Pt = tone2.softmax(tone2.predict_utts(net, kind, [X])[0], T)
        emb = am.posteriors_from_features(am._features(wavs[u["id"]]), True)[1] if am else None
        out = sc.score(wavs[u["id"]], u["ref"], u["spans"], embeddings=emb)
        idx = [i for i, s in enumerate(u["ref"].syllables) if s.tone is not None]
        Pr = np.array([out[i]["tone_probs"] for i in idx])
        Pt = Pt[idx]
        d.append(np.abs(Pr - Pt).max(1))
        agree.append(Pr.argmax(1) == Pt.argmax(1))
        for k, i in enumerate(idx):
            s = u["ref"].syllables[i]
            acc = [t - 1 for t in {s.tone, *s.tone_alts}]
            dacc.append(abs(Pr[k, acc].sum() - Pt[k, acc].sum()))
    d, agree, dacc = np.concatenate(d), np.concatenate(agree), np.array(dacc)
    print(f"{stem}: {len(utts)} clips, {len(d)} syllables; max |dP| {d.max():.4f}, 99th pct {np.quantile(d, .99):.4f}, "
          f"mean {d.mean():.4f}; argmax agreement {agree.mean():.4f}; |dP(acceptable)| max {dacc.max():.4f} "
          f"mean {dacc.mean():.4f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 150)
