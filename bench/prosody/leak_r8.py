"""Round 8 audio-level leak check (independent of the feature-space augmentation): FLEURS
validation clips resynthesised with a flat pitch at the speaker's median (Praat PSOLA,
bench.prosody.leak_test.monotone; offline only). Segments, timing and loudness stay; every model
input (pitch track, low mel, ZIPA activations) is recomputed from the resynthesised audio through
the runtime ProsodyScorer, with the prepared phone spans of the original.

    .venv/bin/python -m bench.prosody.leak_r8 [n_clips_per_language=60]

stress (en de es it pt ru; shipped `stress` vs `stress_v3`): words with >= 2 syllables;
  acc_orig / still_dictionary_mono = argmax of the word's stress distribution on the dictionary
  syllable; mono_auc = P(expected) original vs monotone (higher = the model hears the flattening;
  stress is also carried by loudness and duration, so a flat pitch does not remove it).
(The round-8 Japanese pitch-accent part was removed in round 15.)
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")


def run(n=60):
    from bench import metrics as M
    from bench.prosody.common import load_prepped, save_json
    from bench.prosody.leak_test import monotone
    from bench.prosody.prep import fleurs_validation
    from pronunciada.acoustic import AcousticModel
    from pronunciada.prosody import ProsodyScorer

    am = AcousticModel(threads=2, embeddings=True)
    res = {}
    plan = [(lg, "PRONUNCIADA_STRESS_MODEL", ["stress", "stress_v3"]) for lg in ("en", "de", "es", "it", "pt", "ru")]
    for lang, env, models in plan:
        utts = [u for u in load_prepped("fleurs_val", lang, "dev") if u["ref"].syllables]
        utts = utts[::max(1, len(utts) // n)][:n]
        want = {u["id"] for u in utts}
        wavs = {r["id"]: r["wav"] for r in fleurs_validation(lang) if r["id"] in want}
        inp = {}
        for u in utts:
            w = wavs[u["id"]]
            m = monotone(w)
            inp[u["id"]] = {c: (x, am.posteriors_from_features(am._features(x), True)[1]) for c, x in (("orig", w), ("mono", m))}
        for model in models:
            os.environ[env] = model
            sc = ProsodyScorer(lang)
            po, pm, so, sm = [], [], [], []
            for u in utts:
                outs = {c: sc.score(x, u["ref"], u["spans"], embeddings=e) for c, (x, e) in inp[u["id"]].items()}
                ref = u["ref"]
                for i, s in enumerate(ref.syllables):
                    if outs["orig"][i].get("stress_dist") is None:
                        continue
                    w = ref.words[s.word]
                    j = i - w.syllables[0]
                    do, dm = np.array(outs["orig"][i]["stress_dist"]), np.array(outs["mono"][i]["stress_dist"])
                    po.append(do[j])
                    pm.append(dm[j])
                    so.append(do.argmax() == j)
                    sm.append(dm.argmax() == j)
            lab = np.r_[np.zeros(len(po)), np.ones(len(pm))]
            r = {"n": len(po), "acc_orig": float(np.mean(so)), "still_label_mono": float(np.mean(sm)),
                 "mono_auc": M.roc_auc(lab.astype(int), 1 - np.r_[po, pm]), "mean_p_orig": float(np.mean(po)),
                 "mean_p_mono": float(np.mean(pm))}
            res[f"{lang}:{model}"] = r
            print(lang, model, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
        os.environ.pop(env, None)
    save_json(res, "leak_r8.json")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
