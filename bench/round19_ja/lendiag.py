"""Sound-level view of the length models on the round-19 dev set.

    python -m bench.round19_ja.lendiag [model.json ...]

Per source (cv / fleurs): planted length words (lowest length prob over the planted unit's words)
against control words (each word's lowest), AUC and hit at the 2 % control quantile; also how many
planted length units no sound of which is judged.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from bench.round19_ja.collect import load
from bench.round19_ja.length_fit import WORK, shipped_probs, sound_probs
from bench.round19_ja.unit_eval import roc_auc


def word_min(it, pr, wi):
    w = it["ref"].words[wi]
    v = [pr[i] for i in range(*w.phones) if pr[i] is not None]
    return min(v) if v else None


def unit_words(it):
    u = it["units"][it["label"]["word_index"]]
    return [k for k, (a, b) in enumerate(it["spans"]) if a < u["end"] and b > u["start"]]


def main(argv=None):
    names = (argv if argv is not None else sys.argv[1:])
    dev = load("dev")
    models = {"shipped": shipped_probs(dev)}
    for m in names:
        models[m] = sound_probs(dev, json.loads((WORK / m).read_text()))
    for name, P in models.items():
        for src in ("cv", "fleurs"):
            pos, neg, nj = [], [], 0
            for it, pr in zip(dev, P):
                if it["source"] != src:
                    continue
                if it["label"] and it["label"]["bucket"] == "length":
                    v = [x for x in (word_min(it, pr, k) for k in unit_words(it)) if x is not None]
                    if v:
                        pos.append(min(v))
                    else:
                        nj += 1
                if not it["label"]:
                    neg += [x for x in (word_min(it, pr, k) for k in range(len(it["ref"].words))) if x is not None]
            pos, neg = np.array(pos), np.array(neg)
            t = np.quantile(neg, 0.02)
            auc = roc_auc([1] * len(pos) + [0] * len(neg), list(-pos) + list(-neg))
            print(f"{name:28s} {src:6s} judged {len(pos):3d} not {nj:3d} AUC {auc:.3f} "
                  f"hit@2%neg {np.mean(pos <= t):.2f}", flush=True)


if __name__ == "__main__":
    main()
