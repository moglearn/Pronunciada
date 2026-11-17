"""Can the acoustic model's pooled activations separate Korean lax and tense stops among natives?

    python -m bench.round17.ko_tense

Labels from the spelling: syllable-onset ㄱㄷㅂㅅㅈ (lax) vs ㄲㄸㅃㅆㅉ (tense), where the front-end
writes both as the same plain symbol (k t p s tʃ). Lax onsets after a stop coda inside the word
are dropped (they are pronounced tense by rule). Probe: logistic regression on the round-12b pooled
vector (+ neighbours), fitted on FLEURS train natives (dump-r4-fleurs-train), tested on FLEURS
validation natives (dump-r4-audit clips). Also the shipped aspiration head's score as a 1-d
detector. Writes results/tables/round17_ko_tense.json.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from bench.coach.gather import out_dir
from pronunciada import attr_heads as AH

L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
LAX, TENSE = set("ㄱㄷㅂㅅㅈ"), set("ㄲㄸㅃㅆㅉ")
STOPS = {"k", "t", "p", "s", "tʃ"}


def onset_jamo(ch):
    o = ord(ch) - 0xAC00
    return L[o // 588] if 0 <= o < 11172 else None


def rows(dump, texts):
    from pronunciada.text import get_g2p

    g = get_g2p("ko")
    X, C, y = [], [], []
    for rid, o in dump.items():
        if "error" in o or o.get("lang", "ko") != "ko" or rid not in texts:
            continue
        ref = g(texts[rid])
        if len(ref.phones) != len(o["ipa"]):
            continue
        V = o["vec"].astype(np.float32)
        Cv = AH.context_vectors(V)
        for s in ref.syllables:
            j = onset_jamo(s.text[:1]) if s.text else None
            i = s.phones[0]
            if j is None or ref.phones[i].ipa not in STOPS or not (j in LAX or j in TENSE):
                continue
            w = ref.words[s.word]
            if j in LAX and i > w.phones[0] and ref.phones[i - 1].ipa in ("k", "t", "p"):
                continue
            X.append(V[i])
            C.append(Cv[i])
            y.append(j in TENSE)
    return np.array(X), np.array(C), np.array(y)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from bench.pipeline.common import fleurs_validation_records, records

    tr = pickle.loads((out_dir() / "dump-r4-fleurs-train.pkl").read_bytes())
    tr = {k: v for k, v in tr.items() if v.get("lang") == "ko"}
    texts = {r["id"]: r["text"] for r in records("fleurs", split="train", lang="ko")}
    Xtr, Ctr, ytr = rows(tr, texts)
    va = pickle.loads((out_dir() / "dump-r4-audit.pkl").read_bytes())
    va = {k: v for k, v in va.items() if v.get("lang") == "ko"}
    vtexts = {r["id"]: r["text"] for r in fleurs_validation_records("ko")}
    Xva, Cva, yva = rows(va, vtexts)
    out = {"n_train": int(len(ytr)), "n_tense_train": int(ytr.sum()), "n_dev": int(len(yva)), "n_tense_dev": int(yva.sum())}
    variants = {"own": (Xtr, Xva), "own+ctx": (np.hstack([Xtr, Ctr]), np.hstack([Xva, Cva])),
                "enc1 only": (Xtr[:, :256], Xva[:, :256])}
    for name, (A, B) in variants.items():
        mu, sd = A.mean(0), A.std(0) + 1e-6
        m = LogisticRegression(C=0.05, max_iter=500, class_weight="balanced").fit((A - mu) / sd, ytr)
        out[f"probe_auc_{name}"] = float(roc_auc_score(yva, m.decision_function((B - mu) / sd)))
    h = AH.shipped()
    P = h.predict(Xva, range(len(Xva)), Cva)
    out["aspiration_head_auc"] = float(roc_auc_score(yva, [-p["aspirated"]["yes"] if "aspirated" in p else 0 for p in P]))
    Path("results/tables/round17_ko_tense.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
