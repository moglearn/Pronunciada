"""Round 19 (Korean, part 2): per-sound and per-word evidence from collected rows (factor off).

Per sound (pronunciada.ko_identity on the stored pooled vectors):
  cls    Korean sound class of the reference sound (None: not judged)
  pos    position: "coda", "final" (nucleus or onset of the word's last syllable: particles and
         endings), "initial" (first syllable), "medial"
  s      log P(expected) - log max P(other class)   (identity margin)
  top    the head's most probable class
  llr    aligner llr_sub (L_can - L_wild <= 0: "anything" beats the reference sound by -llr)
  peak   posterior peak of the aligned token; nfr aligned frames
Per word: the sounds above plus
  ins    lowest llr_ins over the boundaries before, inside and after the word (an extra sound)
  dele   lowest llr_del over the word's sounds (a sound missing)
  dur    log(word frames / (number of sounds x the utterance's median frames per sound))
"""
from __future__ import annotations

import numpy as np

from pronunciada import ko_identity as KI


def positions(ref) -> list[str]:
    out = []
    for p in ref.phones:
        w = ref.words[p.word]
        syl = ref.syllables[p.syllable]
        a, b = syl.phones
        nuc = next((k for k in range(a, b) if ref.phones[k].is_nucleus), None)
        i = len(out)
        if nuc is not None and i > nuc:
            out.append("coda")
        elif p.syllable == w.syllables[1] - 1 and w.syllables[1] - w.syllables[0] > 1:
            out.append("final")
        elif p.syllable == w.syllables[0]:
            out.append("initial")
        else:
            out.append("medial")
    return out


def sound_table(r, ref, H=None) -> dict | None:
    ph = r["phones"]
    if ref is None or [p.ipa for p in ref.phones] != list(ph["ipa"]) or ph["V"] is None:
        return None
    H = H or KI._head()
    cls = KI.classes(ref)
    P = KI.probs(ph["V"].astype(np.float32), H)
    lp = np.log(np.clip(P, 1e-9, 1))
    n = len(cls)
    s = np.full(n, np.nan)
    top = [None] * n
    for i, c in enumerate(cls):
        if c is None or c not in H["index"]:
            continue
        j = H["index"][c]
        s[i] = lp[i, j] - np.delete(lp[i], j).max()
        top[i] = str(H["classes"][int(lp[i].argmax())])
    return {"cls": cls, "pos": positions(ref), "s": s, "top": top, "lp": lp,
            "llr": np.asarray(ph["llr_sub"], np.float64), "dele": np.asarray(ph["llr_del"], np.float64),
            "peak": np.asarray(ph.get("peak", np.full(n, np.nan)), np.float64),
            "nfr": np.asarray(ph["n_frames"], np.float64),
            "ins": np.asarray(ph.get("llr_ins", np.full(n, np.nan)), np.float64),
            "start": np.asarray(ph["start"]), "end": np.asarray(ph["end"]), "word": np.asarray(ph["word"])}


def word_feats(r, st) -> dict:
    nw = len(r["words"]["prob"])
    w = st["word"]
    fr = st["nfr"][st["nfr"] > 0]
    rate = float(np.median(fr)) if len(fr) else 3.0
    ins = np.full(nw, 0.0)
    dele = np.full(nw, 0.0)
    dur = np.full(nw, 0.0)
    for j in range(nw):
        idx = np.flatnonzero(w == j)
        if not len(idx):
            continue
        b = [idx[0] - 1] + list(idx)            # boundary after slot k: before word, inside, after
        b = [k for k in b if 0 <= k < len(st["ins"])]
        v = st["ins"][b]
        v = v[np.isfinite(v)]
        ins[j] = float(v.min()) if len(v) else 0.0
        dele[j] = float(np.min(st["dele"][idx]))
        span = st["end"][idx].max() - st["start"][idx].min()
        dur[j] = float(np.log(max(span, 1) / (len(idx) * rate)))
    return {"ins": np.clip(ins, -40, 0), "dele": np.clip(dele, -40, 40), "dur": dur}
