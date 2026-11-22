"""Round 19, part 3: per-word evidence for fitting the word step, from the candidate-2 runtime with the
extra-sound (insertion) evidence on.

    python -m bench.round19_ja.stepfeat SET        # SET = fit2 | dev | devnat | devswap | cvdev

Per word: the base probability the step acts on (sounds x contrast x present, no length factor), and
the raw evidence:
* identity: per judged sound (class, position, margin) of pronunciada.ja_identity;
* length: per judged sound (expected long / short, vowel / consonant, length_prob, counterfactual
  long-vs-short log-likelihood, "how long" inputs of pronunciada.length_r19);
* word duration ratio against its mora count at the local mora rate (all words);
* extra sounds: llr_ins of the slots inside the word and of the gap before it (align.py, insertion);
  free-decode extra count; missing sounds: lowest llr_del, present probability.
Out: <work>/stepfeat/<SET>.pkl (list of records like the dump files, words carrying "f").
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys

import numpy as np

from bench.round19_ja.cv_pool import WORK

INPUTS = {"fit2": "set_fit2.jsonl", "dev": "set_dev.jsonl", "devnat": "devnat_in.jsonl",
          "devswap": "devswap_in.jsonl", "cvdev": "cvdev_in.jsonl"}


def word_feats(sc, res) -> list[dict]:
    from pronunciada import length as L
    from pronunciada import length_r19 as R
    from pronunciada.word_match import extra_counts

    c = res._pronunciada  # noqa: SLF001
    ref, ev, lpa = c["ref"], c["ev"], c["lpa"]
    fb = sc._posteriors(c["wav"], {})[1]  # noqa: SLF001
    inp = L.frame_inputs(ref, ev, lpa, fb)
    lab = L.labels(ref)
    idx, cls, rows = L.features(ref, lab, inp)
    R.extra(ref, inp, idx, rows, lpa=inp.get("lpa"), slot_se=inp.get("slot_se"), fb=inp.get("fb"))
    rowmap = dict(zip(idx, zip(cls, rows)))
    # word duration ratio for every word (judged or not): same arithmetic as length_r19.extra
    on = inp["onset"]
    se = inp["slot_se"]
    nW = len(ref.words)
    wdur = np.zeros(nW)
    wmor = np.zeros(nW)
    for k, w in enumerate(ref.words):
        a, b = w.phones
        s1 = int(on[b]) if b < len(on) else int(se[b - 1][1])
        wdur[k] = max(2 * (s1 - int(on[a])), 1)
        wmor[k] = R._morae(ref, w)  # noqa: SLF001
    wsi = c.get("word_step_in") or {}
    s_id, icls = wsi.get("s"), wsi.get("cls")
    ins = ev.llr_ins if getattr(ev, "llr_ins", None) is not None else np.full(len(ref.phones), np.nan)
    dl = np.asarray(c["tab"]["llr_del"], dtype=np.float64)
    xc = extra_counts(ref, ev)
    out = []
    for k, (w, wr) in enumerate(zip(ref.words, res.words)):
        a, b = w.phones
        others = (wdur.sum() - wdur[k]) / max(wmor.sum() - wmor[k], 1.0)
        md = max(others, 3.0)
        snd = []
        for i, pos in zip(range(a, b), ["final" if i == b - 1 else "first" if i == a else "mid" for i in range(a, b)]):
            d = {"cls": icls[i] if icls else None, "pos": pos,
                 "s": float(s_id[i]) if s_id is not None and np.isfinite(s_id[i]) else None}
            p = wr.phones[i - a]
            if p.length_prob is not None:
                cc, r = rowmap.get(i, (None, {}))
                d.update(lexp=p.length_expected, lp=float(p.length_prob), lcls=cc, cf=r.get("cf_llr"),
                         vstab=r.get("v_stable"), cdip=r.get("c_dip"))
            snd.append(d)
        gap = float(ins[a - 1]) if a > 0 and np.isfinite(ins[a - 1]) else None
        inner = [float(x) for x in ins[a:b - 1] if np.isfinite(x)]
        base = (wr.sounds_prob or 0.0) * (wr.contrast_prob if wr.contrast_prob is not None else 1.0) \
            * (wr.present_prob if wr.present_prob is not None else 1.0)
        out.append({"text": w.text, "span": list(w.char_span), "prob": wr.prob, "base": base,
                    "f": {"sounds": snd, "word_ratio": math.log(wdur[k] / (wmor[k] * md)), "morae": float(wmor[k]),
                          "ins_gap": gap, "ins_in": min(inner) if inner else None,
                          "del_min": float(np.clip(dl[a:b].min(), -50, 50)), "extra": int(xc[k]),
                          "present": wr.present_prob}})
    return out


def run(which: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import warnings

    warnings.simplefilter("ignore")
    from bench.round19_ja.dump import _wav
    from pronunciada.scorer import Scorer

    rows = [json.loads(x) for x in open(WORK / INPUTS[which], encoding="utf-8")]
    sc = Scorer("ja", threads=2)
    out = []
    for k, r in enumerate(rows):
        try:
            res = sc.score((_wav(r["audio"]), 16000), r["text"])
        except Exception:  # noqa: BLE001
            continue
        out.append({**{x: r.get(x) for x in ("id", "text", "units", "label", "labels", "group", "source", "side")},
                    "words": word_feats(sc, res)})
        if k % 100 == 99:
            print(which, k + 1, len(rows), file=sys.stderr, flush=True)
    d = WORK / "stepfeat"
    d.mkdir(exist_ok=True)
    (d / f"{which}.pkl").write_bytes(pickle.dumps(out))


if __name__ == "__main__":
    run(sys.argv[1])
