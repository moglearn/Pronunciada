"""Mechanism check for the word-omission neighbour effect (diagnosis only; an existing Scorer option, no code change).

    python -m bench.round20_native_fa.ea_check --lang en,de,es,fr

Hypothesis: when the prompt has a word the speaker did not say, the aligner must still give that word's
sounds some frames, takes them from the neighbouring words, and the neighbours then look dropped
(presence factor) or wrong. If so, letting that word's sounds be skipped should restore the neighbours.
The existing explain-away pass (Scorer(explain_away=True); pronunciada.align wild_slots: the lowest word's
sounds become a skippable wildcard and the evidence is recomputed) does exactly that. Here it is run on
every word-omission item with the band removed (PRONUNCIADA_EA_BAND=0,1; up to 2 passes), and the
neighbours' and the inserted word's benchmark scores are compared with the default run at the practice
FA@80 threshold. Nothing is selected from this.
"""
from __future__ import annotations

import argparse
import json
import os

from bench.round20_native_fa import core as K


def run(lang: str) -> dict:
    os.environ["PRONUNCIADA_EA_BAND"] = "0,1"
    from bench.third_party.lingolix import align, error_signal, reduce_azure_shape
    from pronunciada import Scorer
    from pronunciada.compat.azure import to_azure

    rows = K.word_rows(lang)
    thr = K.threshold(rows)
    base = {(r["item"], r["k"]): r for r in rows}
    its = [it for it in K.practice_items(lang) if (it.get("label") or {}).get("bucket") == "word_omission"]
    sc = Scorer(lang, threads=2, explain_away=True)
    nb_b = nb_e = ins_b = ins_e = n_nb = 0
    oth_b = oth_e = n_oth = 0
    ds = []
    for it in its:
        res = sc.score(it["audio"], it["target"])
        raw = to_azure(res, granularity="Phoneme", enable_miscue=True, phoneme_alphabet="IPA", layout="rest",
                       grading_system="HundredMark")
        toks = it["target"].split()
        al = align(toks, reduce_azure_shape(raw))
        pos = it["label"]["word_index"]
        for k in range(len(toks)):
            b = base.get((it["id"], k))
            if b is None:
                continue
            e = error_signal(al[k])
            if k == pos:
                ins_b += b["err"] >= thr
                ins_e += e >= thr
            elif abs(k - pos) == 1:
                n_nb += 1
                nb_b += b["err"] >= thr
                nb_e += e >= thr
                ds.append(b["err"] - e)
            else:
                n_oth += 1
                oth_b += b["err"] >= thr
                oth_e += e >= thr
    out = {"items": len(its), "neighbours": n_nb, "neighbour_fa_default": 100 * nb_b / max(1, n_nb),
           "neighbour_fa_explain_away": 100 * nb_e / max(1, n_nb),
           "inserted_caught_default": 100 * ins_b / max(1, len(its)),
           "inserted_caught_explain_away": 100 * ins_e / max(1, len(its)),
           "other_words": n_oth, "other_fa_default": 100 * oth_b / max(1, n_oth),
           "other_fa_explain_away": 100 * oth_e / max(1, n_oth),
           "neighbour_mean_score_gain": sum(ds) / max(1, len(ds))}
    print(lang, json.dumps(out), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    a = ap.parse_args(argv)
    res = {l: run(l) for l in a.lang.split(",")}
    K.save_json("ea_check.json", res)


if __name__ == "__main__":
    main()
