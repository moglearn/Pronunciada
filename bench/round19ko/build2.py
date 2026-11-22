"""Round 19 (Korean, part 2): build the word rows used by fit2.py from the r2 collections.

    python -m bench.round19ko.build2      # -> $CACHE/round19/korean/words_r2.pkl

Sets: dev (Whisper-filtered planted items; controls are also native-pool rows), fleurs-val-cor
(controlled errors, guard), zeroth-test (native pool; Zeroth test speakers are disjoint from the
Zeroth train speakers the head saw). One row per word: base probability, sounds probability, the
deciding-sound candidates of the identity evidence (per sound: class, position, margin, top class,
llr_sub, peak, frames) and the word's insertion / deletion / duration evidence; label = changed word.
"""
from __future__ import annotations

import pickle

import numpy as np

from bench.round19ko import collect as CO
from bench.round19ko import offline as OF
from bench.round19ko import wordfeat as WF
from bench.round19ko.cv_pool import work
from bench.round19ko.select import _ref_from_row


def unit_index(r):
    """index of our word that carries the changed unit (lowest-probability overlap is used by the
    metric; here every overlapping word is marked)."""
    from bench.round19ko.devmetrics import _unit_spans

    lab = r["meta"].get("label")
    if not lab:
        return set()
    a, b = _unit_spans(r["meta"]["target"])[lab["word_index"]]
    return {i for i, (c, d) in enumerate(r["words"]["span"]) if c < b and d > a}


def build(name, keep=None):
    from pronunciada.text import get_g2p

    g = get_g2p("ko")
    R = CO.load(name, "r2")
    if keep is not None:
        R = keep(R)
    out = {}
    for k, r in R.items():
        if "error" in r:
            continue
        m = r["meta"]
        if r.get("site"):                          # contrast-audit edit: rebuild the edited reference
            import zlib

            from bench.audit.contrasts import Site, apply

            ref = apply(g(m["target"]), Site(**r["site"]),
                        np.random.default_rng(zlib.crc32(f"{m['id']}/{r['site']}".encode()))).ref
        else:
            ref = g(m["target"]) if m.get("cor_kind") in (None, "clean") else _ref_from_row(r, g)
        st = WF.sound_table(r, ref)
        if st is None:
            continue
        wf = WF.word_feats(r, st)
        changed = unit_index(r) if name == "dev" else set()
        ww = r.get("word_wrong")
        st.pop("lp")
        out[k] = {"meta": m, "prob": np.asarray(r["words"]["prob"]), "sounds": np.asarray(r["words"]["sounds"]),
                  "span": r["words"]["span"], "st": st, "wf": wf, "changed": changed,
                  "word_wrong": list(ww) if ww is not None else None, "src": m["src"],
                  "site": r.get("site"), "edit_word": r.get("edit_word"), "partner_in_alts": r.get("partner_in_alts")}
    return out


def main():
    import sys

    if sys.argv[1:] == ["test"]:          # HELD-OUT sets, built only after the candidate is frozen
        D = {"swap": build("swap-test"), "test_cor": build("fleurs-test-cor"), "audit": build("audit-test")}
        (work() / "words_r2_test.pkl").write_bytes(pickle.dumps(D))
    else:
        D = {"dev": build("dev", lambda R: OF.keep_items(R, work() / "asr.jsonl")),
             "val_cor": build("fleurs-val-cor"), "zeroth": build("zeroth-test")}
        (work() / "words_r2.pkl").write_bytes(pickle.dumps(D))
    print({k: len(v) for k, v in D.items()})


if __name__ == "__main__":
    main()
