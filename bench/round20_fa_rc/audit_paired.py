"""Rule (c) for the contrast-audit rows: release candidate vs main, paired, on the same edits.

    python -m bench.round20_fa_rc.audit_paired <main chunk dir> <rc chunk dir> OUT.json

For every (language, contrast, direction) row of the audit table: sound hit@5 % and word hit@5 % of main and of
the release candidate, each with its own 5 % native threshold (as bench.audit.contrast_audit.analyse), and the
paired difference on 2,000 resamples of clips (seed 0; natives and edits of a drawn clip are drawn together and
both thresholds are recomputed in every resample); two-sided p. Aggregates only; nothing is printed per clip.
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FIVE = ("en", "de", "es", "fr", "ko")
B = 2000


def load(d: Path):
    clips, edits = {}, []
    for f in sorted(glob.glob(str(d / "chunk_*.pkl"))):
        with open(f, "rb") as fh:
            x = pickle.load(fh)
        for c in x["clips"]:
            clips[c["id"]] = c
        edits += [e for e in x["edits"] if "error" not in e]
    return clips, edits


def main() -> None:
    from bench.audit.contrasts import contrasts_for

    mdir, rdir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    res = {}
    for lang in FIVE:
        cm, em = load(mdir / lang)
        cr, er = load(rdir / lang)
        assert set(cm) == set(cr) and len(em) == len(er)
        ids = sorted(cm)
        ix = {k: i for i, k in enumerate(ids)}
        nat = {}
        for side, C in (("m", cm), ("r", cr)):
            for f in ("prob", "word_prob"):
                nat[side + f] = [np.asarray(C[k][f], dtype=np.float64) for k in ids]
        names = {c.name: c.members for c in contrasts_for(lang)}
        groups = defaultdict(list)
        for a, b in zip(em, er):
            assert (a["id"], a["contrast"], a["direction"]) == (b["id"], b["contrast"], b["direction"])
            if a.get("kind") == "tone":
                continue
            x, y = names[a["contrast"]]
            name = f"said {x}, expected {y}" if a["direction"] == "A>B" else f"said {y}, expected {x}"
            groups[(a["contrast"], name)].append((ix[a["id"]], a["prob"], b["prob"], a["word_prob"], b["word_prob"]))

        def thr(key, pick):
            return np.percentile(np.concatenate([nat[key][i] for i in pick]), 5)

        rng = np.random.default_rng(0)
        draws = [rng.integers(0, len(ids), len(ids)) for _ in range(B)]
        th = [(thr("mprob", d), thr("rprob", d), thr("mword_prob", d), thr("rword_prob", d)) for d in draws]
        full = (thr("mprob", range(len(ids))), thr("rprob", range(len(ids))),
                thr("mword_prob", range(len(ids))), thr("rword_prob", range(len(ids))))
        res[lang] = {}
        for (cn, dn), G in sorted(groups.items()):
            G = np.array(G, dtype=np.float64)
            clip = G[:, 0].astype(int)
            row = {"n": len(G)}
            for which, (cm_, cr_), (tm, tr) in (("sound", (1, 2), (0, 1)), ("word", (3, 4), (2, 3))):
                hm = float(np.mean(G[:, cm_] <= full[tm]))
                hr = float(np.mean(G[:, cr_] <= full[tr]))
                d = []
                by = defaultdict(list)
                for j, c in enumerate(clip):
                    by[c].append(j)
                for b, dr in enumerate(draws):
                    sel = [j for c in dr for j in by.get(c, ())]
                    if not sel:
                        continue
                    g = G[sel]
                    d.append(float(np.mean(g[:, cr_] <= th[b][tr]) - np.mean(g[:, cm_] <= th[b][tm])))
                d = np.array(d)
                p = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))
                row[which] = {"main": hm, "rc": hr, "delta": hr - hm,
                              "ci": [float(np.quantile(d, .025)), float(np.quantile(d, .975))], "p": p,
                              "significantly_worse": bool(hr < hm and p < 0.05)}
            res[lang][f"{cn}: {dn}"] = row
        worse = {k: v for k, v in res[lang].items()
                 if v["sound"]["significantly_worse"] or v["word"]["significantly_worse"]}
        print(lang, len(res[lang]), "rows;", len(worse), "significantly worse:",
              {k: (round(v["word"]["main"], 3), round(v["word"]["rc"], 3), v["word"]["p"]) for k, v in worse.items()},
              flush=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
