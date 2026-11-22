"""Round 20 (Portuguese, time-boxed): does the identity step raise nasal -> oral edits on weak evidence?

    python -m bench.round20_mandarin.pt_nasal collect        # FLEURS VALIDATION audit clips, identity off
    python -m bench.round20_mandarin.pt_nasal diagnose
    python -m bench.round20_mandarin.pt_nasal grid [--save]

Data: the contrast audit's VALIDATION protocol for pt (bench/audit/contrast_audit.py, its saved edit plan,
250 FLEURS validation clips; nothing held out). Each clip is scored on its own text (natives) and on every
planted nasal / oral edit (both directions), identity rule off; per item the rollout row (sounds, front-end
alternatives, pooled vectors, word P, sounds P, presence) is kept so the identity rule is an offline what-if.
Word hit@5 % as the audit: threshold = 5th percentile of the arm's own native word probabilities.
Candidates: (1) no raise for a word with a judged vowel whose evidence is below t
(pronunciada.identity "no_raise_cls"); (2) no raise for a word whose extra-sound evidence (lowest llr_ins
at and between its sounds) is below t ("no_raise_ins_below"; the nasal -> oral edit leaves the coda nasal
in the audio but not in the reference, and the head only judges reference sounds).
Rule (fixed before the grid): the rollout's pt dev guards G1-G5 against off (bench.round19_rollout.fit, on
the stored round-19 pt rows joined to the extra-sound evidence of bench.round20_mandarin.collect), no dev
native row below 0.6 / 0.5 up against the shipped rule, dev fit-fold AUC / FA@80, controlled-error AUC and
the validation near / far / stress / sound-error AUCs not worse than the shipped rule, and the validation
audit nasal -> oral word hit@5 % at least its identity-off value. Pick: the highest audit row.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import zlib
from pathlib import Path

import numpy as np

from bench.round20_mandarin.collect import OUT

LANG = "pt"
D = OUT / LANG
VOWELS = ("a", "e", "ɛ", "i", "o", "ɔ", "u", "aj", "ej", "oj", "ow", "ew", "uj", "ɐ", "ɐ̃")


def cmd_collect():
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "2")
    os.environ["PRONUNCIADA_IDENTITY"] = "0"
    import bench.audit.contrast_audit as CA
    from bench.audit.contrasts import Site, apply
    from bench.round19_rollout.collect import _row as _row0
    from bench.round20_mandarin.collect import extra_row, force_insertion
    from pronunciada import Scorer
    from pronunciada.text import get_g2p

    force_insertion()

    def _row(res, keep_v, bench_words):
        r = _row0(res, keep_v, bench_words)
        r["ins"] = extra_row(res)["ins"]
        return r

    CA.SPLIT = "validation"
    plan = json.loads((CA.cache_dir(LANG) / "plan_250.json").read_text())
    recs = CA.choose(LANG, 250)
    g2p = get_g2p(LANG)
    sc = Scorer(LANG, threads=2)
    D.mkdir(parents=True, exist_ok=True)
    out = D / "audit_val_rows.pkl"
    rows = []
    for k0 in range(0, len(recs), CA.CHUNK):
        for rec, wav in CA._clip_audio(recs[k0:k0 + CA.CHUNK]):  # noqa: SLF001
            try:
                ref = g2p(rec["text"])
                r = _row(sc.score((wav, 16000), ref), True, False)
            except Exception:  # noqa: BLE001
                continue
            r.update(id=rec["id"], kind="native")
            rows.append(r)
            for sd in plan["plan"].get(rec["id"], []):
                site = Site(**sd)
                if site.contrast != "nasal":
                    continue
                rng = np.random.default_rng(zlib.crc32(f"{rec['id']}/{sd}".encode()))
                try:
                    e = apply(ref, site, rng)
                    r2 = _row(sc.score((wav, 16000), e.ref), True, False)
                except Exception:  # noqa: BLE001
                    continue
                r2.update(id=rec["id"], kind="edit", direction=site.direction, edit_word=int(e.word),
                          flag=[int(x) for x in e.flag])
                rows.append(r2)
        print("pt audit val", k0 + CA.CHUNK, len(recs), len(rows), flush=True)
    out.write_bytes(pickle.dumps(rows))


def _rows():
    from bench.round19_rollout.train import Ref
    from pronunciada import identity as ID

    H, cal = ID._head(LANG), ID._cal(LANG)  # noqa: SLF001
    rows = pickle.loads((D / "audit_val_rows.pkl").read_bytes())
    for r in rows:
        s, cls = ID.raw_evidence(Ref(r["ph"]), r.pop("V").astype(np.float32), cal, H)
        r["s"], r["cls"] = s, cls
    return rows


def word_probs(r, rule):
    """Word P of an audit row under rule (None = off), the runtime formula with sound classes."""
    from bench.round19_rollout.fit import pre_prob
    from pronunciada import identity as ID

    pp, pres = pre_prob(r)
    if rule is None:
        return pp * pres
    off = rule["offset"]
    out = []
    for wi in range(len(pp)):
        idx = [i for i, w in enumerate(r["word"]) if w == wi and r["cls"][i] is not None and np.isfinite(r["s"][i])]
        v = [r["s"][i] - off.get(r["cls"][i], 0.0) for i in idx]
        c = [r["cls"][i] for i in idx]
        ins = r.get("ins")
        ins = None if ins is None or not np.isfinite(ins[wi]) else float(ins[wi])
        out.append(ID.apply_word(rule, v, float(pp[wi]), 0.5, sounds=float(r["sounds"][wi]), cls=c, ins=ins)
                   if v else pp[wi])
    return np.array(out) * pres


def audit_rows(rows, rule):
    nat = np.concatenate([word_probs(r, rule) for r in rows if r["kind"] == "native"])
    t5 = np.percentile(nat, 5)
    out = {}
    for d in ("A>B", "B>A"):
        w = np.array([word_probs(r, rule)[r["edit_word"]] for r in rows if r["kind"] == "edit" and r["direction"] == d])
        out[d] = {"n": int(len(w)), "word_hit5": float(np.mean(w <= t5))}
    out["native_below_0.6"] = float(np.mean(nat < 0.6))
    return out


def cmd_diagnose():
    from pronunciada import identity as ID

    rows = _rows()
    cal = ID._cal(LANG)  # noqa: SLF001
    off = cal["offset"]
    for name, rule in (("off", None), ("on", cal)):
        print(name, audit_rows(rows, rule))
    lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))  # noqa: E731
    nat_v = {}
    for r in rows:
        if r["kind"] == "native":
            for x, c in zip(r["s"], r["cls"]):
                if c is not None and np.isfinite(x):
                    nat_v.setdefault(c, []).append(x - off.get(c, 0.0))
    for d in ("A>B", "B>A"):
        E = [r for r in rows if r["kind"] == "edit" and r["direction"] == d]
        dz, sv, cl, smin = [], [], [], []
        for r in E:
            p0, p1 = word_probs(r, None)[r["edit_word"]], word_probs(r, cal)[r["edit_word"]]
            dz.append(lg(p1) - lg(p0))
            i = r["flag"][0]
            c = r["cls"][i]
            sv.append(r["s"][i] - off.get(c, 0.0) if c is not None else np.nan)
            cl.append(c)
            ws = [r["s"][j] - off.get(r["cls"][j], 0.0) for j, w in enumerate(r["word"]) if w == r["edit_word"]
                  and r["cls"][j] is not None and np.isfinite(r["s"][j])]
            smin.append(min(ws) if ws else np.nan)
        dz, sv, smin = map(np.array, (dz, sv, smin))
        print(f"{d}: n {len(E)}, raised {np.mean(dz > 1e-6):.2f}, lowered {np.mean(dz < -1e-6):.2f}, mean dz {dz.mean():+.2f}; "
              f"edited sound s median {np.nanmedian(sv):+.2f}, word min s median {np.nanmedian(smin):+.2f}")
        from collections import Counter
        print("   classes at the edit:", Counter(cl).most_common(8))
        for c, _ in Counter(cl).most_common(5):
            if c in nat_v:
                q = np.quantile(nat_v[c], [0.05, 0.25, 0.5])
                es = sv[[x == c for x in cl]]
                print(f"   {c}: edit s median {np.nanmedian(es):+.2f} | native 5/25/50 % {q.round(2)}")


def cmd_grid(save):
    from bench.round19_rollout import fit as F
    from pronunciada import identity as ID

    from bench.round20_mandarin.collect import load as load_extra

    rows = _rows()
    cal = {k: v for k, v in ID._cal(LANG).items() if k != "round20"}  # noqa: SLF001
    ctx = F.Ctx(LANG)
    for name in ("pool", "dev", "cor", "val"):         # join the word extra-sound evidence (same items)
        ex = load_extra(LANG, {"pool": "cvpool", "cor": "devcor"}.get(name, name))
        for r in getattr(ctx, name):
            x = ex[r["id"]]
            if np.abs(x["prob"] - r["prob"]).max() > 1e-5:
                raise RuntimeError(f"row mismatch {r['id']}")
            r["ins"] = x["ins"]

    def dev_probs(r, rule):
        return word_probs(r, rule)

    b = ctx.measure(lambda r: r["prob"].astype(np.float64))
    on = ctx.measure(lambda r: dev_probs(r, cal))
    a_off, a_on = audit_rows(rows, None), audit_rows(rows, cal)
    print("off audit", a_off, "\non  audit", a_on)
    print(f"on: fit AUC {on['fit_auc']:.4f} FA {100 * on['fit_fa80']:.2f} pool<.6 {100 * on['pool_below_06']:.2f} "
          f"val<.6 {100 * on['val']['native_below_0.6']:.2f} stress {on['val']['stress_auc']:.4f}")
    res = {"off": {"audit": a_off, "m": b}, "on": {"audit": a_on, "m": on}, "grid": []}
    vowels = [c for c in cal["offset"] if c in VOWELS]
    designs = [(f"vowel no-raise below {t:+.1f}", dict(cal, no_raise_cls={c: t for c in vowels}))
               for t in (-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)]
    designs += [(f"extra-sound no-raise below {t:+.1f}", dict(cal, no_raise_ins_below=t))
                for t in (-1.5, -2.0, -2.5, -3.0, -4.0, -5.0, -6.0)]
    for name, rule in designs:
        m = ctx.measure(lambda r, rule=rule: dev_probs(r, rule))
        why = F.guards_ok(m, b, LANG)
        up = [k for k in ("pool_below_06", "pool_below_05", "cvdev_below_06") if m[k] > on[k] + 1e-12]
        up += [k for k in ("native_below_0.6", "swap_native_below_0.6") if m["val"][k] > on["val"][k] + 1e-12]
        worse = [k for k in ("fit_auc", "controlled_auc") if m[k] < on[k] - 1e-4]
        worse += ["fit_fa80"] if m["fit_fa80"] > on["fit_fa80"] + 1e-9 else []
        worse += [f"val {k}" for k in ("swap_near_auc", "swap_far_auc", "stress_auc", "sound_auc")
                  if m["val"][k] < on["val"][k] - 1e-4]
        au = audit_rows(rows, rule)
        ok = not why and not up and not worse and au["A>B"]["word_hit5"] >= a_off["A>B"]["word_hit5"]
        res["grid"].append({"name": name, "rule": {k: v for k, v in rule.items() if k != "offset"}, "fail": why,
                            "natives_up_vs_on": up, "worse_vs_on": worse, "audit": au, "m": m, "ok": ok})
        print(f"{name}: audit nasal->oral {au['A>B']['word_hit5']:.3f} oral->nasal "
              f"{au['B>A']['word_hit5']:.3f} | fit AUC {m['fit_auc']:.4f} FA {100 * m['fit_fa80']:.2f} (on "
              f"{on['fit_auc']:.4f} / {100 * on['fit_fa80']:.2f}) | pool<.6 {100 * m['pool_below_06']:.2f} (on "
              f"{100 * on['pool_below_06']:.2f}) val<.6 {100 * m['val']['native_below_0.6']:.2f} (on "
              f"{100 * on['val']['native_below_0.6']:.2f}) stress {m['val']['stress_auc']:.4f} | {why} up={up} "
              f"worse={worse} {'OK' if ok else ''}")
    ok = [g for g in res["grid"] if g["ok"]]
    pick = max(ok, key=lambda g: (g["audit"]["A>B"]["word_hit5"], -g["m"]["pool_below_06"])) if ok else None
    res["pick"] = pick and pick["name"]
    print("PICK", res["pick"])
    (D / "grid.json").write_text(json.dumps(res, indent=1, default=float))
    if save:
        Path("results/tables/round20_mandarin_pt_nasal_dev.json").write_text(json.dumps(res, indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "diagnose", "grid"])
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args(argv)
    {"collect": cmd_collect, "diagnose": cmd_diagnose, "grid": lambda: cmd_grid(a.save)}[a.cmd]()


if __name__ == "__main__":
    main()
