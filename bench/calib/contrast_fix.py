"""Round 16: contrasts the scorer hides (FLEURS VALIDATION, the contrast audit's own items).

    python -m bench.calib.contrast_fix collect --lang zh,en,ru,pt,ko,ja,de [--tag base]
    python -m bench.calib.contrast_fix alts                     # A: natives' use of the alternatives
    python -m bench.calib.contrast_fix caps                     # B: contrast-specific caps, dev table

collect: the audit's clips and edit plan (bench/audit/contrast_audit.py: 250 clips per language,
plan_250.json, bench/audit/contrasts.apply), scored through the public Scorer; stored per native
sound and per edited (flagged) sound: raw evidence, the top competing token, the shown sound
log-odds. Cached per language and tag under $PRONUNCIADA_CACHE/calib/round16/.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import zlib

import numpy as np

TARGETS = {"en": ("th_s", "dh_z"), "zh": ("aspiration",), "ru": ("hard_soft", "final_devoicing", "voicing"),
           "pt": ("nasal", "mid_vowels"), "ko": ("aspiration",), "ja": ("voicing",),
           "de": ("voicing", "final_devoicing"), "it": ("mid_vowels",)}
FEATS = ("llr_sub", "llr_del", "z_llr_sub", "log_ratio_comp", "peak_post", "n_frames", "comp_mean", "can_mean")


def work_dir():
    from bench.calib.common import calib_cache

    d = calib_cache() / "round16"
    d.mkdir(exist_ok=True)
    return d


def _rows(res, idx=None):
    c = res._pronunciada  # noqa: SLF001
    ref, tab, ev = c["ref"], c["tab"], c["ev"]
    n = len(ref.phones)
    idx = range(n) if idx is None else idx
    out = {k: np.array([float(tab[k][i]) for i in idx], np.float32) for k in FEATS}
    out["comp_tok"] = [tab["comp_tok"][i] for i in idx]
    out["szl"] = np.array([c["s_zl"][i] for i in idx], np.float32)
    out["szs"] = np.array([c["s_zs"][i] for i in idx], np.float32)
    out["ipa"] = [ref.phones[i].ipa for i in idx]
    out["alts"] = [[a for a, _ in ref.phones[i].alts] for i in idx]
    out["alt_index"] = [ev.slots[i].alt_index for i in idx]
    out["pos"] = [i for i in idx]
    out["word"] = [ref.phones[i].word for i in idx]
    return out


def collect(lang, tag="base", threads=2, n_clips=250):
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(threads)
    import warnings

    warnings.simplefilter("ignore")
    from bench.audit.contrast_audit import cache_dir, choose, make_plan
    from bench.audit.contrasts import Site, apply
    from bench.pipeline.common import iter_parquet_audio
    from pronunciada.scorer import Scorer
    from pronunciada.text import get_g2p

    out_p = work_dir() / f"{lang}_{tag}.pkl"
    if out_p.exists():
        return
    recs = choose(lang, n_clips)
    g2p = get_g2p(lang)
    pp = cache_dir(lang) / f"plan_{n_clips}.json"
    plan = json.loads(pp.read_text()) if pp.exists() else make_plan(lang, recs, g2p)
    sc = Scorer(lang, threads=threads)
    if os.environ.get("R16_A"):                  # measured pt nasal coda / it mid vowels
        for attr in ("measured_nasal", "measured_mid"):
            if hasattr(g2p, attr):
                setattr(g2p, attr, True)
        g2p._cache.cache_clear()
    want = set(TARGETS.get(lang, ()))
    data = {"clips": [], "edits": []}
    t0 = time.time()
    by_path = {}
    for r in recs:
        by_path.setdefault(r["audio"].split(":")[1], {})[int(r["audio"].split(":")[2])] = r
    for path, by in by_path.items():
        for i, wav in iter_parquet_audio(path, set(by)):
            rec = by[i]
            try:
                ref = g2p(rec["text"])
                res = sc.score((wav, 16000), ref)
            except Exception:  # noqa: BLE001
                continue
            nat = _rows(res)
            nat["id"] = rec["id"]
            nat["word_prob"] = np.array([w.prob for w in res.words], np.float32)
            data["clips"].append(nat)
            for sd in plan["plan"].get(rec["id"], []):
                if sd["contrast"] not in want:
                    continue
                site = Site(**sd)
                rng = np.random.default_rng(zlib.crc32(f"{rec['id']}/{sd}".encode()))
                try:
                    e = apply(ref, site, rng)
                    r2 = sc.score((wav, 16000), e.ref)
                except Exception:  # noqa: BLE001
                    continue
                row = _rows(r2, e.flag)
                row.update(id=rec["id"], contrast=site.contrast, direction=site.direction, src=site.src,
                           dst=site.dst, kind=site.kind, partner_in_alts=bool(e.partner_in_alts),
                           native_index=site.index, word_prob=float(r2.words[e.word].prob))
                data["edits"].append(row)
    out_p.write_bytes(pickle.dumps(data))
    print(f"{lang}/{tag}: {len(data['clips'])} clips, {len(data['edits'])} edits ({time.time() - t0:.0f}s)", flush=True)


def load(lang, tag="base"):
    return pickle.loads((work_dir() / f"{lang}_{tag}.pkl").read_bytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["collect", "caps", "caps2", "test"])
    ap.add_argument("--lang", default=",".join(TARGETS))
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()
    if a.step == "caps":
        caps(a.tag)
        return
    if a.step == "caps2":
        caps2(a.tag)
        return
    if a.step == "test":
        for lang in a.lang.split(","):
            collect_test(lang, a.tag)
        return
    for lang in a.lang.split(","):
        collect(lang, a.tag)



# ============================================================================ B: contrast caps

VOICED_T = set("bdgvzʒʐʝɣ") | {"dʒ", "dz", "dʑ"}
VOICELESS_T = set("ptkfsʃʂçxθ") | {"tʃ", "ts", "tɕ", "ʈʂ"}
# class -> (languages, hidden direction selector on edits, expected-member test on a native
#           sound, gate: competitor tokens that make the cap apply for an expected symbol)
CLASSES = {
    "aspiration": (("zh", "ko"), lambda e: e["contrast"] == "aspiration" and e["src"].endswith("ʰ"),
                   lambda ipa, lang: ipa in {"zh": ("p", "t", "k", "ts", "ʈʂ", "tɕ"), "ko": ("p", "t", "k", "tʃ")}[lang],
                   lambda ipa: {"ʰ"}),
    "dental_fricative": (("en",), lambda e: e["contrast"] in ("th_s", "dh_z") and e["src"] in ("s", "z"),
                         lambda ipa, lang: ipa in ("θ", "ð"), lambda ipa: {"θ": {"s"}, "ð": {"z"}}[ipa]),
    "palatalisation": (("ru",), lambda e: e["contrast"] == "hard_soft" and e["src"].endswith("ʲ"),
                       lambda ipa, lang: not ipa.endswith("ʲ") and ipa in {"n", "s", "v", "k", "p", "b", "x", "f", "g",
                                                                           "z", "m", "ɭ", "ts", "dz"},
                       lambda ipa: {"ʲ"}),
    "voicing_to_voiceless": (("ja", "de"), lambda e: e["contrast"] == "voicing" and e["src"] in VOICED_T
                             and e["dst"] in VOICELESS_T,
                             lambda ipa, lang: ipa in VOICELESS_T, lambda ipa: VOICED_T),
    "voicing_to_voiced": (("de", "ru"), lambda e: (e["contrast"] == "voicing" and e["src"] in VOICELESS_T
                                                   and e["dst"] in VOICED_T) or e["contrast"] == "final_devoicing",
                          lambda ipa, lang: ipa in VOICED_T, lambda ipa: VOICELESS_T),
}
CAP_FEATS = ("llr_sub", "log_ratio_comp", "z_llr_sub", "peak_post", "comp_mean", "can_mean")
GUARD_PT = 0.005


def _X(d, idx):
    cols = []
    for k in CAP_FEATS:
        v = np.nan_to_num(np.asarray(d[k], np.float64)[idx], nan=0.0, posinf=30, neginf=-30)
        cols.append(np.clip(v, -30, 30))
    return np.column_stack(cols)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def _p(zl, zs):
    from pronunciada.calibrate import blend_logit

    zl, zs = np.asarray(zl, np.float64), np.asarray(zs, np.float64)
    return _sig(blend_logit(zl, np.minimum(zs, zl), 0.5))


def class_rows(cls, lang, tag="base"):
    """(edits: list of (X row, gate, szl, szs)), natives: arrays for the whole language
    (X, gate mask, member mask, szl, szs)."""
    _, sel, member, gate = CLASSES[cls]
    d = load(lang, tag)
    E = []
    for e in d["edits"]:
        if not sel(e):
            continue
        j = int(np.argmin(_p(e["szl"], e["szs"])))           # the flagged sound with the lowest P
        x = _X(e, [j])[0]
        g = e["comp_tok"][j] in gate(e["dst"])
        E.append({"x": x, "gate": g, "szl": float(e["szl"][j]), "szs": float(e["szs"][j]), "clip": e["id"]})
    nat = {"X": [], "gate": [], "member": [], "szl": [], "szs": [], "clip": []}
    for c in d["clips"]:
        n = len(c["ipa"])
        nat["X"].append(_X(c, np.arange(n)))
        nat["member"].append(np.array([member(ip, lang) for ip in c["ipa"]], bool))
        nat["gate"].append(np.array([member(ip, lang) and (c["comp_tok"][i] in gate(c["ipa"][i]))
                                     for i, ip in enumerate(c["ipa"])], bool))
        nat["szl"].append(c["szl"])
        nat["szs"].append(c["szs"])
        nat["clip"].append(np.array([c["id"]] * n))
    nat = {k: np.concatenate(v) for k, v in nat.items()}
    return E, nat


def _gate_keys(cls):
    return {"dental_fricative": ("θ", "ð")}.get(cls, ())


def fit_cap(parts, nat_mask=None, edit_mask=None):
    """Logistic: natives' member sounds (1) vs hidden-direction edits (0), pooled, equal weight per
    language and class side. Returns coef over CAP_FEATS + intercept."""
    from sklearn.linear_model import LogisticRegression

    X, y, w = [], [], []
    for k, (E, nat) in enumerate(parts):
        mm = nat["member"] if nat_mask is None else nat["member"] & nat_mask[k]
        xn = nat["X"][mm]
        xe = np.array([e["x"] for q, e in enumerate(E) if edit_mask is None or edit_mask[k][q]])
        X += [xn, xe]
        y += [np.ones(len(xn)), np.zeros(len(xe))]
        w += [np.full(len(xn), 0.5 / max(len(xn), 1)), np.full(len(xe), 0.5 / max(len(xe), 1))]
    X, y, w = np.concatenate(X), np.concatenate(y), np.concatenate(w) * len(np.concatenate(y))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(C=1.0, max_iter=3000).fit((X - mu) / sd, y, sample_weight=w)
    return {"coef": (m.coef_[0] / sd).tolist(), "b": float(m.intercept_[0] - (m.coef_[0] * mu / sd).sum())}


def apply_cap(model, off, x, gate, zl, zs):
    z2 = x @ np.asarray(model["coef"]) + model["b"] + off
    zl2 = np.where(gate, np.minimum(zl, z2), zl)
    zs2 = np.where(gate, np.minimum(zs, z2), zs)
    return zl2, zs2


def evaluate(E, nat, model=None, off=0.0):
    pn0 = _p(nat["szl"], nat["szs"])
    if model is None:
        pn, pe = pn0, np.array([_p(e["szl"], e["szs"]) for e in E])
    else:
        zl, zs = apply_cap(model, off, nat["X"], nat["gate"], nat["szl"], nat["szs"])
        pn = _p(zl, zs)
        pe = []
        for e in E:
            a, b = apply_cap(model, off, e["x"][None], np.array([e["gate"]]), np.array([e["szl"]]), np.array([e["szs"]]))
            pe.append(float(_p(a, b)[0]))
        pe = np.array(pe)
    thr = np.quantile(pn, 0.05)
    return {"hit5": float(np.mean(pe <= thr)), "native_accept": float(np.mean(pn >= 0.5)),
            "native_accept_drop": float(np.mean(pn0 >= 0.5) - np.mean(pn >= 0.5)),
            "gate_share_edits": float(np.mean([e["gate"] for e in E])), "n": len(E)}


def caps(tag="base"):
    """Dev table: per class and language, today vs the cap (leave-one-language-out where the class
    spans languages), offset grid; chosen offset = the largest hit@5 % with native sound acceptance
    drop <= GUARD_PT in that language."""
    from bench.calib.common import save_json

    res = {}
    for cls, (langs, *_r) in CLASSES.items():
        parts = {l: class_rows(cls, l, tag) for l in langs}
        res[cls] = {}
        for l in langs:
            train = [parts[o] for o in langs if o != l] or [parts[l]]
            cv = "leave-one-language-out" if len(langs) > 1 else "in-language (one language)"
            m = fit_cap(train)
            E, nat = parts[l]
            base = evaluate(E, nat)
            grid = {}
            for off in np.arange(-6, 6.01, 0.5):
                grid[round(float(off), 2)] = evaluate(E, nat, m, off)
            ok = {o: v for o, v in grid.items() if v["native_accept_drop"] <= GUARD_PT}
            best = max(ok, key=lambda o: (ok[o]["hit5"], -o)) if ok else None
            res[cls][l] = {"cv": cv, "today": base, "chosen_offset": best, "at_chosen": ok.get(best),
                           "grid": grid, "model_cv": m}
            print(cls, l, cv, "today hit", round(base["hit5"], 3), "gate", round(base["gate_share_edits"], 2),
                  "-> cap", best, ok.get(best) and {k: round(v, 4) for k, v in ok[best].items()}, flush=True)
        res[cls]["model_all"] = fit_cap(list(parts.values()))
    save_json("round16_caps_dev.json", res)
    return res



def oof_z2(cls, parts, l):
    """Out-of-sample cap scores for language l: leave-one-language-out when the class spans
    languages, else 5-fold grouped by clip. Returns (z2 for natives, z2 for edits)."""
    from sklearn.model_selection import GroupKFold

    langs = CLASSES[cls][0]
    E, nat = parts[l]
    xe = np.array([e["x"] for e in E])
    if len(langs) > 1:
        m = fit_cap([parts[o] for o in langs if o != l])
        return nat["X"] @ np.asarray(m["coef"]) + m["b"], xe @ np.asarray(m["coef"]) + m["b"]
    zn, ze = np.zeros(len(nat["X"])), np.zeros(len(E))
    clips = np.unique(nat["clip"])
    ec = np.array([e["clip"] for e in E])
    for tr, te in GroupKFold(5).split(clips, None, clips):
        trc, tec = set(clips[tr]), set(clips[te])
        nm = np.array([c in trc for c in nat["clip"]])
        em = np.array([c in trc for c in ec])
        m = fit_cap([parts[l]], [nm], [em])
        nte = np.array([c in tec for c in nat["clip"]])
        ete = np.array([c in tec for c in ec])
        zn[nte] = nat["X"][nte] @ np.asarray(m["coef"]) + m["b"]
        ze[ete] = xe[ete] @ np.asarray(m["coef"]) + m["b"]
    return zn, ze


def caps2(tag="base"):
    """Per language: every class of the language at once. Offsets: each class gets an equal share
    of the language's native budget (GUARD_PT / number of classes); the joint drop is reported."""
    from bench.calib.common import save_json

    by_lang = {}
    for cls, (langs, *_r) in CLASSES.items():
        for l in langs:
            by_lang.setdefault(l, []).append(cls)
    res = {"guard": GUARD_PT, "langs": {}, "models": {}}
    for cls, (langs, *_r) in CLASSES.items():
        res["models"][cls] = fit_cap([class_rows(cls, l, tag) for l in langs])
    for l, classes in by_lang.items():
        budget = GUARD_PT / len(classes)
        out = {}
        zl_n = zs_n = None
        for cls in classes:
            parts = {o: class_rows(cls, o, tag) for o in CLASSES[cls][0]}
            E, nat = parts[l]
            zn, ze = oof_z2(cls, parts, l)
            if zl_n is None:
                zl_n, zs_n = nat["szl"].astype(float).copy(), nat["szs"].astype(float).copy()
                p0 = _p(zl_n, zs_n)
            pe0 = np.array([_p(e["szl"], e["szs"]) for e in E])
            gate_e = np.array([e["gate"] for e in E])
            ezl, ezs = np.array([e["szl"] for e in E]), np.array([e["szs"] for e in E])
            best = None
            for off in np.arange(-6, 6.01, 0.25):
                nzl = np.where(nat["gate"], np.minimum(zl_n, zn + off), zl_n)
                nzs = np.where(nat["gate"], np.minimum(zs_n, zn + off), zs_n)
                pn = _p(nzl, nzs)
                drop = float(np.mean(p0 >= 0.5) - np.mean(pn >= 0.5)) - out.get("_used", 0.0)
                pe = _p(np.where(gate_e, np.minimum(ezl, ze + off), ezl), np.where(gate_e, np.minimum(ezs, ze + off), ezs))
                hit = float(np.mean(pe <= np.quantile(pn, 0.05)))
                # ties -> the larger offset (the more lenient cap)
                if drop <= budget + 1e-12 and (best is None or hit >= best[1] - 1e-9):
                    best = (round(float(off), 2), hit, drop, nzl, nzs)
            off, hit, drop, zl_n, zs_n = best
            out["_used"] = out.get("_used", 0.0) + drop
            out[cls] = {"offset": off, "hit5_today": float(np.mean(pe0 <= np.quantile(p0, 0.05))), "hit5_cap": hit,
                        "native_drop": drop, "n": len(E), "gate_share": float(gate_e.mean())}
        pn = _p(zl_n, zs_n)
        out["joint_native_sound_drop"] = float(np.mean(p0 >= 0.5) - np.mean(pn >= 0.5))
        out.pop("_used", None)
        res["langs"][l] = out
        print(l, json.dumps({k: ({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                                 if isinstance(v, dict) else round(v, 4)) for k, v in out.items()}), flush=True)
    save_json("round16_caps2_dev.json", res)
    return res



def planted_guard(caps_path, langs=("zh", "ko", "en", "ru", "ja", "de"), tag="_r16"):
    """Planted-error sound AUC (bench.eval_languages kinds, FLEURS validation) and native sound
    acceptance, before / after the caps (applied offline to the shown log-odds of
    bench/calib/sound2.py collections). Words, near-swap and sentences do not read the shown
    sound log-odds, so they are unchanged by construction."""
    from bench.calib import sound2 as S2
    from bench.calib.common import save_json
    from bench.metrics import roc_auc
    from pronunciada import calibrate as C

    raw = json.loads(open(caps_path).read())
    res = {}
    for lang in langs:
        caps = []
        for cls, off in raw["langs"].get(lang, {}).items():
            m = raw["classes"][cls]
            caps.append({"class": cls, "members": set(m["members"][lang]), "gate": m["gate"],
                         "coef": np.asarray(m["coef"]), "b": float(m["b"]), "offset": float(off)})
        data = S2.load(lang, tag)

        def shown(it, on):
            P = it["P"]
            zl, zs = P["szl"], P["szs"]
            if on:
                t = {k: P[k] for k in C.CAP_FEATS}
                t["phone"], t["comp_tok"] = it["ipa"], P["comp_tok"]
                zl, zs = C.apply_contrast_caps(caps, t, zl, zs)
            return _p(zl, zs)

        out = {}
        for on in (False, True):
            nat, ys, ss = [], [], []
            for d in data.values():
                nat.append(shown(d["items"]["clean"], on))
                for key, it in d["items"].items():
                    if key[4:] in S2.PHONE_KINDS and key.startswith("cor:"):
                        ys.append(it["phone_wrong"])
                        ss.append(shown(it, on))
            nat = np.concatenate(nat)
            ys, ss = np.concatenate(ys).astype(int), np.concatenate(ss)
            out["after" if on else "before"] = {"native_sounds": float(np.mean(nat >= 0.5)),
                                                "planted_sound_auc": float(roc_auc(ys, 1 - ss))}
        res[lang] = out
        print(lang, json.dumps(out), flush=True)
    save_json("round16_planted_guard.json", res)
    return res


def caps3(tag="base", s2tag="_r16", auc_guard=0.003):
    """caps2 plus the planted-error guard: per language, every class gets an equal share of the
    native budget (GUARD_PT, measured on ALL FLEURS-validation natives of bench/calib/sound2.py)
    and of the planted sound-AUC budget (auc_guard); offset = the most lenient one reaching the
    best hit@5 % (out-of-sample cap scores on the audit items) within both shares. A class is
    adopted when its hit@5 % reaches 0.5."""
    from bench.calib import sound2 as S2
    from bench.calib.common import save_json
    from bench.metrics import roc_auc
    from pronunciada import calibrate as C

    by_lang = {}
    for cls, (langs, *_r) in CLASSES.items():
        for l in langs:
            by_lang.setdefault(l, []).append(cls)
    models = {cls: fit_cap([class_rows(cls, l, tag) for l in CLASSES[cls][0]]) for cls in CLASSES}
    res = {"guard_native": GUARD_PT, "guard_planted_auc": auc_guard, "models": models, "langs": {}}
    for l, classes in by_lang.items():
        data = S2.load(l, s2tag)
        # sound2 pools: natives and planted (per item, shown log-odds + cap inputs)
        pools = []
        for d in data.values():
            for key, it in d["items"].items():
                kind = "nat" if key == "clean" else (key[4:] if key.startswith("cor:") else None)
                if kind is None or (kind != "nat" and kind not in S2.PHONE_KINDS):
                    continue
                P = it["P"]
                t = {k: P[k] for k in C.CAP_FEATS}
                t["phone"], t["comp_tok"] = it["ipa"], P["comp_tok"]
                pools.append((kind, t, P["szl"].astype(float), P["szs"].astype(float),
                              it.get("phone_wrong")))

        def guard_metrics(caps):
            nat, ys, ss = [], [], []
            for kind, t, zl, zs, pw in pools:
                a, b = C.apply_contrast_caps(caps, t, zl, zs)
                p = _p(a, b)
                if kind == "nat":
                    nat.append(p)
                else:
                    ys.append(pw)
                    ss.append(p)
            nat = np.concatenate(nat)
            return float(np.mean(nat >= 0.5)), float(roc_auc(np.concatenate(ys).astype(int), 1 - np.concatenate(ss)))

        nat0, auc0 = guard_metrics([])
        adopted, out = [], {}
        k = len(classes)
        for cls in classes:
            parts = {o: class_rows(cls, o, tag) for o in CLASSES[cls][0]}
            E, natA = parts[l]
            zn, ze = oof_z2(cls, parts, l)
            p0 = _p(natA["szl"], natA["szs"])
            gate_e = np.array([e["gate"] for e in E])
            ezl, ezs = np.array([e["szl"] for e in E]), np.array([e["szs"] for e in E])
            pe0 = _p(ezl, ezs)
            m = models[cls]
            best = None
            for off in np.arange(-6, 6.01, 0.25):
                cap = {"class": cls, "members": set(_members(cls, l)), "gate": _gate(cls), "coef": np.asarray(m["coef"]),
                       "b": float(m["b"]), "offset": float(off)}
                nat1, auc1 = guard_metrics(adopted + [cap])
                used_n = nat0 - nat1
                used_a = auc0 - auc1
                if used_n > GUARD_PT * (len(adopted) + 1) / k + 1e-12 or used_a > auc_guard * (len(adopted) + 1) / k + 1e-12:
                    continue
                nzl = np.where(natA["gate"], np.minimum(natA["szl"], zn + off), natA["szl"])
                nzs = np.where(natA["gate"], np.minimum(natA["szs"], zn + off), natA["szs"])
                pn = _p(nzl, nzs)
                pe = _p(np.where(gate_e, np.minimum(ezl, ze + off), ezl), np.where(gate_e, np.minimum(ezs, ze + off), ezs))
                hit = float(np.mean(pe <= np.quantile(pn, 0.05)))
                if best is None or hit >= best[1] - 1e-9:
                    best = (round(float(off), 2), hit, used_n, used_a, cap)
            today = float(np.mean(pe0 <= np.quantile(p0, 0.05)))
            if best is None or best[1] < 0.5:
                out[cls] = {"adopted": False, "hit5_today": today, "hit5_best_within_guards": None if best is None else best[1],
                            "offset": None if best is None else best[0]}
            else:
                adopted.append(best[4])
                out[cls] = {"adopted": True, "offset": best[0], "hit5_today": today, "hit5_cap": best[1],
                            "native_drop_cum": best[2], "planted_auc_drop_cum": best[3]}
        nat1, auc1 = guard_metrics(adopted)
        out["native_sounds"] = [nat0, nat1]
        out["planted_sound_auc"] = [auc0, auc1]
        res["langs"][l] = out
        print(l, json.dumps({kk: ({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}
                                  if isinstance(v, dict) else [round(x, 4) for x in v]) for kk, v in out.items()}), flush=True)
    save_json("round16_caps3_dev.json", res)
    return res


def _members(cls, lang):
    mem = {"aspiration": {"zh": ["p", "t", "k", "ts", "ʈʂ", "tɕ"], "ko": ["p", "t", "k", "tʃ"]},
           "dental_fricative": {"en": ["θ", "ð"]},
           "palatalisation": {"ru": ["n", "s", "v", "k", "p", "b", "x", "f", "g", "z", "m", "ɭ", "ts", "dz"]}}
    if cls in mem:
        return mem[cls][lang]
    return sorted(VOICELESS_T) if cls == "voicing_to_voiceless" else sorted(VOICED_T)


def _gate(cls):
    return {"aspiration": {"*": ["ʰ"]}, "dental_fricative": {"θ": ["s"], "ð": ["z"]}, "palatalisation": {"*": ["ʲ"]},
            "voicing_to_voiceless": {"*": sorted(VOICED_T)}, "voicing_to_voiced": {"*": sorted(VOICELESS_T)}}[cls]


def hits(lang, tag):
    """Per (contrast, direction): sound hit@5 % (threshold from the run's own natives), share the
    front-end accepts (partner listed as alternative at the native position), effective hit =
    flagged AND not accepted, word hit@5 %."""
    d = load(lang, tag)
    pn = np.concatenate([_p(c["szl"], c["szs"]) for c in d["clips"]])
    wn = np.concatenate([c["word_prob"] for c in d["clips"]])
    thr, wthr = np.quantile(pn, 0.05), np.quantile(wn, 0.05)
    out = {}
    for e in d["edits"]:
        k = f"{e['contrast']}: said {e['src']}-type, expected {e['dst']}-type ({e['direction']})"
        key = (e["contrast"], e["direction"])
        o = out.setdefault(key, {"n": 0, "hit": 0, "acc": 0, "eff": 0, "whit": 0, "label": k})
        p = float(np.min(_p(e["szl"], e["szs"])))
        h = p <= thr
        o["n"] += 1
        o["hit"] += h
        o["acc"] += e["partner_in_alts"]
        o["eff"] += h and not e["partner_in_alts"]
        o["whit"] += e["word_prob"] <= wthr
    res = {}
    for (c, dr), o in sorted(out.items()):
        n = o["n"]
        res[f"{c} {dr}"] = {"n": n, "hit5": o["hit"] / n, "accepted": o["acc"] / n, "effective_hit5": o["eff"] / n,
                            "word_hit5": o["whit"] / n}
    res["_native_sound_accept"] = float(np.mean(pn >= 0.5))
    res["_native_word_accept"] = float(np.mean(wn >= 0.5))
    return res


def collect_test(lang, tag, threads=2, n_clips=250):
    """The audit's swap generators on FLEURS TEST (held-out benchmark rows; run once per tag):
    250 clips (numpy default_rng(0) over the sorted test records), the audit's plan rule, only the
    round-16 target contrasts. tag "before": PRONUNCIADA_CONTRAST_CAPS=0 and PRONUNCIADA_PT_NASAL=0
    must be set by the caller; tag "after": the shipped defaults."""
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(threads)
    import warnings

    warnings.simplefilter("ignore")
    from bench.audit.contrast_audit import make_plan
    from bench.audit.contrasts import Site, apply
    from bench.data import load as load_ds
    from pronunciada.scorer import Scorer
    from pronunciada.text import get_g2p

    out_p = work_dir() / f"{lang}_test_{tag}.pkl"
    if out_p.exists():
        return
    recs = sorted(load_ds("fleurs", split="test", lang=lang), key=lambda r: r["id"])
    rng = np.random.default_rng(0)
    recs = [recs[i] for i in sorted(rng.choice(len(recs), min(n_clips, len(recs)), replace=False))]
    g2p = get_g2p(lang)
    pp = work_dir() / f"plan_test_{lang}.json"
    if pp.exists():
        plan = json.loads(pp.read_text())
    else:
        plan = make_plan(lang, recs, g2p)
        pp.write_text(json.dumps(plan, ensure_ascii=False))
    sc = Scorer(lang, threads=threads)
    want = set(TARGETS.get(lang, ()))
    data = {"clips": [], "edits": []}
    for rec in recs:
        try:
            ref = g2p(rec["text"])
            res = sc.score(rec["audio"], ref)
        except Exception:  # noqa: BLE001
            continue
        nat = _rows(res)
        nat["id"] = rec["id"]
        nat["word_prob"] = np.array([w.prob for w in res.words], np.float32)
        data["clips"].append(nat)
        for sd in plan["plan"].get(rec["id"], []):
            if sd["contrast"] not in want:
                continue
            site = Site(**sd)
            rng2 = np.random.default_rng(zlib.crc32(f"{rec['id']}/{sd}".encode()))
            try:
                e = apply(ref, site, rng2)
                r2 = sc.score(rec["audio"], e.ref)
            except Exception:  # noqa: BLE001
                continue
            row = _rows(r2, e.flag)
            row.update(id=rec["id"], contrast=site.contrast, direction=site.direction, src=site.src, dst=site.dst,
                       kind=site.kind, partner_in_alts=bool(e.partner_in_alts), native_index=site.index,
                       word_prob=float(r2.words[e.word].prob))
            data["edits"].append(row)
    out_p.write_bytes(pickle.dumps(data))
    print(f"test {lang}/{tag}: {len(data['clips'])} clips, {len(data['edits'])} edits", flush=True)


if __name__ == "__main__":
    main()
