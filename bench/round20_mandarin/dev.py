"""Round 20 (Mandarin): offline what-ifs of the identity word rule on dev data, tone-aware variants.

    python -m bench.round20_mandarin.dev diagnose         # which words the round-19 rule moved
    python -m bench.round20_mandarin.dev grid [--save]    # every design, guards, pick

Data: the round-19 rollout rows of zh (identity evidence s per sound, ev_<set>.pkl, sets cvpool / dev /
devcor / val; nothing held out) joined to the per-word factors of bench.round20_mandarin.collect
(word tone factor T, contrast factor C, sounds S, presence). Word P before the identity rule is
S x T x C x presence (checked: max relative error 2e-16).

Arms: "off" = no identity rule, "r19" = the shipped round-19 rule (data/calib/identity/zh.json), and
the candidate designs (pronunciada.identity.apply_word with the round-20 keys):
  on        prob      step on the word P (round 19)
            sounds    step on S only; T, C and presence stay multiplicative (design a)
            segmental step on S x C; T and presence stay multiplicative
  tone_gate no raise for a word whose tone factor T is below this value (design b)
  tone_ramp (t0, t1): the raise is scaled by clip((T - t0) / (t1 - t0), 0, 1) (designs b / c)
Guards (results/round19_rollout.md section 3, unchanged): G1-G5 against "off". New, fixed before the
grid was run: on FLEURS validation tone swaps (the val set's fleurs-swap protocol, scoreboard
definitions of bench.eval_swap._metrics) AUC, hit@5%FA and right_word not below "off"; the same three on
the dev set's tone items (CV dev + FLEURS validation clips) not below "off".
Pick: among passing designs the lowest mean of the three dev native rows below 0.6 (pool, CV dev,
FLEURS validation clean); ties: higher val tone AUC.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.round19_rollout import fit as F
from bench.round20_mandarin.collect import OUT, load as load_extra
from pronunciada import identity as ID

LANG = "zh"
TABLES = Path("results/tables")


def r19_cal():
    return json.loads(Path("pronunciada/data/calib/identity/zh.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- data

def attach(rows, extra):
    out = []
    for r in rows:
        x = extra.get(r["id"])
        if x is None:
            continue
        if len(x["prob"]) != len(r["prob"]) or np.abs(x["prob"] - r["prob"]).max() > 1e-5:
            raise RuntimeError(f"row mismatch {r['id']}")
        r = dict(r)
        r["T"] = np.where(np.isfinite(x["tone"]), x["tone"], 1.0)
        r["C"] = np.where(np.isfinite(x["contrast"]), x["contrast"], 1.0)
        r["S"] = x["sounds"]
        r["pres"] = np.where(np.isfinite(x["present"]), x["present"], 1.0)
        r["P0"] = x["prob"]
        out.append(r)
    return out


class Ctx(F.Ctx):
    def __init__(self):
        super().__init__(LANG)
        ex = {n: load_extra(LANG, n) for n in ("cvpool", "dev", "devcor", "val")}
        self.pool = attach(self.pool, ex["cvpool"])
        self.dev = attach(self.dev, ex["dev"])
        self.cor = attach(self.cor, ex["devcor"])
        self.val = attach(self.val, ex["val"])
        self.dev_fit = [r for r in self.dev if r["meta"]["fold"] == "fit"]
        self.dev_chk = [r for r in self.dev if r["meta"]["fold"] == "check"]
        self.nat = self.pool + [r for r in self.dev_fit if not r["meta"].get("label")]
        self.cv_nat = [r for r in self.dev if not r["meta"].get("label") and r["meta"]["src"] == "cv"]
        self.cor_fit = [r for r in self.cor if r["meta"]["fold"] == "fit"]
        self.cor_nat = [r for r in self.dev_fit if not r["meta"].get("label") and r["meta"]["src"] == "cv"]


def new_probs(r, cal):
    """Word P of a stored row under rule cal (None = identity off); the runtime formula."""
    if cal is None:
        return r["P0"].astype(np.float64)
    vals = F.word_vals(r, cal["offset"])
    out = []
    for wi, v in enumerate(vals):
        pre = float(r["S"][wi] * r["T"][wi] * r["C"][wi])
        if not len(v):
            out.append(pre * r["pres"][wi])
            continue
        p = ID.apply_word(cal, v, pre, 0.5, sounds=float(r["S"][wi]), tone=float(r["T"][wi]),
                          contrast=float(r["C"][wi]))
        out.append(p * r["pres"][wi])
    return np.array(out)


# ----------------------------------------------------------------------------- tone-swap metrics

def _units(spans, units, p):
    out = []
    for u in units:
        a, b = u["span"]
        ps = [float(q) for (s, e), q in zip(spans, p) if s < b and e > a]
        out.append(min(ps) if ps else 0.0)
    return out


def swap_clusters(rows, probs_of, kind):
    """bench.eval_swap clusters ({utt: {variant: entry}}) of the val swap set (kind "val") or the dev
    set (kind "dev")."""
    cl = defaultdict(dict)
    for r in rows:
        if kind == "val":
            m = r["meta"]
            if m["kind"] != "swap":
                continue
            v, k, units, utt = m["variant"], m["k"], m["units"], m["utt"]
        else:
            lab = r["meta"].get("label") or {}
            v = lab.get("bucket", "orig")
            k, units, utt = lab.get("word_index"), r["units"], r["id"].rsplit("__", 1)[0]
        us = _units(r["spans"], units, probs_of(r))
        e = {"scores": us, "sent": {}}
        if v != "orig":
            e["pos"] = us[k]
            e["neg"] = [s for i, s in enumerate(us) if i != k]
        cl[utt][v] = e
    return [cl[u] for u in sorted(cl)]


def swap_metrics(rows, probs_of, kind):
    from bench.eval_swap import _metrics

    m = _metrics(swap_clusters(rows, probs_of, kind), ("swap_far", "swap_near", "tone"))
    return {k: m[k] for k in ("tone:auc", "tone:hit@5%FA", "tone:right_word", "swap_near:auc", "swap_far:auc",
                              "native:fa<60", "tone:n")}


def measure(ctx, cal):
    f = lambda r: new_probs(r, cal)  # noqa: E731
    m = ctx.measure(f)
    m["valtone"] = swap_metrics(ctx.val, f, "val")
    m["devtone"] = swap_metrics(ctx.dev, f, "dev")
    m["native_mean06"] = float(np.mean([m["pool_below_06"], m["cvdev_below_06"], m["val"]["native_below_0.6"]]))
    return m


TONE_KEYS = ("tone:auc", "tone:hit@5%FA", "tone:right_word")


def guards(m, b):
    why = F.guards_ok(m, b, LANG)
    for s in ("valtone", "devtone"):
        for k in TONE_KEYS:
            if m[s][k] < b[s][k] - 1e-9:
                why.append(f"T {s} {k.split(':')[1]}")
    return why


# ----------------------------------------------------------------------------- designs

def designs(base):
    keep = {k: v for k, v in base.items()}
    yield "r19", keep
    for on in ("sounds", "segmental"):
        yield f"on={on}", dict(keep, on=on)
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        yield f"gate T<{t}", dict(keep, tone_gate=t)
        yield f"segmental+gate T<{t}", dict(keep, on="segmental", tone_gate=t)
    for t0, t1 in itertools.product((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), (0.5, 0.7, 0.9, 0.95, 0.98, 1.0)):
        if t1 > t0 + 0.1:
            yield f"ramp {t0}-{t1}", dict(keep, tone_ramp=[t0, t1])
            yield f"segmental+ramp {t0}-{t1}", dict(keep, on="segmental", tone_ramp=[t0, t1])


def fmt(name, m, b, why):
    vt, dt = m["valtone"], m["devtone"]
    return (f"{name:26s} nat {100 * m['native_mean06']:.2f} (pool {100 * m['pool_below_06']:.2f} cv "
            f"{100 * m['cvdev_below_06']:.2f} val {100 * m['val']['native_below_0.6']:.2f}) | fitFA "
            f"{100 * m['fit_fa80']:.2f} tonebkt {m['fit_buckets']['tone']:.4f} | val tone {vt['tone:auc']:.4f} "
            f"{vt['tone:hit@5%FA']:.3f} {vt['tone:right_word']:.3f} | dev tone {dt['tone:auc']:.4f} "
            f"{dt['tone:hit@5%FA']:.3f} {dt['tone:right_word']:.3f} | {'OK' if not why else ','.join(why)}")


def cmd_grid(save):
    ctx = Ctx()
    base = r19_cal()
    b = measure(ctx, None)
    print(fmt("OFF", b, b, []), flush=True)
    res = [{"name": "off", "m": b, "fail": []}]
    for name, cal in designs(base):
        m = measure(ctx, cal)
        why = guards(m, b)
        res.append({"name": name, "cal": {k: v for k, v in cal.items() if k != "offset"}, "m": m, "fail": why})
        print(fmt(name, m, b, why), flush=True)
    ok = [x for x in res[1:] if not x["fail"]]
    pick = min(ok, key=lambda x: (round(x["m"]["native_mean06"], 6), -x["m"]["valtone"]["tone:auc"])) if ok else None
    print("PICK", pick["name"] if pick else None)
    out = {"base": b, "grid": res, "pick": pick["name"] if pick else None}
    (OUT / LANG / "grid.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
    if save:
        (TABLES / "round20_mandarin_dev_grid.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))


def cmd_report():
    """Dev tables of the frozen candidate (rule file "round20" keys) against off and r19: rollout dev protocol
    (speaker bootstrap, paired against off) and every guard row."""
    ctx = Ctx()
    base = r19_cal()
    new = {**{k: v for k, v in base.items() if k != "round20"}, **base["round20"]}
    out = {"rule": base["round20"]}
    for arm, cal in (("r19", {k: v for k, v in base.items() if k != "round20"}), ("new", new)):
        f = lambda r, cal=cal: new_probs(r, cal)  # noqa: E731
        for name, rows in (("all", ctx.dev), ("check", ctx.dev_chk)):
            x = F._boot_dev(ctx.items, rows, f)
            out[f"{arm}_dev_{name}"] = x
            print(f"{arm} dev {name:5s} AUC {x['auc']['off']:.4f} -> {x['auc']['on']:.4f} d {x['auc']['delta']:+.4f} "
                  f"[{x['auc']['delta_ci'][0]:+.4f}, {x['auc']['delta_ci'][1]:+.4f}] p={x['auc']['p']:.3f} | FA@80 "
                  f"{100 * x['fa80']['off']:.2f} -> {100 * x['fa80']['on']:.2f} p={x['fa80']['p']:.3f}")
        out[f"{arm}_guards"] = measure(ctx, cal)
    out["off_guards"] = measure(ctx, None)
    (TABLES / "round20_mandarin_dev.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))


def cmd_diagnose():
    """Where the round-19 native gain and the tone losses come from (dev and val only)."""
    ctx = Ctx()
    cal = r19_cal()
    rows = {"natives (pool+fit ctl)": ctx.nat, "cv dev natives": ctx.cv_nat}
    for name, rr in rows.items():
        p0 = np.concatenate([r["P0"] for r in rr])
        p1 = np.concatenate([new_probs(r, cal) for r in rr])
        S = np.concatenate([r["S"] for r in rr])
        T = np.concatenate([r["T"] for r in rr])
        C = np.concatenate([r["C"] for r in rr])
        up = (p0 < 0.6) & (p1 >= 0.6)
        dn = (p0 >= 0.6) & (p1 < 0.6)
        print(f"{name}: words {len(p0)}, below .6 off {100 * np.mean(p0 < .6):.2f} r19 {100 * np.mean(p1 < .6):.2f}; "
              f"lifted over .6: {up.sum()}, pushed under: {dn.sum()}")
        for lab, m in (("lifted", up), ("still low", (p0 < .6) & (p1 < .6))):
            if m.sum():
                print(f"   {lab:9s}: S median {np.median(S[m]):.3f}, T median {np.median(T[m]):.3f}, C median "
                      f"{np.median(C[m]):.3f}; lowest factor = S {np.mean(((S <= T) & (S <= C))[m]):.2f} "
                      f"T {np.mean(((T < S) & (T <= C))[m]):.2f} C {np.mean(((C < S) & (C < T))[m]):.2f}")
        for t in (0.3, 0.5, 0.7):
            print(f"   lifted with T < {t}: {np.sum(up & (T < t))}")
    # tone-swap positives
    for kind, rr in (("val", [r for r in ctx.val if r["meta"]["kind"] == "swap" and r["meta"]["variant"] == "tone"]),
                     ("dev", [r for r in ctx.dev if (r["meta"].get("label") or {}).get("bucket") == "tone"])):
        T, d, P0, P1 = [], [], [], []
        for r in rr:
            if kind == "val":
                k, units = r["meta"]["k"], r["meta"]["units"]
            else:
                k, units = r["meta"]["label"]["word_index"], r["units"]
            a, bb = units[k]["span"]
            wi = [i for i, (s, e) in enumerate(r["spans"]) if s < bb and e > a]
            p1 = new_probs(r, cal)
            for i in wi:
                T.append(r["T"][i])
                P0.append(r["P0"][i])
                P1.append(p1[i])
        T, P0, P1 = map(np.array, (T, P0, P1))
        lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / np.clip(1 - x, 1e-6, 1))  # noqa: E731
        dz = lg(P1) - lg(P0)
        print(f"{kind} tone-swapped words: n {len(T)}, raised {np.mean(dz > 1e-6):.2f}, lowered {np.mean(dz < -1e-6):.2f}, "
              f"median T {np.median(T):.3f}; raised with T<.5: {np.mean((dz > 1e-6) & (T < .5)):.2f}; "
              f"mean dz {dz.mean():+.3f}")
    nT = np.concatenate([r["T"] for r in ctx.nat])
    print("native words T quantiles 1/5/10/25 %:", np.round(np.quantile(nT, [.01, .05, .1, .25]), 3))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diagnose", "grid", "report"])
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "diagnose":
        cmd_diagnose()
    elif a.cmd == "report":
        cmd_report()
    else:
        cmd_grid(a.save)


if __name__ == "__main__":
    main()
