"""Runtime outputs of the round-20 length head on the round-19 dev sets, for a grid of
lower-only settings, in one scoring pass per clip.

    python -m bench.round20_ja_length.devrun SET --head NAME --grid a:q,a:q,... [--shard K --of N] [--tag T]

(a:qL:qS[:cap[:gate[:mode]]] for separate classes; mode 1 = "min"; gate: pronunciada.ja_length20.gated, q = 0 off, cap = largest drop in log-odds; a: slope; q: the native quantile, in %, of the head's match evidence where lowering starts, per
expected class: OUT/knees_<head>.json)

SET: dev (set_dev.jsonl: planted one-word edits), devnat (dev-pool natives), devswap (FLEURS-swap
dev), cvdev (cv-real ja Test A dev half), lentab (the length table on FLEURS VALIDATION: the
planted long / short swaps of bench/length/heldout_runtime.py, same generators and seeds).

Each clip is scored once with PRONUNCIADA_JA_LENGTH20=0 (current main, written as tag "main");
the head's evidence is computed from the same cached activations; then for each grid setting the
length channel's vowel probabilities are lowered (pronunciada.ja_length20.lower) and the result is
re-assembled (Scorer._assemble, the code path of Scorer.rescore), so word probabilities, the word
step and the error flag are exactly what the runtime would give with those parameters. Output:
$PRONUNCIADA_CACHE/round20/jalength/devrun/<SET>_<tag>.jsonl (bench/round19_ja/dump.py records).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from bench.round20_ja_length.common import OUT

R19 = Path.home() / ".cache" / "pronunciada" / "round19" / "japanese"
SETS = {"dev": "set_dev.jsonl", "devnat": "devnat_in.jsonl", "devswap": "devswap_in.jsonl", "cvdev": "cvdev_in.jsonl"}


def install_head(name: str) -> dict:
    """Point pronunciada.ja_length20 at a bench head (OUT/head_<name>.onnx + its config)."""
    import onnxruntime as ort

    from bench.round20_ja_length.train import CFGS, MED_MD
    from pronunciada import ja_length20 as JL

    cfg = CFGS[name]
    P = {"layers": cfg["layers"], "w": cfg["w"], "med_md": MED_MD, "a": 1.0, "k_long": 0.0, "k_short": 0.0,
         "knees": json.loads((OUT / f"knees_{name}.json").read_text())}
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(OUT / f"head_{name}.onnx"), so, providers=["CPUExecutionProvider"])
    JL.params = lambda: P
    JL._session = lambda: sess
    return P


def parse_grid(s: str):
    out = []
    for x in s.split(","):
        v = [float(t) for t in x.split(":")]
        if len(v) == 2:                       # a:q -> the same quantile for both classes, no cap
            v = [v[0], v[1], v[1], 0.0]
        v = v + [0.0] * (6 - len(v))
        v[4], v[5] = int(v[4]), int(v[5])
        out.append(tuple(v))
    return out


def tag_of(a, ql, qs=None, cap=0.0, gate=0, mode=0):
    if qs is None or (qs == ql and not cap and not gate and not mode):
        return f"a{a:g}_q{ql:g}"
    return (f"a{a:g}_L{ql:g}_S{qs:g}" + (f"_c{cap:g}" if cap else "") + (f"_g{gate}" if gate else "")
            + ("_min" if mode else ""))


def knee(cls: str, q):
    """The q % native quantile for one class; q = 0: no lowering for that class."""
    from pronunciada import ja_length20 as JL

    return None if not q else float(JL.params()["knees"][cls][f"{q:g}"])


def knees(q) -> tuple[float, float]:
    """(k_long, k_short): the q % native quantiles of the head's match evidence per expected class
    (out-of-fold on the fit pool natives, bench.round20_ja_length.train knees)."""
    from pronunciada import ja_length20 as JL

    K = JL.params()["knees"]
    return float(K["long"][f"{q:g}"]), float(K["short"][f"{q:g}"])


def variants(sc, res, grid):
    """{tag: re-assembled result} for every grid setting (and "main" = res)."""
    from pronunciada import ja_length20 as JL

    c = res._pronunciada  # noqa: SLF001
    out = {"main": res}
    lres = c.get("length")
    emb = c.get("emb")
    idx = JL.judged_vowels(c["ref"], lres) if lres is not None else []
    if not idx or not emb:
        for g in grid:
            out[tag_of(*g)] = res
        return out, None
    z = JL.head_logits(c["ref"], c["ev"], c["lpa"], emb, idx)
    zh = {i: (float(zi) if lres["expected"][i] == "long" else -float(zi)) for i, zi in zip(idx, z)}
    from pronunciada.ja_length20 import gated
    for g in grid:
        a, ql, qs, cap, gate, mode = g
        kl, ks = knee("long", ql), knee("short", qs)
        prob = list(lres["prob"])
        for i in idx:
            if gated(c["ref"], i, gate) or (gate & 4 and prob[i] < 0.5):
                continue
            prob[i] = JL.lower(prob[i], zh[i], a, kl if lres["expected"][i] == "long" else ks, cap,
                               "min" if mode else "add")
        c2 = dict(c)
        c2["length"] = {**lres, "prob": prob}
        out[tag_of(*g)] = sc._assemble(c2, sc.strictness)  # noqa: SLF001
    return out, zh


def run_set(which: str, head: str, grid, shard: int, of: int, tag: str) -> None:
    import warnings

    warnings.simplefilter("ignore")
    os.environ["PRONUNCIADA_JA_LENGTH20"] = "0"
    from bench.round19_ja.dump import _wav, record
    from pronunciada.scorer import Scorer

    install_head(head)
    rows = [json.loads(x) for x in open(R19 / SETS[which], encoding="utf-8") if x.strip()][shard::of]
    od = OUT / "devrun" / tag
    od.mkdir(parents=True, exist_ok=True)
    tags = ["main"] + [tag_of(*g) for g in grid]
    paths = {t: od / f"{which}_{t}.s{shard:02d}of{of:02d}.jsonl" for t in tags}
    done = set()
    if paths["main"].exists():
        done = {json.loads(x)["id"] for x in open(paths["main"], encoding="utf-8") if x.strip()}
        # keep the files consistent: truncate every file to the clips present in all of them
        common = done
        for t in tags[1:]:
            ids = {json.loads(x)["id"] for x in open(paths[t], encoding="utf-8") if x.strip()} if paths[t].exists() else set()
            common &= ids
        for t in tags:
            if paths[t].exists():
                keep = [x for x in open(paths[t], encoding="utf-8") if x.strip() and json.loads(x)["id"] in common]
                paths[t].write_text("".join(keep), encoding="utf-8")
        done = common
    sc = Scorer("ja", threads=2)
    fh = {t: open(p, "a", encoding="utf-8") for t, p in paths.items()}
    zf = open(od / f"{which}_zh.s{shard:02d}of{of:02d}.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    todo = [r for r in rows if r["id"] not in done]
    for n, r in enumerate(todo):
        try:
            res = sc.score((_wav(r["audio"]), 16000), r["text"])
            V, zh = variants(sc, res, grid)
            recs = {t: {**r, **record(v)} for t, v in V.items()}
        except Exception as e:  # noqa: BLE001
            recs = {t: {**r, "error": f"{type(e).__name__}: {e}"} for t in tags}
            zh = None
        for t in tags:
            fh[t].write(json.dumps(recs[t], ensure_ascii=False) + "\n")
        zf.write(json.dumps({"id": r["id"], "zh": zh and {str(k): v for k, v in zh.items()}}) + "\n")
        if n % 50 == 49:
            for f in fh.values():
                f.flush()
            print(f"{which} s{shard}: {n + 1}/{len(todo)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    for f in fh.values():
        f.close()
    zf.close()
    print(f"done {which} s{shard} {len(todo)} clips {time.time() - t0:.0f}s", file=sys.stderr)


def run_lentab(head: str, grid, shard: int, of: int, tag: str) -> None:
    """The length table (bench/length/heldout_runtime.py) on FLEURS validation, every grid setting."""
    import warnings

    warnings.simplefilter("ignore")
    os.environ["PRONUNCIADA_JA_LENGTH20"] = "0"
    from bench.audit.contrasts import apply, get_contrast, sites
    from bench.length.fit_length import CONTRASTS, PER_DIR
    from bench.pipeline.common import records, stable_seed
    from bench.round19_ja.dump import _wav
    from pronunciada.scorer import Scorer

    install_head(head)
    recs = sorted(records("fleurs-validation", lang="ja"), key=lambda r: r["id"])[shard::of]
    od = OUT / "devrun" / tag
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"lentab.s{shard:02d}of{of:02d}.jsonl"
    done = {json.loads(x)["id"] for x in open(p, encoding="utf-8")} if p.exists() else set()
    sc = Scorer("ja", threads=2)
    t0 = time.time()
    with open(p, "a", encoding="utf-8") as f:
        for n, rec in enumerate(recs):
            if rec["id"] in done:
                continue
            try:
                wav = _wav(rec["audio"])
                res = sc.score((wav, 16000), rec["text"])
            except ValueError:
                continue
            V, _ = variants(sc, res, grid)
            out = {"id": rec["id"], "nat_len": {}, "nat_word": {}, "edits": []}
            for t, v in V.items():
                out["nat_len"][t] = [ph.length_prob for w in v.words for ph in w.phones if ph.length_prob is not None]
                out["nat_word"][t] = [w.prob for w in v.words]
            ref = res._pronunciada["ref"]  # noqa: SLF001
            for cn in CONTRASTS["ja"]:
                st = sites(ref, get_contrast("ja", cn))
                for dr in ("A>B", "B>A"):
                    pool = [s for s in st if s.direction == dr]
                    rng = np.random.default_rng(stable_seed(rec["id"], cn, dr, "length-v1"))
                    for j in rng.permutation(len(pool))[:PER_DIR]:
                        e = apply(ref, pool[j], rng)
                        r2 = sc.score((wav, 16000), e.ref)
                        V2, _ = variants(sc, r2, grid)
                        ed = {"contrast": cn, "direction": dr, "len": {}, "word": {}}
                        for t, v in V2.items():
                            ph = [x for w in v.words for x in w.phones]
                            lv = [ph[i].length_prob for i in e.flag if ph[i].length_prob is not None]
                            ed["len"][t] = min(lv) if lv else None
                            ed["word"][t] = float(v.words[e.word].prob)
                        out["edits"].append(ed)
            f.write(json.dumps(out) + "\n")
            if n % 20 == 19:
                f.flush()
                print(f"lentab s{shard}: {n + 1}/{len(recs)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    print(f"done lentab s{shard} {time.time() - t0:.0f}s", file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=tuple(SETS) + ("lentab",))
    ap.add_argument("--head", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--tag", default="g1")
    a = ap.parse_args(argv)
    grid = parse_grid(a.grid)
    if a.set == "lentab":
        run_lentab(a.head, grid, a.shard, a.of, a.tag)
    else:
        run_set(a.set, a.head, grid, a.shard, a.of, a.tag)


if __name__ == "__main__":
    main()
