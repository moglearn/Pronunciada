"""Part 1: warm the shared posterior cache (bench.postcache) for every record of every dataset
plus the FLEURS validation split, model zipa-small-crctc-500k fp32.

    python -m bench.pipeline.warm --dataset all [--shard 0/2] [--threads 3]
    python -m bench.pipeline.warm --report        # hours / files / size -> results/tables/warm.json
"""
from __future__ import annotations

import json
import time

from bench.pipeline.common import (ALL_DATASETS, LANGS, MODEL, PRECISION, iter_parquet_audio,
                                   lp_path, log, records, repo_root)


def warm_dataset(name: str, shard: tuple[int, int], threads: int) -> tuple[int, int]:
    from bench.postcache import get_posteriors, model

    k, n = shard
    done = computed = 0
    if name == "fleurs-validation":
        for lang in LANGS:
            recs = records(name, lang=lang)
            mine = [r for i, r in enumerate(recs) if i % n == k]
            todo = {int(r["audio"].rsplit(":", 1)[1]): r for r in mine if not lp_path(name, r["id"]).exists()}
            done += len(mine)
            if not todo:
                continue
            path = mine[0]["audio"].split(":", 1)[1].rsplit(":", 1)[0]
            model(MODEL, PRECISION, threads=threads)
            for row, wav in iter_parquet_audio(path, set(todo)):
                get_posteriors(name, todo[row]["id"], wav, MODEL, PRECISION)
                computed += 1
        return done, computed
    recs = records(name)
    model(MODEL, PRECISION, threads=threads)
    t0 = time.time()
    for i, r in enumerate(recs):
        if i % n != k:
            continue
        done += 1
        if not lp_path(name, r["id"]).exists():
            get_posteriors(name, r["id"], r["audio"], MODEL, PRECISION)
            computed += 1
            if computed % 1000 == 0:
                log(f"{name}: {computed} computed ({time.time() - t0:.0f}s)")
    return done, computed


def report() -> dict:
    out = {"model": f"{MODEL}-{PRECISION}", "datasets": {}}
    tot_h = tot_b = tot_n = missing = 0
    for name in ALL_DATASETS:
        recs = records(name)
        h = sum(r["dur"] for r in recs) / 3600
        b = 0
        miss = 0
        for r in recs:
            p = lp_path(name, r["id"])
            if p.exists():
                b += p.stat().st_size
            else:
                miss += 1
        out["datasets"][name] = {"n": len(recs), "hours": round(h, 2), "gb": round(b / 1e9, 3),
                                 "missing": miss}
        tot_h += h; tot_b += b; tot_n += len(recs); missing += miss
    out["total"] = {"n": tot_n, "hours": round(tot_h, 2), "gb": round(tot_b / 1e9, 3), "missing": missing}
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        rep = report()
        prev = repo_root() / "results" / "tables" / "warm.json"
        if prev.exists():
            old = json.loads(prev.read_text())
            rep["wall_s"] = old.get("wall_s")
        prev.parent.mkdir(parents=True, exist_ok=True)
        prev.write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))
        return
    k, n = map(int, a.shard.split("/"))
    names = ALL_DATASETS if a.dataset == "all" else a.dataset.split(",")
    # FLEURS (train/test) and the dev split first: the adapter needs them.
    order = sorted(names, key=lambda x: {"fleurs-validation": 0, "fleurs": 1}.get(x, 2))
    t0 = time.time()
    for name in order:
        t = time.time()
        d, c = warm_dataset(name, (k, n), a.threads)
        log(f"{name} shard {k}/{n}: {d} records, {c} computed in {time.time() - t:.0f}s")
    log(f"shard {k}/{n} total wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
