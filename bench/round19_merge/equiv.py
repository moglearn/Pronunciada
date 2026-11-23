"""Round 19 merge check: dump every output number of the public Scorer on a fixed clip set, and
compare two dumps.

    # run with the code under test first on the path (any checkout of the repository):
    PYTHONPATH=<checkout> python bench/round19_merge/equiv.py dump --lang ko --set fleurs-val \
        --precision fp32 --out ko.jsonl
    python bench/round19_merge/equiv.py compare a.jsonl b.jsonl [--tol 1e-6]

Clip sets (dev / validation data only, never a held-out test):
  fleurs-val   100 FLEURS validation clips (seed 19), each scored twice: with its own text and
               with one word replaced by the word at the same place in the next clip's text
               (exercises the error paths: identity steps, word recalibration, coaching)
  so762-train  100 speechocean762 TRAIN clips (seed 19), English learners, own text
  cv-dev-short (round 20, latency) cv-real DEV-half clips of 2.5-5 s (seed 19), own text
  items:<file> (round 20, false-alarm release check) a fixed list {lang: [[id, audio, text], ...]}, e.g. practice
               items written by bench/round20_fa_rc/practice_items.py
  fleurs-val-cut (round 20, latency) FLEURS validation clips cut at a word boundary near 3.5 s, text
               cut to the same words (bench/round20_latency/make_cut.py writes the cut list)

Per score the dump holds: SentenceResult.to_dict() without timing_ms, the same result rescored
at strictness 0 and 1, Scorer.details(), coaching() and compat.azure.to_azure(). The compare step
flattens both and reports the largest absolute difference and every key whose values differ.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _clips(set_name: str, lang: str, n: int):
    rng = np.random.default_rng(19)
    if set_name == "fleurs-val":
        from bench.pipeline.common import iter_parquet_audio, records

        recs = records("fleurs-validation", lang=lang)
        pick = sorted(rng.choice(len(recs), min(n, len(recs)), replace=False))
        recs = [recs[i] for i in pick]
        by = {int(r["audio"].split(":")[2]): r for r in recs}
        path = recs[0]["audio"].split(":")[1]
        order = [int(r["audio"].split(":")[2]) for r in recs]
        nxt = {order[k]: by[order[(k + 1) % len(order)]]["text"] for k in range(len(order))}
        for i, wav in iter_parquet_audio(path, set(by)):
            r = by[i]
            yield r["id"], wav, r["text"]
            yield r["id"] + "#swap", wav, _swap(r["text"], nxt[i], lang)
    elif set_name == "so762-train":
        import soundfile as sf

        from bench.pipeline.common import cache_root

        man = cache_root() / "data" / "speechocean762" / "manifest.jsonl"
        recs = [json.loads(l) for l in man.read_text().splitlines() if l.strip()]
        recs = [r for r in recs if r["split"] == "train"]
        pick = sorted(rng.choice(len(recs), min(n, len(recs)), replace=False))
        for i in pick:
            r = recs[i]
            audio = r["audio"].replace("/.cache/pronkit/", "/.cache/pronunciada/")
            if not os.path.exists(audio):
                audio = r["audio"]
            wav, sr = sf.read(audio, dtype="float32")
            assert sr == 16000
            yield r["id"], wav, r["text"]
    elif set_name == "cv-dev-short":
        import soundfile as sf

        from bench.cv_build import OUT_DIR, audio_path

        recs = [json.loads(l) for l in (OUT_DIR / f"{lang}.jsonl").read_text(encoding="utf-8").splitlines()]
        recs = [r for r in recs if r["half"] == "dev" and 2.5 <= r["dur"] <= 5.0]
        pick = sorted(rng.choice(len(recs), min(n, len(recs)), replace=False))
        for i in pick:
            r = recs[i]
            wav, sr = sf.read(str(audio_path(r["locale"], r["path"].rsplit(".", 1)[0])), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(1)
            assert sr == 16000, sr
            yield r["id"], wav, r["sentence"]
    elif set_name == "fleurs-val-cut":
        from bench.pipeline.common import cache_root, iter_parquet_audio

        man = cache_root() / "round20" / "latency" / f"fleurs_cut_{lang}.jsonl"
        recs = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines()][:n]
        by = {r["row"]: r for r in recs}
        for i, wav in iter_parquet_audio(recs[0]["parquet"], set(by)):
            r = by[i]
            yield r["id"], wav[:r["cut"]], r["text"]
    elif set_name.startswith("items:"):
        # (round 20, false-alarm release check) a fixed item list: {lang: [[id, audio path, text], ...]}
        from pronunciada.audio import load_audio

        with open(set_name[len("items:"):], encoding="utf-8") as f:
            its = json.load(f).get(lang, [])
        for cid, audio, text in its[:n]:
            yield cid, load_audio(audio), text
    else:
        raise SystemExit(f"unknown set {set_name}")


def _swap(text: str, other: str, lang: str) -> str:
    """Deterministic one-word change: the middle word of `text` becomes the word at the same
    relative place of `other` (two characters for zh / ja, which have no spaces)."""
    if lang in ("zh", "ja"):
        a, b = list(text), list(other)
        i = len(a) // 2
        j = min(len(b) - 2, max(0, int(len(b) * i / max(len(a), 1))))
        rep = b[j:j + 2]
        return "".join(a[:i] + rep + a[i + 2:]) if rep else text
    a, b = text.split(), other.split()
    if len(a) < 2 or not b:
        return text
    i = len(a) // 2
    j = min(len(b) - 1, int(len(b) * i / len(a)))
    if a[i] == b[j]:
        j = (j + 1) % len(b)
    a[i] = b[j]
    return " ".join(a)


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return repr(x)
    return x


def dump(a):
    import pronunciada
    from pronunciada import Scorer
    from pronunciada.compat.azure import to_azure

    root = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
    src = os.path.dirname(os.path.dirname(os.path.abspath(pronunciada.__file__)))
    if root and os.path.realpath(src) != os.path.realpath(root):
        raise SystemExit(f"pronunciada imported from {src}, not from {root}")
    sc = Scorer(a.lang, precision=a.precision, threads=a.threads)
    n_done = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for cid, wav, text in _clips(a.set, a.lang, a.n):
            try:
                res = sc.score((wav, 16000), text)
            except ValueError as e:
                f.write(json.dumps({"id": cid, "error": str(e)}, ensure_ascii=False) + "\n")
                continue
            d = res.to_dict()
            d.pop("timing_ms", None)
            rec = {"id": cid, "text": text, "result": d,
                   "s0": [w.prob for w in sc.rescore(res, 0.0).words] + [sc.rescore(res, 0.0).prob],
                   "s1": [w.prob for w in sc.rescore(res, 1.0).words] + [sc.rescore(res, 1.0).prob],
                   "details": sc.details(res), "coaching": res.coaching(),
                   "azure": to_azure(res)}
            f.write(json.dumps(_plain(rec), ensure_ascii=False) + "\n")
            n_done += 1
    print(json.dumps({"lang": a.lang, "set": a.set, "precision": a.precision, "scored": n_done,
                      "src": src, "out": a.out}))


def _flat(x, pre="", out=None):
    out = {} if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            _flat(v, f"{pre}.{k}" if pre else k, out)
    elif isinstance(x, list):
        out[pre + ".len"] = len(x)
        for i, v in enumerate(x):
            _flat(v, f"{pre}[{i}]", out)
    else:
        out[pre] = x
    return out


def compare(paths, tol: float, quiet: bool = False) -> dict:
    A = [json.loads(l) for l in open(paths[0], encoding="utf-8")]
    B = [json.loads(l) for l in open(paths[1], encoding="utf-8")]
    out = {"a": paths[0], "b": paths[1], "items_a": len(A), "items_b": len(B), "values": 0,
           "max_abs_diff": 0.0, "n_numeric_over_tol": 0, "n_other_diff": 0, "examples": []}
    if [r["id"] for r in A] != [r["id"] for r in B]:
        out["n_other_diff"] += 1
        out["examples"].append("item ids differ")
    for ra, rb in zip(A, B):
        fa, fb = _flat(ra), _flat(rb)
        for k in sorted(set(fa) | set(fb)):
            va, vb = fa.get(k, "<missing>"), fb.get(k, "<missing>")
            out["values"] += 1
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) \
                    and not isinstance(va, bool) and not isinstance(vb, bool):
                d = abs(float(va) - float(vb))
                out["max_abs_diff"] = max(out["max_abs_diff"], d)
                if d > tol:
                    out["n_numeric_over_tol"] += 1
                    if len(out["examples"]) < 12:
                        out["examples"].append(f"{ra['id']} {k}: {va} vs {vb}")
            elif va != vb:
                out["n_other_diff"] += 1
                if len(out["examples"]) < 12:
                    out["examples"].append(f"{ra['id']} {k}: {str(va)[:60]!r} vs {str(vb)[:60]!r}")
    out["identical_within_tol"] = out["n_numeric_over_tol"] == 0 and out["n_other_diff"] == 0
    if not quiet:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dump")
    d.add_argument("--lang", required=True)
    d.add_argument("--set", default="fleurs-val")
    d.add_argument("--n", type=int, default=100)
    d.add_argument("--precision", default="fp32")
    d.add_argument("--threads", type=int, default=2)
    d.add_argument("--out", required=True)
    c = sp.add_parser("compare")
    c.add_argument("paths", nargs=2)
    c.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a)
    else:
        r = compare(a.paths, a.tol)
        sys.exit(0 if r["identical_within_tol"] else 1)


if __name__ == "__main__":
    main()
