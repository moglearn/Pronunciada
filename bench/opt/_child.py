"""One Scorer in a fresh process, driven over stdin / stdout (used by profile.py and parity.py).

Started with PYTHONPATH = the source tree under test and run by path (runpy), so it imports nothing
from bench/ and the `pronunciada` package is the tree's. Arguments: --lang --precision --threads
[--trace-files] [--exit-after-load]. Protocol: one JSON command per stdin line, one "@@"-prefixed
JSON reply per command on stdout:

    {"op": "score", "id": ..., "audio": path, "text": ..., "dump": dir|null}
        -> {"id", "wall_ms", "timing_ms"} (+ "error"); with "dump", the outputs (SentenceResult
           without timing, rescoring at strictness 0 and 1, details(), coaching(), Azure layout,
           tips) go to <dir>/outputs.jsonl and the raw / adapted log-posteriors to <dir>/post/<n>.npz
    {"op": "rss"}    -> {"maxrss_mb"}
    {"op": "files"}  -> {"files": [...]} files opened since start (--trace-files)
    {"op": "quit"}
The posterior cache is cleared before every score, so each call runs the whole pipeline.
"""
from __future__ import annotations

import time

T_START = time.time()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402

OPENED: set[str] = set()


def _maxrss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def _emit(**kw) -> None:
    sys.stdout.write("@@" + json.dumps(kw) + "\n")
    sys.stdout.flush()


def _trace() -> None:
    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = os.fsdecode(args[0])
            if os.path.isabs(p):
                OPENED.add(p)

    sys.addaudithook(hook)
    import onnxruntime as ort

    real = ort.InferenceSession.__init__

    def init(self, path_or_bytes, *a, **k):
        if isinstance(path_or_bytes, (str, os.PathLike)):
            OPENED.add(os.path.abspath(os.fspath(path_or_bytes)))
        return real(self, path_or_bytes, *a, **k)

    ort.InferenceSession.__init__ = init


def _plain(x):
    import numpy as np

    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float) and not np.isfinite(x):
        return repr(x)
    return x


def _outputs(sc, res) -> dict:
    d = res.to_dict()
    d.pop("timing_ms", None)
    r0, r1 = sc.rescore(res, 0.0), sc.rescore(res, 1.0)
    rec = {"result": d, "s0": [w.prob for w in r0.words] + [r0.prob], "s1": [w.prob for w in r1.words] + [r1.prob],
           "details": sc.details(res), "coaching": res.coaching()}
    try:
        from pronunciada.compat.azure import to_azure

        rec["azure"] = to_azure(res)
    except ImportError:
        pass
    try:
        from pronunciada.tips import tips

        rec["tips"] = tips(res.coaching(), 3, res)
    except ImportError:
        pass
    return _plain(rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--precision", default="int8")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--trace-files", action="store_true")
    ap.add_argument("--exit-after-load", action="store_true")
    a = ap.parse_args()
    if a.trace_files:
        _trace()
    import numpy as np

    import pronunciada
    from pronunciada.scorer import Scorer

    sc = Scorer(a.lang, precision=a.precision, threads=a.threads)
    _emit(event="loaded", t=time.time(), t_start=T_START, maxrss_mb=_maxrss_mb(), src=pronunciada.__file__,
          backbone=getattr(sc.am.session, "_model_path", None), embeddings=bool(sc.am.embeddings))
    if a.exit_after_load:
        return
    n_dump = 0
    for line in sys.stdin:
        cmd = json.loads(line)
        op = cmd["op"]
        if op == "quit":
            _emit(event="bye", maxrss_mb=_maxrss_mb())
            return
        if op == "rss":
            _emit(event="rss", maxrss_mb=_maxrss_mb())
            continue
        if op == "files":
            _emit(event="files", files=sorted(OPENED))
            continue
        if op != "score":
            _emit(event="error", error=f"unknown op {op}")
            continue
        sc._lp_cache.clear()
        t0 = time.perf_counter()
        try:
            res = sc.score(cmd["audio"], cmd["text"])
        except ValueError as e:
            _emit(event="scored", id=cmd["id"], error=str(e), t=time.time())
            continue
        wall = (time.perf_counter() - t0) * 1e3
        out = {"event": "scored", "id": cmd["id"], "wall_ms": wall, "timing_ms": res.timing_ms, "t": time.time()}
        if cmd.get("dump"):
            d = cmd["dump"]
            os.makedirs(os.path.join(d, "post"), exist_ok=True)
            lpa, x, _emb = next(reversed(sc._lp_cache.values()))
            raw = sc.am.posteriors_from_features(x)
            np.savez(os.path.join(d, "post", f"{n_dump:05d}.npz"), raw=raw, adapted=lpa)
            with open(os.path.join(d, "outputs.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": cmd["id"], "n": n_dump, **_outputs(sc, res)}, ensure_ascii=False) + "\n")
            n_dump += 1
        _emit(**out)


if __name__ == "__main__":
    main()
