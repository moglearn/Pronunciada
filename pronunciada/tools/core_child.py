"""Run score_phones in a fresh process and report what it loaded (used by tests and packaging tools).

    python -m pronunciada.tools.core_child job.json  -> one JSON line on stdout

job.json: {"lang": "es", "items": [{"id": ..., "samples": "<path of a float32 .npy, 16 kHz>",
           "target": {... pronunciada.core target ...}}], "kw": {"precision": "fp32", "threads": 2},
           "rescore": [0.0, 1.0]}
Output: {"results": {id: {"result": to_dict() without timing_ms, "rescore": {s: to_dict()}, "details": ...}},
         "modules": sorted top-level modules loaded, "pronunciada_modules": [...],
         "opened": every file path opened (audit hook), "peak_rss_bytes": ..., "seconds": {...}}
Only the standard library is imported before the audit hook is installed.
"""
from __future__ import annotations

import json
import sys
import time

_OPENED: list[str] = []


def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        p = args[0].decode() if isinstance(args[0], bytes) else args[0]
        _OPENED.append(p)


def _plain(x):
    import numpy as np

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
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and x != x:
        return "NaN"
    return x


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    sys.addaudithook(_hook)
    job = json.loads(open(argv[0], encoding="utf-8").read())
    t0 = time.perf_counter()
    import numpy as np

    from pronunciada.core import get_scorer, score_phones

    kw = job.get("kw") or {}
    sc = get_scorer(job["lang"], **kw)
    t_load = time.perf_counter() - t0
    out, lat = {}, []
    for it in job["items"]:
        wav = np.load(it["samples"])
        t = time.perf_counter()
        r = score_phones(wav, 16000, it["target"], job["lang"], scorer=sc)
        lat.append(time.perf_counter() - t)
        d = r.to_dict()
        d.pop("timing_ms", None)
        rec = {"result": _plain(d)}
        for s in job.get("rescore") or ():
            x = sc.rescore(r, float(s)).to_dict()
            x.pop("timing_ms", None)
            rec.setdefault("rescore", {})[str(s)] = _plain(x)
        if job.get("details"):
            rec["details"] = _plain(sc.details(r))
        out[it["id"]] = rec
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss if sys.platform == "darwin" else rss * 1024          # macOS: bytes, Linux: KiB
    mods = sorted({m.split(".")[0] for m in sys.modules})
    print(json.dumps({"results": out, "modules": mods,
                      "pronunciada_modules": sorted(m for m in sys.modules if m.startswith("pronunciada")),
                      "opened": sorted(set(_OPENED)), "peak_rss_bytes": int(rss),
                      "seconds": {"load": t_load, "per_clip": lat}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
