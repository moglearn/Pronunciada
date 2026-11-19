"""    python bench/speed_same_clips.py en zh ja es fr de it pt ru ko

Time Pronunciada on the same recordings and text Azure was timed on (fleurs-swap, 'orig' requests).
One process, one CPU thread, full (fp32) model, posterior cache cleared per clip. Azure times are the cached
wall-clock times of the original requests; nothing is sent anywhere."""
import gc, glob, json, os, sys, time
import numpy as np
from pronunciada.scorer import Scorer

AZ = os.path.expanduser("~/.cache/pronunciada/azure/fleurs-swap")
MAN = {}
for l in open(os.path.expanduser("~/.cache/pronunciada/data/fleurs-swap/manifest.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    if r["labels"]["variant"] == "orig":
        MAN[r["id"]] = r
out = {}
for lang in sys.argv[1:]:
    az = {}
    for f in glob.glob(f"{AZ}/fleurs-{lang}-*__orig.json"):
        o = json.load(open(f))
        if o.get("latency_s") is not None and o["id"] in MAN:
            az[o["id"]] = (o["latency_s"], o["attempts"])
    ids = sorted(az)
    sc = Scorer(lang, threads=1)
    sc.score(MAN[ids[0]]["audio"], MAN[ids[0]]["text"])
    ms, dur, used = [], [], []
    for i in ids:
        r = MAN[i]
        sc._lp_cache.clear()
        try:
            t = time.perf_counter(); sc.score(r["audio"], r["text"]); ms.append((time.perf_counter() - t) * 1e3)
        except ValueError:
            continue
        dur.append(r["dur"]); used.append(i)
    a = np.array([az[i][0] for i in used])
    out[lang] = {"n": len(used), "ours_median_ms": float(np.median(ms)), "ours_p95_ms": float(np.percentile(ms, 95)),
                 "azure_median_s": float(np.median(a)), "azure_p95_s": float(np.percentile(a, 95)),
                 "azure_retried": int(sum(az[i][1] > 1 for i in used)),
                 "median_audio_s": float(np.median(dur)), "mean_audio_s": float(np.mean(dur))}
    print(lang, json.dumps(out[lang]), flush=True)
    del sc; gc.collect()
