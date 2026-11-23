"""End-to-end parity and mobile-readiness numbers: existing pipeline vs the mobile path.

    python scripts/mobile_parity.py [--langs all] [--precisions fp32,int8] [--so762 100] \
        [--bundles DIR] [--backbone-cache DIR] [--out-dir results/tables]

Existing pipeline: Scorer(lang).score(audio file, text) with the full installation (fresh process).
Mobile path: phones_for(text, lang) -> score_phones(16 kHz samples, target, lang) in a fresh process
whose package copy holds only the language bundle (scripts/build_language_bundles.py) and whose
model cache holds only the exported backbone (scripts/export_sherpa_onnx.py --fp32 --scoring; the
scoring variants serve as the extra-output files). Audio decoding / resampling (pronunciada.audio
.load_audio) happens before the mobile process, as on a phone.
Clips: the practice list (bench/opt/clips.json, FLEURS train, 20 per language) and, for English,
--so762 speechocean762 TRAIN clips (sorted by id, numpy default_rng(20261123) permutation, first N).
Compared per clip: the SentenceResult (without timing), rescore at strictness 0 and 1, details().

Readiness (indicative, this laptop, CPU): peak RSS of a fresh mobile-path process scoring 5 clips,
model load time, and median per-clip latency at 1, 2 and 4 intra-op threads.
Writes <out-dir>/mobile_parity.json and <out-dir>/mobile_readiness.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _bundles_mod():
    spec = importlib.util.spec_from_file_location("build_language_bundles", ROOT / "scripts/build_language_bundles.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


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


def compare(a: dict, b: dict) -> dict:
    fa, fb = _flat(a), _flat(b)
    keys = set(fa) | set(fb)
    diff, mx = [], 0.0
    for k in keys:
        x, y = fa.get(k), fb.get(k)
        if x == y:
            continue
        diff.append(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            mx = max(mx, abs(float(x) - float(y)) if math.isfinite(x) and math.isfinite(y) else math.inf)
    return {"values": len(keys), "differing": len(diff), "max_abs_diff": mx, "examples": sorted(diff)[:5]}


def so762_clips(n: int) -> list[dict]:
    import numpy as np

    from bench.data import load

    recs = sorted(load("speechocean762", split="train"), key=lambda r: r["id"])
    order = np.random.default_rng(20261123).permutation(len(recs))[:n]
    return [{"id": recs[i]["id"], "lang": "en", "audio": recs[i]["audio"], "text": recs[i]["text"]} for i in order]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="all")
    ap.add_argument("--precisions", default="fp32,int8")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--so762", type=int, default=100)
    ap.add_argument("--bundles", default=str(Path("~/.cache/pronunciada/opt/mobile/bundles").expanduser()))
    ap.add_argument("--backbone-cache", default=str(Path("~/.cache/pronunciada/opt/mobile/cache_bb").expanduser()))
    ap.add_argument("--threads", default="1,2,4")
    ap.add_argument("--out-dir", default=str(ROOT / "results/tables"))
    a = ap.parse_args(argv)
    import numpy as np

    from bench.opt.common import practice_clips
    from pronunciada.audio import load_audio
    from pronunciada.core import phones_for

    B = _bundles_mod()
    langs = B.LANGS if a.langs == "all" else a.langs.split(",")
    tmp = Path(tempfile.mkdtemp(prefix="mobile_parity_"))
    bundles = Path(a.bundles).expanduser()
    par, ready = {"precisions": {}}, {"threads": [int(t) for t in a.threads.split(",")], "languages": {}}
    for lang in langs:
        bd = bundles / lang / "pronunciada" / "data"
        if not bd.is_dir():
            raise SystemExit(f"no bundle for {lang} at {bd}: run scripts/build_language_bundles.py first")
        pk = B.package_copy(tmp / f"pkg_{lang}", bd)
        clips = [dict(c, set="practice") for c in practice_clips([lang], n=a.n)]
        if lang == "en" and a.so762:
            clips += [dict(c, set="speechocean762-train") for c in so762_clips(a.so762)]
        text_items, core_items = [], []
        for c in clips:
            p = tmp / f"{c['id']}.npy"
            np.save(p, load_audio(c["audio"]))
            text_items.append({"id": c["id"], "audio": c["audio"], "text": c["text"]})
            core_items.append({"id": c["id"], "samples": str(p), "target": phones_for(c["text"], lang)})
        man = json.loads((bundles / lang / "manifest.json").read_text(encoding="utf-8"))
        for prec in a.precisions.split(","):
            kw = {"precision": prec, "threads": 2}
            old = _run(B, ROOT, lang, text_items, tmp, None, kw, mode="text")
            new = _run(B, pk, lang, core_items, tmp, a.backbone_cache, kw)
            rows = {}
            for c in clips:
                rows[c["id"]] = dict(compare(old["results"][c["id"]], new["results"][c["id"]]), set=c["set"])
            sets = sorted({c["set"] for c in clips})
            summ = {}
            for s in sets:
                rs = [r for r in rows.values() if r["set"] == s]
                summ[s] = {"clips": len(rs), "clips_identical": sum(r["differing"] == 0 for r in rs),
                           "values": sum(r["values"] for r in rs), "values_differing": sum(r["differing"] for r in rs),
                           "max_abs_diff": max(r["max_abs_diff"] for r in rs)}
            par["precisions"].setdefault(prec, {})[lang] = {
                "summary": summ, "identical": all(r["differing"] == 0 for r in rows.values()),
                "differing_clips": {k: v for k, v in rows.items() if v["differing"]}}
            print(f"{prec} {lang}: " + "; ".join(f"{s} {v['clips_identical']}/{v['clips']} clips identical, "
                                                 f"max|diff| {v['max_abs_diff']:.3g}" for s, v in summ.items()),
                  flush=True)
        # readiness: fresh process, 5 clips, threads 1 / 2 / 4
        rr = {"bundle_bytes": man["total_bytes"]}
        for prec in a.precisions.split(","):
            from pronunciada.acoustic import PRECISION_FILES

            bb = Path(a.backbone_cache).expanduser() / "models" / "zipa-small-crctc-500k"
            size = (bb / PRECISION_FILES[prec]).stat().st_size + (bb / "tokens.txt").stat().st_size
            rp = {"backbone_bytes": size, "backbone_plus_bundle_bytes": size + man["total_bytes"]}
            for th in ready["threads"]:
                r = _run(B, pk, lang, core_items[:5], tmp, a.backbone_cache, {"precision": prec, "threads": th},
                         rescore=False)
                lat = r["seconds"]["per_clip"]
                rp[f"threads_{th}"] = {"peak_rss_mb": round(r["peak_rss_bytes"] / 2**20, 1),
                                       "load_s": round(r["seconds"]["load"], 3),
                                       "latency_ms_median": round(1e3 * float(np.median(lat[1:] or lat)), 1),
                                       "latency_ms_first": round(1e3 * lat[0], 1),
                                       "clip_seconds": [round(len(np.load(it["samples"])) / 16000, 2)
                                                        for it in core_items[:5]]}
            rr[prec] = rp
            print(f"  readiness {prec} {lang}: " + ", ".join(
                f"{th} thr {rp[f'threads_{th}']['latency_ms_median']} ms / {rp[f'threads_{th}']['peak_rss_mb']} MB"
                for th in ready["threads"]), flush=True)
        ready["languages"][lang] = rr
    meta = {"machine": _machine(), "generated": time.strftime("%Y-%m-%d %H:%M"),
            "command": "python scripts/mobile_parity.py " + " ".join(argv if argv is not None else sys.argv[1:]),
            "note": "indicative numbers from one laptop CPU; not a phone measurement"}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mobile_parity.json").write_text(json.dumps({**meta, **par}, indent=1, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    (out / "mobile_readiness.json").write_text(json.dumps({**meta, **ready}, indent=1) + "\n", encoding="utf-8")
    ok = all(v["identical"] for v in par["precisions"].get("fp32", {}).values())
    return 0 if ok else 1


def _run(B, pkg, lang, items, tmp, cache, kw, mode=None, rescore=True):
    job = tmp / f"job_{lang}_{mode or 'core'}_{kw['precision']}_{kw['threads']}_{abs(hash(str(pkg))) % 10**6}.json"
    body = {"lang": lang, "items": items, "kw": kw}
    if rescore:
        body.update(rescore=[0.0, 1.0], details=True)
    if mode:
        body["mode"] = mode
    job.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    env = B._clean_env({"PYTHONPATH": str(pkg)} | ({"PRONUNCIADA_CACHE": str(Path(cache).expanduser())} if cache else {}))
    r = subprocess.run([sys.executable, "-m", "pronunciada.tools.core_child", str(job)], capture_output=True,
                       text=True, env=env, cwd=tmp, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-4000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def _machine() -> str:
    import platform

    try:
        import onnxruntime as ort

        v = ort.__version__
    except Exception:  # noqa: BLE001
        v = "?"
    return f"{platform.system()} {platform.machine()}, Python {platform.python_version()}, onnxruntime {v}"


if __name__ == "__main__":
    sys.exit(main())
