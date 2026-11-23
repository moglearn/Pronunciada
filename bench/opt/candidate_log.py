"""The candidate log table of docs/optimization_and_mobile_report.md, from the result files.

    python -m bench.opt.candidate_log [--fill]

Reads $PRONUNCIADA_CACHE/opt/results/<name>.{parity,screen,gate}.json, the candidates' build.json,
and the paired latency / RSS tables in results/tables/opt_profile_cand_*.json, opt_rss_cand_*.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import REPO, opt_root

KIND = {"ort-": "a. ONNX Runtime options", "slim-": "b. lossless graph", "ortopt-": "b. lossless graph",
        "prep-": "b. lossless graph", "f16store": "c. float16 storage", "adapters-": "d. adapters",
        "int8x-": "e. int8, MatMuls kept fp32", "nbits4": "e. 4-bit weights"}
BASE_BYTES = {"fp32": 260_267_872, "int8": 70_677_672}


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def rows() -> list[dict]:
    spec = json.loads((Path(__file__).with_name("candidates.json")).read_text())
    R = opt_root() / "results"
    lat, rss = {}, {}
    for p in sorted((REPO / "results" / "tables").glob("opt_profile_cand_*.json")):
        for x in json.loads(p.read_text())["paired"]:
            lat.setdefault(x["b"], {})[f"{x['lang']} t{x['threads']}"] = x
    for p in sorted((REPO / "results" / "tables").glob("opt_rss_cand_*.json")):
        d = json.loads(p.read_text())
        for x in d["results"]:
            rss.setdefault(x["config"], {})[x["threads"]] = x["after_n_mb"]
    base_rss = rss.get("fp32", {})
    out = []
    for name, s in spec.items():
        if name.startswith("_"):
            continue
        par, scr, gate = (_load(R / f"{name}.{k}.json") for k in ("parity", "screen", "gate"))
        b = _load(opt_root() / "candidates" / name / "build.json")
        size = None
        if b and "bytes" in b:
            size = b["bytes"] / 1e6
        elif b and "compact_total" in b:
            size = b["compact_total"] / 1e6
            b["settings"] = f"{b['variant']} storage of all ten adapters, compressed npz (pronunciada/data/adapters: 4.57 MB)"
        if par is None:
            verdict = "not run"
        elif par["identical"]:
            verdict = "exactly identical"
        else:
            pk = "adapted" if name.startswith("adapters-") else "raw"
            verdict = f"not identical ({par['n_clips_any_score_differs']}/{par['n_clips']} clips differ; {pk} posteriors " \
                      f"max {par['max_abs_diff_posteriors'][pk]:.3g})"
        if scr:
            verdict += f"; screen: {scr['verdict'].split(' (')[0]}" + (f" ({len(scr['reasons'])} reasons)" if scr["reasons"] else "")
        if gate:
            verdict += f"; full gate: {gate['verdict']}" + (" by identity" if gate.get("by_identity") else "")
        L = lat.get(name, {})
        lt = "; ".join(f"{k} {v['mean_diff_ms']:+.1f} ms [{v['mean_diff_ci'][0]:+.1f}, {v['mean_diff_ci'][1]:+.1f}]"
                       for k, v in sorted(L.items()))
        r = rss.get(name, {})
        base_rss = rss.get(s.get("precision", "fp32"), {})
        rs ="; ".join(f"t{t} {mb - base_rss.get(t, mb):+.0f} MB" for t, mb in sorted(r.items())) if r else ""
        out.append({"name": name, "kind": next((v for k, v in KIND.items() if name.startswith(k)), "other"),
                    "precision": s.get("precision", "fp32"), "size_mb": size, "verdict": verdict, "latency": lt, "rss": rs,
                    "settings": (b or {}).get("settings") or json.dumps(s.get("ort_options") or s.get("adapter_dir"))})
    return out


def markdown(rs: list[dict]) -> str:
    L = ["| candidate | kind | precision | size MB (default fp32 260.3, int8 70.7; adapters 4.57 in total) | result | "
         "latency vs default, mean | peak RSS vs default |", "|---|---|---|--:|---|---|---|"]
    for r in rs:
        sz = "–" if r["size_mb"] is None else f"{r['size_mb']:.1f}"
        L.append(f"| {r['name']} | {r['kind']} | {r['precision']} | {sz} | {r['verdict']} | {r['latency'] or '–'} | "
                 f"{r['rss'] or '–'} |")
    L += ["", "How each was made:", ""] + [f"- **{r['name']}**: {r['settings']}" for r in rs]
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fill", action="store_true")
    a = ap.parse_args(argv)
    md = markdown(rows())
    if a.fill:
        from .report import DOC, fill

        fill(DOC, "candidates", md)
    print(md)


if __name__ == "__main__":
    main()
