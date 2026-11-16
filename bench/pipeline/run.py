"""Run the whole phase-2 pipeline for some languages (each stage is idempotent):

    python -m bench.pipeline.run --lang all [--workers 6] [--force]

warm (posterior cache) -> train_adapter -> native_stats -> features (FLEURS validation + test;
English learner corpora when en is included) -> analyze (tables + results/phase2_evidence.md).
Re-run after a front-end change: adapters / native tables / features are keyed on the front-end
source hash (bench.pipeline.common.g2p_version) and the shipped adapter / native table.
"""
from __future__ import annotations

import subprocess
import sys

from bench.pipeline.common import base_parser, parse_langs


def sh(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--skip-warm", action="store_true")
    a = ap.parse_args()
    langs = parse_langs(a.lang)
    w = str(a.workers)
    f = ["--force"] if a.force else []
    if not a.skip_warm:
        sh("bench.pipeline.warm", "--threads", str(min(a.workers, 6)))
    for lang in langs:
        sh("bench.pipeline.train_adapter", "--lang", lang, "--workers", w, *f)
        sh("bench.pipeline.native_stats", "--lang", lang, "--workers", w, *f)
        sh("bench.pipeline.features", "--lang", lang, "--dataset", "fleurs-validation,fleurs",
           "--workers", w, *f)
    if "en" in langs:
        sh("bench.pipeline.features", "--lang", "en", "--dataset", "speechocean762,l2arctic,epadb",
           "--workers", w, *f)
        sh("bench.pipeline.analyze", "--lang", ",".join(langs), "--steps", "so762_adapter,calib,errdet",
           "--workers", w)
    sh("bench.pipeline.warm", "--report")
    sh("bench.pipeline.analyze", "--lang", "all", "--steps", "ctl,report", "--workers", w)


if __name__ == "__main__":
    main()
