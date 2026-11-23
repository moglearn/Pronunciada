"""Check the exported folder with the sherpa-onnx Python package: tokens equal the repo's greedy decode.

    python scripts/verify_sherpa_onnx.py --sherpa-python /path/to/venv/bin/python \
        [--dir ~/.cache/pronunciada/opt/mobile/sherpa-onnx-zipa-small] [--langs en,de,ja,ko,zh] [--n 2] \
        [--models model.int8.onnx,model.onnx] [--out report.json]

The repo side (this interpreter) decodes the practice clips (bench/opt/clips.json, FLEURS train)
to 16 kHz float32, computes fbank + backbone log-posteriors with pronunciada.acoustic for the same
file, and takes the collapsed argmax with only the blank (id 0) removed (sherpa-onnx's greedy search
keeps every other id). The sherpa side (`--sherpa-python`, an interpreter where sherpa-onnx is
installed; it needs only sherpa_onnx and numpy) loads the exported model with
OfflineRecognizer.from_zipformer_ctc and decodes the same samples. Reported per clip: token
sequences equal or not (sherpa-onnx prints the word-boundary token ▁ as a space), and the median
ratio of sherpa-onnx timestamps to the true frame start 0.02 j (expected 2.0: sherpa-onnx assumes
subsampling 4, the model subsamples by 2; source: sherpa-onnx/csrc/offline-zipformer-ctc-model.cc,
SubsamplingFactor()).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_DIR = Path("~/.cache/pronunciada/opt/mobile/sherpa-onnx-zipa-small").expanduser()

CHILD = r"""
import json, sys
import numpy as np
import sherpa_onnx
job = json.load(open(sys.argv[1]))
out = {}
for model in job["models"]:
    rec = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(model=job["dir"] + "/" + model,
                                                           tokens=job["dir"] + "/tokens.txt", num_threads=2)
    for c in job["clips"]:
        s = rec.create_stream()
        s.accept_waveform(16000, np.load(c["npy"]).astype(np.float32))
        rec.decode_stream(s)
        r = s.result
        out[model + ":" + c["id"]] = {"tokens": list(r.tokens), "timestamps": [float(x) for x in r.timestamps]}
print(json.dumps({"version": getattr(sherpa_onnx, "__version__", "?"), "out": out}, ensure_ascii=False))
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sherpa-python", required=True)
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--langs", default="en,de,es,ja,ko,zh")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--models", default="model.int8.onnx,model.onnx")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import numpy as np

    from bench.opt.common import practice_clips
    from pronunciada.acoustic import AcousticModel
    from pronunciada.audio import load_audio

    d = Path(a.dir).expanduser()
    models = [m for m in a.models.split(",") if (d / m).exists()]
    clips = practice_clips(a.langs.split(","), n=a.n)
    tmp = Path(tempfile.mkdtemp(prefix="sherpa_verify_"))
    job = {"dir": str(d), "models": models, "clips": []}
    repo = {}
    ams = {}
    for m in models:
        prec = "int8" if "int8" in m else "fp32"
        os.environ["PRONUNCIADA_BACKBONE"] = str(d / m)          # the exported file itself
        ams[m] = AcousticModel(precision=prec, threads=2)
        del os.environ["PRONUNCIADA_BACKBONE"]
    for c in clips:
        wav = load_audio(c["audio"])
        p = tmp / f"{c['id']}.npy"
        np.save(p, wav)
        job["clips"].append({"id": c["id"], "npy": str(p)})
        for m, am in ams.items():
            ids = am.posteriors(wav).argmax(1)
            seq = [(int(ids[t]), t) for t in range(len(ids)) if ids[t] != 0 and (t == 0 or ids[t] != ids[t - 1])]
            repo[m + ":" + c["id"]] = {"tokens": [am.tokens[i] for i, _ in seq], "frames": [t for _, t in seq]}
    (tmp / "job.json").write_text(json.dumps(job), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    r = subprocess.run([a.sherpa_python, "-I", "-c", CHILD, str(tmp / "job.json")], capture_output=True, text=True,
                       env=env, timeout=1200)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        return 2
    sh = json.loads(r.stdout.strip().splitlines()[-1])
    rows = []
    for key, rp in repo.items():
        so = sh["out"][key]
        st = [" " if t == "▁" else t for t in rp["tokens"]]
        ratio = [b / (0.02 * f) for f, b in zip(rp["frames"], so["timestamps"]) if f > 0]
        rows.append({"model": key.split(":")[0], "clip": key.split(":", 1)[1], "repo_tokens": len(st),
                     "sherpa_tokens": len(so["tokens"]), "identical": st == so["tokens"],
                     "timestamp_ratio_median": float(np.median(ratio)) if ratio else None})
    rep = {"sherpa_onnx_version": sh["version"], "dir": str(d), "rows": rows,
           "all_identical": all(x["identical"] for x in rows)}
    for x in rows:
        print(f"{x['model']:18s} {x['clip']:42s} repo {x['repo_tokens']:4d} sherpa {x['sherpa_tokens']:4d} "
              f"identical {x['identical']}  ts ratio {x['timestamp_ratio_median']}")
    print("sherpa-onnx", sh["version"], "all identical:", rep["all_identical"])
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if rep["all_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
