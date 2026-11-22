"""Round 19 (encoder adaptation): export the adapted stack 5 of each CV fold to ONNX (fp32 and dynamic
int8), recompute the test fold's per-sound stack-5 / final vectors with onnxruntime, and re-score
the fold with the fold's head. Gives the out-of-fold correlation of the deployable int8 piece next to
the fp32 torch numbers of train_cv.

    python -m bench.round19_encoder.int8_cost --name NAME [--seeds 0]
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from bench.round19_encoder import en_data as RD
from bench.round19_encoder import en_exp_seq as ES
from bench.round19_encoder import en_seqhead as SH
from bench.round19_encoder import train_cv as TC


def export_top(merged: dict, path: Path, int8: bool = False) -> Path:
    """top.onnx with the adapted initializers, outputs stack 5 + final only (the CTC layer dropped)."""
    import onnx
    from onnx import numpy_helper

    from bench.round19_encoder import cut as C

    m = onnx.load(str(C.model_dir() / "top.onnx"))
    for i, init in enumerate(m.graph.initializer):
        if init.name in merged:
            m.graph.initializer[i].CopyFrom(numpy_helper.from_array(np.asarray(merged[init.name], np.float32), init.name))
    tmp = Path(tempfile.mkdtemp()) / "full.onnx"
    onnx.save(m, str(tmp))
    fp = path if not int8 else path.with_suffix(".fp32.onnx")
    C._extract(tmp, fp, list(C.CUT.values()), [C.OUT["enc5"], C.OUT["final"]])
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(fp), str(path), weight_type=QuantType.QInt8)
        fp.unlink()
    return path


def ort_phone_vectors(sess, items):
    """per clip {"enc5": [n, 256], "final": [n, 512]} per-sound means over the extract_en windows."""
    from bench.round19_encoder.verify_torch import feeds_np

    out = []
    for it in items:
        e5, fi = sess.run(None, feeds_np(it["cut"]))
        e5, fi = e5[:, 0], fi[0]
        out.append({"enc5": np.stack([e5[a:b].mean(0) for a, b in it["win"]]),
                    "final": np.stack([fi[a:b].mean(0) for a, b in it["win"]])})
    return out


def with_vectors(items_norm, vecs, lnorm):
    out = []
    for it, v in zip(items_norm, vecs):
        it = dict(it)
        it["xl"] = dict(it["xl"])
        for k in ("enc5", "final"):
            it["xl"][k] = ((v[k] - lnorm[k][0]) / lnorm[k][1]).astype(np.float16)
        out.append(it)
    return out


def main():
    import torch

    from bench.calib.common import load_json, save_json
    from bench.round19_encoder import cut as C

    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    items, vocab = TC.load_items()
    lnorm = ES.layer_norms(items, TC.LAYERS)
    spk = np.array([it["spk"] for it in items])
    age = np.array([it["age"] for it in items])
    cache = TC.work_dir() / "ckpt"
    res = {}
    d = Path(tempfile.mkdtemp())
    for seed in [int(s) for s in a.seeds.split(",")]:
        S = {k: np.zeros((len(items), 4)) for k in ("fp32", "int8")}
        P = {k: [None] * len(items) for k in S}
        W = {k: [None] * len(items) for k in S}
        for fi, (tr, te) in enumerate(RD.folds(spk, age, seed)):
            ck = torch.load(cache / f"{a.name}_s{seed}_f{fi}.pt", weights_only=False)
            h1, tabn, uttn = TC.stage1(items, tr, lnorm, vocab, seed, fi, cache)
            h1.load_state_dict({k: v.cpu() for k, v in ck["head"].items()})
            h1.eval()
            ite = ES.apply_norms(items, te, tabn, uttn, lnorm, TC.LAYERS)
            for it, j in zip(ite, te):
                it["cut"], it["win"] = items[j]["cut"], items[j]["win"]
            for kind in ("fp32", "int8"):
                p = export_top(ck["top"], d / f"top_{kind}.onnx", int8=kind == "int8")
                sess = C.session(p)
                s, pp, w = SH.predict(h1, with_vectors(ite, ort_phone_vectors(sess, ite), lnorm), TC.LAYERS)
                S[kind][te] = s
                for k, i in enumerate(te):
                    P[kind][i], W[kind][i] = pp[k], w[k]
            print(f"seed {seed} fold {fi} done", flush=True)
        for kind in S:
            r = ES.evaluate(items, S[kind], P[kind], W[kind])
            res.setdefault(kind, {})[seed] = r
            print(kind, seed, {k: round(v, 4) for k, v in r.items()}, flush=True)
        res["MB_int8"] = (d / "top_int8.onnx").stat().st_size / 1e6
    R = load_json("round19_encoder_int8.json") or {}
    R[a.name] = res
    save_json("round19_encoder_int8.json", R)


if __name__ == "__main__":
    main()
