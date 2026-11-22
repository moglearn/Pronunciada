"""Round 19 (encoder adaptation): cut the ZIPA-small ONNX graph below its last encoder stack.

    python -m bench.round19_encoder.cut [--time]

The Zipformer's six stacks run one after the other at 50 Hz (stack 5 internally at 25 Hz); the
encoder output is concat(stack 5 [256], stack 4 channels 256:384, stack 3 channels 384:512),
then the (trivial) output downsampler and the CTC layer. So the smallest cut above which only
stack 5 and the output layers remain is four tensors:

    in0  /encoder/Slice_5_output_0   stack 4 output, channels 0:256   (input of stack 5)
    in1  /encoder/Slice_6_output_0   stack 4 output, channels 256:384
    in2  /encoder/Slice_7_output_0   stack 3 output, channels 384:512
    mask /encoder/1/encoder/0/self_attn_weights/Unsqueeze_15_output_0   the stride-2 key padding
         mask (bool [N, 1, T2]; computed once from x_lens and shared by stacks 1 and 5)

Writes under $PRONUNCIADA_CACHE/round19/encoder/models/:
  cut.onnx  the full model with the four cut tensors (and stack 5 / final) as extra outputs
            (the shared backbone itself is not changed; this is the same graph with more outputs)
  top.onnx  the part above the cut: inputs in0, in1, in2, mask; outputs enc5, final, log_probs
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

CUT = {
    "in0": "/encoder/Slice_5_output_0",
    "in1": "/encoder/Slice_6_output_0",
    "in2": "/encoder/Slice_7_output_0",
    "mask": "/encoder/1/encoder/0/self_attn_weights/Unsqueeze_15_output_0",
}
OUT = {"enc5": "/encoder/5/out_combiner/Add_output_0", "final": "/Transpose_1_output_0", "log_probs": "log_probs"}
# the cut tensors for the two-stack cut (stack 4 + 5), research only (too large to ship)
CUT2 = {
    "in0": "/encoder/Slice_4_output_0",          # stack 3 output, channels 0:384 (input of stack 4)
    "in2": "/encoder/Slice_7_output_0",
    "mask4": "/encoder/2/encoder/0/self_attn_weights/Unsqueeze_15_output_0",
    "mask": CUT["mask"],
}


def model_dir() -> Path:
    from bench.data.common import cache_root

    d = cache_root() / "round19" / "encoder" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def src_model(precision="fp32") -> Path:
    from bench.data.common import cache_root

    f = "model.onnx" if precision == "fp32" else "model.int8.onnx"
    return cache_root() / "models" / "zipa-small-crctc-500k" / f


def _extract(src_path, dst_path, inputs, outputs):
    """onnx.utils.extract_model, plus the outer-scope initializers that If-subgraphs read
    (the extractor drops them)."""
    import onnx

    onnx.utils.extract_model(str(src_path), str(dst_path), inputs, outputs, check_model=False)
    src = onnx.load(str(src_path))
    m = onnx.load(str(dst_path))
    have = {i.name for i in m.graph.initializer} | {o for n in m.graph.node for o in n.output}
    init = {i.name: i for i in src.graph.initializer}
    consts = {n.output[0]: n for n in src.graph.node if n.op_type == "Constant"}
    need = set()
    for n in m.graph.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                for sn in a.g.node:
                    need.update(i for i in sn.input if (i in init or i in consts) and i not in have)
    for k in sorted(need):
        if k in init:
            m.graph.initializer.append(init[k])
        else:
            m.graph.node.insert(0, consts[k])
    onnx.save(m, str(dst_path))


def build(precision="fp32"):
    import onnx
    from onnx import TensorProto, helper

    m = onnx.load(str(src_model(precision)))
    g = m.graph
    have = {o.name for o in g.output}
    extra = {**CUT, **{k: v for k, v in OUT.items() if k != "log_probs"}}
    for k, name in extra.items():
        if name in have:
            continue
        et = TensorProto.BOOL if k == "mask" else TensorProto.FLOAT
        g.output.append(helper.make_tensor_value_info(name, et, None))
    sfx = "" if precision == "fp32" else ".int8"
    cut_path = model_dir() / f"cut{sfx}.onnx"
    onnx.save(m, str(cut_path))
    top_path = model_dir() / f"top{sfx}.onnx"
    _extract(cut_path, top_path, list(CUT.values()), list(OUT.values()))
    return cut_path, top_path


def build_top2():
    import onnx

    cut_path = model_dir() / "cut.onnx"
    p = model_dir() / "top2.onnx"
    _extract(cut_path, p, list(CUT2.values()), list(OUT.values()))
    return p


def session(path, threads=2):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.enable_mem_pattern = False
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run_cut(sess, x):
    """x: fbank [T, 80] -> dict of cut tensors (batch 1, as the graph produces them) + enc5/final/log_probs."""
    names = list(CUT.values()) + [OUT["enc5"], OUT["final"], "log_probs"]
    r = sess.run(names, {"x": x[None].astype(np.float32), "x_lens": np.array([len(x)], np.int64)})
    keys = list(CUT) + ["enc5", "final", "log_probs"]
    return dict(zip(keys, r))


def timing(threads=4, secs=3.5, reps=30):
    """CPU time of the full model, and of the top part alone, on a `secs` clip."""
    from pronunciada.acoustic import AcousticModel

    am = AcousticModel(precision="fp32", threads=threads)
    rng = np.random.default_rng(0)
    wav = (rng.standard_normal(int(16000 * secs)) * 0.05).astype(np.float32)
    x = am._features(wav)  # noqa: SLF001
    out = {}
    cut = session(model_dir() / "cut.onnx", threads)
    c = run_cut(cut, x)
    for name, path in [("full_fp32", src_model("fp32")), ("full_int8", src_model("int8")),
                       ("top_fp32", model_dir() / "top.onnx"), ("top_int8", model_dir() / "top.int8q.onnx")]:
        if not Path(path).exists():
            continue
        s = session(path, threads)
        if name.startswith("full"):
            feed = {"x": x[None], "x_lens": np.array([len(x)], np.int64)}
        else:
            feed = {CUT[k]: c[k] for k in CUT}
        for _ in range(3):
            s.run(None, feed)
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            s.run(None, feed)
            t.append((time.perf_counter() - t0) * 1e3)
        out[name] = {"ms_median": float(np.median(t)), "ms_p90": float(np.percentile(t, 90)),
                     "MB": os.path.getsize(path) / 1e6}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--top2", action="store_true")
    a = ap.parse_args()
    if not (model_dir() / "top.onnx").exists():
        cut_path, top_path = build()
        print("wrote", cut_path, top_path)
    if a.top2 and not (model_dir() / "top2.onnx").exists():
        print("wrote", build_top2())
    if a.time:
        for k, v in timing().items():
            print(k, v)


if __name__ == "__main__":
    main()
