"""Round 19 (encoder adaptation): check that the torch port of the top graph reproduces onnxruntime,
that batching with padding leaves the real frames (almost) unchanged, and time a training step.

    python -m bench.round19_encoder.verify_torch [--device cpu|mps] [--n 8]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def feeds_np(cut):
    """one clip's cached cut [T, 512] -> ORT feeds (batch 1, unpadded)."""
    from bench.round19_encoder.cut import CUT

    c = cut.astype(np.float32)[:, None, :]
    T = len(c)
    return {CUT["in0"]: c[..., :256], CUT["in1"]: c[..., 256:384], CUT["in2"]: c[..., 384:],
            CUT["mask"]: np.zeros((1, 1, (T + 1) // 2), bool)}


def feeds_torch(cuts, device):
    """list of [T_i, 512] -> padded torch feeds [T, N, C] + lengths."""
    import torch

    from bench.round19_encoder.cut import CUT

    T = max(len(c) for c in cuts)
    N = len(cuts)
    x = torch.zeros(T, N, 512)
    for i, c in enumerate(cuts):
        x[:len(c), i] = torch.from_numpy(np.asarray(c, np.float32))
        x[len(c):, i] = x[len(c) - 1, i]   # edge padding: the stride-2 downsampler then sees what it sees unpadded
    lens = torch.tensor([len(c) for c in cuts])
    T2 = (T + 1) // 2
    mask = (torch.arange(0, T, 2)[None, :] >= lens[:, None])[:, None, :T2]
    x = x.to(device)
    return {CUT["in0"]: x[..., :256], CUT["in1"]: x[..., 256:384], CUT["in2"]: x[..., 384:], CUT["mask"]: mask}, lens


def main():
    import pickle

    import torch

    from bench.round19_encoder import cut as C
    from bench.round19_encoder import onnx_torch as OT
    from bench.data.common import cache_root

    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    d = cache_root() / "round19" / "encoder" / "cut" / "speechocean762_train"
    D = pickle.loads(sorted(d.glob("shard*.pkl"))[0].read_bytes())
    ids = list(D)[:a.n]
    cuts = [D[i]["cut"] for i in ids]
    sess = C.session(C.model_dir() / "top.onnx")
    ref = [sess.run(None, feeds_np(c)) for c in cuts]
    g = OT.build(C.model_dir() / "top.onnx", device=a.device)
    res = {"device": a.device}
    err1 = []
    with torch.no_grad():
        for c, r in zip(cuts, ref):
            f, _ = feeds_torch([c], a.device)
            o = g(**f)
            err1.append([float((o[j].float().cpu() - torch.from_numpy(r[j])).abs().max()) for j in range(3)])
    res["unpadded_maxabs_enc5_final_logp"] = np.max(err1, 0).tolist()
    res["ref_absmean_enc5_final"] = [float(np.abs(ref[0][0]).mean()), float(np.abs(ref[0][1]).mean())]
    with torch.no_grad():
        f, lens = feeds_torch(cuts, a.device)
        o = g(**f)
    err2 = []
    for i, r in enumerate(ref):
        n = int(lens[i])
        e5 = (o[0][:n, i].cpu() - torch.from_numpy(r[0][:, 0])).abs()
        fi = (o[1][i, :n].cpu() - torch.from_numpy(r[1][0])).abs()
        err2.append([float(e5.max()), float(e5.mean()), float(fi.max()), float(fi.mean())])
    res["padded_batch_enc5_max_mean_final_max_mean"] = np.max(err2, 0).tolist()
    # LoRA starts at the frozen model; time one fwd+bwd step on the batch
    g2 = OT.build(C.model_dir() / "top.onnx", device=a.device,
                  lora={"rank": 8, "alpha": 8, "match": lambda n: True})
    print("lora matrices", len(g2.lora_of), "params", sum(p.numel() for p in OT.trainable_params(g2)))
    opt = torch.optim.AdamW(OT.trainable_params(g2), 1e-3)
    ts = []
    for _ in range(4):
        t0 = time.perf_counter()
        o = g2(**f)
        loss = o[1].pow(2).mean() + o[0].pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if a.device == "mps":
            torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    res["step_s_lora_batch"] = [round(t, 3) for t in ts]
    res["batch_frames"] = [int(x) for x in lens]
    g3 = OT.build(C.model_dir() / "top.onnx", device=a.device, full=lambda n: True)
    opt = torch.optim.AdamW(OT.trainable_params(g3), 1e-4)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        o = g3(**f)
        loss = o[1].pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if a.device == "mps":
            torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    res["step_s_full_batch"] = [round(t, 3) for t in ts]
    res["full_trainable_params"] = int(sum(p.numel() for p in OT.trainable_params(g3)))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
