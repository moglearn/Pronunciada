"""Round 19: train the sequence sentence head on all of speechocean762 TRAIN and export it for
pronunciada.sentence_head (ONNX network + npz of normalisation / vocabulary / column names).

    python -m bench.round19.train_seq_final [--name seqhead_en]

Configuration frozen from CV (bench/round19/exp_seq.py run "s6_smallreg"): every encoder stack,
d 64, 1 transformer layer, ff 128, dropout 0.2, 25 epochs, correlation-loss weight 1.0, utterance
inputs SENT_NUM (out-of-fold) + delivery + F0 summary. Checks that onnxruntime reproduces torch.
"""
from __future__ import annotations

import argparse

import numpy as np

from bench.round19 import data as RD
from bench.round19 import exp_seq as ES
from bench.round19 import seqhead as SH
from pronunciada import delivery as DV
from pronunciada import features as F
from pronunciada.sentence_head import F0_KEYS

CFG = dict(d=64, n_layers=1, ff=128, dropout=0.2, epochs=25, w_corr=1.0)


def export_onnx(m, layers, n_tab, n_utt, path):
    import torch

    class Wrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.m = inner

        def forward(self, *args):
            # the module's forward re-written with shape-free reshapes (nn.MultiheadAttention
            # traces the sequence length as a constant); same weights, batch of one, no padding
            import torch.nn.functional as Fn

            mm = self.m
            xl = list(args[:len(layers)])
            xt, pid, ppos, xu = args[len(layers):]
            wts = torch.softmax(mm.alpha, 0)
            h = sum(wts[i] * mm.lin[i](x) for i, x in enumerate(xl))
            h = h + mm.tab(xt) + mm.ph(pid) + mm.pos(ppos)
            z = torch.cat([mm.q[None], mm.utt(xu)[:, None, :], h], 1)          # [1, T, d]
            for lay in mm.enc.layers:
                a = lay.self_attn
                nh = a.num_heads
                y = lay.norm1(z)
                qkv = Fn.linear(y, a.in_proj_weight, a.in_proj_bias)            # [1, T, 3d]
                q, k, v = qkv.chunk(3, dim=-1)
                dh = q.shape[-1] // nh
                q = q.reshape(1, -1, nh, dh).transpose(1, 2)
                k = k.reshape(1, -1, nh, dh).transpose(1, 2)
                v = v.reshape(1, -1, nh, dh).transpose(1, 2)
                att = torch.softmax(q @ k.transpose(-1, -2) / float(dh) ** 0.5, -1)
                o = (att @ v).transpose(1, 2).reshape(1, -1, nh * dh)
                z = z + a.out_proj(o)
                y = lay.norm2(z)
                z = z + lay.linear2(Fn.relu(lay.linear1(y)))
            z = mm.norm(z)
            return torch.diagonal(mm.head_s(z[:, :4]), dim1=1, dim2=2)

    w = Wrap(m).eval()
    N = 7
    ex = tuple(torch.randn(1, N, SH.LAYER_DIMS[k]) for k in layers) + (
        torch.randn(1, N, n_tab), torch.ones(1, N, dtype=torch.long), torch.ones(1, N, dtype=torch.long),
        torch.randn(1, n_utt))
    names = [f"x_{k}" for k in layers] + ["xt", "pid", "ppos", "xu"]
    dyn = {n: {1: "n"} for n in names if n != "xu"}
    torch.onnx.export(w, ex, str(path), input_names=names, output_names=["sentence"], dynamic_axes=dyn,
                      opset_version=17, dynamo=False)
    return w, ex, names


def main():
    import torch

    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="seqhead_en")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from bench.calib.common import calib_cache, save_json, ship_dir

    layers = list(RD.LAYERS)
    U = RD.load("train", keep=["speaker", "labels", "delivery", "f0", "ph_layers", "tab_num", "tab_names", "phones",
                               "phone_word", "words", "dur"])
    z = np.load(calib_cache() / "round2_sentence_inputs_clean.npz", allow_pickle=False)
    pos = {u: i for i, u in enumerate(z["utt"])}
    U = [u for u in U if u["id"] in pos]
    tab_cols = list(U[0]["tab_names"])
    items, vocab = ES.build(U, z["XB"][[pos[u["id"]] for u in U]], layers, ["xb", "deliv", "f0"])
    lnorm = ES.layer_norms(items, layers)
    tabn = SH.Norm(np.concatenate([it["xt_raw"] for it in items]))
    uttn = SH.Norm(np.stack([it["xu_raw"] for it in items]))
    itr = ES.apply_norms(items, np.arange(len(items)), tabn, uttn, lnorm, layers)
    m = SH.build_module(itr[0]["xt"].shape[1], len(itr[0]["xu"]), len(vocab), layers, d=CFG["d"],
                        n_layers=CFG["n_layers"], ff=CFG["ff"], dropout=CFG["dropout"])
    SH.train(m, itr, layers, epochs=CFG["epochs"], w_corr=CFG["w_corr"], seed=a.seed)
    m.eval()
    dest = ship_dir()
    onnx_path = dest / f"{a.name}.onnx"
    export_onnx(m, layers, itr[0]["xt"].shape[1], len(itr[0]["xu"]), onnx_path)
    utt_cols = list(F.SENT_NUM) + list(DV.ALL) + list(F0_KEYS)
    assert len(utt_cols) == len(itr[0]["xu"])
    arrays = {"layers": np.array(layers), "tab_cols": np.array(tab_cols), "utt_cols": np.array(utt_cols),
              "vocab": np.array(sorted(vocab, key=vocab.get)), **tabn.state("tab"), **uttn.state("utt")}
    for k in layers:
        arrays[f"lay_{k}_mu"] = lnorm[k][0].astype(np.float32)
        arrays[f"lay_{k}_sd"] = lnorm[k][1].astype(np.float32)
    np.savez_compressed(dest / f"{a.name}.npz", **arrays)
    # parity: onnxruntime vs torch on 50 training utterances
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    diffs = []
    for it in itr[:50]:
        b = SH.pad_batch([it], layers)
        with torch.no_grad():
            st, _, _ = m(b["xl"], b["xt"], b["pid"], b["ppos"], b["xu"], b["mask"])
        feeds = {f"x_{k}": b["xl"][i].numpy() for i, k in enumerate(layers)}
        feeds.update(xt=b["xt"].numpy(), pid=b["pid"].numpy(), ppos=b["ppos"].numpy(), xu=b["xu"].numpy())
        so = sess.run(["sentence"], feeds)[0]
        diffs.append(float(np.abs(so - st.numpy()).max()))
    out = {"params": SH.n_params(m), "onnx_kb": onnx_path.stat().st_size / 1024,
           "npz_kb": (dest / f"{a.name}.npz").stat().st_size / 1024, "onnx_vs_torch_max_abs": max(diffs),
           "layer_weights": torch.softmax(m.alpha, 0).detach().numpy().round(4).tolist(), "config": CFG}
    print(out, flush=True)
    save_json("round19_english_seqhead_fit.json", out)


if __name__ == "__main__":
    main()
