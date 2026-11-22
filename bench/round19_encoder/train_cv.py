"""Round 19 (encoder adaptation): does adapting the top ZIPA-small encoder stack (stack 5) to the
expert scores lift the English sequence head beyond the frozen-feature version? speechocean762 TRAIN only.

    python -m bench.round19_encoder.train_cv --name NAME --mode {frozen,lora,full} [--seeds 0,1,2]
        [--e2 8] [--rank 8] [--lr-top 3e-4] [--lr-head 3e-4] [--l2sp 0] [--soup 1] [--device mps]

Head, folds, inputs and metrics are the English engineer's (bench/round19_encoder/en_*.py, copied):
5-fold speaker-grouped, child-balanced folds, seeds 0,1,2; the GOPT-like head "s6_smallreg"
(d 64, 1 layer, ff 128, dropout 0.2, 25 epochs, corr weight 1) over per-sound means of all seven
encoder outputs + the aligner evidence + utterance cues.

Two stages per (seed, fold), paired across modes:
  stage 1  the head on frozen features, exactly the English recipe (cached, CPU; shared by all modes).
  stage 2  E2 more epochs, starting from the stage-1 head, with the per-sound stack-5 and final-encoder
           vectors RECOMPUTED by the top graph (bench/round19_encoder/onnx_torch.py) from the cached cut
           tensors, so the gradient reaches stack 5:
             frozen  stack 5 fixed (control: the same extra epochs, no adaptation)
             lora    low-rank adapters on every weight matrix of stack 5 (attention, feed-forward,
                     convolution in/out, non-linear attention), rank r
             full    all of stack 5 trainable, L2 pull to the released weights (l2sp)
           --soup K: K stage-2 runs with different seeds from the same stage-1 head; their head and
           (merged) stack-5 weights are averaged ("model soup") and the average is evaluated.
Out-of-fold predictions of every head; results in results/tables/round19_encoder_cv.json (one entry per
name, per seed, plus the mean). Resumable: finished folds are kept in a per-name state file.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np

from bench.round19_encoder import en_data as RD
from bench.round19_encoder import en_exp_seq as ES
from bench.round19_encoder import en_seqhead as SH

LAYERS = list(RD.LAYERS)
FROZEN = ["enc0", "enc1", "enc2", "enc3", "enc4"]
S6 = dict(d=64, n_layers=1, ff=128, dropout=0.2)
S6_TRAIN = dict(epochs=25, w_corr=1.0)


def work_dir() -> Path:
    from bench.data.common import cache_root

    d = cache_root() / "round19" / "encoder"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_cuts(name: str) -> dict:
    d = work_dir() / "cut" / name
    out = {}
    for p in sorted(d.glob("shard*.pkl")):
        for k, v in pickle.loads(p.read_bytes()).items():
            out[k] = v["cut"]
    return out


def windows(start, end, T):
    """The per-sound frame windows of bench/round19/extract_en.layer_phone_means."""
    out = []
    for a, b in zip(start, end):
        a, b = int(a), int(b)
        if b <= a:
            a, b = a - 1, a + 1
        a, b = max(a - 1, 0), min(b + 1, T)
        if b <= a:
            a = max(min(a, T - 1), 0)
            b = a + 1
        out.append((a, b))
    return out


def load_items(split="train", cuts_name="speechocean762_train"):
    keep = ["speaker", "labels", "delivery", "f0", "ph_layers", "tab_num", "phones", "phone_word", "words", "dur",
            "start", "end", "n_frames"]
    from bench.calib.common import calib_cache

    U = RD.load(split, keep=keep)
    z = np.load(calib_cache() / "round2_sentence_inputs_clean.npz", allow_pickle=False)
    pos = {u: i for i, u in enumerate(z["utt"])}
    U = [u for u in U if u["id"] in pos]
    XB = z["XB"][[pos[u["id"]] for u in U]]
    items, vocab = ES.build(U, XB, LAYERS, ["xb", "deliv", "f0"])
    cuts = load_cuts(cuts_name)
    for it, u in zip(items, U):
        c = cuts[u["id"]]
        assert len(c) == u["n_frames"], (u["id"], len(c), u["n_frames"])
        it["cut"] = c
        it["win"] = windows(u["start"], u["end"], len(c))
    return items, vocab


# ----------------------------------------------------------------------------- torch pieces

def pool_matrix(items, T, device):
    import torch

    B = len(items)
    N = max(len(it["pid"]) for it in items)
    P = torch.zeros(B, N, T)
    for b, it in enumerate(items):
        for i, (a, e) in enumerate(it["win"]):
            P[b, i, a:e] = 1.0 / (e - a)
    return P.to(device)


def cut_feeds(items, device):
    from bench.round19_encoder.verify_torch import feeds_torch

    return feeds_torch([it["cut"] for it in items], device)[0]


class Model:
    """stage-2 model: the head + the (adaptable) top graph."""

    def __init__(self, head, top, lnorm, device):
        import torch

        self.head, self.top, self.device = head, top, device
        self.budget = 8000    # padded frames (50 Hz) per top-graph call
        self.mu = {k: torch.tensor(lnorm[k][0], dtype=torch.float32, device=device) for k in ("enc5", "final")}
        self.sd = {k: torch.tensor(lnorm[k][1], dtype=torch.float32, device=device) for k in ("enc5", "final")}

    def forward(self, items, b):
        """b: SH.pad_batch(items, LAYERS) moved to device (xl of enc5 / final get replaced)."""
        import torch

        from torch.utils.checkpoint import checkpoint

        from bench.round19_encoder.cut import CUT

        N = max(len(it["pid"]) for it in items)
        names = list(CUT.values())

        def run(P, *feeds):
            enc5, final = self.top(**dict(zip(names, feeds)))[:2]   # [T, b, 256], [b, T, 512]
            return torch.einsum("bnt,tbc->bnc", P, enc5), torch.einsum("bnt,btc->bnc", P, final)

        # micro-batches of at most `budget` padded frames; when a batch is split, each part is
        # recomputed in backward (activation checkpointing) to bound GPU memory
        order = sorted(range(len(items)), key=lambda j: len(items[j]["cut"]))
        parts, cur = [], []
        for j in order:
            if cur and (len(cur) + 1) * len(items[j]["cut"]) > self.budget:
                parts.append(cur)
                cur = []
            cur.append(j)
        parts.append(cur)
        inv = np.argsort([j for p in parts for j in p])
        E5, FI = [], []
        for part in parts:
            mb = [items[j] for j in part]
            f = cut_feeds(mb, self.device)
            P = pool_matrix(mb, f[names[0]].shape[0], self.device)
            feeds = [f[k] for k in names]
            if torch.is_grad_enabled() and len(parts) > 1:
                e5, fi = checkpoint(run, P, *feeds, use_reentrant=False)
            else:
                e5, fi = run(P, *feeds)
            pad = N - e5.shape[1]
            E5.append(torch.nn.functional.pad(e5, (0, 0, 0, pad)))
            FI.append(torch.nn.functional.pad(fi, (0, 0, 0, pad)))
        inv_t = torch.as_tensor(inv, device=self.device)
        e5, fi = torch.cat(E5)[inv_t], torch.cat(FI)[inv_t]
        e5 = (e5 - self.mu["enc5"]) / self.sd["enc5"]
        fi = (fi - self.mu["final"]) / self.sd["final"]
        xl = list(b["xl"][:5]) + [e5, fi]
        return self.head(xl, b["xt"], b["pid"], b["ppos"], b["xu"], b["mask"])


def to_dev(b, device):
    import torch

    out = {}
    for k, v in b.items():
        out[k] = [x.to(device) for x in v] if isinstance(v, list) else v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def losses(s, p, pw, b, w_word=0.5, w_phone=0.5, w_corr=1.0):
    import torch

    loss = ((s - b["ys"]) ** 2).mean()
    if w_corr and len(s) > 4:
        loss = loss + w_corr * SH.corr_loss(s, b["ys"])
    okp = b["mask"] & torch.isfinite(b["yp"])
    if w_phone and okp.any():
        loss = loss + w_phone * ((p[okp] - b["yp"][okp]) ** 2).mean()
    wm = word_means_dev(pw, b["wid"], b["mask"])
    okw = b["mask"] & torch.isfinite(b["yw"])
    if w_word and okw.any():
        loss = loss + w_word * ((wm[okw] - b["yw"][okw]) ** 2).mean()
    return loss


def word_means_dev(pw, wid, mask):
    """SH.word_means on any device."""
    import torch

    B, N = pw.shape
    W = max(int(wid.max().item()) + 1, 1)
    ok = mask & (wid >= 0)
    oh = torch.nn.functional.one_hot(wid.clamp(min=0), W).float() * ok[..., None].float()
    s = torch.einsum("bn,bnw->bw", pw, oh)
    c = oh.sum(1).clamp(min=1.0)
    return torch.einsum("bw,bnw->bn", s / c, oh)


def length_batches(items, bs, rng, pool=8):
    """Random batches of similar length (less padding, bounded GPU memory): shuffle, cut into pools
    of pool*bs clips, sort each pool by length, deal it into batches, shuffle the batches."""
    order = rng.permutation(len(items))
    out = []
    for i in range(0, len(order), pool * bs):
        p = sorted(order[i:i + pool * bs], key=lambda j: len(items[j]["cut"]))
        out += [p[k:k + bs] for k in range(0, len(p), bs)]
    return [out[k] for k in rng.permutation(len(out))]


def stage2(head, items_tr, lnorm, a, seed, device, log):
    """Train head + top for a.e2 epochs; returns (head, top)."""
    import torch

    from bench.round19_encoder import cut as C
    from bench.round19_encoder import onnx_torch as OT

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lora = {"rank": a.rank, "alpha": a.rank, "seed": seed, "match": lambda n: n.startswith("/encoder/5/")} \
        if a.mode == "lora" else None
    full = (lambda n: n.startswith("encoder.encoders.5.") or n.startswith("onnx::MatMul")) if a.mode == "full" else None
    top = OT.build(C.model_dir() / "top.onnx", device=device, lora=lora, full=full)
    if a.mode == "full":
        # only stack-5 weights: the MatMul initializers of the stack (anonymous names) are found via the nodes
        s5 = {ins[1] for n, op, ins, outs, at in top.nodes if op == "MatMul" and n.startswith("/encoder/5/")}
        for name, k in top.pkey.items():
            top.params[k].requires_grad = name.startswith("encoder.encoders.5.") or name in s5
        w0 = {k: p.detach().clone() for k, p in top.params.items() if p.requires_grad}
    head = head.to(device)
    m = Model(head, top, lnorm, device)
    groups = [{"params": list(head.parameters()), "lr": a.lr_head, "weight_decay": 1e-2}]
    tp = OT.trainable_params(top)
    if tp:
        groups.append({"params": tp, "lr": a.lr_top, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    bs = 32
    steps = a.e2 * math.ceil(len(items_tr) / bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps,
                                              pct_start=0.1)
    head.train()
    top.train()
    for ep in range(a.e2):
        tot = 0.0
        for idx in length_batches(items_tr, bs, rng):
            chunk = [items_tr[j] for j in idx]
            b = to_dev(SH.pad_batch(chunk, LAYERS), device)
            s, p, pw = m.forward(chunk, b)
            loss = losses(s, p, pw, b)
            if a.mode == "full" and a.l2sp:
                loss = loss + a.l2sp * sum(((top.params[k] - w0[k]) ** 2).sum() for k in w0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([q for g in groups for q in g["params"]], 1.0)
            opt.step()
            sch.step()
            tot += float(loss.detach()) * len(chunk)
            if str(device).startswith("mps"):
                torch.mps.empty_cache()
        log(f"    stage2 epoch {ep + 1}/{a.e2}: loss {tot / len(items_tr):.4f}")
    head.eval()
    top.eval()
    return head, top


def predict2(head, top, items, lnorm, device, bs=32):
    import torch

    m = Model(head, top, lnorm, device)
    order = sorted(range(len(items)), key=lambda j: len(items[j]["cut"]))
    items = [items[j] for j in order]
    S, P, Wd = [], [], []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            b = to_dev(SH.pad_batch(chunk, LAYERS), device)
            s, p, pw = m.forward(chunk, b)
            wm = word_means_dev(pw, b["wid"], b["mask"])
            if str(device).startswith("mps"):
                torch.mps.empty_cache()
            for k, it in enumerate(chunk):
                n = len(it["pid"])
                S.append(s[k].cpu().numpy())
                P.append(p[k, :n].cpu().numpy())
                Wd.append(wm[k, :n].cpu().numpy())
    inv = np.argsort(order)
    return np.stack(S)[inv], [P[j] for j in inv], [Wd[j] for j in inv]


def soup(heads, tops):
    """Average head state dicts and merged top weights (all runs share the stage-1 start)."""
    import torch

    h = copy.deepcopy(heads[0])
    sd = {k: sum(x.state_dict()[k].float() for x in heads) / len(heads) for k in heads[0].state_dict()}
    h.load_state_dict(sd)
    merged = [t.merged_initializers() for t in tops]
    t0 = tops[0]
    from bench.round19_encoder import cut as C
    from bench.round19_encoder import onnx_torch as OT

    t = OT.build(C.model_dir() / "top.onnx", device=t0.device_)
    with torch.no_grad():
        for name in merged[0]:
            avg = np.mean([mm[name] for mm in merged], 0)
            t.params[t.pkey[name]].copy_(torch.from_numpy(avg))
    t.eval()
    return h, t


def stage1(items, tr, lnorm, vocab, seed, fi, cache):
    """The English recipe on frozen cached features; state cached per (seed, fold)."""
    import torch

    p = cache / f"stage1_s{seed}_f{fi}.pt"
    tabn = SH.Norm(np.concatenate([items[i]["xt_raw"] for i in tr]))
    uttn = SH.Norm(np.stack([items[i]["xu_raw"] for i in tr]))
    itr = ES.apply_norms(items, tr, tabn, uttn, lnorm, LAYERS)
    m = SH.build_module(itr[0]["xt"].shape[1], len(itr[0]["xu"]), len(vocab), LAYERS, **S6)
    if p.exists():
        m.load_state_dict(torch.load(p))
        m.eval()
    else:
        SH.train(m, itr, LAYERS, epochs=S6_TRAIN["epochs"], w_corr=S6_TRAIN["w_corr"], seed=seed * 10 + fi)
        torch.save(m.state_dict(), p)
    return m, tabn, uttn


def main():
    import os

    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.2")
    import torch

    torch.set_num_threads(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["stage1", "frozen", "lora", "full"], required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--folds", default="")
    ap.add_argument("--e2", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr-top", type=float, default=3e-4)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--l2sp", type=float, default=0.0)
    ap.add_argument("--soup", type=int, default=1)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    from bench.calib.common import load_json, save_json

    t0 = time.time()

    def log(s):
        print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)

    items, vocab = load_items()
    lnorm = ES.layer_norms(items, LAYERS)
    spk = np.array([it["spk"] for it in items])
    age = np.array([it["age"] for it in items])
    log(f"loaded {len(items)} utts")
    cache = work_dir() / "ckpt"
    cache.mkdir(exist_ok=True)
    statef = work_dir() / f"cv_state_{a.name}.pkl"
    state = pickle.loads(statef.read_bytes()) if statef.exists() else {}
    device = a.device if a.mode != "stage1" else "cpu"
    for seed in [int(s) for s in a.seeds.split(",")]:
        fl = RD.folds(spk, age, seed)
        sel = [int(f) for f in a.folds.split(",")] if a.folds else range(len(fl))
        for fi in sel:
            if (seed, fi) in state:
                continue
            tr, te = fl[fi]
            h1, tabn, uttn = stage1(items, tr, lnorm, vocab, seed, fi, cache)
            ite = ES.apply_norms(items, te, tabn, uttn, lnorm, LAYERS)
            if a.mode == "stage1":
                s, p, w = SH.predict(h1, ite, LAYERS)
            else:
                itr = ES.apply_norms(items, tr, tabn, uttn, lnorm, LAYERS)
                for it, j in zip(itr, tr):
                    it["cut"], it["win"] = items[j]["cut"], items[j]["win"]
                for it, j in zip(ite, te):
                    it["cut"], it["win"] = items[j]["cut"], items[j]["win"]
                heads, tops = [], []
                for k in range(a.soup):
                    h, t = stage2(copy.deepcopy(h1), itr, lnorm, a, seed * 100 + fi * 10 + k, device, log)
                    heads.append(h)
                    tops.append(t)
                h, t = soup(heads, tops) if a.soup > 1 else (heads[0], tops[0])
                s, p, w = predict2(h, t, ite, lnorm, device)
                if a.mode in ("lora", "full"):
                    torch.save({"head": h.state_dict(), "top": t.merged_initializers()},
                               cache / f"{a.name}_s{seed}_f{fi}.pt")
            state[(seed, fi)] = {"te": te, "S": s, "P": p, "W": w}
            statef.write_bytes(pickle.dumps(state, protocol=5))
            log(f"seed {seed} fold {fi} done")
    R = load_json("round19_encoder_cv.json") or {}
    res = {"args": vars(a), "seeds": {}}
    for seed in sorted({k[0] for k in state}):
        ks = sorted(k for k in state if k[0] == seed)
        if len(ks) < 5:
            continue
        S = np.zeros((len(items), 4))
        P = [None] * len(items)
        W = [None] * len(items)
        for k in ks:
            st = state[k]
            S[st["te"]] = st["S"]
            for j, i in enumerate(st["te"]):
                P[i], W[i] = st["P"][j], st["W"][j]
        r = ES.evaluate(items, S, P, W)
        res["seeds"][seed] = r
        np.savez(work_dir() / f"oof_{a.name}_seed{seed}.npz", ids=np.array([it["id"] for it in items]), S=S * 10)
        log(f"seed {seed}: " + " ".join(f"{k}={v:.4f}" for k, v in r.items()))
    if res["seeds"]:
        ks = list(next(iter(res["seeds"].values())))
        res["mean"] = {k: float(np.mean([r[k] for r in res["seeds"].values()])) for k in ks}
        log("MEAN " + " ".join(f"{k}={v:.4f}" for k, v in res["mean"].items()))
        R[a.name] = res
        save_json("round19_encoder_cv.json", R)
    print(json.dumps(res.get("mean", {})))


if __name__ == "__main__":
    main()
