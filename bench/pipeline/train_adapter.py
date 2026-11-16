"""Part 2: per-language output adapter.

    python -m bench.pipeline.train_adapter --lang en|all [--kinds linear,ctx,conv] [--force]

For each language: train three adapter sizes with CTC loss (torch, CPU) on native FLEURS-train
speech towards the canonical reference tokens (blank = logaddexp(lp'0, lp'3), as
pronunciada.align.emission_table), early stopping on FLEURS-validation CTC loss; then score raw vs
each adapter with the controlled-error protocol on FLEURS validation (bench.pipeline.evalctl,
kinds phone_sub_near / phone_sub_far / word_swap, 2 errors per utterance, fixed seeds) and pick
the simplest variant whose mean AUC is within noise of the best (paired utterance bootstrap);
ship it only if it beats raw. Writes pronunciada/data/adapters/<lang>.npz (or removes it) and
results/tables/adapter_<lang>.json.
"""
from __future__ import annotations

import json
import shutil
import time

import numpy as np

from bench.pipeline.common import (LANGS, build_ref, g2p_version, load_lp, log, near_silent, parse_langs,
                                   pipe_dir, records, repo_root)

V = 127
FLOOR, SCALE = -20.0, 10.0
KINDS = ("linear", "ctx", "conv")
SIMPLICITY = {"raw": 0, "linear": 1, "ctx": 2, "conv": 3}
HP = {"lr": 1e-3, "weight_decay": 1e-2, "dropout": 0.1, "batch": 8, "max_epochs": 40,
      "patience": 4, "ctx": 2, "conv_k": 5, "conv_h": 128, "seed": 0}


# ----------------------------------------------------------------------------- data

def load_ctc_data(dataset: str, lang: str, split: str | None):
    """[(id, lp float16 [T,V], target ids)] for records with posteriors and a reference."""
    from pronunciada.text.ipa import to_ids

    out = []
    for r in records(dataset, split=split, lang=lang):
        ref = build_ref(lang, r["text"])
        if ref is None:
            continue
        lp = load_lp(dataset, r)
        if lp is None or near_silent(lp, r["dur"]):
            continue
        tgt = np.array([i for p in ref.phones for i in to_ids(p.ipa)], dtype=np.int64)
        out.append((r["id"], lp.astype(np.float16), tgt))
    return out


# ----------------------------------------------------------------------------- model

def make_net(kind: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.drop = nn.Dropout(HP["dropout"])
            if kind == "linear":
                self.lin = nn.Linear(V, V)
                nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)
            elif kind == "ctx":
                self.k = 2 * HP["ctx"] + 1
                self.conv = nn.Conv1d(V, V, self.k)
                nn.init.zeros_(self.conv.weight); nn.init.zeros_(self.conv.bias)
            elif kind == "conv":
                self.k = HP["conv_k"]
                H = HP["conv_h"]
                self.c1 = nn.Conv1d(V, H, self.k)
                self.c2 = nn.Conv1d(H, H, self.k)
                self.out = nn.Linear(H, V)
                nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
            else:
                raise ValueError(kind)

        def _rep(self, h):       # [N, C, T] edge-replicated "same" padding
            c = self.k // 2
            return F.pad(h, (c, c), mode="replicate")

        def forward(self, lp):   # lp [N, T, V] -> adapted log-probs [N, T, V]
            x = self.drop(torch.clamp(lp, min=FLOOR) / SCALE)
            if self.kind == "linear":
                f = self.lin(x)
            elif self.kind == "ctx":
                f = self.conv(self._rep(x.transpose(1, 2))).transpose(1, 2)
            else:
                h = torch.relu(self.c1(self._rep(x.transpose(1, 2))))
                h = self.drop(h)
                h = torch.relu(self.c2(self._rep(h)))
                h = self.drop(h)
                f = self.out(h.transpose(1, 2))
            return torch.log_softmax(lp + f, dim=-1)

    return Net()


def export(net) -> dict:
    """torch Net -> numpy params of pronunciada.adapter.Adapter."""
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    out = {"kind": np.array(net.kind), "floor": np.array(FLOOR), "scale": np.array(SCALE)}
    if net.kind == "linear":
        out["W"] = sd["lin.weight"].T.copy(); out["b"] = sd["lin.bias"]
    elif net.kind == "ctx":
        out["W"] = np.transpose(sd["conv.weight"], (2, 1, 0)).copy(); out["b"] = sd["conv.bias"]
    else:
        out["W1"] = np.transpose(sd["c1.weight"], (2, 1, 0)).copy(); out["b1"] = sd["c1.bias"]
        out["W2"] = np.transpose(sd["c2.weight"], (2, 1, 0)).copy(); out["b2"] = sd["c2.bias"]
        out["Wo"] = sd["out.weight"].T.copy(); out["bo"] = sd["out.bias"]
    return out


def _batches(data, bs, rng=None):
    order = np.argsort([len(d[1]) for d in data])
    chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
    if rng is not None:
        rng.shuffle(chunks)
    return chunks


def _collate(data, idx):
    import torch

    T = max(len(data[i][1]) for i in idx)
    x = np.empty((len(idx), T, V), dtype=np.float32)
    for b, i in enumerate(idx):
        lp = data[i][1].astype(np.float32)
        x[b, :len(lp)] = lp
        x[b, len(lp):] = lp[-1]          # replicate the last frame (matches runtime edge padding)
    tg = [data[i][2] for i in idx]
    tgt = torch.from_numpy(np.concatenate(tg))
    tgt = torch.where(tgt > 3, tgt - 1, tgt)                    # merged vocabulary (126)
    return (torch.from_numpy(x), tgt, torch.tensor([len(data[i][1]) for i in idx]),
            torch.tensor([len(t) for t in tg]))


def ctc(lpa, tgt, il, tl):
    import torch
    import torch.nn.functional as F

    blank = torch.logaddexp(lpa[..., 0], lpa[..., 3]).unsqueeze(-1)
    m = torch.cat([blank, lpa[..., 1:3], lpa[..., 4:]], dim=-1)          # [N, T, 126]
    return F.ctc_loss(m.transpose(0, 1), tgt, il, tl, blank=0, reduction="sum",
                      zero_infinity=True)


def eval_loss(net, data) -> float:
    import torch

    net.eval()
    tot = ntok = 0.0
    with torch.no_grad():
        for idx in _batches(data, 16):
            x, tgt, il, tl = _collate(data, idx)
            tot += float(ctc(net(x), tgt, il, tl))
            ntok += float(tl.sum())
    return tot / ntok


def train_one(kind, train, val, log_fn=log):
    import torch

    torch.manual_seed(HP["seed"])
    net = make_net(kind)
    opt = torch.optim.AdamW(net.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    rng = np.random.default_rng(HP["seed"])
    hist = [{"epoch": 0, "val": eval_loss(net, val)}]
    best, best_state, bad = hist[0]["val"], {k: v.clone() for k, v in net.state_dict().items()}, 0
    log_fn(f"  {kind}: epoch 0 val CTC/token {best:.4f} (identity)")
    for ep in range(1, HP["max_epochs"] + 1):
        net.train()
        t0 = time.time()
        tot = ntok = 0.0
        for idx in _batches(train, HP["batch"], rng):
            x, tgt, il, tl = _collate(train, idx)
            loss = ctc(net(x), tgt, il, tl)
            opt.zero_grad()
            (loss / tl.sum()).backward()
            opt.step()
            tot += loss.item(); ntok += float(tl.sum())
        v = eval_loss(net, val)
        hist.append({"epoch": ep, "train": tot / ntok, "val": v, "s": round(time.time() - t0, 1)})
        log_fn(f"  {kind}: epoch {ep} train {tot / ntok:.4f} val {v:.4f} ({time.time() - t0:.0f}s)")
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, hist, best


# ----------------------------------------------------------------------------- selection

def select(lang: str, dfs: dict, summ: dict, n_boot: int = 500) -> dict:
    """dfs: name -> slot rows (raw, linear, ctx, conv). Primary metric: mean over the three kinds
    of AUC(-llr_sub) for corrupted vs untouched slots. Pick the simplest within noise of the
    best (paired bootstrap 95 % CI of best - candidate includes 0); ship only if it beats raw
    (paired bootstrap CI of candidate - raw > 0)."""
    from bench.pipeline.evalctl import mean_auc_untouched, utt_bootstrap

    score = {k: mean_auc_untouched(d) for k, d in dfs.items()}
    best = max(score, key=score.get)
    within = {}
    for k in dfs:
        if k == best:
            within[k] = {"delta": 0.0, "lo": 0.0, "hi": 0.0}
            continue
        b = utt_bootstrap(dfs[best], dfs[k], mean_auc_untouched, n_boot=n_boot)
        within[k] = {"delta": b["delta"], "lo": b["lo"], "hi": b["hi"]}
    cands = sorted([k for k in dfs if k == best or within[k]["lo"] <= 0.0], key=SIMPLICITY.get)
    pick = cands[0]
    vs_raw = None
    if pick != "raw":
        vs_raw = utt_bootstrap(dfs[pick], dfs["raw"], mean_auc_untouched, n_boot=n_boot)
        if not vs_raw["lo"] > 0:
            # the simplest-within-noise adapter does not beat raw: is any adapter significantly better?
            sig = []
            for k in dfs:
                if k == "raw":
                    continue
                b = utt_bootstrap(dfs[k], dfs["raw"], mean_auc_untouched, n_boot=n_boot)
                if b["lo"] > 0:
                    sig.append((SIMPLICITY[k], k, b))
            if sig:
                _, pick, vs_raw = min(sig)
            else:
                pick = "raw"
    return {"mean_auc": score, "best": best, "within_noise_of_best": within, "choice": pick,
            "choice_vs_raw": vs_raw}


def summarise(lang, dfs, summs, hist, choice, sel):
    from bench.pipeline.evalctl import KINDS_AUC, auc_table, boot_ci, mean_auc_untouched

    tab = {}
    for k, d in dfs.items():
        t = auc_table(d)
        t.update({f"med_{a}": b for a, b in auc_table(d, "llr_sub_med").items()})
        t["mean_auc"] = mean_auc_untouched(d)
        t["mean_auc_med"] = mean_auc_untouched(d, "llr_sub_med")
        lo, hi = boot_ci(d, mean_auc_untouched, n_boot=300)
        t["mean_auc_ci"] = [lo, hi]
        clean = d[d.kind == "clean"].llr_sub
        t["native_llr_sub_p05"] = float(np.quantile(clean, 0.05))
        t["native_llr_sub_median"] = float(np.median(clean))
        t["ter"] = summs[k]["ter"]
        t["applied"] = summs[k]["applied"]
        t["n_ok"] = summs[k]["n_ok"]
        tab[k] = t
    return {"lang": lang, "g2p_version": g2p_version(lang), "hp": HP, "history": hist,
            "table": tab, "selection": sel, "choice": choice}


def run_lang(lang: str, kinds=KINDS, force=False, workers=6):
    import torch

    from bench.pipeline import evalctl

    torch.set_num_threads(workers)
    out_json = repo_root() / "results" / "tables" / f"adapter_{lang}.json"
    cand_dir = pipe_dir("adapters", lang)
    ver = g2p_version(lang)
    if out_json.exists() and not force:
        old = json.loads(out_json.read_text())
        if old.get("g2p_version") == ver and old.get("hp") == HP and set(old["table"]) == {"raw", *kinds}:
            log(f"[{lang}] adapter selection up to date (g2p {ver}): {old['choice']}")
            return old
    t0 = time.time()
    train = load_ctc_data("fleurs", lang, "train")
    val = load_ctc_data("fleurs-validation", lang, None)
    log(f"[{lang}] train {len(train)} utts, val {len(val)} utts (g2p {ver})")
    hist = {}
    specs = {"raw": None}
    for kind in kinds:
        p = cand_dir / f"{kind}-{ver}.npz"
        hp_ = cand_dir / f"{kind}-{ver}.json"
        if p.exists() and hp_.exists() and not force:
            hist[kind] = json.loads(hp_.read_text())
        else:
            net, h, best = train_one(kind, train, val)
            np.savez(p, **export(net))
            hp_.write_text(json.dumps(h))
            hist[kind] = h
            # numpy runtime must reproduce torch
            from pronunciada.adapter import Adapter

            a = Adapter.from_file(p)
            x = val[0][1].astype(np.float32)
            with torch.no_grad():
                ref_out = net(torch.from_numpy(x)[None])[0].numpy()
            err = float(np.abs(a(x) - ref_out).max())
            log(f"  {kind}: numpy vs torch max abs diff {err:.2e}")
            assert err < 1e-3, err
        specs[kind] = str(p)
    del train
    recs = records("fleurs-validation", lang=lang)
    dfs, summs = {}, {}
    for k, spec in specs.items():
        dfs[k], summs[k] = evalctl.run("fleurs-validation", recs, spec, workers=workers,
                                       desc=f"{lang}/{k}")
        log(f"[{lang}] {k}: mean AUC {evalctl.mean_auc_untouched(dfs[k]):.4f} TER {summs[k]['ter']:.4f}")
        dfs[k].to_parquet(cand_dir / f"eval-{k}-{ver}.parquet")
    sel = select(lang, dfs, summs)
    choice = sel["choice"]
    dst = repo_root() / "pronunciada" / "data" / "adapters" / f"{lang}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if choice == "raw":
        if dst.exists():
            dst.unlink()
    else:
        shutil.copyfile(specs[choice], dst)
    res = summarise(lang, dfs, summs, hist, choice, sel)
    res["wall_s"] = round(time.time() - t0, 1)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1, default=float))
    log(f"[{lang}] choice: {choice}  ({res['wall_s']:.0f}s)")
    return res


def main():
    from bench.pipeline.common import base_parser

    ap = base_parser(__doc__)
    ap.add_argument("--kinds", default=",".join(KINDS))
    a = ap.parse_args()
    for lang in parse_langs(a.lang):
        run_lang(lang, tuple(a.kinds.split(",")), a.force, a.workers)


if __name__ == "__main__":
    main()
