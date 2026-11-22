"""Round 19 (encoder adaptation): run an ONNX graph in torch, with trainable weights.

A small interpreter (training time only; nothing here ships): every node of the graph is executed
with the equivalent torch op, the float initializers become nn.Parameters (frozen unless chosen),
so gradients flow through the exact exported computation. Shape arithmetic stays on the CPU as
int64 tensors; float work runs on `device`.

Adaptation forms:
  * LoRA: for chosen MatMul nodes x @ W (W a 2-D initializer), y = x @ W + (x @ A) @ B * (alpha / r),
    A [in, r] small random, B [r, out] zero (so training starts at the frozen model).
  * full: chosen initializers get requires_grad.
`merged_initializers()` returns {initializer name: numpy array} with the LoRA deltas folded in,
and `export(path)` writes the graph with them (the ONNX that ships has the same ops as the cut).

Verified against onnxruntime by bench/round19_encoder/verify_torch.py.
"""
from __future__ import annotations

import numpy as np

_DT = None


def _dtypes():
    global _DT
    if _DT is None:
        import torch

        _DT = {1: torch.float32, 2: torch.uint8, 3: torch.int8, 5: torch.int16, 6: torch.int32, 7: torch.int64,
               9: torch.bool, 10: torch.float16, 11: torch.float64}
    return _DT


_SAFE_EXP = None


def _safe_exp():
    """exp with a backward of 0 where the incoming gradient is 0. The export computes softplus-like
    activations as log(1 + exp(x)) inside a Where that keeps x for large x: the forward is finite,
    but exp overflows to inf and inf * 0 would make the gradient NaN. Forward is unchanged."""
    global _SAFE_EXP
    if _SAFE_EXP is None:
        import torch

        class SafeExp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = torch.exp(x)
                ctx.save_for_backward(y)
                return y

            @staticmethod
            def backward(ctx, g):
                (y,) = ctx.saved_tensors
                return torch.where(g == 0, torch.zeros_like(g), g * y)

        _SAFE_EXP = SafeExp.apply
    return _SAFE_EXP


def build(model, device="cpu", lora=None, full=None):
    """model: onnx ModelProto or path. lora: None or {"rank": r, "alpha": a, "match": callable(node_name)}.
    full: None or callable(initializer_name) -> bool (trainable)."""
    import onnx
    import torch
    from onnx import helper, numpy_helper
    from torch import nn
    from torch.nn import functional as F

    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(str(model))
    DT = _dtypes()

    def attrs(n):
        return {a.name: helper.get_attribute_value(a) for a in n.attribute}

    class G(nn.Module):
        def __init__(self):
            super().__init__()
            g = model.graph
            self.model = model
            self.device_ = torch.zeros(1, device=device).device  # "mps" -> "mps:0"
            self.in_names = [i.name for i in g.input]
            self.out_names = [o.name for o in g.output]
            self.params = nn.ParameterDict()
            self.pkey = {}
            self.consts = {}
            for init in g.initializer:
                arr = numpy_helper.to_array(init).copy()
                t = torch.from_numpy(arr)
                if t.is_floating_point() and t.ndim >= 1 and t.numel() > 1:
                    k = f"w{len(self.pkey)}"
                    self.params[k] = nn.Parameter(t.to(self.device_), requires_grad=bool(full and full(init.name)))
                    self.pkey[init.name] = k
                else:
                    self.consts[init.name] = t
            self.nodes = self._prep(g)
            self.lora = nn.ModuleDict()
            self.lora_of = {}
            self.scale = 0.0
            if lora:
                r, alpha = int(lora["rank"]), float(lora.get("alpha", lora["rank"]))
                self.scale = alpha / r
                gen = torch.Generator().manual_seed(int(lora.get("seed", 0)))
                for n, op, ins, outs, at in self.nodes:
                    if op == "MatMul" and ins[1] in self.pkey and lora["match"](n):
                        W = self.params[self.pkey[ins[1]]]
                        if W.ndim != 2:
                            continue
                        k = f"l{len(self.lora_of)}"
                        m = nn.Module()
                        m.A = nn.Parameter((torch.randn(W.shape[0], r, generator=gen) / np.sqrt(W.shape[0])).to(self.device_))
                        m.B = nn.Parameter(torch.zeros(r, W.shape[1], device=self.device_))
                        self.lora[k] = m
                        self.lora_of[n] = (k, ins[1])
            self._dev_cache = {}

        def _prep(self, g):
            out = []
            for n in g.node:
                at = attrs(n)
                if n.op_type == "Constant":
                    self.consts[n.output[0]] = torch.from_numpy(numpy_helper.to_array(at["value"]).copy())
                    continue
                if n.op_type == "If":
                    at = {"then": self._prep(at["then_branch"]), "else": self._prep(at["else_branch"]),
                          "then_out": [o.name for o in at["then_branch"].output],
                          "else_out": [o.name for o in at["else_branch"].output]}
                out.append((n.name, n.op_type, list(n.input), list(n.output), at))
            return out

        # -------------------------------------------------------------- helpers
        def _d(self, t):
            """float / bool data to the compute device (constants cached)."""
            if not isinstance(t, torch.Tensor) or t.device == self.device_:
                return t
            if t.ndim == 0 or (t.ndim == 1 and not t.is_floating_point()):
                return t.to(self.device_)
            key = id(t)
            c = self._dev_cache.get(key)
            if c is None or c[0] is not t:
                c = (t, t.to(self.device_))
                self._dev_cache[key] = c
            return c[1]

        @staticmethod
        def _ints(t):
            return [int(v) for v in t.reshape(-1).tolist()]

        def forward(self, **feeds):
            env = dict(self.consts)
            for k, v in self.pkey.items():
                env[k] = self.params[v]
            env.update(feeds)
            self._run(self.nodes, env)
            return [env[o] for o in self.out_names]

        def _run(self, nodes, env):
            for name, op, ins, outs, at in nodes:
                x = [env[i] if i else None for i in ins]
                y = self._op(name, op, x, at, env)
                if isinstance(y, (list, tuple)):
                    for o, v in zip(outs, y):
                        env[o] = v
                else:
                    env[outs[0]] = y

        def _op(self, name, op, x, at, env):  # noqa: C901
            d = self._d
            if op in ("Add", "Sub", "Mul", "Div", "Pow", "Equal", "GreaterOrEqual", "Where"):
                if any(isinstance(v, torch.Tensor) and v.device == self.device_ for v in x):
                    x = [d(v) for v in x]
                a = x[0]
                if op == "Add":
                    return a + x[1]
                if op == "Sub":
                    return a - x[1]
                if op == "Mul":
                    return a * x[1]
                if op == "Div":
                    if not a.is_floating_point() and not x[1].is_floating_point():
                        return torch.div(a, x[1], rounding_mode="trunc")
                    return a / x[1]
                if op == "Pow":
                    return torch.pow(a, x[1])
                if op == "Equal":
                    return a == x[1]
                if op == "GreaterOrEqual":
                    return a >= x[1]
                if op == "Where":
                    return torch.where(a, x[1], x[2])
            if op == "MatMul":
                a, b = d(x[0]), d(x[1])
                y = torch.matmul(a, b)
                lo = self.lora_of.get(name)
                if lo is not None:
                    m = self.lora[lo[0]]
                    y = y + torch.matmul(torch.matmul(a, m.A), m.B) * self.scale
                return y
            if op == "Shape":
                return torch.tensor(list(x[0].shape), dtype=torch.int64)
            if op == "Gather":
                ax = at.get("axis", 0)
                src, idx = x[0], x[1]
                idx = idx.to(src.device) if src.device != idx.device else idx
                n = src.shape[ax]
                idx = torch.where(idx < 0, idx + n, idx)
                if idx.ndim == 0:
                    return src.select(ax, int(idx))
                out = torch.index_select(src, ax, idx.reshape(-1))
                return out.reshape(src.shape[:ax] + idx.shape + src.shape[ax + 1:])
            if op == "Unsqueeze":
                y = x[0]
                axes = sorted(self._ints(x[1]) if len(x) > 1 and x[1] is not None else at["axes"])
                r = y.ndim + len(axes)
                for a_ in sorted(a % r for a in axes):
                    y = y.unsqueeze(a_)
                return y
            if op == "Squeeze":
                y = x[0]
                axes = self._ints(x[1]) if len(x) > 1 and x[1] is not None else at.get("axes")
                if axes is None:
                    return y.squeeze()
                for a_ in sorted((a % y.ndim for a in axes), reverse=True):
                    y = y.squeeze(a_)
                return y
            if op == "Concat":
                vs = [v for v in x if v is not None]
                if any(v.device == self.device_ for v in vs):
                    vs = [d(v) for v in vs]
                return torch.cat(vs, at["axis"])
            if op == "Reshape":
                shp = self._ints(x[1])
                src = x[0]
                shp = [src.shape[i] if s == 0 else s for i, s in enumerate(shp)]
                return src.reshape(shp)
            if op == "Transpose":
                return x[0].permute(*at["perm"]) if "perm" in at else x[0].permute(*reversed(range(x[0].ndim)))
            if op == "Cast":
                return x[0].to(DT[at["to"]])
            if op == "Slice":
                src = x[0]
                st, en = self._ints(x[1]), self._ints(x[2])
                axes = self._ints(x[3]) if len(x) > 3 and x[3] is not None else list(range(len(st)))
                steps = self._ints(x[4]) if len(x) > 4 and x[4] is not None else [1] * len(st)
                sl = [slice(None)] * src.ndim
                for s, e, a, stp in zip(st, en, axes, steps):
                    a = a % src.ndim
                    n = src.shape[a]
                    if stp < 0:
                        raise NotImplementedError("negative slice step")
                    s = max(0, min(n, s + n if s < 0 else s))
                    e = max(0, min(n, e + n if e < 0 else e))
                    sl[a] = slice(s, e, stp)
                return src[tuple(sl)]
            if op == "Expand":
                shp = self._ints(x[1])
                src = x[0]
                tgt = torch.broadcast_shapes(tuple(src.shape), tuple(shp))
                return src.expand(tgt)
            if op == "ConstantOfShape":
                v = at.get("value")
                val = numpy_helper.to_array(v) if v is not None else np.zeros(1, np.float32)
                t = torch.from_numpy(val.copy()).reshape(-1)
                shp = self._ints(x[0])
                return torch.full(shp, t[0].item(), dtype=t.dtype)
            if op == "Range":
                s, e, st = x[0].item(), x[1].item(), x[2].item()
                return torch.arange(s, e, st, dtype=x[0].dtype)
            if op == "Tile":
                return x[0].repeat(*self._ints(x[1]))
            if op == "GatherElements":
                src, idx = x[0], x[1].to(x[0].device)
                ax = at.get("axis", 0)
                idx = torch.where(idx < 0, idx + src.shape[ax], idx)
                return torch.gather(src, ax, idx)
            if op == "ScatterND":
                src, idx, upd = x[0], x[1], x[2]
                out = src.clone()
                upd = upd.to(src.device)
                k = idx.shape[-1]
                ii = idx.reshape(-1, k)
                uu = upd.reshape((-1,) + tuple(src.shape[k:]))
                out[tuple(ii[:, j] for j in range(k))] = uu.to(out.dtype)
                return out
            if op == "Softmax":
                return torch.softmax(x[0], at.get("axis", -1))
            if op == "LogSoftmax":
                return torch.log_softmax(x[0], at.get("axis", -1))
            if op in ("ReduceMean", "ReduceSum"):
                axes = at.get("axes")
                if axes is None and len(x) > 1 and x[1] is not None:
                    axes = self._ints(x[1])
                kd = bool(at.get("keepdims", 1))
                src = x[0]
                if axes is None:
                    axes = list(range(src.ndim))
                f = torch.mean if op == "ReduceMean" else torch.sum
                return f(src, dim=tuple(axes), keepdim=kd)
            if op == "Conv":
                w = d(x[1])
                b = d(x[2]) if len(x) > 2 and x[2] is not None else None
                pads = at.get("pads", [0] * (2 * len(at["kernel_shape"])))
                k = len(at["kernel_shape"])
                assert k == 1, "only 1-D conv"
                src = d(x[0])
                if pads[0] != pads[1]:
                    src = F.pad(src, (pads[0], pads[1]))
                    p = 0
                else:
                    p = pads[0]
                return F.conv1d(src, w, b, stride=at.get("strides", [1])[0], padding=p,
                                dilation=at.get("dilations", [1])[0], groups=at.get("group", 1))
            if op == "If":
                c = bool(x[0].reshape(-1)[0].item())
                br, outs = (at["then"], at["then_out"]) if c else (at["else"], at["else_out"])
                self._run(br, env)
                return [env[o] for o in outs]
            if op == "Exp" and x[0].is_floating_point():
                return _safe_exp()(x[0])
            un = {"Exp": torch.exp, "Log": torch.log, "Sigmoid": torch.sigmoid, "Tanh": torch.tanh,
                  "Abs": torch.abs, "Neg": torch.neg, "Sign": torch.sign, "Sin": torch.sin, "Cos": torch.cos,
                  "Atan": torch.atan, "Sqrt": torch.sqrt, "Relu": torch.relu}
            if op in un:
                return un[op](x[0])
            if op == "Identity":
                return x[0]
            raise NotImplementedError(op)

        # -------------------------------------------------------------- export
        def merged_initializers(self) -> dict:
            out = {}
            with torch.no_grad():
                for name, k in self.pkey.items():
                    p = self.params[k]
                    if p.requires_grad:
                        out[name] = p.detach().cpu().numpy().astype(np.float32)
                for n, (k, wname) in self.lora_of.items():
                    m = self.lora[k]
                    W = out.get(wname)
                    if W is None:
                        W = self.params[self.pkey[wname]].detach().cpu().numpy().astype(np.float32)
                    out[wname] = (W + (m.A @ m.B).detach().cpu().numpy() * self.scale).astype(np.float32)
            return out

        def export(self, path, keep_outputs=None):
            import copy

            m = copy.deepcopy(self.model)
            new = self.merged_initializers()
            for i, init in enumerate(m.graph.initializer):
                if init.name in new:
                    m.graph.initializer[i].CopyFrom(numpy_helper.from_array(new[init.name], init.name))
            if keep_outputs is not None:
                keep = [o for o in m.graph.output if o.name in keep_outputs]
                del m.graph.output[:]
                m.graph.output.extend(keep)
            onnx.save(m, str(path))
            return path

    return G()


def trainable_params(g):
    return [p for p in g.parameters() if p.requires_grad]
