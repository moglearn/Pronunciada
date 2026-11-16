"""Export trained prosody models to ONNX + JSON metadata under pronunciada/data/prosody/.

    .venv/bin/python -m bench.prosody.export tone tone_ctx            # -> tone_zh.onnx / .json
    .venv/bin/python -m bench.prosody.export pitch pitch_ja
    .venv/bin/python -m bench.prosody.export stress stress_multi
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from bench.prosody.common import MODELS_DIR, OUT
from pronunciada.prosody.features import CTX_LEN, N_IN

DATA = Path(__file__).resolve().parents[2] / "pronunciada" / "data" / "prosody"


def _export(net, path: Path, inputs: dict, dyn: dict):
    net.eval()
    names = list(inputs)
    torch.onnx.export(net, tuple(inputs.values()), str(path), input_names=names, output_names=["logits"],
                      dynamic_axes=dyn, opset_version=17, dynamo=False)
    import onnx

    m = onnx.load(str(path))
    onnx.save(m, str(path))      # strip nothing; re-save to normalise
    return path


def export_tone(name: str, out: str = "tone_zh"):
    from bench.prosody.tone_eval import load_model

    net, ck = load_model(name)
    DATA.mkdir(parents=True, exist_ok=True)
    _export(net, DATA / f"{out}.onnx", {"x": torch.zeros(2, CTX_LEN, N_IN)}, {"x": {0: "n"}, "logits": {0: "n"}})
    res = json.load(open(OUT / "results" / f"{name}.json"))
    meta = {"model": name, "temperature": res["temperature"], "platt": res["platt"], "ctx_len": CTX_LEN}
    json.dump(meta, open(DATA / f"{out}.json", "w"), indent=1)
    return net


def export_tone_v2(name: str, out: str = "tone_zh_v2"):
    """Utterance-level model: input x [S, 160, 10] = all syllables of one utterance in order."""
    from bench.prosody.tone2 import load2

    net, kind, ck = load2(name)

    class Utt(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.forward_utt(x)

    _export(Utt(net), DATA / f"{out}.onnx", {"x": torch.zeros(5, CTX_LEN, N_IN)},
            {"x": {0: "n"}, "logits": {0: "n"}})
    res = json.load(open(OUT / "results" / f"{name}.json"))
    meta = {"model": name, "temperature": res["temperature"], "platt": res["platt"], "ctx_len": CTX_LEN,
            "sequence": True}
    json.dump(meta, open(DATA / f"{out}.json", "w"), indent=1)
    return Utt(net)


def export_tone_v3(name: str, out: str = "tone_zh_v3", out_dir: str | None = None, leniency: str = ""):
    """tone v3 (bench.prosody.tone3): x [S, 160, 10 + extras]; meta "inputs" says which extra
    frame columns the runtime must append (register features, low-frequency log-mel)."""
    from bench.prosody.tone3 import load3, n_in

    net, kind, ck = load3(name)

    class Utt(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.forward_utt(x)

    d = Path(out_dir) if out_dir else DATA
    d.mkdir(parents=True, exist_ok=True)
    _export(Utt(net), d / f"{out}.onnx", {"x": torch.zeros(5, CTX_LEN, n_in(ck["cfg"]))},
            {"x": {0: "n"}, "logits": {0: "n"}})
    res = json.load(open(OUT / "results" / f"{name}.json"))
    meta = {"model": name, "temperature": res["temperature"], "platt": res["platt"], "ctx_len": CTX_LEN,
            "sequence": True, "inputs": {"reg": bool(ck["cfg"]["reg"]), "lowmel": bool(ck["cfg"]["lowmel"])},
            "leniency": [x for x in leniency.split(",") if x]}
    json.dump(meta, open(d / f"{out}.json", "w"), indent=1)
    x = np.random.default_rng(0).normal(size=(7, CTX_LEN, n_in(ck["cfg"]))).astype(np.float32)
    print("onnx vs torch max |dlogit|", parity(d / f"{out}.onnx", Utt(net), {"x": x}))
    return Utt(net)


def export_pitch(name: str):
    from bench.prosody.tone_eval import load_model

    net, ck = load_model(name)
    _export(net, DATA / "pitch_ja.onnx", {"x": torch.zeros(2, CTX_LEN, N_IN)}, {"x": {0: "n"}, "logits": {0: "n"}})
    res = json.load(open(OUT / "results" / f"{name}.json"))
    json.dump({"model": name, "platt": res["platt"], "ctx_len": CTX_LEN}, open(DATA / "pitch_ja.json", "w"), indent=1)


def export_stress(name: str):
    from bench.prosody.stress_train import StressNet

    ck = torch.load(MODELS_DIR / f"{name}.pt", weights_only=False)
    net = StressNet(**ck["cfg"])
    net.load_state_dict(ck["state"])
    from pronunciada.prosody.features import N_STRESS_EXTRA

    inputs = {"x": torch.zeros(2, CTX_LEN, N_IN), "extra": torch.zeros(2, N_STRESS_EXTRA)}
    dyn = {"x": {0: "n"}, "extra": {0: "n"}, "logits": {0: "n"}}
    if ck["cfg"].get("n_lang"):
        inputs["lang"] = torch.zeros(2, dtype=torch.int64)
        dyn["lang"] = {0: "n"}
    _export(net, DATA / "stress.onnx", inputs, dyn)
    res = json.load(open(OUT / "results" / f"{name}.json"))
    json.dump({"model": name, "langs": ck["langs"], "platt": res["platt"], "ctx_len": CTX_LEN,
               "n_lang": ck["cfg"].get("n_lang", 0)}, open(DATA / "stress.json", "w"), indent=1)


def parity(onnx_path: Path, net, feeds: dict) -> float:
    import onnxruntime as ort

    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = net(*[torch.tensor(v) for v in feeds.values()]).numpy()
    got = s.run(None, {k: v for k, v in feeds.items()})[0]
    return float(np.abs(ref - got).max())


if __name__ == "__main__":
    what, name = sys.argv[1], sys.argv[2]
    if what == "tone_v3":     # tone_v3 <ckpt> [out_stem] [out_dir] [leniency rules]
        a = sys.argv[3:] + [None] * 3
        export_tone_v3(name, a[0] or "tone_zh_v3", a[1] or None, a[2] or "")
    else:
        {"tone": export_tone, "tone_v2": export_tone_v2, "pitch": export_pitch, "stress": export_stress}[what](name)
    print("exported", what, name)
