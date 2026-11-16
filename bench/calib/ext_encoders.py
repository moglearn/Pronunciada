"""Round 6: pooled hidden states of OUTSIDE pre-trained speech encoders on speechocean762 TRAIN.

    python -m bench.calib.ext_encoders --model wav2vec2-base [--device mps]

Candidates (model cards checked 2026-11-16; licence of the released weights):
  whisper-tiny   openai/whisper-tiny            Apache-2.0   (encoder only, 4 layers x 384)
  whisper-base   openai/whisper-base            Apache-2.0   (encoder only, 6 layers x 512)
  wav2vec2-base  facebook/wav2vec2-base         Apache-2.0   (12 layers x 768, self-supervised)
  hubert-base    facebook/hubert-base-ls960     Apache-2.0   (12 layers x 768, self-supervised)
  WavLM base / base-plus: no licence on the model card -> skipped.

Whisper runs WITHOUT the 30 s padding (the encoder is applied to the clip's own frames with the
first T positional embeddings), which is what a runtime export can afford on short clips.
Per utterance and layer (index 0 = the input of the first transformer layer): mean ++ std over
the SPEECH frames of the ZIPA alignment (segments from bench/calib/delivery_extract.py's
_align.pkl; frame k counts as speech when its centre (k + 0.5) * 20 ms lies in a segment).
Output: $PRONUNCIADA_CACHE/calib/ext_<model>_<split>.npz: utt, L<i> [N, 2C] float32.
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np

MODELS = {
    "whisper-tiny": ("openai/whisper-tiny", "whisper", "Apache-2.0"),
    "whisper-base": ("openai/whisper-base", "whisper", "Apache-2.0"),
    "wav2vec2-base": ("facebook/wav2vec2-base", "w2v", "Apache-2.0"),
    "hubert-base": ("facebook/hubert-base-ls960", "w2v", "Apache-2.0"),
}
HOP = 0.02
T0 = 0.03


class Encoder:
    def __init__(self, name: str, device: str = "cpu"):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self.torch = torch
        repo, self.kind, _ = MODELS[name]
        self.fe = AutoFeatureExtractor.from_pretrained(repo)
        m = AutoModel.from_pretrained(repo)
        self.model = (m.encoder if self.kind == "whisper" else m).eval().to(device)
        self.device = device

    def hidden(self, wav: np.ndarray) -> list[np.ndarray]:
        """[layer][T, C] float32 hidden states (layer 0 = embeddings)."""
        torch = self.torch
        with torch.no_grad():
            if self.kind == "whisper":
                f = self.fe(wav, sampling_rate=16000, return_tensors="pt", padding="longest")
                x = f["input_features"]
                if x.shape[-1] % 2:
                    x = x[..., :-1]
                return [h[0].float().cpu().numpy() for h in whisper_hidden(self.model, x.to(self.device))]
            f = self.fe(wav, sampling_rate=16000, return_tensors="pt")
            out = self.model(f["input_values"].to(self.device), output_hidden_states=True)
            return [h[0].float().cpu().numpy() for h in out.hidden_states]


def whisper_hidden(enc, x):
    """Unpadded Whisper encoder: conv stem, the first T positional embeddings, every layer;
    returns [embeddings, layer 1, ..., layer L (with the final layer norm)]."""
    import torch.nn.functional as Fn

    h = Fn.gelu(enc.conv1(x))
    h = Fn.gelu(enc.conv2(h)).permute(0, 2, 1)
    h = h + enc.embed_positions.weight[: h.shape[1]]
    hs = [h]
    for i, layer in enumerate(enc.layers):
        h = layer(h, None, layer_head_mask=None)[0]
        hs.append(enc.layer_norm(h) if i == len(enc.layers) - 1 else h)
    return hs


def speech_mask(n: int, segs) -> np.ndarray:
    c = (np.arange(n) + 0.5) * HOP
    m = np.zeros(n, bool)
    for a, b in segs:
        m |= (c >= T0 + a * HOP) & (c < T0 + b * HOP)
    return m


def pool(hs, segs):
    out = []
    for h in hs:
        m = speech_mask(len(h), segs)
        x = h[m] if m.sum() >= 2 else h
        out.append(np.concatenate([x.mean(0), x.std(0)]).astype(np.float32))
    return out


def main():
    import torch

    from bench.calib.common import calib_cache
    from bench.pipeline.common import records
    from pronunciada.audio import load_audio

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    torch.set_num_threads(4)
    al = pickle.loads((calib_cache() / f"delivery_speechocean762_{a.split}_align.pkl").read_bytes())
    recs = [r for r in records("speechocean762") if r["split"] == a.split and r["id"] in al]
    enc = Encoder(a.model, a.device)
    utt, layers = [], None
    t0 = time.time()
    for k, r in enumerate(recs):
        hs = enc.hidden(load_audio(r["audio"]))
        p = pool(hs, al[r["id"]]["segs"])
        layers = layers or [[] for _ in p]
        for i, v in enumerate(p):
            layers[i].append(v)
        utt.append(r["id"])
        if (k + 1) % 500 == 0:
            print(a.model, k + 1, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(calib_cache() / f"ext_{a.model}_{a.split}.npz", utt=np.array(utt),
             **{f"L{i}": np.stack(v) for i, v in enumerate(layers)})
    print(a.model, len(utt), "utts", len(layers), "layers", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
