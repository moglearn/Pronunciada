"""ZIPA ONNX wrapper: audio -> frame log-posteriors over the packaged token list.

Frame timing (measured on zipa-small-crctc-500k): with T fbank frames (10 ms, snip_edges=False,
i.e. T = (num_samples + 80) // 160) the model emits T' = (T - 7) // 2 frames (needs T >= 9).
Output frame j depends mostly on fbank frames 2j .. 2j+7, centre 2j + 3.5, i.e. centre time
0.02 j + 0.04 s. `frame_to_time(j)` = 0.02 j + 0.03 is the START of the 20 ms interval of
frame j, so a span [s, e) of frames covers [frame_to_time(s), frame_to_time(e)).
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np

from .audio import fbank, load_audio
from .text.ipa import load_tokens

HF_URL = "https://huggingface.co/anyspeech/{name}/resolve/main/{file}"
PRECISION_FILES = {"fp32": "model.onnx", "fp16": "model.fp16.onnx", "int8": "model.int8.onnx"}
NON_SPEECH = 4                  # ids 0..3: <blk> <sos/eos> <unk> ▁
MIN_FBANK_FRAMES = 9            # model needs >= 9 input frames to emit one output frame
SILENCE_FEAT = float(np.log(np.finfo(np.float32).eps))   # log-mel value of digital silence

# Intermediate encoder activations exposed as extra graph outputs (round 5, sentence "delivery"
# features, pronunciada.delivery). Names are the same in the fp32, fp16 and int8 exports of
# zipa-small-crctc-500k; shapes [T', 1, C] for the stacks and [1, T', C] for the final output
# (the input of the CTC layer), all on the 20 ms output grid.
EMBED_OUTPUTS = {
    "enc1": "/encoder/1/out_combiner/Add_output_0",      # stack 1 (early), 256-d
    "enc3": "/encoder/3/out_combiner/Add_output_0",      # stack 3 (middle), 512-d
    "enc5": "/encoder/5/out_combiner/Add_output_0",      # stack 5 (late), 256-d
    "final": "/Transpose_1_output_0",                    # encoder output before the CTC layer, 512-d
}
# Round 19: the output of every encoder stack (layer fusion for the English sentence head).
# Stack 0 runs at the full rate and has no out_combiner; its output is its last layer's bypass.
# Same [T', 1, C] layout as the stacks above.
ALL_LAYER_OUTPUTS = {
    "enc0": "/encoder/0/layers.1/bypass/Add_output_0",
    "enc1": EMBED_OUTPUTS["enc1"],
    "enc2": "/encoder/2/out_combiner/Add_output_0",
    "enc3": EMBED_OUTPUTS["enc3"],
    "enc4": "/encoder/4/out_combiner/Add_output_0",
    "enc5": EMBED_OUTPUTS["enc5"],
    "final": EMBED_OUTPUTS["final"],
}


def embed_outputs_for(layers) -> dict:
    """{key: tensor name} for a layer selection: True = EMBED_OUTPUTS (round 5), "all" =
    ALL_LAYER_OUTPUTS, or an iterable of keys of ALL_LAYER_OUTPUTS (EMBED_OUTPUTS are always
    included, since earlier rounds' inputs read them)."""
    if layers is True:
        return dict(EMBED_OUTPUTS)
    if layers == "all":
        return dict(ALL_LAYER_OUTPUTS)
    keys = set(layers) | set(EMBED_OUTPUTS)
    return {k: v for k, v in ALL_LAYER_OUTPUTS.items() if k in keys}


# --------------------------------------------------------------------------- protobuf surgery
# Adding graph outputs needs no `onnx` package: a repeated protobuf field may be extended by
# appending its encoding to the message, so the ModelProto bytes are edited in place
# (ModelProto.graph = field 7, GraphProto.output = field 12, ValueInfoProto.name = field 1,
# ValueInfoProto.type = 2 -> TypeProto.tensor_type = 1 -> {elem_type = 1 (float), shape = 2 with
# three symbolic dims}).

def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _ld(field: int, payload: bytes) -> bytes:
    """Length-delimited protobuf field."""
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _value_info(name: str) -> bytes:
    shape = b"".join(_ld(1, _ld(2, f"{name}_d{k}".encode())) for k in range(3))
    tensor = _varint(1 << 3) + _varint(1) + _ld(2, shape)
    return _ld(1, name.encode("utf-8")) + _ld(2, _ld(1, tensor))


def _read_varint(buf, i: int) -> tuple[int, int]:
    shift = n = 0
    while True:
        b = buf[i]
        i += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, i
        shift += 7


def add_graph_outputs(model_bytes: bytes, names) -> bytes:
    """Serialized ONNX ModelProto -> the same model with `names` (existing tensor names)
    appended to graph.output as float tensors of unspecified shape."""
    buf = memoryview(model_bytes)
    i = 0
    while i < len(buf):
        start = i
        key, i = _read_varint(buf, i)
        field, wt = key >> 3, key & 7
        if wt == 0:
            _, i = _read_varint(buf, i)
        elif wt == 1:
            i += 8
        elif wt == 5:
            i += 4
        elif wt == 2:
            ln, j = _read_varint(buf, i)
            if field == 7:
                extra = b"".join(_ld(12, _value_info(n)) for n in names)
                payload = bytes(buf[j:j + ln]) + extra
                return bytes(buf[:start]) + _ld(7, payload) + bytes(buf[j + ln:])
            i = j + ln
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
    raise ValueError("no graph in the ONNX model")


def _copy_range(src, dst, start: int, end: int, chunk: int = 1 << 22) -> None:
    src.seek(start)
    left = end - start
    while left > 0:
        b = src.read(min(chunk, left))
        if not b:
            raise ValueError("truncated ONNX model")
        dst.write(b)
        left -= len(b)


def write_with_graph_outputs(src_path: str, dest_path: str, names) -> None:
    """add_graph_outputs from file to file, streamed in 4 MB chunks: the same bytes, but without
    holding the model in memory (three copies of it at once otherwise: ~1 GB for fp32)."""
    size = os.path.getsize(src_path)
    with open(src_path, "rb") as f:
        i = 0
        while i < size:
            f.seek(i)
            head = f.read(32)
            key, j = _read_varint(head, 0)
            field, wt = key >> 3, key & 7
            if wt == 0:
                _, j = _read_varint(head, j)
                i += j
            elif wt == 1:
                i += j + 8
            elif wt == 5:
                i += j + 4
            elif wt == 2:
                ln, k = _read_varint(head, j)
                if field == 7:
                    extra = b"".join(_ld(12, _value_info(n)) for n in names)
                    with open(dest_path, "wb") as out:
                        _copy_range(f, out, 0, i)
                        out.write(_varint((7 << 3) | 2) + _varint(ln + len(extra)))
                        _copy_range(f, out, i + k, i + k + ln)
                        out.write(extra)
                        _copy_range(f, out, i + k + ln, size)
                    return
                i += k + ln
            else:
                raise ValueError(f"unsupported protobuf wire type {wt}")
    raise ValueError("no graph in the ONNX model")


def embedding_model_path(name: str, precision: str, cache_dir: str | None = None,
                         outputs: dict | None = None) -> str:
    """Path of the model variant with `outputs` (default EMBED_OUTPUTS) as extra outputs, built
    (once) next to the original in the model cache. The round-5 set keeps the file name
    `<model>.emb.onnx`; any other set gets `<model>.emb-<keys>.onnx`."""
    outputs = dict(EMBED_OUTPUTS) if outputs is None else outputs
    src = ensure_model_file(name, PRECISION_FILES[precision], cache_dir)
    tag = "" if list(outputs) == list(EMBED_OUTPUTS) else "-" + "-".join(outputs)
    dest = src[:-len(".onnx")] + f".emb{tag}.onnx"
    if not os.path.exists(dest) or os.path.getmtime(dest) < os.path.getmtime(src):
        part = f"{dest}.{os.getpid()}.part"
        try:
            write_with_graph_outputs(src, part, list(outputs.values()))
        except BaseException:
            if os.path.exists(part):
                os.remove(part)
            raise
        os.replace(part, dest)
    return dest


def cache_root(cache_dir: str | None = None) -> str:
    return cache_dir or os.environ.get("PRONUNCIADA_CACHE") or os.path.expanduser("~/.cache/pronunciada")


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "pronunciada"})
    with urllib.request.urlopen(req) as r, open(part, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(part, dest)


def ensure_model_file(name: str, file: str, cache_dir: str | None = None) -> str:
    path = os.path.join(cache_root(cache_dir), "models", name, file)
    if not os.path.exists(path):
        _download(HF_URL.format(name=name, file=file), path)
    return path


class AcousticModel:
    frames_per_input = 2            # fbank frames per output frame
    fbank_shift = 0.01
    frame_shift = 0.02              # seconds per output frame
    time_offset = 0.03              # start time of output frame 0 (centre at 0.04 s)

    def __init__(self, name: str = "zipa-small-crctc-500k", precision: str = "fp32",
                 threads: int | None = None, providers: list | None = None,
                 cache_dir: str | None = None, embeddings=False):
        import onnxruntime as ort

        if precision not in PRECISION_FILES:
            raise ValueError(f"precision must be one of {sorted(PRECISION_FILES)}")
        self.name, self.precision = name, precision
        model_path = ensure_model_file(name, PRECISION_FILES[precision], cache_dir)
        self.embeddings = False
        self.embed_outputs = embed_outputs_for(embeddings) if embeddings else {}
        if embeddings:
            try:
                model_path = embedding_model_path(name, precision, cache_dir, self.embed_outputs)
                self.embeddings = True
            except (OSError, ValueError):
                pass        # fall back to the plain model: no embeddings, same log-probs
        tok_path = ensure_model_file(name, "tokens.txt", cache_dir)

        ort.set_default_logger_severity(3)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        # The memory-pattern planner keeps one preallocated block per input shape it has seen,
        # and every clip length is a new shape: about +20 MB after a few calls, growing with
        # use. Without it the outputs are bit-identical and a run is ~0.5 ms slower.
        so.enable_mem_pattern = False
        if threads:
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, so,
                                            providers=providers or ["CPUExecutionProvider"])

        self.tokens = load_tokens()
        with open(tok_path, encoding="utf-8") as f:
            model_tokens = [ln.split()[0] for ln in f if ln.strip()]
        self.model_tokens = tuple(model_tokens)
        if self.model_tokens == self.tokens:
            self._remap = None
        else:
            pos = {t: i for i, t in enumerate(model_tokens)}
            # column k of the output = model column remap[k], or -1 (-> -inf) if missing
            self._remap = np.array([pos.get(t, -1) for t in self.tokens], dtype=np.int64)
        if self.embeddings:
            outs = {o.name for o in self.session.get_outputs()}
            if not all(v in outs for v in self.embed_outputs.values()):
                self.embeddings = False
        if not self.embeddings:
            self.embed_outputs = {}
        self._out_names = ["log_probs", "log_probs_len"] + list(self.embed_outputs.values())

    # ------------------------------------------------------------------ timing
    @staticmethod
    def num_output_frames(num_fbank_frames: int) -> int:
        return max((max(int(num_fbank_frames), MIN_FBANK_FRAMES) - 7) // 2, 0)

    def frame_to_time(self, t):
        """Start time (s) of output frame t (works on scalars and arrays)."""
        if isinstance(t, (list, tuple)):
            t = np.asarray(t)
        return t * self.frame_shift + self.time_offset

    # ------------------------------------------------------------------ inference
    def _features(self, wav) -> np.ndarray:
        if not isinstance(wav, np.ndarray) or wav.dtype != np.float32 or wav.ndim != 1:
            wav = load_audio(wav)
        x = fbank(wav)
        if len(x) < MIN_FBANK_FRAMES:
            pad = np.full((MIN_FBANK_FRAMES - len(x), x.shape[1] if x.size else 80), SILENCE_FEAT,
                          dtype=np.float32)
            x = np.concatenate([x.reshape(-1, 80), pad], axis=0)
        return x

    def _fix(self, lp: np.ndarray) -> np.ndarray:
        lp = lp.astype(np.float32, copy=False)
        if self._remap is None:
            return lp
        out = lp[:, np.maximum(self._remap, 0)]
        out[:, self._remap < 0] = -np.inf
        return out

    def posteriors_from_features(self, x: np.ndarray, return_embeddings: bool = False):
        """[T', 127] log-posteriors; with return_embeddings also {layer: [T', C] float32}
        (None when the model was loaded without embeddings)."""
        res = self.session.run(self._out_names if return_embeddings else self._out_names[:2],
                               {"x": x[None].astype(np.float32),
                                "x_lens": np.array([len(x)], dtype=np.int64)})
        n = int(res[1][0])
        lp = self._fix(res[0][0, :n])
        if not return_embeddings:
            return lp
        if not self.embeddings:
            return lp, None
        emb = {}
        for k, v in zip(self.embed_outputs, res[2:]):
            v = v[:, 0] if v.shape[1] == 1 and v.shape[0] != 1 else v[0]
            emb[k] = np.ascontiguousarray(v[:n], dtype=np.float32)
        return lp, emb

    def posteriors(self, wav) -> np.ndarray:
        """[T', 127] float32 log-posteriors in packaged token order."""
        return self.posteriors_from_features(self._features(wav))

    def posteriors_batch(self, wavs, max_pad_ratio: float = 1.0) -> list[np.ndarray]:
        """Length-bucketed batch inference.

        Padding is NOT neutral for this export: padding an utterance by even one frame changes
        its log-probs by up to ~0.1 (and ~1-2 when padded 4x), argmax unchanged. So by default
        (max_pad_ratio=1.0) only utterances with identical fbank length share a batch and the
        results equal the unbatched ones (to ~1e-5). `max_pad_ratio > 1` opts in to padded
        batching (faster for many similar-length clips, slightly different numbers)."""
        feats = [self._features(w) for w in wavs]
        order = sorted(range(len(feats)), key=lambda i: len(feats[i]))
        out: list[np.ndarray | None] = [None] * len(feats)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and len(feats[order[j]]) <= max_pad_ratio * len(feats[order[i]]):
                j += 1
            group = order[i:j]
            tmax = max(len(feats[k]) for k in group)
            x = np.full((len(group), tmax, 80), SILENCE_FEAT, dtype=np.float32)
            lens = np.zeros(len(group), dtype=np.int64)
            for b, k in enumerate(group):
                x[b, : len(feats[k])] = feats[k]
                lens[b] = len(feats[k])
            lp, n = self.session.run(None, {"x": x, "x_lens": lens})
            for b, k in enumerate(group):
                out[k] = self._fix(lp[b, : int(n[b])])
            i = j
        return out  # type: ignore[return-value]

    # ------------------------------------------------------------------ decoding
    def greedy(self, lp: np.ndarray) -> list[tuple[str, int, int]]:
        return greedy_tokens(lp, self.tokens)


def greedy_tokens(lp: np.ndarray, tokens=None) -> list[tuple[str, int, int]]:
    """Collapsed greedy decode -> [(token, start_frame, end_frame_exclusive)]; ids 0-3 are
    non-speech."""
    tokens = tokens or load_tokens()
    if len(lp) == 0:
        return []
    ids = np.asarray(lp).argmax(-1)
    change = np.flatnonzero(np.diff(ids)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(ids)]])
    return [(tokens[ids[s]], int(s), int(e)) for s, e in zip(starts, ends) if ids[s] >= NON_SPEECH]
