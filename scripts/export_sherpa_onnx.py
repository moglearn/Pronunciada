"""Write the ZIPA-small backbone in sherpa-onnx's offline Zipformer CTC layout.

    python scripts/export_sherpa_onnx.py [--out DIR] [--fp32] [--scoring] [--verify]

Output folder (default ~/.cache/pronunciada/opt/mobile/sherpa-onnx-zipa-small/, outside the repo):
  model.int8.onnx        the downloaded int8 backbone + descriptive metadata (below)
  model.onnx             the fp32 backbone + the same metadata (--fp32; the product default precision)
  tokens.txt             127 lines "<symbol> <id>" (= pronunciada/data/zipa_tokens.txt and the model's own)
  model.int8.scoring.onnx / model.scoring.onnx   (--scoring) the same graph with the encoder
                         activations the scorer reads added as graph outputs (pronunciada.acoustic
                         ALL_LAYER_OUTPUTS); for direct ONNX Runtime use by a phone app, not for sherpa-onnx
  MODEL_CARD.md          description, licence, I/O, conventions and known gaps
  manifest.json          file, bytes, sha256, licence, source; the metadata added; the checks run

How the files are written: the source ONNX file's bytes are copied unchanged and new
`metadata_props` entries (ModelProto field 14, a repeated StringStringEntryProto) are appended at the
end of the file. Protobuf concatenates repeated fields, so the graph and every initializer stay
byte-identical; only keys the model does not carry yet are added (sherpa-onnx's
OfflineZipformerCtcModel reads no metadata key at all; the existing model_type=zipformer2_ctc,
version, model_author and comment are kept). --verify checks, with ONNX Runtime, that the exported
files give bit-identical log-posteriors to the source files on a synthetic and (when the FLEURS
cache is present) two practice clips.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = "zipa-small-crctc-500k"
DEFAULT_OUT = Path("~/.cache/pronunciada/opt/mobile/sherpa-onnx-zipa-small").expanduser()
LICENCE = "CC-BY-4.0"
SOURCE_URL = f"https://huggingface.co/anyspeech/{MODEL}"
METADATA = {
    "model_name": MODEL,
    "url": SOURCE_URL,
    "license": LICENCE,
    "language": "multilingual (IPA phone recognition)",
    "vocab_size": "127",
    "subsampling_factor": "2",
    "frame_shift_ms": "20",
    "feature": "kaldi fbank 80 bins, 16 kHz, 25 ms / 10 ms, povey, preemph 0.97, dither 0, snip_edges false, "
               "low 20 Hz, high -400 Hz, log power, no normalisation",
    "packaged_by": "pronunciada scripts/export_sherpa_onnx.py",
}


# ----------------------------------------------------------------------------- protobuf helpers

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


def metadata_entry(key: str, value: str) -> bytes:
    """ModelProto.metadata_props (field 14) entry: StringStringEntryProto{key = 1, value = 2}."""
    return _ld(14, _ld(1, key.encode("utf-8")) + _ld(2, value.encode("utf-8")))


def read_metadata(path: Path) -> dict:
    """Top-level metadata_props of an ONNX file (skips over the graph without parsing it)."""
    from pronunciada.acoustic import _read_varint

    out = {}
    size = path.stat().st_size
    with open(path, "rb") as f:
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
                if field == 14:
                    f.seek(i + k)
                    entry = f.read(ln)
                    kv, p = {}, 0
                    while p < len(entry):
                        kk, p = _read_varint(entry, p)
                        n, p = _read_varint(entry, p)
                        kv[kk >> 3] = entry[p:p + n].decode("utf-8")
                        p += n
                    out[kv.get(1, "")] = kv.get(2, "")
                i += k + ln
            else:
                raise ValueError(f"unsupported protobuf wire type {wt}")
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def export_file(src: Path, dest: Path, extra: dict) -> dict:
    have = read_metadata(src)
    add = {k: v for k, v in extra.items() if k not in have}
    part = dest.with_suffix(".part")
    shutil.copyfile(src, part)
    with open(part, "ab") as f:
        for k, v in add.items():
            f.write(metadata_entry(k, v))
    os.replace(part, dest)
    return {"kept": have, "added": add}


# ----------------------------------------------------------------------------- checks

def _posteriors(path: Path, feats) -> list:
    import numpy as np
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    out = []
    for x in feats:
        r = sess.run(names, {"x": x[None].astype(np.float32), "x_lens": np.array([len(x)], np.int64)})
        out.append(r)
    return out


def verify_identical(src: Path, dest: Path) -> dict:
    """Graph bytes and log-posteriors of dest equal src's."""
    import numpy as np

    from pronunciada.audio import fbank, load_audio

    n = src.stat().st_size
    prefix_equal = True
    with open(src, "rb") as a, open(dest, "rb") as b:
        left = n
        while left > 0 and prefix_equal:
            k = min(1 << 22, left)
            prefix_equal = a.read(k) == b.read(k)
            left -= k
    wavs = [np.clip(np.sin(np.arange(48000) * 0.05).astype(np.float32) * 0.1, -1, 1)]
    try:
        from bench.opt.common import practice_clips

        wavs += [load_audio(c["audio"]) for c in practice_clips(["en", "zh"], n=1)]
    except Exception:  # noqa: BLE001  (no FLEURS cache)
        pass
    feats = [fbank(w) for w in wavs]
    pa, pb = _posteriors(src, feats), _posteriors(dest, feats)
    equal = all(all(np.array_equal(x, y) for x, y in zip(ra, rb)) for ra, rb in zip(pa, pb))
    return {"source_bytes_are_a_prefix": bool(prefix_equal), "appended_bytes": dest.stat().st_size - n,
            "clips": len(feats), "outputs_bit_identical": bool(equal)}


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--fp32", action="store_true", help="also write model.onnx (fp32)")
    ap.add_argument("--scoring", action="store_true", help="also write the extra-output variants")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    from pronunciada.acoustic import ALL_LAYER_OUTPUTS, PRECISION_FILES, ensure_model_file, write_with_graph_outputs

    out = Path(a.out).expanduser()
    if ROOT in out.resolve().parents or out.resolve() == ROOT:
        raise SystemExit("the export folder must be outside the repository")
    out.mkdir(parents=True, exist_ok=True)
    files, meta, checks = [], {}, {}
    precisions = ["int8"] + (["fp32"] if a.fp32 else [])
    for prec in precisions:
        src = Path(ensure_model_file(MODEL, PRECISION_FILES[prec]))
        dest = out / PRECISION_FILES[prec]
        meta[dest.name] = export_file(src, dest, METADATA)
        files.append((dest, f"{SOURCE_URL}/resolve/main/{PRECISION_FILES[prec]} + metadata entries appended"))
        if a.verify:
            checks[dest.name] = verify_identical(src, dest)
        if a.scoring:
            sd = out / (dest.name[:-len(".onnx")] + ".scoring.onnx")
            write_with_graph_outputs(str(dest), str(sd), list(ALL_LAYER_OUTPUTS.values()))
            files.append((sd, f"{dest.name} + graph outputs {sorted(ALL_LAYER_OUTPUTS)} "
                              "(pronunciada.acoustic.write_with_graph_outputs)"))
            if a.verify:
                checks[sd.name] = {"log_probs_bit_identical_to_" + dest.name: _check_scoring(dest, sd)}
    tok_src = Path(ensure_model_file(MODEL, "tokens.txt"))
    pkg_tok = ROOT / "pronunciada/data/zipa_tokens.txt"
    if tok_src.read_bytes() != pkg_tok.read_bytes():
        raise SystemExit("model tokens.txt differs from pronunciada/data/zipa_tokens.txt")
    shutil.copyfile(pkg_tok, out / "tokens.txt")
    files.append((out / "tokens.txt", f"{SOURCE_URL}/resolve/main/tokens.txt (= pronunciada/data/zipa_tokens.txt)"))
    (out / "MODEL_CARD.md").write_text(model_card(precisions, a.scoring), encoding="utf-8")
    files.append((out / "MODEL_CARD.md", "written by scripts/export_sherpa_onnx.py"))
    man = {"model": MODEL, "layout": "sherpa-onnx offline zipformer CTC (--zipformer-ctc-model, --tokens)",
           "files": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p),
                      "licence": LICENCE if p.suffix in (".onnx", ".txt") else "Apache-2.0", "source": s}
                     for p, s in files],
           "metadata": meta, "checks": checks}
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for f in man["files"]:
        print(f"{f['file']:32s} {f['bytes']:>12,d}  {f['sha256'][:16]}")
    if checks:
        print(json.dumps(checks, indent=1))
    ok = all(v.get("outputs_bit_identical", True) and v.get("source_bytes_are_a_prefix", True)
             and all(x for k, x in v.items() if k.startswith("log_probs_bit_identical")) for v in checks.values())
    return 0 if ok else 1


def _check_scoring(plain: Path, scoring: Path) -> bool:
    import numpy as np

    x = (np.sin(np.arange(300 * 80) * 0.01).reshape(300, 80) * 3 - 8).astype(np.float32)
    a, b = _posteriors(plain, [x])[0], _posteriors(scoring, [x])[0]
    return bool(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))


def model_card(precisions, scoring) -> str:
    names = " and ".join(f"`{'model.int8.onnx' if p == 'int8' else 'model.onnx'}`" for p in precisions)
    return f"""# ZIPA-small (zipa-small-crctc-500k) for sherpa-onnx

Phone recognizer: Zipformer2 CTC acoustic model that outputs International Phonetic Alphabet
symbols (127 tokens: `<blk>`, `<sos/eos>`, `<unk>`, word boundary `▁`, and 123 IPA code points).
Model by the ZIPA authors, published at {SOURCE_URL} under {LICENCE}; this folder repackages
those files unchanged ({names}, `tokens.txt`) with descriptive metadata appended (the graph and
weights are byte-identical; ONNX Runtime outputs are bit-identical to the original files).

## Use with sherpa-onnx

    sherpa-onnx-offline --zipformer-ctc-model=model.int8.onnx --tokens=tokens.txt input.wav

Python: `sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(model="model.int8.onnx", tokens="tokens.txt")`.
Greedy search gives the collapsed phone sequence; on practice clips it equals the token sequence of
the Pronunciada repository's own greedy decode.

## Inputs and outputs

- `x`: float32 `[N, T, 80]` log-mel fbank (kaldi-native-fbank with sherpa-onnx's default feature
  options: 16 kHz, 25 ms window, 10 ms shift, povey window, pre-emphasis 0.97, dither 0,
  snip_edges false, 80 bins, 20 Hz to 7600 Hz, log power, no normalisation); `x_lens`: int64 `[N]`.
- `log_probs`: float32 `[N, T', 127]`, `log_probs_len`: int64 `[N]`, with T' = (T - 7) // 2 (T >= 9).

## Known differences from sherpa-onnx's Zipformer CTC conventions

- Subsampling is 2 (one output frame per 20 ms), while `OfflineZipformerCtcModel::SubsamplingFactor()`
  returns 4: sherpa-onnx timestamps are twice the true time. The start of output frame j is
  0.02 j + 0.03 s.
- Input shorter than 9 fbank frames (about 90 ms) gives no output frame; Pronunciada pads such
  input with log(float32 eps).
- The int8 file uses dynamic quantisation (DynamicQuantizeLinear + MatMulInteger): its outputs are
  sensitive to tiny feature differences. The fp32 file is not.
{"" if not scoring else '''
## Scoring variants

`*.scoring.onnx` carry encoder activations as extra graph outputs (names in manifest.json); Pronunciada's
scorer reads them. sherpa-onnx can load them (extra outputs are computed and ignored), but they are meant
for direct ONNX Runtime use.
'''}
## Licence

{LICENCE} (model and tokens, from the ZIPA release). Attribution: ZIPA, {SOURCE_URL}.
"""


if __name__ == "__main__":
    sys.exit(main())
