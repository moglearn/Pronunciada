"""Objective check for a text front-end: how well does its reference match what ZIPA hears
from native speakers?  Uses the FLEURS *validation* split (our dev set; train/test are reserved
for benchmarks).  Self-contained on purpose (no dependency on pronunciada.acoustic).

    .venv/bin/python scripts/g2p_eval.py en --n 150
    .venv/bin/python scripts/g2p_eval.py zh --n 150 --show 5

Prints token error rate (reference tokens vs ZIPA greedy tokens), the most frequent
substitutions / deletions / insertions (ref -> heard), and failures.  Lower is better; what
remains after conventions are fixed is the model's own error plus natural reductions.
"""
from __future__ import annotations

import argparse
import collections
import io
import os
import sys
import urllib.request

import numpy as np

CACHE = os.environ.get("PRONUNCIADA_CACHE", os.path.expanduser("~/.cache/pronunciada"))
FLEURS = {"en": "en_us", "zh": "cmn_hans_cn", "ja": "ja_jp", "es": "es_419", "fr": "fr_fr",
          "de": "de_de", "it": "it_it", "pt": "pt_br", "ru": "ru_ru", "ko": "ko_kr"}


def fleurs_validation(lang: str):
    """Yield (id, raw_transcription, wav16k float32). Downloads one parquet (~150-250 MB) once."""
    import pyarrow.parquet as pq
    import soundfile as sf

    cfg = FLEURS[lang]
    path = os.path.join(CACHE, "data", "fleurs", cfg, "validation-0.parquet")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"https://huggingface.co/api/datasets/google/fleurs/parquet/{cfg}/validation/0.parquet"
        print(f"downloading {url}", file=sys.stderr)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=16, columns=["id", "raw_transcription", "audio"]):
        for row in batch.to_pylist():
            wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(1)
            assert sr == 16000, sr
            yield row["id"], row["raw_transcription"], wav


class Zipa:
    def __init__(self, name="zipa-small-crctc-500k", file="model.onnx", threads=4):
        import onnxruntime as ort

        d = os.path.join(CACHE, "models", name)
        self.tokens = [l.split()[0] for l in open(os.path.join(d, "tokens.txt"), encoding="utf-8")]
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(os.path.join(d, file), so, providers=["CPUExecutionProvider"])

    @staticmethod
    def fbank(wav):
        import kaldi_native_fbank as knf

        o = knf.FbankOptions()
        o.frame_opts.dither = 0.0
        o.frame_opts.snip_edges = False
        o.frame_opts.samp_freq = 16000
        o.mel_opts.num_bins = 80
        o.mel_opts.high_freq = -400
        f = knf.OnlineFbank(o)
        f.accept_waveform(16000, wav.tolist())
        f.input_finished()
        return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)

    def greedy(self, wav) -> list[str]:
        x = self.fbank(wav)
        lp = self.sess.run(None, {"x": x[None], "x_lens": np.array([len(x)], dtype=np.int64)})[0][0]
        ids = lp.argmax(-1)
        return [self.tokens[i] for j, i in enumerate(ids) if i > 3 and (j == 0 or ids[j - 1] != i)]


def align(ref: list[str], hyp: list[str]):
    """Levenshtein alignment -> list of (ref_tok | None, hyp_tok | None)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            out.append((ref[i - 1], None)); i -= 1
        else:
            out.append((None, hyp[j - 1])); j -= 1
    return out[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--show", type=int, default=3, help="print this many example pairs")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--model", default="zipa-small-crctc-500k")
    ap.add_argument("--file", default="model.onnx")
    a = ap.parse_args()

    from pronunciada.text import get_g2p

    g2p, zipa = get_g2p(a.lang), Zipa(a.model, a.file)
    errs = tot = fails = 0
    subs, dels, ins, warns = (collections.Counter() for _ in range(4))
    ref_count = collections.Counter()
    for k, (uid, text, wav) in enumerate(fleurs_validation(a.lang)):
        if k >= a.n:
            break
        try:
            ref = g2p(text)
            ref.validate()
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {uid} {text!r}: {type(e).__name__}: {e}")
            continue
        for w in ref.warnings:
            warns[w] += 1
        r = [c for p in ref.phones for c in p.ipa]
        h = zipa.greedy(wav)
        if k < a.show:
            print(f"TEXT {text}\n REF {' '.join(''.join(ph.ipa for ph in ref.phones[w.phones[0]:w.phones[1]]) for w in ref.words)}\n HYP {''.join(h)}")
        for x, y in align(r, h):
            if x is not None:
                tot += 1
                ref_count[x] += 1
            if x == y:
                continue
            errs += 1
            (dels if y is None else ins if x is None else subs)[(x, y)] += 1
    print(f"\n[{a.lang}] utts={k} failed={fails} ref_tokens={tot} token_error_rate={errs / max(tot, 1):.3f} "
          f"(sub={sum(subs.values()) / max(tot, 1):.3f} del={sum(dels.values()) / max(tot, 1):.3f} ins={sum(ins.values()) / max(tot, 1):.3f})")
    print("top substitutions ref->heard (count, share of that ref token):")
    print("  " + "  ".join(f"{x}->{y}:{c}({c / ref_count[x]:.0%})" for (x, y), c in subs.most_common(a.top)))
    print("top deletions (ref token missing):")
    print("  " + "  ".join(f"{x}:{c}({c / ref_count[x]:.0%})" for (x, _), c in dels.most_common(a.top)))
    print("top insertions (heard, not in ref):")
    print("  " + "  ".join(f"{y}:{c}" for (_, y), c in ins.most_common(a.top)))
    if warns:
        print("warnings:", "; ".join(f"{w} x{c}" for w, c in warns.most_common(15)))


if __name__ == "__main__":
    main()
