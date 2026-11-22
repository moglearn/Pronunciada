"""Round 19 (Korean): score item sets through the public Scorer and keep what the offline fits need.

    python -m bench.round19ko.collect --set dev   [--tag base]   # the planted dev items (plant.py)
    python -m bench.round19ko.collect --set fleurs-train         # FLEURS ko TRAIN natives (head fitting)
    python -m bench.round19ko.collect --set fleurs-test          # FLEURS ko TEST natives (final guard only)

One process, 2 threads (the laptop is shared). Per item: per word the probability at strictness
0.5 and its parts, the word-model inputs (feature table W); per sound the symbol, word, vowel
class of its syllable (nucleus only), tense flag, shown log-odds and the pooled enc1|enc3|enc5
vector (float16; the input of the attribute / contrast heads). Pickle: $CACHE/round19/korean/
collect/<set>-<tag>.pkl, written in chunks (resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from bench.round19ko.cv_pool import work

VCLASS = {"ㅏ": "a", "ㅑ": "a", "ㅘ": "a", "ㅓ": "ʌ", "ㅕ": "ʌ", "ㅝ": "ʌ", "ㅗ": "o", "ㅛ": "o", "ㅜ": "u",
          "ㅠ": "u", "ㅡ": "ɯ", "ㅣ": "i", "ㅟ": "i", "ㅐ": "e", "ㅒ": "e", "ㅔ": "e", "ㅖ": "e", "ㅙ": "e",
          "ㅚ": "e", "ㅞ": "e"}


def vowel_class(syl_text: str) -> str | None:
    from pronunciada.text.ko import decompose

    d = decompose(syl_text[0]) if syl_text else None
    return VCLASS.get(d[1]) if d else None


def out_dir() -> Path:
    d = work() / "collect"
    d.mkdir(parents=True, exist_ok=True)
    return d


def items(name: str) -> list[dict]:
    from bench.pipeline.common import records

    if name == "dev":
        return [json.loads(x) for x in (work() / "dev_items.jsonl").read_text(encoding="utf-8").splitlines()]
    if name == "fleurs-train":
        return [{"item": r["id"], "audio": r["audio"], "target": r["text"], "src": "fleurs-train",
                 "speaker": f"fleurs-sent-{r['labels']['fleurs_id']}", "label": None}
                for r in records("fleurs", split="train", lang="ko")]
    if name == "fleurs-val-cor":                  # bench.eval_languages controlled errors (guard)
        from bench.eval_languages import KINDS

        return [{"item": f"{r['id']}#{k}", "id": r["id"], "audio": r["audio"], "target": r["text"], "cor_kind": k,
                 "src": "fleurs-val", "speaker": f"fleurs-sent-{r['labels']['fleurs_id']}", "label": None}
                for r in records("fleurs-validation", lang="ko") for k in ("clean",) + KINDS]
    if name == "fleurs-test-cor":                 # HELD-OUT guard: FLEURS ko TEST + eval_languages errors
        from bench.eval_languages import KINDS

        return [{"item": f"{r['id']}#{k}", "id": r["id"], "audio": r["audio"], "target": r["text"], "cor_kind": k,
                 "src": "fleurs-test", "speaker": f"fleurs-sent-{r['labels']['fleurs_id']}", "label": None}
                for r in records("fleurs", split="test", lang="ko") for k in ("clean",) + KINDS]
    if name == "swap-test":                       # HELD-OUT guard: bench.eval_swap fleurs-swap TEST, ko
        from bench.data import load as dload

        return [{"item": r["id"], "id": r["labels"]["src_id"], "audio": r["audio"], "target": r["text"],
                 "src": "swap-test", "speaker": r["labels"]["src_id"],
                 "label": None if r["labels"]["variant"] == "orig" else
                 {"bucket": r["labels"]["variant"], "word_index": r["labels"]["word_index"]}}
                for r in dload("fleurs-swap", split="test") if r["lang"] == "ko"]
    if name == "audit-test":                      # HELD-OUT: contrast audit, FLEURS TEST (round-17 plan, read only)
        from bench.round17.collect import clips, root

        plan = json.loads((root("test", "ko", "plan") / "plan.json").read_text())
        plan = plan.get("plan", plan)
        out = []
        for r in clips("ko", "test"):
            out.append({"item": f"{r['id']}#clean", "id": r["id"], "audio": r["audio"], "target": r["text"],
                        "src": "audit-test", "speaker": r["id"], "label": None})
            for j, sd in enumerate(plan.get(r["id"], [])):
                out.append({"item": f"{r['id']}#e{j}", "id": r["id"], "audio": r["audio"], "target": r["text"],
                            "src": "audit-test", "speaker": r["id"], "label": None, "site": sd})
        return out
    if name == "zeroth-test":                     # Zeroth test split: speakers disjoint from its train (native pool)
        import pyarrow.parquet as pq

        f = work() / "zeroth" / "test-00000-of-00001.parquet"
        t = pq.read_table(f, columns=["id", "speaker_id", "text"])
        return [{"item": f"zeroth-{uid}", "audio": f"zeroth:{f}:{i}", "target": text, "src": "zeroth-test",
                 "speaker": f"zeroth-{spk}", "label": None}
                for i, (uid, spk, text) in enumerate(zip(*[t[c].to_pylist() for c in ("id", "speaker_id", "text")]))]
    if name == "zeroth-train2":                   # Zeroth train shards 2-5 (more speakers), every 6th clip
        import pyarrow.parquet as pq

        out = []
        for f in sorted((work() / "zeroth").glob("train-*.parquet"))[2:6]:
            t = pq.read_table(f, columns=["id", "speaker_id", "text"])
            for i, (uid, spk, text) in enumerate(zip(*[t[c].to_pylist() for c in ("id", "speaker_id", "text")])):
                if i % 6 == 0:
                    out.append({"item": f"zeroth-{uid}", "audio": f"zeroth:{f}:{i}", "target": text,
                                "src": "zeroth-train", "speaker": f"zeroth-{spk}", "label": None})
        return out
    if name == "zeroth-train":                    # Zeroth-Korean (OpenSLR 40, CC BY 4.0), train shards 0-1, every 3rd
        import pyarrow.parquet as pq

        out = []
        for f in sorted((work() / "zeroth").glob("train-*.parquet"))[:2]:
            t = pq.read_table(f, columns=["id", "speaker_id", "text"])
            for i, (uid, spk, text) in enumerate(zip(*[t[c].to_pylist() for c in ("id", "speaker_id", "text")])):
                if i % 3 == 0:
                    out.append({"item": f"zeroth-{uid}", "audio": f"zeroth:{f}:{i}", "target": text,
                                "src": "zeroth-train", "speaker": f"zeroth-{spk}", "label": None})
        return out[:2500]
    if name == "fleurs-test":
        return [{"item": r["id"], "audio": r["audio"], "target": r["text"], "src": "fleurs-test",
                 "speaker": f"fleurs-sent-{r['labels']['fleurs_id']}", "label": None}
                for r in records("fleurs", split="test", lang="ko")]
    raise KeyError(name)


def summarize(res) -> dict:
    from pronunciada.sound_emb import phone_vectors

    c = res._pronunciada  # noqa: SLF001
    ref, ev = c["ref"], c["ev"]
    n = len(ref.phones)
    V = phone_vectors(c["emb"], ev, n).astype(np.float16) if c.get("emb") else None
    ph = {"ipa": [p.ipa for p in ref.phones], "word": np.array([p.word for p in ref.phones]),
          "nuc": np.array([p.is_nucleus for p in ref.phones]),
          "vclass": [vowel_class(ref.syllables[p.syllable].text) if p.is_nucleus else None for p in ref.phones],
          "tense": [getattr(p, "tense", None) for p in ref.phones],
          "zl": np.asarray(c["s_zl"], np.float32), "zs": np.asarray(c["s_zs"], np.float32),
          "contrast_z": np.asarray(c.get("contrast_z"), np.float32) if c.get("contrast_z") is not None else None,
          "llr_sub": np.asarray(c["tab"]["llr_sub"], np.float32), "llr_del": np.asarray(c["tab"]["llr_del"], np.float32),
          "n_frames": np.asarray(c["tab"]["n_frames"], np.float32),
          "start": np.array([s.start_frame for s in ev.slots[:n]]), "end": np.array([s.end_frame for s in ev.slots[:n]]),
          "heard": [s.heard for s in ev.slots[:n]], "peak": np.array([s.peak_post for s in ev.slots[:n]], np.float32),
          "alt": np.array([s.alt_index for s in ev.slots[:n]]),
          "llr_ins": (np.asarray(ev.llr_ins, np.float32) if getattr(ev, "llr_ins", None) is not None
                      else np.full(n, np.nan, np.float32)),
          "syl": np.array([p.syllable for p in ref.phones]), "T": int(ev.n_frames),
          "V": V}
    W = {k: np.asarray(v, np.float32) for k, v in c["W"].items() if np.asarray(v).dtype.kind in "fib"}
    wd = {"text": [w.text for w in res.words], "span": [tuple(w.char_span) for w in ref.words],
          "prob": np.array([w.prob for w in res.words]), "sounds": np.array([w.sounds_prob for w in res.words]),
          "present": np.array([np.nan if w.present_prob is None else w.present_prob for w in res.words]),
          "w_zl": np.asarray(c["w_zl"], np.float32), "w_zs": np.asarray(c["w_zs"], np.float32), "W": W}
    return {"phones": ph, "words": wd, "prob": res.prob, "dur": c["dur"], "warnings": res.warnings,
            "ms": res.timing_ms.get("total")}


def run(name: str, tag: str, limit: int | None = None) -> Path:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "2")
    from bench.pipeline.common import iter_parquet_audio
    from pronunciada import Scorer

    sc = Scorer("ko", threads=2)
    its = items(name)[:limit]
    p = out_dir() / f"{name}-{tag}.pkl"
    out = pickle.loads(p.read_bytes()) if p.exists() else {}
    todo = [it for it in its if it["item"] not in out]
    pq = [it for it in todo if str(it["audio"]).startswith("parquet:")]
    t0 = time.time()

    def one(it, audio):
        try:
            text, extra = it["target"], {}
            if it.get("site"):
                import zlib

                from bench.audit.contrasts import Site, apply

                site = Site(**it["site"])
                e = apply(sc.g2p(text), site, np.random.default_rng(zlib.crc32(f"{it['id']}/{it['site']}".encode())))
                text, extra = e.ref, {"site": it["site"], "edit_word": int(e.word), "edit_flag": list(e.flag),
                                      "partner_in_alts": bool(e.partner_in_alts)}
            if it.get("cor_kind") not in (None, "clean"):
                from bench.corrupt import NotApplicable, corrupt
                from bench.pipeline.common import stable_seed
                from bench.pipeline.evalctl import inventory

                try:
                    cor = corrupt(sc.g2p(text), [it["cor_kind"]], inventory("ko"), n_errors=2,
                                  seed=stable_seed(it["id"], it["cor_kind"], "features-v1"), strict=False)
                except NotApplicable:
                    return
                if not cor.errors:
                    return
                text, extra = cor.ref, {"word_wrong": list(cor.word_wrong), "phone_wrong": list(cor.phone_wrong)}
            res = sc.score(audio, text)
            out[it["item"]] = {"meta": {k: it.get(k) for k in ("src", "speaker", "label", "target", "id", "cor_kind")},
                               **extra, **summarize(res)}
        except ValueError as e:
            out[it["item"]] = {"meta": {k: it.get(k) for k in ("src", "speaker", "label", "target", "id")},
                               "error": str(e)}

    k = 0
    if pq:
        path = pq[0]["audio"].split(":")[1]
        by = {}
        for it in pq:
            by.setdefault(int(it["audio"].split(":")[2]), []).append(it)
        for i, wav in iter_parquet_audio(path, set(by)):
            for it in by[i]:
                one(it, (wav, 16000))
                k += 1
    zq = [it for it in todo if str(it["audio"]).startswith("zeroth:")]
    if zq:
        import io

        import pyarrow.parquet as papq
        import soundfile as sf

        from bench.data.common import resample

        byf = {}
        for it in zq:
            _, f, i = it["audio"].rsplit(":", 2)
            byf.setdefault(f, {})[int(i)] = it
        for f, want in byf.items():
            pf = papq.ParquetFile(f)
            row = 0
            for g in range(pf.num_row_groups):
                col = pf.read_row_group(g, columns=["audio"]).column("audio").to_pylist()
                for j, a in enumerate(col):
                    if row + j in want:
                        x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
                        one(want[row + j], (resample(x.mean(axis=1), sr), 16000))
                        k += 1
                row += len(col)
                del col
            p.write_bytes(pickle.dumps(out))
            print(f"{name}-{tag}: {len(out)} / {len(its)} [{time.time() - t0:.0f}s]", flush=True)
    for it in todo:
        if it in pq or it in zq:
            continue
        one(it, it["audio"])
        k += 1
        if k % 200 == 0:
            p.write_bytes(pickle.dumps(out))
            print(f"{name}-{tag}: {len(out)} / {len(its)} [{time.time() - t0:.0f}s]", flush=True)
    p.write_bytes(pickle.dumps(out))
    print(f"{name}-{tag}: {len(out)} done [{time.time() - t0:.0f}s]", flush=True)
    return p


def load(name: str, tag: str) -> dict:
    return pickle.loads((out_dir() / f"{name}-{tag}.pkl").read_bytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run(a.set, a.tag, a.limit)
