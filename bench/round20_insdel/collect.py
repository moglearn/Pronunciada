"""Round 20 (insdel): score dev item sets once with the current runtime and keep small arrays.

    python -m bench.round20_insdel.collect --lang ko --set dev
    python -m bench.round20_insdel.collect --lang ja --set dev

Per item: the shipped word probabilities (and their character spans), per reference sound the
symbol, its allowed alternatives, word index, aligned span, aligner evidence (llr_sub, llr_del,
llr_ins where computed), the collapsed greedy decode with frame spans, and per frame the top-4
tokens of the adapter-corrected log-probabilities with their log-probabilities (float16). Nothing
else (no audio, no full posteriors). Output: $PRONUNCIADA_CACHE/round20/insdel/collect/<lang>-<set>.pkl,
resumable, checkpointed every 100 items. One process, 2 threads.

Sets:
* ko: dev (round-19 planted dev set, CV dev + FLEURS validation), val-cor (FLEURS validation with the
  bench.eval_languages controlled errors; clean = natives), zeroth-test (native pool).
* ja: dev, devnat, devswap, cvdev (the round-19 Japanese dev set and its dev guard sets).
* fr / es / de / en: dev, valnat (bench/round19_rollout dev set and FLEURS validation natives).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np


def cache_root() -> Path:
    return Path(os.environ.get("PRONUNCIADA_CACHE", Path.home() / ".cache" / "pronunciada"))


def work() -> Path:
    d = cache_root() / "round20" / "insdel"
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_path(lang: str, name: str) -> Path:
    d = work() / "collect"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{lang}-{name}.pkl"


# ----------------------------------------------------------------------------- item lists

def _ja_rows(fname: str) -> list[dict]:
    keep = ("id", "audio", "text", "group", "source", "units", "label", "labels", "side", "prompt")
    src = cache_root() / "round19" / "japanese" / fname
    out = []
    for x in src.read_text(encoding="utf-8").splitlines():
        if not x.strip():
            continue
        r = json.loads(x)
        out.append({"item": r["id"], "audio": r["audio"], "text": r["text"],
                    "meta": {k: r.get(k) for k in keep if k != "audio"}})
    return out


def items(lang: str, name: str) -> list[dict]:
    if lang == "ko":
        from bench.round19ko import collect as KC

        src = {"dev": "dev", "val-cor": "fleurs-val-cor", "zeroth-test": "zeroth-test"}[name]
        out = []
        for it in KC.items(src):
            out.append({"item": it["item"], "audio": it["audio"], "text": it["target"],
                        "cor": it.get("cor_kind"), "cor_id": it.get("id"),
                        "meta": {k: it.get(k) for k in ("src", "speaker", "label", "target", "id", "cor_kind")}})
        return out
    if lang == "ja":
        f = {"dev": "dev2_r19.jsonl", "devnat": "devnat2_r19.jsonl", "devswap": "devswap2_r19.jsonl",
             "cvdev": "cvdev2_r19.jsonl"}[name]
        return _ja_rows(f)
    if lang == "en" and name == "so762-train":
        from bench.data import load as dload

        return [{"item": r["id"], "audio": r["audio"], "text": r["text"], "so762": r,
                 "meta": {"speaker": r["speaker"], "src": "so762-train"}} for r in dload("speechocean762", split="train")]
    if name == "val-cor":           # FLEURS validation + the bench.eval_languages controlled errors
        from bench.eval_languages import KINDS
        from bench.pipeline.common import records

        return [{"item": f"{r['id']}#{k}", "audio": r["audio"], "text": r["text"], "cor": k, "cor_id": r["id"],
                 "meta": {"src": "fleurs-val", "speaker": f"fleurs-sent-{r['labels']['fleurs_id']}", "id": r["id"],
                          "cor_kind": k, "label": None}}
                for r in records("fleurs-validation", lang=lang) for k in ("clean",) + KINDS]
    raise KeyError((lang, name))


def so762_labels(rec: dict, ref) -> dict:
    """Per OUR sound: the expert 0-2 score (NaN if unmatched) and whether the expert marked it
    deleted (<DEL>); per OUR word the mean expert word accuracy (bench.round19.extract_en rules)."""
    from bench.pipeline.labels import align_reference, arpa_to_ours

    W = rec["labels"]["words"]
    tp = [[arpa_to_ours(p) for p in w["phones"]] for w in W]
    o2t, wpairs = align_reference(ref, [w["text"] for w in W], tp)
    n = len(ref.phones)
    ph = np.full(n, np.nan)
    dl = np.zeros(n, bool)
    dels = {(j, m["index"]) for j, w in enumerate(W) for m in w.get("mispronunciations", [])
            if m.get("pronounced_phone") == "<DEL>"}
    for i, (j, k) in o2t.items():
        ph[i] = float(W[j]["phones_accuracy"][k])
        dl[i] = (j, k) in dels
    tw = {}
    for i, (j, _) in o2t.items():
        tw.setdefault(ref.phones[i].word, set()).add(j)
    for i, j in wpairs:
        tw.setdefault(i, set()).add(j)
    wd = np.full(len(ref.words), np.nan)
    for i, js in tw.items():
        wd[i] = float(np.mean([W[j]["accuracy"] for j in js]))
    return {"phone_score": ph, "phone_del": dl, "word_acc": wd}


# ----------------------------------------------------------------------------- audio

def iter_audio(todo: list[dict]):
    """Yield (item, waveform) grouping parquet / zeroth rows by file (each file read once)."""
    from pronunciada.audio import load_audio

    by: dict[tuple[str, str], dict[int, list[dict]]] = {}
    plain = []
    for it in todo:
        a = str(it["audio"])
        if a.startswith("parquet:") or a.startswith("zeroth:"):
            kind, rest = a.split(":", 1)
            path, row = rest.rsplit(":", 1)
            by.setdefault((kind, path), {}).setdefault(int(row), []).append(it)
        else:
            plain.append(it)
    for (kind, path), want in by.items():
        if kind == "parquet":
            from bench.pipeline.common import iter_parquet_audio

            for i, wav in iter_parquet_audio(path, set(want)):
                for it in want[i]:
                    yield it, wav
        else:
            import pyarrow.parquet as papq
            import soundfile as sf

            from bench.data.common import resample

            pf = papq.ParquetFile(path)
            row = 0
            for g in range(pf.num_row_groups):
                col = pf.read_row_group(g, columns=["audio"]).column("audio").to_pylist()
                for j, a in enumerate(col):
                    if row + j in want:
                        x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
                        w = resample(x.mean(axis=1), sr)
                        for it in want[row + j]:
                            yield it, w
                row += len(col)
                del col
    for it in plain:
        try:
            w = load_audio(it["audio"])
        except Exception:  # noqa: BLE001  (missing / unreadable file: recorded as an error)
            w = None
        yield it, w


# ----------------------------------------------------------------------------- summary

def summarize(res) -> dict:
    c = res._pronunciada  # noqa: SLF001
    ref, ev, lpa = c["ref"], c["ev"], np.asarray(c["lpa"], np.float32)
    n = len(ref.phones)
    T = lpa.shape[0]
    top = np.argpartition(-lpa, 4, axis=1)[:, :4]
    tv = np.take_along_axis(lpa, top, axis=1)
    o = np.argsort(-tv, axis=1)
    top = np.take_along_axis(top, o, axis=1).astype(np.uint8)
    tv = np.take_along_axis(tv, o, axis=1).astype(np.float16)
    blank = np.logaddexp(lpa[:, 0], lpa[:, 3]).astype(np.float16)
    ins = getattr(ev, "llr_ins", None)
    ph = {"ipa": [p.ipa for p in ref.phones], "alts": [[(a, float(w)) for a, w in p.alts] for p in ref.phones],
          "word": np.array([p.word for p in ref.phones], np.int32),
          "start": np.array([s.start_frame for s in ev.slots[:n]], np.int32),
          "end": np.array([s.end_frame for s in ev.slots[:n]], np.int32),
          "alt": np.array([s.alt_index for s in ev.slots[:n]], np.int16),
          "llr_sub": np.array([s.llr_sub for s in ev.slots[:n]], np.float32),
          "llr_del": np.array([s.llr_del for s in ev.slots[:n]], np.float32),
          "llr_ins": (np.asarray(ins, np.float32) if ins is not None else None)}
    words = {"text": [w.text for w in res.words], "span": [tuple(w.char_span) for w in ref.words],
             "phones": [tuple(w.phones) for w in ref.words],
             "prob": np.array([w.prob for w in res.words]), "sounds": np.array([w.sounds_prob for w in res.words])}
    return {"phones": ph, "words": words, "prob": res.prob, "T": T, "L_can": float(ev.L_can),
            "free": [(t, int(a), int(b)) for t, a, b in ev.free_tokens],
            "top": top, "topv": tv, "blank": blank, "ms": res.timing_ms.get("total"),
            "s_zl": np.asarray(c["s_zl"], np.float32), "s_zs": np.asarray(c["s_zs"], np.float32),
            "coach_zl": np.asarray(c.get("coach_zl", c["s_zl"]), np.float32),
            "coach_zs": np.asarray(c.get("coach_zs", c["s_zs"]), np.float32),
            "w_zl": np.asarray(c["w_zl"], np.float32), "w_zs": np.asarray(c["w_zs"], np.float32)}


def run(lang: str, name: str, limit: int | None = None) -> Path:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "2")
    import warnings

    warnings.simplefilter("ignore")
    from pronunciada import Scorer

    sc = Scorer(lang, threads=2)
    its = items(lang, name)[:limit]
    p = out_path(lang, name)
    out = pickle.loads(p.read_bytes()) if p.exists() else {}
    todo = [it for it in its if it["item"] not in out]
    t0 = time.time()
    k = 0
    for it, wav in iter_audio(todo):
        extra = {}
        if wav is None:
            out[it["item"]] = {"meta": it["meta"], "error": "audio unreadable"}
            continue
        try:
            text = it["text"]
            if it.get("cor") not in (None, "clean"):
                from bench.corrupt import NotApplicable, corrupt
                from bench.pipeline.common import stable_seed
                from bench.pipeline.evalctl import inventory

                try:
                    cor = corrupt(sc.g2p(text), [it["cor"]], inventory(lang), n_errors=2,
                                  seed=stable_seed(it["cor_id"], it["cor"], "features-v1"), strict=False)
                except NotApplicable:
                    continue
                if not cor.errors:
                    continue
                text, extra = cor.ref, {"word_wrong": list(cor.word_wrong), "phone_wrong": list(cor.phone_wrong)}
            res = sc.score((wav, 16000), text)
            if it.get("so762") is not None:
                extra["labels"] = so762_labels(it["so762"], res._pronunciada["ref"])  # noqa: SLF001
            out[it["item"]] = {"meta": it["meta"], **extra, **summarize(res)}
        except ValueError as e:
            out[it["item"]] = {"meta": it["meta"], "error": str(e)}
        k += 1
        if k % 100 == 0:
            p.write_bytes(pickle.dumps(out))
            print(f"{lang}-{name}: {len(out)} / {len(its)} [{time.time() - t0:.0f}s]", flush=True)
    p.write_bytes(pickle.dumps(out))
    print(f"{lang}-{name}: {len(out)} done [{time.time() - t0:.0f}s]", flush=True)
    return p


def load(lang: str, name: str) -> dict:
    return pickle.loads(out_path(lang, name).read_bytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--set", required=True, nargs="+")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    for s in a.set:
        run(a.lang, s, a.limit)
    sys.exit(0)
