"""Prepare prosody training / eval material: reference, ZIPA alignment spans, pitch track.

    .venv/bin/python -m bench.prosody.prep aishell3 train
    .venv/bin/python -m bench.prosody.prep fleurs train zh
    .venv/bin/python -m bench.prosody.prep fleurs_val dev zh
    .venv/bin/python -m bench.prosody.prep zhtw test
    .venv/bin/python -m bench.prosody.prep speechocean762 test en

Output: ~/.cache/pronunciada/prosody/<dataset>-<lang>-<split>.pkl = list of dicts
  id, speaker, dur, ref (pronunciada Reference), spans [(s, e)] per phone in ZIPA frames,
  track (f0 f16, voiced bool, voicing f16, energy_db f16), noisy (same, from the audio with
  added noise at a random 12-30 dB SNR; train splits only), ext (dataset labels, e.g. AISHELL tones
  per syllable), L_can.
ZIPA posteriors come from bench.postcache (shared cache).
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402

from bench import postcache  # noqa: E402
from pronunciada.acoustic import cache_root  # noqa: E402
from pronunciada.align import _spans, emission_table  # noqa: E402
from pronunciada.audio import load_audio  # noqa: E402
from pronunciada.lattice import build_lattice, slots_from_reference  # noqa: E402
from pronunciada.prosody.f0 import track  # noqa: E402
from pronunciada.text import get_g2p  # noqa: E402

OUT = Path(cache_root()) / "prosody"


def out_path(dataset: str, lang: str, split: str) -> Path:
    return OUT / f"{dataset}-{lang}-{split}.pkl"


FLEURS_CFG = {"en": "en_us", "zh": "cmn_hans_cn", "ja": "ja_jp", "es": "es_419", "fr": "fr_fr",
              "de": "de_de", "it": "it_it", "pt": "pt_br", "ru": "ru_ru", "ko": "ko_kr"}


def fleurs_validation(lang: str):
    """FLEURS validation (dev set) rows; ids follow the shared posterior-cache convention
    `fleurs-<lang>-validation-<wav stem>` (dataset "fleurs-validation"); FLEURS' own `id` is the
    sentence id and is shared by several recordings, so it must not be used as a key."""
    import io

    import pyarrow.parquet as pq
    import soundfile as sf

    path = Path(cache_root()) / "data" / "fleurs" / FLEURS_CFG[lang] / "validation-0.parquet"
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=16, columns=["id", "path", "num_samples", "raw_transcription", "audio"]):
        for row in batch.to_pylist():
            if row["num_samples"] / 16000 > 30:
                continue
            wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(1)
            assert sr == 16000
            yield {"id": f"fleurs-{lang}-validation-{Path(row['path']).stem}", "text": row["raw_transcription"],
                   "speaker": None, "lang": lang, "labels": {"fleurs_id": row["id"]}, "wav": wav}


def records(dataset: str, split: str, lang: str | None):
    if dataset == "fleurs_val":
        yield from fleurs_validation(lang)
        return
    from bench.data import load

    for r in load(dataset, split=split, lang=lang):
        yield r


def pack(tr):
    return {"f0": tr.f0.astype(np.float16), "voiced": tr.voiced, "voicing": tr.voicing.astype(np.float16),
            "energy_db": tr.energy_db.astype(np.float16)}


def aishell_ext(ref, labels):
    """AISHELL tone/pinyin per syllable (its first character's hanzi entry; AISHELL groups erhua
    as one entry, e.g. 孩儿 hair2), None if the texts do not match."""
    hz, py = labels.get("hanzi", []), labels.get("pinyin", [])
    if len(hz) != len(py) or "".join(s.text for s in ref.syllables) != "".join(hz):
        return None
    hz_of = [k for k, h in enumerate(hz) for _ in h]
    out, c = [], 0
    for s in ref.syllables:
        k = hz_of[c]
        p = py[k]
        out.append({"pinyin": p, "tone": int(p[-1]) if p[-1:].isdigit() else None,
                    "erhua": len(hz[k]) > 1 or len(s.text) > 1})
        c += len(s.text)
    return out


def main(dataset: str, split: str, lang: str | None = None):
    lang_c = lang or "zh"
    g2p = get_g2p(lang_c)
    path = out_path(dataset, lang_c, split)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    noisy = split == "train"
    out, fails = [], 0
    t0 = time.time()
    pc_name = "fleurs-validation" if dataset == "fleurs_val" else dataset
    for k, r in enumerate(records(dataset, split, lang)):
        try:
            wav = r["wav"] if "wav" in r else load_audio(r["audio"])
            ref = g2p(r["text"])
            if not ref.phones:
                fails += 1
                continue
            lp = postcache.get_posteriors(pc_name, r["id"], wav)
            slots = slots_from_reference(ref)
            lat = build_lattice(slots)
            tab = emission_table(lp)
            from pronunciada.align import viterbi

            score, _ = viterbi(lat, tab)
            spans = _spans(lat, slots, tab, len(lp), bool(np.isfinite(score)))
            if not np.isfinite(score):
                fails += 1
                continue
            item = {"id": r["id"], "speaker": r.get("speaker"), "dur": len(wav) / 16000, "ref": ref,
                    "spans": [(s, e) for s, e, _ in spans], "track": pack(track(wav)),
                    "L_can_per_frame": float(score) / max(len(lp), 1), "T_zipa": len(lp)}
            if noisy:
                snr = rng.uniform(12, 30)
                p = float(np.mean(wav ** 2)) + 1e-10
                n = rng.standard_normal(len(wav)).astype(np.float32)
                if rng.random() < 0.5:            # pink-ish noise
                    n = np.cumsum(n) * 0.05
                    n -= np.convolve(n, np.ones(401) / 401, mode="same")
                n *= np.sqrt(p / (np.mean(n ** 2) + 1e-12) / 10 ** (snr / 10))
                item["noisy"] = pack(track(np.clip(wav + n, -1, 1)))
            if dataset == "aishell3":
                item["ext"] = aishell_ext(ref, r["labels"])
            out.append(item)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"fail {r['id']}: {e!r}", file=sys.stderr)
        if k % 200 == 0:
            print(f"{dataset} {split} {lang_c}: {k} done, {fails} fails, {time.time() - t0:.0f}s", flush=True)
    with open(path.with_suffix(".tmp"), "wb") as f:
        pickle.dump(out, f, protocol=5)
    os.replace(path.with_suffix(".tmp"), path)
    print(f"wrote {path}: {len(out)} utts, {fails} fails, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2] if len(a) > 2 else None)
