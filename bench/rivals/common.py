"""Contract for scoring an outside system on the scoreboard's held-out records.

Every outside system (another model or service) is run on exactly the records of
`bench.eval_zipa_baseline.tasks()` and writes one JSON per record with `write_record`. The scores
are attached to Pronunciada's reference for the prompt (`build_ref(lang, text)`): one score per
reference word, and one per reference sound when the system has that resolution. `bench.eval_external`
then feeds those files to the same test code, item sets, labels, resampling and seeds that produced
results/SCOREBOARD.md and results/zipa_baseline.md.

Score convention: a real number, higher = better pronunciation. Any monotone scale works for the
rank-based tests (AUC, hit rates); the Pearson rows use the numbers as given, so write the scale a
user of that system would actually read (a probability, a mean log-probability, the system's own
0-100 score). Values are clipped to [FLOOR, +inf); FLOOR (-60) is also what a record that the system
could not score receives in the tests, exactly as for the ZIPA baseline, so every valid score must
be above it.

Cache: $PRONUNCIADA_CACHE/external/<system>/<dataset>/<id>.json
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

FLOOR = -60.0
_BUNDLE = None


def _bundle():
    """Off-machine runs (bench/external/cloud/): $PRONUNCIADA_EXT_BUNDLE points to a pickle made by
    bench.external.cloud.make_bundle with the task list and the references already built, so the job
    needs neither the corpora's manifests nor the text front ends. Audio paths in it are relative to
    the cache root."""
    global _BUNDLE
    p = os.environ.get("PRONUNCIADA_EXT_BUNDLE")
    if not p:
        return None
    if _BUNDLE is None:
        import pickle

        from pronunciada.acoustic import cache_root

        with open(p, "rb") as f:
            _BUNDLE = pickle.load(f)
        for ts in _BUNDLE["tasks"].values():
            for t in ts:
                t["audio"] = os.path.join(cache_root(), t["audio"])
    return _BUNDLE


def tasks() -> dict[str, list[dict]]:
    """{dataset: [{dataset, id, lang, audio, text, g2p, dur}]}: every held-out record of the scoreboard."""
    b = _bundle()
    if b is not None:
        return b["tasks"]
    from bench.eval_zipa_baseline import tasks as _t

    return _t()


def build_ref(lang: str, text: str):
    """Pronunciada's reference for a prompt (None when the front end cannot read it; then call write_error)."""
    b = _bundle()
    if b is not None:
        return b["refs"].get((lang, text))
    from bench.eval_zipa_baseline import _ref

    return _ref(lang, text)


def ext_dir(system: str, dataset: str) -> Path:
    from pronunciada.acoustic import cache_root

    return Path(cache_root()) / "external" / system / dataset


def rec_path(system: str, task: dict) -> Path:
    return ext_dir(system, task["dataset"]) / f"{task['id'].replace('/', '_')}.json"


def cached_ok(system: str, task: dict) -> bool:
    p = rec_path(system, task)
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text()).get("g2p") == task["g2p"]
    except (OSError, ValueError):
        return False


def _dump(system: str, task: dict, o: dict) -> None:
    o.update(id=task["id"], lang=task["lang"], g2p=task["g2p"], dur=task["dur"], system=system)
    p = rec_path(system, task)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(o, ensure_ascii=False))
    os.replace(tmp, p)


def _clean(v, n: int, what: str) -> list[float]:
    v = [float(x) for x in v]
    if len(v) != n:
        raise ValueError(f"{what}: {len(v)} scores for {n} reference units")
    if not all(math.isfinite(x) for x in v):
        raise ValueError(f"{what}: non-finite score")
    return [max(x, FLOOR) for x in v]


def write_record(system: str, task: dict, ref, *, word, sentence, phone=None, secondary: dict | None = None,
                 aligned: bool = True, extra: dict | None = None) -> None:
    """Store one record's scores.

    word      one score per ref.words (required)
    sentence  one score for the recording (required)
    phone     one score per ref.phones, or None when the system has no per-sound output: each sound
              then inherits its word's score and the record is marked phone_from_word
    secondary optional second scoring of the same system, {"word": [...], "sentence": x, "phone": [...] | None}
              (reported as a second column); defaults to a copy of the primary
    aligned   False when the system had to fall back (e.g. the text did not fit the audio)
    extra     anything worth keeping (raw response, timings); never read by the tests
    """
    nw, npn = len(ref.words), len(ref.phones)
    w = _clean(word, nw, "word")
    from_word = phone is None
    ph = [w[p.word] for p in ref.phones] if from_word else _clean(phone, npn, "phone")
    s = _clean([sentence], 1, "sentence")[0]
    o = {"phones": [p.ipa for p in ref.phones], "word_text": [x.text for x in ref.words],
         "word_span": [list(x.char_span) for x in ref.words], "word_phones": [list(x.phones) for x in ref.words],
         "aligned": bool(aligned), "phone_from_word": from_word, "gop": ph, "gop_w": w, "gop_s": s,
         "tokens": [ref.text]}                       # the tests only compare this between prompts
    if secondary is None:
        o.update(match=ph, match_w=w, match_s=s, has_secondary=False)
    else:
        w2 = _clean(secondary["word"], nw, "secondary word")
        p2 = secondary.get("phone")
        o.update(match=[w2[p.word] for p in ref.phones] if p2 is None else _clean(p2, npn, "secondary phone"),
                 match_w=w2, match_s=_clean([secondary["sentence"]], 1, "secondary sentence")[0], has_secondary=True)
    if extra:
        o["extra"] = extra
    _dump(system, task, o)


def write_error(system: str, task: dict, reason: str) -> None:
    """The system could not score this record (it is then scored FLOOR in every test, and counted)."""
    _dump(system, task, {"error": str(reason)[:500]})


def todo_groups(system: str, datasets=None, limit: int | None = None) -> list[list[dict]]:
    """Records still to score, grouped by audio file (fleurs-swap has several prompts per recording),
    longest audio first. `limit` keeps the first N audio files per dataset *in manifest order* (for smoke tests)."""
    out = []
    for ds, ts in tasks().items():
        if datasets and ds not in datasets:
            continue
        by = defaultdict(list)
        for t in ts:
            by[t["audio"]].append(t)
        gs = list(by.values())[:limit] if limit else list(by.values())
        out += [g2 for g2 in ([t for t in g if not cached_ok(system, t)] for g in gs) if g2]
    out.sort(key=lambda g: -g[0]["dur"])
    return out


def load_audio(path: str):
    """float32 mono 16 kHz waveform, exactly as Pronunciada reads it."""
    from pronunciada.audio import load_audio as _l

    return _l(path)


def coverage(system: str) -> dict:
    out = {}
    for ds, ts in tasks().items():
        c = {"records": len(ts), "scored": 0, "error": 0, "missing": 0, "not_aligned": 0, "phone_from_word": 0}
        for t in ts:
            p = rec_path(system, t)
            if not p.exists():
                c["missing"] += 1
                continue
            o = json.loads(p.read_text())
            if "error" in o:
                c["error"] += 1
            else:
                c["scored"] += 1
                c["not_aligned"] += not o.get("aligned", True)
                c["phone_from_word"] += bool(o.get("phone_from_word"))
        out[ds] = c
    return out
