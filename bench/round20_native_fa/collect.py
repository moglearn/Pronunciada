"""Score a practice set with the current code and keep, per word, everything the false-alarm audit reads.

    python -m bench.round20_native_fa.collect --lang en [--part 0/2] [--limit N]

Each item goes through Scorer(lang).score(audio, target) with default settings, then
pronunciada.compat.azure.to_azure and the benchmark's Azure reduction (bench.third_party.lingolix),
exactly as the Lingolix runner does, so "words" is what the benchmark-style metric reads.
Beside it, per reference word: the probability and its factors (sounds, stress, length, contrast,
presence, the sound-identity step where the language ships one), times, and per sound the symbol,
chosen alternative, shown probability, what the decoder heard, llr_sub / llr_del and native z.
Diagnostic only (does not change any score): the extra-sound evidence llr_ins from a second
alignment pass with insertion=True on the same posteriors, and the free greedy decode of the clip.
Nothing under pronunciada/ is modified; the recorders below wrap two module functions in this process.
Output: $CACHE/round20/nativefa/dumps/<lang>.jsonl (resumable; parts append to <lang>.p<k>.jsonl).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from bench.round20_native_fa.rebuild import ROOT, manifest_path

_REC: list = []
DUMPS = ROOT / "dumps"


def items(lang: str) -> list[dict]:
    rows = [json.loads(l) for l in manifest_path(lang).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    if lang == "ko":
        asr = {}
        p = ROOT / "korean" / "asr.jsonl"
        if p.exists():
            for l in p.read_text(encoding="utf-8").splitlines():
                d = json.loads(l)
                asr[d["id"]] = d["cer"]
        for r in rows:
            audio = r["audio"]
            if str(audio).startswith("parquet:"):      # FLEURS validation rows, written once as FLAC
                audio = str(ROOT / "rollout" / "ko" / "fleurs_val_audio" / f"{r['id']}.flac")
            out.append({"id": r["item"], "clip": r["id"], "audio": audio, "target": r["target"],
                        "prompt": r["prompt"], "label": r["label"], "speaker": r["speaker"],
                        "source": "cv" if r["src"] == "cv" else "fleurs", "lang": "ko",
                        "asr_cer": asr.get(r["id"]),
                        "fold": "fit" if _fold_hash(r["speaker"]) % 2 == 0 else "check"})
        return out
    for r in rows:
        out.append({"id": r["id"], "clip": r["id"].rsplit("__", 1)[0], "audio": r["audio"], "target": r["target"],
                    "prompt": r["prompt"], "label": r["label"], "speaker": r["speaker"], "source": r["source"],
                    "lang": lang.split("@")[0], "fold": r.get("fold")})
    return out


def _fold_hash(spk: str) -> int:
    """Speaker-disjoint fit / check halves for Korean, the rule of the other builders (sha256 of "fold|speaker")."""
    import hashlib

    return int(hashlib.sha256(f"fold|{spk}".encode()).hexdigest()[:12], 16)


def _f(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


def _install_recorders() -> None:
    import pronunciada.de_identity as DI
    import pronunciada.identity as ID

    if getattr(ID.apply_word, "_rec", False):
        return
    f_apply, f_de = ID.apply_word, DI.word_factor

    def apply_word(cal, s_word, prob, *a, **k):
        out = f_apply(cal, s_word, prob, *a, **k)
        _REC.append(("id", out / prob if prob > 0 else float("nan")))
        return out

    def word_factor(*a, **k):
        out = f_de(*a, **k)
        _REC.append(("de_id", float("nan") if out is None else out))
        return out

    apply_word._rec = True
    ID.apply_word, DI.word_factor = apply_word, word_factor


def detail(sc, res) -> dict:
    from pronunciada.align import score_reference
    from pronunciada.identity import word_ins

    c = res._pronunciada  # noqa: SLF001
    ref, ev, tab = c["ref"], c["ev"], c["tab"]
    ins = ev.llr_ins
    if ins is None:
        try:
            ins = score_reference(c["lpa"], ref, insertion=True).llr_ins
        except Exception:  # noqa: BLE001
            ins = None
    rec = [v for _, v in _REC]
    kid = c.get("ko_identity_s")
    f2t = sc.am.frame_to_time
    words = []
    for wi, (w, wr) in enumerate(zip(ref.words, res.words)):
        a, b = w.phones
        ph = []
        for i in range(a, b):
            p, sl = ref.phones[i], ev.slots[i]
            ph.append({"ipa": p.ipa, "nalt": len(p.alts), "alt": sl.alt_index,
                       "alt_ipa": (p.alts[sl.alt_index - 1][0] if sl.alt_index > 0 else ("" if sl.alt_index < 0 else p.ipa)),
                       "p": _f(wr.phones[i - a].prob) if i - a < len(wr.phones) else None, "zl": _f(c["s_zl"][i]), "heard": sl.heard,
                       "llr_sub": _f(sl.llr_sub), "llr_del": _f(sl.llr_del), "z_sub": _f(tab["z_llr_sub"][i])
                       if "z_llr_sub" in tab else None, "nf": int(sl.n_frames), "peak": _f(sl.peak_post),
                       "sf": int(sl.start_frame), "ef": int(sl.end_frame),
                       "stress": ref.syllables[p.syllable].stress, "nuc": bool(p.is_nucleus)})
        words.append({"text": w.text, "span": list(w.char_span), "prob": _f(wr.prob), "sounds": _f(wr.sounds_prob),
                      "stress": _f(wr.stress_prob), "length": _f(wr.length_prob), "contrast": _f(wr.contrast_prob),
                      "present": _f(wr.present_prob), "omitted": bool(wr.omitted),
                      "idf": _f(rec[wi]) if len(rec) == len(ref.words) else None,
                      "ko_delta": _f(kid["delta"][wi]) if isinstance(kid, dict) else None,
                      "ins": _f(word_ins(ins, a, b)) if ins is not None else None,
                      "start": wr.start, "end": wr.end, "w_zl": _f(c["w_zl"][wi]), "w_zs": _f(c["w_zs"][wi]),
                      "nsyl": w.syllables[1] - w.syllables[0], "phones": ph})
    free = [[t, int(s), int(e)] for t, s, e in (ev.free_tokens or [])]
    return {"rwords": words, "free": free, "frame_shift": float(ev.frame_shift), "n_frames": int(ev.n_frames),
            "sent_prob": _f(res.prob), "warnings": list(res.warnings), "audio_q": res.audio,
            "t0": float(f2t(0))}


def run(lang: str, part: str | None, limit: int | None, threads: int = 2, tag: str | None = None,
        only: set | None = None, oracle: str | None = None) -> Path:
    from bench.third_party.lingolix import reduce_azure_shape
    from pronunciada import Scorer
    from pronunciada.compat.azure import to_azure

    _install_recorders()
    k_part, n_part = (int(x) for x in part.split("/")) if part else (0, 1)
    d = DUMPS / tag if tag else DUMPS
    d.mkdir(parents=True, exist_ok=True)
    out = d / (f"{lang}.jsonl" if n_part == 1 else f"{lang}.p{k_part}.jsonl")
    done = set()
    for p in [q for q in d.glob(f"{lang}.*jsonl") if q.name == f"{lang}.jsonl" or q.name.startswith(f"{lang}.p")]:
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["id"])
    its = items(lang)
    clips = sorted({it["clip"] for it in its})
    mine = {c for j, c in enumerate(clips) if j % n_part == k_part}
    its = [it for it in its if it["clip"] in mine and (only is None or it["id"] in only)]
    if limit:
        its = its[:limit]
    sc = Scorer(lang.split("@")[0], threads=threads)
    t0, n = time.time(), 0
    with open(out, "a", encoding="utf-8") as f:
        for it in its:
            if it["id"] in done:
                continue
            _REC.clear()
            t = time.perf_counter()
            try:
                if oracle:                       # part 3 phase 0: guessed words given an oracle reading
                    from bench.round20_native_fa.enlex import oracle_reference

                    res = sc.score(it["audio"], oracle_reference(sc.g2p, it["target"], oracle))
                else:
                    res = sc.score(it["audio"], it["target"])
                lat = (time.perf_counter() - t) * 1000
                raw = to_azure(res, granularity="Phoneme", enable_miscue=True, phoneme_alphabet="IPA",
                               layout="rest", grading_system="HundredMark")
                vw = reduce_azure_shape(raw)
                rec = {"id": it["id"], "latency_ms": round(lat, 1), **detail(sc, res),
                       "r20": res._pronunciada.get("r20"),  # noqa: SLF001
                       "words": [{"text": w["text"], "score": w["score"], "error_type": w["error_type"]} for w in vw]}
            except Exception as e:  # noqa: BLE001
                rec = {"id": it["id"], "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                f.flush()
                print(lang, k_part, n, len(its), f"{time.time() - t0:.0f}s", flush=True)
    print(lang, f"part {k_part}/{n_part}: {n} new items in {time.time() - t0:.0f}s", flush=True)
    return out


def load(lang: str, tag: str | None = None) -> dict[str, dict]:
    out = {}
    d = DUMPS / tag if tag else DUMPS
    for p in sorted(q for q in d.glob(f"{lang}.*jsonl") if q.name == f"{lang}.jsonl" or q.name.startswith(f"{lang}.p")):
        if True:
            for l in p.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    out[r["id"]] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--part")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag")
    ap.add_argument("--only", help="json file: item ids to score (the rest is taken from the untagged dump)")
    ap.add_argument("--oracle", help="part 3 phase 0: morph | ls | g2pen | best")
    a = ap.parse_args(argv)
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    if a.workers > 1:
        import subprocess
        import sys

        ps = [subprocess.Popen([sys.executable, "-m", "bench.round20_native_fa.collect", "--lang", a.lang,
                                "--part", f"{k}/{a.workers}"] + (["--tag", a.tag] if a.tag else [])
                             + (["--oracle", a.oracle] if a.oracle else [])
                             + (["--only", a.only] if a.only else []))
              for k in range(a.workers)]
        codes = [p.wait() for p in ps]
        if any(codes):
            raise SystemExit(f"worker failed: {codes}")
        return
    only = set(json.loads(Path(a.only).read_text())) if a.only else None
    for lang in a.lang.split(","):
        run(lang, a.part, a.limit, tag=a.tag, only=only, oracle=a.oracle)


if __name__ == "__main__":
    main()
