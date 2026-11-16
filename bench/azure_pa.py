"""Azure Pronunciation Assessment runner (REST API for short audio) + response parser.

    python -m bench.azure_pa run   <dataset> [--split test] [--lang en] [--limit N] [--dry-run]
    python -m bench.azure_pa parse <dataset> [--split test]   -> ~/.cache/pronunciada/azure/<dataset>/flat.jsonl

Credentials come ONLY from the environment: AZURE_SPEECH_KEY and AZURE_SPEECH_REGION
(optional AZURE_SPEECH_ENDPOINT = full https URL of the short-audio recognition endpoint, for
resources that require a custom-domain endpoint). The key is never printed, logged or cached.

Raw responses are cached per utterance in ~/.cache/pronunciada/azure/<dataset>/<id>.json together with
the request parameters (never the key); an utterance with a cache file is never sent again.

Locale facts (Microsoft Learn, checked 2026-11-15; see docs/benchmarks_data.md):
  * all 10 locales below support pronunciation assessment (33 locales in total);
  * prosody assessment: en-US only; phoneme names / IPA alphabet / syllables / NBest spoken
    phonemes: en-US only (zh-CN has SAPI phoneme names); other locales return per-phoneme
    scores without phoneme names;
  * REST short audio: <= 30 s of audio for pronunciation assessment, WAV PCM 16 kHz mono.

UNTESTED AGAINST THE LIVE SERVICE (no key was available); the parser is tested against the
response examples in the Azure documentation (both the REST flat layout and the Speech SDK
nested "PronunciationAssessment" layout).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

LOCALES = {"en": "en-US", "zh": "zh-CN", "ja": "ja-JP", "es": "es-MX", "fr": "fr-FR", "de": "de-DE",
           "it": "it-IT", "pt": "pt-BR", "ru": "ru-RU", "ko": "ko-KR"}
PROSODY_LOCALES = {"en-US"}
IPA_LOCALES = {"en-US"}
MAX_SECONDS = 30.0
TICKS = 1e7  # Offset / Duration unit: 100 ns


def cache_dir(dataset: str) -> Path:
    base = os.environ.get("PRONUNCIADA_CACHE")
    root = Path(base).expanduser() if base else Path.home() / ".cache" / "pronunciada"
    return root / "azure" / dataset


# ----------------------------------------------------------------------------- request

def pa_params(text: str, locale: str) -> dict:
    p = {"ReferenceText": text, "GradingSystem": "HundredMark", "Granularity": "Phoneme",
         "Dimension": "Comprehensive", "EnableMiscue": "True"}
    if locale in PROSODY_LOCALES:
        p["EnableProsodyAssessment"] = "True"
    if locale in IPA_LOCALES:
        p["PhonemeAlphabet"] = "IPA"
        p["NBestPhonemeCount"] = 5
    return p


def endpoint(region: str, locale: str) -> str:
    base = os.environ.get("AZURE_SPEECH_ENDPOINT") or (
        f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1")
    return f"{base}?language={locale}&format=detailed"


def wav_bytes(path: str) -> bytes:
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g)
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1), 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def credentials() -> tuple[str, str]:
    key, region = os.environ.get("AZURE_SPEECH_KEY"), os.environ.get("AZURE_SPEECH_REGION")
    if not key or not (region or os.environ.get("AZURE_SPEECH_ENDPOINT")):
        sys.exit("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set in the environment "
                 "(they are never read from files). Nothing was sent.")
    return key, region or ""


class RateLimiter:
    def __init__(self, rps: float):
        self.dt = 1.0 / rps if rps > 0 else 0.0
        self.last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now - self.last < self.dt:
            time.sleep(self.dt - (now - self.last))
        self.last = time.monotonic()


def send(session, key: str, region: str, locale: str, params: dict, audio: bytes,
         max_tries: int = 6, limiter: RateLimiter | None = None) -> dict:
    hdr = base64.b64encode(json.dumps(params, ensure_ascii=False).encode("utf-8")).decode("ascii")
    headers = {"Ocp-Apim-Subscription-Key": key, "Pronunciation-Assessment": hdr,
               "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
               "Accept": "application/json"}
    delay = 2.0
    for attempt in range(max_tries):
        if limiter:
            limiter.wait()
        try:
            r = session.post(endpoint(region, locale), headers=headers, data=audio, timeout=60)
        except Exception as e:  # network error: retry (message has no key: headers aren't in it)
            err = type(e).__name__
            status = None
        else:
            status = r.status_code
            if status == 200:
                return r.json()
            if status in (400, 401, 403):
                raise RuntimeError(f"Azure returned HTTP {status}: {r.text[:300]}")
            err = f"HTTP {status}"
            ra = r.headers.get("Retry-After")
            if ra and ra.isdigit():
                delay = max(delay, float(ra))
        print(f"  {err}; retry {attempt + 1}/{max_tries} in {delay:.0f}s", file=sys.stderr)
        time.sleep(delay)
        delay = min(delay * 2, 120)
    raise RuntimeError(f"giving up after {max_tries} tries")


def _audio_ok(rec: dict) -> bool:
    return rec["dur"] <= MAX_SECONDS


def run(dataset: str, split: str | None, lang: str | None, limit: int | None, dry_run: bool,
        rps: float) -> None:
    from bench.data import load

    recs = [r for r in load(dataset, split=split, lang=lang) if r["lang"] in LOCALES]
    out = cache_dir(dataset)
    todo = [r for r in recs if not (out / f"{r['id']}.json").exists()]
    too_long = [r for r in todo if not _audio_ok(r)]
    n_cached = len(recs) - len(todo)
    todo = [r for r in todo if _audio_ok(r)]
    if limit is not None:
        todo = todo[:limit]
    secs: dict[str, float] = {}
    for r in todo:
        secs[LOCALES[r["lang"]]] = secs.get(LOCALES[r["lang"]], 0.0) + r["dur"]
    print(f"{dataset}: {len(recs)} records, {n_cached} already cached, {len(too_long)} skipped "
          f"(> {MAX_SECONDS:.0f} s), {len(todo)} to send" + (f" (--limit {limit})" if limit else ""))
    for loc, s in sorted(secs.items()):
        print(f"  {loc}: {s:.0f} audio-seconds ({s / 3600:.2f} h)")
    print(f"  total: {sum(secs.values()):.0f} audio-seconds ({sum(secs.values()) / 3600:.2f} h)")
    if dry_run or not todo:
        return
    key, region = credentials()
    import requests

    out.mkdir(parents=True, exist_ok=True)
    lim = RateLimiter(rps)
    with requests.Session() as s:
        for i, r in enumerate(todo):
            locale = LOCALES[r["lang"]]
            params = pa_params(r["text"], locale)
            resp = send(s, key, region, locale, params, wav_bytes(r["audio"]), limiter=lim)
            rec = {"id": r["id"], "locale": locale, "params": params,
                   "audio_sha1": hashlib.sha1(Path(r["audio"]).read_bytes()).hexdigest(),
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "response": resp}
            tmp = out / f"{r['id']}.json.tmp"
            tmp.write_text(json.dumps(rec, ensure_ascii=False))
            os.replace(tmp, out / f"{r['id']}.json")
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- parser

def _pa(d: dict) -> dict:
    """Scores live either flat on the object (REST) or under "PronunciationAssessment" (SDK)."""
    pa = d.get("PronunciationAssessment")
    return pa if isinstance(pa, dict) else d


def _num(x):
    return None if x is None else float(x)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(ch for ch in s if unicodedata.category(ch)[0] in "LN")


def tokenize(text: str) -> list[str]:
    """Default manifest word list: whitespace tokens with surrounding punctuation stripped."""
    return [t for t in (re.sub(r"^\W+|\W+$", "", w) for w in text.split()) if t]


def parse_response(resp: dict) -> dict:
    """Flatten one Azure response -> {status, sentence{...}, words[...]} (Azure's own word list)."""
    status = resp.get("RecognitionStatus")
    nbest = resp.get("NBest") or []
    if status not in ("Success", 0) or not nbest:
        return {"status": status, "sentence": None, "words": []}
    best = nbest[0]
    pa = _pa(best)
    sentence = {k: _num(pa.get(k)) for k in
                ("AccuracyScore", "FluencyScore", "CompletenessScore", "ProsodyScore", "PronScore")}
    sentence["recognized"] = best.get("Lexical")
    words = []
    for w in best.get("Words") or []:
        wpa = _pa(w)
        fb = (w.get("Feedback") or wpa.get("Feedback") or {}).get("Prosody") or {}
        phones = []
        for p in w.get("Phonemes") or []:
            ppa = _pa(p)
            nb = ppa.get("NBestPhonemes") or p.get("NBestPhonemes") or []
            phones.append({"phoneme": p.get("Phoneme"), "accuracy": _num(ppa.get("AccuracyScore")),
                           "start": _sec(p.get("Offset")), "dur": _sec(p.get("Duration")),
                           "nbest": [(n.get("Phoneme"), _num(n.get("Score"))) for n in nb]})
        syls = [{"syllable": s.get("Syllable"), "accuracy": _num(_pa(s).get("AccuracyScore")),
                 "start": _sec(s.get("Offset")), "dur": _sec(s.get("Duration"))}
                for s in w.get("Syllables") or []]
        brk = fb.get("Break") or {}
        words.append({
            "word": w.get("Word"), "accuracy": _num(wpa.get("AccuracyScore")),
            "error_type": wpa.get("ErrorType") or "None",
            "start": _sec(w.get("Offset")), "dur": _sec(w.get("Duration")),
            "break_errors": brk.get("ErrorTypes") or [],
            "unexpected_break_conf": _num((brk.get("UnexpectedBreak") or {}).get("Confidence")),
            "missing_break_conf": _num((brk.get("MissingBreak") or {}).get("Confidence")),
            "intonation_errors": (fb.get("Intonation") or {}).get("ErrorTypes") or [],
            "syllables": syls, "phones": phones,
        })
    return {"status": "Success", "sentence": sentence, "words": words}


def _sec(t):
    return None if t is None else float(t) / TICKS


def align_words(parsed: dict, words: list[str]) -> list[dict]:
    """Map Azure's words onto our manifest words (character-level alignment of normalised strings,
    so it works for CJK as well as space-separated languages). Insertions are ignored.
    Returns one entry per manifest word: {text, azure_words, accuracy (min over matched Azure
    words, None if unmatched), error_types, omitted, phones}."""
    az = [w for w in parsed["words"] if w["error_type"] != "Insertion"]
    a_chars, a_owner = [], []
    for k, w in enumerate(az):
        for ch in _norm(w["word"] or ""):
            a_chars.append(ch)
            a_owner.append(k)
    m_chars, m_owner = [], []
    for k, w in enumerate(words):
        for ch in _norm(w):
            m_chars.append(ch)
            m_owner.append(k)
    hits: list[set[int]] = [set() for _ in words]
    sm = SequenceMatcher(None, m_chars, a_chars, autojunk=False)
    for blk in sm.get_matching_blocks():
        for t in range(blk.size):
            hits[m_owner[blk.a + t]].add(a_owner[blk.b + t])
    out = []
    for k, w in enumerate(words):
        ks = sorted(hits[k])
        matched = [az[j] for j in ks]
        accs = [m["accuracy"] for m in matched if m["accuracy"] is not None]
        ets = sorted({m["error_type"] for m in matched})
        out.append({"text": w, "azure_words": [m["word"] for m in matched],
                    "accuracy": min(accs) if accs else None, "error_types": ets,
                    "omitted": (not matched) or ets == ["Omission"],
                    "phones": [p for m in matched for p in m["phones"]]})
    return out


def flatten(resp: dict, words: list[str]) -> dict:
    p = parse_response(resp)
    return {"status": p["status"], "sentence": p["sentence"], "words": align_words(p, words),
            "azure_words": p["words"]}


def manifest_words(rec: dict) -> list[str]:
    lab = rec.get("labels") or {}
    if isinstance(lab.get("words"), list) and lab["words"] and isinstance(lab["words"][0], dict):
        return [w["text"] for w in lab["words"]]
    return tokenize(rec["text"])


def parse_cached(dataset: str, split: str | None) -> Path:
    from bench.data import load

    d = cache_dir(dataset)
    out = d / "flat.jsonl"
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in load(dataset, split=split):
            p = d / f"{r['id']}.json"
            if not p.exists():
                continue
            raw = json.loads(p.read_text())
            fl = flatten(raw["response"], manifest_words(r))
            f.write(json.dumps({"id": r["id"], "locale": raw["locale"], **fl}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} flattened results -> {out}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("dataset")
    r.add_argument("--split")
    r.add_argument("--lang")
    r.add_argument("--limit", type=int)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--rps", type=float, default=1.0, help="max requests per second")
    p = sub.add_parser("parse")
    p.add_argument("dataset")
    p.add_argument("--split")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.dataset, a.split, a.lang, a.limit, a.dry_run, a.rps)
    else:
        parse_cached(a.dataset, a.split)


if __name__ == "__main__":
    main()
