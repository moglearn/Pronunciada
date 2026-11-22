"""Head-to-head: the Lingolix pronunciation API measured by us, against Pronunciada, on the same items.

    export LINGOLIX_API_KEY=...                                  # read from the environment only
    python -m bench.third_party.lingolix_live --plan-only        # what would be sent, and the cost
    python -m bench.third_party.lingolix_live --yes              # send, cache, then write the table
    python -m bench.third_party.lingolix_live --report-only      # table from the cache, sends nothing
    python -m bench.third_party.lingolix_live --dry-run          # whole pipeline on a local fake

What is sent. Only public benchmark recordings and their prompt texts, nothing else:
  * test 1 of the Lingolix benchmark (lingolix/pronunciation-benchmark v2026.10): the 200 Common Voice
    clips (CC0) per language for en, fr, es, de, ja, ko, each with the manifest's `target` text, the
    WAV files of the benchmark's release sent byte for byte;
  * test 2: the 500 speechocean762 test clips (CC BY 4.0) with their `target` text, read from
    Pronunciada's local lossless copy and sent as 16 kHz mono 16-bit WAV (the format build_tier2.py
    makes);
  * with --fleurs-swap: Pronunciada's fleurs-swap test records (FLEURS test, CC BY 4.0; the original
    prompt and prompts with one word changed) for the languages Lingolix lists, as 16 kHz WAV.
One multipart POST per (recording, text): fields `sentence`, `language_code`, file `speechdata`, header
`Authorization: Bearer <key>`, exactly as the benchmark's Lingolix adapter (vendors.lingolix) sends it.
No user data, no Pronunciada output, no other file leaves the machine.

The key. Read only from LINGOLIX_API_KEY; never printed, logged or written (every file this module
writes is checked for it and scrubbed). Without it the module refuses to send; --plan-only,
--report-only and --dry-run need no key.

Cost control. The plan (requests, audio hours, estimated cost from the public price list) is printed
before anything is sent and nothing is sent without --yes. --max-requests N caps a run. Every raw JSON
response is cached under $PRONUNCIADA_CACHE/third_party/lingolix_live/responses/, keyed by a hash of
(audio bytes, text, language, API options) and written atomically as soon as it arrives, so an
interrupted run resumes and nothing is paid for twice; cached items are never requested again unless
--refresh is given.

Rate limits. At most --rps requests per second (default 1), --concurrency requests in flight
(default 1, at most 4). 429 (rate limit) and 5xx are retried with exponential backoff and jitter,
honouring Retry-After. 401, 403, 402 and the documented quota 429 stop the run cleanly.

Run log. $PRONUNCIADA_CACHE/third_party/lingolix_live/runs/<UTC time>.json: test date and time (UTC),
the model_version the API returned (and any version header), endpoint, options, counts, failures, the
git commit of this repository and the benchmark version. The table repeats the model versions and test
dates of the responses it used.

Scoring. Responses are reduced exactly as the benchmark's Lingolix adapter reduces them (word score =
`accuracy`, error type from the syllables' is_missing / is_extra) and scored with the benchmark's metric
rules as re-implemented in bench/third_party/lingolix.py (v2026.10 rules and the later `main` rules),
including report.py's Lingolix-specific error flag (expected vs detected syllable IPA for ja, fr, es)
and sentence score (`overall`). Output: results/tables/lingolix_live.json and results/lingolix_live.md.

Dry run. --dry-run replaces the HTTP transport with a local fake that returns responses in the
documented format with random scores; a socket guard refuses any connection to a non-local host while
it runs. Its cache, run logs and tables go to $PRONUNCIADA_CACHE/third_party/lingolix_live/DRYRUN_FAKE/,
never to results/, and every table it writes carries a "DRY RUN, FAKE DATA" banner.

ASSUMPTIONS about the API that the benchmark repository and the public documentation
(lingolix.com/en/api-portal/docs, /api-docs.md and /en/pricing, read 2026-11-22) do not settle. They are
all checked in `parse_response` / `classify_status`; a real response that breaks one stops the run with
a message naming it (the response is already cached, so fixing the adapter costs nothing):
  A1  Endpoint: POST {LINGOLIX_BASE_URL or https://api.lingolix.com}/api/pronunciation/v3/check, as in
      vendors.py; the docs name the path and the host but not the full base URL.
  A2  Request: multipart/form-data with `sentence`, `language_code` and the file field `speechdata`
      (vendors.py; the docs list the fields but not the encoding).
  A3  Every score (`overall`, word `accuracy`) is a number in [0, 1] (docs). A value above 1 (for
      example a 0-100 scale) stops the run instead of being silently rescaled.
  A4  `words` is a list of objects with `text` and `accuracy`, in the order of the sentence; for every
      language except Japanese at least half of the sentence's words are found among them.
  A5  `syllables[*]` carry `expected_ipa`, `detected_ipa`, `is_missing`, `is_extra`, `is_scored` (docs);
      the benchmark's error flag depends on them.
  A6  `model_version` (top level, string or null) is the version string to cite; no version header is
      documented, so any response header whose name contains "version" or "model" is also recorded.
  A7  A 429 whose body has `remaining_minutes`, `upgrade_url` or `plan`, or whose X-RateLimit-Remaining
      is 0, is the documented monthly-quota error and stops the run; any other 429 is a transient rate
      limit and is retried (no per-second limit and no Retry-After are documented).
  A8  Billing is per audio minute (usage measured per second, whole minutes charged; whether per request
      or per month is not documented), so the cost estimate is a range.
  A9  Language codes are the benchmark's (en fr es de ja ko; fleurs-swap: also it pt ru), all in the
      documented list. Recordings over 60 s or 10 MiB are rejected by the API (docs) and are not sent.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import email.utils
import hashlib
import io
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench.third_party import lingolix as L

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada")
LIVE = CACHE / "third_party" / "lingolix_live"
DRY_SUBDIR = "DRYRUN_FAKE"
OUT_JSON = ROOT / "results" / "tables" / "lingolix_live.json"
OUT_MD = ROOT / "results" / "lingolix_live.md"
BANNER = "DRY RUN, FAKE DATA: the Lingolix column comes from a local fake and says nothing about Lingolix."

KEY_ENV = "LINGOLIX_API_KEY"
DEFAULT_BASE = "https://api.lingolix.com"           # vendors.lingolix (A1)
ENDPOINT_PATH = "/api/pronunciation/v3/check"
API_HOSTS = ("api.lingolix.com",)
OPTIONS = {"endpoint": ENDPOINT_PATH, "method": "POST", "encoding": "multipart/form-data",
           "fields": ["sentence", "language_code"], "file_field": "speechdata", "content_type": "audio/wav"}
ADAPTER = "vendors.lingolix of lingolix/pronunciation-benchmark (identical at v2026.10 and main f8750aa)"
# Documented language codes (api-portal/docs, 2026-11-22)
SUPPORTED = ("ja", "en", "de", "fr", "es", "hu", "bg", "cs", "el", "fi", "it", "ko", "nl", "pl", "pt", "sv",
             "uk", "da", "nb", "ru", "tr")
MAX_AUDIO_S, MAX_AUDIO_BYTES = 60.0, 10 * 1024 * 1024
PRICES = {"source": "https://lingolix.com/en/pricing (read 2026-11-22)", "currency": "EUR",
          "overage_per_min": [0.02, 0.05],
          "plans_included_min": {"Free (non-commercial)": 15, "Starter": 180, "Learner": 720, "Pro": 3600,
                                 "Scale": 13200},
          "billing": "usage measured per second, whole minutes charged"}
TESTS = tuple(f"tier1_{x}" for x in L.LANGS) + ("tier2",)
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", ""}


class Fatal(RuntimeError):
    """Stop the run (bad key, no subscription, quota, a response the adapter does not understand)."""


class ResponseFormatError(Fatal):
    pass


class TransientError(RuntimeError):
    """Network-level failure worth retrying (timeout, connection reset)."""


class NetworkBlocked(RuntimeError):
    pass


# ── items ──────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Job:
    test: str          # tier1_<lang>, tier2, fleurs-swap
    id: str
    lang: str          # the benchmark's language
    api_lang: str      # language_code sent
    audio: Path
    text: str
    passthrough: bool  # True: send the file's bytes as they are (benchmark WAV); False: decode to 16 kHz WAV


_AUDIO_MEMO: dict = {}
_MEMO_LOCK = threading.Lock()


def audio_bytes(job: Job) -> bytes:
    """The exact bytes sent for this job (memoised for the last file: fleurs-swap reuses recordings)."""
    k = str(job.audio)
    with _MEMO_LOCK:
        if k in _AUDIO_MEMO:
            return _AUDIO_MEMO[k]
    if job.passthrough:
        b = Path(job.audio).read_bytes()
    else:
        import soundfile as sf

        x, sr = sf.read(str(job.audio), dtype="int16", always_2d=True)
        if sr != 16000:
            raise ValueError(f"{job.audio}: {sr} Hz, expected 16 kHz")
        buf = io.BytesIO()
        sf.write(buf, x[:, 0], 16000, format="WAV", subtype="PCM_16")
        b = buf.getvalue()
    with _MEMO_LOCK:
        _AUDIO_MEMO.clear()
        _AUDIO_MEMO[k] = b
    return b


def wav_seconds(b: bytes) -> float:
    import soundfile as sf

    i = sf.info(io.BytesIO(b))
    return i.frames / i.samplerate


def cache_key(audio: bytes, text: str, lang: str, options: dict = OPTIONS) -> tuple[str, dict]:
    inputs = {"audio_sha256": hashlib.sha256(audio).hexdigest(), "text": text, "language": lang,
              "options": options}
    s = json.dumps(inputs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest(), inputs


def benchmark_jobs(langs: list[str], tests: list[str]) -> list[Job]:
    T = L.tests()
    jobs = []
    for test in tests:
        if test.startswith("tier1_") and test[6:] in langs:
            for it in T[test]:
                lang = it.get("vendor_lang", {}).get("lingolix", it["lang"])     # run.py's rule
                jobs.append(Job(test, it["id"], it["lang"], lang, L.tier1_wav(it), it["target"], True))
        elif test == "tier2" and "en" in langs:
            paths = L.so762_paths([x["id"] for x in T["tier2"]])
            for it in T["tier2"]:
                jobs.append(Job("tier2", it["id"], "en", "en", paths[it["id"]], it["target"], False))
    return jobs


def fleurs_swap_jobs(langs: list[str]) -> list[Job]:
    from bench.data import load

    out = []
    for r in load("fleurs-swap", split="test"):
        if r["lang"] in langs and r["lang"] in SUPPORTED:
            p = Path(r["audio"].replace("/.cache/pronkit/", "/.cache/pronunciada/"))
            out.append(Job("fleurs-swap", r["id"], r["lang"], r["lang"], p, r["text"], False))
    return out


# ── cache ──────────────────────────────────────────────────────────────────────

def live_root(dry: bool, base: Path | None = None) -> Path:
    b = Path(base) if base else LIVE
    return b / DRY_SUBDIR if dry else b


def resp_path(root: Path, key: str) -> Path:
    return root / "responses" / key[:2] / f"{key}.json"


def scrub(s: str, secret: str | None) -> str:
    if secret and len(secret) >= 4 and secret in s:
        s = s.replace(secret, "[REDACTED]")
    return s


def write_atomic(p: Path, text: str, secret: str | None = None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(scrub(text, secret), encoding="utf-8")
    os.replace(tmp, p)


def read_cached(root: Path, key: str) -> dict | None:
    p = resp_path(root, key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None                       # a torn file cannot exist (atomic writes), but never trust one


_KEEP_HEADERS = re.compile(r"version|model|request-id|ratelimit|retry-after|^date$|^x-lingolix", re.I)


def keep_headers(h: dict) -> dict:
    return {k: v for k, v in h.items() if _KEEP_HEADERS.search(k) and k.lower() not in ("authorization", "set-cookie")}


# ── the adapter (all API-format knowledge lives here) ──────────────────────────

def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def lingolix_error(w: dict) -> str | None:
    """vendors._lingolix_error, verbatim."""
    sylls = w.get("syllables") or []
    if sylls and all(s.get("is_missing") for s in sylls):
        return "Omission"
    if any(s.get("is_extra") for s in sylls):
        return "Insertion"
    return None


def parse_response(raw, text: str, lang: str) -> list[dict]:
    """The benchmark's reduction of a v3 response (vendors._lingolix_call), after checking assumptions
    A3-A6. Raises ResponseFormatError naming the assumption a response breaks."""
    def bad(a: str, msg: str):
        raise ResponseFormatError(f"Lingolix response does not match assumption {a}: {msg}. The adapter "
                                  "(bench/third_party/lingolix_live.py, parse_response) must be checked "
                                  "against this response before any number is computed.")

    if not isinstance(raw, dict):
        bad("A4", f"top level is {type(raw).__name__}, expected an object")
    words = raw.get("words")
    if not isinstance(words, list):
        bad("A4", "no `words` list")
    if text.strip() and not words:
        bad("A4", "empty `words` for a non-empty sentence")
    for name in ("overall", "accuracy"):
        v = raw.get(name)
        if v is not None and (not _num(v) or not 0.0 <= v <= 1.0):
            bad("A3", f"`{name}` = {v!r}, expected a number in [0, 1]")
    mv = raw.get("model_version")
    if mv is not None and not isinstance(mv, str):
        bad("A6", f"`model_version` = {mv!r}, expected a string or null")
    for k, w in enumerate(words):
        if not isinstance(w, dict) or not isinstance(w.get("text"), str):
            bad("A4", f"word {k} has no `text` string")
        if not _num(w.get("accuracy")) or not 0.0 <= w["accuracy"] <= 1.0:
            bad("A3", f"word {k} ({w.get('text')!r}) `accuracy` = {w.get('accuracy')!r}, expected [0, 1]")
        sy = w.get("syllables")
        if sy is not None:
            if not isinstance(sy, list) or not all(isinstance(s, dict) for s in sy):
                bad("A5", f"word {k} `syllables` is not a list of objects")
            for s in sy:
                for f in ("is_missing", "is_extra", "is_scored"):
                    if f in s and not isinstance(s[f], bool):
                        bad("A5", f"word {k} syllable `{f}` = {s[f]!r}, expected a boolean")
                for f in ("expected_ipa", "detected_ipa"):
                    if s.get(f) is not None and not isinstance(s[f], str):
                        bad("A5", f"word {k} syllable `{f}` = {s[f]!r}, expected a string")
    if lang != "ja" and text.strip():
        want = [L.key(t) for t in text.split() if L.key(t)]
        got = {L.key(w["text"]) for w in words}
        hit = sum(t in got for t in want)
        if want and hit < 0.5 * len(want):
            bad("A4", f"only {hit} of {len(want)} words of the sentence appear in `words`")
    return [{"text": w["text"], "score": w["accuracy"], "scored": w.get("is_scored", True),
             "error_type": lingolix_error(w)} for w in words]


def to_record(job_id: str, cached: dict, text: str, lang: str) -> dict:
    """A benchmark response record (run.py's shape) from a cached raw response."""
    raw = cached["response"]
    return {"id": job_id, "vendor": "lingolix", "words": parse_response(raw, text, lang), "raw": raw,
            "latency_ms": cached.get("latency_ms")}


# ── transports ─────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class Response:
    status: int
    headers: dict
    body: bytes


class RequestsTransport:
    """The real one (requests, already in requirements-bench.txt)."""

    def post(self, url: str, *, headers: dict, data: dict, file: tuple, timeout: float) -> Response:
        import requests

        try:
            r = requests.post(url, headers=headers, data=data, files={OPTIONS["file_field"]: file},
                              timeout=timeout, allow_redirects=False)
        except requests.RequestException as e:      # message names the URL only, never the headers
            raise TransientError(type(e).__name__) from None
        return Response(r.status_code, dict(r.headers), r.content)

    def __repr__(self):
        return "RequestsTransport()"


class FakeTransport:
    """Local stand-in for the API: documented response format, random scores seeded by the request.
    `transient_every` / `unavailable_every` make every n-th call a 429 (Retry-After: 0) / 503 so the retry
    path runs too. Never touches the network."""

    def __init__(self, transient_every: int = 0, unavailable_every: int = 0, script: dict | None = None):
        self.calls: list[dict] = []
        self.transient_every, self.unavailable_every = transient_every, unavailable_every
        self.script = dict(script or {})          # call number (1-based) -> Response
        self._lock = threading.Lock()

    def post(self, url: str, *, headers: dict, data: dict, file: tuple, timeout: float) -> Response:
        with self._lock:
            n = len(self.calls) + 1
            self.calls.append({"url": url, "data": dict(data), "file": file[0], "n_bytes": len(file[1]),
                               "auth": headers.get("Authorization", "")})
        if n in self.script:
            return self.script[n]
        if self.transient_every and n % self.transient_every == 0:
            return Response(429, {"Retry-After": "0"}, b'{"message": "slow down"}')
        if self.unavailable_every and n % self.unavailable_every == 0:
            return Response(503, {}, b'{"message": "unavailable"}')
        body = fake_response(data["sentence"], data["language_code"], hashlib.sha256(file[1]).hexdigest())
        return Response(200, {"Content-Type": "application/json", "X-Request-Id": f"fake-{n}"},
                        json.dumps(body, ensure_ascii=False).encode())


_JA_SPLIT = re.compile(r"[、。！？!?,.\s「」『』（）]+")


def _fake_ipa(t: str) -> str:
    return "".join(c.lower() if c.isascii() and c.isalpha() else "ka" for c in t) or "a"


def fake_response(text: str, lang: str, audio_sha: str) -> dict:
    rng = random.Random(hashlib.sha256(f"{audio_sha}|{text}|{lang}".encode()).hexdigest())
    pieces = []                                               # (text, char_start, char_end)
    if lang == "ja":
        for seg in _JA_SPLIT.split(text):
            i = 0
            while i < len(seg):
                n = rng.randint(2, 4)
                pieces.append((seg[i:i + n], -1, -1))
                i += n
    else:
        for m in re.finditer(r"\S+", text):
            core = re.sub(r"^\W+|\W+$", "", m.group())
            if core:
                a = m.start() + m.group().find(core)
                pieces.append((core, a, a + len(core)))
    words = []
    for t, a, b in pieces:
        acc = rng.uniform(0.05, 0.6) if rng.random() < 0.1 else min(1.0, rng.betavariate(9, 1.3))
        sylls = []
        for j in range(0, len(t), 2):
            exp = _fake_ipa(t[j:j + 2])
            det = exp if acc >= 0.6 else ("x" + exp[1:] if lang != "ja" else exp + "ː")
            sylls.append({"text": t[j:j + 2], "expected_ipa": exp, "detected_ipa": det, "accuracy": round(acc, 3),
                          "completeness": 1.0, "pitch": "unknown", "duration_ms": 180, "start_ms": 0, "end_ms": 0,
                          "is_missing": acc < 0.15, "is_extra": False, "is_scored": True})
        words.append({"text": t, "accuracy": round(acc, 3), "completeness": 1.0, "start_ms": 0, "end_ms": 0,
                      "char_start": a, "char_end": b, "is_scored": True, "is_error": acc < 0.7,
                      "hesitation_ms": None, "syllables": sylls})
    acc = round(sum(w["accuracy"] for w in words) / len(words), 3) if words else 0.0
    nsyl = sum(len(w["syllables"]) for w in words)
    return {"text": text, "overall": acc, "accuracy": acc, "fluency": None if nsyl < 4 else round(rng.uniform(0.6, 1), 3),
            "completeness": 1.0, "speaking_rate": 5.0, "target_speaking_rate": None,
            "model_version": "FAKE-dry-run", "words": words}


@contextlib.contextmanager
def no_network_guard():
    """While active, any socket connection or name lookup for a non-local host raises NetworkBlocked."""
    real_connect, real_gai, real_cc = socket.socket.connect, socket.getaddrinfo, socket.create_connection

    def local(host) -> bool:
        h = host if isinstance(host, str) else (host.decode() if isinstance(host, bytes) else "")
        return h in LOCAL_HOSTS or h.startswith("127.")

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not local(address[0]):
            raise NetworkBlocked(f"dry run: connection to {address[0]} refused by the guard")
        return real_connect(self, address)

    def gai(host, *a, **k):
        if not local(host):
            raise NetworkBlocked(f"dry run: name lookup of {host} refused by the guard")
        return real_gai(host, *a, **k)

    def cc(address, *a, **k):
        if not local(address[0]):
            raise NetworkBlocked(f"dry run: connection to {address[0]} refused by the guard")
        return real_cc(address, *a, **k)

    socket.socket.connect, socket.getaddrinfo, socket.create_connection = connect, gai, cc
    try:
        yield
    finally:
        socket.socket.connect, socket.getaddrinfo, socket.create_connection = real_connect, real_gai, real_cc


# ── rate limit, retry ──────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, rps: float, clock=time.monotonic, sleep=time.sleep):
        self.interval = 1.0 / rps if rps > 0 else 0.0
        self.clock, self.sleep = clock, sleep
        self.next = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = self.clock()
            t = max(now, self.next)
            self.next = t + self.interval
        if t > now:
            self.sleep(t - now)


@dataclasses.dataclass
class Policy:
    max_attempts: int = 6
    base_s: float = 2.0
    cap_s: float = 60.0
    max_retry_after_s: float = 900.0
    timeout_s: float = 60.0


def retry_after_s(headers: dict) -> float | None:
    v = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    if v is None:
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        try:
            d = email.utils.parsedate_to_datetime(v)
            return max(0.0, (d - dt.datetime.now(dt.timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None


def _json_or_none(b: bytes):
    try:
        return json.loads(b.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def classify_status(resp: Response) -> str:
    """ok | retry | fatal | fail (this item only)."""
    s = resp.status
    if 200 <= s < 300:
        return "ok"
    if s in (401, 403, 402):
        return "fatal"
    if s == 429:
        body = _json_or_none(resp.body)
        rem = next((v for k, v in resp.headers.items() if k.lower() == "x-ratelimit-remaining"), None)
        if (isinstance(body, dict) and any(k in body for k in ("remaining_minutes", "upgrade_url", "plan"))) \
                or (rem is not None and str(rem).strip() == "0"):
            return "fatal"                                    # A7: documented monthly quota
        return "retry"
    if s in (408, 425) or s >= 500:
        return "retry"
    return "fail"


_FATAL_HELP = {401: "the API key was rejected (401): check LINGOLIX_API_KEY",
               402: "payment required (402)",
               403: "no active API subscription for this key (403)",
               429: "the monthly audio-minute quota is used up (429); nothing more will be sent"}


def send_one(job: Job, audio: bytes, *, transport, url: str, key: str, limiter: RateLimiter, policy: Policy,
             sleep=time.sleep, rng: random.Random | None = None) -> tuple[dict | None, dict]:
    """(raw JSON or None, meta). Raises Fatal to stop the run."""
    rng = rng or random.Random()
    meta = {"attempts": 0, "retries": []}
    for attempt in range(policy.max_attempts):
        limiter.wait()
        meta["attempts"] += 1
        t0 = time.perf_counter()
        try:
            resp = transport.post(url, headers={"Authorization": f"Bearer {key}"},
                                  data={"sentence": job.text, "language_code": job.api_lang},
                                  file=(Path(job.audio).with_suffix(".wav").name, audio, OPTIONS["content_type"]),
                                  timeout=policy.timeout_s)
        except TransientError as e:
            resp, why = None, str(e)
        meta["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if resp is not None:
            kind = classify_status(resp)
            if kind == "ok":
                raw = _json_or_none(resp.body)
                if raw is None:
                    raise ResponseFormatError("Lingolix answered 200 with a body that is not JSON (A4)")
                meta.update(status=resp.status, headers=keep_headers(resp.headers))
                return raw, meta
            if kind == "fatal":
                raise Fatal(_FATAL_HELP.get(resp.status, f"HTTP {resp.status}") + "; run stopped")
            if kind == "fail":
                body = resp.body[:300].decode("utf-8", "replace")
                meta.update(status=resp.status, error=f"HTTP {resp.status}: {scrub(body, key)}")
                return None, meta
            why = f"HTTP {resp.status}"
            ra = retry_after_s(resp.headers)
        else:
            ra = None
        if attempt == policy.max_attempts - 1:
            break
        if ra is not None:
            if ra > policy.max_retry_after_s:
                raise Fatal(f"the API asked to wait {ra:.0f} s (Retry-After); run stopped, rerun later to resume")
            delay = ra + rng.uniform(0, min(1.0, 0.1 * ra + 0.05))
        else:
            delay = min(policy.cap_s, policy.base_s * 2 ** attempt) * rng.uniform(0.5, 1.0)
        meta["retries"].append({"why": why, "retry_after_s": ra, "slept_s": round(delay, 3)})
        sleep(delay)
    meta["error"] = f"gave up after {meta['attempts']} attempts ({why})"
    return None, meta


# ── one run ────────────────────────────────────────────────────────────────────

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_state() -> dict:
    try:
        c = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        d = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, timeout=10)
        return {"commit": c.stdout.strip() or None, "dirty": bool(d.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def benchmark_version() -> dict:
    return {"repo": f"https://github.com/{L.REPO}", "manifests": f"v2026.10 ({L.SOURCES['repo_v2026.10'][:7]})",
            "main_rules": f"main {L.SOURCES['repo_main_f8750aa'][:7]}", "tier1_audio": L.TIER1_ZIP[0],
            "adapter": ADAPTER}


def plan(jobs: list[Job], root: Path, *, refresh: bool, max_requests: int | None) -> dict:
    rows, todo = {}, []
    for j in jobs:
        a = audio_bytes(j)
        k, _ = cache_key(a, j.text, j.api_lang)
        sec = wav_seconds(a)
        r = rows.setdefault(j.test, {"items": 0, "cached": 0, "over_limit": 0, "to_send": 0, "send_s": 0.0,
                                     "all_s": 0.0})
        r["items"] += 1
        r["all_s"] += sec
        if not refresh and read_cached(root, k) is not None:
            r["cached"] += 1
        elif sec > MAX_AUDIO_S or len(a) > MAX_AUDIO_BYTES:
            r["over_limit"] += 1
        else:
            todo.append((j, k, sec))
    if max_requests is not None:
        todo = todo[:max(0, max_requests)]
    for j, _, sec in todo:
        rows[j.test]["to_send"] += 1
        rows[j.test]["send_s"] += sec
    send_s = sum(s for _, _, s in todo)
    minutes = send_s / 60
    per_req_min = sum(math.ceil(s / 60) for _, _, s in todo)
    lo, hi = PRICES["overage_per_min"]
    return {"rows": rows, "todo": todo, "n_send": len(todo), "send_hours": send_s / 3600,
            "cost": {"minutes": round(minutes, 1), "minutes_if_each_request_rounded_up": per_req_min,
                     "eur_overage_if_billed_per_second_total": [round(minutes * lo, 2), round(minutes * hi, 2)],
                     "eur_overage_if_each_request_rounded_up": [round(per_req_min * lo, 2), round(per_req_min * hi, 2)],
                     "note": "a plan's included minutes are used first; Free (15 min) is non-commercial", **PRICES}}


def print_plan(p: dict, *, url: str, dry: bool, rps: float, concurrency: int) -> None:
    out = sys.stderr
    print(("DRY RUN, FAKE DATA: nothing leaves this machine; a local fake answers.\n" if dry else
           "LIVE: these requests are billed by Lingolix.\n") + f"endpoint  POST {url}", file=out)
    print(f"{'test':<14}{'items':>7}{'cached':>8}{'>limit':>8}{'to send':>9}{'audio min':>11}", file=out)
    for t, r in p["rows"].items():
        print(f"{t:<14}{r['items']:>7}{r['cached']:>8}{r['over_limit']:>8}{r['to_send']:>9}{r['send_s'] / 60:>11.1f}",
              file=out)
    c = p["cost"]
    print(f"requests to send: {p['n_send']}   audio: {p['send_hours']:.2f} h ({c['minutes']} min)", file=out)
    if p["n_send"]:
        a, b = c["eur_overage_if_billed_per_second_total"], c["eur_overage_if_each_request_rounded_up"]
        print(f"estimated cost (exact billing unit not documented): {c['minutes']} audio min, i.e. EUR "
              f"{a[0]:.2f}-{a[1]:.2f} at the published overage rates (EUR {c['overage_per_min'][0]}-"
              f"{c['overage_per_min'][1]}/min); if every request were rounded up to a whole minute: "
              f"{c['minutes_if_each_request_rounded_up']} min, EUR {b[0]:.2f}-{b[1]:.2f}. A plan's included "
              "minutes are used first (per month: " + ", ".join(f"{k} {v}" for k, v in c["plans_included_min"].items())
              + f"). Source: {c['source']}.", file=out)
        eta = p["n_send"] / max(rps, 1e-9)
        print(f"rate: <= {rps:g} requests/s, {concurrency} in flight; at least {eta / 60:.1f} min", file=out)


def run_jobs(todo: list[tuple[Job, str, float]], *, transport, key: str, root: Path, url: str, rps: float,
             concurrency: int, policy: Policy, sleep=time.sleep, clock=time.monotonic, progress=True) -> dict:
    """Send every (job, cache key) in `todo`, caching each answer as it arrives. Returns run counters."""
    limiter = RateLimiter(rps, clock=clock, sleep=sleep)
    stop = threading.Event()
    lock = threading.Lock()
    st = {"sent": 0, "succeeded": 0, "failed": 0, "retries": 0, "failures": [], "model_versions": {},
          "version_headers": {}, "stopped": None}
    done_by_test: dict = {}

    def one(item):
        job, key_hash, _sec = item
        if stop.is_set():
            return
        audio = audio_bytes(job) if job.audio else b""
        _, inputs = cache_key(audio, job.text, job.api_lang)
        try:
            raw, meta = send_one(job, audio, transport=transport, url=url, key=key, limiter=limiter, policy=policy,
                                 sleep=sleep)
        except Fatal as e:
            stop.set()
            with lock:
                st["stopped"] = st["stopped"] or scrub(str(e), key)
            return
        with lock:
            st["sent"] += 1
            st["retries"] += len(meta["retries"])
        if raw is None:
            with lock:
                st["failed"] += 1
                st["failures"].append({"test": job.test, "id": job.id, "error": scrub(meta.get("error", ""), key),
                                       "attempts": meta["attempts"]})
            return
        rec = {"schema": 1, "key": key_hash, "key_inputs": inputs, "job": {"test": job.test, "id": job.id},
               "received_utc": utc_now(), "endpoint": url, "status": meta["status"], "headers": meta["headers"],
               "latency_ms": meta["latency_ms"], "attempts": meta["attempts"], "response": raw}
        p = resp_path(root, key_hash)
        write_atomic(p, json.dumps(rec, ensure_ascii=False), secret=key)     # paid for: keep it first
        try:
            parse_response(raw, job.text, job.lang)
        except ResponseFormatError as e:
            stop.set()
            with lock:
                st["stopped"] = f"{e} Raw response: {p}"
            return
        with lock:
            st["succeeded"] += 1
            v = raw.get("model_version")
            st["model_versions"][str(v)] = st["model_versions"].get(str(v), 0) + 1
            for h, hv in meta["headers"].items():
                if re.search("version|model", h, re.I):
                    st["version_headers"].setdefault(h, {})
                    st["version_headers"][h][hv] = st["version_headers"][h].get(hv, 0) + 1
            n = done_by_test[job.test] = done_by_test.get(job.test, 0) + 1
        if progress and n % 25 == 0:
            print(f"  {job.test}: {n} answered", file=sys.stderr, flush=True)

    try:
        if concurrency <= 1:
            for item in todo:
                if stop.is_set():
                    break
                one(item)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                list(ex.map(one, todo))
    except KeyboardInterrupt:
        stop.set()
        st["stopped"] = "interrupted (Ctrl-C); rerun the same command to resume"
    st["not_sent"] = len(todo) - st["sent"]
    return st


# ── comparison table ───────────────────────────────────────────────────────────

def pronunciada_responses(pcache: Path, test: str) -> dict:
    p = pcache / "third_party" / "lingolix" / "responses" / "pronunciada" / f"{test}.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def lingolix_records(jobs: list[Job], root: Path) -> tuple[dict, dict]:
    """{test: {id: record}} from the cache, and provenance (model versions, test dates, counts)."""
    recs: dict = {}
    prov = {"model_versions": {}, "version_headers": {}, "first_utc": None, "last_utc": None, "responses": 0}
    for j in jobs:
        a = audio_bytes(j)
        k, _ = cache_key(a, j.text, j.api_lang)
        c = read_cached(root, k)
        if c is None:
            continue
        try:
            recs.setdefault(j.test, {})[j.id] = to_record(j.id, c, j.text, j.lang)
        except ResponseFormatError as e:
            raise ResponseFormatError(f"{e} Raw response: {resp_path(root, k)}") from None
        prov["responses"] += 1
        v = str(c["response"].get("model_version"))
        prov["model_versions"][v] = prov["model_versions"].get(v, 0) + 1
        for h, hv in (c.get("headers") or {}).items():
            if re.search("version|model", h, re.I):
                prov["version_headers"].setdefault(h, {})
                prov["version_headers"][h][hv] = prov["version_headers"][h].get(hv, 0) + 1
        t = c.get("received_utc")
        if t:
            prov["first_utc"] = min(x for x in (prov["first_utc"], t) if x)
            prov["last_utc"] = max(x for x in (prov["last_utc"], t) if x)
    return recs, prov


def _p_two(d) -> float:
    import numpy as np

    d = np.asarray(d, dtype=float)
    if not len(d):
        return math.nan
    return float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1))))


def _q(v) -> list:
    import numpy as np

    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [math.nan, math.nan]


def paired_units(manifest, ra, rb, fn, **kw) -> dict:
    """Pronunciada - Lingolix on the same 1,000 utterance resamples (report.py's resampling, seed 0)."""
    ua = {u[2]["id"]: u for u in L.collect(manifest, ra, **kw)["units"]}
    ub = {u[2]["id"]: u for u in L.collect(manifest, rb, **kw)["units"]}
    ids = [i for i in ua if i in ub]
    rng = random.Random(0)
    d = []
    for _ in range(L.BOOTSTRAP):
        pick = [ids[rng.randrange(len(ids))] for _ in ids]
        va = fn([x for i in pick for x in ua[i][0]], [x for i in pick for x in ua[i][1]])
        vb = fn([x for i in pick for x in ub[i][0]], [x for i in pick for x in ub[i][1]])
        if not (math.isnan(va) or math.isnan(vb)):
            d.append(va - vb)
    return {"n_common": len(ids), "delta_ci": _q(d), "p": _p_two(d)}


def length_caught(manifest, ra, rb) -> dict:
    """Japanese length items: caught = the system's own error flag on the planted unit (report.py verdicts)."""
    def caught(R):
        out = {}
        for labels, _, item, flags in L.collect(manifest, R)["units"]:
            if (item.get("label") or {}).get("bucket") == "length" and 1 in labels:
                out[item["id"]] = None if flags is None else bool(flags[labels.index(1)])
        return out

    a, b = caught(ra), caught(rb)
    ids = [i for i in a if i in b and a[i] is not None and b[i] is not None]
    rng = random.Random(0)
    d = []
    for _ in range(L.BOOTSTRAP if ids else 0):
        pick = [ids[rng.randrange(len(ids))] for _ in ids]
        d.append(sum(a[i] for i in pick) - sum(b[i] for i in pick))
    n_items = sum(1 for it in manifest if (it.get("label") or {}).get("bucket") == "length")
    return {"n_items": n_items, "pronunciada": sum(bool(v) for v in a.values()),
            "lingolix": None if any(v is None for v in b.values()) else sum(bool(v) for v in b.values()),
            "lingolix_answered": len(b), "n_common": len(ids),
            "delta": (sum(a[i] for i in ids) - sum(b[i] for i in ids)) if ids else None,
            "delta_ci": _q(d), "p": _p_two(d)}


def _locate(words: list[dict], text: str) -> list[tuple[int, int, float]]:
    """Lingolix words to character spans: char_start/char_end when given (>= 0), else found in order
    (report.align_spans: Japanese and extra words); words that cannot be placed are left out."""
    spans, cur = [], 0
    for w in words:
        a, b = w.get("char_start", -1), w.get("char_end", -1)
        if isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(text):
            spans.append((a, b, float(w["accuracy"])))
            cur = b
            continue
        t = L._PUNCT.sub("", w.get("text", ""))
        j = text.find(t, cur) if t else -1
        if j >= 0:
            spans.append((j, j + len(t), float(w["accuracy"])))
            cur = j + len(t)
    return spans


def swap_table(jobs: list[Job], root: Path, pcache: Path) -> dict:
    """fleurs-swap (bench/eval_swap.py metrics): far AUC, near AUC, near hit at 5 % false alarms, on the
    clusters both systems scored completely; paired 1,000 cluster resamples, seed 0."""
    import pickle

    import numpy as np

    from bench.data import load
    from bench.eval_swap import _metrics, collect

    by_id = {j.id: j for j in jobs if j.test == "fleurs-swap"}
    recs = [r for r in load("fleurs-swap", split="test") if r["id"] in by_id]
    lx, failed = {}, {}
    for r in recs:
        j = by_id[r["id"]]
        c = read_cached(root, cache_key(audio_bytes(j), j.text, j.api_lang)[0])
        if c is None:
            failed[r["lang"]] = failed.get(r["lang"], 0) + 1
            continue
        raw = c["response"]
        parse_response(raw, j.text, j.lang)
        lx[r["id"]] = {"status": "Success", "unmapped": [], "words": _locate(raw["words"], j.text),
                       "sentence": {"overall": raw.get("overall")}}
    ours = {}
    for lang in sorted({r["lang"] for r in recs}):
        p = pcache / "calib" / "api" / f"fleurs-swap-{lang}.pkl"
        if p.exists():
            ours.update({k: v["on"] for k, v in pickle.loads(p.read_bytes()).items()})
    A, B = collect(recs, ours), collect(recs, lx)
    keys = (("swap_far:auc", "far AUC"), ("swap_near:auc", "near AUC"), ("swap_near:hit@5%FA", "near hit@5% FA"))
    out = {}
    for lang in sorted(set(A) & set(B)):
        ca, cb = A[lang]["clusters"], B[lang]["clusters"]
        pa, pb = [], []
        for k in sorted(set(ca) & set(cb)):
            vs = set(ca[k]) & set(cb[k])
            if "orig" in vs and len(vs) > 1:
                pa.append({v: ca[k][v] for v in vs})
                pb.append({v: cb[k][v] for v in vs})
        if not pa:
            continue
        variants = ["swap_far", "swap_near"]
        ma, mb = _metrics(pa, variants), _metrics(pb, variants)
        rng = np.random.default_rng(0)
        ba, bb, bd = {k: [] for k, _ in keys}, {k: [] for k, _ in keys}, {k: [] for k, _ in keys}
        for _ in range(1000):
            idx = rng.integers(0, len(pa), len(pa))
            xa, xb = _metrics([pa[i] for i in idx], variants), _metrics([pb[i] for i in idx], variants)
            for k, _n in keys:
                if np.isfinite(xa.get(k, np.nan)):
                    ba[k].append(xa[k])
                if np.isfinite(xb.get(k, np.nan)):
                    bb[k].append(xb[k])
                if np.isfinite(xa.get(k, np.nan) - xb.get(k, np.nan)):
                    bd[k].append(xa[k] - xb[k])
        out[lang] = {"clusters": len(pa), "lingolix_not_answered": failed.get(lang, 0)}
        for k, name in keys:
            out[lang][name] = {"pronunciada": {"value": ma.get(k), "ci": _q(ba[k])},
                               "lingolix": {"value": mb.get(k), "ci": _q(bb[k])},
                               "delta": (ma.get(k, math.nan) - mb.get(k, math.nan)), "delta_ci": _q(bd[k]),
                               "p": _p_two(bd[k]), "published": None}
    return out


def build_table(jobs: list[Job], root: Path, pcache: Path, *, dry: bool) -> dict:
    from bench.rivals.eval_lingolix import paired_t2

    recs, prov = lingolix_records(jobs, root)
    T = L.tests()
    unv = {}
    for lang in L.LANGS:
        for it in L.load_manifest("repo_main_f8750aa", f"tier1/manifest_{lang}.jsonl"):
            unv[it["id"]] = it.get("unverified", [])
    pub = L.PUBLISHED
    res = {"tier1": {}, "tier1_main_rules": {}, "ja_length_caught": None, "tier2": {}, "fleurs_swap": {}}
    for lang in L.LANGS:
        test = f"tier1_{lang}"
        if test not in recs:
            continue
        man, PA, LX = T[test], pronunciada_responses(pcache, test), recs[test]
        if not PA:
            continue
        for rules, kw in (("tier1", {}), ("tier1_main_rules", {"exclude_unverified": unv})):
            mp, ml = L.metrics(man, PA, **kw), L.metrics(man, LX, **kw)
            row = {"n_items": len(man), "lingolix_answered": len(LX), "lingolix_failed_or_missing": ml["n_failed"],
                   "pronunciada_failed": mp["n_failed"],
                   "lingolix_verdicts": ml["verdicts"], "pronunciada_verdicts": mp["verdicts"]}
            for k, ci, fn, pk in (("roc_auc", "roc_auc_ci", L.roc_auc, "tier1_auc"),
                                  ("fpr_at_recall_80", "fpr_at_recall_80_ci",
                                   lambda l, s: L.fpr_at_recall(l, s, 0.80), "tier1_fa_at_80")):
                row[k] = {"pronunciada": {"value": mp[k], "ci": mp[ci]}, "lingolix": {"value": ml[k], "ci": ml[ci]},
                          "delta": mp[k] - ml[k], **paired_units(man, PA, LX, fn, **kw),
                          "published": pub[pk]["lingolix"][lang] if rules == "tier1" else None}
            res[rules][lang] = row
        if lang == "ja":
            res["ja_length_caught"] = {**length_caught(man, PA, LX),
                                       "published": pub["ja_length_caught_of_40"]["lingolix"]}
    if "tier2" in recs:
        man, PA, LX = T["tier2"], pronunciada_responses(pcache, "tier2"), recs["tier2"]
        if PA:
            mp, ml = L.metrics(man, PA), L.metrics(man, LX)
            for which, k in (("word", "pcc_word"), ("utt", "pcc_utterance")):
                res["tier2"][k] = {"pronunciada": {"value": mp.get(k), "ci": L._pcc_ci(man, PA, which)},
                                   "lingolix": {"value": ml.get(k), "ci": L._pcc_ci(man, LX, which)},
                                   "delta": (mp.get(k) or math.nan) - (ml.get(k) or math.nan),
                                   **paired_t2(man, PA, LX, which),
                                   "published": pub["tier2_pcc"]["lingolix"]["word" if which == "word" else "utterance"]}
            res["tier2"]["counts"] = {"n_items": len(man), "lingolix_failed_or_missing": ml["n_failed"],
                                      "lingolix_word_coverage": ml.get("word_coverage")}
    if any(j.test == "fleurs-swap" for j in jobs):
        res["fleurs_swap"] = swap_table(jobs, root, pcache)
    mv = prov["model_versions"]
    out = {"dry_run": dry, "banner": BANNER if dry else None,
           "lingolix": {"endpoint": f"POST {base_url()}{ENDPOINT_PATH}", "options": OPTIONS,
                        "model_versions": mv, "mixed_model_versions": len(mv) > 1,
                        "version_headers": prov["version_headers"],
                        "test_dates_utc": [prov["first_utc"], prov["last_utc"]], "responses_used": prov["responses"]},
           "pronunciada": {"outputs_from": _public_path(pcache), "version": _pronunciada_version()},
           "benchmark": benchmark_version(), "git": git_state(), "made_utc": utc_now(),
           "method": "metrics of bench/third_party/lingolix.py (report.py re-implemented, v2026.10 and main rules); "
                     "Lingolix responses reduced with the benchmark's Lingolix adapter; 95% intervals: 1,000 "
                     "utterance resamples (test 1), speaker resamples (test 2), cluster resamples (fleurs-swap), "
                     "seed 0; delta = Pronunciada - Lingolix on the same resamples, two-sided p; FA@80 lower is "
                     "better; 'published' = Lingolix's own figure from its 11 Oct 2026 post (rounded, no interval)",
           **res}
    return L._clean(out)


def _public_path(p: Path) -> str:
    """A path without the user's home directory in it (tables are committed)."""
    p = Path(p).expanduser().absolute()
    for base, name in ((CACHE.absolute(), "$PRONUNCIADA_CACHE"), (Path.home(), "~")):
        try:
            rel = p.relative_to(base)
            return name if str(rel) == "." else f"{name}/{rel}"
        except ValueError:
            pass
    return p.name


def _pronunciada_version():
    try:
        import pronunciada

        return getattr(pronunciada, "__version__", None)
    except Exception:
        return None


def _fmt(v, pct=False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{100 * v:.1f}%" if pct else f"{v:.3f}"


def _cell(x: dict, pct=False) -> str:
    ci = x.get("ci") or [None, None]
    if ci[0] is None:
        return _fmt(x.get("value"), pct)
    s = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.3f}")
    return f"{_fmt(x.get('value'), pct)} [{s(ci[0])}, {s(ci[1])}]"


def _delta(r: dict, pct=False) -> str:
    d, ci, p = r.get("delta"), r.get("delta_ci") or [None, None], r.get("p")
    if d is None:
        return "n/a"
    f = (lambda v: f"{100 * v:+.1f}") if pct else (lambda v: f"{v:+.3f}")
    c = f" [{f(ci[0])}, {f(ci[1])}]" if ci[0] is not None else ""
    return f"{f(d)}{c}, p {p:.3f}" if p is not None else f"{f(d)}{c}"


def markdown(t: dict) -> str:
    lx = t["lingolix"]
    out = []
    if t.get("dry_run"):
        out += [f"# {BANNER}", ""]
    out += ["# Lingolix API measured by us, against Pronunciada", "",
            f"Lingolix: {lx['endpoint']}; model_version {', '.join(f'{k} ({v})' for k, v in lx['model_versions'].items()) or 'n/a'}"
            f"{' (MIXED VERSIONS)' if lx['mixed_model_versions'] else ''}; tested {lx['test_dates_utc'][0]} to "
            f"{lx['test_dates_utc'][1]} (UTC); {lx['responses_used']} responses.",
            f"Pronunciada outputs read from {t['pronunciada']['outputs_from']} (installed package "
            f"{t['pronunciada']['version']}); "
            f"repository commit {(t['git']['commit'] or '?')[:10]}{' (uncommitted changes)' if t['git']['dirty'] else ''}; "
            f"benchmark {t['benchmark']['manifests']}.", "",
            "| test | metric | Pronunciada [95% CI] | Lingolix measured [95% CI] | Pronunciada − Lingolix [95% CI], p | Lingolix published |",
            "|---|---|---|---|---|---|"]
    for rules, tag in (("tier1", "v2026.10 rules"), ("tier1_main_rules", "main rules")):
        for lang, r in (t.get(rules) or {}).items():
            part = "" if r["lingolix_answered"] == r["n_items"] else \
                f", INCOMPLETE: Lingolix answered {r['lingolix_answered']} of {r['n_items']}"
            for k, name, pct in (("roc_auc", "word AUC", False), ("fpr_at_recall_80", "FA@80", True)):
                x = r[k]
                pubv = x.get("published")
                out.append(f"| test 1 {lang} ({tag}{part}) | {name} | {_cell(x['pronunciada'], pct)} | {_cell(x['lingolix'], pct)} | "
                           f"{_delta(x, pct)} | {'—' if pubv is None else (f'{100 * pubv:.0f}%' if pct else f'{pubv:.2f}')} |")
    j = t.get("ja_length_caught")
    if j:
        d, ci, p = j["delta"], j["delta_ci"], j["p"]
        ds = "n/a" if d is None else f"{d:+d}"
        if d is not None and ci[0] is not None:
            ds += f" [{ci[0]:+.0f}, {ci[1]:+.0f}]"
        if d is not None and p is not None:
            ds += f", p {p:.3f}"
        lx_n = "n/a" if j["lingolix"] is None else j["lingolix"]
        out.append(f"| test 1 ja | length errors caught (of {j['n_items']}) | {j['pronunciada']} | {lx_n} | "
                   f"{ds} | {j['published']} |")
    for k, name in (("pcc_word", "word Pearson"), ("pcc_utterance", "sentence Pearson")):
        x = (t.get("tier2") or {}).get(k)
        if x:
            miss = t["tier2"]["counts"]["lingolix_failed_or_missing"]
            part = f", INCOMPLETE: {miss} of 500 not answered" if miss else ""
            out.append(f"| test 2 (speechocean762{part}) | {name} | {_cell(x['pronunciada'])} | {_cell(x['lingolix'])} | "
                       f"{_delta(x)} | {x['published']:.2f} |")
    for lang, r in (t.get("fleurs_swap") or {}).items():
        for name in ("far AUC", "near AUC", "near hit@5% FA"):
            x = r[name]
            part = f", INCOMPLETE: {r['lingolix_not_answered']} prompts not answered" if r["lingolix_not_answered"] else ""
            out.append(f"| fleurs-swap {lang} ({r['clusters']} recordings{part}) | {name} | {_cell(x['pronunciada'])} | "
                       f"{_cell(x['lingolix'])} | {_delta(x)} | — |")
    out += ["", "FA@80: share of correct words flagged at the threshold that catches 80% of the planted errors "
            "(lower is better); its differences are in percentage points. Intervals: 1,000 resamples of "
            "utterances (test 1), speakers (test 2) or recordings (fleurs-swap); p two-sided, paired on the "
            "same resamples. Length errors caught: each system's own error flag on the 40 Japanese length "
            "items (Lingolix: report.py's flag, expected vs detected syllable IPA). Failed or unanswered "
            "requests: see the JSON.", ""]
    if t.get("dry_run"):
        out += [f"**{BANNER}**", ""]
    return "\n".join(out)


def base_url() -> str:
    return os.environ.get("LINGOLIX_BASE_URL", DEFAULT_BASE).rstrip("/")


def output_paths(dry: bool, root: Path) -> tuple[Path, Path]:
    if dry:
        a, b = root / "tables" / "lingolix_live.json", root / "tables" / "lingolix_live.md"
        assert ROOT / "results" not in a.parents, "dry-run output must never go to results/"
        return a, b
    return OUT_JSON, OUT_MD


# ── command line ───────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="local fake API, no network, output under DRYRUN_FAKE/")
    ap.add_argument("--yes", action="store_true", help="send the planned requests (required for a live run)")
    ap.add_argument("--plan-only", action="store_true", help="print the plan and stop")
    ap.add_argument("--report-only", action="store_true", help="table from cached responses; sends nothing")
    ap.add_argument("--max-requests", type=int, help="send at most N requests in this run")
    ap.add_argument("--langs", help="comma list (default: every language Lingolix lists: en fr es de ja ko for "
                                    "the benchmark, also it pt ru for fleurs-swap)")
    ap.add_argument("--no-benchmark", action="store_true", help="skip tests 1 and 2")
    ap.add_argument("--fleurs-swap", action="store_true", help="also run Pronunciada's fleurs-swap test records")
    ap.add_argument("--refresh", action="store_true", help="request again even when a response is cached")
    ap.add_argument("--rps", type=float, help="requests per second (default 1; dry run 500)")
    ap.add_argument("--concurrency", type=int, default=1, help="requests in flight (default 1, max 4)")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--pronunciada-cache", type=Path, default=CACHE,
                    help="cache root holding Pronunciada's outputs (third_party/lingolix/responses, calib/api)")
    ap.add_argument("--cache-dir", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    dry = a.dry_run
    langs = [x for x in a.langs.split(",") if x] if a.langs else list(SUPPORTED)
    bad = [x for x in langs if x not in SUPPORTED]
    if bad:
        ap.error(f"Lingolix does not list language(s) {bad}")
    concurrency = max(1, min(a.concurrency, 4))
    rps = a.rps if a.rps else (500.0 if dry else 1.0)
    root = live_root(dry, a.cache_dir)

    key = None
    if not dry and not a.plan_only and not a.report_only:
        key = (os.environ.get(KEY_ENV) or "").strip()
        if not key:
            print(f"refusing to start: set {KEY_ENV} in the environment (or use --dry-run, --plan-only, "
                  "--report-only)", file=sys.stderr)
            return 2
    guard = no_network_guard() if dry else contextlib.nullcontext()
    with guard:
        jobs = [] if a.no_benchmark else benchmark_jobs(langs, [t for t in TESTS])
        if a.fleurs_swap:
            jobs += fleurs_swap_jobs(langs)
        url = f"{(DEFAULT_BASE if dry else base_url())}{ENDPOINT_PATH}"
        p = plan(jobs, root, refresh=a.refresh, max_requests=a.max_requests)
        print_plan(p, url=url, dry=dry, rps=rps, concurrency=concurrency)
        if a.plan_only:
            return 0
        st = None
        if p["n_send"] and not a.report_only:
            if not dry and not a.yes:
                print("nothing sent: add --yes to send these requests", file=sys.stderr)
                return 1
            transport = FakeTransport(transient_every=97, unavailable_every=89) if dry else RequestsTransport()
            policy = Policy(max_attempts=a.max_attempts, timeout_s=a.timeout,
                            **({"base_s": 0.01, "cap_s": 0.05} if dry else {}))
            started = utc_now()
            st = run_jobs(p["todo"], transport=transport, key="dry-run-placeholder" if dry else key, root=root,
                          url=url, rps=rps, concurrency=concurrency, policy=policy)
            log = {"banner": BANNER if dry else None, "dry_run": dry, "started_utc": started, "finished_utc": utc_now(),
                   "endpoint": f"POST {url}", "options": OPTIONS, "language_codes": sorted({j.api_lang for j in jobs}),
                   "rps": rps, "concurrency": concurrency, "policy": dataclasses.asdict(policy),
                   "counts": {"items": len(jobs), "planned": p["n_send"], "sent": st["sent"],
                              "succeeded": st["succeeded"], "failed": st["failed"], "retries": st["retries"],
                              "not_sent": st["not_sent"],
                              "cached_before": sum(r["cached"] for r in p["rows"].values()),
                              "over_documented_limit": sum(r["over_limit"] for r in p["rows"].values())},
                   "model_versions": st["model_versions"], "version_headers": st["version_headers"],
                   "failures": st["failures"], "stopped": st["stopped"], "git": git_state(),
                   "benchmark": benchmark_version(), "audio_hours_planned": round(p["send_hours"], 3),
                   "cost_estimate": p["cost"], "python": sys.version.split()[0]}
            lp = root / "runs" / f"{started.replace(':', '')}.json"
            write_atomic(lp, json.dumps(log, indent=1, ensure_ascii=False), secret=key)
            print(f"run log {lp}: sent {st['sent']}, ok {st['succeeded']}, failed {st['failed']}, "
                  f"retries {st['retries']}, model_version {st['model_versions']}", file=sys.stderr)
            if dry:
                print(f"fake transport calls: {len(transport.calls)}", file=sys.stderr)
            if st["stopped"]:
                print(f"STOPPED: {st['stopped']}", file=sys.stderr)
                return 3
        t = build_table(jobs, root, a.pronunciada_cache, dry=dry)
        jp, mp = output_paths(dry, root)
        write_atomic(jp, json.dumps(t, indent=1, ensure_ascii=False) + "\n", secret=key)
        md = markdown(t)
        write_atomic(mp, md + "\n", secret=key)
        print(md)
        print(f"wrote {jp} and {mp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
