"""Chivox pronunciation assessment (global developer portal) measured by us, against Pronunciada.

    export CHIVOX_API_KEY=...                                          # read from the environment only
    python -m bench.third_party.chivox_live --plan                     # what a full run sends and costs
    python -m bench.third_party.chivox_live --pilot --plan             # the 300 + 300 pilot
    python -m bench.third_party.chivox_live --limit 1 --yes --points-cap 2      # smoke test, 2 points
    python -m bench.third_party.chivox_live --pilot --yes --points-cap 600      # the pilot
    python -m bench.third_party.chivox_live --yes --points-cap 3245             # full run
    python -m bench.third_party.chivox_live --report-only [--pilot]    # results/chivox_live.md from the cache
    python -m bench.third_party.chivox_live --dry-run --limit 20       # the whole pipeline on a local fake

Documentation (read 2026-11-23, all public, no login): https://api-portal.cloud.chivox.com/global,
/docs, /pricing, /runtime, /legal/terms, /legal/privacy, and the vendor's MIT repository
github.com/chivox-developer/chivox-speech-eval-mcp (docs/api-reference.md, examples/quick-test/python).
Write-up: results/chivox_harness.md.

What is sent. Only speechocean762 TEST (2,500 clips, Apache-2.0; prompt = the scored words joined by
single spaces, the Lingolix all-2,500 convention, bench.third_party.lingolix.tier2x_items) and the
OMPAL TEST half (745 clips incl. natives, CC BY 4.0; bench.eval_ompal.records("test")). Any other set
is refused (live_core.check_datasets). Audio: decoded locally to 16 kHz mono 16-bit PCM WAV (the
docs recommend 16 kHz mono WAV or MP3), sent inline as `audio_base64`.

Request (MCP over Streamable HTTP, JSON-RPC 2.0; docs/api-reference.md §4):
    POST https://mcp-global.cloud.chivox.com/      Authorization: Bearer $CHIVOX_API_KEY
    1. initialize (once per run; not an evaluation, 0 points; the server's name and version are recorded)
    2. notifications/initialized
    3. per clip: tools/call {"name": "en_sentence_eval" | "cn_sentence_eval",
                             "arguments": {"ref_text", "audio_base64", ...}}
       English: accent 3 (neutral, the documented default), rank 100 (0-100 scale, the default).
       Mandarin: ref_text = the OMPAL prompt in Simplified characters (OpenCC t2s, the converter the
       Azure zh-CN run used; one character for one character), age_group "adult", rank 100.
    The evaluation JSON is the text of result.content[0] (examples/quick-test/python/test_eval.py).

Billing: points; 1 point per word / phrase / sentence evaluation, 2 per paragraph, 0 for a failed call
(/pricing). Both tools used here are sentence evaluations: 1 point per request. The cap counts 1 point
for EVERY attempt (retries included), so the ledger is an upper bound.

FIELD MAPPING, fixed from the documentation before any response existed (docs "Response fields"):
    sentence total        <- overall            ("Weighted roll-up")
    sentence accuracy     <- accuracy           ("Phoneme / syllable correctness")
    sentence fluency      <- fluency.overall
    sentence completeness <- integrity          (reported in the JSON only; no table row uses it)
    sentence prosodic     <- none documented    (row shown as "—")
    secondary             <- pron (speechocean762: vs expert accuracy), overall (OMPAL: vs teacher
                             accuracy, the slot Azure's PronScore takes in results/ompal.md)
    word                  <- details[i].score   (details[i].char = the word; Mandarin: the character)
    sound                 <- details[i].phone[j].score (phoneme in .phoneme; dp_type kept, not used)
    tone (Mandarin)       <- details[i].tone_ref vs tone_detected: 100 if equal, 0 if not (diagnostic
                             tone channel); the Mandarin "sounds" channel = min phone score of the unit.
All scores are on 0-100 ("all 0-100 unless noted").

ASSUMPTIONS the documentation does not settle; checked on every answer (`validate`); a real answer
that breaks one stops the run after it is cached, so fixing the adapter costs nothing:
  C1  The MCP endpoint above is the scoring path for audio_base64 (the function-calling host
      fc-global.cloud.chivox.com/v1/call has no documented body and is not used).
  C2  The server reads the WAV container (no audio_format / sample_rate argument is documented for
      the one-shot tools; the streaming session's defaults are wav-or-mp3, 16000 Hz, 1 channel, 2 bytes).
  C3  overall (and accuracy / pron / integrity when present) is a number in [0, 100]; fluency is a
      number or an object with `overall`.
  C4  details is a list of objects with `char` (string) and `score` (number in [0, 100]) in the order
      of the text; at least half of the reference words (English) / characters (Mandarin) appear.
  C5  details[i].phone, when present, is a list of objects with `phoneme` and `score` in [0, 100].
  C6  No version field is documented; the version string is the MCP serverInfo name + version of
      the session, plus any top-level `version` / `engine_version` / `res` field of the result.
  C7  Quota / billing errors: HTTP 401, 402, 403 ("API Key invalid, disabled, or quota exhausted"),
      429 ("structured 429" when a key cap is reached; any 429 without Retry-After stops the run),
      or a JSON-RPC error whose message names quota / points / balance / credits. They stop the run.
"""
from __future__ import annotations

import base64
import hashlib
import json
import random
import re
import sys

from bench.third_party import live_core as C

NAME = "Chivox"
UNIT = "point"
KEY_ENV = "CHIVOX_API_KEY"
CAP_FLAG = "--points-cap"
WRITEUP = "chivox_harness.md"
ALLOWED = ("speechocean762", "ompal")
PILOT = {"speechocean762": 300, "ompal": 300}            # 600 points = the free allowance
FULL_SIZE = {"speechocean762": 2500, "ompal": 745}
HOST = "mcp-global.cloud.chivox.com"
URL = f"https://{HOST}/"
PROTOCOL = "2025-03-26"
TOOLS = {"en": "en_sentence_eval", "zh": "cn_sentence_eval"}
ARGS = {"en": {"accent": 3, "rank": 100}, "zh": {"age_group": "adult", "rank": 100}}
OPTIONS = {"endpoint": URL, "transport": "MCP Streamable HTTP, JSON-RPC 2.0 tools/call", "tools": TOOLS, "args": ARGS,
           "audio": "audio_base64 of a 16 kHz mono PCM16 WAV", "zh_text": "Simplified (OpenCC t2s)"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
PRICES = {"source": "https://api-portal.cloud.chivox.com/pricing (read 2026-11-23)", "currency": "USD",
          "per_sentence": {"Standard": 0.0038, "Advanced": 0.0035, "Flagship": 0.0032},
          "free_points": 600, "free_valid_days": 30}
MAPPING = {
    "sentence_total": "overall", "sentence_accuracy": "accuracy", "sentence_fluency": "fluency.overall",
    "sentence_completeness": "integrity", "sentence_prosodic": None,
    "secondary": {"speechocean762": "pron", "ompal": "overall"},
    "word": "details[i].score", "sound": "details[i].phone[j].score",
    "tone": "details[i].tone_ref == details[i].tone_detected -> 100 else 0 (Mandarin, diagnostic)",
    "mandarin_sounds_channel": "min details[i].phone[j].score",
    "scale": "0-100", "source": "https://api-portal.cloud.chivox.com/docs (Response fields), read 2026-11-23"}

_QUOTA = re.compile(r"quota|insufficient|\bpoints?\b|balance|credit|exhaust|billing|payment|cap (?:reached|exceeded)",
                    re.I)
_AUTH = re.compile(r"unauthori[sz]ed|api[ _-]?key|invalid key|forbidden|permission|jwt|token", re.I)
_TRANSIENT = re.compile(r"timeout|timed out|upstream|capacity|busy|temporar|unavailable|try again", re.I)
_SESSION = re.compile(r"session", re.I)


def api_lang(item: C.Item) -> str:
    return item.lang


def cost(seconds: float) -> float:
    return 1.0


def over_limit(item: C.Item, audio: bytes) -> str | None:
    return "over 50 MB" if len(audio) * 4 / 3 > MAX_UPLOAD_BYTES else None


def endpoint_desc() -> str:
    return f"POST {URL} (MCP tools/call {TOOLS['en']} / {TOOLS['zh']})"


def price_note(points: float) -> str:
    p = PRICES["per_sentence"]
    return (f"Free allowance: {PRICES['free_points']} points, valid {PRICES['free_valid_days']} days. At the published "
            f"per-sentence prices this run would cost USD {points * p['Flagship']:.2f}-{points * p['Standard']:.2f} "
            f"(Flagship-Standard, before any bonus points). Source: {PRICES['source']}.")


def check_key(key: str) -> str | None:
    return None if len(key) >= 8 else f"{KEY_ENV} looks too short to be a key"


def policy() -> C.Policy:
    return C.Policy(rps=0.5)


def real_transport():
    return C.RequestsTransport((HOST,))


def fake_transport():
    return FakeTransport()


# ── MCP plumbing ───────────────────────────────────────────────────────────────

def new_session() -> dict:
    return {}


def session_info(s: dict) -> dict:
    return {"server": s.get("server"), "protocol": s.get("protocol"), "has_session_id": bool(s.get("id"))}


def _headers(key: str, s: dict) -> dict:
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
         "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": PROTOCOL}
    if s.get("id"):
        h["Mcp-Session-Id"] = s["id"]
    return h


def rpc_messages(resp: C.Response) -> list[dict]:
    """JSON-RPC messages of a Streamable HTTP answer: a JSON body, or server-sent events."""
    ctype = next((v for k, v in resp.headers.items() if k.lower() == "content-type"), "") or ""
    text = resp.body.decode("utf-8", "replace")
    if "text/event-stream" in ctype or text.lstrip().startswith(("event:", "data:")):
        out, buf = [], []
        for line in text.splitlines() + [""]:
            if line.startswith("data:"):
                buf.append(line[5:].lstrip())
            elif not line.strip() and buf:
                try:
                    m = json.loads("\n".join(buf))
                    out += m if isinstance(m, list) else [m]
                except ValueError:
                    pass
                buf = []
        return [m for m in out if isinstance(m, dict)]
    m = resp.json()
    if isinstance(m, list):
        return [x for x in m if isinstance(x, dict)]
    return [m] if isinstance(m, dict) else []


def rpc_reply(resp: C.Response, want_id) -> dict | None:
    msgs = rpc_messages(resp)
    for m in msgs:
        if m.get("id") == want_id and ("result" in m or "error" in m):
            return m
    return next((m for m in msgs if "result" in m or "error" in m), None)


def ensure_session(s: dict, *, transport, key: str, budget: C.Budget, policy: C.Policy) -> None:
    if s.get("ready"):
        return
    body = {"jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": PROTOCOL, "capabilities": {},
                       "clientInfo": {"name": "pronunciada-bench", "version": "1"}}}
    if budget.ledger is not None:
        budget.ledger.append({"kind": "session", "utc": C.utc_now(), "vendor": NAME, "method": "initialize",
                              "cost": 0, "unit": UNIT})
    try:
        r = transport.post(URL, headers=_headers(key, {}), json_body=body, timeout=policy.timeout_s)
    except C.TransientError as e:
        raise C.Fatal(f"Chivox initialize failed ({e}); nothing scored, rerun later") from None
    if budget.ledger is not None:
        budget.ledger.append({"kind": "session_result", "utc": C.utc_now(), "vendor": NAME, "status": r.status})
    if r.status in (401, 402, 403):
        raise C.Fatal(f"Chivox refused the key at initialize (HTTP {r.status}); check {KEY_ENV}")
    if not 200 <= r.status < 300:
        raise C.Fatal(f"Chivox initialize answered HTTP {r.status}; run stopped")
    m = rpc_reply(r, 0) or {}
    if "error" in m:
        raise C.Fatal(C.scrub(f"Chivox initialize error: {str(m['error'])[:200]}", key))
    res = m.get("result") or {}
    sid = next((v for k, v in r.headers.items() if k.lower() == "mcp-session-id"), None)
    info = res.get("serverInfo") or {}
    s.update(id=sid, server={"name": info.get("name"), "version": info.get("version")},
             protocol=res.get("protocolVersion"), next_id=1, ready=True)
    try:
        transport.post(URL, headers=_headers(key, s), json_body={"jsonrpc": "2.0", "method": "notifications/initialized"},
                       timeout=policy.timeout_s)
    except C.TransientError:
        pass


def request_body(s: dict, item: C.Item, audio: bytes) -> dict:
    s["next_id"] = s.get("next_id", 1) + 1
    s["last_id"] = s["next_id"]
    args = {"ref_text": item.text, "audio_base64": base64.b64encode(audio).decode("ascii"), **ARGS[item.lang]}
    return {"jsonrpc": "2.0", "id": s["last_id"], "method": "tools/call",
            "params": {"name": TOOLS[item.lang], "arguments": args}}


def send(s: dict, item: C.Item, audio: bytes, h: str, *, transport, key: str, policy: C.Policy) -> C.Response:
    return transport.post(URL, headers=_headers(key, s), json_body=request_body(s, item, audio), timeout=policy.timeout_s)


def ledger_extra(resp: C.Response) -> dict:
    return {}


def _tool_text(m: dict) -> tuple[bool, str]:
    res = m.get("result") or {}
    parts = [c.get("text", "") for c in res.get("content") or [] if isinstance(c, dict) and c.get("type") == "text"]
    return bool(res.get("isError")), "\n".join(parts)


def _message_kind(msg: str) -> str:
    if _QUOTA.search(msg) or _AUTH.search(msg):
        return "fatal"
    if _TRANSIENT.search(msg):
        return "retry"
    if _SESSION.search(msg) and re.search(r"not found|expired|invalid", msg, re.I):
        return "session"
    return "rejected"


def classify(resp: C.Response, s: dict | None = None) -> tuple[str, str]:
    """answered | rejected (final, cached) | retry | session (re-initialize, retry) | fatal | fail."""
    st = resp.status
    snippet = resp.body[:300].decode("utf-8", "replace")
    if st in (401, 402, 403):
        return "fatal", {401: "key missing or rejected (401)", 402: "payment required (402)",
                         403: "key invalid, disabled, quota exhausted, or no permission for the tool (403)"}[st]
    if st == 404:
        return "session", "session not found (404)"
    if st == 429:
        if C.retry_after_s(resp.headers) is not None and not _QUOTA.search(snippet):
            return "retry", "rate limited (429 with Retry-After)"
        return "fatal", "429: key cap or quota reached (structured 429, /runtime); nothing more will be sent"
    if st in (408, 425) or st >= 500:
        return "retry", f"HTTP {st}"
    if st == 400:
        return "rejected", f"HTTP 400: {snippet}"
    if not 200 <= st < 300:
        return "fail", f"HTTP {st}: {snippet}"
    m = rpc_reply(resp, (s or {}).get("last_id"))
    if m is None:
        return "fatal", "a 2xx answer without a JSON-RPC result (C1); the adapter must be checked"
    if "error" in m:
        msg = json.dumps(m["error"], ensure_ascii=False)[:300]
        return _message_kind(msg), f"JSON-RPC error: {msg}"
    is_err, text = _tool_text(m)
    if is_err:
        return _message_kind(text), f"tool error: {text[:300]}"
    try:
        ev = json.loads(text)
    except ValueError:
        return "rejected", f"tool answer is not JSON: {text[:200]}"
    if not isinstance(ev, dict):
        return "rejected", "tool answer is not a JSON object"
    ev = _unwrap(ev)
    if "overall" not in ev and "details" not in ev:
        return _message_kind(text), f"tool answer without scores: {text[:200]}"
    return "answered", "ok"


def _unwrap(ev: dict) -> dict:
    """The Quickstart reads result.details: accept the evaluation either at the top or under `result`."""
    if "overall" not in ev and isinstance(ev.get("result"), dict):
        return ev["result"]
    return ev


def evaluation(rec_or_resp) -> dict:
    if isinstance(rec_or_resp, C.Response):
        resp = rec_or_resp
    else:
        resp = C.Response(int(rec_or_resp.get("status") or 200), rec_or_resp.get("headers") or {},
                          rec_or_resp["body"].encode("utf-8"))
    m = rpc_reply(resp, None) or {}
    _, text = _tool_text(m)
    return _unwrap(json.loads(text))


def version(resp: C.Response, s: dict) -> str:
    srv = (s or {}).get("server") or {}
    v = f"{srv.get('name') or 'mcp'}-{srv.get('version') or 'unknown'}"
    try:
        ev = evaluation(resp)
    except (ValueError, TypeError):
        ev = {}
    extra = [str(ev[k]) for k in ("version", "engine_version", "res") if isinstance(ev.get(k), (str, int, float))]
    return "_".join([v] + extra)


# ── checking and reading an answer ─────────────────────────────────────────────

def _num01(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 100


def _fluency(ev):
    f = ev.get("fluency")
    if isinstance(f, dict):
        return f.get("overall")
    return f


def validate_eval(ev: dict, text: str, lang: str) -> None:
    def bad(a, msg):
        raise C.ResponseFormatError(f"Chivox answer does not match assumption {a}: {msg}. Check the adapter "
                                    "(bench/third_party/chivox_live.py) against the cached answer.")

    if not _num01(ev.get("overall")):
        bad("C3", f"`overall` = {ev.get('overall')!r}, expected a number in [0, 100]")
    for k in ("accuracy", "pron", "integrity"):
        if ev.get(k) is not None and not _num01(ev[k]):
            bad("C3", f"`{k}` = {ev[k]!r}")
    f = _fluency(ev)
    if f is not None and not _num01(f):
        bad("C3", f"fluency = {ev.get('fluency')!r}")
    det = ev.get("details")
    if not isinstance(det, list) or (text.strip() and not det):
        bad("C4", "no `details` list")
    for i, d in enumerate(det):
        if not isinstance(d, dict) or not isinstance(d.get("char"), str) or not _num01(d.get("score")):
            bad("C4", f"details[{i}] = {str(d)[:120]}")
        ph = d.get("phone")
        if ph is not None:
            if not isinstance(ph, list) or not all(isinstance(p, dict) and isinstance(p.get("phoneme"), str)
                                                   and _num01(p.get("score")) for p in ph):
                bad("C5", f"details[{i}].phone = {str(ph)[:120]}")
    if lang == "en":
        from bench.third_party import lingolix as L
        want = [L.key(t) for t in text.split() if L.key(t)]
        got = {L.key(d["char"]) for d in det}
        hit = sum(w in got for w in want)
        if want and hit < 0.5 * len(want):
            bad("C4", f"only {hit} of {len(want)} words of the sentence appear in `details`")
    else:
        han = [ch for ch in text if "一" <= ch <= "鿿"]
        got = "".join(d["char"] for d in det)
        hit = sum(ch in got for ch in han)
        if han and hit < 0.5 * len(han):
            bad("C4", f"only {hit} of {len(han)} characters of the sentence appear in `details`")


def validate(resp: C.Response, item: C.Item) -> None:
    validate_eval(evaluation(resp), item.text, item.lang)


def _tone(x):
    m = re.search(r"[0-5]", str(x)) if x is not None else None
    return m.group() if m else None


def view(rec: dict, item: C.Item) -> dict | None:
    ev = evaluation(rec)
    validate_eval(ev, item.text, item.lang)
    sent = {"total": ev.get("overall"), "accuracy": ev.get("accuracy"), "fluency": _fluency(ev), "prosodic": None,
            "completeness": ev.get("integrity"),
            "secondary": ev.get("pron") if item.lang == "en" else ev.get("overall")}
    words = [{"text": d["char"], "score": float(d["score"]),
              "phones": [(p["phoneme"], float(p["score"])) for p in d.get("phone") or []]} for d in ev["details"]]
    out = {"sentence": sent, "words": words}
    if item.lang == "zh":
        units, cur = [], 0
        for d in ev["details"]:
            t = d["char"].strip()
            j = item.text.find(t, cur) if t else -1
            if j < 0:
                continue
            tr, td = _tone(d.get("tone_ref")), _tone(d.get("tone_detected"))
            tone = None if tr is None or td is None else (100.0 if tr == td else 0.0)
            ph = [float(p["score"]) for p in d.get("phone") or []]
            units.append((j, j + len(t), float(d["score"]), tone, min(ph) if ph else None))
            cur = j + len(t)
        out["zh_units"] = units
    return out


# ── the local fake ─────────────────────────────────────────────────────────────

def fake_eval(text: str, lang: str, audio_sha: str) -> dict:
    """An evaluation in the documented shape (docs "What the engine returns"), random scores."""
    rng = random.Random(hashlib.sha256(f"{audio_sha}|{text}|{lang}".encode()).hexdigest())
    units = [c for c in text if "一" <= c <= "鿿"] if lang == "zh" else re.findall(r"[\w']+", text)
    det = []
    for u in units:
        sc = rng.uniform(20, 60) if rng.random() < 0.12 else rng.uniform(70, 100)
        n = 2 if lang == "zh" else max(1, min(len(u), 6) - 1)
        phones = [{"phoneme": u[k % len(u)].lower(), "score": round(min(100, max(0, sc + rng.uniform(-15, 10))), 1),
                   "dp_type": "normal" if sc > 60 else "mispron"} for k in range(n)]
        d = {"char": u, "score": round(sc, 1), "phone": phones}
        if lang == "zh":
            t = rng.randint(1, 4)
            d.update(tone_ref=f"T{t}", tone_detected=f"T{t if sc > 55 else (t % 4) + 1}")
        det.append(d)
    m = sum(d["score"] for d in det) / len(det) if det else 0.0
    return {"overall": round(m, 1), "accuracy": round(m - 2, 1), "pron": round(m + 1, 1), "integrity": 100,
            "fluency": {"overall": round(rng.uniform(60, 100), 1), "speed": 65, "pause": rng.randint(0, 3)},
            "details": det}


class FakeTransport:
    """Local stand-in for the MCP server. `script` maps a tools/call number (1-based) to a Response.
    Records every call (auth header kept only to test that it is never logged elsewhere)."""

    def __init__(self, script: dict | None = None, sse: bool = False, session_id: str = "fake-session-1"):
        self.calls: list[dict] = []
        self.script = dict(script or {})
        self.sse, self.session_id = sse, session_id
        self.n_tool = 0

    def post(self, url, *, headers, json_body=None, data=None, files=None, timeout=60.0) -> C.Response:
        self.calls.append({"url": url, "headers": dict(headers), "body": json_body})
        method = (json_body or {}).get("method")
        if method == "initialize":
            m = {"jsonrpc": "2.0", "id": json_body["id"], "result": {
                "protocolVersion": PROTOCOL, "capabilities": {"tools": {}},
                "serverInfo": {"name": "FAKE-chivox", "version": "dry-run"}}}
            return self._reply(m, {"Mcp-Session-Id": self.session_id})
        if method == "notifications/initialized":
            return C.Response(202, {}, b"")
        self.n_tool += 1
        if self.n_tool in self.script:
            return self.script[self.n_tool]
        args = json_body["params"]["arguments"]
        lang = "zh" if json_body["params"]["name"].startswith("cn_") else "en"
        sha = hashlib.sha256(base64.b64decode(args["audio_base64"])).hexdigest()
        ev = fake_eval(args["ref_text"], lang, sha)
        return self._reply({"jsonrpc": "2.0", "id": json_body["id"],
                            "result": {"content": [{"type": "text", "text": json.dumps(ev, ensure_ascii=False)}]}}, {})

    def _reply(self, m: dict, extra: dict) -> C.Response:
        if self.sse:
            return C.Response(200, {"Content-Type": "text/event-stream", **extra},
                              f"event: message\ndata: {json.dumps(m, ensure_ascii=False)}\n\n".encode())
        return C.Response(200, {"Content-Type": "application/json", **extra}, json.dumps(m, ensure_ascii=False).encode())


def main(argv=None) -> int:
    return C.main(sys.modules[__name__], argv)


if __name__ == "__main__":
    sys.exit(main())
