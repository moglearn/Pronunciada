"""bench.third_party.lingolix_live: caching, resume, retries, key handling, parsing, dry-run guard.

No test sends anything over the network: every request goes to a local fake transport."""
import json
import socket
import struct
import wave
from pathlib import Path

import pytest

pytest.importorskip("soundfile")
pytest.importorskip("requests")

from bench.third_party import lingolix as L  # noqa: E402
from bench.third_party import lingolix_live as LL  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "lingolix_v3_check_response.json"
SECRET = "lx_TEST_SECRET_4f9a2c71e0"
URL = LL.DEFAULT_BASE + LL.ENDPOINT_PATH


def _wav(p: Path, seconds=0.5, tone=1) -> Path:
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", (i * tone) % 300) for i in range(int(16000 * seconds))))
    return p


def _jobs(tmp_path, n=3, lang="en"):
    out = []
    for i in range(n):
        out.append(LL.Job("tier1_en", f"en-{i:04d}", lang, lang, _wav(tmp_path / f"a{i}.wav", tone=i + 1),
                          f"hello world number {i}", True))
    return out


def _todo(jobs, root):
    return LL.plan(jobs, root, refresh=False, max_requests=None)["todo"]


def _run(todo, transport, root, sleeps=None, key=SECRET, **kw):
    sleeps = [] if sleeps is None else sleeps
    return LL.run_jobs(todo, transport=transport, key=key, root=root, url=URL, rps=0, concurrency=1,
                       policy=LL.Policy(), sleep=sleeps.append, progress=False, **kw)


# ── parsing ──

def test_parse_fixture_like_the_benchmark_adapter():
    raw = json.loads(FIXTURE.read_text())
    words = LL.parse_response(raw, "The quick brown fox.", "en")
    assert [w["text"] for w in words] == ["The", "quick", "brown", "fox"]
    assert [w["score"] for w in words] == [0.96, 0.62, 0.0, 0.88]          # accuracy, already 0-1
    assert [w["error_type"] for w in words] == [None, None, "Omission", "Insertion"]
    rec = {"id": "x", "vendor": "lingolix", "words": words, "raw": raw}
    assert L.utterance_score(rec) == 0.71                                  # report.py: `overall`
    assert L.vendor_flags(rec, "en") is None                               # report.py: undefined for en
    # es: flag also when expected and detected syllable IPA differ (after report.py's merges)
    assert L.vendor_flags(rec, "es") == [False, True, True, True]
    man = [{"id": "x", "lang": "en", "target": "The quick brown fox.",
            "label": {"word_index": 2, "bucket": "sub_gross"}}]
    c = L.collect(man, {"x": rec})
    assert c["units"][0][1] == [pytest.approx(0.04), pytest.approx(0.38), 1.0, pytest.approx(0.12)]


@pytest.mark.parametrize("mutate, assumption", [
    (lambda r: r["words"][1].__setitem__("accuracy", 62), "A3"),           # a 0-100 scale is refused
    (lambda r: r.__setitem__("overall", 71.0), "A3"),
    (lambda r: r.pop("words"), "A4"),
    (lambda r: [w.__setitem__("text", "zzz") for w in r["words"]], "A4"),  # words that are not the sentence
    (lambda r: r["words"][0]["syllables"][0].__setitem__("is_missing", "no"), "A5"),
    (lambda r: r.__setitem__("model_version", 3), "A6"),
])
def test_unexpected_format_fails_loudly(mutate, assumption):
    raw = json.loads(FIXTURE.read_text())
    mutate(raw)
    with pytest.raises(LL.ResponseFormatError, match=assumption):
        LL.parse_response(raw, "The quick brown fox.", "en")


# ── cache and resume ──

def test_cache_hit_means_no_second_request(tmp_path):
    root = tmp_path / "cache"
    jobs = _jobs(tmp_path, 2)
    fake = LL.FakeTransport()
    st = _run(_todo(jobs, root), fake, root)
    assert st["succeeded"] == 2 and len(fake.calls) == 2
    p = LL.plan(jobs, root, refresh=False, max_requests=None)
    assert p["n_send"] == 0 and sum(r["cached"] for r in p["rows"].values()) == 2
    _run(p["todo"], fake, root)
    assert len(fake.calls) == 2
    assert LL.plan(jobs, root, refresh=True, max_requests=None)["n_send"] == 2   # only with --refresh
    # the key covers audio, text, language and options: a changed text is a new request
    jobs[0].text = "hello world changed"
    assert LL.plan(jobs, root, refresh=False, max_requests=None)["n_send"] == 1


def test_resume_after_interruption(tmp_path):
    root = tmp_path / "cache"
    jobs = _jobs(tmp_path, 5)

    class Dies(LL.FakeTransport):
        def post(self, url, **kw):
            if len(self.calls) == 2:
                raise KeyboardInterrupt
            return super().post(url, **kw)

    first = Dies()
    st = _run(_todo(jobs, root), first, root)
    assert st["stopped"].startswith("interrupted") and st["succeeded"] == 2
    assert not list(root.rglob("*.tmp"))                                  # atomic writes leave no debris
    second = LL.FakeTransport()
    todo = _todo(jobs, root)
    assert len(todo) == 3
    _run(todo, second, root)
    sent = [c["data"]["sentence"] for c in first.calls[:2] + second.calls]
    assert sorted(sent) == sorted(j.text for j in jobs)                   # each item answered exactly once
    assert LL.plan(jobs, root, refresh=False, max_requests=None)["n_send"] == 0


def test_max_requests_caps_a_run(tmp_path):
    root = tmp_path / "cache"
    p = LL.plan(_jobs(tmp_path, 4), root, refresh=False, max_requests=1)
    assert p["n_send"] == 1


# ── retries and stops ──

def test_backoff_on_429_honours_retry_after(tmp_path):
    root = tmp_path / "cache"
    fake = LL.FakeTransport(script={1: LL.Response(429, {"Retry-After": "7"}, b'{"message": "slow down"}'),
                                    2: LL.Response(503, {}, b"{}"), 3: LL.Response(502, {}, b"{}")})
    sleeps = []
    st = _run(_todo(_jobs(tmp_path, 1), root), fake, root, sleeps)
    assert st["succeeded"] == 1 and len(fake.calls) == 4
    assert 7.0 <= sleeps[0] <= 8.0                                        # Retry-After plus a little jitter
    base = LL.Policy().base_s
    assert base * 2 * 0.5 <= sleeps[1] <= base * 2 and base * 4 * 0.5 <= sleeps[2] <= base * 4   # exponential


def test_retry_after_http_date():
    import email.utils
    import time

    d = email.utils.formatdate(time.time() + 30, usegmt=True)
    assert 25 <= LL.retry_after_s({"retry-after": d}) <= 31


@pytest.mark.parametrize("resp, expect", [
    (LL.Response(401, {}, b'{"detail": "bad key"}'), "401"),
    (LL.Response(403, {}, b"{}"), "subscription"),
    (LL.Response(429, {"X-RateLimit-Remaining": "0"}, b'{"message": "quota", "plan": "free",'
                   b' "remaining_minutes": 0, "upgrade_url": "x"}'), "quota"),
])
def test_stops_cleanly_on_auth_and_quota(tmp_path, resp, expect):
    root = tmp_path / "cache"
    fake = LL.FakeTransport(script={1: resp})
    sleeps = []
    st = _run(_todo(_jobs(tmp_path, 3), root), fake, root, sleeps)
    assert expect in st["stopped"] and len(fake.calls) == 1 and not sleeps and st["succeeded"] == 0


def test_client_error_is_a_failure_not_a_retry(tmp_path):
    root = tmp_path / "cache"
    fake = LL.FakeTransport(script={1: LL.Response(400, {}, b'{"detail": "audio longer than 60 s"}')})
    st = _run(_todo(_jobs(tmp_path, 2), root), fake, root)
    assert st["failed"] == 1 and st["succeeded"] == 1 and len(fake.calls) == 2 and st["stopped"] is None


def test_bad_response_stops_and_keeps_the_paid_answer(tmp_path):
    root = tmp_path / "cache"
    raw = json.loads(FIXTURE.read_text())
    raw["words"][0]["accuracy"] = 96
    fake = LL.FakeTransport(script={1: LL.Response(200, {}, json.dumps(raw).encode())})
    st = _run(_todo(_jobs(tmp_path, 2), root), fake, root)
    assert "A3" in st["stopped"] and len(fake.calls) == 1
    assert len(list((root / "responses").rglob("*.json"))) == 1           # cached before it was checked


def test_rate_limiter_spaces_requests():
    t = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    rl = LL.RateLimiter(2.0, clock=lambda: t[0], sleep=sleep)
    for _ in range(4):
        rl.wait()
    assert slept == [0.5, 0.5, 0.5]


# ── the key ──

def test_refuses_without_key(monkeypatch, capsys):
    monkeypatch.delenv(LL.KEY_ENV, raising=False)
    monkeypatch.setattr(LL, "RequestsTransport", lambda: pytest.fail("transport built without a key"))
    monkeypatch.setattr(LL, "benchmark_jobs", lambda *a: pytest.fail("work started without a key"))
    assert LL.main(["--yes"]) == 2
    assert LL.KEY_ENV in capsys.readouterr().err


class _Echo(LL.FakeTransport):
    """A hostile server: echoes the Authorization header back in the body and in a header."""

    def post(self, url, *, headers, data, file, timeout):
        r = super().post(url, headers=headers, data=data, file=file, timeout=timeout)
        if r.status == 200:
            body = json.loads(r.body)
            body["debug_auth"] = headers["Authorization"]
            return LL.Response(200, {**r.headers, "X-Model-Version-Debug": headers["Authorization"]},
                               json.dumps(body).encode())
        return r


def test_key_never_in_logs_or_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(LL.KEY_ENV, SECRET)
    jobs = _jobs(tmp_path, 3)
    echo = _Echo(script={2: LL.Response(400, {}, f'{{"detail": "bad request for Bearer {SECRET}"}}'.encode())})
    monkeypatch.setattr(LL, "benchmark_jobs", lambda *a: jobs)
    monkeypatch.setattr(LL, "RequestsTransport", lambda: echo)
    monkeypatch.setattr(LL, "build_table", lambda *a, **k: {"note": "stub"})
    monkeypatch.setattr(LL, "markdown", lambda t: "stub table")
    out = tmp_path / "out"
    monkeypatch.setattr(LL, "output_paths", lambda dry, root: (out / "t.json", out / "t.md"))
    assert LL.main(["--yes", "--cache-dir", str(tmp_path / "cache"), "--rps", "1000"]) == 0
    assert any(SECRET in c["auth"] for c in echo.calls)                   # it was sent, as a header
    printed = capsys.readouterr()
    assert SECRET not in printed.out + printed.err
    files = [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix in (".json", ".md")]
    assert any("runs" in p.parts for p in files) and len(list((tmp_path / "cache").rglob("responses/*/*.json"))) == 2
    for p in files:
        assert SECRET not in p.read_text(encoding="utf-8"), p
    log = json.loads(next((tmp_path / "cache" / "runs").glob("*.json")).read_text())
    assert log["counts"]["failed"] == 1 and "[REDACTED]" in log["failures"][0]["error"]
    assert log["git"]["commit"] and log["endpoint"].endswith(LL.ENDPOINT_PATH)


# ── dry run never leaves the machine ──

def test_guard_blocks_non_local_hosts():
    with LL.no_network_guard():
        with pytest.raises(LL.NetworkBlocked):
            socket.create_connection(("api.lingolix.com", 443), timeout=1)
        with pytest.raises(LL.NetworkBlocked):
            socket.getaddrinfo("api.lingolix.com", 443)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(LL.NetworkBlocked):
                s.connect(("93.184.216.34", 443))
        finally:
            s.close()
        with pytest.raises((LL.NetworkBlocked, LL.TransientError)):
            LL.RequestsTransport().post(URL, headers={}, data={"sentence": "x", "language_code": "en"},
                                        file=("a.wav", b"RIFF", "audio/wav"), timeout=1)
    assert socket.getaddrinfo is not None and socket.create_connection.__name__ == "create_connection"


def test_dry_run_contacts_no_remote_host(tmp_path, monkeypatch, capsys):
    """Fails if a dry run would build the real transport or open a connection to a non-local host."""
    attempts = []
    real_connect, real_gai = socket.socket.connect, socket.getaddrinfo

    def spy_connect(self, address):
        attempts.append(address)
        return real_connect(self, address)

    def spy_gai(host, *a, **k):
        attempts.append(host)
        return real_gai(host, *a, **k)

    monkeypatch.setattr(socket.socket, "connect", spy_connect)
    monkeypatch.setattr(socket, "getaddrinfo", spy_gai)
    monkeypatch.setenv(LL.KEY_ENV, SECRET)
    monkeypatch.setattr(LL, "RequestsTransport", lambda: pytest.fail("real transport built in a dry run"))
    jobs = _jobs(tmp_path, 3)
    monkeypatch.setattr(LL, "benchmark_jobs", lambda *a: jobs)
    seen = {}
    monkeypatch.setattr(LL, "build_table", lambda j, root, pc, dry: seen.update(root=root, dry=dry) or
                        {"dry_run": dry, "banner": LL.BANNER})
    monkeypatch.setattr(LL, "markdown", lambda t: LL.BANNER)
    assert LL.main(["--dry-run", "--cache-dir", str(tmp_path / "cache")]) == 0
    assert attempts == []
    assert seen["dry"] and seen["root"] == tmp_path / "cache" / LL.DRY_SUBDIR
    files = list((tmp_path / "cache" / LL.DRY_SUBDIR).rglob("*.json"))
    assert files and all(SECRET not in p.read_text() for p in files)
    assert all(json.loads(p.read_text()).get("response", {}).get("model_version", "FAKE-dry-run") == "FAKE-dry-run"
               for p in files if "responses" in p.parts)
    assert not (tmp_path / "cache" / "responses").exists()               # never the real cache
    jp, mp = LL.output_paths(True, tmp_path / "cache" / LL.DRY_SUBDIR)
    assert LL.ROOT / "results" not in jp.parents and LL.ROOT / "results" not in mp.parents
    assert LL.BANNER in mp.read_text() and "DRY RUN, FAKE DATA" in capsys.readouterr().err


def test_fake_responses_follow_the_documented_format():
    for text, lang in (("Bonjour, le monde !", "fr"), ("学生は、レポートを置くと、出て行った。", "ja")):
        raw = LL.fake_response(text, lang, "0" * 64)
        LL.parse_response(raw, text, lang)
        assert set(raw) >= {"overall", "accuracy", "fluency", "model_version", "words"}
