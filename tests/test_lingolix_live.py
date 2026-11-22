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
    monkeypatch.setattr(LL, "tier2x_jobs", lambda: [])
    monkeypatch.setattr(LL, "RequestsTransport", lambda: echo)
    monkeypatch.setattr(LL, "build_table", lambda *a, **k: {"note": "stub"})
    monkeypatch.setattr(LL, "markdown", lambda t: "stub table")
    out = tmp_path / "out"
    monkeypatch.setattr(LL, "output_paths", lambda dry, root: (out / "t.json", out / "t.md"))
    assert LL.main(["--yes", "--cache-dir", str(tmp_path / "cache"), "--rps", "1000", "--quota-cap-min", "5"]) == 0
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
    monkeypatch.setattr(LL, "tier2x_jobs", lambda: [])
    seen = {}
    monkeypatch.setattr(LL, "build_table", lambda j, root, pc, dry, **k: seen.update(root=root, dry=dry) or
                        {"dry_run": dry, "banner": LL.BANNER})
    monkeypatch.setattr(LL, "markdown", lambda t: LL.BANNER)
    assert LL.main(["--dry-run", "--cache-dir", str(tmp_path / "cache"), "--quota-cap-min", "1"]) == 0
    assert attempts == []
    assert seen["dry"] and seen["root"] == tmp_path / "cache" / LL.DRY_SUBDIR
    files = list((tmp_path / "cache" / LL.DRY_SUBDIR).rglob("*.json"))
    assert files and all(SECRET not in p.read_text() for p in files)
    assert all(json.loads(p.read_text()).get("response", {}).get("model_version", "FAKE-dry-run") == "FAKE-dry-run"
               for p in files if "responses" in p.parts)
    assert not (tmp_path / "cache" / "responses").exists()               # never the real cache
    assert not (tmp_path / "cache" / LL.LEDGER).exists()                 # nor the real ledger
    dry_ledger = LL.Ledger(tmp_path / "cache" / LL.DRY_SUBDIR / LL.LEDGER)
    assert sum(e["kind"] == "attempt" for e in dry_ledger.entries()) >= 3
    assert sorted(p.relative_to(tmp_path / "cache").parts[0] for p in (tmp_path / "cache").rglob("*")
                  if p.is_file()) == [LL.DRY_SUBDIR] * len([p for p in (tmp_path / "cache").rglob("*") if p.is_file()])
    jp, mp = LL.output_paths(True, tmp_path / "cache" / LL.DRY_SUBDIR)
    assert LL.ROOT / "results" not in jp.parents and LL.ROOT / "results" not in mp.parents
    assert LL.BANNER in mp.read_text() and "DRY RUN, FAKE DATA" in capsys.readouterr().err


def test_fake_responses_follow_the_documented_format():
    for text, lang in (("Bonjour, le monde !", "fr"), ("学生は、レポートを置くと、出て行った。", "ja")):
        raw = LL.fake_response(text, lang, "0" * 64)
        LL.parse_response(raw, text, lang)
        assert set(raw) >= {"overall", "accuracy", "fluency", "model_version", "words"}


def test_nested_quota_answer_stops_the_run():
    """The shape the API really returns when the monthly quota is used up (seen 2026-11-22)."""
    body = (b'{"error":{"code":"HTTP_429","message":{"error":"quota_exceeded","message":"Monthly quota exceeded.",'
            b'"quota_percentage":99.8,"remaining_minutes":0.03,"plan":"api_free","upgrade_url":"https://example.invalid"},'
            b'"path":"/api/pronunciation/v3/check","method":"POST"}}')
    assert LL.classify_status(LL.Response(429, {}, body)) == "fatal"
    assert LL.classify_status(LL.Response(429, {}, b'{"error":{"code":"HTTP_429","message":"slow down"}}')) == "retry"


# ── audio-minute budget, ledger, speechocean762 extension ──

def _spk_jobs(tmp_path, spec):
    """spec: [(speaker, [seconds, ...]), ...] -> tier2x jobs (job order as given)."""
    out = []
    for spk, secs in spec:
        for k, s in enumerate(secs):
            p = _wav(tmp_path / f"s{spk}_{k}.wav", seconds=s, tone=k + 2)
            out.append(LL.Job(LL.TIER2X, f"{spk}{k:04d}", "en", "en", p, f"word {spk} {k}", True, spk))
    return out


def test_speaker_selection_never_splits_and_stops_at_first_misfit(tmp_path):
    root = tmp_path / "cache"
    # job order is not speaker order: selection must sort by speaker ID (string order)
    jobs = _spk_jobs(tmp_path, [("0300", [0.25]), ("0003", [0.5, 0.5, 0.5]), ("0024", [1.0, 1.0]),
                                ("0111", [0.25])])
    # 0003 (1.5 s) fits, 0024 (2.0 s) does not; 0111 and 0300 would fit but come after: never skipped ahead
    p = LL.plan(jobs, root, refresh=False, max_requests=None, used_s=10.0, cap_s=10.0 + 3.0)
    assert [j.group for j, _, _ in p["todo"]] == ["0003"] * 3
    s = p["selection"]
    assert s["speakers"] == ["0003"] and s["clips"] == 3 and s["next_speaker"]["speaker"] == "0024"
    assert p["budget"]["held_back"] == 4 and p["budget"]["stopped_by"] == "--quota-cap-min"
    assert p["rows"][LL.TIER2X]["held_back"] == 4
    # room for 0003, 0024 and 0111 but not 0300
    p = LL.plan(jobs, root, refresh=False, max_requests=None, used_s=0.0, cap_s=3.75 + 2e-6)
    assert [j.group for j, _, _ in p["todo"]] == ["0003"] * 3 + ["0024"] * 2 + ["0111"]
    assert p["budget"]["add_s"] <= p["budget"]["cap_s"]
    # max-requests never splits a speaker either
    p = LL.plan(jobs, root, refresh=False, max_requests=4, used_s=0.0, cap_s=None)
    assert [j.group for j, _, _ in p["todo"]] == ["0003"] * 3
    # cached clips are not counted: a speaker contributes only what is still to send
    _run(_todo(jobs[1:3], root), LL.FakeTransport(), root)
    p = LL.plan(jobs, root, refresh=False, max_requests=None, used_s=0.0, cap_s=2.6)   # 0.5 + 2.0 left, not 0111
    assert [j.id for j, _, _ in p["todo"]] == [jobs[3].id, jobs[4].id, jobs[5].id]
    assert p["rows"][LL.TIER2X]["cached"] == 2


@pytest.mark.parametrize("concurrency", [1, 4])
def test_cap_is_never_crossed_with_retries_failures_and_concurrency(tmp_path, concurrency):
    for trial in range(5):
        root = tmp_path / f"cache{trial}"
        jobs = _spk_jobs(tmp_path, [(f"{trial}{k:03d}", [0.5, 0.25, 0.75]) for k in range(6)])
        fake = LL.FakeTransport(transient_every=3, unavailable_every=4,
                                script={2: LL.Response(400, {}, b'{"detail": "bad audio"}')})
        ledger = LL.Ledger(root / LL.LEDGER)
        used, cap = 60.0, 60.0 + 4.1          # room for some clips and a few retries, not for all
        budget = LL.Budget(cap, used, ledger)
        todo = LL.plan(jobs, root, refresh=False, max_requests=None)["todo"]
        st = LL.run_jobs(todo, transport=fake, key=SECRET, root=root, url=URL, rps=0, concurrency=concurrency,
                         policy=LL.Policy(max_attempts=6, base_s=0, cap_s=0), sleep=lambda s: None, progress=False,
                         budget=budget)
        assert budget.used_s <= cap
        assert used + ledger.total_s() == pytest.approx(budget.used_s) and ledger.total_s() <= cap - used
        attempts = [e for e in ledger.entries() if e["kind"] == "attempt"]
        assert len(attempts) == len(fake.calls) == budget.attempts          # every network call was budgeted
        assert len([e for e in ledger.entries() if e["kind"] == "result"]) == len(attempts)
        assert {str(e["status"]) for e in ledger.entries() if e["kind"] == "result"} >= {"200", "429", "503"}
        assert "cap" in st["stopped"]
        assert st["succeeded"] + st["failed"] < len(todo)


def test_a_retry_is_refused_when_it_would_cross_the_cap(tmp_path):
    root = tmp_path / "cache"
    jobs = _jobs(tmp_path, 1)                 # 0.5 s
    fake = LL.FakeTransport(script={n: LL.Response(503, {}, b"{}") for n in range(1, 10)})
    budget = LL.Budget(1.2, 0.0, LL.Ledger(root / LL.LEDGER))
    st = _run(_todo(jobs, root), fake, root, budget=budget)
    assert len(fake.calls) == 2 and budget.used_s == pytest.approx(1.0) and "cap" in st["stopped"]


def test_ledger_backfill_is_idempotent(tmp_path):
    root = tmp_path / "cache"
    jobs = _jobs(tmp_path, 3)
    _run(_todo(jobs, root), LL.FakeTransport(), root)          # cached without a ledger (as before it existed)
    ledger = LL.Ledger(root / LL.LEDGER)
    resolve = LL.job_resolver(jobs)
    first = LL.backfill_ledger(root, ledger, resolve)
    assert first["added"] == 3 and first["seconds"] == pytest.approx(1.5)
    for _ in range(2):
        assert LL.backfill_ledger(root, ledger, resolve)["added"] == 0
    assert len(ledger.entries()) == 3 and ledger.total_s() == pytest.approx(1.5)
    # a response sent under the budget is in the ledger as an attempt: it is never backfilled as well
    (tmp_path / "more").mkdir()
    more = [LL.Job("tier1_en", "en-9999", "en", "en", _wav(tmp_path / "more" / "b.wav", seconds=0.5, tone=9),
                   "one more", True)]
    _run(_todo(more, root), LL.FakeTransport(), root, budget=LL.Budget(None, ledger.total_s(), ledger))
    assert LL.backfill_ledger(root, ledger, LL.job_resolver(jobs + more))["added"] == 0
    assert ledger.total_s() == pytest.approx(2.0)


def test_unreadable_ledger_fails_closed(tmp_path):
    ledger = LL.Ledger(tmp_path / LL.LEDGER)
    ledger.append({"kind": "backfill", "key": "k", "seconds": 1.0})
    with open(ledger.path, "a") as f:
        f.write('{"kind": "attempt", "secon')
    with pytest.raises(LL.Fatal, match="cannot be read"):
        ledger.total_s()


@pytest.mark.parametrize("extra", [[], ["--so762-extend"]])
def test_live_send_refuses_without_a_cap(monkeypatch, capsys, extra):
    monkeypatch.setenv(LL.KEY_ENV, SECRET)
    monkeypatch.setattr(LL, "RequestsTransport", lambda: pytest.fail("transport built without a cap"))
    monkeypatch.setattr(LL, "benchmark_jobs", lambda *a: pytest.fail("work started without a cap"))
    monkeypatch.setattr(LL, "tier2x_jobs", lambda: pytest.fail("work started without a cap"))
    assert LL.main(["--yes"] + extra) == 2
    err = capsys.readouterr().err
    assert "--quota-cap-min" in err and SECRET not in err


def test_speaker_bootstrap_matches_the_metric_and_is_seeded():
    import random

    from scipy.stats import pearsonr

    rng = random.Random(1)
    man, ra, rb, spk = [], {}, {}, {}
    for s in range(12):
        for k in range(4):
            i = f"{s:04d}{k}"
            words = [{"text": t, "accuracy": rng.randint(3, 10), "mispronounced": False} for t in ("aa", "bb", "cc")]
            man.append({"id": i, "lang": "en", "target": "aa bb cc", "label": None,
                        "human": {"utterance": {"total": rng.randint(3, 10)}, "words": words}})
            spk[i] = f"{s:04d}"
            for R, noise in ((ra, 0.5), (rb, 2.0)):
                sc = [min(1.0, max(0.0, w["accuracy"] / 10 + rng.gauss(0, noise / 10))) for w in words]
                R[i] = {"id": i, "words": [{"text": t, "score": x, "error_type": None}
                                           for t, x in zip(("aa", "bb", "cc"), sc)],
                        "raw": {"overall": sum(sc) / 3}}
    for which, k in (("utt", "pcc_utterance"), ("word", "pcc_word")):
        bs = LL.speaker_bootstrap(man, ra, rb, which, spk, n=2000)
        c = L.collect(man, ra)[f"{which}_pairs"]
        assert bs["full_sample"][0] == pytest.approx(pearsonr([p[0] for p in c], [p[1] for p in c]).statistic)
        assert bs["full_sample"][0] == pytest.approx(L.metrics(man, ra)[k])
        assert bs["full_sample"][1] == pytest.approx(L.metrics(man, rb)[k])
        d = bs["full_sample"][0] - bs["full_sample"][1]
        assert bs["n_speakers"] == 12 and bs["delta_ci"][0] < d < bs["delta_ci"][1]
        assert bs == LL.speaker_bootstrap(man, ra, rb, which, spk, n=2000)          # seeded


_so762 = pytest.mark.skipif(not L.SO762_LOCAL.exists() or not L.repo_dir("repo_v2026.10").exists(),
                            reason="speechocean762 manifest or the benchmark repository not present")


@_so762
def test_tier2x_items_reproduce_the_benchmark_and_its_cache_keys():
    import re

    items, spk = L.tier2x_items()
    assert len(items) == 2500 and len(set(spk.values())) == 125
    assert L.tier2x_mismatches(items, L.tests()["tier2"]) == []           # same target text, same human labels
    by_id = {it["id"]: it for it in items}
    for r in L.so762_test_records():          # the target equals the corpus text without punctuation but apostrophes
        assert by_id[r["id"].removeprefix("so762-")]["target"] == " ".join(re.sub(r"[^\w\s']", " ", r["text"]).split())
    t2 = {j.id: j for j in LL.benchmark_jobs(["en"], ["tier2"])}
    tx = {j.id: j for j in LL.tier2x_jobs() if j.id in t2}
    assert len(tx) == 500
    for i in list(t2)[:50]:                   # same request, same cache key (all 500: see the plan's cached count)
        assert LL.key_seconds(tx[i])[0] == LL.key_seconds(t2[i])[0]
