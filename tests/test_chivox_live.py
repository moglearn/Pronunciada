"""bench.third_party.chivox_live (+ live_core): cap, --yes, cache, ledger, refusals, parsing, secrets.

No test sends anything over the network: every request goes to a local fake transport, and the
datasets are replaced by synthetic recordings."""
import json
import socket
import struct
import wave
from pathlib import Path

import pytest

pytest.importorskip("soundfile")

from bench.third_party import chivox_live as CX  # noqa: E402
from bench.third_party import live_core as C  # noqa: E402

SECRET = "cvx_TEST_SECRET_9b1f3c55aa"


def _wav(p: Path, seconds=0.5, tone=1) -> Path:
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", (i * tone) % 300) for i in range(int(16000 * seconds))))
    return p


def _items(tmp_path, n=3, lang="en"):
    ds = "speechocean762" if lang == "en" else "ompal"
    text = (lambda i: f"HELLO WORLD NUMBER {i}") if lang == "en" else (lambda i: "他們都很忙"[: 3 + i % 3])
    return [C.Item(ds, f"{ds}-{i:04d}", f"spk{i % 2}", _wav(tmp_path / f"{ds}{i}.wav", tone=i + 1), text(i), lang)
            for i in range(n)]


def _run(items, root, transport, cap=100.0, key=SECRET, used=0.0):
    ledger = C.Ledger(root / C.LEDGER)
    p = C.plan(CX, items, root, used=used, cap=None)
    budget = C.Budget(cap, used, ledger, CX.UNIT, CX.NAME)
    st = C.run(CX, p["todo"], transport=transport, key=key, root=root, budget=budget,
               policy=C.Policy(rps=0, base_s=0), sleep=lambda s: None, progress=False)
    return st, ledger


def _tool_calls(t):
    return [c for c in t.calls if (c["body"] or {}).get("method") == "tools/call"]


# ── cap and consent ──

def test_cap_refuses_before_any_network_call(tmp_path):
    t = CX.FakeTransport()
    st, ledger = _run(_items(tmp_path, 3), tmp_path / "root", t, cap=2)
    assert len(_tool_calls(t)) == 2                         # the third was refused before being sent
    assert st["answered"] == 2 and "cap reached" in st["stopped"]
    assert ledger.spent() == 2


def test_retries_reserve_against_the_cap(tmp_path):
    t = CX.FakeTransport(script={1: C.Response(503, {}, b"busy")})
    st, ledger = _run(_items(tmp_path, 1), tmp_path / "root", t, cap=1)
    assert len(_tool_calls(t)) == 1                         # the retry would cross the cap: not sent
    assert "cap reached" in st["stopped"]
    assert ledger.spent() == 1


def test_cap_counts_points_already_in_the_ledger(tmp_path):
    t = CX.FakeTransport()
    st, _ = _run(_items(tmp_path, 3), tmp_path / "root", t, cap=600, used=599)
    assert len(_tool_calls(t)) == 1


def test_nothing_sent_without_yes_or_cap(tmp_path, monkeypatch, capsys):
    items = _items(tmp_path, 2)
    monkeypatch.setattr(C, "select_items", lambda *a, **k: items)
    monkeypatch.setattr(C, "live_root", lambda v, dry, base=None: tmp_path / ("dry" if dry else "live"))
    monkeypatch.setattr(CX, "real_transport", lambda: pytest.fail("a transport was built"))
    monkeypatch.setenv(CX.KEY_ENV, SECRET)
    assert CX.main(["--datasets", "speechocean762", "--points-cap", "5"]) == 2          # no --yes
    assert CX.main(["--datasets", "speechocean762", "--yes"]) == 2                      # no cap
    monkeypatch.delenv(CX.KEY_ENV)
    assert CX.main(["--datasets", "speechocean762", "--yes", "--points-cap", "5"]) == 2  # no key
    assert CX.main(["--datasets", "speechocean762", "--plan"]) == 0
    err = capsys.readouterr().err
    assert "nothing sent" in err and SECRET not in err
    assert not (tmp_path / "live" / C.LEDGER).exists()


def test_refuses_other_datasets(tmp_path):
    for bad in ("l2arctic", "epadb", "saa", "leap", "common_voice", "heroico", "lingolix_test2"):
        with pytest.raises(C.DatasetRefused):
            C.check_datasets([bad], CX.ALLOWED)
        with pytest.raises(C.DatasetRefused):
            C.load_items(bad, CX.ALLOWED)
    assert CX.main(["--datasets", "epadb", "--plan"]) == 2


# ── cache and ledger ──

def test_cache_hit_means_no_request(tmp_path):
    items, root = _items(tmp_path, 3), tmp_path / "root"
    t1 = CX.FakeTransport()
    _run(items, root, t1)
    assert len(_tool_calls(t1)) == 3
    p = C.plan(CX, items, root, used=0, cap=None)
    assert p["n_send"] == 0 and p["rows"]["speechocean762"]["answered"] == 3
    t2 = CX.FakeTransport()
    st, _ = _run(items, root, t2)
    assert t2.calls == [] and st["sent"] == 0


def test_cached_answer_written_at_once_with_version(tmp_path):
    items, root = _items(tmp_path, 1), tmp_path / "root"
    _run(items, root, CX.FakeTransport())
    files = list((root / "responses").rglob("*.json"))
    assert len(files) == 1
    assert files[0].name.endswith("__speechocean762-0000__FAKE-chivox-dry-run.json")
    rec = json.loads(files[0].read_text())
    assert rec["final"] == "answered" and rec["version"] == "FAKE-chivox-dry-run" and rec["status"] == 200


def test_ledger_arithmetic(tmp_path):
    t = CX.FakeTransport(script={2: C.Response(500, {}, b"oops")})
    st, ledger = _run(_items(tmp_path, 3), tmp_path / "root", t)
    E = ledger.entries()
    att = [e for e in E if e["kind"] == "attempt"]
    res = [e for e in E if e["kind"] == "result"]
    assert len(att) == len(res) == 4                       # 3 items + 1 retry
    assert ledger.spent() == 4 == st["booked"]
    assert [r["status"] for r in res].count(500) == 1
    assert all(a["unit"] == "point" and a["cost"] == 1 and a["id"] and a["utc"] for a in att)
    assert {e["kind"] for e in E} >= {"session", "session_result"}       # initialize logged at 0 cost
    assert sum(e.get("cost") or 0 for e in E if e["kind"] == "session") == 0


def test_unreadable_ledger_fails_closed(tmp_path):
    p = tmp_path / C.LEDGER
    p.write_text('{"kind": "attempt", "cost": 1}\nnot json\n')
    with pytest.raises(C.Fatal, match="cannot be read"):
        C.Ledger(p).spent()


# ── stopping on errors ──

@pytest.mark.parametrize("resp", [
    C.Response(401, {}, b"{}"), C.Response(403, {}, b'{"error": "quota exhausted"}'),
    C.Response(429, {}, b'{"error": "key cap reached"}'),
    C.Response(200, {"Content-Type": "application/json"},
               json.dumps({"jsonrpc": "2.0", "id": 2, "error": {"code": -32000,
                                                                "message": "Evaluation failed: insufficient points"}}).encode()),
])
def test_quota_billing_auth_errors_stop_at_once(tmp_path, resp):
    t = CX.FakeTransport(script={1: resp})
    st, ledger = _run(_items(tmp_path, 3), tmp_path / "root", t)
    assert len(_tool_calls(t)) == 1 and st["stopped"]
    assert ledger.spent() == 1


def test_transient_errors_are_retried_and_bounded(tmp_path):
    t = CX.FakeTransport(script={k: C.Response(503, {}, b"") for k in range(1, 20)})
    st, _ = _run(_items(tmp_path, 1), tmp_path / "root", t)
    assert len(_tool_calls(t)) == C.Policy().max_attempts and st["failed"] == 1


def test_session_expiry_reinitializes(tmp_path):
    t = CX.FakeTransport(script={1: C.Response(404, {}, b"session not found")})
    st, _ = _run(_items(tmp_path, 1), tmp_path / "root", t)
    assert st["answered"] == 1
    assert sum((c["body"] or {}).get("method") == "initialize" for c in t.calls) == 2


def test_rejected_request_is_cached_and_not_resent(tmp_path):
    items, root = _items(tmp_path, 1), tmp_path / "root"
    bad = json.dumps({"jsonrpc": "2.0", "id": 2, "error": {"code": -32000, "message": "Evaluation failed: bad ref_text"}})
    _run(items, root, CX.FakeTransport(script={1: C.Response(200, {}, bad.encode())}))
    p = C.plan(CX, items, root, used=0, cap=None)
    assert p["n_send"] == 0 and p["rows"]["speechocean762"]["rejected"] == 1
    assert C.plan(CX, items, root, used=0, cap=None, refresh_rejected=True)["n_send"] == 1


# ── secrets ──

def test_credential_never_written_or_printed(tmp_path, capsys):
    root = tmp_path / "root"
    leak = json.dumps({"jsonrpc": "2.0", "id": 2, "error": {"code": -1, "message": f"bad ref for key {SECRET}"}})
    t = CX.FakeTransport(script={1: C.Response(200, {"Authorization": f"Bearer {SECRET}"}, leak.encode())})
    _run(_items(tmp_path, 2), root, t)
    assert all(c["headers"]["Authorization"] == f"Bearer {SECRET}" for c in t.calls)   # it is sent...
    for p in root.rglob("*"):
        if p.is_file():
            assert SECRET not in p.read_text(encoding="utf-8"), p                         # ...never stored
    out = capsys.readouterr()
    assert SECRET not in out.err + out.out


# ── request and response format ──

def test_request_body_follows_the_docs(tmp_path):
    t = CX.FakeTransport()
    _run(_items(tmp_path, 1) + _items(tmp_path, 1, "zh"), tmp_path / "root", t)
    init = t.calls[0]["body"]
    assert init["method"] == "initialize" and t.calls[0]["url"] == "https://mcp-global.cloud.chivox.com/"
    en, zh = (c["body"]["params"] for c in _tool_calls(t))
    assert en["name"] == "en_sentence_eval" and en["arguments"]["accent"] == 3 and en["arguments"]["rank"] == 100
    assert zh["name"] == "cn_sentence_eval" and zh["arguments"]["age_group"] == "adult"
    assert set(en["arguments"]) == {"ref_text", "audio_base64", "accent", "rank"}
    assert all(c["headers"].get("Mcp-Session-Id") == "fake-session-1" for c in _tool_calls(t))


# README / docs "What the engine returns" example (hello, four phones)
DOC_EXAMPLE = {"overall": 85, "accuracy": 82, "pron": 88, "fluency": {"overall": 78, "speed": 65, "pause": 2},
               "integrity": 95, "details": [{"char": "hello", "score": 85, "phone": [
                   {"phoneme": "h", "score": 90, "dp_type": "normal"}, {"phoneme": "ɛ", "score": 82, "dp_type": "normal"},
                   {"phoneme": "l", "score": 88, "dp_type": "normal"}, {"phoneme": "oʊ", "score": 80, "dp_type": "normal"}]}]}
# /products/mandarin-chinese-assessment example shape
DOC_ZH = {"overall": 78, "pron": 74, "fluency": {"overall": 81, "speed": 72},
          "details": [{"char": "你", "score": 70, "tone_ref": "T3", "tone_detected": "T2", "dp_type": "normal"},
                      {"char": "好", "score": 90, "tone_ref": "T3", "tone_detected": "T3"}]}


def _rec(ev, sse=False):
    m = {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": json.dumps(ev, ensure_ascii=False)}]}}
    if sse:
        return {"status": 200, "headers": {"Content-Type": "text/event-stream"},
                "body": f"event: message\ndata: {json.dumps(m, ensure_ascii=False)}\n\n"}
    return {"status": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps(m, ensure_ascii=False)}


@pytest.mark.parametrize("sse", [False, True])
def test_documented_example_maps_to_the_fixed_fields(sse):
    it = C.Item("speechocean762", "x", "s", Path("x.wav"), "HELLO", "en")
    v = CX.view(_rec(DOC_EXAMPLE, sse), it)
    assert v["sentence"] == {"total": 85, "accuracy": 82, "fluency": 78, "prosodic": None, "completeness": 95,
                             "secondary": 88}
    assert v["words"] == [{"text": "hello", "score": 85.0,
                           "phones": [("h", 90.0), ("ɛ", 82.0), ("l", 88.0), ("oʊ", 80.0)]}]
    r = C.Response(200, _rec(DOC_EXAMPLE, sse)["headers"], _rec(DOC_EXAMPLE, sse)["body"].encode())
    assert CX.classify(r, {"last_id": 2}) == ("answered", "ok")


def test_quickstart_nesting_under_result_is_accepted():
    it = C.Item("speechocean762", "x", "s", Path("x.wav"), "HELLO", "en")
    assert CX.view(_rec({"result": DOC_EXAMPLE}), it)["sentence"]["total"] == 85


def test_mandarin_units_and_tone_channel():
    it = C.Item("ompal", "z", "s", Path("z.wav"), "你好。", "zh")
    v = CX.view(_rec(DOC_ZH), it)
    assert v["zh_units"] == [(0, 1, 70.0, 0.0, None), (1, 2, 90.0, 100.0, None)]
    assert v["sentence"]["secondary"] == 78 and v["sentence"]["accuracy"] is None


@pytest.mark.parametrize("mutate, assumption", [
    (lambda e: e.__setitem__("overall", 850), "C3"),
    (lambda e: e.pop("details"), "C4"),
    (lambda e: e["details"][0].__setitem__("score", "high"), "C4"),
    (lambda e: e["details"][0]["phone"][0].__setitem__("score", -3), "C5"),
    (lambda e: e["details"][0].__setitem__("char", "zzz"), "C4"),
])
def test_unexpected_format_fails_loudly(mutate, assumption):
    ev = json.loads(json.dumps(DOC_EXAMPLE))
    mutate(ev)
    with pytest.raises(C.ResponseFormatError, match=assumption):
        CX.view(_rec(ev), C.Item("speechocean762", "x", "s", Path("x.wav"), "HELLO", "en"))


def test_format_error_stops_run_after_caching(tmp_path):
    bad = dict(DOC_EXAMPLE, overall=850)
    body = json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": json.dumps(bad)}]}})
    root = tmp_path / "root"
    st, _ = _run(_items(tmp_path, 2), root, CX.FakeTransport(script={1: C.Response(200, {}, body.encode())}))
    assert "C3" in st["stopped"] and len(list((root / "responses").rglob("*.json"))) == 1


# ── pilot sample and dry-run guard ──

def test_pilot_order_is_seeded_speaker_balanced_and_nested(tmp_path):
    its = [C.Item("speechocean762", f"u{s}{k}", f"s{s}", Path("a"), "x", "en") for s in range(5) for k in range(4)]
    o = C.pilot_order(its, 0)
    assert len({i.speaker for i in o[:5]}) == 5                       # one clip per speaker first
    assert [i.id for i in C.pilot_order(its, 0)] == [i.id for i in o]  # deterministic
    assert [i.id for i in C.pilot_order(its, 1)] != [i.id for i in o]
    assert sorted(i.id for i in o) == sorted(i.id for i in its)


def test_default_pilot_fits_the_free_points():
    assert sum(CX.PILOT.values()) == 600 and CX.cost(3.0) == 1


def test_dry_run_guard_blocks_the_network():
    with C.no_network_guard():
        with pytest.raises(Exception, match="refused by the guard"):
            socket.create_connection(("mcp-global.cloud.chivox.com", 443), timeout=1)


def test_real_transport_only_talks_to_the_documented_host():
    with pytest.raises(C.Fatal, match="refusing"):
        C.RequestsTransport((CX.HOST,)).post("https://example.org/", headers={}, json_body={})


# ── evaluation on synthetic speechocean762 rows ──

def test_so762_rows_line_up_with_the_scoreboard_rows():
    import numpy as np
    from bench.third_party import live_eval as LE

    rng = np.random.default_rng(0)
    human = {}
    P = {k: [] for k in ("y", "spk", "az", "B", "key")}
    Wd = {k: [] for k in ("y", "spk", "az", "ours", "key")}
    S = {k: [] for k in ("accuracy", "fluency", "prosodic", "total", "spk", "key")}
    S.update({f"az_{k}": [] for k in ("accuracy", "fluency", "prosodic", "total")},
             ours={k: [] for k in ("accuracy", "fluency", "prosodic", "total")})
    views = {}
    for u in range(12):
        uid, spk = f"so762-{u:04d}", f"s{u % 4}"
        acc = [float(rng.integers(3, 11)) for _ in range(2)]
        human[uid] = [{"text": "HELLO", "phones": ["HH", "AH0", "L", "OW1"], "accuracy": acc[0]},
                      {"text": "WORLD", "phones": ["W", "ER1", "L", "D"], "accuracy": acc[1]}]
        views[uid] = {"sentence": {"accuracy": 10 * sum(acc), "total": 9 * sum(acc), "fluency": None, "prosodic": None},
                      "words": [{"text": "hello", "score": 10 * acc[0],
                                 "phones": [("h", 1.0), ("ə", 2.0), ("l", 3.0), ("oʊ", 4.0)]},
                                {"text": "world", "score": 10 * acc[1], "phones": []}]}
        for j, a in enumerate(acc):
            for k, v in (("y", a), ("spk", spk), ("az", a + rng.normal()), ("key", (uid, j)), ("ours", a + rng.normal())):
                Wd[k].append(v)
            for k in range(4):
                for name, v in (("y", a / 5), ("spk", spk), ("az", rng.normal()), ("B", rng.normal()), ("key", (uid, j, k))):
                    P[name].append(v)
        for k in ("accuracy", "fluency", "prosodic", "total"):
            S[k].append(sum(acc))
            S[f"az_{k}"].append(rng.normal())
            S["ours"][k].append(rng.normal())
        S["spk"].append(spk)
        S["key"].append(uid)
    answered = dict(list(views.items())[:10])
    R = LE.eval_so762(answered, ref=(P, Wd, S, 0), human=human, n_boot=30, vendor="chivox")
    assert R["coverage"] == {"words": 20, "words_unmatched": 0, "phones": 80, "phones_word_fallback": 40}
    assert R["words_pearson"]["n"] == 20 and R["words_pearson"]["chivox"]["value"] == pytest.approx(1.0)
    assert R["sounds_pearson"]["n"] == 80
    assert R["sentence_accuracy_pearson"]["n"] == 10
    assert R["sentence_accuracy_pearson"]["chivox"]["value"] == pytest.approx(1.0)
    assert not R["sentence_fluency_pearson"]["vendor_has_score"] and "chivox" not in R["sentence_fluency_pearson"]
    assert R["low_word_auc"]["chivox"]["value"] == pytest.approx(1.0)
    assert set(R["words_pearson"]["delta"]) == {"value", "ci", "p", "of"}
