"""bench.eval_cv --half: the dev (tuning) half uses its own caches and never touches the test
report; the default stays the test half. No network, no scoring."""
import json

import pytest

from bench import cv_build as CB
from bench import eval_cv as EV


@pytest.fixture
def half(monkeypatch):
    def set_half(h):
        monkeypatch.setattr(EV, "HALF", h)
    yield set_half


def test_default_half_is_test(tmp_path, monkeypatch):
    monkeypatch.setenv("PRONUNCIADA_CACHE", str(tmp_path))
    assert EV.HALF == "test"
    assert EV.ours_root() == tmp_path / "cv-real" / "ours"
    assert EV.azure_dir() == tmp_path / "azure" / "cv-real"


def test_dev_half_has_separate_caches(tmp_path, monkeypatch, half):
    monkeypatch.setenv("PRONUNCIADA_CACHE", str(tmp_path))
    half("dev")
    assert EV.ours_root() == tmp_path / "cv-real-dev" / "ours"
    assert EV.azure_dir() == tmp_path / "azure" / "cv-real-dev"


def _manifest(tmp_path, monkeypatch):
    rows = [{"id": f"cv22-ja-{h}{i}", "locale": "ja", "lang": "ja", "test": "A", "half": h, "label": i % 2,
             "path": f"x{h}{i}.mp3", "client_id": f"{h}{i}", "sentence": "テスト", "n_chars": 3, "dur": 2.0}
            for h in ("test", "dev") for i in range(4)]
    d = tmp_path / "manifests"
    d.mkdir()
    (d / "ja.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(CB, "OUT_DIR", d)


def test_records_follow_half(tmp_path, monkeypatch, half):
    _manifest(tmp_path, monkeypatch)
    assert {r["half"] for r in EV.records(["ja"])} == {"test"}
    half("dev")
    assert {r["half"] for r in EV.records(["ja"])} == {"dev"}
    assert {r["half"] for r in EV.records(["ja"], half="test")} == {"test"}


def test_dev_report_writes_only_dev_table(tmp_path, monkeypatch, half):
    _manifest(tmp_path, monkeypatch)
    monkeypatch.setenv("PRONUNCIADA_CACHE", str(tmp_path / "cache"))
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    monkeypatch.setattr(EV, "REPO", repo)
    half("dev")
    d = EV.ours_root() / "fp"
    d.mkdir(parents=True)
    (EV.ours_root() / "latest.json").write_text(json.dumps({"fingerprint": "fp"}))
    for r in EV.records(["ja"]):
        (d / f"{r['id']}.json").write_text(json.dumps({"status": "Success", "prob": 0.9 - 0.5 * r["label"]}))
    EV.report(["ja"])
    assert (repo / "results" / "tables" / "cv_real_dev.json").exists()
    assert not (repo / "results" / "cv_real.md").exists()
    assert not (repo / "results" / "tables" / "cv_real.json").exists()
