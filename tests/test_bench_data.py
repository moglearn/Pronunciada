import json
import os
from pathlib import Path

import numpy as np
import pytest

from bench.data import common, load
from bench.data.l2arctic import labels_from_textgrid, parse_textgrid

TG = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.0
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = ""
        intervals [2]:
            xmin = 0.1
            xmax = 1.0
            text = "this"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 4
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = "sil"
        intervals [2]:
            xmin = 0.1
            xmax = 0.4
            text = "DH,D,s"
        intervals [3]:
            xmin = 0.4
            xmax = 0.7
            text = "IH1"
        intervals [4]:
            xmin = 0.7
            xmax = 1.0
            text = "S,sil,d"
'''


def test_textgrid_and_l2arctic_labels():
    t = parse_textgrid(TG)
    assert [x[2] for x in t["words"]] == ["", "this"]
    lab = labels_from_textgrid(TG, "NJS")
    assert lab["l1"] == "Spanish" and lab["words"] == [{"text": "this", "start": 0.1, "end": 1.0}]
    assert [(p["canonical"], p["perceived"], p["error"], p["word"]) for p in lab["phones"]] == [
        ("DH", "D", "s", 0), ("IH1", "IH1", None, 0), ("S", "sil", "d", 0)]


def test_flac_roundtrip_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("PRONUNCIADA_CACHE", str(tmp_path))
    root = common.data_dir("toy")
    x = (0.1 * np.sin(np.arange(44100) / 10)).astype(np.float32)
    dur = common.write_flac(root / "a.flac", x, 44100)
    assert dur == pytest.approx(1.0, abs=1e-3)
    import soundfile as sf
    info = sf.info(str(root / "a.flac"))
    assert info.samplerate == 16000 and info.channels == 1 and info.format == "FLAC"
    pw = common.PartWriter(root, "test")
    rec = common.make_record("u1", root / "a.flac", "en", "hi", "test", None, dur, {"k": 1})
    pw.add(rec)
    pw.add(rec)  # idempotent
    pw.finish()
    assert common.PartWriter(root, "test").done
    common.merge_parts(root, ["test"])
    assert load("toy") == [rec] and load("toy", split="train") == []
    assert list(rec) == list(common.MANIFEST_KEYS)


REAL = ["speechocean762", "fleurs", "l2arctic", "epadb", "aishell3", "zhtw"]


@pytest.mark.parametrize("name", REAL)
def test_real_manifests(name):
    p = common.data_dir(name) / "manifest.jsonl"
    if not p.exists():
        pytest.skip(f"{name} not fetched")
    recs = load(name)
    assert recs and len({r["id"] for r in recs}) == len(recs)
    for r in recs[:: max(1, len(recs) // 50)]:
        assert list(r) == list(common.MANIFEST_KEYS)
        assert r["lang"] in common.LANGS and r["split"] in ("train", "test")
        assert os.path.isabs(r["audio"]) and Path(r["audio"]).exists() and r["dur"] > 0
    if name == "fleurs":
        assert all(r["split"] != "validation" and "/validation" not in r["audio"] for r in recs)
        assert max(r["dur"] for r in recs) <= 30.0
    if name == "speechocean762":
        assert len(load(name, "train")) == 2500 and len(load(name, "test")) == 2500
        w = recs[0]["labels"]["words"][0]
        assert len(w["phones"]) == len(w["phones_accuracy"])
    if name == "aishell3":
        tr = {r["speaker"] for r in load(name, "train")}
        te = {r["speaker"] for r in load(name, "test")}
        assert not (tr & te)
