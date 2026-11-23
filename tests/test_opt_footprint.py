"""bench/opt/profile.py footprint table and helpers (synthetic inputs, no model)."""
from bench.opt.profile import classify, footprint_table, paired_diff, summarize


def test_classify():
    assert classify("pronunciada/scorer.py") == "code"
    assert classify("pronunciada/data/adapters/en.npz") == "adapter"
    assert classify("pronunciada/data/lex/de.tsv.gz") == "lexicon"
    assert classify("pronunciada/data/calib/sound_B.npz") == "calibration"
    assert classify("pronunciada/data/coach/model.json") == "coaching"
    assert classify("models/model.int8.onnx") == "backbone"
    assert classify("models/model.int8.emb-enc0-enc1.onnx").startswith("backbone (runtime-built")
    assert classify("models/tokens.txt") == "backbone tokens"


def test_footprint_table_per_language():
    files = {"pronunciada/scorer.py": 100, "pronunciada/data/adapters/en.npz": 1000,
             "pronunciada/data/adapters/de.npz": 2000, "pronunciada/data/lex/de.tsv.gz": 500,
             "pronunciada/data/calib/config.json": 10, "pronunciada/data/coach/unused.npz": 7,
             "models/model.int8.onnx": 67_000, "models/model.onnx": 248_000, "models/tokens.txt": 1}
    used = {"en": {"pronunciada/data/adapters/en.npz", "pronunciada/data/calib/config.json", "models/tokens.txt"},
            "de": {"pronunciada/data/adapters/de.npz", "pronunciada/data/lex/de.tsv.gz",
                   "pronunciada/data/calib/config.json", "models/tokens.txt"}}
    t = footprint_table(files, used, backbone_for={"en": ["models/model.int8.onnx"], "de": ["models/model.int8.onnx"]},
                        third_party={"de": {"somepkg": 5}})
    en, de = t["smallest_install"]["en"], t["smallest_install"]["de"]
    assert en["code_bytes"] == de["code_bytes"] == 100
    assert en["data_bytes"] == 1000 + 10 + 1 + 67_000
    assert de["data_bytes"] == 2000 + 500 + 10 + 1 + 67_000
    assert de["total_bytes"] == 100 + de["data_bytes"] and de["third_party_bytes"] == 5
    assert en["by_category"]["backbone"] == 67_000 and "lexicon" not in en["by_category"]
    assert t["package_bytes"] == 100 + 1000 + 2000 + 500 + 10 + 7
    assert t["by_category"]["adapter"] == {"files": 2, "bytes": 3000}
    assert set(t["data_files_opened_by_no_language"]) == {"pronunciada/data/coach/unused.npz", "models/model.onnx"}
    shared = next(r for r in t["files"] if r["file"] == "pronunciada/data/calib/config.json")
    assert shared["used_by"] == ["de", "en"]


def test_summarize_and_paired_diff():
    s = summarize([1.0, 2.0, 3.0, None])
    assert s["n"] == 3 and s["p50"] == 2.0 and s["mean"] == 2.0
    a = {f"c{i}": [10.0, 10.0, 12.0] for i in range(30)}
    b = {f"c{i}": [11.0, 11.0, 13.0] for i in range(30)}
    d = paired_diff(a, b)
    assert d["n_clips"] == 30 and abs(d["mean_diff_ms"] - 1.0) < 1e-12
    assert d["mean_diff_ci"][0] <= 1.0 <= d["mean_diff_ci"][1]
