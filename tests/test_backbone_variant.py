"""One runtime-built backbone variant serves every set of extra encoder outputs."""
import os
from pathlib import Path

import numpy as np
import pytest

from pronunciada import acoustic as A

MODEL = "zipa-small-crctc-500k"
UNION = "model.int8.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.onnx"


def _fake_cache(tmp_path, real=False):
    d = tmp_path / "models" / MODEL
    d.mkdir(parents=True)
    if real:
        src = Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada") / "models" / MODEL
        for f in ("model.int8.onnx", "tokens.txt"):
            (d / f).symlink_to(src / f)
    else:
        (d / "model.int8.onnx").write_bytes(b"x")
        (d / "tokens.txt").write_text("<blk> 0\n")
    return d


def test_every_output_set_maps_to_one_file(monkeypatch, tmp_path):
    monkeypatch.delenv("PRONUNCIADA_BACKBONE", raising=False)
    d = _fake_cache(tmp_path)
    calls = []

    def fake_write(src, dest, names):
        calls.append(list(names))
        Path(dest).write_bytes(b"y")

    monkeypatch.setattr(A, "write_with_graph_outputs", fake_write)
    paths = {A.embedding_model_path(MODEL, "int8", str(tmp_path), A.embed_outputs_for(sel))
             for sel in (True, "all", ["enc0"], ["enc2", "enc4"])}
    paths.add(A.embedding_model_path(MODEL, "int8", str(tmp_path)))
    assert paths == {str(d / UNION)}
    assert calls == [list(A.ALL_LAYER_OUTPUTS.values())]           # built once, with every output
    assert sorted(os.listdir(d)) == sorted(["model.int8.onnx", "tokens.txt", UNION])
    # a set outside the known stacks still gets a file of its own
    other = A.embedding_model_path(MODEL, "int8", str(tmp_path), {"x": "/some/other_output"})
    assert other == str(d / "model.int8.emb-x.onnx") and calls[-1] == ["/some/other_output"]


def test_stale_variants_listed_and_removed_on_request(tmp_path):
    d = _fake_cache(tmp_path)
    old = ["model.int8.emb.onnx", "model.emb.onnx", "model.emb-enc0-enc1-enc2-enc3-enc5-final.onnx",
           "model.fp16.emb-enc1.onnx"]
    keep = [UNION, "model.emb-enc0-enc1-enc2-enc3-enc4-enc5-final.onnx", "model.onnx", "notes.onnx"]
    for f in old + keep:
        (d / f).write_bytes(b"z")
    assert A.stale_model_variants(MODEL, str(tmp_path)) == sorted(str(d / f) for f in old)
    assert sorted(A.remove_stale_model_variants(MODEL, str(tmp_path))) == sorted(str(d / f) for f in old)
    assert sorted(os.listdir(d)) == sorted(keep + ["model.int8.onnx", "tokens.txt"])
    assert A.stale_model_variants(MODEL, str(tmp_path / "none")) == []


@pytest.mark.model
def test_union_file_outputs_bit_identical(monkeypatch, tmp_path):
    """Round-5 outputs, every-stack outputs and the plain model's log-probs, from the single file."""
    monkeypatch.delenv("PRONUNCIADA_BACKBONE", raising=False)
    d = _fake_cache(tmp_path, real=True)
    x = np.random.default_rng(1).standard_normal((420, 80)).astype(np.float32) - 5
    plain = A.AcousticModel(MODEL, "int8", threads=1, cache_dir=str(tmp_path))
    r5 = A.AcousticModel(MODEL, "int8", threads=1, cache_dir=str(tmp_path), embeddings=True)
    al = A.AcousticModel(MODEL, "int8", threads=1, cache_dir=str(tmp_path), embeddings="all")
    assert r5.embeddings and al.embeddings
    assert list(r5.embed_outputs) == list(A.EMBED_OUTPUTS) and list(al.embed_outputs) == list(A.ALL_LAYER_OUTPUTS)
    assert sorted(f for f in os.listdir(d) if ".emb" in f) == [UNION]
    lp0 = plain.posteriors_from_features(x)
    lp5, e5 = r5.posteriors_from_features(x, return_embeddings=True)
    lpa, ea = al.posteriors_from_features(x, return_embeddings=True)
    assert np.array_equal(lp0, lp5) and np.array_equal(lp0, lpa)
    assert set(e5) == set(A.EMBED_OUTPUTS)
    for k in e5:
        assert np.array_equal(e5[k], ea[k])
    # batching fetches the log-probs only, whatever the session carries
    assert np.array_equal(r5.posteriors_batch([np.zeros(16000, np.float32)])[0],
                          plain.posteriors_batch([np.zeros(16000, np.float32)])[0])
