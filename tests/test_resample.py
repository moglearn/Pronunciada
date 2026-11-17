"""pronunciada._resample (numpy resample_poly, keeps scipy.signal out of the scoring path) and the
streamed embedding-model writer."""
import numpy as np
import pytest

from pronunciada._resample import resample_poly


@pytest.mark.parametrize("up,down", [(1, 2), (160, 441), (1, 3), (2, 3), (4, 1), (160, 147)])
@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-12), (np.float32, 1e-5)])
def test_resample_poly_matches_scipy(up, down, dtype, tol):
    ss = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(up * 1000 + down)
    for n in (1, 2, 7, 100, 1001, 16000 * 3 + 5):
        x = (0.3 * rng.standard_normal(n)).astype(dtype)
        a, b = ss.resample_poly(x, up, down), resample_poly(x, up, down)
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.max(np.abs(a.astype(np.float64) - b)) < tol


def test_resample_poly_identity_copies():
    x = np.arange(5, dtype=np.float32)
    y = resample_poly(x, 3, 3)
    assert y is not x and np.array_equal(x, y)


def test_scoring_path_does_not_import_scipy():
    import subprocess
    import sys

    code = ("import sys, numpy as np\n"
            "from pronunciada.prosody.f0 import track\n"
            "from pronunciada.audio import load_audio\n"
            "track(np.zeros(16000, np.float32)); load_audio((np.zeros(4410, np.float32), 44100))\n"
            "print(any(m == 'scipy' or m.startswith('scipy.') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_write_with_graph_outputs_matches_in_memory(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from pronunciada.acoustic import add_graph_outputs, write_with_graph_outputs

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    g = helper.make_graph([helper.make_node("Relu", ["x"], ["h"]), helper.make_node("Neg", ["h"], ["y"])],
                          "g", [x], [y])
    b = helper.make_model(g, producer_name="t").SerializeToString()
    (tmp_path / "a.onnx").write_bytes(b)
    write_with_graph_outputs(str(tmp_path / "a.onnx"), str(tmp_path / "b.onnx"), ["h"])
    assert (tmp_path / "b.onnx").read_bytes() == add_graph_outputs(b, ["h"])
    assert [o.name for o in onnx.load(str(tmp_path / "b.onnx")).graph.output] == ["y", "h"]
