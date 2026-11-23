"""bench/opt/suite.py scheduling with stand-in steps (no scoring): dependencies, parallel start, done
markers, missing outputs, and the step environment (overrides set, credentials removed)."""
import json
import sys

import pytest

from bench.opt import suite as S


def _fake(tmp_path, monkeypatch, steps):
    run = tmp_path / "run"

    def prepare(spec, run_):
        (run_ / "src" / "results" / "tables").mkdir(parents=True, exist_ok=True)
        (run_ / "cache").mkdir(exist_ok=True)
        (run_ / "logs").mkdir(exist_ok=True)
        return {"spec": spec}

    monkeypatch.setattr(S, "prepare", prepare)
    monkeypatch.setattr(S, "STEPS", steps)
    monkeypatch.setattr(S, "BY_NAME", {s.name: s for s in steps})
    monkeypatch.setattr(S, "free_pct", lambda: 90)
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    return run


def _writer(name, out, extra=""):
    # (the stand-in must not name the rival services itself: check_plan refuses such a step)
    code = (f"import json, os, time; open('results/tables/{out}', 'w').write(json.dumps("
            f"{{'step': '{name}', 't': time.time(), 'secrets': [k for k in os.environ if k.endswith(('_SPEECH_KEY', "
            f"'_API_KEY'))], 'env': {{k: v for k, v in os.environ.items() if k.startswith('PRONUNCIADA_')}}}})){extra}")
    return ["runpy_stub", "-c", code]


def test_dag_order_markers_and_env(tmp_path, monkeypatch):
    steps = [S.Step("a", [], outputs=["a.json"]), S.Step("b", [], outputs=["b.json"]),
             S.Step("c", [], deps=["a", "b"], outputs=["c.json"])]
    for s in steps:
        s.args = _writer(s.name, s.outputs[0])
    run = _fake(tmp_path, monkeypatch, steps)
    import subprocess

    def launch(step, spec, run_):        # S._launch runs `python -m <module>`; the stand-ins run `python -c`
        log = open(run_ / "logs" / f"{step.name}.log", "a")
        p = subprocess.Popen([sys.executable, *step.args[1:]], cwd=run_ / "src", env=S.step_env(spec, run_),
                             stdout=log, stderr=subprocess.STDOUT)
        return p, log

    monkeypatch.setattr(S, "_launch", launch)
    monkeypatch.setenv("AZURE_SPEECH_KEY", "x")
    monkeypatch.setenv("LINGOLIX_API_KEY", "y")
    spec = {"name": "t", "code": "HEAD", "precision": "int8", "adapter_dir": str(tmp_path)}
    S.run_suite(spec, run, parallel=2, log=lambda *a: None)
    T = {n: json.loads((run / "src" / "results" / "tables" / f"{n}.json").read_text()) for n in "abc"}
    assert T["c"]["t"] >= max(T["a"]["t"], T["b"]["t"])
    for n in "abc":
        assert (run / "logs" / f"{n}.done").exists()
        env = T[n]["env"]
        assert T[n]["secrets"] == []
        assert env["PRONUNCIADA_ADAPTER_DIR"] == str(tmp_path)
        assert env["PRONUNCIADA_BACKBONE"].endswith("model.int8.onnx")
        assert env["PRONUNCIADA_CACHE"] == str(run / "cache")
    # a second run skips every finished step
    (run / "src" / "results" / "tables" / "a.json").unlink()
    S.run_suite(spec, run, parallel=2, log=lambda *a: None)
    assert not (run / "src" / "results" / "tables" / "a.json").exists()


def test_missing_output_stops_the_run(tmp_path, monkeypatch):
    steps = [S.Step("a", ["x", "-c", "pass"], outputs=["a.json"])]
    run = _fake(tmp_path, monkeypatch, steps)
    import subprocess

    def launch(step, spec, run_):
        log = open(run_ / "logs" / f"{step.name}.log", "a")
        return subprocess.Popen([sys.executable, *step.args[1:]], cwd=run_ / "src", stdout=log,
                                stderr=subprocess.STDOUT), log

    monkeypatch.setattr(S, "_launch", launch)
    with pytest.raises(SystemExit, match="without writing"):
        S.run_suite({"name": "t", "code": "HEAD", "precision": "fp32"}, run, log=lambda *a: None)
    assert not (run / "logs" / "a.done").exists()


def test_real_plan_outputs_cover_the_scoreboard_inputs():
    names = set(S.table_outputs())
    for t in ("english_vs_azure.json", "swap_ours_vs_azure.json", "cv_real.json", "ompal.json", "languages.json",
              "contrast_audit_test.json", "length_runtime_ja.json", "coaching_tips_r19.json",
              "coaching_azure_yardstick.json", "coaching_r4_heldout.json"):
        assert t in names
