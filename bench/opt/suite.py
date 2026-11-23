"""The held-out suite behind results/SCOREBOARD.md, run for one configuration in isolation.

    python -m bench.heavy --lane cpu --need 25 --floor 12 -- \
        python -m bench.opt.suite run --spec '{"name":"int8"}' --run-dir <dir> [--parallel 3] [--serial]
    python -m bench.opt.suite plan          # the steps, their inputs and outputs

The steps are those of the round-19 frozen run (bench/round19_merge/frozen_run.sh) that write a
table bench/scoreboard.py reads, with the same arguments and worker counts; left out: the Lingolix
benchmark (not part of this gate), the ZIPA-alone ablation (no row of its own reaches the Azure
tally; its Pronunciada values are recomputed from the other tables), the round-12b coaching table
(coaching_12b_heldout.json, not read by the scoreboard), the learner suite (not read by the
scoreboard) and eval_english's int8 / speed steps (not read by the scoreboard).

Isolation, as in the frozen runs:
  * code: a private copy of the configuration's source tree (<run>/src); the scripts write their
    tables into <run>/src/results/, never into a checkout. The tables the suite must write are
    deleted from the copy first, so a step that fails to write one cannot leave the committed
    version in place;
  * outputs: PRONUNCIADA_CACHE = <run>/cache, an overlay of the real cache
    (bench/round19_merge/overlay.py) whose output folders start empty, so every Pronunciada output
    is computed by the configuration under test; per-clip outputs of the coaching-tips and length
    scripts go to <run>/cache/opt_dumps (PRONUNCIADA_BENCH_DUMP);
  * Azure / Lingolix: every credential variable is removed from every step's environment; Azure
    numbers come from the cached responses only (no step of this suite sends a request);
  * the configuration's overrides (PRONUNCIADA_BACKBONE / _ADAPTER_DIR / _ORT_OPTIONS) are set in
    every step's environment; precision int8 = PRONUNCIADA_BACKBONE set to the int8 model file.

Scheduling: --serial runs the steps one after another in the frozen-run order. The default runs
independent steps side by side (at most --parallel at once, and a step starts only while macOS
reports at least --need % memory free); each step keeps its own worker count and thread settings,
so every per-clip computation is the same as in the serial order. Finished steps are marked in
<run>/logs/<step>.done; a restarted run (e.g. by bench.heavy after a memory stop) skips them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .common import cache_root, child_env, copy_tree, free_pct, load_spec, machine, materialize


@dataclass
class Step:
    name: str
    args: list[str]
    deps: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)      # results/tables/*.json written
    workers: int = 1                                       # worker processes it starts


STEPS = [
    Step("english", ["bench.eval_english", "--steps", "run,so762,calib,errdet", "--workers", "4"],
         outputs=["english_vs_azure.json"], workers=4),
    Step("languages", ["bench.eval_languages", "--steps", "run,eval,swap", "--workers", "3"],
         outputs=["languages.json", "swap_ours_vs_azure.json"], workers=3),
    Step("cv_real", ["bench.eval_cv", "--steps", "ours,report", "--workers", "3"], outputs=["cv_real.json"], workers=3),
    Step("ompal", ["bench.eval_ompal", "--steps", "ours,report", "--workers", "3"], outputs=["ompal.json"], workers=3),
    Step("coach_dump", ["bench.coach.dump", "--sets", "so762-test", "--workers", "2"], workers=2),
    Step("coach_dump_r4", ["bench.coach.dump_r4", "--sets", "so762-test,l2arctic-test,epadb-test,ompal-test",
                           "--workers", "3"], workers=3),
    Step("yardstick", ["bench.coach.azure_yardstick"], deps=["coach_dump"], outputs=["coaching_azure_yardstick.json"]),
    Step("fit_r4_eval", ["bench.coach.fit_r4", "eval"], deps=["coach_dump_r4"], outputs=["coaching_r4_heldout.json"]),
    Step("audit", ["bench.audit.contrast_audit", "run", "--split", "test", "--workers", "2"], workers=2),
    Step("audit_report", ["bench.audit.contrast_audit", "report", "--split", "test"], deps=["audit"],
         outputs=["contrast_audit_test.json"]),
    Step("length", ["bench.length.heldout_runtime", "--workers", "2"], outputs=["length_runtime_ja.json"], workers=2),
    Step("tips", ["bench.coach.tips_r19", "--workers", "2"], outputs=["coaching_tips_r19.json"], workers=2),
    Step("languages_report", ["bench.eval_languages", "--steps", "report"], deps=["languages", "english"]),
    Step("english_report", ["bench.eval_english", "--steps", "report"], deps=["english", "languages_report"]),
]
STEPS.append(Step("scoreboard", ["bench.scoreboard"], deps=[s.name for s in STEPS],
                  outputs=["scoreboard_tally.json"]))
BY_NAME = {s.name: s for s in STEPS}
FORBIDDEN_ARGS = ("azure", "lingolix")


def check_plan(steps=STEPS) -> None:
    for s in steps:
        low = " ".join(s.args).lower()
        if any(f in low.replace("azure_yardstick", "") for f in FORBIDDEN_ARGS):
            raise RuntimeError(f"step {s.name} would contact a rival service: {s.args}")
        for d in s.deps:
            if d not in BY_NAME:
                raise RuntimeError(f"step {s.name}: unknown dependency {d}")


def table_outputs() -> list[str]:
    return sorted({o for s in STEPS for o in s.outputs})


# ----------------------------------------------------------------------------- preparing a run

def prepare(spec: dict, run: Path) -> dict:
    """<run>/src (private tree copy, outputs deleted), <run>/cache (fresh overlay), <run>/logs."""
    from bench.round19_merge.overlay import build

    run.mkdir(parents=True, exist_ok=True)
    meta_p = run / "run.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if meta["spec"] != spec:
            raise SystemExit(f"{run} holds a run of another configuration: {meta['spec']}")
        return meta
    tree = materialize(spec["code"])
    src = copy_tree(tree, run / "src")
    for name in table_outputs():
        (src / "results" / "tables" / name).unlink(missing_ok=True)
    (src / "results" / "SCOREBOARD.md").unlink(missing_ok=True)
    build(cache_root(), run / "cache")
    (run / "logs").mkdir(exist_ok=True)
    meta = {"spec": spec, "tree": tree.name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": machine()}
    meta_p.write_text(json.dumps(meta, indent=1))
    return meta


def step_env(spec: dict, run: Path) -> dict:
    return child_env(spec, run / "src", suite=True,
                     extra={"PRONUNCIADA_CACHE": str(run / "cache"), "OMP_NUM_THREADS": "2",
                            "PRONUNCIADA_BENCH_DUMP": str(run / "cache" / "opt_dumps")})


# ----------------------------------------------------------------------------- running

def _launch(step: Step, spec: dict, run: Path):
    log = open(run / "logs" / f"{step.name}.log", "a")
    log.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} START load {os.getloadavg()[0]:.1f} "
              f"free {free_pct()} % {' '.join(step.args)}\n")
    log.flush()
    p = subprocess.Popen([sys.executable, "-m", *step.args], cwd=run / "src", env=step_env(spec, run),
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return p, log


def run_suite(spec: dict, run: Path, parallel: int = 3, need: int = 25, serial: bool = False,
              only: list[str] | None = None, log=print) -> dict:
    check_plan()
    meta = prepare(spec, run)
    steps = [s for s in STEPS if only is None or s.name in only]
    done = {s.name for s in STEPS if (run / "logs" / f"{s.name}.done").exists()}
    if only is not None:
        done |= {s.name for s in STEPS if s.name not in only}
    times_p = run / "logs" / "steps.tsv"
    t_suite = time.time()
    running: dict[str, tuple] = {}
    pending = [s for s in steps if s.name not in done]
    failed = None
    while pending or running:
        for name, (p, fh, t0, st) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            dt = time.time() - t0
            fh.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} END rc={rc} {dt:.0f} s\n")
            fh.close()
            with open(times_p, "a") as f:
                f.write(f"{name}\t{rc}\t{dt:.0f}\t{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))}\t"
                        f"{os.getloadavg()[0]:.1f}\n")
            del running[name]
            if rc == 0:
                missing = [o for o in st.outputs if not (run / "src" / "results" / "tables" / o).exists()]
                if missing:
                    failed = f"{name} finished without writing {missing}"
                else:
                    (run / "logs" / f"{name}.done").write_text(f"{dt:.0f}\n")
                    done.add(name)
                    log(f"[suite {time.strftime('%H:%M:%S')}] {name} done in {dt:.0f} s")
            else:
                failed = f"{name} failed (rc={rc}); see {run / 'logs' / (name + '.log')}"
        if failed:
            for p, fh, _, _ in running.values():
                p.terminate()
            raise SystemExit(failed)
        ready = [s for s in pending if all(d in done for d in s.deps)]
        cap = 1 if serial else parallel
        while ready and len(running) < cap:
            if running and free_pct() < need:
                break
            s = ready.pop(0)
            pending.remove(s)
            p, fh = _launch(s, spec, run)
            running[s.name] = (p, fh, time.time(), s)
            log(f"[suite {time.strftime('%H:%M:%S')}] start {s.name} ({s.workers} workers; running: "
                f"{', '.join(running)})")
            if serial:
                break
        time.sleep(5)
    total = time.time() - t_suite
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta.setdefault("wall_s", []).append(round(total))
    meta["mode"] = "serial" if serial else f"parallel {parallel}"
    (run / "run.json").write_text(json.dumps(meta, indent=1))
    log(f"[suite] all steps done in {total:.0f} s -> {run}")
    return meta


def step_times(run: Path) -> dict:
    p = run / "logs" / "steps.tsv"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            name, rc, dt, start, load = line.split("\t")
            if rc == "0":
                out[name] = {"s": int(dt), "start": start, "load_after": float(load)}
    return out


def time_language(spec: dict, run: Path, lang: str, donor: Path, log=print) -> dict:
    """Wall time of the per-language scoring steps for one language (for the full-run estimate):
    eval_languages run,eval,swap --lang L, contrast_audit run --lang L, eval_cv ours --locales L.
    The other languages' swap outputs are copied from `donor` (an earlier run's cache) so the swap
    step scores only L; nothing of this run is a result."""
    prepare(spec, run)
    api = run / "cache" / "calib" / "api"
    for p in sorted((donor / "calib" / "api").glob("fleurs-swap-*.pkl")):
        if p.stem.split("-")[-1] != lang and not (api / p.name).exists():
            shutil.copy2(p, api / p.name)
    locs = {"zh": "zh-CN,zh-TW"}.get(lang, lang)
    cmds = [("languages", ["bench.eval_languages", "--steps", "run,eval,swap", "--lang", lang, "--workers", "3"]),
            ("audit", ["bench.audit.contrast_audit", "run", "--split", "test", "--lang", lang, "--workers", "2"]),
            ("cv_real", ["bench.eval_cv", "--steps", "ours", "--locales", locs, "--workers", "3"])]
    out = {"lang": lang, "steps": {}, "machine": machine()}
    env = step_env(spec, run)
    for name, args in cmds:
        t0 = time.time()
        with open(run / "logs" / f"time_{name}.log", "w") as fh:
            rc = subprocess.run([sys.executable, "-m", *args], cwd=run / "src", env=env, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        out["steps"][name] = {"s": round(time.time() - t0, 1), "rc": rc, "load_after": os.getloadavg()[0]}
        log(f"[time {lang}] {name}: {out['steps'][name]}")
    (run / "time_language.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("time-language")
    t.add_argument("--spec", default=None)
    t.add_argument("--run-dir", required=True)
    t.add_argument("--lang", default="de")
    t.add_argument("--donor", required=True, help="cache overlay of an earlier run (swap outputs of other languages)")
    r = sp.add_parser("run")
    r.add_argument("--spec", default=None)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--parallel", type=int, default=3)
    r.add_argument("--need", type=int, default=25)
    r.add_argument("--serial", action="store_true")
    r.add_argument("--only", default=None, help="comma list of step names (the rest must be done already)")
    sp.add_parser("plan")
    a = ap.parse_args(argv)
    if a.cmd == "time-language":
        time_language(load_spec(a.spec), Path(a.run_dir).expanduser(), a.lang, Path(a.donor).expanduser())
        return
    if a.cmd == "plan":
        check_plan()
        for s in STEPS:
            print(f"{s.name:18s} deps={','.join(s.deps) or '-':40s} writes={','.join(s.outputs) or '(cache)'}  "
                  f"python -m {' '.join(s.args)}")
        return
    run_suite(load_spec(a.spec), Path(a.run_dir).expanduser(), a.parallel, a.need, a.serial,
              a.only.split(",") if a.only else None)


if __name__ == "__main__":
    main()
