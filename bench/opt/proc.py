"""Driver side of bench/opt/_child.py: one Scorer per fresh process, talked to over pipes."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .common import child_env

CHILD = Path(__file__).resolve().parent / "_child.py"


class Child:
    """A fresh process holding one Scorer for (spec, lang, threads). `launched` is the wall clock
    just before the process was started; `loaded` / `first` are the child's own time stamps."""

    def __init__(self, spec: dict, tree: Path, lang: str, threads: int, trace_files: bool = False,
                 exit_after_load: bool = False, precision: str | None = None):
        self.spec, self.lang, self.threads = spec, lang, threads
        args = ["--lang", lang, "--precision", precision or spec["precision"], "--threads", str(threads)]
        if trace_files:
            args.append("--trace-files")
        if exit_after_load:
            args.append("--exit-after-load")
        code = ("import runpy, sys; sys.argv = [sys.argv[0]] + sys.argv[1:]; "
                f"runpy.run_path({str(CHILD)!r}, run_name='__main__')")
        env = child_env(spec, tree, threads=threads)
        self.err = tempfile.TemporaryFile("w+")      # a pipe could fill up and block the child
        self.launched = time.time()
        self.p = subprocess.Popen([sys.executable, "-c", code, *args], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.err, text=True, env=env,
                                  cwd=str(tree), bufsize=1)
        self.info = self._read()
        if self.info.get("event") != "loaded":
            raise RuntimeError(f"child failed to load: {self.info}")
        src = Path(self.info["src"]).resolve()
        if Path(tree).resolve() not in src.parents:
            self.close()
            raise RuntimeError(f"pronunciada imported from {src}, not from {tree}")

    def _read(self) -> dict:
        while True:
            line = self.p.stdout.readline()
            if not line:
                self.p.wait(timeout=30)
                self.err.seek(0)
                err = self.err.read()[-3000:]
                raise RuntimeError(f"child exited ({self.p.poll()}): {err}")
            if line.startswith("@@"):
                return json.loads(line[2:])

    def ask(self, **cmd) -> dict:
        self.p.stdin.write(json.dumps(cmd) + "\n")
        self.p.stdin.flush()
        return self._read()

    def score(self, clip: dict, dump: str | None = None) -> dict:
        return self.ask(op="score", id=clip["id"], audio=clip["audio"], text=clip["text"], dump=dump)

    def close(self) -> dict:
        out = {}
        if self.p.poll() is None:
            try:
                out = self.ask(op="quit")
            except (RuntimeError, BrokenPipeError):
                pass
            try:
                self.p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.p.kill()
        return out


def load_only(spec: dict, tree: Path, lang: str, threads: int) -> dict:
    """Peak RSS and wall time of a fresh process that only builds the Scorer."""
    c = Child(spec, tree, lang, threads, exit_after_load=True)
    c.p.wait(timeout=120)
    return {"load_s": c.info["t"] - c.launched, "maxrss_mb": c.info["maxrss_mb"],
            "embeddings": c.info.get("embeddings"), "backbone": c.info.get("backbone")}
