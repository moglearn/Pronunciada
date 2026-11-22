"""Run one memory-heavy benchmark job at a time, and stop it before the machine runs out of memory.

    python -m bench.heavy [--need 35] [--floor 15] -- <command> [args...]

Every job that loads a large model (anything in torch, a 300 M+ ONNX model, several workers) goes
through this wrapper. It
  1. takes an exclusive lock ($PRONUNCIADA_CACHE/heavy.lock), so only one such job runs on the machine
     (--lane NAME is a second queue for a CPU-only job that fits beside the GPU one);
  2. waits until macOS reports at least --need percent of memory free before starting;
  3. checks free memory every 10 s while the job runs; under --floor percent it stops the job, waits
     for memory to recover and starts it again (the runners are resumable, finished records are kept).
Exit code = the job's. Prints one line per event with the time.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import signal
import subprocess
import sys
import time


def free_pct() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else 100


def say(msg: str) -> None:
    print(f"[heavy {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    except ProcessLookupError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--need", type=int, default=35, help="percent of memory free required to start")
    ap.add_argument("--floor", type=int, default=15, help="percent free under which the job is stopped")
    ap.add_argument("--max-restarts", type=int, default=6)
    ap.add_argument("--lane", default="", help="a second queue (e.g. 'cpu') that may run beside the main one")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    root = os.path.expanduser(os.environ.get("PRONUNCIADA_CACHE", "~/.cache/pronunciada"))
    lock = open(os.path.join(root, f"heavy{'-' + a.lane if a.lane else ''}.lock"), "w")
    say(f"waiting for the lock: {' '.join(cmd)[:120]}")
    fcntl.flock(lock, fcntl.LOCK_EX)
    restarts = 0
    while True:
        while (f := free_pct()) < a.need:
            say(f"memory free {f} % < {a.need} %, waiting")
            time.sleep(30)
        say(f"start (memory free {f} %)")
        p = subprocess.Popen(cmd, start_new_session=True)
        killed = False
        while p.poll() is None:
            time.sleep(10)
            f = free_pct()
            if f < a.floor and p.poll() is None:
                say(f"memory free {f} % < {a.floor} %: stopping the job")
                stop(p)
                killed = True
        if not killed:
            say(f"finished, exit code {p.returncode}")
            return p.returncode
        restarts += 1
        if restarts > a.max_restarts:
            say("too many restarts, giving up")
            return 75
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
