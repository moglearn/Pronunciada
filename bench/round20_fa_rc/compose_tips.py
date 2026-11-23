"""Coaching-tips table of the final per-language table, without scoring again.

    python -m bench.round20_fa_rc.compose_tips <frozen-run coaching_tips_r19.json> <languages kept on>

bench.coach.tips_r19 keeps no per-clip cache. The repository's table is main's; the natives entries of the
languages kept on are taken from the frozen held-out run's table (identical outputs: run_equiv_final.sh).
English is off, so the speechocean762 entry stays main's; the OMPAL entry (zh) is main's in both.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    rc = json.loads(Path(sys.argv[1]).read_text())
    keep = [x for x in sys.argv[2].split(",") if x]
    from bench.calib.common import load_json, save_json

    t = load_json("coaching_tips_r19.json")
    for lang in keep:
        t["natives_fleurs_test"][lang] = rc["natives_fleurs_test"][lang]
    assert "en" not in keep
    save_json("coaching_tips_r19.json", t)
    print("composed coaching_tips_r19.json; natives from the frozen run:", keep)


if __name__ == "__main__":
    main()
