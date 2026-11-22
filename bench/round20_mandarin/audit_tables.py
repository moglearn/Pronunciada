"""Round 20: contrast-audit rows of the three arms (bench/round20_mandarin/audit_arms.sh), per-arm word hit@5 %
and paired arm-vs-arm differences (bench/round19_merge/audit_paired.py: clip bootstrap, 2,000 draws).

    python -m bench.round20_mandarin.audit_tables <work dir> <lang> <split> <out json> [arm ...]
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path


def main():
    work, lang, split, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    arms = sys.argv[5:] or ["off", "r19", "new"]
    sub = "contrast_v1_test" if split == "test" else "contrast_v1"
    R = {a: json.loads((work / f"report_{a}.json").read_text())["languages"][lang] for a in arms}
    rows = {}
    for c, v in R[arms[0]]["contrasts"].items():
        for name, d in v.get("directions", {}).items():
            code = "A>B" if name.startswith(f"said {v['members'][0]},") else "B>A"
            rows[f"{lang}:{c}:{code}"] = {"label": f"{c}: {name}", "n": d["n"],
                                          **{a: R[a]["contrasts"][c]["directions"][name]["word_hit5"] for a in arms}}
    from bench.round19_merge import audit_paired as AP

    pairs = [(arms[i], arms[j]) for i in range(len(arms)) for j in range(i + 1, len(arms))]
    for a, b in pairs:
        argv, env = sys.argv, os.environ.get("OUT")
        sys.argv = ["x", str(work / f"ov_{a}" / "audit" / sub), str(work / f"ov_{b}" / "audit" / sub), *rows]
        tmp = work / f"paired_{a}_{b}.json"
        os.environ["OUT"] = str(tmp)
        try:
            with redirect_stdout(io.StringIO()):
                AP.main()
        finally:
            sys.argv = argv
            if env is None:
                os.environ.pop("OUT", None)
        for r in json.loads(tmp.read_text()):
            rows[r["row"]][f"{b}-{a}"] = {"delta": r["delta"], "ci": r["ci"], "p": r["p"], "n_paired": r["n_edits_paired"]}
    for k, r in rows.items():
        print(f"{r['label']:55s} n {r['n']:3d} " + " ".join(f"{a} {r[a]:.3f}" for a in arms) + " | " +
              " ".join(f"{p} {x['delta']:+.3f} p={x['p']:.2f}" for p, x in r.items() if isinstance(x, dict)))
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
