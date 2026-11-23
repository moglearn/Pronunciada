"""Print the taxonomy tables of several languages side by side (reads tables/taxonomy_<lang>.json).

    python -m bench.round20_native_fa.summary --lang en,de,ko,es,fr [--keys driver_all,by_n_sounds]
"""
from __future__ import annotations

import argparse
import json

from bench.round20_native_fa.rebuild import ROOT


def load(lang):
    return json.loads((ROOT / "tables" / f"taxonomy_{lang}.json").read_text())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--keys", default="")
    a = ap.parse_args(argv)
    for lang in a.lang.split(","):
        d = load(lang)
        print(f"== {lang}: AUC {d['auc']:.4f} [{d['auc_ci'][0]:.4f}, {d['auc_ci'][1]:.4f}]  FA@80 {100 * d['fa80']:.2f} % "
              f"[{100 * d['fa80_ci'][0]:.2f}, {100 * d['fa80_ci'][1]:.2f}]  thr score {d['thr_score']:.3f}  "
              f"items {d['n_items']} speakers {d['n_speakers']} words {d['n_words']} changed {d['n_changed']}")
        for pop in ("FA", "low06"):
            x = d[pop]
            print(f"  {pop} n={x['n']}: " + "  ".join(f"{b} {v['pct']:.1f} [{v['ci'][0]:.1f},{v['ci'][1]:.1f}]"
                                                      for b, v in x["buckets"].items()))
            print("    overlap " + "  ".join(f"{t} {v:.1f}" for t, v in x["overlap"].items()))
            print("    driver", x["driver"], "low", x["low_kind"])
        print("  base   " + "  ".join(f"{t} {v:.1f}" for t, v in d["FA"]["base_rate"].items()))
        for k in [k for k in a.keys.split(",") if k]:
            print(f"  {k}: {json.dumps(d.get(k), ensure_ascii=False)}")


if __name__ == "__main__":
    main()
