"""Round 20 false-alarm release check, step 4: the numbers the pre-registered decision rule reads.

    PRONUNCIADA_CACHE=<overlay> python -m bench.round20_fa_rc.verdict <main results dir> <rc results dir> \
        <main Pronunciada test-1/2 answers dir> <rc Pronunciada test-1/2 answers dir> OUT.json

(a) the 84-row Azure tally: main's class and the release candidate's class of every row;
(b) Lingolix test 1, per language: release candidate - main, AUC and false alarms at 80 % recall, on the
    harness's 1,000 utterance resamples (bench.third_party.lingolix_live.paired_units, seed 0, two-sided p),
    benchmark rules of v2026.10 and of main; English test 2 and all 2,500 speechocean762 test clips: sentence
    and word correlations of both (from the comparison tables of lingolix_live);
(c) every scoreboard row that changed (bench/round19_merge/scoreboard_diff.py keys).
Only Pronunciada's own answers are compared here; no Lingolix answer is read.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

CLASS = {"ahead (significant)": 2, "ahead (not significant)": 1, "level": 1, "behind (not significant)": 1,
         "behind (significant)": 0}
NAME = {2: "ahead", 1: "level", 0: "behind"}
FIVE = ("en", "de", "es", "fr", "ko")


def _jsonl(p: Path) -> dict:
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def tally(main_dir: Path, rc_dir: Path) -> dict:
    A = json.loads((main_dir / "tables" / "scoreboard_tally.json").read_text())
    B = json.loads((rc_dir / "tables" / "scoreboard_tally.json").read_text())
    assert len(A) == len(B) == 84, (len(A), len(B))
    rows = []
    for a, b in zip(A, B):
        assert (a["area"], a["lang"], a["measure"]) == (b["area"], b["lang"], b["measure"]), (a, b)
        rows.append({"area": a["area"], "lang": a["lang"], "measure": a["measure"], "main": a["cat"], "rc": b["cat"],
                     "class_main": NAME[CLASS[a["cat"]]], "class_rc": NAME[CLASS[b["cat"]]],
                     "down": CLASS[b["cat"]] < CLASS[a["cat"]], "up": CLASS[b["cat"]] > CLASS[a["cat"]]})

    def count(key):
        c = {"ahead": 0, "level": 0, "behind": 0}
        for r in rows:
            c[r[key]] += 1
        return c

    return {"main": count("class_main"), "rc": count("class_rc"),
            "changed": [r for r in rows if r["main"] != r["rc"]], "down": [r for r in rows if r["down"]]}


def lingolix(main_ans: Path, rc_ans: Path) -> dict:
    from bench.third_party import lingolix as L
    from bench.third_party.lingolix_live import paired_units

    T = L.tests()
    unv = {}
    for lang in L.LANGS:
        for it in L.load_manifest("repo_main_f8750aa", f"tier1/manifest_{lang}.jsonl"):
            unv[it["id"]] = it.get("unverified", [])
    out = {}
    for lang in L.LANGS:
        test = f"tier1_{lang}"
        man = T[test]
        A, B = _jsonl(rc_ans / f"{test}.jsonl"), _jsonl(main_ans / f"{test}.jsonl")
        if not A or not B:
            out[lang] = {"missing": True}
            continue
        same = all(json.dumps(A[k], sort_keys=True) == json.dumps(B.get(k), sort_keys=True) for k in A) and \
            set(A) == set(B)
        row = {"answers_identical": same}
        for rules, kw in (("v2026.10", {}), ("main", {"exclude_unverified": unv})):
            ma, mb = L.metrics(man, A, **kw), L.metrics(man, B, **kw)
            r = {}
            for k, fn in (("roc_auc", L.roc_auc), ("fpr_at_recall_80", lambda l, s: L.fpr_at_recall(l, s, 0.80))):
                pu = paired_units(man, A, B, fn, **kw)
                r[k] = {"main": mb[k], "rc": ma[k], "delta": ma[k] - mb[k], **pu}
                worse = (r[k]["delta"] < 0) if k == "roc_auc" else (r[k]["delta"] > 0)
                r[k]["significantly_worse"] = bool(worse and pu["p"] is not None and pu["p"] < 0.05)
            row[rules] = r
        out[lang] = row
    t2 = {}
    A, B = _jsonl(rc_ans / "tier2.jsonl"), _jsonl(main_ans / "tier2.jsonl")
    man = T["tier2"]
    ma, mb = L.metrics(man, A), L.metrics(man, B)
    for k in ("pcc_word", "pcc_utterance"):
        t2[k] = {"main": mb.get(k), "rc": ma.get(k), "delta": (ma.get(k) or math.nan) - (mb.get(k) or math.nan)}
    t2["answers_identical"] = set(A) == set(B) and all(
        json.dumps(A[k], sort_keys=True) == json.dumps(B[k], sort_keys=True) for k in A)
    out["tier2"] = t2
    return out


def tier2x(main_dir: Path, rc_dir: Path) -> dict:
    A = json.loads((main_dir / "tables" / "lingolix_live.json").read_text())
    B = json.loads((rc_dir / "tables" / "lingolix_live.json").read_text())
    out = {}
    for row in ("a", "b"):
        for k in ("pcc_utterance", "pcc_word"):
            m = A["so762_more_speakers"]["rows"][row][k]["pronunciada"]["value"]
            r = B["so762_more_speakers"]["rows"][row][k]["pronunciada"]["value"]
            out[f"{row}:{k}"] = {"name": A["so762_more_speakers"]["rows"][row]["name"], "main": m, "rc": r,
                                 "delta": r - m}
    for k in ("pcc_word", "pcc_utterance"):
        out[f"tier2:{k}"] = {"main": A["tier2"][k]["pronunciada"]["value"], "rc": B["tier2"][k]["pronunciada"]["value"],
                             "delta": B["tier2"][k]["pronunciada"]["value"] - A["tier2"][k]["pronunciada"]["value"]}
    for lang in ("en", "fr", "es", "de", "ja", "ko"):
        for k in ("roc_auc", "fpr_at_recall_80"):
            a, b = A["tier1"][lang][k], B["tier1"][lang][k]
            out[f"tier1_{lang}:{k}"] = {"main": a["pronunciada"]["value"], "rc": b["pronunciada"]["value"],
                                        "vs_lingolix_main": [a["delta"], a["p"]], "vs_lingolix_rc": [b["delta"], b["p"]]}
    return out


def board(main_dir: Path, rc_dir: Path) -> list:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round19_merge"))
    from scoreboard_diff import rows

    old, new = rows(main_dir / "SCOREBOARD.md"), rows(rc_dir / "SCOREBOARD.md")
    out = []
    for k in list(old) + [k for k in new if k not in old]:
        a, b = old.get(k), new.get(k)
        if a != b:
            out.append({"section": k[0], "header": k[1], "row": k[2], "occurrence": k[4], "main": a, "rc": b})
    return out


def main() -> None:
    main_dir, rc_dir, main_ans, rc_ans, out = (Path(x) for x in sys.argv[1:6])
    res = {"tally": tally(main_dir, rc_dir), "lingolix_paired": lingolix(main_ans, rc_ans),
           "lingolix_tables": tier2x(main_dir, rc_dir), "scoreboard_rows_changed": board(main_dir, rc_dir)}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    t = res["tally"]
    print("tally main", t["main"], "rc", t["rc"])
    for r in t["changed"]:
        print("  tally row changed:", r["area"], r["lang"], r["measure"], r["main"], "->", r["rc"])
    for lang, r in res["lingolix_paired"].items():
        print(" ", lang, json.dumps(r)[:400])
    print(len(res["scoreboard_rows_changed"]), "scoreboard rows changed")


if __name__ == "__main__":
    main()
