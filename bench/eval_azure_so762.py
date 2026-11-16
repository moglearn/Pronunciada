"""Azure Pronunciation Assessment vs human labels on speechocean762 (from cached responses).

    python -m bench.eval_azure_so762 [--split test]

Prints Pearson correlations at sentence / word / phone level and writes per-item Azure scores to
~/.cache/pronunciada/azure/speechocean762/aligned_<split>.jsonl so our system can be compared on the
exact same items (paired tests).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from bench.data import load

ARPA = {"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d",
        "DH": "ð", "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "F": "f", "G": "g", "HH": "h", "IH": "ɪ", "IY": "i",
        "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ", "P": "p",
        "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
        "Y": "j", "Z": "z", "ZH": "ʒ"}
SAME = [{"ʌ", "ə"}, {"ɝ", "ɚ", "ɜ"}, {"g", "ɡ"}, {"ɔ", "ɑ"}, {"i", "ɪ"}, {"u", "ʊ"}]


def close(a: str, b: str) -> bool:
    return a == b or any(a in s and b in s for s in SAME)


def expand(azure_phones: list[dict]) -> list[tuple[str, float]]:
    """Azure merges vowel + r ("ɑɹ"); split so both halves carry the same score."""
    out = []
    for p in azure_phones:
        ph, sc = p["phoneme"], p["accuracy"]
        if len(ph) > 1 and ph.endswith("ɹ"):
            out += [(ph[:-1], sc), ("ɹ", sc)]
        else:
            out.append((ph, sc))
    return out


def align(ref: list[str], hyp: list[tuple[str, float]]) -> list[float | None]:
    """Score for each ref phone from the aligned Azure phone (None if unaligned)."""
    n, m = len(ref), len(hyp)
    if n == m:
        return [s for _, s in hyp]
    d = np.zeros((n + 1, m + 1))
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (0 if close(ref[i - 1], hyp[j - 1][0]) else 1))
    out: list[float | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + (0 if close(ref[i - 1], hyp[j - 1][0]) else 1):
            out[i - 1] = hyp[j - 1][1]
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def pcc(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    root = os.path.expanduser(os.path.join(os.environ.get("PRONUNCIADA_CACHE", "~/.cache/pronunciada"), "azure", "speechocean762"))
    flat = {}
    for line in open(os.path.join(root, "flat.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        flat[r["id"]] = r
    recs = load("speechocean762", a.split)
    sent = {k: ([], []) for k in ("accuracy", "fluency", "prosodic", "completeness", "total")}
    az_key = {"accuracy": "AccuracyScore", "fluency": "FluencyScore", "prosodic": "ProsodyScore",
              "completeness": "CompletenessScore", "total": "PronScore"}
    w_h, w_a, p_h, p_a = [], [], [], []
    n_ph = n_fallback = n_missing = n_fail = 0
    out_rows = []
    for rec in recs:
        f = flat.get(rec["id"])
        if f is None:
            n_missing += 1
            continue
        if f["status"] != "Success" or not f.get("words"):
            n_fail += 1
        s = f.get("sentence") or {}
        for k, (h, z) in sent.items():
            v = s.get(az_key[k])
            # a failed recognition is Azure's answer too: count it as the lowest score
            h.append(rec["labels"]["sentence"][k])
            z.append(0.0 if v is None else v)
        row = {"id": rec["id"], "sentence": {k: s.get(az_key[k]) for k in az_key}, "words": []}
        fw = f.get("words") or []
        for wi, w in enumerate(rec["labels"]["words"]):
            aw = fw[wi] if wi < len(fw) else None
            acc = aw["accuracy"] if aw and aw.get("accuracy") is not None else 0.0
            w_h.append(w["accuracy"])
            w_a.append(acc)
            ref = [ARPA["".join(c for c in p if c.isalpha()).upper()] for p in w["phones"]]
            sc = align(ref, expand(aw["phones"])) if aw and aw.get("phones") else [None] * len(ref)
            for h, v in zip(w["phones_accuracy"], sc):
                n_ph += 1
                if v is None:
                    n_fallback += 1
                    v = acc
                p_h.append(h)
                p_a.append(v)
            row["words"].append({"accuracy": acc, "phones": [acc if v is None else v for v in sc]})
        out_rows.append(row)
    with open(os.path.join(root, f"aligned_{a.split}.jsonl"), "w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"speechocean762 {a.split}: {len(recs)} utts, {n_missing} without an Azure response, {n_fail} where Azure returned no assessment")
    for k, (h, z) in sent.items():
        print(f"  sentence {k:13s} PCC = {pcc(h, z):.3f}  (Azure {az_key[k]})")
    print(f"  word accuracy          PCC = {pcc(w_h, w_a):.3f}  (n={len(w_h)})")
    print(f"  phone accuracy         PCC = {pcc(p_h, p_a):.3f}  (n={n_ph}, {n_fallback / max(n_ph, 1):.1%} fell back to the word score)")


if __name__ == "__main__":
    main()
