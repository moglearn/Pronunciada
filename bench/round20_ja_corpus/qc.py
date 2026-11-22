"""Quality checks of the Japanese length corpus.

    python -m bench.round20_ja_corpus.qc durations   # short/long and single/geminate duration separation
    python -m bench.round20_ja_corpus.qc anchors --n 300   # aligner vowel midpoints vs the runtime aligner's
    python -m bench.round20_ja_corpus.qc xcheck --n 400    # independent energy / autocorrelation measure
    python -m bench.round20_ja_corpus.qc counts            # hours, speakers, contrast counts

Results go to <work>/qc/*.json; the small tables are copied to results/tables/round20_ja_corpus_*.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bench.round20_ja_corpus.common import CORPUS, WORK

REPO = Path(__file__).resolve().parents[2]
TABLES = REPO / "results" / "tables"
STOPS = {"k", "t", "p", "tʃ", "ts", "s", "ʃ", "kj", "pj", "tj", "h", "f"}
VOICELESS = {"k", "t", "p", "tʃ", "ts", "s", "ʃ", "kj", "pj", "tj", "h", "hj", "f", "Q"}
SPLITS = ("train", "validation", "heldback")


def rows(split: str | None = None, corpus: Path = CORPUS):
    for s in ([split] if split else SPLITS):
        p = corpus / f"{s}.jsonl"
        if p.exists():
            yield from map(json.loads, open(p, encoding="utf-8"))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))     # average ranks for ties
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def overlap(a: np.ndarray, b: np.ndarray, bins=np.arange(0, 0.5, 0.005)) -> float:
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    return float(np.minimum(ha, hb).sum() * (bins[1] - bins[0]))


def _q(x) -> dict:
    x = np.asarray(x)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean_ms": round(float(x.mean() * 1000), 1),
            **{f"p{q}_ms": round(float(np.percentile(x, q) * 1000), 1) for q in (10, 50, 90)}}


def tokens(corpus: Path = CORPUS):
    """(utt, seg index, seg, duration, rate) for vowels and consonants."""
    for r in rows(corpus=corpus):
        segs = r["segments"]
        for i, s in enumerate(segs):
            yield r, i, s


def durations(corpus: Path = CORPUS) -> dict:
    V = defaultdict(lambda: {"short": [], "long": []})
    C = defaultdict(lambda: {"single": [], "gem": []})
    Vr = defaultdict(lambda: {"short": [], "long": []})
    Cr = defaultdict(lambda: {"single": [], "gem": []})
    rate_bins = [0, 6, 7, 8, 9, 100]
    for r in rows(corpus=corpus):
        segs = r["segments"]
        rb = int(np.digitize(r["rate"], rate_bins)) - 1
        for i, s in enumerate(segs):
            d = s["t1"] - s["t0"]
            if s["k"] == "V" and not s.get("dv") and not s["pf"]:
                key = "long" if s["long"] else "short"
                V[s["s"][0]][key].append(d)
                Vr[rb][key].append(d)
            if s["k"] == "C" and s["s"] in STOPS and s["mora"] is not None:
                if s.get("gem"):
                    q = segs[i - 1]
                    dd = s["t1"] - q["t0"]       # closure (Q) + release
                    C[s["s"]]["gem"].append(dd)
                    Cr[rb]["gem"].append(dd)
                elif i > 0 and segs[i - 1]["k"] in ("V", "N") and segs[i - 1]["next"] == s["s"]:
                    C[s["s"]]["single"].append(d)
                    Cr[rb]["single"].append(d)
    out = {"note": "vowels: not devoiced, not phrase-final; consonants: word-medial after a vowel/N, "
                   "geminate = Q closure + release; rate bins in morae per second of speech",
           "rate_bins": rate_bins, "vowel": {}, "vowel_by_rate": {}, "consonant": {}, "consonant_by_rate": {}}
    allv = {"short": [], "long": []}
    for v, d in sorted(V.items()):
        out["vowel"][v] = {"short": _q(d["short"]), "long": _q(d["long"]),
                           "auc": round(auc(np.array(d["long"]), np.array(d["short"])), 4),
                           "overlap": round(overlap(np.array(d["long"]), np.array(d["short"])), 4)}
        allv["short"] += d["short"]
        allv["long"] += d["long"]
    out["vowel"]["all"] = {"short": _q(allv["short"]), "long": _q(allv["long"]),
                           "auc": round(auc(np.array(allv["long"]), np.array(allv["short"])), 4),
                           "overlap": round(overlap(np.array(allv["long"]), np.array(allv["short"])), 4)}
    for b, d in sorted(Vr.items()):
        out["vowel_by_rate"][str(b)] = {"short": _q(d["short"]), "long": _q(d["long"]),
                                        "auc": round(auc(np.array(d["long"]), np.array(d["short"])), 4)}
    allc = {"single": [], "gem": []}
    for c, d in sorted(C.items()):
        out["consonant"][c] = {"single": _q(d["single"]), "gem": _q(d["gem"]),
                               "auc": round(auc(np.array(d["gem"]), np.array(d["single"])), 4)}
        allc["single"] += d["single"]
        allc["gem"] += d["gem"]
    out["consonant"]["all"] = {"single": _q(allc["single"]), "gem": _q(allc["gem"]),
                               "auc": round(auc(np.array(allc["gem"]), np.array(allc["single"])), 4),
                               "overlap": round(overlap(np.array(allc["gem"]), np.array(allc["single"])), 4)}
    for b, d in sorted(Cr.items()):
        out["consonant_by_rate"][str(b)] = {"single": _q(d["single"]), "gem": _q(d["gem"]),
                                            "auc": round(auc(np.array(d["gem"]), np.array(d["single"])), 4)}
    # rate-normalised duration (duration x local rate) as a second baseline
    vn = {"short": [], "long": []}
    cn = {"single": [], "gem": []}
    for r in rows(corpus=corpus):
        for i, s in enumerate(r["segments"]):
            d = (s["t1"] - s["t0"]) * r["rate"]
            if s["k"] == "V" and not s.get("dv") and not s["pf"]:
                vn["long" if s["long"] else "short"].append(d)
            if s["k"] == "C" and s["s"] in STOPS and s.get("gem"):
                cn["gem"].append((s["t1"] - r["segments"][i - 1]["t0"]) * r["rate"])
            elif s["k"] == "C" and s["s"] in STOPS and i > 0 and r["segments"][i - 1]["k"] in ("V", "N") \
                    and r["segments"][i - 1]["next"] == s["s"]:
                cn["single"].append(d)
    out["rate_normalised_auc"] = {"vowel": round(auc(np.array(vn["long"]), np.array(vn["short"])), 4),
                                  "consonant": round(auc(np.array(cn["gem"]), np.array(cn["single"])), 4)}
    return out


def counts(corpus: Path = CORPUS) -> dict:
    out: dict = {"by_split": {}, "by_source": {}, "contrasts": {}}
    hrs = defaultdict(float)
    spk = defaultdict(set)
    utt = Counter()
    con = defaultdict(Counter)
    for split in SPLITS:
        for r in rows(split, corpus):
            key = f"{r['source']}/{r.get('tier', '')}"
            hrs[(split, key)] += r["duration"] / 3600
            spk[(split, key)].add(r["speaker"])
            utt[(split, key)] += 1
            for s in r["segments"]:
                pos = "phrase_final" if s["pf"] else "nonfinal"
                if s["k"] == "V":
                    lab = f"{s['s'][0]}{':' if s['long'] else ''}"
                    con[pos][lab] += 1
                    if s.get("dv"):
                        con[pos][lab + "(devoiced)"] += 1
                elif s["k"] == "C":
                    con[pos][f"{s['s']}{'(gem)' if s.get('gem') else ''}"] += 1
                elif s["k"] == "N":
                    con[pos]["N"] += 1
                elif s["k"] == "Q":
                    con[pos]["Q_final" if s.get("fq") else "Q"] += 1
    for (split, key), h in sorted(hrs.items()):
        out["by_split"].setdefault(split, {})[key] = {"hours": round(h, 2), "speakers": len(spk[(split, key)]),
                                                      "utterances": utt[(split, key)]}
    lic = defaultdict(float)
    for (split, key), h in hrs.items():
        lic[key.split("/")[0]] += h
    out["hours_by_source"] = {k: round(v, 2) for k, v in lic.items()}
    out["hours_total"] = round(sum(hrs.values()), 2)
    out["speakers_total"] = len(set().union(*spk.values())) if spk else 0
    tot = Counter()
    for c in con.values():
        tot.update(c)
    out["contrasts"] = {"all": dict(sorted(tot.items())),
                        "phrase_final": dict(sorted(con["phrase_final"].items()))}
    return out


# ---------------------------------------------------------------- anchors vs the runtime aligner

def anchors(n: int, corpus: Path = CORPUS) -> dict:
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import warnings

    warnings.simplefilter("ignore")
    from pronunciada.audio import load_audio
    from pronunciada.scorer import Scorer

    pool = [r for s in ("validation", "heldback") for r in rows(s, corpus)]
    pool += [r for r in rows("train", corpus) if r["source"] != "cv22"]
    random.Random(3).shuffle(pool)
    by_src = defaultdict(list)
    for r in pool:
        by_src[r["source"]].append(r)
    take = by_src["cv22"][: int(n * 0.7)] + by_src["fleurs"][: n // 6] + by_src["kokoro"][: n // 6]
    sc = Scorer("ja", threads=2)
    diffs = defaultdict(list)
    bnd = defaultdict(list)
    skipped = Counter()
    for r in take:
        res = sc.score((load_audio(str(WORK / r["audio"])), 16000), r["text"])
        ref = res._pronunciada["ref"]  # noqa: SLF001
        flat = [p for w in res.words for p in w.phones]
        if len(flat) != len(ref.phones):
            skipped["phone_count"] += 1
            continue
        nuc = [(rp, pr) for rp, pr in zip(ref.phones, flat) if rp.is_nucleus]
        ours = [s for s in r["segments"] if s["k"] in ("V", "N")]
        if len(nuc) != len(ours):
            skipped["nucleus_count"] += 1
            continue
        for (rp, pr), s in zip(nuc, ours):
            if pr.end <= pr.start:
                skipped["zero_length_anchor"] += 1
                continue
            mid_z = (pr.start + pr.end) / 2
            mid_a = (s["t0"] + s["t1"]) / 2
            key = "N" if s["k"] == "N" else ("long" if s["long"] else "short")
            diffs[key].append(mid_z - mid_a)
            bnd[key + "_start"].append(pr.start - s["t0"])
            bnd[key + "_end"].append(pr.end - s["t1"])
    out = {"n_utterances": len(take), "skipped": dict(skipped),
           "note": "difference = runtime (ZIPA) aligner minus corpus aligner, ms; nucleus phones only"}
    for k, v in sorted({**diffs, **bnd}.items()):
        x = np.array(v) * 1000
        out[k] = {"n": len(x), "mean": round(float(x.mean()), 1), "sd": round(float(x.std()), 1),
                  **{f"p{q}": round(float(np.percentile(x, q)), 1) for q in (5, 10, 25, 50, 75, 90, 95)},
                  "abs_median": round(float(np.median(np.abs(x))), 1),
                  "within_20ms": round(float((np.abs(x) <= 20).mean()), 3),
                  "within_40ms": round(float((np.abs(x) <= 40).mean()), 3)}
    return out


# ---------------------------------------------------------------- independent cross-check

def _frames(x: np.ndarray, sr: int = 16000, win: float = 0.03, hop: float = 0.005):
    w, h = int(win * sr), int(hop * sr)
    n = max(0, 1 + (len(x) - w) // h)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    return x[idx] * np.hanning(w)[None, :], hop


def voicing(x: np.ndarray, sr: int = 16000):
    """Per 5 ms frame: log energy (dB) and normalised autocorrelation peak in the 70-400 Hz lag range."""
    fr, hop = _frames(x, sr)
    e = 10 * np.log10((fr ** 2).mean(axis=1) + 1e-10)
    spec = np.fft.rfft(fr, n=2 * fr.shape[1], axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, : fr.shape[1]]
    ac = ac / (ac[:, :1] + 1e-12)
    lo, hi = int(sr / 400), int(sr / 70)
    peak = ac[:, lo:hi].max(axis=1)
    return e, peak, hop


def xcheck(n: int, corpus: Path = CORPUS) -> dict:
    """Vowels between voiceless consonants: the voiced run around the vowel midpoint (energy within
    25 dB of the clip's 95th percentile and autocorrelation peak >= 0.45). Geminate vs single voiceless
    stops after a vowel: the low-energy run (energy 25 dB under the vowel) around the closure midpoint."""
    from pronunciada.audio import load_audio

    pool = [r for r in rows(corpus=corpus)]
    random.Random(5).shuffle(pool)
    vt, ct = [], []
    for r in pool:
        segs = r["segments"]
        for i, s in enumerate(segs):
            if s["k"] == "V" and not s.get("dv") and not s.get("de") and 0 < i < len(segs) - 1 \
                    and segs[i - 1]["s"] in VOICELESS and segs[i + 1]["s"] in VOICELESS \
                    and s["prev"] != "#" and s["next"] not in ("#", "_"):
                vt.append((r, s))
            if s["k"] == "C" and s["s"] in ("k", "t", "p") and i > 0 and segs[i - 1]["k"] in ("V", "Q") \
                    and s["prev"] not in ("#", "_"):
                ct.append((r, i))
        if len(vt) >= n * 4 and len(ct) >= n * 4:
            break
    longs = [t for t in vt if t[1]["long"]][: n // 2]
    shorts = [t for t in vt if not t[1]["long"]][: n // 2]
    gems = [t for t in ct if t[0]["segments"][t[1]].get("gem")][: n // 2]
    sings = [t for t in ct if not t[0]["segments"][t[1]].get("gem")][: n // 2]
    cache: dict[str, tuple] = {}

    def feats(r):
        if r["id"] not in cache:
            if len(cache) > 200:
                cache.clear()
            x = load_audio(str(WORK / r["audio"]))
            cache[r["id"]] = voicing(x)
        return cache[r["id"]]

    vres = []
    for r, s in longs + shorts:
        e, pk, hop = feats(r)
        thr = np.percentile(e, 95) - 25
        voiced = (e > thr) & (pk >= 0.45)
        m = int(((s["t0"] + s["t1"]) / 2) / hop)
        if m >= len(voiced) or not voiced[m]:
            vres.append((s["long"], s["t1"] - s["t0"], None))
            continue
        a = m
        while a > 0 and voiced[a - 1]:
            a -= 1
        b = m
        while b + 1 < len(voiced) and voiced[b + 1]:
            b += 1
        vres.append((s["long"], s["t1"] - s["t0"], (b - a + 1) * hop))
    cres = []
    for r, i in gems + sings:
        segs = r["segments"]
        s = segs[i]
        e, pk, hop = feats(r)
        t0 = segs[i - 1]["t0"] if s.get("gem") else s["t0"]
        clos_mid = (t0 + s["t1"]) / 2 if not s.get("gem") else (segs[i - 1]["t0"] + segs[i - 1]["t1"]) / 2
        if s.get("gem"):
            vow = segs[i - 2] if i >= 2 else None
        else:
            vow = segs[i - 1]
        if vow is None or vow["k"] != "V":
            continue
        va, vb = int(vow["t0"] / hop), max(int(vow["t1"] / hop), int(vow["t0"] / hop) + 1)
        thr = e[va:vb].max() - 25 if vb <= len(e) else None
        if thr is None:
            continue
        low = e < thr
        _ = clos_mid
        # the longest low-energy run that overlaps the aligned consonant interval (closure + release)
        ia, ib = int(t0 / hop), min(int(s["t1"] / hop), len(low) - 1)
        best = 0
        k = ia
        while k <= ib:
            if low[k]:
                a = k
                while a > 0 and low[a - 1]:
                    a -= 1
                b = k
                while b + 1 < len(low) and low[b + 1]:
                    b += 1
                best = max(best, b - a + 1)
                k = b + 1
            else:
                k += 1
        cres.append((bool(s.get("gem")), s["t1"] - t0, best * hop if best else None))

    def summ(res, pos_name, neg_name):
        ok = [t for t in res if t[2] is not None]
        al = np.array([t[1] for t in ok])
        ind = np.array([t[2] for t in ok])
        lab = np.array([t[0] for t in ok])
        return {"tokens": len(res), "measured": len(ok),
                "pearson_r": round(float(np.corrcoef(al, ind)[0, 1]), 3) if len(ok) > 2 else None,
                "auc_aligner": round(auc(al[lab], al[~lab]), 4),
                "auc_independent": round(auc(ind[lab], ind[~lab]), 4),
                pos_name: {"aligner": _q(al[lab]), "independent": _q(ind[lab])},
                neg_name: {"aligner": _q(al[~lab]), "independent": _q(ind[~lab])},
                "label_agreement_at_median_split": round(float(((ind > np.median(ind)) == lab).mean()), 3)}
    return {"vowels_between_voiceless": summ(vres, "long", "short"),
            "voiceless_stops_k_t_p": summ(cres, "geminate", "single"),
            "method": "voiced run: 5 ms frames, energy > p95-25 dB and autocorrelation peak >= 0.45 (70-400 Hz);"
                      " closure: longest run 25 dB below the preceding vowel's peak that overlaps the aligned consonant (gem: Q+release)"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("durations", "anchors", "xcheck", "counts"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--table", action="store_true", help="also write results/tables/round20_ja_corpus_<cmd>.json")
    a = ap.parse_args(argv)
    corpus = Path(a.corpus)
    fn = {"durations": lambda: durations(corpus), "counts": lambda: counts(corpus),
          "anchors": lambda: anchors(a.n, corpus), "xcheck": lambda: xcheck(a.n, corpus)}[a.cmd]
    out = fn()
    (WORK / "qc").mkdir(exist_ok=True)
    json.dump(out, open(WORK / "qc" / f"{a.cmd}.json", "w"), indent=1, ensure_ascii=False)
    if a.table:
        json.dump(out, open(TABLES / f"round20_ja_corpus_{a.cmd}.json", "w"), indent=1, ensure_ascii=False)
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
