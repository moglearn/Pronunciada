"""Round 20 (Japanese vowel length): per-vowel windows of encoder activations, natural and
WSOLA-stretched, clean and noisy.

    python -m bench.round20_ja_length.extract POOL --shard K --of N [--layers enc1,enc3] [--probe]

POOL:
* ``fit``  Common Voice ja train clips of the round-19 fit pools (cv_fit + cv_fit2: no speaker of
           Lingolix ja, no cv-real test-half speaker, no benchmark prompt) + FLEURS ja train;
* ``dev``  the round-19 dev-pool natives (Common Voice ja dev + FLEURS validation, devnat_in.jsonl).

For every clip the runtime Scorer (encoder outputs of every stack exposed) scores the natural audio
with its own text; every vowel the length channel judges gives one example: the window of
activations centred on the vowel's CTC peak (+-W output frames, 20 ms each), the local and utterance
mora duration, the text's long / short label, the shipped length probability. Then "versions":
the same clip with up to three judged vowels (>= 450 ms apart) time-stretched by WSOLA (factor per
vowel; the rest untouched), scored again with the ORIGINAL text, so the audio disagrees with the
text at those vowels; optionally with DEMAND noise mixed in. Targets get a soft "audio is long"
label from the source label and the factor (bench.round20_ja_length.common.soft_label).

fit versions: natural clean (all vowels), WSOLA clean, WSOLA + noise (5-20 dB, training
environments), natural + noise (every second clip). dev versions: natural clean, WSOLA at clean /
20 / 10 / 5 dB (held-back noise environments). Output: shards of float16 windows
($PRONUNCIADA_CACHE/round20/jalength/win_<pool>[_probe]/shard_KKK_of_NNN.npz); resumable per shard.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from bench.round20_ja_length.common import (DEV_ENVS, OUT, TRAIN_ENVS, W, add_noise, noise_bank, soft_label)

R19 = Path.home() / ".cache" / "pronunciada" / "round19" / "japanese"
LONG_F = (0.8, 0.67, 0.55, 0.45, 0.4)
SHORT_F = (1.3, 1.5, 1.7, 2.0, 2.3, 2.6)


def pool_rows(pool: str) -> list[dict]:
    if pool == "dev":
        rows = [json.loads(x) for x in open(R19 / "devnat_in.jsonl", encoding="utf-8")]
        return [{"id": r["id"], "audio": r["audio"], "text": r["text"], "group": r["group"], "source": r["source"]}
                for r in rows]
    rows = []
    for f in ("cv_fit.jsonl", "cv_fit2.jsonl"):
        for x in open(R19 / f, encoding="utf-8"):
            r = json.loads(x)
            rows.append({"id": r["id"], "audio": r["audio"], "text": r["text"], "group": r["speaker"], "source": "cv"})
    from bench.pipeline.common import records

    for r in records("fleurs", split="train", lang="ja"):
        # FLEURS has no speaker ids; the sentence id groups the readings of one sentence
        rows.append({"id": r["id"], "audio": r["audio"], "text": r["text"],
                     "group": f"fl{r['labels']['fleurs_id']}", "source": "fleurs"})
    return rows


def make_scorer(threads: int = 2):
    from pronunciada.acoustic import AcousticModel
    from pronunciada.scorer import Scorer

    sc = Scorer("ja", threads=threads)
    sc.am = AcousticModel(sc.am.name, sc.am.precision, threads, embeddings="all")
    return sc


def steady_span(fb: np.ndarray, E: np.ndarray, sil: float, c: int, cos: float = 0.85) -> tuple[int, int]:
    """[L, R] (10 ms fbank frames) of the vowel's stable-spectrum run around frame c
    (pronunciada.length_r19's v_stable)."""
    F = fb - fb.mean(0, keepdims=True)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-6)
    n = len(Fn)
    c = min(max(c, 0), n - 1)
    rv = Fn[max(c - 1, 0):c + 2].mean(0)
    rv /= np.linalg.norm(rv) + 1e-6
    L = R = c
    while L - 1 >= 0 and float(Fn[L - 1] @ rv) >= cos and E[L - 1] >= sil:
        L -= 1
    while R + 1 < n and float(Fn[R + 1] @ rv) >= cos and E[R + 1] >= sil:
        R += 1
    return L, R


def analyse(sc, wav: np.ndarray, text: str, layers):
    """Score once; per judged vowel: dict with window and meta (see module docstring)."""
    from pronunciada import length as L
    from pronunciada.length_r19 import _morae
    from pronunciada.text.ipa import is_vowel, token_ids

    res = sc.score((wav, 16000), text)
    c = res._pronunciada  # noqa: SLF001
    ref, ev, lres, emb, lpa = c["ref"], c["ev"], c.get("length"), c.get("emb"), c["lpa"]
    if lres is None or emb is None:
        return [], None
    fb = sc._lp_cache[next(reversed(sc._lp_cache))][1]  # noqa: SLF001  (the fbank of this clip)
    tid = token_ids()
    lab = L.labels(ref)
    X = np.concatenate([np.asarray(emb[k], np.float32) for k in layers], axis=1)
    T = len(X)
    on = np.array([s.start_frame for s in ev.slots], np.int64)
    # mora durations (20 ms frames): the utterance, and the utterance outside the vowel's word
    nW = len(ref.words)
    wdur, wmor = np.zeros(nW), np.zeros(nW)
    for k, w in enumerate(ref.words):
        a, b = w.phones
        s1 = int(on[b]) if b < len(on) else int(ev.slots[b - 1].end_frame)
        wdur[k] = max(s1 - int(on[a]), 1)
        wmor[k] = _morae(ref, w)
    tot_d, tot_m = wdur.sum(), wmor.sum()
    md_utt = max(tot_d / max(tot_m, 1.0), 1.5)
    E = np.log(np.exp(np.asarray(fb, np.float64)).sum(1) + 1e-10)
    loud, floor = float(np.percentile(E, 95)), float(np.percentile(E, 5))
    sil = max(loud - 3.0 * math.log(10.0), floor + 0.35 * (loud - floor))
    out = []
    for i, p in enumerate(ref.phones):
        if lab[i] is None or not is_vowel(p.ipa) or lres["prob"][i] is None:
            continue
        sl = ev.slots[i]
        s, e = int(sl.start_frame), int(sl.end_frame)
        tk = tid.get(p.ipa[0])
        if e > s and tk is not None:
            cf = s + int(np.argmax(lpa[s:e, tk]))
        else:
            cf = s
        cf = min(max(cf, 0), T - 1)
        win = np.zeros((2 * W + 1, X.shape[1]), np.float16)
        a, b = cf - W, cf + W + 1
        aa, bb = max(a, 0), min(b, T)
        win[aa - a: aa - a + (bb - aa)] = X[aa:bb]
        k = p.word
        md_loc = (tot_d - wdur[k]) / max(tot_m - wmor[k], 1.0) if tot_m - wmor[k] >= 3 else md_utt
        md_loc = max(md_loc, 1.5)
        # 10 ms centre of the vowel as length_r19 takes it, and its stable-spectrum run
        a10 = int(round((on[i] * 0.02 + 0.03) / 0.01))
        b10 = int(round((on[i + 1] * 0.02 + 0.03) / 0.01)) if i + 1 < len(on) else a10 + 8
        Ls, Rs = steady_span(np.asarray(fb, np.float32), E, sil, (a10 + b10) // 2)
        out.append({"i": i, "win": win, "valid": (aa - a, aa - a + (bb - aa)), "peak": cf,
                    "span": (s, e), "next_on": int(on[i + 1]) if i + 1 < len(on) else e,
                    "md_loc": md_loc, "md_utt": md_utt, "text_long": float(lab[i] == "long"),
                    "vowel": p.ipa[0], "lprob": float(lres["prob"][i]), "steady": (Ls, Rs),
                    "pos_s": cf * 0.02 + 0.03})
    return out, res


def choose_targets(items, rng, n: int = 3, gap_s: float = 0.45):
    cand = [it for it in items if it["steady"][1] - it["steady"][0] + 1 >= 4]
    rng.shuffle(cand)
    chosen = []
    for it in cand:
        if all(abs(it["pos_s"] - o["pos_s"]) >= gap_s for o in chosen):
            chosen.append(it)
        if len(chosen) >= n:
            break
    return chosen


def factor_for(it, rng) -> float:
    if rng.random() < 0.2:                       # control: a near-identity stretch, natural label
        return float(rng.uniform(0.9, 1.1))
    return float(rng.choice(LONG_F if it["text_long"] else SHORT_F))


def modified(wav, targets, factors):
    from bench.round20_ja_length.wsola import stretch_segment

    y = wav
    # right to left, so earlier sample positions stay valid
    for it, f in sorted(zip(targets, factors), key=lambda z: -z[0]["steady"][0]):
        Ls, Rs = it["steady"]
        y = stretch_segment(y, Ls * 160, (Rs + 1) * 160, f)
    return y


def rows_from(items, keep_i=None, info=None):
    rows = []
    for it in items:
        if keep_i is not None and it["i"] not in keep_i:
            continue
        r = {k: it[k] for k in ("i", "md_loc", "md_utt", "text_long", "vowel", "lprob", "peak")}
        r["win"] = it["win"]
        r["v0"], r["v1"] = it["valid"]
        r.update(info or {})
        rows.append(r)
    return rows


def run(pool: str, shard: int, of: int, layers, probe: bool, threads: int = 2) -> None:
    import warnings

    warnings.simplefilter("ignore")
    from bench.pipeline.common import stable_seed
    from bench.round19_ja.dump import _wav

    rows = pool_rows(pool)
    rows = rows[shard::of]
    if probe:
        rows = rows[: max(len(rows) // 4, 1)]
    od = OUT / (f"win_{pool}" + ("_probe" if probe else ""))
    od.mkdir(parents=True, exist_ok=True)
    dest = od / f"shard_{shard:03d}_of_{of:03d}.npz"
    if dest.exists():
        print("exists", dest, file=sys.stderr)
        return
    sc = make_scorer(threads)
    nb = noise_bank(DEV_ENVS if pool == "dev" else TRAIN_ENVS)
    acc: list[dict] = []
    t0 = time.time()
    for n, r in enumerate(rows):
        rng = np.random.default_rng(stable_seed(r["id"], "r20len"))
        try:
            wav = _wav(r["audio"]).astype(np.float32)
            items, _ = analyse(sc, wav, r["text"], layers)
        except Exception as e:  # noqa: BLE001
            print("skip", r["id"], type(e).__name__, e, file=sys.stderr)
            continue
        base = {"clip": r["id"], "group": r["group"], "source": r["source"]}
        if not items:
            continue
        if probe:                                # layer probe: every long vowel, a third of the short ones
            items = [it for it in items if it["text_long"] or rng.random() < 0.35]
            acc += rows_from(items, info={**base, "kind": 0, "factor": 1.0, "snr": 99.0, "y": None})
            continue
        if pool == "fit":                        # natural short vowels are plentiful: keep half
            nat_items = [it for it in items if it["text_long"] or rng.random() < 0.5]
        else:
            nat_items = items
        acc += rows_from(nat_items, info={**base, "kind": 0, "factor": 1.0, "snr": 99.0, "y": None})
        versions = ([("wsola", None), ("wsola", "train")] + ([("natural", "train")] if n % 2 == 0 else [])
                    if pool == "fit" else [("wsola", None), ("wsola", 20.0), ("wsola", 10.0), ("wsola", 5.0)])
        for vk, (kind, snr) in enumerate(versions):
            if kind == "wsola":
                tg = choose_targets(items, rng)
                if not tg:
                    continue
                fs = [factor_for(it, rng) for it in tg]
                y = modified(wav, tg, fs)
            else:
                tg, fs, y = [], [], wav
            s_db = float(rng.uniform(5, 20)) if snr == "train" else snr
            if s_db is not None:
                y = add_noise(y, nb, s_db, rng)
            try:
                it2, _ = analyse(sc, y, r["text"], layers)
            except Exception as e:  # noqa: BLE001
                print("skip version", r["id"], vk, type(e).__name__, e, file=sys.stderr)
                continue
            fmap = {it["i"]: f for it, f in zip(tg, fs)}
            keep = set(fmap) if kind == "wsola" else (
                {it["i"] for it in it2 if it["text_long"] or rng.random() < 0.5} if pool == "fit" else None)
            for row in rows_from(it2, keep_i=keep, info={**base, "kind": 1 + vk, "snr": 99.0 if s_db is None else s_db}):
                f = fmap.get(row["i"], 1.0)
                row["factor"] = f
                row["y"] = soft_label(row["text_long"], f)
                acc.append(row)
        if n % 25 == 24:
            print(f"shard {shard}: {n + 1}/{len(rows)} clips, {len(acc)} rows, {time.time() - t0:.0f}s",
                  file=sys.stderr, flush=True)
    save(dest, acc, layers)
    print(f"done shard {shard}: {len(acc)} rows in {time.time() - t0:.0f}s -> {dest}", file=sys.stderr)


def save(dest: Path, acc: list[dict], layers) -> None:
    win = np.stack([a["win"] for a in acc]).astype(np.float16) if acc else np.zeros((0, 2 * W + 1, 1), np.float16)
    meta = {k: [a[k] for a in acc] for k in acc[0] if k != "win"} if acc else {}
    meta["y"] = [(a["y"] if a["y"] is not None else a["text_long"]) for a in acc]
    tmp = dest.with_suffix(".tmp.npz")
    np.savez(tmp, win=win, meta=np.array(json.dumps({"layers": list(layers), "rows": meta}, ensure_ascii=False)))
    os.replace(tmp, dest)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", choices=("fit", "dev"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--layers", default="enc0,enc1,enc2,enc3,enc4,enc5,final")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    run(a.pool, a.shard, a.of, a.layers.split(","), a.probe, a.threads)


if __name__ == "__main__":
    main()
