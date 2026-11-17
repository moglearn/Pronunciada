"""Contrast audit: for which specific sound contrasts is the scorer blind, per language?

    python -m bench.audit.contrast_audit run [--lang all] [--clips 250] [--workers 2]
    python -m bench.audit.contrast_audit asp --lang zh,ko   # revised aspiration measures (natives)
    python -m bench.audit.contrast_audit report      # -> results/tables/contrast_audit.json

FLEURS VALIDATION only, native audio, ten languages. For every contrast of
bench/audit/contrasts.py, eligible positions of each clip's reference are collected, a fixed-seed
sample is drawn (TARGET edits per (language, contrast), split over the two directions, at most
PER_CLIP per clip and direction), the reference is edited at that one position (the native audio
keeps member A, the reference now demands member B) and the edited reference is scored through the
public Scorer (default settings). Recorded per edit: the edited sound's probability, its word's
probability, raw evidence (llr_sub, llr_del, heard) and the same quantities at the same position of
the unedited native. Each clip is also scored on its correct reference (the false-alarm pool).

Natives also feed two "is a small detector feasible" measurements that need no edit:
* length / gemination: onset-to-onset intervals from the canonical alignment (20 ms frames),
  normalised by the utterance's median for that sound class, long vs short by the text's labels;
* aspiration (zh, ko): burst -> voicing-onset gap (burst = steepest rise of 2.5-8 kHz energy near
  the aligned stop, voicing from pronunciada/prosody/f0.py), high-frequency energy at the burst,
  the stop's onset-to-onset interval, and (ko) f0 at the vowel onset.

Resumable: everything is cached per language and chunk of 25 clips under
$PRONUNCIADA_CACHE/audit/contrast_v1/<lang>/.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

LANGS = ("en", "zh", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")
TARGET = 240          # edited items per (language, contrast)
PER_CLIP = 2          # at most per (clip, contrast, direction)
CHUNK = 25
VERSION = "contrast_v1"
HF_LO, HF_HI = 2500.0, 8000.0


def cache_dir(lang: str) -> Path:
    from bench.data.common import cache_root

    p = cache_root() / "audit" / VERSION / lang
    p.mkdir(parents=True, exist_ok=True)
    return p


def choose(lang: str, n: int) -> list[dict]:
    from bench.pipeline.common import records

    recs = records("fleurs-validation", lang=lang)
    rng = np.random.default_rng(0)
    return [recs[i] for i in sorted(rng.choice(len(recs), min(n, len(recs)), replace=False))]


# ----------------------------------------------------------------------------- plan


def _alloc(avail: dict[str, int], total: int) -> dict[str, int]:
    """Split `total` evenly over directions, giving what a short direction cannot use to the others."""
    out = {d: 0 for d in avail}
    left = total
    open_ = [d for d in sorted(avail) if avail[d] > 0]
    while left > 0 and open_:
        share = max(left // len(open_), 1)
        for d in list(open_):
            k = min(share, avail[d] - out[d], left)
            out[d] += k
            left -= k
            if out[d] >= avail[d]:
                open_.remove(d)
            if left <= 0:
                break
    return out


def make_plan(lang: str, recs: list[dict], g2p) -> dict:
    """{clip id: [site dicts]} — deterministic given lang and recs."""
    from .contrasts import contrasts_for, sites

    cands: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    n_sites: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for r in recs:
        try:
            ref = g2p(r["text"])
            ref.validate()
        except Exception:  # noqa: BLE001
            continue
        for c in contrasts_for(lang):
            for s in sites(ref, c):
                cands[(c.name, s.direction)].append((r["id"], s.__dict__.copy()))
                n_sites[c.name][s.direction] += 1
    plan: dict[str, list[dict]] = defaultdict(list)
    for c in contrasts_for(lang):
        dirs = {d: len(v) for (cn, d), v in cands.items() if cn == c.name}
        rng = np.random.default_rng(zlib.crc32(f"{lang}/{c.name}".encode()))
        # per-clip cap first, then the quota
        capped = {}
        for d in sorted(dirs):
            pool = cands[(c.name, d)]
            order = rng.permutation(len(pool))
            per: dict[str, int] = defaultdict(int)
            keep = []
            for j in order:
                cid, s = pool[j]
                if per[cid] < PER_CLIP:
                    per[cid] += 1
                    keep.append((cid, s))
            capped[d] = keep
        quota = _alloc({d: len(v) for d, v in capped.items()}, TARGET)
        for d, v in capped.items():
            for cid, s in v[:quota[d]]:
                plan[cid].append(s)
    return {"plan": dict(plan), "n_sites": {k: dict(v) for k, v in n_sites.items()}}


# ----------------------------------------------------------------------------- acoustic measures


def _hf_track(wav: np.ndarray, sr: int = 16000):
    """(times, log HF power, log total power) on a 5 ms hop, 16 ms window."""
    n, hop = 256, 80
    if len(wav) < n:
        wav = np.pad(wav, (0, n - len(wav)))
    fr = np.lib.stride_tricks.sliding_window_view(wav, n)[::hop] * np.hanning(n)
    spec = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    hf = spec[:, (f >= HF_LO) & (f <= HF_HI)].sum(1)
    tot = spec[:, f >= 80].sum(1)
    t = (np.arange(len(fr)) * hop + n / 2) / sr
    return t, np.log10(hf + 1e-10), np.log10(tot + 1e-10)


def stop_measures(t_c: float, t_next: float, hf, f0) -> dict:
    """Burst / voicing measures around an aligned stop whose CTC spike starts at t_c.

    The voiceless stretch of the pitch tracker that overlaps t_c +- 50 ms is the stop's
    closure + release (none: voiced throughout, gap 0). Its end is the voicing onset; the burst is
    the steepest rise of 2.5-8 kHz energy inside it; vot = voicing onset - burst."""
    t, lhf, ltot = hf
    ft = np.arange(len(f0.f0)) * 0.01 + 0.005
    v = np.asarray(f0.voiced, dtype=bool)
    near = np.flatnonzero((ft >= t_c - 0.05) & (ft <= t_c + 0.05) & ~v)
    f0v = np.asarray(f0.f0, dtype=np.float64)
    vv = f0v[v & (f0v > 0)]
    out = {"c_to_next": float(t_next - t_c)}
    if len(near) == 0:
        tv = t_c
        out.update(gap=0.0, vot=0.0, hf_burst=float("nan"), hf_ratio=float("nan"))
    else:
        j = near[np.argmin(np.abs(ft[near] - t_c))]
        j0 = j
        while j0 > 0 and not v[j0 - 1] and ft[j] - ft[j0 - 1] < 0.4:
            j0 -= 1
        j1 = j
        while j1 + 1 < len(v) and not v[j1 + 1] and ft[j1 + 1] - ft[j] < 0.4:
            j1 += 1
        off, tv = ft[j0] - 0.005, ft[j1] + 0.005
        if j0 == 0 or j1 + 1 >= len(v):
            return {}                                     # utterance edge: no voiced neighbours
        m = np.flatnonzero((t >= off) & (t <= tv))
        if len(m) >= 2:
            k = m[1:][int(np.argmax(np.diff(lhf[m])))]
            bt = float(t[k])
        else:
            bt = off
        ref95 = float(np.percentile(lhf, 95))
        w = (t >= bt) & (t <= bt + 0.02)
        asp = (t >= bt) & (t <= tv)
        out.update(gap=float(tv - off), vot=float(max(tv - bt, 0.0)),
                   hf_burst=float(lhf[w].mean() - ref95) if w.any() else float("nan"),
                   hf_ratio=float((lhf[asp] - ltot[asp]).mean()) if asp.any() else float("nan"))
    on = (ft >= tv) & (ft <= tv + 0.05) & v & (f0v > 0)
    out["f0_onset_st"] = (float(12 * np.log2(np.median(f0v[on]) / np.median(vv)))
                          if on.any() and len(vv) else float("nan"))
    return out


KO_INITIAL = {0: "lax", 1: "tense", 3: "lax", 4: "tense", 7: "lax", 8: "tense", 12: "lax", 13: "tense",
              14: "aspirated", 15: "aspirated", 16: "aspirated", 17: "aspirated"}
ZH_STOPS = {"p", "t", "k", "ts", "ʈʂ", "tɕ"}


def aspiration_rows(lang: str, ref, ev, wav, f0) -> list[dict]:
    if lang not in ("zh", "ko"):
        return []
    hf = _hf_track(np.asarray(wav, dtype=np.float32))
    fs = ev.frame_shift
    sl = ev.slots
    cons = [sl[i + 1].start_frame - sl[i].start_frame for i in range(len(sl) - 1)
            if not ref.phones[i].is_nucleus and sl[i].alt_index >= 0 and sl[i + 1].alt_index >= 0]
    med = float(np.median(cons)) * fs if cons else float("nan")
    out = []
    for i, p in enumerate(ref.phones):
        base = p.ipa.replace("ʰ", "")
        if base not in ZH_STOPS or i + 1 >= len(ref.phones) or sl[i].alt_index < 0 or sl[i + 1].alt_index < 0:
            continue
        s = ref.syllables[p.syllable]
        if s.phones[0] != i:
            continue
        label = "aspirated" if p.ipa.endswith("ʰ") else "plain"
        if lang == "ko":
            w = ref.words[p.word]
            if w.syllables[0] != p.syllable or not s.text or not ("가" <= s.text[0] <= "힣"):
                continue
            label = KO_INITIAL.get((ord(s.text[0]) - 0xAC00) // 588)
            if label is None:
                continue
        mm = stop_measures(sl[i].start_frame * fs, sl[i + 1].start_frame * fs, hf, f0)
        if mm:
            mm.update(label=label, ipa=p.ipa, stop=base in ("p", "t", "k"), c_to_next_norm=mm["c_to_next"] / med
                      if med and np.isfinite(med) else float("nan"))
            out.append(mm)
    return out


# ----------------------------------------------------------------------------- scoring


def _phone_probs(res) -> list[float]:
    return [p.prob for w in res.words for p in w.phones]


def native_rows(res) -> dict:
    c = res._pronunciada  # noqa: SLF001
    ref, tab, ev = c["ref"], c["tab"], c["ev"]
    probs = _phone_probs(res)
    assert len(probs) == len(ref.phones)
    out = {"ipa": [p.ipa for p in ref.phones], "word": [p.word for p in ref.phones],
           "nucleus": [p.is_nucleus for p in ref.phones],
           "stress": [ref.syllables[p.syllable].stress for p in ref.phones],
           "alts": [[a for a, _ in p.alts] for p in ref.phones],
           "prob": np.asarray(probs, dtype=np.float32),
           "llr_sub": np.asarray(tab["llr_sub"], dtype=np.float32),
           "llr_del": np.asarray(tab["llr_del"], dtype=np.float32),
           "heard": [s.heard for s in ev.slots], "alt_index": [s.alt_index for s in ev.slots],
           "start": np.asarray([s.start_frame for s in ev.slots], dtype=np.int32),
           "word_prob": np.asarray([w.prob for w in res.words], dtype=np.float32),
           "word_text": [w.text for w in res.words]}
    tones = []
    for w in res.words:
        tones += [s.tone_prob for s in w.syllables]
    out["tone_prob"] = tones
    return out


def edited_row(res, e, nat: dict) -> dict:
    c = res._pronunciada  # noqa: SLF001
    ref, tab, ev = c["ref"], c["tab"], c["ev"]
    s = e.site
    row = {"contrast": s.contrast, "direction": s.direction, "src": s.src, "dst": s.dst,
           "kind": s.kind, "alt": bool(e.partner_in_alts), "word_prob": float(res.words[e.word].prob),
           "nat_word_prob": float(nat["word_prob"][e.word])}
    if s.kind == "tone":
        w = ref.words[e.word]
        k = e.flag[0] - w.syllables[0]
        row["prob"] = res.words[e.word].syllables[k].tone_prob
        row["nat_prob"] = nat["tone_prob"][e.flag[0]]
        row.update(llr_sub=float("nan"), llr_del=float("nan"), nat_llr_sub=float("nan"), heard="",
                   nat_heard="")
        return row
    probs = _phone_probs(res)
    f = e.flag
    j = min(f, key=lambda i: probs[i])
    row.update(prob=float(min(probs[i] for i in f)), llr_sub=float(min(tab["llr_sub"][i] for i in f)),
               llr_del=float(min(tab["llr_del"][i] for i in f)), heard=ev.slots[j].heard,
               nat_prob=float(nat["prob"][s.index]), nat_llr_sub=float(nat["llr_sub"][s.index]),
               nat_heard=nat["heard"][s.index])
    return row


def run_lang(lang: str, n_clips: int, threads: int = 2) -> None:
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(threads)
    import warnings

    warnings.simplefilter("ignore")
    from bench.pipeline.common import iter_parquet_audio
    from pronunciada.scorer import Scorer
    from pronunciada.text import get_g2p

    from .contrasts import Site, apply

    d = cache_dir(lang)
    recs = choose(lang, n_clips)
    g2p = get_g2p(lang)
    pp = d / f"plan_{n_clips}.json"
    if pp.exists():
        plan = json.loads(pp.read_text())
    else:
        plan = make_plan(lang, recs, g2p)
        pp.write_text(json.dumps(plan, ensure_ascii=False))
    sc = None
    for k0 in range(0, len(recs), CHUNK):
        out = d / f"chunk_{n_clips}_{k0:04d}.pkl"
        if out.exists():
            continue
        if sc is None:
            sc = Scorer(lang, threads=threads)
        chunk = recs[k0:k0 + CHUNK]
        by = {int(r["audio"].split(":")[2]): r for r in chunk}
        path = chunk[0]["audio"].split(":")[1]
        rows = {"clips": [], "edits": [], "asp": []}
        for i, wav in iter_parquet_audio(path, set(by)):
            rec = by[i]
            try:
                ref = g2p(rec["text"])
                res = sc.score((wav, 16000), ref)
            except Exception:  # noqa: BLE001
                continue
            nat = native_rows(res)
            nat["id"] = rec["id"]
            rows["clips"].append(nat)
            c = res._pronunciada  # noqa: SLF001
            if c.get("f0") is not None:
                for a in aspiration_rows(lang, ref, c["ev"], wav, c["f0"]):
                    a["id"] = rec["id"]
                    rows["asp"].append(a)
            for sd in plan["plan"].get(rec["id"], []):
                site = Site(**sd)
                rng = np.random.default_rng(zlib.crc32(f"{rec['id']}/{sd}".encode()))
                try:
                    e = apply(ref, site, rng)
                    r2 = sc.score((wav, 16000), e.ref)
                except Exception as ex:  # noqa: BLE001
                    rows["edits"].append({"contrast": site.contrast, "error": f"{type(ex).__name__}: {ex}"})
                    continue
                row = edited_row(r2, e, nat)
                row["id"] = rec["id"]
                rows["edits"].append(row)
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(rows))
        os.replace(tmp, out)
        print(f"{lang} clips {k0 + len(chunk)}/{len(recs)} edits {len(rows['edits'])}", flush=True)


def run_asp(lang: str, n_clips: int, threads: int = 2) -> None:
    """Aspiration measures only (natives, no edits), cached as asp_<clips>.pkl; supersedes the
    rows stored in the chunks (lets the measure be revised without re-running the edits)."""
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(threads)
    import warnings

    warnings.simplefilter("ignore")
    from bench.pipeline.common import iter_parquet_audio
    from pronunciada.scorer import Scorer
    from pronunciada.text import get_g2p

    out = cache_dir(lang) / f"asp_{n_clips}.pkl"
    if out.exists() or lang not in ("zh", "ko"):
        return
    recs = choose(lang, n_clips)
    by = {int(r["audio"].split(":")[2]): r for r in recs}
    sc = Scorer(lang, threads=threads)
    g2p = get_g2p(lang)
    rows = []
    for i, wav in iter_parquet_audio(recs[0]["audio"].split(":")[1], set(by)):
        rec = by[i]
        try:
            ref = g2p(rec["text"])
            res = sc.score((wav, 16000), ref)
        except Exception:  # noqa: BLE001
            continue
        c = res._pronunciada  # noqa: SLF001
        trk = c.get("f0")
        if trk is None:                  # no prosody channel (ko): run the same pitch tracker
            from pronunciada.prosody.f0 import track

            trk = track(np.asarray(wav, dtype=np.float32))
        for a in aspiration_rows(lang, ref, c["ev"], wav, trk):
            a["id"] = rec["id"]
            rows.append(a)
    out.write_bytes(pickle.dumps(rows))
    print(lang, "aspiration rows", len(rows), flush=True)


def load(lang: str, n_clips: int) -> dict:
    d = cache_dir(lang)
    rows = {"clips": [], "edits": [], "asp": []}
    for p in sorted(d.glob(f"chunk_{n_clips}_*.pkl")):
        r = pickle.loads(p.read_bytes())
        for k in rows:
            rows[k] += r[k]
    plan = json.loads((d / f"plan_{n_clips}.json").read_text())
    rows["n_sites"] = plan["n_sites"]
    ap = d / f"asp_{n_clips}.pkl"
    if ap.exists():
        rows["asp"] = pickle.loads(ap.read_bytes())
    return rows


# ----------------------------------------------------------------------------- analysis


def _auc(pos, neg) -> float:
    from bench.metrics import roc_auc

    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return roc_auc(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def verdict(hit5: float) -> str:
    return "seen" if hit5 >= 0.6 else ("weak" if hit5 >= 0.3 else "blind")


def _intervals(nat: dict):
    """Onset-to-onset interval of every phone (frames) to the next valid phone; nan if unusable."""
    st = nat["start"].astype(np.float64)
    ok = np.asarray(nat["alt_index"]) >= 0
    n = len(st)
    own = np.full(n, np.nan)
    nxt = np.full(n, -1)
    prv = np.full(n, -1)
    last = -1
    for i in range(n):
        if ok[i]:
            prv[i] = last
            last = i
    last = -1
    for i in range(n - 1, -1, -1):
        if ok[i]:
            nxt[i] = last
            last = i
    for i in range(n):
        if ok[i] and nxt[i] >= 0:
            own[i] = st[nxt[i]] - st[i]
    return st, ok, own, nxt, prv


def duration_feasibility(lang: str, clips: list[dict]) -> dict:
    """AUC long-vs-short of rate-normalised onset-to-onset intervals among natives."""
    from pronunciada.text.ipa import is_vowel

    from .contrasts import get_contrast

    out = {}
    fams = {"ja": ["vowel_length", "gemination"], "de": ["vowel_length"], "it": ["gemination"],
            "en": ["tense_lax"]}.get(lang, [])
    for name in fams:
        c = get_contrast(lang, name)
        long_set = {a for a, _ in c.pairs}
        short_set = {b for _, b in c.pairs}
        vals = {"long": [], "short": []}
        vals_raw = {"long": [], "short": []}
        per_pair = defaultdict(lambda: {"long": [], "short": []})
        for nat in clips:
            st, ok, own, nxt, prv = _intervals(nat)
            ipa = nat["ipa"]
            words = nat["word"]
            if name in ("vowel_length", "tense_lax"):
                vmask = np.array([nat["nucleus"][i] and is_vowel(ipa[i]) for i in range(len(ipa))])
                ref_med = np.nanmedian(own[vmask]) if np.isfinite(own[vmask]).any() else np.nan
                for i, p in enumerate(ipa):
                    if not np.isfinite(own[i]) or not np.isfinite(ref_med) or ref_med <= 0:
                        continue
                    if name == "vowel_length" and lang == "de" and nat["stress"][i] != 1:
                        continue
                    if name == "tense_lax" and nat["stress"][i] != 1:
                        continue
                    lab = "long" if p in long_set else ("short" if p in short_set else None)
                    if lab:
                        vals[lab].append(own[i] / ref_med)
                        vals_raw[lab].append(own[i])
                        key = p if lab == "long" else next(a for a, b in c.pairs if b == p)
                        per_pair[key][lab].append(own[i] / ref_med)
            else:   # gemination: prev-vowel onset -> next-vowel onset across the consonant(s)
                spans = []
                for i, p in enumerate(ipa):
                    if is_vowel(p) or not ok[i]:
                        continue
                    j0 = i - 1
                    gem_it = lang == "it" and i + 1 < len(ipa) and ipa[i + 1] == p and words[i + 1] == words[i]
                    if lang == "it" and i > 0 and ipa[i - 1] == p:
                        continue                                  # second half handled with the first
                    j1 = i + 2 if gem_it else i + 1
                    if j0 < 0 or j1 >= len(ipa) or words[j0] != words[i] or words[j1] != words[i]:
                        continue
                    if not (is_vowel(ipa[j0]) and is_vowel(ipa[j1]) and ok[j0] and ok[j1]):
                        continue
                    span = st[j1] - st[j0]
                    if lang == "ja":
                        lab = "long" if p in long_set else ("short" if p in short_set else None)
                    else:
                        lab = "long" if gem_it else "short"
                    spans.append((lab, span))
                single = [s for lab, s in spans if lab == "short"]
                if not single:
                    continue
                med = float(np.median(single))
                if med <= 0:
                    continue
                for lab, s in spans:
                    if lab:
                        vals[lab].append(s / med)
                        vals_raw[lab].append(s)
        out[name] = {"n_long": len(vals["long"]), "n_short": len(vals["short"]),
                     "auc_normalised": _auc(vals["long"], vals["short"]),
                     "auc_raw_frames": _auc(vals_raw["long"], vals_raw["short"]),
                     "median_norm_long": float(np.median(vals["long"])) if vals["long"] else None,
                     "median_norm_short": float(np.median(vals["short"])) if vals["short"] else None,
                     "per_pair": {k: {"n_long": len(v["long"]), "n_short": len(v["short"]),
                                      "auc": _auc(v["long"], v["short"])} for k, v in per_pair.items()}}
        # what a one-threshold detector would do: threshold at the 5 % / 10 % false alarm on the
        # members the learner is asked to make long (natives' long items flagged as "too short")
        lo = np.asarray(vals["long"])
        sh = np.asarray(vals["short"])
        if len(lo) and len(sh):
            for q in (5, 10):
                t = np.percentile(lo, q)
                out[name][f"short_caught_at_{q}pct_FA_on_long"] = float(np.mean(sh < t))
                t2 = np.percentile(sh, 100 - q)
                out[name][f"long_caught_at_{q}pct_FA_on_short"] = float(np.mean(lo > t2))
    return out


def aspiration_feasibility(lang: str, asp: list[dict]) -> dict:
    if not asp:
        return {}
    feats = ("gap", "vot", "hf_burst", "hf_ratio", "c_to_next_norm", "f0_onset_st")
    out = {}
    pairs = ([("aspirated", "plain")] if lang == "zh" else
             [("aspirated", "lax"), ("aspirated", "tense"), ("lax", "tense")])
    for subset in ("all", "stops"):
        rows = [a for a in asp if subset == "all" or a["stop"]]
        for a_lab, b_lab in pairs:
            A = [r for r in rows if r["label"] == a_lab]
            B = [r for r in rows if r["label"] == b_lab]
            key = f"{subset}:{a_lab}_vs_{b_lab}"
            out[key] = {"n": [len(A), len(B)]}
            for f in feats:
                xa = [r.get(f, np.nan) for r in A]
                xb = [r.get(f, np.nan) for r in B]
                out[key][f] = {"auc": _auc(xa, xb),
                               "median": [float(np.nanmedian(xa)) if xa else None,
                                          float(np.nanmedian(xb)) if xb else None]}
            # two-cue logistic regression, 5-fold CV grouped by clip (VOT + HF ratio + interval [+ f0])
            fs = ["gap", "vot", "c_to_next_norm"] + (["f0_onset_st"] if lang == "ko" else [])
            X = np.array([[r.get(f, np.nan) for f in fs] for r in A + B], dtype=np.float64)
            y = np.r_[np.ones(len(A)), np.zeros(len(B))]
            g = np.array([r["id"] for r in A + B])
            okr = np.isfinite(X).all(1)
            if okr.sum() > 50 and len(set(y[okr])) == 2 and len(set(g[okr])) >= 5:
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import GroupKFold

                X, y, g = X[okr], y[okr], g[okr]
                p = np.zeros(len(y))
                for tr, te in GroupKFold(5).split(X, y, g):
                    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                    m = LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
                    p[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
                out[key]["combined_cv_auc"] = _auc(p[y == 1], p[y == 0])
                out[key]["combined_features"] = fs
    return out


def native_alt_usage(lang: str, clips: list[dict]) -> dict:
    """Where the reference expects A and lists B as an alternative: how often the native alignment
    picked B (= natives who do not make the contrast, as far as the acoustic model hears)."""
    from .contrasts import contrasts_for

    out = {}
    for c in contrasts_for(lang):
        res = {}
        if c.kind == "sub":
            for a, b in c.pairs:
                for x, y in ((a, b), (b, a)):
                    n = k = 0
                    for nat in clips:
                        for i, ipa in enumerate(nat["ipa"]):
                            if ipa == x and y in nat["alts"][i]:
                                n += 1
                                ai = nat["alt_index"][i]
                                k += ai > 0 and nat["alts"][i][ai - 1] == y
                    if n:
                        res[f"{x} (alt {y})"] = {"n": n, "native_chose_alt": k / n}
        elif c.kind == "gem_it":
            n = k = 0
            for nat in clips:
                for i, ipa in enumerate(nat["ipa"][:-1]):
                    if nat["ipa"][i + 1] == ipa and "" in nat["alts"][i] and nat["word"][i + 1] == nat["word"][i]:
                        n += 1
                        ai = nat["alt_index"][i]
                        k += ai == -1 or (ai > 0 and nat["alts"][i][ai - 1] == "")
            if n:
                res["geminate first copy (alt ∅)"] = {"n": n, "native_chose_alt": k / n}
        if res:
            out[c.name] = res
    return out


def analyse(lang: str, rows: dict) -> dict:
    from .contrasts import NOT_REPRESENTABLE, contrasts_for

    clips = rows["clips"]
    nat_p = np.concatenate([c["prob"] for c in clips])
    nat_llr = np.concatenate([c["llr_sub"] for c in clips])
    nat_w = np.concatenate([c["word_prob"] for c in clips])
    nat_t = np.array([t for c in clips for t in c["tone_prob"] if t is not None], dtype=np.float64)
    th = {q: float(np.percentile(nat_p, q)) for q in (5, 10)}
    thw = {q: float(np.percentile(nat_w, q)) for q in (5, 10)}
    tht = {q: float(np.percentile(nat_t, q)) for q in (5, 10)} if len(nat_t) else {}
    edits = [e for e in rows["edits"] if "error" not in e]
    errors = [e for e in rows["edits"] if "error" in e]
    res = {"n_clips": len(clips), "n_native_sounds": int(len(nat_p)), "n_native_words": int(len(nat_w)),
           "thresholds": {"sound": th, "word": thw, "tone": tht}, "n_edit_errors": len(errors),
           "contrasts": {}}
    for c in contrasts_for(lang):
        E = [e for e in edits if e["contrast"] == c.name]
        if not E:
            res["contrasts"][c.name] = {"label": c.label, "n": 0, "n_sites": rows["n_sites"].get(c.name, {})}
            continue
        p = np.array([e["prob"] for e in E], dtype=np.float64)
        w = np.array([e["word_prob"] for e in E], dtype=np.float64)
        llr = np.array([e["llr_sub"] for e in E], dtype=np.float64)
        tone = c.kind == "tone"
        T, NP = (tht, nat_t) if tone else (th, nat_p)

        def block(sel):
            ps, ws, ls = p[sel], w[sel], llr[sel]
            Es = [e for e, s in zip(E, sel) if s]
            nat_same = np.array([e["nat_prob"] for e in Es], dtype=np.float64)
            heard_partner = [e["nat_heard"] == e["dst"] for e in Es if e["kind"] in ("sub", "final_dev",
                                                                                     "ru_unstressed")]
            b = {"n": int(sel.sum()),
                 "sound_hit5": float(np.mean(ps <= T[5])), "sound_hit10": float(np.mean(ps <= T[10])),
                 "word_hit5": float(np.mean(ws <= thw[5])), "word_hit10": float(np.mean(ws <= thw[10])),
                 "auc": _auc(-ps, -NP), "auc_paired": _auc(-ps, -nat_same),
                 "raw_auc": float("nan") if tone else _auc(-ls, -nat_llr),
                 "raw_auc_paired": float("nan") if tone else _auc(
                     -ls, -np.array([e["nat_llr_sub"] for e in Es], dtype=np.float64)),
                 "share_above_0.9": float(np.mean(ps > 0.9)),
                 "median_prob": float(np.median(ps)), "median_llr_sub": float(np.nanmedian(ls)) if not tone else None,
                 "native_heard_as_partner": float(np.mean(heard_partner)) if heard_partner else None,
                 "native_same_pos_below_t5": float(np.mean(nat_same <= T[5]))}
            return b

        allsel = np.ones(len(E), dtype=bool)
        r = {"label": c.label, "members": list(c.members), "kind": c.kind,
             "n_sites": rows["n_sites"].get(c.name, {}), **block(allsel), "directions": {}}
        # "front-end accepts": the edit at a native position with member A asks whether the
        # reference of A lists B as an alternative, i.e. whether a learner who says B where A is
        # expected passes by construction. That is the learner error of the OPPOSITE edit
        # direction, so each direction gets the alternative share measured on the other one
        # (one-way contrasts: final_dev / ru_unstressed are built so that their own share is the
        # learner error; tone has no alternatives).
        alt_by_dir = {d: float(np.mean([e["alt"] for e in E if e["direction"] == d]))
                      for d in {e["direction"] for e in E}}
        a, b = c.members
        acc_n = []
        for d in sorted(alt_by_dir):
            sel = np.array([e["direction"] == d for e in E])
            name = f"said {a}, expected {b}" if d == "A>B" else f"said {b}, expected {a}"
            blk = block(sel)
            opp = "B>A" if d == "A>B" else "A>B"
            if c.kind in ("final_dev", "ru_unstressed", "tone"):
                acc = alt_by_dir[d]
            else:
                acc = alt_by_dir.get(opp)
            blk["learner_error_accepted"] = acc
            blk["verdict"] = verdict(blk["sound_hit5"])
            blk["verdict_today"] = ("not representable (accepted as alternative)"
                                    if acc is not None and acc >= 0.5 else blk["verdict"])
            r["directions"][name] = blk
            if acc is not None:
                acc_n.append((acc, blk["n"]))
        r["learner_error_accepted"] = (sum(x * n for x, n in acc_n) / sum(n for _, n in acc_n)
                                       if acc_n else None)
        pairs = defaultdict(list)
        for k, e in enumerate(E):
            pairs[f"{e['src']}>{e['dst']}"].append(k)
        r["by_symbol"] = {k: {"n": len(v), "sound_hit5": float(np.mean(p[v] <= T[5])),
                              "raw_auc": float("nan") if tone else _auc(-llr[v], -nat_llr)}
                          for k, v in sorted(pairs.items(), key=lambda t: -len(t[1]))}
        r["verdict"] = verdict(r["sound_hit5"])
        acc_dirs = [dn for dn, d in r["directions"].items() if d["verdict_today"].startswith("not repr")]
        if acc_dirs and len(acc_dirs) == len(r["directions"]):
            r["verdict_today"] = "not representable (accepted as alternative)"
        elif acc_dirs:
            r["verdict_today"] = f"{r['verdict']}; not representable: {', '.join(acc_dirs)}"
        else:
            r["verdict_today"] = r["verdict"]
        res["contrasts"][c.name] = r
    res["not_representable"] = NOT_REPRESENTABLE.get(lang, [])
    res["duration_feasibility"] = duration_feasibility(lang, clips)
    res["native_alt_usage"] = native_alt_usage(lang, clips)
    res["aspiration_feasibility"] = aspiration_feasibility(lang, rows["asp"])
    return res


def probe_pairs() -> dict:
    """What the front-end emits for both members of known minimal pairs."""
    from pronunciada.text import get_g2p

    from .contrasts import NOT_REPRESENTABLE

    extra = {"ja": [["おばさん", "おばあさん"], ["きて", "きって"]], "it": [["fato", "fatto"]],
             "de": [["Miete", "Mitte"], ["Rad", "Rat"]], "zh": [["怕", "爸"], ["山", "三"]],
             "es": [["pero", "perro"]], "pt": [["lã", "lá"], ["avô", "avó"], ["caro", "carro"]],
             "fr": [["vu", "vous"], ["bon", "beau"]], "ko": [["탈", "달"]], "en": [["sheep", "ship"]],
             "ru": [["брат", "брать"], ["мат", "мать"], ["рад", "ряд"], ["мол", "мёл"], ["лук", "люк"]]}
    out = {}
    for lang in LANGS:
        g = get_g2p(lang)
        pairs = [x["pair"] for x in NOT_REPRESENTABLE.get(lang, [])] + extra.get(lang, [])
        out[lang] = {}
        for pr in pairs:
            for wtxt in pr:
                r = g(wtxt)
                out[lang][wtxt] = " ".join(p.ipa + ("{" + ",".join(a or "∅" for a, _ in p.alts) + "}"
                                                    if p.alts else "") for p in r.phones)
    return out


# ----------------------------------------------------------------------------- CLI


def _job(args):
    run_lang(*args)
    return args[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "asp", "report"])
    ap.add_argument("--lang", default="all")
    ap.add_argument("--clips", type=int, default=250)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/tables/contrast_audit.json")
    a = ap.parse_args()
    langs = list(LANGS) if a.lang == "all" else a.lang.split(",")
    if a.cmd == "asp":
        for lg in langs:
            run_asp(lg, a.clips, a.threads)
        return
    if a.cmd == "run":
        workers = min(a.workers, 2, len(langs))          # hard limit: 2 processes
        if workers <= 1:
            for lg in langs:
                run_lang(lg, a.clips, a.threads)
        else:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(workers) as ex:
                for lg in ex.map(_job, [(lg, a.clips, a.threads) for lg in langs]):
                    print("done", lg, flush=True)
        return
    import warnings

    warnings.simplefilter("ignore")
    out = {"protocol": {"data": "FLEURS validation (dev), native audio", "target_per_contrast": TARGET,
                        "per_clip_direction_cap": PER_CLIP, "clips_per_language": a.clips,
                        "scorer": "pronunciada.Scorer(lang) defaults", "seeds": "numpy default_rng(0) clip "
                        "choice; crc32(lang/contrast) site sampling"},
           "languages": {}, "front_end_probe": probe_pairs()}
    for lg in langs:
        rows = load(lg, a.clips)
        if not rows["clips"]:
            continue
        out["languages"][lg] = analyse(lg, rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(markdown_tables(out))


def _pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.0f} %"


def _f3(x) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def markdown_tables(out: dict) -> str:
    """One table per language (pooled over directions) plus a per-direction table."""
    lines = []
    for lg, r in out["languages"].items():
        th = r["thresholds"]
        lines += [f"\n### {lg} ({r['n_clips']} clips; flag thresholds at 5 % native FA: sound P ≤ "
                  f"{th['sound']['5']:.3f}, word P ≤ {th['word']['5']:.3f})\n" if isinstance(th['sound'], dict)
                  and '5' in th['sound'] else
                  f"\n### {lg} ({r['n_clips']} clips; flag thresholds at 5 % native FA: sound P ≤ "
                  f"{th['sound'][5]:.3f}, word P ≤ {th['word'][5]:.3f})\n",
                  "| contrast | n | sound hit@5 % | @10 % | word hit@5 % | @10 % | AUC | raw-evidence AUC "
                  "| P > 0.9 | front-end accepts the error | natives heard as B | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for cn, c in r["contrasts"].items():
            if not c.get("n"):
                lines.append(f"| {c['label']} | 0 | | | | | | | | | | no eligible position |")
                continue
            lines.append(
                f"| {c['label']} | {c['n']} | {_pct(c['sound_hit5'])} | {_pct(c['sound_hit10'])} | "
                f"{_pct(c['word_hit5'])} | {_pct(c['word_hit10'])} | {_f3(c['auc'])} | {_f3(c['raw_auc'])} | "
                f"{_pct(c['share_above_0.9'])} | {_pct(c['learner_error_accepted'])} | "
                f"{_pct(c['native_heard_as_partner'])} | {c['verdict_today']} |")
        for x in r.get("not_representable", []):
            lines.append(f"| {x['contrast']} | – | | | | | | | | | | not representable"
                         f"{'' if x['phonemic'] else ' (not phonemic)'} |")
        lines += ["", "| direction (said / expected) | n | sound hit@5 % | word hit@5 % | AUC | raw AUC | "
                  "front-end accepts this error | verdict |", "|---|---|---|---|---|---|---|---|"]
        for cn, c in r["contrasts"].items():
            for dn, d in c.get("directions", {}).items():
                lines.append(f"| {cn}: {dn} | {d['n']} | {_pct(d['sound_hit5'])} | {_pct(d['word_hit5'])} | "
                             f"{_f3(d['auc'])} | {_f3(d['raw_auc'])} | {_pct(d['learner_error_accepted'])} | {d['verdict_today']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
