"""OMPAL benchmark: French-L1 learners of Taiwan Mandarin, expert word / sentence marks
(bench/ompal_build.py), Pronunciada vs Azure Pronunciation Assessment.

    python -m bench.ompal_build --clone                      # once: data + manifest
    python -m bench.eval_ompal --steps ours,report [--split dev|test] [--force]
    python -m bench.eval_ompal --steps azure [--dry-run]     # TEST half only, cached, never re-sent

`ours` scores through the public API only (pronunciada.Scorer("zh").score(wav, text)), 3 worker
processes, one JSON per utterance under
    $PRONUNCIADA_CACHE/ompal/ours/<tone model>-<onnx sha1[:8]>__<pronunciada fingerprint>/
so it is idempotent: a re-run with nothing changed scores nothing. The tone model is the one the
package would load (PRONUNCIADA_TONE_MODEL if set, else pronunciada.prosody.TONE_MODEL); report
writes results/ompal.md, results/tables/ompal.json and results/tables/ompal_<tone model>.json.

`azure` sends the TEST recordings and their prompt text to Azure (zh-TW with the Traditional
prompt = PRIMARY, because the corpus is Taiwan Mandarin read from Traditional-script prompts;
zh-CN with the prompt converted to Simplified by OpenCC t2s, the converter pronunciada.text.zh
uses = SENSITIVITY). Responses are cached in $PRONUNCIADA_CACHE/azure/ompal-<locale>/<id>.json.

PRE-SPECIFIED METRICS (fixed 2026-11-16 before any system output on OMPAL was computed or seen)
Units: OMPAL word units (one Hanzi each; bench/ompal_build.unit_spans). Labels: the shipped
majority-of-three labels. "correct" = tone, consonant and vowel all marked correct.
Unit scores (higher = better):
  ours  word   = min WordResult.prob (default strictness) over our words overlapping the unit
        tone   = min SyllableResult.tone_prob over our syllables inside the unit  [diagnostic]
        sounds = min WordResult.sounds_prob over our words overlapping the unit   [diagnostic]
  azure word   = min word AccuracyScore/100 over Azure words overlapping the unit (character
                 alignment of bench.eval_swap.azure_map); Omission = 0.
  A unit no system word covers, an utterance the system failed on (Azure status != Success,
  our ValueError) -> 0, i.e. the system's lowest score; never dropped. Counts are reported.
Word level, TEST (all 21 speakers, natives included); learners-only as a secondary row:
  primary  AUC(any-wrong vs correct)
  (a)      AUC(tone-wrong-only vs correct): ours word, Azure word; ours tone [diagnostic]
  (b)      AUC(consonant-or-vowel-wrong vs correct): ours word, Azure word; ours sounds [diag.]
  hit rate at 5 % / 10 % false alarms, threshold from the correct units of (i) the native
  speakers, (ii) the learners; positives = learners' units of the class; ties at the threshold
  split fractionally (bench.eval_swap._hit_at_fa).
Sentence level, TEST learners (primary; natives are all 5/5/5 by construction, an
"incl. natives" row is secondary): Pearson and Spearman with the mean teacher score:
  accuracy: ours scores["accuracy"] vs Azure AccuracyScore
  fluency:  ours scores["fluency"]  vs Azure FluencyScore
  prosody:  ours scores["prosodic"] vs Azure PronScore (nearest available; Azure has no
            Mandarin prosody score)
  secondary: ours SentenceResult.prob and Azure PronScore vs teacher accuracy.
Uncertainty: 95 % percentile bootstrap, 2000 resamples, (1) over speakers, (2) over utterances;
paired bootstrap of ours - Azure on the same resamples, two-sided p = 2 min(P(d<=0), P(d>=0)).
DEV (28 learners): ours only, the same word metrics (learner FA only) and sentence correlations.
Nothing is tuned on either half.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from bench.metrics import pearson, roc_auc, spearman

ROOT = Path(__file__).resolve().parents[1]
DATASET = "ompal"
AZ_LOCALES = ("zh-TW", "zh-CN")          # [0] = primary
N_BOOT = 2000
MD_OUT = ROOT / "results" / "ompal.md"
JSON_OUT = ROOT / "results" / "tables" / "ompal.json"


# ============================================================================ helpers

def cache_root() -> Path:
    from bench.data.common import cache_root as cr
    return cr()


def tone_model() -> tuple[str, str]:
    """(name, onnx sha1[:8]) of the tone model pronunciada would load now."""
    from pronunciada import prosody

    stem = os.environ.get("PRONUNCIADA_TONE_MODEL", prosody.TONE_MODEL)
    d = prosody.DATA
    if os.sep in stem:
        d, stem = Path(stem).parent, Path(stem).name
    p = Path(d) / f"{stem}.onnx"
    sha = hashlib.sha1(p.read_bytes()).hexdigest()[:8] if p.exists() else "missing"
    return stem, sha


def fingerprint() -> str:
    h = hashlib.sha256()
    root = ROOT / "pronunciada"
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def records(split: str | None = None) -> list[dict]:
    from bench.data import load
    return load(DATASET, split=split)


def overlap(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


def word_char_spans(text: str, words: list[str]) -> list[tuple[int, int] | None]:
    """Char span of each system word in `text`, matched left to right (None if not found)."""
    out, pos = [], 0
    for w in words:
        k = text.find(w, pos) if w else -1
        if k < 0:
            out.append(None)
        else:
            out.append((k, k + len(w)))
            pos = k + len(w)
    return out


def syllable_char_spans(span: tuple[int, int], n_syl: int) -> list[tuple[int, int]]:
    """One char per syllable when the counts agree, else every syllable gets the whole word."""
    a, b = span
    if n_syl == b - a:
        return [(a + i, a + i + 1) for i in range(n_syl)]
    return [(a, b)] * n_syl


def unit_score(span, items, missing: float = 0.0) -> float:
    """min score over items (a, b, s) overlapping span; s None ignored; nothing -> `missing`."""
    ss = [s for a, b, s in items if s is not None and overlap(span[0], span[1], a, b)]
    return float(min(ss)) if ss else missing


# ============================================================================ ours

_SC = None


def _init() -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    global _SC
    import pronunciada

    _SC = pronunciada.Scorer("zh", threads=1)


def summarize(res, text: str) -> dict:
    spans = word_char_spans(text, [w.text for w in res.words])
    words, syls = [], []
    for w, sp in zip(res.words, spans):
        if sp is None:
            continue
        words.append([sp[0], sp[1], float(w.prob), None if w.sounds_prob is None else float(w.sounds_prob),
                      None if w.tone_prob is None else float(w.tone_prob)])
        for (a, b), s in zip(syllable_char_spans(sp, len(w.syllables)), w.syllables):
            syls.append([a, b, None if s.tone_prob is None else float(s.tone_prob)])
    sc = dict(res.scores or {})
    return {"status": "Success", "words": words, "syllables": syls,
            "n_words_unplaced": sum(s is None for s in spans),
            "sentence": {"prob": float(res.prob),
                         **{k: (None if sc.get(k) is None else float(sc[k]))
                            for k in ("accuracy", "fluency", "prosodic", "total")}},
            "warnings": list(res.warnings)}


def _score_one(rec: dict) -> tuple[str, dict]:
    t = time.perf_counter()
    try:
        o = summarize(_SC.score(rec["audio"], rec["text"]), rec["text"])
    except ValueError as e:
        o = {"status": "error", "error": str(e)[:200]}
    o["seconds"] = round(time.perf_counter() - t, 3)
    return rec["id"], o


def ours_dir(tm=None, fp=None) -> Path:
    name, sha = tm or tone_model()
    return cache_root() / DATASET / "ours" / f"{name}-{sha}__{fp or fingerprint()}"


def run_ours(split: str | None, force: bool = False, workers: int = 3) -> Path:
    tm, fp = tone_model(), fingerprint()
    d = ours_dir(tm, fp)
    d.mkdir(parents=True, exist_ok=True)
    recs = records(split)
    todo = [r for r in recs if force or not (d / f"{r['id']}.json").exists()]
    print(f"ours: tone model {tm[0]} ({tm[1]}), pronunciada {fp}: {len(recs)} utts, "
          f"{len(recs) - len(todo)} cached, {len(todo)} to score", flush=True)
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(min(workers, 3), initializer=_init) as ex:
            for uid, o in ex.map(_score_one, todo, chunksize=8):
                tmp = d / f"{uid}.json.tmp"
                tmp.write_text(json.dumps(o, ensure_ascii=False))
                os.replace(tmp, d / f"{uid}.json")
        print(f"  scored {len(todo)} in {time.time() - t0:.0f} s", flush=True)
    (d / "_meta.json").write_text(json.dumps({"tone_model": tm[0], "tone_model_sha1": tm[1],
                                              "fingerprint": fp, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    latest = cache_root() / DATASET / "ours" / f"latest_{tm[0]}.json"
    latest.write_text(json.dumps({"dir": d.name}))
    return d


def ours_outputs(recs: list[dict]) -> tuple[dict, dict]:
    """Outputs of the latest ours run for the current tone model (a note if pronunciada/ changed)."""
    tm = tone_model()
    d = ours_dir(tm)
    latest = cache_root() / DATASET / "ours" / f"latest_{tm[0]}.json"
    if not d.exists() and latest.exists():
        d = d.parent / json.loads(latest.read_text())["dir"]
        print(f"note: pronunciada/ or the tone model changed since the last ours run; reporting {d.name}")
    meta = json.loads((d / "_meta.json").read_text()) if (d / "_meta.json").exists() else {}
    out = {}
    for r in recs:
        p = d / f"{r['id']}.json"
        if p.exists():
            out[r["id"]] = json.loads(p.read_text())
    meta["dir"] = d.name
    return out, meta


# ============================================================================ azure

class _TimedSession:
    def __init__(self, s):
        self.s, self.last = s, None

    def post(self, *a, **k):
        t = time.perf_counter()
        try:
            return self.s.post(*a, **k)
        finally:
            self.last = time.perf_counter() - t


_T2S = None


def azure_text(text: str, locale: str) -> str:
    global _T2S
    if locale == "zh-CN":
        if _T2S is None:
            from opencc import OpenCC
            _T2S = OpenCC("t2s")
        return _T2S.convert(text)
    return text


def az_dir(locale: str) -> Path:
    from bench.azure_pa import cache_dir
    return cache_dir(f"{DATASET}-{locale}")


def run_azure(locales=AZ_LOCALES, dry_run: bool = False, workers: int = 4, max_tries: int = 12) -> None:
    from bench import azure_pa as AZ

    recs = [r for r in records("test") if r["dur"] <= AZ.MAX_SECONDS]
    jobs = [(loc, r) for loc in locales for r in recs if not (az_dir(loc) / f"{r['id']}.json").exists()]
    print(f"{DATASET} TEST: {len(records('test'))} clips ({len(recs)} <= 30 s); to send: {len(jobs)} requests, "
          f"{sum(r['dur'] for _, r in jobs) / 3600:.2f} audio-h", flush=True)
    for loc in locales:
        js = [r for l2, r in jobs if l2 == loc]
        print(f"  {loc}: {len(js)} requests, {sum(r['dur'] for r in js) / 3600:.2f} h")
    if dry_run or not jobs:
        return
    key, region = AZ.credentials()
    import requests

    for loc in locales:
        az_dir(loc).mkdir(parents=True, exist_ok=True)
    local = threading.local()

    def one(loc: str, r: dict) -> str:
        if not hasattr(local, "s"):
            local.s = requests.Session()
        text = azure_text(r["text"], loc)
        params = AZ.pa_params(text, loc)
        audio = AZ.wav_bytes(r["audio"])
        rnd = random.Random(r["id"] + loc)
        for attempt in range(max_tries):
            ts = _TimedSession(local.s)
            try:
                resp = AZ.send(ts, key, region, loc, params, audio, max_tries=1)
            except RuntimeError as e:
                if str(e).startswith("Azure returned HTTP"):
                    return f"fail {loc} {r['id']}: {str(e)[:120]}"
                time.sleep(min(300.0, 5.0 * 2 ** attempt) * (0.75 + 0.5 * rnd.random()))
                continue
            rec = {"id": r["id"], "locale": loc, "params": params,
                   "audio_sha1": hashlib.sha1(Path(r["audio"]).read_bytes()).hexdigest(),
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "latency_s": ts.last,
                   "attempts": attempt + 1, "audio_dur": r["dur"], "response": resp}
            out = az_dir(loc)
            tmp = out / f"{r['id']}.json.tmp{threading.get_ident()}"
            tmp.write_text(json.dumps(rec, ensure_ascii=False))
            os.replace(tmp, out / f"{r['id']}.json")
            return "ok"
        return f"fail {loc} {r['id']}: giving up after {max_tries} tries"

    n_ok = n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, loc, r) for loc, r in jobs]
        for i, f in enumerate(as_completed(futs)):
            res = f.result()
            if res == "ok":
                n_ok += 1
            else:
                n_fail += 1
                print(res, flush=True)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)} ({n_fail} failed, {time.time() - t0:.0f} s)", flush=True)
    print(f"done: {n_ok} ok, {n_fail} failed, {time.time() - t0:.0f} s", flush=True)


def azure_outputs(recs: list[dict], locale: str) -> dict:
    from bench.azure_pa import parse_response
    from bench.eval_swap import azure_map

    d = az_dir(locale)
    outs = {}
    for r in recs:
        p = d / f"{r['id']}.json"
        if not p.exists():
            continue
        raw = json.loads(p.read_text())
        parsed = parse_response(raw["response"])
        o = {"status": parsed["status"], "latency_s": raw.get("latency_s"), "audio_dur": r["dur"]}
        if parsed["status"] == "Success":
            text = azure_text(r["text"], locale)
            assert len(text) == len(r["text"])          # t2s is char-for-char on this corpus
            m = azure_map(text, "zh", parsed)
            o["words"] = [list(w) for w in m["words"]]
            o["unmapped"] = [list(u) for u in m["unmapped"]]
            az = [w for w in parsed["words"] if w["error_type"] != "Insertion"]
            o["n_az_words"] = len(az)
            o["n_az_words_single_char"] = sum(len(w["word"] or "") == 1 for w in az)
            s = parsed["sentence"] or {}
            o["sentence"] = {k: (None if s.get(k) is None else s[k] / 100.0)
                             for k in ("AccuracyScore", "FluencyScore", "PronScore", "CompletenessScore")}
        outs[r["id"]] = o
    return outs


def azure_usage(recs: list[dict]) -> dict:
    out = {}
    for loc in AZ_LOCALES:
        d = az_dir(loc)
        n, secs, lat, statuses = 0, 0.0, [], defaultdict(int)
        for r in recs:
            p = d / f"{r['id']}.json"
            if p.exists():
                raw = json.loads(p.read_text())
                n += 1
                secs += r["dur"]
                statuses[str(raw["response"].get("RecognitionStatus"))] += 1
                if raw.get("latency_s") is not None:
                    lat.append(raw["latency_s"])
        out[loc] = {"requests_cached": n, "audio_hours": round(secs / 3600, 3), "statuses": dict(statuses),
                    "median_latency_s": float(np.median(lat)) if lat else None}
    return out


# ============================================================================ tables

SENT_PAIRS = [  # (row, teacher key, ours key, azure key, note)
    ("accuracy", "accuracy", "accuracy", "AccuracyScore", ""),
    ("fluency", "fluency", "fluency", "FluencyScore", ""),
    ("prosody", "prosody", "prosodic", "PronScore", "nearest available; Azure has no Mandarin prosody score"),
    ("accuracy (secondary: ours sentence prob, Azure PronScore)", "accuracy", "prob", "PronScore", "secondary"),
]


def unit_table(recs: list[dict], ours: dict, azs: dict[str, dict]) -> dict:
    """Flat per-unit arrays: labels, speaker / utterance / native, and every system's unit score."""
    cols = defaultdict(list)
    cnt = defaultdict(int)
    for r in recs:
        lab = r["labels"]
        if lab["word_level_dropped"]:
            cnt["utts_word_level_dropped"] += 1
            continue
        o = ours.get(r["id"]) or {"status": "missing"}
        ok = o.get("status") == "Success"
        cnt["ours_failed_utts"] += 0 if ok else 1
        az_ok = {}
        for loc, A in azs.items():
            a = A.get(r["id"]) or {"status": "missing"}
            az_ok[loc] = a if a.get("status") == "Success" else None
            cnt[f"azure_{loc}_failed_utts"] += 0 if az_ok[loc] else 1
        for u in lab["units"]:
            if u["span"] is None:
                cnt["units_not_in_prompt"] += 1
                continue
            sp = tuple(u["span"])
            cols["utt"].append(r["id"])
            cols["spk"].append(r["speaker"])
            cols["native"].append(lab["native"])
            for k in ("any_wrong", "tone_only", "seg_wrong", "correct", "tone", "consonant", "vowel"):
                cols[k].append(u[k])
            if ok:
                w = [(a, b, s) for a, b, s, _, _ in o["words"]]
                if not any(overlap(*sp, a, b) for a, b, _ in w):
                    cnt["ours_units_uncovered"] += 1
                cols["ours_word"].append(unit_score(sp, w))
                cols["ours_sounds"].append(unit_score(sp, [(a, b, s) for a, b, _, s, _ in o["words"]]))
                ts = unit_score(sp, [tuple(x) for x in o["syllables"]], missing=np.nan)
                if not np.isfinite(ts):
                    ts = unit_score(sp, [(a, b, t) for a, b, _, _, t in o["words"]], missing=np.nan)
                    cnt["ours_units_tone_from_word"] += int(np.isfinite(ts))
                cols["ours_tone"].append(ts)
            else:
                for k in ("ours_word", "ours_sounds", "ours_tone"):
                    cols[k].append(0.0)
            for loc, a in az_ok.items():
                if a is None:
                    cols[f"az_{loc}"].append(0.0)
                    continue
                items = [tuple(x) for x in a["words"]]
                if not any(overlap(*sp, x[0], x[1]) for x in items):
                    cnt[f"azure_{loc}_units_uncovered"] += 1
                cols[f"az_{loc}"].append(unit_score(sp, items))
    T = {k: np.asarray(v) for k, v in cols.items()}
    return {"cols": T, "counts": dict(cnt)}


def sent_table(recs: list[dict], ours: dict, azs: dict[str, dict]) -> dict:
    cols = defaultdict(list)
    for r in recs:
        lab = r["labels"]
        cols["utt"].append(r["id"])
        cols["spk"].append(r["speaker"])
        cols["native"].append(lab["native"])
        for k, v in lab["sentence"].items():
            cols[f"t_{k}"].append(v)
        o = ours.get(r["id"]) or {}
        s = o.get("sentence") if o.get("status") == "Success" else None
        for k in ("prob", "accuracy", "fluency", "prosodic"):
            cols[f"ours_{k}"].append(0.0 if not s or s.get(k) is None else s[k])
        for loc, A in azs.items():
            a = A.get(r["id"]) or {}
            s = a.get("sentence") if a.get("status") == "Success" else None
            for k in ("AccuracyScore", "FluencyScore", "PronScore"):
                cols[f"az_{loc}_{k}"].append(0.0 if not s or s.get(k) is None else s[k])
    return {k: np.asarray(v) for k, v in cols.items()}


# ============================================================================ bootstrap

def boot_indices(clusters: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> list[np.ndarray]:
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
            for _ in range(n_boot)]


def _q(v) -> list[float]:
    v = np.asarray([x for x in v if np.isfinite(x)])
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [np.nan, np.nan]


def stat_with_ci(f, boots: dict[str, list[np.ndarray]], n: int, ref=None) -> dict:
    """f(idx) -> float for the system; ref(idx) -> float for the comparator (Azure) if given."""
    full = np.arange(n)
    out = {"value": float(f(full))}
    if ref is not None:
        out["azure"] = float(ref(full))
        out["delta"] = out["value"] - out["azure"]
    for name, B in boots.items():
        v = np.array([f(i) for i in B])
        out[f"ci_{name}"] = _q(v)
        if ref is not None:
            w = np.array([ref(i) for i in B])
            d = (v - w)[np.isfinite(v - w)]
            out[f"azure_ci_{name}"] = _q(w)
            out[f"delta_ci_{name}"] = _q(d)
            out[f"p_{name}"] = float(min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1),
                                                      (np.sum(d >= 0) + 1) / (len(d) + 1))))
    return out


def auc_fn(pos: np.ndarray, neg: np.ndarray, score: np.ndarray):
    """AUC for detecting pos (vs neg) with score = goodness (low = flagged), on an index subset."""
    def f(idx):
        p, q = pos[idx], neg[idx]
        sel = p | q
        if p.sum() == 0 or q.sum() == 0:
            return np.nan
        s = score[idx][sel]
        ok = np.isfinite(s)
        y = p[sel][ok]
        if y.min() == y.max():
            return np.nan
        return roc_auc(y.astype(int), -s[ok])
    return f


def hit_fn(pos: np.ndarray, neg: np.ndarray, score: np.ndarray, alpha: float):
    from bench.eval_swap import _hit_at_fa

    def f(idx):
        s = score[idx]
        a, b = s[pos[idx] & np.isfinite(s)], s[neg[idx] & np.isfinite(s)]
        return _hit_at_fa(b, a, alpha) if len(a) and len(b) else np.nan
    return f


def corr_fn(y: np.ndarray, s: np.ndarray, kind: str):
    m = pearson if kind == "pearson" else spearman

    def f(idx):
        yy, ss = y[idx], s[idx]
        if np.std(yy) == 0 or np.std(ss) == 0:
            return np.nan
        return m(ss, yy)
    return f


# ============================================================================ evaluation

CLASSES = [("primary: any error", "any_wrong", "ours_word", None),
           ("(a) tone-only error", "tone_only", "ours_word", "ours_tone"),
           ("(b) consonant or vowel error", "seg_wrong", "ours_word", "ours_sounds")]


def word_eval(U: dict, azure_locs: list[str], learners_only: bool = False) -> dict:
    c = U["cols"]
    keep = ~c["native"] if learners_only else np.ones(len(c["utt"]), bool)
    c = {k: v[keep] for k, v in c.items()}
    n = len(c["utt"])
    boots = {"speaker": boot_indices(c["spk"], seed=0), "utterance": boot_indices(c["utt"], seed=1)}
    neg = c["correct"].astype(bool)
    out = {"n_units": n, "n_correct": int(neg.sum())}
    for name, lab, sys_col, diag in CLASSES:
        pos = c[lab].astype(bool)
        row = {"n_pos": int(pos.sum())}
        ref = auc_fn(pos, neg, c[f"az_{azure_locs[0]}"]) if azure_locs else None
        row["ours"] = stat_with_ci(auc_fn(pos, neg, c[sys_col]), boots, n, ref)
        for loc in azure_locs[1:]:
            row[f"ours_vs_{loc}"] = stat_with_ci(auc_fn(pos, neg, c[sys_col]), boots, n,
                                                 auc_fn(pos, neg, c[f"az_{loc}"]))
        if diag:
            row["ours_diag"] = stat_with_ci(auc_fn(pos, neg, c[diag]), boots, n)
            row["diag_channel"] = diag
        out[name] = row
    return out


def hit_eval(U: dict, azure_locs: list[str]) -> dict:
    c = U["cols"]
    n = len(c["utt"])
    boots = {"speaker": boot_indices(c["spk"], seed=2)}
    nat = c["native"].astype(bool)
    corr = c["correct"].astype(bool)
    out = {}
    for fa_name, negmask in (("native", corr & nat), ("learner", corr & ~nat)):
        if negmask.sum() == 0:
            continue
        for name, lab, sys_col, diag in CLASSES:
            pos = c[lab].astype(bool) & ~nat
            for alpha in (0.05, 0.10):
                key = f"{name} | FA on {fa_name} correct units | {int(alpha * 100)} %"
                row = {"n_pos": int(pos.sum()), "n_neg": int(negmask.sum())}
                ref = hit_fn(pos, negmask, c[f"az_{azure_locs[0]}"], alpha) if azure_locs else None
                row["ours"] = stat_with_ci(hit_fn(pos, negmask, c[sys_col], alpha), boots, n, ref)
                for loc in azure_locs[1:]:
                    row[f"azure_{loc}"] = float(hit_fn(pos, negmask, c[f"az_{loc}"], alpha)(np.arange(n)))
                if diag:
                    row["ours_diag"] = float(hit_fn(pos, negmask, c[diag], alpha)(np.arange(n)))
                out[key] = row
    return out


def sent_eval(S: dict, azure_locs: list[str], include_natives: bool = False) -> dict:
    keep = np.ones(len(S["utt"]), bool) if include_natives else ~S["native"].astype(bool)
    c = {k: v[keep] for k, v in S.items()}
    n = len(c["utt"])
    boots = {"speaker": boot_indices(c["spk"], seed=3), "utterance": boot_indices(c["utt"], seed=4)}
    out = {"n_utts": n, "n_speakers": int(len(set(c["spk"])))}
    for row, tk, ok, ak, note in SENT_PAIRS:
        y = c[f"t_{tk}"].astype(float)
        r = {"note": note}
        for kind in ("pearson", "spearman"):
            ref = corr_fn(y, c[f"az_{azure_locs[0]}_{ak}"].astype(float), kind) if azure_locs else None
            r[kind] = stat_with_ci(corr_fn(y, c[f"ours_{ok}"].astype(float), kind), boots, n, ref)
            for loc in azure_locs[1:]:
                r[f"{kind}_vs_{loc}"] = stat_with_ci(corr_fn(y, c[f"ours_{ok}"].astype(float), kind), boots, n,
                                                     corr_fn(y, c[f"az_{loc}_{ak}"].astype(float), kind))
        out[row] = r
    return out


def label_stats(recs: list[dict]) -> dict:
    out = {}
    for split in ("test", "dev"):
        for grp, nat in (("learners", False), ("natives", True)):
            us = [u for r in recs if r["split"] == split and r["labels"]["native"] == nat
                  and not r["labels"]["word_level_dropped"] for u in r["labels"]["units"] if u["span"]]
            if not us:
                continue
            n = len(us)
            out[f"{split}/{grp}"] = {
                "utterances": sum(r["split"] == split and r["labels"]["native"] == nat for r in recs),
                "speakers": len({r["speaker"] for r in recs if r["split"] == split and r["labels"]["native"] == nat}),
                "units": n, **{f"share_{k}": sum(u[k] for u in us) / n
                               for k in ("any_wrong", "tone_only", "seg_wrong")},
                "share_tone_wrong": sum(1 - u["tone"] for u in us) / n,
                "share_consonant_wrong": sum(1 - u["consonant"] for u in us) / n,
                "share_vowel_wrong": sum(1 - u["vowel"] for u in us) / n}
    return out


def evaluate(force_locs=None) -> dict:
    recs_all = records()
    test, dev = [r for r in recs_all if r["split"] == "test"], [r for r in recs_all if r["split"] == "dev"]
    ours, meta = ours_outputs(recs_all)
    azs = {loc: azure_outputs(test, loc) for loc in (force_locs or AZ_LOCALES)}
    locs = [loc for loc in azs if len(azs[loc]) == len(test)]
    for loc in azs:
        if loc not in locs:
            print(f"note: Azure {loc} has {len(azs[loc])}/{len(test)} TEST responses cached; not reported")
    azs = {k: azs[k] for k in locs}
    R = {"dataset": DATASET, "tone_model": meta.get("tone_model"), "tone_model_sha1": meta.get("tone_model_sha1"),
         "pronunciada_fingerprint": meta.get("fingerprint"), "ours_cache": meta.get("dir"),
         "azure_locales": locs, "azure_primary": locs[0] if locs else None,
         "labels": label_stats(recs_all), "n_boot": N_BOOT}
    missing = [r["id"] for r in recs_all if r["id"] not in ours]
    if missing:
        print(f"note: {len(missing)} utterances have no ours output (scored as failures)")
    R["ours_missing"] = len(missing)
    R["ours_errors"] = sorted(k for k, o in ours.items() if o.get("status") != "Success")
    for name, recs, A in (("test", test, azs), ("dev", dev, {})):
        U = unit_table(recs, ours, A)
        S = sent_table(recs, ours, A)
        L = list(A)
        R[name] = {"counts": U["counts"], "word": word_eval(U, L), "word_learners_only": word_eval(U, L, True),
                   "hits": hit_eval(U, L), "sentence": sent_eval(S, L)}
        if name == "test":
            R[name]["sentence_incl_natives"] = sent_eval(S, L, True)
            R[name]["azure_single_char_word_share"] = {
                loc: float(sum(o.get("n_az_words_single_char", 0) for o in A[loc].values())
                           / max(1, sum(o.get("n_az_words", 0) for o in A[loc].values()))) for loc in L}
    R["azure_usage"] = azure_usage(test)
    return R


# ============================================================================ report

def _ci(x: dict, key: str = "value", ci: str = "ci_speaker", pct: bool = False) -> str:
    v = x.get(key)
    if v is None or not np.isfinite(v):
        return "–"
    sc, fmt = (100, "{:.1f}") if pct else (1, "{:.3f}")
    s = fmt.format(v * sc)
    c = x.get(ci)
    if c and np.isfinite(c[0]):
        s += f" [{fmt.format(c[0] * sc)}, {fmt.format(c[1] * sc)}]"
    return s


def _pv(p) -> str:
    return "–" if p is None else (f"{p:.3f}" if p >= 0.001 else "< 0.001")


def _word_rows(W: dict, has_az: bool, loc2: str | None) -> list[str]:
    L = ["| class | n wrong / n correct | Pronunciada word [speaker CI] | [utterance CI] | "
         + ("Azure word [speaker CI] | [utterance CI] | Δ ours − Azure | p (speaker) | p (utterance) | "
            + (f"Azure {loc2} (sensitivity) | " if loc2 else "") if has_az else "")
         + "our diagnostic channel (Azure has no equivalent) [speaker CI] |",
         "|---|---|---|---|" + ("---|---|---|---|---|" + ("---|" if loc2 else "") if has_az else "") + "---|"]
    for name, _, _, _ in CLASSES:
        r = W[name]
        o = r["ours"]
        cells = [name, f"{r['n_pos']} / {W['n_correct']}", _ci(o), _ci(o, ci="ci_utterance")]
        if has_az:
            cells += [_ci(o, "azure", "azure_ci_speaker"), _ci(o, "azure", "azure_ci_utterance"),
                      f"{o['delta']:+.3f}", _pv(o.get("p_speaker")), _pv(o.get("p_utterance"))]
            if loc2:
                o2 = r[f"ours_vs_{loc2}"]
                cells.append(f"{o2['azure']:.3f} (Δ {o2['delta']:+.3f}, p {_pv(o2.get('p_speaker'))})")
        cells.append(f"{r['diag_channel'].replace('ours_', '')}: {_ci(r['ours_diag'])}" if "ours_diag" in r else "–")
        L.append("| " + " | ".join(cells) + " |")
    return L


def _sent_rows(Sx: dict, has_az: bool, loc2: str | None) -> list[str]:
    L = ["| teacher score | system scores | Pronunciada Pearson [speaker CI] [utterance CI] | "
         + ("Azure Pearson [speaker CI] | Δ, p (speaker / utterance) | " if has_az else "")
         + "Pronunciada Spearman [speaker CI] | " + ("Azure Spearman [speaker CI] | Δ, p (speaker) | " if has_az else "")
         + (f"Azure {loc2} Pearson / Spearman (sensitivity) |" if loc2 else ""),
         "|---|---|---|" + ("---|---|" if has_az else "") + "---|" + ("---|---|" if has_az else "")
         + ("---|" if loc2 else "")]
    for row, tk, ok, ak, note in SENT_PAIRS:
        r = Sx[row]
        p, s = r["pearson"], r["spearman"]
        sysd = f"ours `{ok}`" + (f" / Azure `{ak}`" if has_az else "") + (f" ({note})" if has_az and note and note != "secondary" else "")
        cells = [row, sysd, f"{_ci(p)} {_ci(p, ci='ci_utterance').split(' ', 1)[-1] if np.isfinite(p['value']) else ''}"]
        if has_az:
            cells += [_ci(p, "azure", "azure_ci_speaker"),
                      f"{p['delta']:+.3f}, p {_pv(p.get('p_speaker'))} / {_pv(p.get('p_utterance'))}"]
        cells.append(_ci(s))
        if has_az:
            cells += [_ci(s, "azure", "azure_ci_speaker"), f"{s['delta']:+.3f}, p {_pv(s.get('p_speaker'))}"]
        if loc2:
            cells.append(f"{r[f'pearson_vs_{loc2}']['azure']:.3f} / {r[f'spearman_vs_{loc2}']['azure']:.3f}")
        L.append("| " + " | ".join(cells) + " |")
    return L


def _hit_rows(H: dict, has_az: bool, loc2: str | None) -> list[str]:
    L = ["| class | false alarms set on | FA rate | Pronunciada word [speaker CI] | "
         + ("Azure word [speaker CI] | p (speaker) | " + (f"Azure {loc2} | " if loc2 else "") if has_az else "")
         + "our diagnostic channel |",
         "|---|---|---|---|" + ("---|---|" + ("---|" if loc2 else "") if has_az else "") + "---|"]
    for key, r in H.items():
        name, fa, rate = [x.strip() for x in key.split("|")]
        o = r["ours"]
        cells = [name, fa.replace("FA on ", "") + f" (n = {r['n_neg']})", rate, _ci(o, pct=True)]
        if has_az:
            cells += [_ci(o, "azure", "azure_ci_speaker", pct=True), _pv(o.get("p_speaker"))]
            if loc2:
                cells.append(f"{100 * r[f'azure_{loc2}']:.1f}")
        cells.append(f"{100 * r['ours_diag']:.1f}" if "ours_diag" in r else "–")
        L.append("| " + " | ".join(cells) + " |")
    return L


def verdicts(R: dict) -> list[str]:
    out = []
    T = R["test"]
    loc = R["azure_primary"]
    if not loc:
        return out

    def say(label, o):
        d, p = o["delta"], o.get("p_speaker")
        who = "Pronunciada ahead" if d > 0 else "Azure ahead"
        sig = "significant" if p is not None and p < 0.05 else "not significant"
        return f"- {label}: {o['value']:.3f} vs {o['azure']:.3f} — {who} by {abs(d):.3f}, {sig} (speaker-bootstrap p {_pv(p) if _pv(p).startswith("<") else "= " + _pv(p)})."
    for name, *_ in CLASSES:
        out.append(say(f"Word AUC, {name}", T["word"][name]["ours"]))
    for key, r in T["hits"].items():
        name, fa, rate = [x.strip() for x in key.split("|")]
        out.append(say(f"Hit rate, {name}, {rate} {fa.replace('FA on ', 'false alarms on ')}", r["ours"]))
    for row, *_ in SENT_PAIRS:
        out.append(say(f"Sentence Pearson, {row}", T["sentence"][row]["pearson"]))
    return out


def markdown(R: dict) -> str:
    T, D = R["test"], R["dev"]
    locs = R["azure_locales"]
    has_az = bool(locs)
    loc2 = locs[1] if len(locs) > 1 else None
    lab = R["labels"]
    ag = json.loads((ROOT / "results" / "tables" / "ompal_manifest.json").read_text())["agreement"]
    from bench.ompal_build import CITATION, LICENCE

    L = [f"# OMPAL: French-speaking learners of Taiwan Mandarin, Pronunciada vs Azure\n",
         f"Generated by `python -m bench.eval_ompal --steps ours,report` (tone model **{R['tone_model']}**, "
         f"onnx sha1 {R['tone_model_sha1']}, pronunciada fingerprint {R['pronunciada_fingerprint']}). "
         f"Azure: {', '.join(locs) if locs else 'not run'}"
         + (f"; **{locs[0]} is the pre-declared primary**, {loc2} is a sensitivity check." if has_az else ".") + "\n",
         "## How to read this\n",
         "This is the first expert-marked learner benchmark outside English for this project. It is small and "
         "narrow: one first language (French), 18 test learners (first- and second-year students; the shipped test "
         "files contain 18 distinct learners, not 20), and Taiwan Mandarin read from Traditional-script prompts, "
         "marked by Taiwanese teachers — so Taiwan norms (e.g. less retroflexion, fewer neutral tones, "
         "different readings of some characters) are the reference, while Pronunciada's pronunciations come from a "
         "Mainland-oriented G2P and its tone model was trained on Mainland and Taiwan native speech. Speaker-level "
         "confidence intervals are wide because there are only 21 test speakers (18 learners); the utterance-level "
         "column is narrower but treats one speaker's utterances as independent, which they are not. Nothing in "
         "Pronunciada has ever seen this data, and nothing was tuned on DEV or TEST. The metric definitions below "
         "were fixed before any system output on OMPAL was computed. Word units are single characters but both "
         "systems score whole words: a two-character word gets one score for both characters, for both systems.\n",
         "## Pre-specified metrics (fixed 2026-11-16, before any result was seen)\n",
         "- Units: OMPAL word units = one Hanzi each; labels = the shipped majority of three raters; "
         "\"correct\" = tone, consonant and vowel all marked correct.",
         "- Unit score = lowest score over the system's words overlapping the unit (character spans, as in "
         "`bench/eval_swap.py`). Ours: `word.prob` at default strictness; diagnostic channels: lowest syllable "
         "`tone_prob` inside the unit, lowest `word.sounds_prob`. Azure: word `AccuracyScore` (Omission = 0).",
         "- A unit no system word covers, and every unit of an utterance the system failed on, gets that system's "
         "lowest score (0); nothing is dropped for a system failure.",
         "- Word level, TEST all speakers (natives included): primary AUC any-error vs correct; (a) tone-only "
         "error vs correct; (b) consonant-or-vowel error vs correct; learners-only as a secondary table; hit rate "
         "at 5 % / 10 % false alarms with the threshold set on the correct units of the natives and, separately, "
         "of the learners (positives = learners' units).",
         "- Sentence level, TEST learners (natives are all 5/5/5, so they are a secondary row): Pearson and "
         "Spearman with the mean teacher score — accuracy (ours `accuracy` / Azure AccuracyScore), fluency "
         "(ours `fluency` / Azure FluencyScore), prosody (ours `prosodic` / Azure PronScore, the nearest "
         "available: Azure has no Mandarin prosody score); secondary: ours sentence `prob` / Azure PronScore vs "
         "teacher accuracy.",
         f"- 95 % percentile bootstrap, {R['n_boot']} resamples, over speakers and over utterances; Δ = ours − "
         "Azure on the same resamples, two-sided paired-bootstrap p.\n",
         "## TEST: word level (all 21 speakers)\n"]
    L += _word_rows(T["word"], has_az, loc2)
    L += ["\n### TEST: word level, learners only (secondary)\n"]
    L += _word_rows(T["word_learners_only"], has_az, loc2)
    L += ["\n### TEST: hit rate at fixed false-alarm rates (positives = learners' units; speaker CI)\n"]
    L += _hit_rows(T["hits"], has_az, loc2)
    L += ["\n## TEST: sentence level, learners (primary)\n"]
    L += _sent_rows(T["sentence"], has_az, loc2)
    L += [f"\nn = {T['sentence']['n_utts']} utterances, {T['sentence']['n_speakers']} speakers. "
          "Utterance CI for ours is the second bracket in the Pearson column.\n",
          "### TEST: sentence level including the native utterances (secondary)\n"]
    L += _sent_rows(T["sentence_incl_natives"], has_az, loc2)
    if has_az:
        L += ["\n## Who is ahead (TEST, primary Azure locale, speaker bootstrap)\n"] + verdicts(R)
    L += ["\n## DEV (28 learners, Pronunciada only — no Azure on DEV)\n"]
    L += _word_rows(D["word"], False, None)
    L += ["\nHit rate, false alarms set on the learners' correct units:\n"]
    L += _hit_rows(D["hits"], False, None)
    L += ["\nSentence level:\n"]
    L += _sent_rows(D["sentence"], False, None)
    L += ["\n## Labels and data\n",
          "| split / group | speakers | utterances | units | any error % | tone wrong % | consonant wrong % | "
          "vowel wrong % | tone-only % | consonant-or-vowel % |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in lab.items():
        L.append(f"| {k} | {v['speakers']} | {v['utterances']} | {v['units']} | {100 * v['share_any_wrong']:.1f} | "
                 f"{100 * v['share_tone_wrong']:.1f} | {100 * v['share_consonant_wrong']:.1f} | "
                 f"{100 * v['share_vowel_wrong']:.1f} | {100 * v['share_tone_only']:.1f} | {100 * v['share_seg_wrong']:.1f} |")
    L += ["",
          "- Source: OMPAL corpus (github.com/phantomhsieh/OMPAL-corpus), 1,850 utterances, 49 speakers "
          "(3 native, 46 French-L1 learners), 84 distinct Traditional-script prompts (short read sentences). "
          f"Licence {LICENCE}. Citation: {CITATION}.",
          "- Raters: the README says four teachers; the per-rater file has exactly three ratings per learner item. "
          "The shipped word labels equal the majority of the three (verified on every judgement); sentence scores "
          "are the mean of the three. Native recordings have only aggregated labels (all 5/5/5; the only "
          "native judgements marked wrong are the four units of 00100217 that are not in its prompt, i.e. not said, "
          "and these are dropped, so every evaluated native unit is correct).",
          f"- Inter-rater agreement (learners, three ratings): Fleiss' κ tone {ag['word_tone_fleiss_kappa']:.2f}, "
          f"consonant {ag['word_phoneme_consonant_fleiss_kappa']:.2f}, vowel {ag['word_phoneme_vowel_fleiss_kappa']:.2f} "
          f"(unanimous on {100 * ag['word_tone_unanimous_share']:.0f} / {100 * ag['word_phoneme_consonant_unanimous_share']:.0f} / "
          f"{100 * ag['word_phoneme_vowel_unanimous_share']:.0f} % of units); sentence ratings, one rater vs the mean of "
          f"the other two (assuming list position = rater): accuracy r = {ag['sentence_accuracy_rater_vs_mean_of_others_pearson_mean']:.2f}, "
          f"fluency {ag['sentence_fluency_rater_vs_mean_of_others_pearson_mean']:.2f}, prosody "
          f"{ag['sentence_prosody_rater_vs_mean_of_others_pearson_mean']:.2f}. These bound what any system can reach.",
          "- Split (by speaker, fixed before scoring): TEST = speakers of the shipped `test/test_[1-5]_scores.json` "
          "(18 learners + natives SPEAKER01001/01002) plus native SPEAKER01003, which no shipped split uses; DEV = the "
          "other 28 learners. No speaker is in both. The shipped test files do not match the speaker lists printed in "
          "the corpus README; the files were used. Manifest: `results/tables/ompal_manifest.json`.",
          "- Units: one character each; 9 learner units list two characters where the speaker inserted or swapped a "
          "word (e.g. 還也 against the prompt's 還): the unit keeps the prompt character it matches. Dropped: "
          f"{T['counts'].get('units_not_in_prompt', 0)} TEST units whose character is not in the prompt (a native "
          "recording annotated with four extra characters) and the word level of "
          f"{T['counts'].get('utts_word_level_dropped', 0)} TEST utterance (00100218, whose unit list interleaves "
          "two sentences); its sentence scores are kept. One learner prompt (00204117) has four trailing characters "
          "with no units; they are simply not evaluated.",
          f"- Failures and coverage, TEST: {json.dumps(T['counts'], ensure_ascii=False)}; DEV: "
          f"{json.dumps(D['counts'], ensure_ascii=False)}. Our scoring errors: {len(R['ours_errors'])}.",
          ]
    if has_az:
        L.append("- Azure words per character: " + ", ".join(
            f"{k}: {100 * v:.1f} % of Azure words are a single character" for k, v in T["azure_single_char_word_share"].items())
            + "; Pronunciada scores jieba words (often two characters).")
        L.append("- Azure usage (TEST only, every response cached under `$PRONUNCIADA_CACHE/azure/ompal-<locale>/`, "
                 "nothing sent twice): " + "; ".join(
                     f"{k}: {v['requests_cached']} requests, {v['audio_hours']:.2f} audio-hours, statuses {v['statuses']}, "
                     f"median latency {v['median_latency_s']:.2f} s" for k, v in R["azure_usage"].items()
                     if v["requests_cached"]) + ". Only the public OMPAL recordings and their prompt text were sent.")
    return "\n".join(L) + "\n"


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def report() -> None:
    R = evaluate()
    md = markdown(R)
    MD_OUT.write_text(md)
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    js = json.dumps(_clean(R), indent=1, ensure_ascii=False)
    JSON_OUT.write_text(js)
    (JSON_OUT.parent / f"ompal_{R['tone_model']}.json").write_text(js)
    print(md)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="ours,report", help="comma list of ours, azure, report")
    ap.add_argument("--split", choices=("dev", "test"), help="ours step: score only this half")
    ap.add_argument("--force", action="store_true", help="ours step: re-score even if cached")
    ap.add_argument("--dry-run", action="store_true", help="azure step: only count what would be sent")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    for s in steps:
        if s not in ("ours", "azure", "report"):
            sys.exit(f"unknown step {s!r}")
    if "ours" in steps:
        run_ours(a.split, force=a.force, workers=a.workers)
    if "azure" in steps:
        run_azure(dry_run=a.dry_run)
    if "report" in steps:
        report()


if __name__ == "__main__":
    main()
