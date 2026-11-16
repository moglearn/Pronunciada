"""Evaluate Pronunciada and Azure on cv-real (bench/cv_build.py), TEST half only.

    python -m bench.eval_cv --steps ours,report              # re-score ours (cached) + report
    python -m bench.eval_cv --steps ours,report --clear-ours # drop our cached outputs first
    python -m bench.eval_cv --steps azure [--dry-run]        # send uncached test clips to Azure
    python -m bench.eval_cv --steps report

PRE-SPECIFIED (fixed before any result was looked at):
  primary score  ours  = SentenceResult.prob (sentence acceptance probability, default strictness)
                 Azure = PronScore
  primary metric AUC of "bad" = rejected (Test A) / non-native (Test B) vs accepted / native,
                 using 1 - prob and 100 - PronScore; 95 % percentile bootstrap CI resampling
                 speakers (client_id); paired bootstrap of AUC(ours) - AUC(Azure) on the same
                 speaker resamples, two-sided p = 2 * min(P(delta <= 0), P(delta >= 0)).
  secondary      ours: mean word prob, lowest word prob, scores["accuracy"], fluency
                 (scores["fluency"] when present, else SentenceResult.fluency), scores["total"];
                 Azure: AccuracyScore, FluencyScore, CompletenessScore.
  failures       a system that returns no assessment (Azure: RecognitionStatus != Success or
                 an HTTP error; ours: ValueError) gets its lowest score (0) for every score;
                 counted, never dropped.

Our outputs are cached per clip under $PRONUNCIADA_CACHE/cv-real/ours/<fingerprint>/ where
fingerprint = hash of the pronunciada/ package files, so a changed scorer is re-run
automatically; --clear-ours deletes all cached outputs. Azure responses are cached forever under
$PRONUNCIADA_CACHE/azure/cv-real/<id>.json (bench.azure_pa layout) and never re-sent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from bench import cv_build as CB
from bench.metrics import roc_auc

DATASET = "cv-real"
AZURE_LOCALE = {"en": "en-US", "zh-CN": "zh-CN", "zh-TW": "zh-TW", "ja": "ja-JP", "es": "es-MX",
                "fr": "fr-FR", "de": "de-DE", "it": "it-IT", "pt": "pt-BR", "ru": "ru-RU", "ko": "ko-KR"}
OURS_PRIMARY = "prob"
AZ_PRIMARY = "PronScore"
OURS_SECONDARY = ("word_mean", "word_min", "accuracy", "fluency", "total")
AZ_SECONDARY = ("AccuracyScore", "FluencyScore", "CompletenessScore")
REPO = Path(__file__).resolve().parents[1]
N_BOOT = 2000


def records(locales=CB.LOCALES) -> list[dict]:
    recs = CB.load_manifest(locales, half="test")
    assert all(r["half"] == "test" for r in recs)
    return recs


def unique_clips(recs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in recs:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


# ============================================================================ Azure

def azure_dir() -> Path:
    from bench.azure_pa import cache_dir

    return cache_dir(DATASET)


def run_azure(locales, dry_run: bool = False, workers: int = 4, max_tries: int = 10,
              limit: int | None = None) -> None:
    from bench import azure_pa as AZ

    clips = [r for r in unique_clips(records(locales)) if r["half"] == "test"]
    out = azure_dir()
    todo = [r for r in clips if not (out / f"{r['id']}.json").exists()]
    n_cached = len(clips) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    secs = defaultdict(float)
    for r in todo:
        secs[AZURE_LOCALE[r["locale"]]] += r["dur"]
    print(f"{DATASET}: {len(clips)} test clips, {n_cached} cached, {len(todo)} to send, "
          f"{sum(secs.values()) / 3600:.2f} audio-h", flush=True)
    for loc, s in sorted(secs.items()):
        print(f"  {loc}: {sum(1 for r in todo if AZURE_LOCALE[r['locale']] == loc)} requests, {s / 3600:.2f} h")
    if dry_run or not todo:
        return
    key, region = AZ.credentials()
    import requests

    out.mkdir(parents=True, exist_ok=True)
    local = threading.local()
    abort = threading.Event()

    def one(r: dict) -> str:
        if abort.is_set():
            return "skipped"
        assert r["half"] == "test"
        if not hasattr(local, "s"):
            local.s = requests.Session()
        locale = AZURE_LOCALE[r["locale"]]
        params = AZ.pa_params(r["sentence"], locale)
        audio = AZ.wav_bytes(r["audio"])
        resp, attempts = None, 0
        for attempt in range(max_tries):
            attempts = attempt + 1
            t = time.perf_counter()
            try:
                resp = AZ.send(local.s, key, region, locale, params, audio, max_tries=1)
                break
            except RuntimeError as e:
                msg = str(e)
                if msg.startswith("Azure returned HTTP 401") or msg.startswith("Azure returned HTTP 403"):
                    abort.set()
                    return f"auth failure ({msg[:40]}); stopping"
                if msg.startswith("Azure returned HTTP 400"):
                    resp = {"RecognitionStatus": "HTTP400", "error": msg[:300]}
                    break
                time.sleep(min(120.0, 4.0 * 2 ** attempt))
        if resp is None:
            return f"fail {r['id']}: giving up after {max_tries} tries (not cached)"
        rec = {"id": r["id"], "locale": locale, "params": params,
               "audio_sha1": hashlib.sha1(Path(r["audio"]).read_bytes()).hexdigest(),
               "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "latency_s": time.perf_counter() - t,
               "attempts": attempts, "audio_dur": r["dur"], "response": resp}
        tmp = out / f"{r['id']}.json.tmp{threading.get_ident()}"
        tmp.write_text(json.dumps(rec, ensure_ascii=False))
        os.replace(tmp, out / f"{r['id']}.json")
        return "ok"

    n_ok = n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for i, f in enumerate(as_completed(futs)):
            res = f.result()
            if res == "ok":
                n_ok += 1
            elif res != "skipped":
                n_fail += 1
                print(res, flush=True)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(todo)} ({n_fail} failed, {time.time() - t0:.0f} s)", flush=True)
    print(f"azure done: {n_ok} ok, {n_fail} failed, {time.time() - t0:.0f} s", flush=True)


def azure_scores(recs: list[dict]) -> dict:
    """{id: {"status", AZ_PRIMARY, *AZ_SECONDARY, "requests", "audio_dur"}} from the cache."""
    from bench.azure_pa import parse_response

    d = azure_dir()
    out = {}
    for r in unique_clips(recs):
        p = d / f"{r['id']}.json"
        if not p.exists():
            continue
        raw = json.loads(p.read_text())
        resp = raw["response"]
        parsed = parse_response(resp) if "NBest" in resp or "RecognitionStatus" in resp else {"status": None}
        o = {"status": parsed["status"] or "unknown", "requests": raw.get("attempts", 1),
             "audio_dur": raw.get("audio_dur", r["dur"]), "recognized": None}
        s = parsed.get("sentence") if parsed["status"] == "Success" else None
        for k in (AZ_PRIMARY,) + AZ_SECONDARY:
            v = None if s is None else s.get(k)
            o[k] = 0.0 if v is None else float(v)
        if s is None:
            o["status"] = str(parsed["status"])
        else:
            o["recognized"] = s.get("recognized")
            if s.get(AZ_PRIMARY) is None:
                o["status"] = "no_PronScore"
        out[r["id"]] = o
    return out


# ============================================================================ ours

def fingerprint() -> str:
    h = hashlib.sha256()
    root = REPO / "pronunciada"
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",):
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def ours_root() -> Path:
    return CB.cache_root() / "cv-real" / "ours"


_SC = None


def _init(lang: str) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    global _SC
    import pronunciada

    _SC = pronunciada.Scorer(lang, threads=1)


def summarize_result(res) -> dict:
    wp = [float(w.prob) for w in res.words]
    sc = dict(getattr(res, "scores", None) or {})
    flu = sc.get("fluency", res.fluency)
    return {"status": "Success", "prob": float(res.prob),
            "word_mean": float(np.mean(wp)) if wp else 0.0, "word_min": float(np.min(wp)) if wp else 0.0,
            "accuracy": None if sc.get("accuracy") is None else float(sc["accuracy"]),
            "fluency": None if flu is None else float(flu),
            "total": None if sc.get("total") is None else float(sc["total"]),
            "completeness": None if res.completeness is None else float(res.completeness),
            "n_words": len(wp), "warnings": list(res.warnings)}


def _score_one(rec: dict) -> tuple[str, dict]:
    t = time.perf_counter()
    try:
        res = _SC.score(rec["audio"], rec["sentence"])
        o = summarize_result(res)
    except ValueError as e:
        o = {"status": "error", "error": str(e)[:200]}
    o["seconds"] = round(time.perf_counter() - t, 3)
    return rec["id"], o


def run_ours(locales, workers: int = 3, clear: bool = False) -> dict:
    root = ours_root()
    if clear and root.exists():
        shutil.rmtree(root)
        print(f"cleared {root}", flush=True)
    fp = fingerprint()
    d = root / fp
    d.mkdir(parents=True, exist_ok=True)
    clips = unique_clips(records(locales))
    by_lang = defaultdict(list)
    for r in clips:
        if not (d / f"{r['id']}.json").exists():
            by_lang[r["lang"]].append(r)
    n = sum(len(v) for v in by_lang.values())
    print(f"ours (pronunciada fingerprint {fp}): {len(clips)} test clips, {len(clips) - n} cached, {n} to score",
          flush=True)
    t0 = time.time()
    for lang, recs in sorted(by_lang.items()):
        with ProcessPoolExecutor(min(workers, 3), initializer=_init, initargs=(lang,)) as ex:
            for k, (uid, o) in enumerate(ex.map(_score_one, recs, chunksize=4)):
                tmp = d / f"{uid}.json.tmp"
                tmp.write_text(json.dumps(o, ensure_ascii=False))
                os.replace(tmp, d / f"{uid}.json")
        print(f"  {lang}: {len(recs)} clips scored ({time.time() - t0:.0f} s)", flush=True)
    meta = {"fingerprint": fp, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (d / "_meta.json").write_text(json.dumps(meta))
    (root / "latest.json").write_text(json.dumps(meta))
    return meta


def ours_scores(recs: list[dict]) -> tuple[dict, str]:
    """Outputs of the most recent `ours` run (its fingerprint is reported; a warning is printed
    when pronunciada/ has changed since, i.e. `--steps ours,report` would re-score)."""
    fp = fingerprint()
    latest = ours_root() / "latest.json"
    if latest.exists():
        run_fp = json.loads(latest.read_text())["fingerprint"]
        if run_fp != fp:
            print(f"note: pronunciada/ changed since the last ours run ({run_fp} -> {fp}); "
                  f"reporting the outputs of {run_fp}", flush=True)
        fp = run_fp
    d = ours_root() / fp
    out = {}
    for r in unique_clips(recs):
        p = d / f"{r['id']}.json"
        if p.exists():
            o = json.loads(p.read_text())
            if o["status"] != "Success":
                o = {"status": o["status"], "error": o.get("error"),
                     **{k: 0.0 for k in (OURS_PRIMARY,) + OURS_SECONDARY}}
            out[r["id"]] = o
    return out, fp


# ============================================================================ metrics

def _auc_bad(y: np.ndarray, good: np.ndarray) -> float:
    """AUC for detecting y = 1 with score = -good (higher good = better pronunciation)."""
    return roc_auc(y, -good)


def speaker_bootstrap(y, a, b, spk, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """AUC of a, of b (both 'goodness' scores) with speaker-resampled CIs, and the paired delta."""
    y, a, b = np.asarray(y), np.asarray(a, float), np.asarray(b, float)
    uniq, inv = np.unique(np.asarray(spk), return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    va, vb, dd = [], [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        x1, x2 = _auc_bad(yy, a[idx]), _auc_bad(yy, b[idx])
        va.append(x1)
        vb.append(x2)
        dd.append(x1 - x2)
    va, vb, dd = map(np.asarray, (va, vb, dd))
    A, Bv = _auc_bad(y, a), _auc_bad(y, b)
    p = min(1.0, 2 * min((np.sum(dd <= 0) + 1) / (len(dd) + 1), (np.sum(dd >= 0) + 1) / (len(dd) + 1)))
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return {"ours": A, "ours_ci": q(va), "azure": Bv, "azure_ci": q(vb), "delta": A - Bv, "delta_ci": q(dd),
            "p_two_sided": float(p), "n_boot": int(len(dd))}


def single_bootstrap(y, a, spk, n_boot: int = N_BOOT, seed: int = 1) -> dict:
    y, a = np.asarray(y), np.asarray(a, float)
    uniq, inv = np.unique(np.asarray(spk), return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        if y[idx].min() != y[idx].max():
            v.append(_auc_bad(y[idx], a[idx]))
    return {"auc": _auc_bad(y, a), "ci": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]}


def evaluate(recs: list[dict], ours: dict, az: dict) -> dict:
    res = {}
    groups = defaultdict(list)
    for r in recs:
        groups[(r["locale"], r["test"])].append(r)
    for (loc, test), rs in sorted(groups.items(), key=lambda kv: (CB.LOCALES.index(kv[0][0]), kv[0][1])):
        have = [r for r in rs if r["id"] in ours and r["id"] in az]
        e = {"locale": loc, "test": test, "n_pos": sum(r["label"] for r in rs),
             "n_neg": sum(1 - r["label"] for r in rs), "n_speakers": len({r["client_id"] for r in rs}),
             "n_scored_both": len(have), "missing_ours": sum(r["id"] not in ours for r in rs),
             "missing_azure": sum(r["id"] not in az for r in rs),
             "azure_failures": dict(Counter(az[r["id"]]["status"] for r in have if az[r["id"]]["status"] != "Success")),
             "ours_failures": dict(Counter(ours[r["id"]]["status"] for r in have if ours[r["id"]]["status"] != "Success")),
             "azure_failures_by_side": {s: sum(1 for r in have if r["label"] == s and az[r["id"]]["status"] != "Success")
                                        for s in (1, 0)},
             "mean_dur": {s: float(np.mean([r["dur"] for r in rs if r["label"] == s] or [np.nan])) for s in (1, 0)},
             "mean_chars": {s: float(np.mean([r["n_chars"] for r in rs if r["label"] == s] or [np.nan])) for s in (1, 0)}}
        if have and 0 < sum(r["label"] for r in have) < len(have):
            y = np.array([r["label"] for r in have])
            spk = [r["client_id"] for r in have]
            a = np.array([ours[r["id"]][OURS_PRIMARY] for r in have])
            b = np.array([az[r["id"]][AZ_PRIMARY] for r in have])
            e["primary"] = speaker_bootstrap(y, a, b, spk)
            sec = {}
            for k in OURS_SECONDARY:
                v = [ours[r["id"]].get(k) for r in have]
                if all(x is not None for x in v):
                    sec[f"ours:{k}"] = single_bootstrap(y, np.array(v, float), spk)
            for k in AZ_SECONDARY:
                sec[f"azure:{k}"] = single_bootstrap(y, np.array([az[r["id"]][k] for r in have]), spk)
            # length-only baselines (should be ~0.5 if the matching worked)
            sec["baseline:-duration"] = single_bootstrap(y, -np.array([r["dur"] for r in have]), spk)
            sec["baseline:-chars"] = single_bootstrap(y, -np.array([r["n_chars"] for r in have], float), spk)
            e["secondary"] = sec
        res[f"{loc}/{test}"] = e
    return res


# ============================================================================ report

HOW_TO_READ = """\
**How to read this.** Test A asks whether a system scores a recording lower when Common Voice
volunteers voted it down ("does this recording match the sentence?") than when they accepted it
unanimously. Reviewer rejections are not mispronunciation labels: they include misread or skipped
words, wrong sentences, truncated or noisy recordings, very strong accents, and reviewers' own
mistakes; acceptances can still contain mild errors. Volunteers are not trained raters and each
clip has only a handful of votes. A high AUC therefore means "agrees with crowd rejection", which
mixes pronunciation with reading accuracy and audio quality; neither system was built for exactly
this question, and absolute values are not comparable with the English expert-rated benchmarks.
Test A-weak (Korean only) adds clips with a single down-vote and no up-vote from
the not-yet-decided pool; one down-vote is a much weaker label than the strict rule (≥ 2 down-votes
and down − up ≥ 2): many such clips would have been accepted with more votes.
Test B compares accepted (correctly read) clips of speakers who self-declare a non-native accent
with speakers who declare a native variety. A pronunciation scorer is expected to give accented
speech somewhat lower scores, but a lower score here is **not** an error: the words were read
correctly and judged acceptable by reviewers, and accent labels are self-reported and sparse.
Both sides of every comparison are matched on prompt length and duration (stratified sampling);
the "length-only" rows show what prompt length or duration alone would achieve (≈ 0.5 means the
matching removed that shortcut)."""


def _ci(v, ci) -> str:
    return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _bold_pair(ev: dict) -> tuple[str, str, str]:
    p = ev["primary"]
    o, a = _ci(p["ours"], p["ours_ci"]), _ci(p["azure"], p["azure_ci"])
    if p["p_two_sided"] < 0.05:
        if p["delta"] > 0:
            o = f"**{o}**"
        else:
            a = f"**{a}**"
    verdict = ("ours ahead" if p["delta"] > 0 else "Azure ahead") + (
        "" if p["p_two_sided"] < 0.05 else " (n.s.)")
    return o, a, verdict


def markdown(ev: dict, build: dict, usage: dict, fp: str) -> str:
    L = ["# cv-real: real Common Voice speech judged by real people (ours vs Azure)", "",
         f"Generated by `python -m bench.eval_cv --steps report` ({time.strftime('%Y-%m-%d')}); "
         f"Pronunciada package fingerprint `{fp}`. Test half only; the dev half "
         "(same size, disjoint speakers) is reserved for future tuning and has never been scored or sent to Azure.",
         "", "Pre-specified before any result was seen: primary score ours = sentence acceptance probability "
         "(`SentenceResult.prob`, default strictness), Azure = `PronScore`; primary metric = AUC with 95 % "
         "bootstrap CIs resampling speakers and a paired speaker-bootstrap two-sided p-value for ours − Azure. "
         "Everything in the sensitivity table is secondary. Nothing in `pronunciada/` was tuned on this data.",
         "", "Our numbers come from the Pronunciada tree at the fingerprint above; the English sentence scorer "
         "and the Mandarin tone model were being changed by other work at the time, so re-run "
         "`python -m bench.eval_cv --steps ours,report` before quoting them (Azure responses are cached and are "
         "not re-sent).", "", HOW_TO_READ, ""]
    for test, title in (("A", "Test A — reviewer-rejected vs reviewer-accepted clips"),
                        ("Aweak", "Test A-weak (ko only) — weak-label rejected vs reviewer-accepted clips"),
                        ("B", "Test B — self-declared non-native vs native speakers (accepted clips only)")):
        rows = [e for e in ev.values() if e["test"] == test]
        L += [f"## {title}", "",
              "| locale | rejected / accepted | speakers | Pronunciada AUC [95 % CI] | Azure AUC [95 % CI] | ours − Azure [95 % CI] | p | |"
              if test in ("A", "Aweak") else
              "| locale | non-native / native | speakers | Pronunciada AUC [95 % CI] | Azure AUC [95 % CI] | ours − Azure [95 % CI] | p | |",
              "|---|---|---|---|---|---|---|---|"]
        for e in rows:
            if "primary" not in e:
                L.append(f"| {e['locale']} | {e['n_pos']} / {e['n_neg']} | {e['n_speakers']} | – | – | – | – | not scored |")
                continue
            o, a, verdict = _bold_pair(e)
            p = e["primary"]
            L.append(f"| {e['locale']} | {e['n_pos']} / {e['n_neg']} | {e['n_speakers']} | {o} | {a} | "
                     f"{p['delta']:+.3f} [{p['delta_ci'][0]:+.3f}, {p['delta_ci'][1]:+.3f}] | {p['p_two_sided']:.3f} | {verdict} |")
        if test == "Aweak":
            for loc, v in build["locales"].items():
                w = v.get("testAweak_eligibility")
                if not w:
                    continue
                t = v["tests"].get("Aweak", {}).get("test", {})
                L += ["", f"{loc}: positives = the strict Test A rejected candidates plus `other.tsv` clips with "
                          f"down_votes ≥ 1 and up_votes = 0 (`other.tsv` = clips with too few votes to be validated or "
                          f"invalidated; vote table up/down → clips: "
                          + ", ".join(f"{k} → {n}" for k, n in w["other_vote_table"].items())
                          + f"). {w['other_weak_candidates']} weak candidates; after the duration/text rules "
                          f"{w['positives_after_cleaning']} positives ({w['positives_strict']} strict + "
                          f"{w['positives_weak']} weak) from {w['positive_speakers']} speakers (build threshold "
                          f"{CB.WEAK_MIN_POS}). Test half: {t.get('n_pos', 0)} per side after the speaker split, "
                          f"speaker/sentence disjointness, 2-per-speaker cap, matching and the near-silent rule "
                          f"({t.get('audio_rules_on_fetched') or 0}); the manifest flags each positive `strict` "
                          "true/false. Negatives and every other rule are as in Test A. The 11-per-side strict "
                          "Test A row above is unchanged.",
                      "", "`reported.tsv` cannot be used for this: its header (ko) is `sentence_id, sentence, "
                          "locale, reason`, i.e. reports on sentence prompts, with no clip or speaker."]
        if test == "B":
            no_b = [l for l, v in build["locales"].items() if "B" not in v["tests"]]
            L += ["", f"Test B exists only where ≥ {CB.TESTB_MIN_CLIPS} clips from ≥ {CB.TESTB_MIN_SPK} self-declared "
                      f"non-native speakers were available after cleaning: not run for {', '.join(no_b)} "
                      "(eligible non-native clips / speakers: " + "; ".join(
                          f"{l} {build['locales'][l]['testB_eligibility']['nonnative_clips_capped']}/"
                          f"{build['locales'][l]['testB_eligibility']['nonnative_speakers']}" for l in no_b) + ")."]
        L.append("")
    L += ["## Sensitivity (secondary scores; AUC [95 % CI], speaker bootstrap)", "",
          "Ours: `word_mean` / `word_min` = mean / lowest word probability, `accuracy`, `fluency`, `total` = "
          "regression-style sentence scores. Azure: Accuracy, Fluency, Completeness. `-duration` / `-chars` = "
          "length-only baselines (shorter = 'worse').", ""]
    keys = []
    for e in ev.values():
        for k in e.get("secondary", {}):
            if k not in keys:
                keys.append(k)
    L += ["| locale / test | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for e in ev.values():
        s = e.get("secondary", {})
        L.append(f"| {e['locale']} {e['test']} | " + " | ".join(
            (f"{s[k]['auc']:.3f} [{s[k]['ci'][0]:.2f}, {s[k]['ci'][1]:.2f}]" if k in s else "–") for k in keys) + " |")
    L += ["", "## Failures (counted as the lowest score, never dropped)", "",
          "| locale / test | Azure failures (rejected / accepted side) | Azure statuses | ours failures | missing ours / Azure |",
          "|---|---|---|---|---|"]
    for e in ev.values():
        fb = e["azure_failures_by_side"]
        L.append(f"| {e['locale']} {e['test']} | {fb[1]} / {fb[0]} | {e['azure_failures'] or '–'} | "
                 f"{e['ours_failures'] or '–'} | {e['missing_ours']} / {e['missing_azure']} |")
    L += ["", "## Data: source, cleaning and matching", "",
          f"Source for every locale: **{build['source']['dataset']}**, licence **{build['source']['license']}** "
          f"(stated on the mirror's dataset card), Hugging Face mirror `{build['source']['mirror'].split('datasets/')[-1]}` "
          f"at revision `{build['source']['revision']}`. Only the TSV metadata and the tar shards named below were "
          "read; shards were streamed and only selected clips were stored (16 kHz FLAC, outside the repo). "
          "Manifests (clip ids, speaker hashes, votes, accents, source shard): `bench/data/cv_real/<locale>.jsonl`; "
          "build log with all counts: `bench/data/cv_real/build.json`.", "",
          "Rules: " + "; ".join(f"{k}: {v}" for k, v in build["rules"].items()) + ".", "",
          "Rejected clips come from invalidated shard 0 only (the first rows of `invalidated.tsv`) where a locale has "
          "several invalidated shards (en, zh-CN, es, fr, de), to limit download; accepted clips come from the official "
          "dev + test splits (validated).", "",
          "| locale | shards read (GB streamed) | after vote rule (rej / acc) | duration + text rules removed (rej; acc) | "
          "speaker / sentence overlap and per-speaker cap removed (test half) | near-silent / undecodable among fetched (test half) | region rule |",
          "|---|---|---|---|---|---|---|"]
    for loc, v in build["locales"].items():
        c = v["counts"]
        ta = v["tests"]["A"]["test"]
        reg = c.get("region", {}).get("rule", "–")
        L.append(f"| {loc} | {', '.join(s.split('/', 1)[1] for s in v['shards_read'])} ({v['streamed_gb']}) | "
                 f"{c['after_votes']['rejected']} / {c['after_votes']['accepted']} | "
                 f"{c['rules_rejected'] or 0}; {c['rules_accepted'] or 0} | {ta['dedup_cap']} | "
                 f"{ta['audio_rules_on_fetched'] or 0} | {reg} |")
    short = [l for l, v in build["locales"].items() if v["tests"]["A"]["test"]["n_pos"] < CB.TARGET]
    if short:
        L += ["", "Fewer than 200 clips per side in the Test A test half: " + ", ".join(
            f"{l} ({build['locales'][l]['tests']['A']['test']['n_pos']} per side, "
            f"pool after all metadata rules {build['locales'][l]['tests']['A']['test']['pool']['pos']} rejected clips "
            f"from {build['locales'][l]['tests']['A']['test']['pool']['pos_spk']} speakers)" for l in short) +
            ". Apart from ko, where Common Voice 22.0 has only 279 clearly rejected clips from 47 speakers in total, "
            "the 1–7 missing pairs are strata where the near-silent rule removed more clips than the fetched reserve "
            "(35 % + 2 per stratum); a stratum is always trimmed on both sides so the matching stays exact."]
    L += ["", "The near-silent rule (system-independent) removes almost only rejected clips (e.g. en: 122 of 638 "
          "fetched rejected candidates vs 8 of 628 accepted, both halves): empty or near-empty recordings are an "
          "obvious reason for rejection and would have made Test A easier for both systems."]
    L += ["", "## Azure usage", "",
          f"{usage['clips']} test clips sent once each ({usage['requests']} HTTP requests including retries), "
          f"{usage['hours']:.2f} h of audio, Azure region from the environment; responses cached in "
          "`$PRONUNCIADA_CACHE/azure/cv-real/`. Locales: " + ", ".join(f"{k} → {v}" for k, v in AZURE_LOCALE.items()) +
          ". Only public CC0 Common Voice recordings and their prompts were sent.", ""]
    return "\n".join(L)


def azure_usage(az: dict) -> dict:
    return {"clips": len(az), "requests": int(sum(o.get("requests", 1) for o in az.values())),
            "hours": float(sum(o.get("audio_dur", 0) for o in az.values()) / 3600)}


def report(locales) -> None:
    recs = records(locales)
    ours, fp = ours_scores(recs)
    az = azure_scores(recs)
    ev = evaluate(recs, ours, az)
    build = json.loads((CB.OUT_DIR / "build.json").read_text())
    usage = azure_usage(az)
    tables = REPO / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "cv_real.json").write_text(json.dumps(
        {"fingerprint": fp, "pre_specified": {"ours": OURS_PRIMARY, "azure": AZ_PRIMARY, "metric": "AUC",
                                              "ci": "speaker bootstrap 95 %", "n_boot": N_BOOT},
         "azure_locales": AZURE_LOCALE, "azure_usage": usage, "results": ev,
         "build": {k: build[k] for k in ("source", "rules")}}, indent=1, ensure_ascii=False))
    (REPO / "results" / "cv_real.md").write_text(markdown(ev, build, usage, fp))
    for e in ev.values():
        if "primary" in e:
            p = e["primary"]
            print(f"{e['locale']:6s} {e['test']}  n={e['n_pos']}/{e['n_neg']}  ours {p['ours']:.3f}  "
                  f"azure {p['azure']:.3f}  delta {p['delta']:+.3f}  p={p['p_two_sided']:.3f}")
    print(f"wrote results/cv_real.md and results/tables/cv_real.json; azure usage {usage}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="ours,report", help="comma list of azure, ours, report")
    ap.add_argument("--locales", default="all")
    ap.add_argument("--workers", type=int, default=3, help="local scoring processes (max 3)")
    ap.add_argument("--azure-workers", type=int, default=4)
    ap.add_argument("--clear-ours", action="store_true", help="delete all cached outputs of ours first")
    ap.add_argument("--dry-run", action="store_true", help="azure: only count what would be sent")
    ap.add_argument("--limit", type=int, help="azure: send at most N uncached clips")
    a = ap.parse_args(argv)
    locs = list(CB.LOCALES) if a.locales == "all" else a.locales.split(",")
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    for s in steps:
        if s not in ("azure", "ours", "report"):
            sys.exit(f"unknown step {s}")
    if "azure" in steps:
        run_azure(locs, a.dry_run, a.azure_workers, limit=a.limit)
    if "ours" in steps:
        run_ours(locs, a.workers, a.clear_ours)
    if "report" in steps:
        report(locs)


if __name__ == "__main__":
    main()
