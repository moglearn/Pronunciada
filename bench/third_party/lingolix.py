"""Pronunciada on the Lingolix pronunciation-assessment API benchmark (v2026.10).

    python -m bench.third_party.lingolix fetch            # download the test items (no account needed)
    python -m bench.third_party.lingolix score [--workers 2]
    python -m bench.third_party.lingolix report           # writes results/tables/third_party_lingolix.json
    python -m bench.third_party.lingolix all

Benchmark: https://github.com/lingolix/pronunciation-benchmark (code MIT, Lingolix UG), results
post: https://lingolix.com/en/blog/best-pronunciation-assessment-api-2026 (11 October 2026).

Everything downloaded is treated as data. Nothing from the benchmark repository is imported or run:
the manifests are read as JSON, and the metrics of its report.py are re-implemented below from
reading the source (rules of tag v2026.10, the version current on the post's date; the later
`unverified` exclusion of commit f8750aa is computed as a second variant).

Pronunciada is scored with its public API and default settings (Scorer(lang): fp32, strictness 0.5).
Each result is converted with pronunciada.compat.azure.to_azure (REST layout, Granularity=Phoneme,
EnableMiscue=True, the settings the benchmark uses for Azure) and then reduced exactly as the
benchmark's Azure adapter reduces Azure's response: word score = AccuracyScore / 100 (0 when
absent), word flag = non-empty ErrorType, sentence score = NBest[0] PronScore. So Pronunciada goes
through the same code path as Azure, and no threshold is chosen here: the threshold-free metrics
(AUC, PR-AUC) need none, FA@N uses the benchmark's rule (the threshold at which N% of the planted
errors are caught, applied per system), and the verdicts use the system's own flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import tarfile
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada")
BASE = CACHE / "third_party" / "lingolix"
OUT_JSON = ROOT / "results" / "tables" / "third_party_lingolix.json"

REPO = "lingolix/pronunciation-benchmark"
SOURCES = {
    # tag v2026.10 (11 Oct 2026): manifests and metric rules as of the blog post
    "repo_v2026.10": "e021b08c0b6549306f21ba053e7884bd73d37ff5",
    # main (18 Oct 2026): same items, adds per-item `unverified` word lists to the tier 1 manifests
    "repo_main_f8750aa": "f8750aa76d86efa67cdc0c5f2fea4b7c5b5a97c4",
}
TIER1_ZIP = (f"https://github.com/{REPO}/releases/download/v2026.10/pronbench-tier1-audio-v2026.10.zip",
             138190959)
SO762_URL = "https://www.openslr.org/resources/101/speechocean762.tar.gz"
SO762_LOCAL = CACHE / "data" / "speechocean762" / "manifest.jsonl"   # Pronunciada's own copy, if present
LANGS = ("en", "fr", "es", "de", "ja", "ko")
VENDOR = "pronunciada"

# Published by the benchmark's author (Lingolix), not re-run here. Source: the blog post above.
PUBLISHED = {
    "source": "https://lingolix.com/en/blog/best-pronunciation-assessment-api-2026",
    "date": "2026-10-11 (TonePerfect note added 2026-10-17)",
    "tested": "October 2026, each API in its 'documented best configuration'; no API versions given",
    "tier1_auc": {
        "lingolix": {"en": 0.97, "fr": 0.96, "es": 0.97, "de": 0.95, "ja": 0.95, "ko": 0.95},
        "azure": {"en": 0.94, "fr": 0.91, "es": 0.91, "de": 0.87, "ja": 0.93, "ko": 0.94},
        "toneperfect": {"en": 0.84, "fr": 0.80, "es": 0.79, "de": 0.79, "ja": 0.55, "ko": 0.67},
    },
    "tier1_fa_at_80": {
        "lingolix": {"en": 0.01, "fr": 0.01, "es": 0.02, "de": 0.02, "ja": 0.05, "ko": 0.03},
        "azure": {"en": 0.06, "fr": 0.13, "es": 0.14, "de": 0.22, "ja": 0.03, "ko": 0.07},
        "toneperfect": {"en": 0.26, "fr": 0.26, "es": 0.32, "de": 0.42, "ja": 0.74, "ko": 0.56},
    },
    "ja_length_caught_of_40": {"lingolix": 37, "azure": 36, "toneperfect": 0},
    "tier2_pcc": {
        "lingolix": {"utterance": 0.74, "word": 0.51},
        "azure": {"utterance": 0.76, "word": 0.61},
        "toneperfect": {"utterance": 0.59, "word": 0.29},
    },
    "latency_s": {"lingolix": 0.8, "azure": 1.0, "toneperfect": 2.5},
}


# ── fetching (data only) ───────────────────────────────────────────────────────

def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dst)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def repo_dir(name: str) -> Path:
    """Extracted source tarball of one pinned commit (each in its own directory)."""
    return BASE / name / "src"


def fetch_repo(name: str) -> Path:
    src = repo_dir(name)
    if (src / "data" / "tier1" / "manifest_en.jsonl").exists():
        return src
    sha = SOURCES[name]
    tgz = BASE / name / f"{sha}.tar.gz"
    if not tgz.exists():
        print(f"downloading {REPO}@{sha[:7]}", file=sys.stderr)
        _download(f"https://codeload.github.com/{REPO}/tar.gz/{sha}", tgz)
    tmp = BASE / name / "extract"
    with tarfile.open(tgz) as t:
        t.extractall(tmp, filter="data")
    (top,) = list(tmp.iterdir())
    top.rename(src)
    tmp.rmdir()
    return src


def tier1_audio_dir() -> Path:
    return BASE / "tier1_audio_v2026.10" / "unzipped"


def fetch_tier1_audio() -> Path:
    d = tier1_audio_dir()
    if d.exists() and any(d.rglob("*.wav")):
        return d
    url, size = TIER1_ZIP
    z = BASE / "tier1_audio_v2026.10" / Path(url).name
    if not z.exists() or z.stat().st_size != size:
        print(f"downloading {url} ({size / 1e6:.0f} MB)", file=sys.stderr)
        _download(url, z)
    with zipfile.ZipFile(z) as zf:
        for m in zf.namelist():           # data only: refuse anything that is not plain audio
            if not m.endswith("/") and not m.lower().endswith((".wav", ".txt", ".md", ".json", ".jsonl")):
                raise RuntimeError(f"unexpected member in audio zip: {m}")
            if m.startswith("/") or ".." in Path(m).parts:
                raise RuntimeError(f"unsafe path in audio zip: {m}")
        zf.extractall(d)
    return d


def tier1_wav(item: dict) -> Path:
    d = tier1_audio_dir()
    for cand in (d / item["audio"], d / Path(item["audio"]).name, d / "tier1" / item["audio"]):
        if cand.exists():
            return cand
    hits = list(d.rglob(Path(item["audio"]).name))
    if not hits:
        raise FileNotFoundError(item["audio"])
    return hits[0]


def so762_paths(uids: list[str]) -> dict[str, Path]:
    """Audio of the 500 test 2 clips: Pronunciada's local speechocean762 copy when it has every
    clip (same recordings, lossless FLAC), otherwise the OpenSLR archive in its own directory."""
    if SO762_LOCAL.exists():
        m = {}
        for line in SO762_LOCAL.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            m[d["id"].removeprefix("so762-")] = (d, Path(d["audio"].replace("/.cache/pronkit/", "/.cache/pronunciada/")))
        if all(u in m and m[u][1].exists() for u in uids):
            assert all(m[u][0]["split"] == "test" for u in uids)
            return {u: m[u][1] for u in uids}
    root = BASE / "speechocean762_openslr"
    tree = root / "speechocean762"
    if not tree.exists():
        tgz = root / "speechocean762.tar.gz"
        if not tgz.exists():
            print(f"downloading {SO762_URL} (~520 MB)", file=sys.stderr)
            _download(SO762_URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(root, filter="data")
    scp = {}
    for line in (tree / "test" / "wav.scp").read_text(encoding="utf-8").splitlines():
        if line.strip():
            u, rel = line.split(maxsplit=1)
            scp[u] = tree / rel
    return {u: scp[u] for u in uids}


def load_manifest(repo: str, rel: str) -> list[dict]:
    with open(repo_dir(repo) / "data" / rel, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tests() -> dict[str, list[dict]]:
    out = {f"tier1_{lang}": load_manifest("repo_v2026.10", f"tier1/manifest_{lang}.jsonl") for lang in LANGS}
    out["tier2"] = load_manifest("repo_v2026.10", "tier2/manifest_speechocean762.jsonl")
    return out


def cmd_fetch() -> None:
    for name in SOURCES:
        print(name, fetch_repo(name))
    print("tier1 audio", fetch_tier1_audio())
    t2 = load_manifest("repo_v2026.10", "tier2/manifest_speechocean762.jsonl")
    p = so762_paths([x["id"] for x in t2])
    print("tier2 audio", len(p), "clips, e.g.", next(iter(p.values())))
    z = BASE / "tier1_audio_v2026.10" / Path(TIER1_ZIP[0]).name
    if z.exists():
        print("tier1 zip sha256", _sha256(z))


# ── scoring with Pronunciada's public API ──────────────────────────────────────

_SCORERS: dict = {}


def _scorer(lang: str, threads: int):
    if lang not in _SCORERS:
        from pronunciada import Scorer
        _SCORERS.clear()
        _SCORERS[lang] = Scorer(lang, threads=threads)        # defaults: fp32, strictness 0.5
    return _SCORERS[lang]


def reduce_azure_shape(raw: dict) -> list[dict]:
    """The benchmark's Azure adapter (vendors.azure), applied to Pronunciada's Azure-layout output."""
    def field(w, name):
        return w[name] if name in w else (w.get("PronunciationAssessment") or {}).get(name)

    nbest = (raw.get("NBest") or [{}])[0]
    out = []
    for w in nbest.get("Words", []):
        acc = field(w, "AccuracyScore")
        out.append({"text": w["Word"], "score": 0.0 if acc is None else float(acc) / 100,
                    "scored": True, "error_type": field(w, "ErrorType")})
    return out


def _score_chunk(jobs: list[tuple[str, str, str, str]], threads: int) -> list[dict]:
    from pronunciada.compat.azure import to_azure

    recs = []
    for iid, lang, wav, text in jobs:
        t0 = time.perf_counter()
        try:
            res = _scorer(lang, threads).score(wav, text)
            raw = to_azure(res, granularity="Phoneme", enable_miscue=True, phoneme_alphabet="IPA",
                           layout="rest", grading_system="HundredMark")
            rec = {"id": iid, "vendor": VENDOR, "words": reduce_azure_shape(raw), "raw": raw,
                   "pronunciada": {"prob": res.prob, "scores": res.scores, "warnings": res.warnings,
                                   "word_probs": [w.prob for w in res.words]}}
        except Exception as e:  # counted as a failed request, as the benchmark does
            rec = {"id": iid, "vendor": VENDOR, "error": f"{type(e).__name__}: {e}"}
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        recs.append(rec)
    return recs


def responses_path(test: str) -> Path:
    return BASE / "responses" / VENDOR / f"{test}.jsonl"


def read_responses(test: str) -> dict[str, dict]:
    p = responses_path(test)
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def cmd_score(workers: int, threads: int, only: list[str] | None) -> None:
    workers = max(1, min(workers, 2))           # the laptop is shared: never more than 2 processes
    T = tests()
    so = so762_paths([x["id"] for x in T["tier2"]])
    jobs_by_test = {}
    for test, items in T.items():
        if only and test not in only:
            continue
        done = {k for k, r in read_responses(test).items() if not r.get("error")}
        jobs = []
        for it in items:
            if it["id"] in done:
                continue
            wav = so[it["id"]] if test == "tier2" else tier1_wav(it)
            jobs.append((it["id"], it.get("lang", "en"), str(wav), it["target"]))
        jobs_by_test[test] = jobs
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for test, jobs in jobs_by_test.items():
            if not jobs:
                continue
            p = responses_path(test)
            p.parent.mkdir(parents=True, exist_ok=True)
            chunks = [jobs[i:i + 25] for i in range(0, len(jobs), 25)]
            t0, n = time.time(), 0
            with open(p, "a", encoding="utf-8") as f:
                for fut in as_completed([ex.submit(_score_chunk, c, threads) for c in chunks]):
                    for r in fut.result():
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        n += 1
                    f.flush()
            print(f"{test}: scored {n} in {time.time() - t0:.0f}s", file=sys.stderr)


# ── metrics, re-implemented from report.py of the benchmark ────────────────────

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
BOOTSTRAP = 1000


def key(w: str) -> str:
    return _PUNCT.sub("", w).lower()


def roc_auc(labels, scores) -> float:
    """Mann-Whitney U with ties averaged; `scores` is the error signal (higher = worse)."""
    n1 = sum(labels)
    n0 = len(labels) - n1
    if not n1 or not n0:
        return math.nan
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    s = sum(r for r, l in zip(ranks, labels) if l)
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)


def average_precision(labels, scores) -> float:
    n1 = sum(labels)
    if not n1:
        return math.nan
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp, ap = 0, 0.0
    for k, i in enumerate(order, 1):
        if labels[i]:
            tp += 1
            ap += tp / k
    return ap / n1


def fpr_at_recall(labels, scores, target: float) -> float:
    """Share of correct words at or above the error-signal threshold that catches `target` of the errors."""
    pos = sorted((s for s, l in zip(scores, labels) if l), reverse=True)
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return math.nan
    thr = pos[min(int(math.ceil(target * len(pos))) - 1, len(pos) - 1)]
    return sum(1 for s in neg if s >= thr) / len(neg)


def bootstrap_ci(units, fn, seed=0):
    """Resample whole utterances (random.Random(0), 1000 draws, 2.5/97.5 percentiles), as report.py."""
    rng = random.Random(seed)
    vals = []
    for _ in range(BOOTSTRAP):
        pick = [units[rng.randrange(len(units))] for _ in range(len(units))]
        v = fn([l for u in pick for l in u[0]], [s for u in pick for s in u[1]])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return [math.nan, math.nan]
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)]]


def _opcodes(a, b):
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        sys.path.insert(0, str(BASE / "_pydeps"))      # rapidfuzz 3.14.6, the version in their uv.lock
        from rapidfuzz.distance import Levenshtein
    return Levenshtein.opcodes(a, b)


def align(target_words, vendor_words):
    a = [key(w) for w in target_words]
    b = [key(w.get("text", "")) for w in vendor_words]
    out = [None] * len(a)
    for op in _opcodes(a, b):
        if op.tag in ("equal", "replace"):
            for k, i in enumerate(range(op.src_start, op.src_end)):
                j = op.dest_start + k
                if j < op.dest_end:
                    out[i] = vendor_words[j]
    return out


def align_spans(text, units, vendor_words, pick=None):
    spans, cur = [], 0
    for w in vendor_words:
        t = _PUNCT.sub("", w.get("text", ""))
        j = text.find(t, cur) if t else -1
        if j >= 0:
            spans.append((j, j + len(t), w))
            cur = j + len(t)
    out = []
    for u in units:
        hits = [w for s, e, w in spans if s < u["end"] and e > u["start"]]
        if not hits:
            out.append(None)
        elif pick:
            out.append(pick(hits))
        else:
            out.append(min(hits, key=lambda w: float(w.get("score", 0.0))))
    return out


# report.vendor_flags for Lingolix-shaped records (raw["words"][k]["syllables"]): the phones the API
# heard are compared with the phones it expected, after report.py's per-language canonical spelling.
_MERGERS_FR = {"e": "ɛ", "o": "ɔ", "ø": "œ", "ɑ": "a", "œ̃": "ɛ̃", "ɥ": "y", "w": "u", "j": "i"}
_MERGERS_ES = {"θ": "s", "ʎ": "i", "ʝ": "i", "j": "i", "w": "u", "β": "b", "ð": "d", "ɣ": "ɡ", "h": "x", "ŋ": "n"}
_VOWELS = set("iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɝɞʌɔæɐaɶɑɒ")
_FT = None


def _segs(ipa: str) -> tuple[str, ...]:
    global _FT
    if _FT is None:
        import panphon
        _FT = panphon.FeatureTable()
    return tuple(_FT.ipa_segs(ipa))


def _merge(phones: tuple[str, ...], lang: str) -> tuple[str, ...]:
    if lang == "fr":
        return tuple(_MERGERS_FR.get(q, q) for q in (p.replace("ː", "") for p in phones) if q != "ə")
    m = [_MERGERS_ES.get(q, q) for q in phones]
    return tuple(q for i, q in enumerate(m) if not (q == "s" and (i == len(m) - 1 or m[i + 1] not in _VOWELS)))


def _canon(ipa: str, lang: str) -> str:
    if lang in ("fr", "es"):
        return "".join(_merge(_segs(ipa), lang))
    if lang != "ja":
        return ipa
    s = ipa.replace("̥", "").replace("ʲ", "j").replace("ɸ", "f").replace("ç", "h")
    return re.sub("[ɴŋɲm]|n(?![aieoɯ])", "N", s)


def vendor_flags(rec: dict, lang: str = "") -> list[bool] | None:
    """report.vendor_flags. Azure-shaped records (Pronunciada's): a non-empty ErrorType. Lingolix-shaped
    records: additionally, expected and detected syllable IPA differ (ja, fr, es); None (verdicts
    undefined) for the other languages, as in report.py."""
    words = rec.get("words", [])
    raw_words = (rec.get("raw") or {}).get("words")
    parallel = isinstance(raw_words, list) and len(raw_words) == len(words)
    out = []
    for k, w in enumerate(words):
        flag = w.get("error_type") not in (None, "None")
        syl = [x for x in (raw_words[k].get("syllables") or []) if x.get("is_scored", True)] if parallel else []
        if syl and lang not in ("ja", "fr", "es"):
            return None
        if syl:
            exp = _canon("".join(x.get("expected_ipa") or "" for x in syl), lang)
            det = _canon("".join(x.get("detected_ipa") or "" for x in syl), lang)
            flag = flag or exp != det
        out.append(flag)
    return out


def error_signal(entry) -> float:
    return 1.0 if entry is None else 1.0 - float(entry.get("score", 0.0))


def utterance_score(rec: dict, which: str = "benchmark"):
    """benchmark: report.utterance_score on the Azure layout = NBest[0] PronScore (= 100 x scores['total']).
    prob: Pronunciada's headline sentence probability (reported as a side note only)."""
    if which == "prob":
        return rec["pronunciada"]["prob"]
    raw = rec.get("raw") or {}
    if isinstance(raw.get("overall"), (int, float)):      # Lingolix (report.utterance_score order)
        return float(raw["overall"])
    if isinstance(raw.get("accuracy"), (int, float)):
        return float(raw["accuracy"])
    tp = raw.get("scores") or {}
    if isinstance(tp.get("overall"), (int, float)):       # TonePerfect
        return float(tp["overall"]) / 100
    nbest = raw.get("NBest") or [{}]
    pa = nbest[0].get("PronunciationAssessment") or nbest[0]
    return float(pa["PronScore"]) if "PronScore" in pa else None


def collect(manifest, responses, exclude_unverified: dict | None = None, utt_which="benchmark"):
    units, loc_hit, loc_n, latency = [], 0, 0, []
    word_pairs, utt_pairs = [], []
    n_failed, human_words = 0, 0
    for item in manifest:
        rec = responses.get(item["id"])
        if not rec or rec.get("error"):
            n_failed += 1
            h = item.get("human")
            if h and len(h["words"]) == len(item["target"].split()):
                human_words += len(h["words"])
            continue
        latency.append(rec.get("latency_ms", math.nan))
        vflags = vendor_flags(rec, item.get("lang", ""))
        vwords = [{**w, "flag": f} for w, f in zip(rec.get("words", []), vflags or [False] * len(rec.get("words", [])))]
        if "units" in item:
            target = [u["text"] for u in item["units"]]
            aligned = align_spans(item["target"], item["units"], vwords)
            judged = align_spans(item["target"], item["units"], vwords,
                                 pick=lambda hits: {"flag": any(h["flag"] for h in hits)})
        else:
            target = item["target"].split()
            aligned = judged = align(target, vwords)
        scores = [error_signal(e) for e in aligned]
        flags = None if vflags is None else [True if e is None else bool(e["flag"]) for e in judged]

        human = item.get("human")
        if human:
            hw = human["words"]
            if len(hw) != len(target):
                continue
            human_words += len(hw)
            labels = [int(w["mispronounced"]) for w in hw]
            for w, e in zip(hw, aligned):
                if e is not None:
                    word_pairs.append((float(e.get("score", 0.0)), float(w["accuracy"])))
            total = (human.get("utterance") or {}).get("total")
            vendor_utt = [1.0 - x for x in scores]
            served = utterance_score(rec, utt_which)
            if total is not None and (served is not None or vendor_utt):
                utt = served if served is not None else sum(vendor_utt) / len(vendor_utt)
                utt_pairs.append((utt, float(total)))
            units.append((labels, scores, item, flags))
            continue

        label = item.get("label")
        if label and not label.get("word_level_scorable", True):
            continue
        pos = label.get("word_index") if label else None
        keep = list(range(len(target)))
        if exclude_unverified is not None:            # main-branch rule (commit f56cfff, 14 Oct 2026)
            unv = set(exclude_unverified.get(item["id"], ()))
            keep = [i for i in keep if i == pos or i not in unv]
        labels = [int(i == pos) for i in keep]
        scores = [scores[i] for i in keep]
        flags = None if flags is None else [flags[i] for i in keep]
        units.append((labels, scores, item, flags))
        if pos is not None:
            loc_n += 1
            loc_hit += int(keep[max(range(len(scores)), key=lambda i: scores[i])] == pos)
    return {"units": units, "localisation": (loc_hit, loc_n), "latency": latency, "word_pairs": word_pairs,
            "utt_pairs": utt_pairs, "n_failed": n_failed,
            "word_coverage": len(word_pairs) / human_words if human_words else math.nan}


def verdict_metrics(units) -> dict:
    tp = fp = fn = tn = 0
    by_bucket = defaultdict(lambda: [0, 0])
    if any(u[3] is None for u in units):                  # report.py: verdicts undefined
        return {"recall": math.nan, "precision": math.nan, "f1": math.nan, "fa": math.nan, "recall_by_bucket": {}}
    for labels, _, item, flags in units:
        label = item.get("label") or {}
        b = label.get("bucket") or label.get("category") or "human"
        for l, f in zip(labels, flags):
            tp += l and f
            fn += l and not f
            fp += (not l) and f
            tn += (not l) and not f
            if l:
                by_bucket[b][0] += f
                by_bucket[b][1] += 1
    rec = tp / (tp + fn) if tp + fn else math.nan
    prec = tp / (tp + fp) if tp + fp else math.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"recall": rec, "precision": prec, "f1": f1, "fa": fp / (fp + tn) if fp + tn else math.nan,
            "recall_by_bucket": {b: list(v) for b, v in sorted(by_bucket.items())}}


def metrics(manifest, responses, **kw) -> dict:
    from scipy.stats import pearsonr, spearmanr

    c = collect(manifest, responses, **kw)
    units = c["units"]
    labels = [l for u in units for l in u[0]]
    scores = [s for u in units for s in u[1]]
    pairs = [(u[0], u[1]) for u in units]
    hit, n = c["localisation"]
    lat = sorted(x for x in c["latency"] if not math.isnan(x))
    buckets = defaultdict(lambda: ([], []))
    for ls, ss, item, _ in units:
        b = (item.get("label") or {}).get("bucket") or (item.get("label") or {}).get("category", "control")
        for l, s in zip(ls, ss):
            if l:
                buckets[b][0].append(l)
                buckets[b][1].append(s)
    for b in buckets:
        for l, s in zip(labels, scores):
            if not l:
                buckets[b][0].append(l)
                buckets[b][1].append(s)
    m = {"n_utterances": len(units), "n_words": len(labels), "n_positive": sum(labels),
         "n_failed": c["n_failed"], "roc_auc": roc_auc(labels, scores),
         "roc_auc_ci": bootstrap_ci(pairs, roc_auc), "pr_auc": average_precision(labels, scores),
         "fpr_at_recall_50": fpr_at_recall(labels, scores, 0.50),
         "fpr_at_recall_80": fpr_at_recall(labels, scores, 0.80),
         "fpr_at_recall_80_ci": bootstrap_ci(pairs, lambda l, s: fpr_at_recall(l, s, 0.80)),
         "localisation_top1": hit / n if n else math.nan,
         "latency_p50_ms": lat[len(lat) // 2] if lat else math.nan,
         "latency_p95_ms": lat[int(0.95 * len(lat))] if lat else math.nan,
         "by_bucket": {b: roc_auc(*buckets[b]) for b in sorted(buckets) if b != "control"},
         "verdicts": verdict_metrics(units)}
    for name, prs in (("word", c["word_pairs"]), ("utterance", c["utt_pairs"])):
        if len(prs) >= 3:
            a, b = [p[0] for p in prs], [p[1] for p in prs]
            m[f"pcc_{name}"] = float(pearsonr(a, b).statistic)
            m[f"srcc_{name}"] = float(spearmanr(a, b).statistic)
    if c["word_pairs"]:
        m["word_coverage"] = c["word_coverage"]
    return m


def _pcc_ci(manifest, responses, which: str, utt_which="benchmark") -> list[float]:
    """Speaker-level bootstrap of a tier 2 Pearson correlation (our addition; report.py gives none)."""
    import numpy as np
    from scipy.stats import pearsonr

    c = collect(manifest, responses, utt_which=utt_which)
    by_spk = defaultdict(list)
    ok = [it for it in manifest if responses.get(it["id"]) and not responses[it["id"]].get("error")]
    per_item = {}
    for it in ok:
        cc = collect([it], {it["id"]: responses[it["id"]]}, utt_which=utt_which)
        per_item[it["id"]] = cc[f"{which}_pairs"]
        by_spk[it["id"][:4]].append(it["id"])
    spk = sorted(by_spk)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(1000):
        pick = rng.integers(0, len(spk), len(spk))
        prs = [p for s in pick for i in by_spk[spk[s]] for p in per_item[i]]
        vals.append(pearsonr([p[0] for p in prs], [p[1] for p in prs]).statistic)
    del c
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def _clean(x):
    if isinstance(x, float):
        return None if math.isnan(x) else round(x, 4)
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def cmd_report() -> None:
    T = tests()
    main_unv = {}
    for lang in LANGS:
        for it in load_manifest("repo_main_f8750aa", f"tier1/manifest_{lang}.jsonl"):
            main_unv[it["id"]] = it.get("unverified", [])
        # same items in both versions (checked here, not assumed)
        a = [(x["id"], x["target"], x["label"]) for x in T[f"tier1_{lang}"]]
        b = [(x["id"], x["target"], x["label"]) for x in load_manifest("repo_main_f8750aa", f"tier1/manifest_{lang}.jsonl")]
        assert a == b, f"tier 1 {lang} items differ between v2026.10 and main"
    out = {"benchmark": {"repo": f"https://github.com/{REPO}", "rules": "report.py at tag v2026.10 (e021b08)",
                         "variant_main": "report.py at main f8750aa: tier 1 words in `unverified` dropped as negatives",
                         "system": "pronunciada (Scorer(lang) defaults: fp32, strictness 0.5) via to_azure, "
                                   "reduced with the benchmark's Azure adapter"},
           "published_not_rerun_here": PUBLISHED, "tier1": {}, "tier1_main_rules": {}, "tier2": {}}
    try:
        import pronunciada
        out["benchmark"]["pronunciada_version"] = getattr(pronunciada, "__version__", None)
    except Exception:
        pass
    for lang in LANGS:
        R = read_responses(f"tier1_{lang}")
        if not R:
            continue
        out["tier1"][lang] = metrics(T[f"tier1_{lang}"], R)
        out["tier1_main_rules"][lang] = metrics(T[f"tier1_{lang}"], R, exclude_unverified=main_unv)
    R = read_responses("tier2")
    if R:
        m = metrics(T["tier2"], R)
        m["pcc_word_ci_speakers"] = _pcc_ci(T["tier2"], R, "word")
        m["pcc_utterance_ci_speakers"] = _pcc_ci(T["tier2"], R, "utt")
        alt = metrics(T["tier2"], R, utt_which="prob")
        m["side_note_sentence_prob"] = {"pcc_utterance": alt.get("pcc_utterance"),
                                        "srcc_utterance": alt.get("srcc_utterance")}
        m["split"] = "speechocean762 test (all 500 clips; build_tier2.py --split test --seed 42)"
        out["tier2"] = m
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(_clean(out), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_JSON}", file=sys.stderr)
    for lang, m in out["tier1"].items():
        mm = out["tier1_main_rules"][lang]
        print(f"{lang}  AUC {m['roc_auc']:.3f} [{m['roc_auc_ci'][0]:.3f},{m['roc_auc_ci'][1]:.3f}]  "
              f"FA@80 {m['fpr_at_recall_80']:.3f}  loc@1 {m['localisation_top1']:.3f}  failed {m['n_failed']}  "
              f"| main rules AUC {mm['roc_auc']:.3f} FA@80 {mm['fpr_at_recall_80']:.3f}  "
              f"| verdicts R {m['verdicts']['recall']:.3f} FA {m['verdicts']['fa']:.3f}")
    if out["tier2"]:
        m = out["tier2"]
        print(f"tier2  PCC word {m['pcc_word']:.3f}  PCC utt {m['pcc_utterance']:.3f}  "
              f"AUC {m['roc_auc']:.3f}  FA@80 {m['fpr_at_recall_80']:.3f}  coverage {m['word_coverage']:.3f}  "
              f"failed {m['n_failed']}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("fetch", "score", "report", "all"))
    ap.add_argument("--workers", type=int, default=2, help="processes (capped at 2)")
    ap.add_argument("--threads", type=int, default=2, help="threads per Scorer")
    ap.add_argument("--only", nargs="*", help="tests to score, e.g. tier1_en tier2")
    a = ap.parse_args(argv)
    if a.cmd in ("fetch", "all"):
        cmd_fetch()
    if a.cmd in ("score", "all"):
        cmd_score(a.workers, a.threads, a.only)
    if a.cmd in ("report", "all"):
        cmd_report()


if __name__ == "__main__":
    main()
