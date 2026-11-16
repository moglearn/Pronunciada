"""Evaluate any pronunciation scorer on fleurs-swap (text-level planted errors, bench/textswap.py).

    python -m bench.eval_swap run-azure [--dry-run] [--workers 4] [--lang xx] [--limit N]
    python -m bench.eval_swap azure  [--md results/azure_swap_tables.md] [--tables results/tables]
    python -m bench.eval_swap jsonl <outputs.jsonl> --name ours [--md ...] [--tables ...]

SYSTEM-AGNOSTIC INPUT ("outputs"): {record id: output}, one per fleurs-swap record, where
    output = {"status": "Success" (anything else = the system gave no assessment -> excluded),
              "words": [(char_start, char_end, score in [0,1]), ...]   over the record's text,
              "sentence": {"<name>": score in [0,1], ...}               (optional),
              "unmapped": [(char_start, char_end), ...]                 (optional: text regions
                           the adapter could not place the system's output on -> "missing")}
The JSONL form has one {"id": ..., **output} per line (spans as 2-/3-element lists).

Unit scoring: a unit (labels.words[i].span) gets the MINIMUM score over system words whose span
overlaps it; no overlapping word -> 0 (omission), unless the unit overlaps an "unmapped" region
-> None (missing, not an error; reported as the unmapped rate). A swapped record whose replaced
region (labels.char_span_in_new_text) is missing is excluded and counted ("rewritten"), since the
system (e.g. Azure's text normaliser) did not score the word we planted.

Plugging in pronunciada: build, per record, a SentenceResult whose `words` follow the record's text
and the char span of each word (pronunciada RefWord.char_span), then
    outs = {rec["id"]: sentence_result_output(res, spans) for ...}
    evaluate(load("fleurs-swap", "test"), outs)
or write those dicts to JSONL and run `python -m bench.eval_swap jsonl ours.jsonl --name ours`.

Metrics (per language x variant, utterance-clustered bootstrap 95 % CIs, clusters = source
utterance, so orig and swapped versions of one recording are resampled together):
  auc            ROC AUC of (1 - unit score): replaced unit (positive) vs the untouched units
                 of the same swapped records
  hit@5%FA, hit@2%FA
                 threshold set on ORIG records so that 5 % / 2 % of native units are flagged
                 (score below it; ties at the threshold are split fractionally so the native
                 rate is exactly 5 % / 2 %), share of replaced units flagged
  hit<60, hit<80 share of replaced units with score < 0.6 / 0.8 (Azure's natural thresholds)
  right_word     share of swapped records whose replaced unit has the lowest score in the
                 sentence (ties for the lowest share the credit 1/k)
  sent_auc_<k>   ROC AUC of (- sentence score k): swapped records vs their orig records
  native: fa<60, fa<80 (share of native units flagged), mean / median native unit score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from bench.metrics import roc_auc

DATASET = "fleurs-swap"
CJK_CHAR_LANGS = ("zh", "ja")
VARIANTS = ("swap_far", "swap_near", "tone")
_STRIP = re.compile(r"^\W+|\W+$")
ROOT = Path(__file__).resolve().parents[1]


# ============================================================================ Azure runner

class _TimedSession:
    """requests.Session wrapper that records the wall-clock time of the last POST."""

    def __init__(self, s):
        self.s = s
        self.last = None

    def post(self, *a, **k):
        t = time.perf_counter()
        try:
            return self.s.post(*a, **k)
        finally:
            self.last = time.perf_counter() - t


def run_azure(split: str = "test", lang: str | None = None, limit: int | None = None,
              dry_run: bool = False, workers: int = 4, max_tries: int = 12) -> None:
    """Concurrent, resumable Azure run over fleurs-swap. Same cache layout as bench.azure_pa
    (one <id>.json per record, written atomically; cached ids are never re-sent), plus
    latency_s (wall-clock of the successful POST, includes network) and attempts."""
    from bench import azure_pa as AZ
    from bench.data import load

    recs = [r for r in load(DATASET, split=split, lang=lang) if r["lang"] in AZ.LOCALES]
    out = AZ.cache_dir(DATASET)
    todo = [r for r in recs if not (out / f"{r['id']}.json").exists() and r["dur"] <= AZ.MAX_SECONDS]
    n_pending = len(todo)
    if limit is not None:
        todo = todo[:limit]
    secs = defaultdict(float)
    for r in todo:
        secs[AZ.LOCALES[r["lang"]]] += r["dur"]
    print(f"{DATASET}: {len(recs)} records, {len(recs) - n_pending} cached or > 30 s, {len(todo)} to send, "
          f"{sum(secs.values()) / 3600:.2f} audio-h", flush=True)
    for loc, s in sorted(secs.items()):
        print(f"  {loc}: {sum(1 for r in todo if AZ.LOCALES[r['lang']] == loc)} requests, {s / 3600:.2f} h")
    if dry_run or not todo:
        return
    key, region = AZ.credentials()
    import requests

    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".run.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)
            sys.exit(f"another run (pid {pid}) is active on {out}; not starting a second one")
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    lock.write_text(str(os.getpid()))
    local = threading.local()

    def one(r: dict) -> str:
        if not hasattr(local, "s"):
            local.s = requests.Session()
        locale = AZ.LOCALES[r["lang"]]
        params = AZ.pa_params(r["text"], locale)
        audio = AZ.wav_bytes(r["audio"])
        rnd = random.Random(r["id"])
        for attempt in range(max_tries):
            ts = _TimedSession(local.s)
            try:
                resp = AZ.send(ts, key, region, locale, params, audio, max_tries=1)
            except RuntimeError as e:
                if str(e).startswith("Azure returned HTTP"):
                    return f"fail {r['id']}: {str(e)[:120]}"
                time.sleep(min(300.0, 5.0 * 2 ** attempt) * (0.75 + 0.5 * rnd.random()))
                continue
            rec = {"id": r["id"], "locale": locale, "params": params,
                   "audio_sha1": hashlib.sha1(Path(r["audio"]).read_bytes()).hexdigest(),
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "latency_s": ts.last,
                   "attempts": attempt + 1, "audio_dur": r["dur"], "response": resp}
            tmp = out / f"{r['id']}.json.tmp{threading.get_ident()}"
            tmp.write_text(json.dumps(rec, ensure_ascii=False))
            os.replace(tmp, out / f"{r['id']}.json")
            return "ok"
        return f"fail {r['id']}: giving up after {max_tries} tries"

    n_ok = n_fail = 0
    t0 = time.time()
    try:
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(one, r) for r in todo]
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res == "ok":
                    n_ok += 1
                else:
                    n_fail += 1
                    print(res, flush=True)
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(todo)} ({n_fail} failed, {time.time() - t0:.0f} s)", flush=True)
    finally:
        lock.unlink(missing_ok=True)
    print(f"done: {n_ok} ok, {n_fail} failed, {time.time() - t0:.0f} s", flush=True)


# ============================================================================ adapters

def _norm(s: str) -> str:
    from bench.azure_pa import _norm as n
    return n(s)


def _text_tokens(text: str) -> list[tuple[str, int, int]]:
    out = []
    for m in re.finditer(r"\S+", text):
        tok = m.group(0)
        core = _STRIP.sub("", tok)
        if not core:
            continue
        a = m.start() + tok.index(core)
        out.append((core, a, a + len(core)))
    return out


def _az_score(w: dict) -> float | None:
    if w["error_type"] == "Omission":
        return 0.0
    return None if w["accuracy"] is None else w["accuracy"] / 100.0


def azure_map(text: str, lang: str, parsed: dict) -> dict:
    """Place Azure's word list (bench.azure_pa.parse_response) onto the record text.

    Space-delimited languages: sequential matching of normalised tokens (difflib opcodes);
    'equal' runs map one-to-one; a 'replace' run whose two sides spell the same letters
    (e.g. a hyphenated token split by Azure) is mapped by characters; a single text token with a
    digit replaced by Azure's spelled-out words gets all of them. zh / ja: character offsets
    (character-level matching of the normalised strings). Insertions are ignored; Omission -> 0.
    Text that cannot be placed is returned as "unmapped"."""
    az = [w for w in parsed["words"] if w["error_type"] != "Insertion"]
    words, unmapped = [], []
    if lang in CJK_CHAR_LANGS:
        t_chars = [(c, i) for i, ch in enumerate(text) for c in _norm(ch)]
        a_chars = [(c, k) for k, w in enumerate(az) for c in _norm(w["word"] or "")]
        sm = SequenceMatcher(None, [c for c, _ in t_chars], [c for c, _ in a_chars], autojunk=False)
        hit = {}
        for b in sm.get_matching_blocks():
            for t in range(b.size):
                hit[t_chars[b.a + t][1]] = a_chars[b.b + t][1]
        for c, i in t_chars:
            if i in hit:
                s = _az_score(az[hit[i]])
                if s is None:
                    unmapped.append((i, i + 1))
                else:
                    words.append((i, i + 1, s))
            else:
                unmapped.append((i, i + 1))
        return {"words": words, "unmapped": unmapped}
    toks = [t for t in _text_tokens(text) if _norm(t[0])]
    tn = [_norm(t[0]) for t in toks]
    an = [_norm(w["word"] or "") for w in az]
    sm = SequenceMatcher(None, tn, an, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for d in range(i2 - i1):
                s = _az_score(az[j1 + d])
                t = toks[i1 + d]
                (words.append((t[1], t[2], s)) if s is not None else unmapped.append((t[1], t[2])))
        elif op == "replace" and "".join(tn[i1:i2]) == "".join(an[j1:j2]):
            owner = [k for k in range(j1, j2) for _ in an[k]]
            pos = 0
            for i in range(i1, i2):
                ks = sorted(set(owner[pos:pos + len(tn[i])]))
                pos += len(tn[i])
                ss = [_az_score(az[k]) for k in ks]
                ss = [s for s in ss if s is not None]
                t = toks[i]
                (words.append((t[1], t[2], min(ss))) if ss else unmapped.append((t[1], t[2])))
        elif op == "replace" and i2 - i1 == 1 and any(ch.isdigit() for ch in toks[i1][0]):
            ss = [s for s in (_az_score(az[k]) for k in range(j1, j2)) if s is not None]
            t = toks[i1]
            (words.append((t[1], t[2], min(ss))) if ss else unmapped.append((t[1], t[2])))
        else:
            for i in range(i1, i2):
                unmapped.append((toks[i][1], toks[i][2]))
    return {"words": words, "unmapped": unmapped}


def azure_outputs(records: list[dict]) -> dict:
    """Outputs (system-agnostic format) from the cached Azure responses of fleurs-swap.
    Sentence scores: AccuracyScore and PronScore, /100. Adds latency_s / audio_dur."""
    from bench.azure_pa import cache_dir, parse_response

    d = cache_dir(DATASET)
    outs = {}
    for r in records:
        p = d / f"{r['id']}.json"
        if not p.exists():
            continue
        raw = json.loads(p.read_text())
        parsed = parse_response(raw["response"])
        o = {"status": parsed["status"], "latency_s": raw.get("latency_s"), "audio_dur": r["dur"],
             "locale": raw.get("locale")}
        if parsed["status"] == "Success":
            o.update(azure_map(r["text"], r["lang"], parsed))
            s = parsed["sentence"] or {}
            o["sentence"] = {k: (None if s.get(k) is None else s[k] / 100.0)
                             for k in ("AccuracyScore", "PronScore")}
            o["recognized"] = s.get("recognized")
        outs[r["id"]] = o
    return outs


def sentence_result_output(res, char_spans: list[tuple[int, int]]) -> dict:
    """Adapter for pronunciada: res = pronunciada.types.SentenceResult for the record's text, char_spans =
    the char span (in the record text) of each res.words[i] (e.g. from RefWord.char_span).
    The word score is WordResult.prob, which since round 4 already contains the probability that
    the word was said at all (earlier rounds scored a word with the hard omitted flag as 0).
    Sentence score key "prob"."""
    if len(char_spans) != len(res.words):
        raise ValueError("one char span per SentenceResult word is required")
    words = [(int(a), int(b), float(w.prob) if w.present_prob is not None or not w.omitted else 0.0)
             for w, (a, b) in zip(res.words, char_spans)]
    return {"status": "Success", "words": words, "sentence": {"prob": float(res.prob)}, "unmapped": []}


def load_jsonl_outputs(path: str) -> dict:
    outs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                o.setdefault("status", "Success")
                outs[o.pop("id")] = o
    return outs


# ============================================================================ scoring

def _overlap(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


def unit_scores(units: list[dict], out: dict) -> list[float | None]:
    """Score per unit: min over overlapping system words; none -> 0 (omission) unless the unit
    overlaps an unmapped region -> None."""
    words = out.get("words") or []
    um = out.get("unmapped") or []
    res = []
    for u in units:
        a, b = u["span"]
        ss = [w[2] for w in words if _overlap(a, b, w[0], w[1])]
        if ss:
            res.append(float(min(ss)))
        elif any(_overlap(a, b, x[0], x[1]) for x in um):
            res.append(None)
        else:
            res.append(0.0)
    return res


def _hit_at_fa(native: np.ndarray, pos: np.ndarray, alpha: float) -> float:
    """Flag = score < t, plus a fraction q of scores == t, with t, q set so that exactly alpha of
    native scores are flagged; returns the share of positives flagged."""
    if len(native) == 0 or len(pos) == 0:
        return float("nan")
    ns = np.sort(native)
    k = alpha * len(ns)
    idx = min(int(np.floor(k)), len(ns) - 1)
    t = ns[idx]
    below = np.mean(native < t)
    eq = np.mean(native == t)
    q = 0.0 if eq == 0 else min(1.0, max(0.0, (alpha - below) / eq))
    return float(np.mean(pos < t) + q * np.mean(pos == t))


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc(y, np.r_[-pos, -neg])


def collect(records: list[dict], outputs: dict) -> dict:
    """Per language: clusters {src_id: {...}} + counts."""
    by_lang: dict = defaultdict(lambda: {"clusters": defaultdict(dict), "counts": defaultdict(int)})
    for r in records:
        lab = r["labels"]
        L = by_lang[r["lang"]]
        v = lab["variant"]
        c = L["counts"]
        c[f"{v}:records"] += 1
        o = outputs.get(r["id"])
        if o is None:
            c[f"{v}:no_output"] += 1
            continue
        if o.get("status") != "Success":
            c[f"{v}:failed"] += 1
            continue
        us = unit_scores(lab["words"], o)
        c["units"] += len(us)
        c["units_unmapped"] += sum(s is None for s in us)
        c[f"{v}:units"] += len(us)
        c[f"{v}:units_unmapped"] += sum(s is None for s in us)
        sent = {k: s for k, s in (o.get("sentence") or {}).items() if s is not None}
        entry = {"scores": us, "sent": sent}
        if v != "orig":
            k = lab["word_index"]
            a, b = lab["char_span_in_new_text"]
            rewritten = us[k] is None or any(_overlap(a, b, x[0], x[1]) for x in (o.get("unmapped") or []))
            if rewritten:
                c[f"{v}:rewritten"] += 1
                continue
            entry["pos"] = us[k]
            entry["neg"] = [s for i, s in enumerate(us) if i != k and s is not None]
        c[f"{v}:used"] += 1
        L["clusters"][lab["src_id"]][v] = entry
    return by_lang


def _metrics(clusters: list[dict], variants) -> dict:
    out = {}
    native = np.array([s for c in clusters if "orig" in c for s in c["orig"]["scores"] if s is not None])
    out["native:n_units"] = len(native)
    out["native:fa<60"] = float(np.mean(native < 0.6)) if len(native) else float("nan")
    out["native:fa<80"] = float(np.mean(native < 0.8)) if len(native) else float("nan")
    out["native:mean"] = float(native.mean()) if len(native) else float("nan")
    out["native:median"] = float(np.median(native)) if len(native) else float("nan")
    for v in variants:
        sw = [c for c in clusters if v in c]
        if not sw:
            continue
        pos = np.array([c[v]["pos"] for c in sw])
        neg = np.array([s for c in sw for s in c[v]["neg"]])
        out[f"{v}:n"] = len(sw)
        out[f"{v}:auc"] = _auc(pos, neg)
        out[f"{v}:hit@5%FA"] = _hit_at_fa(native, pos, 0.05)
        out[f"{v}:hit@2%FA"] = _hit_at_fa(native, pos, 0.02)
        out[f"{v}:hit<60"] = float(np.mean(pos < 0.6))
        out[f"{v}:hit<80"] = float(np.mean(pos < 0.8))
        out[f"{v}:others_fa<60"] = float(np.mean(neg < 0.6)) if len(neg) else float("nan")
        out[f"{v}:pos_mean"] = float(pos.mean())
        cred = []
        for c in sw:
            p, ng = c[v]["pos"], c[v]["neg"]
            m = min([p] + ng)
            cred.append(0.0 if p > m else 1.0 / (1 + sum(1 for s in ng if s == m)))
        out[f"{v}:right_word"] = float(np.mean(cred))
        keys = sorted({k for c in sw for k in c[v]["sent"]})
        for k in keys:
            pairs = [(c[v]["sent"][k], c["orig"]["sent"][k]) for c in sw
                     if "orig" in c and k in c[v]["sent"] and k in c["orig"]["sent"]]
            if pairs:
                ps, os_ = np.array(pairs).T
                out[f"{v}:sent_auc_{k}"] = _auc(ps, os_)
                out[f"{v}:sent_drop_{k}"] = float(np.mean(os_ - ps))
    return out


def evaluate(records: list[dict], outputs: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    """{lang: {"counts": {...}, "metrics": {name: {value, lo, hi}}}} -- see module docstring."""
    res = {}
    for lang, L in sorted(collect(records, outputs).items()):
        cl = [L["clusters"][k] for k in sorted(L["clusters"])]
        variants = [v for v in VARIANTS if any(v in c for c in cl)]
        point = _metrics(cl, variants)
        rng = np.random.default_rng(seed)
        boots = defaultdict(list)
        for _ in range(n_boot):
            idx = rng.integers(0, len(cl), len(cl))
            m = _metrics([cl[i] for i in idx], variants)
            for k, v in m.items():
                if isinstance(v, float) and np.isfinite(v):
                    boots[k].append(v)
        met = {}
        for k, v in point.items():
            b = boots.get(k)
            if isinstance(v, float) and b:
                lo, hi = np.quantile(b, [0.025, 0.975])
                met[k] = {"value": v, "lo": float(lo), "hi": float(hi)}
            else:
                met[k] = {"value": v}
        cnt = dict(L["counts"])
        cnt["unmapped_rate"] = cnt.get("units_unmapped", 0) / max(cnt.get("units", 0), 1)
        res[lang] = {"counts": cnt, "metrics": met}
    return res


def latency_summary(outputs: dict) -> dict:
    by = defaultdict(list)
    for o in outputs.values():
        if o.get("latency_s") is not None:
            by[o.get("locale") or "?"].append((o["latency_s"], o["audio_dur"]))
    out = {}
    for loc, xs in sorted(by.items()):
        lat = np.array([x[0] for x in xs])
        dur = np.array([x[1] for x in xs])
        out[loc] = {"n": len(xs), "median_s": float(np.median(lat)), "p90_s": float(np.quantile(lat, 0.9)),
                    "mean_s": float(lat.mean()), "mean_audio_s": float(dur.mean()),
                    "median_rtf": float(np.median(lat / dur))}
    return out


# ============================================================================ report

def _f(m: dict, k: str, pct: bool = False, ci: bool = True) -> str:
    x = m.get(k)
    if not x or x.get("value") is None or (isinstance(x["value"], float) and not np.isfinite(x["value"])):
        return "–"
    sc = 100 if pct else 1
    fmt = "{:.1f}" if pct else "{:.3f}"
    s = fmt.format(x["value"] * sc)
    if ci and "lo" in x:
        s += f" [{fmt.format(x['lo'] * sc)}, {fmt.format(x['hi'] * sc)}]"
    return s


def markdown(res: dict, name: str) -> str:
    L = []
    for v in VARIANTS:
        rows = [(lang, r) for lang, r in res.items() if f"{v}:auc" in r["metrics"]]
        if not rows:
            continue
        L.append(f"\n### {name}: {v}\n")
        L.append("| lang | n | AUC | hit@5%FA % | hit@2%FA % | hit<60 % | hit<80 % | right word % | "
                 "sent AUC (Acc) | sent AUC (Pron) | excluded (rewritten / failed) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for lang, r in rows:
            m, c = r["metrics"], r["counts"]
            sk = sorted(k for k in m if k.startswith(f"{v}:sent_auc_"))
            sa = {k.split("sent_auc_")[1]: _f(m, k, ci=True) for k in sk}
            L.append(f"| {lang} | {m[f'{v}:n']['value']} | {_f(m, f'{v}:auc')} | {_f(m, f'{v}:hit@5%FA', True)} | "
                     f"{_f(m, f'{v}:hit@2%FA', True)} | {_f(m, f'{v}:hit<60', True, False)} | "
                     f"{_f(m, f'{v}:hit<80', True, False)} | {_f(m, f'{v}:right_word', True)} | "
                     f"{sa.get('AccuracyScore', sa.get('prob', '–'))} | {sa.get('PronScore', '–')} | "
                     f"{c.get(f'{v}:rewritten', 0)} / {c.get(f'{v}:failed', 0)} |")
    L.append(f"\n### {name}: native (orig) units\n")
    L.append("| lang | orig records used | units | FA <60 % | FA <80 % | mean | median | unmapped units % | failed |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for lang, r in res.items():
        m, c = r["metrics"], r["counts"]
        L.append(f"| {lang} | {c.get('orig:used', 0)} | {m['native:n_units']['value']} | "
                 f"{_f(m, 'native:fa<60', True)} | {_f(m, 'native:fa<80', True)} | "
                 f"{_f(m, 'native:mean', ci=False)} | {_f(m, 'native:median', ci=False)} | "
                 f"{100 * c['unmapped_rate']:.2f} | {sum(v for k, v in c.items() if k.endswith(':failed'))} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run-azure")
    r.add_argument("--lang")
    r.add_argument("--limit", type=int)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--workers", type=int, default=4)
    for name in ("azure", "jsonl"):
        p = sub.add_parser(name)
        if name == "jsonl":
            p.add_argument("path")
            p.add_argument("--name", default="system")
        p.add_argument("--md")
        p.add_argument("--tables", help="directory for <name>_swap_<lang>.json")
        p.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    if a.cmd == "run-azure":
        run_azure(lang=a.lang, limit=a.limit, dry_run=a.dry_run, workers=a.workers)
        return
    from bench.data import load

    recs = load(DATASET, split="test")
    if a.cmd == "azure":
        name, outs = "azure", azure_outputs(recs)
    else:
        name, outs = a.name, load_jsonl_outputs(a.path)
    res = evaluate(recs, outs, n_boot=a.n_boot)
    lat = latency_summary(outs)
    md = markdown(res, name)
    print(md)
    if lat:
        print(json.dumps(lat, indent=1))
    if a.md:
        Path(a.md).write_text(md)
    if a.tables:
        Path(a.tables).mkdir(parents=True, exist_ok=True)
        from bench.azure_pa import LOCALES
        for lang, rr in res.items():
            rr = dict(rr, latency=lat.get(LOCALES.get(lang, "?")), system=name, dataset=DATASET)
            (Path(a.tables) / f"{name}_swap_{lang}.json").write_text(json.dumps(rr, indent=1))


if __name__ == "__main__":
    main()
