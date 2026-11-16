"""Build the "cv-real" benchmark: real human judgements on real Common Voice speech.

    python -m bench.cv_build meta     [--locales all]   # TSVs (votes, accents, invalidated list)
    python -m bench.cv_build select   [--locales all]   # selection with reserves (metadata only)
    python -m bench.cv_build fetch    [--locales all]   # stream the needed tar shards, keep only
                                                        # the selected clips (16 kHz FLAC)
    python -m bench.cv_build finalize [--locales all]   # silence rule + final matched manifest
    python -m bench.cv_build all      [--locales all]

Source (all locales): Mozilla Common Voice Corpus 22.0, CC0-1.0, Hugging Face mirror
``fsicoli/common_voice_22_0`` pinned at revision REV (transcript/<locale>/*.tsv and
audio/<locale>/<split>/<locale>_<split>_<k>.tar). Only the TSVs and the tar shards listed in
the manifest's ``source`` are read; tar shards are streamed and only the selected clips are
written to disk.

Test A (reviewer judgement): positives = clips in ``invalidated`` with down_votes >= 2 and
up_votes <= down_votes - 2; negatives = clips in ``dev`` / ``test`` (validated, i.e. accepted)
with up_votes >= 2 and down_votes == 0.
Test B (self-declared accent, accepted clips only): positives = speakers whose ``accents``
label marks them as non-native, negatives = speakers labelled with a native variety.

Common rules (both tests): clips < 1 s or > 15 s dropped (clip_durations.tsv); prompts with
digits or Latin letters dropped in non-Latin-script locales; near-silent audio dropped
(fewer than 0.3 s of 25 ms frames above -45 dBFS; applied after fetching); at most 2 clips
per speaker; no speaker and no (normalised) sentence on both sides; dev / test halves split by
speaker hash (seed SEED); the two sides are matched by stratified sampling on (characters,
duration) bins, with the same number of clips per stratum on each side.

Outputs: bench/data/cv_real/<locale>.jsonl (ids + metadata, no audio, no paths),
bench/data/cv_real/build.json (cleaning counts, sources). Audio:
$PRONUNCIADA_CACHE/data/cv-real/audio/<locale>/<stem>.flac.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = "fsicoli/common_voice_22_0"
REV = "ae911de250fe9375d08e2daa2105b671d659d446"
CV_VERSION = "Common Voice Corpus 22.0"
LICENSE = "CC0-1.0"
HF = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}"
SEED = 20261116
LOCALES = ("en", "zh-CN", "zh-TW", "ja", "es", "fr", "de", "it", "pt", "ru", "ko")
SCORER_LANG = {"en": "en", "zh-CN": "zh", "zh-TW": "zh", "ja": "ja", "es": "es", "fr": "fr",
               "de": "de", "it": "it", "pt": "pt", "ru": "ru", "ko": "ko"}
NON_LATIN = {"zh-CN", "zh-TW", "ja", "ru", "ko"}
TARGET = 200            # per side per half
PER_SPEAKER = 2
MIN_DUR, MAX_DUR = 1.0, 15.0
LEN_BINS, DUR_BINS = 5, 4
RESERVE = 1.35          # fetch this many candidates per stratum slot (silence-rule reserve)
TESTB_MIN_CLIPS, TESTB_MIN_SPK = 100, 30
SHARD_ROWS = 40000      # rows per tar shard in the mirror
WEAK_LOCALES = ("ko",)  # Test A-weak: other.tsv single-down-vote clips added to the positives
WEAK_MIN_POS = 60       # A-weak is built only if this many positives survive the cleaning rules
SIL_DBFS, SIL_MIN_ACTIVE_S = -45.0, 0.3

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "bench" / "data" / "cv_real"
csv.field_size_limit(1 << 30)


def cache_root() -> Path:
    base = os.environ.get("PRONUNCIADA_CACHE")
    return Path(base).expanduser() if base else Path.home() / ".cache" / "pronunciada"


def work_dir(*p) -> Path:
    d = cache_root().joinpath("data", "cv-real", *p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def audio_path(locale: str, stem: str) -> Path:
    return cache_root() / "data" / "cv-real" / "audio" / locale / f"{stem}.flac"


# ============================================================================ pure rules

_LATIN = re.compile(r"[A-Za-zÀ-ɏＡ-Ｚａ-ｚ]")
_DIGIT = re.compile(r"[0-9０-９]")


def text_rule(sentence: str, locale: str) -> str | None:
    """Reason a prompt is not usable, or None. Only non-Latin-script locales are checked."""
    if not sentence or not sentence.strip():
        return "empty_prompt"
    if locale in NON_LATIN:
        if _DIGIT.search(sentence):
            return "digits_in_prompt"
        if _LATIN.search(sentence):
            return "latin_in_prompt"
    return None


def dur_rule(dur: float | None) -> str | None:
    if dur is None:
        return "no_duration"
    if dur < MIN_DUR:
        return "too_short"
    if dur > MAX_DUR:
        return "too_long"
    return None


def is_rejected(up: int, down: int) -> bool:
    return down >= 2 and up <= down - 2


def is_weak_rejected(up: int, down: int) -> bool:
    """other.tsv (too few votes to be validated / invalidated): at least one down-vote, no up-vote."""
    return down >= 1 and up == 0


def is_accepted(up: int, down: int) -> bool:
    return up >= 2 and down == 0


def norm_sentence(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(ch for ch in s if unicodedata.category(ch)[0] in "LN")


def n_chars(s: str) -> int:
    return len(norm_sentence(s))


def frame_levels_db(x: np.ndarray, sr: int = 16000, win: float = 0.025, hop: float = 0.010) -> np.ndarray:
    w, h = int(win * sr), int(hop * sr)
    if len(x) < w:
        x = np.pad(x, (0, w - len(x)))
    n = 1 + (len(x) - w) // h
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(x[idx].astype(np.float64) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def active_seconds(x: np.ndarray, sr: int = 16000, thr_db: float = SIL_DBFS) -> float:
    return float((frame_levels_db(x, sr) > thr_db).sum() * 0.010)


def near_silent(x: np.ndarray, sr: int = 16000) -> bool:
    return active_seconds(x, sr) < SIL_MIN_ACTIVE_S


def speaker_half(client_id: str, seed: int = SEED) -> str:
    h = hashlib.sha256(f"{seed}|{client_id}".encode()).digest()
    return "test" if h[0] % 2 == 0 else "dev"


def _coin(key: str, seed: int = SEED) -> bool:
    return hashlib.sha256(f"{seed}|side|{key}".encode()).digest()[0] % 2 == 0


# ----------------------------------------------------------------------------- accents

LATAM_KW = ("méxico", "mexic", "andino", "caribe", "rioplatense", "américa central", "america central",
            "chileno", "chile", "argentin", "colombia", "venezuel", "perú", "peru", "ecuador",
            "bolivia", "uruguay", "paraguay", "cuba", "puerto rico", "dominican", "panamá", "guatemala",
            "honduras", "salvador", "nicaragua", "costa rica", "latin")
SPAIN_KW = ("españa", "espana", "castellano", "peninsular", "andaluz", "canari", "catalu", "madrid")
BRAZIL_KW = ("brasil", "brazil", "paulist", "carioca", "mineir", "nordestin", "gaúch", "gauch", "sulista",
             "baian", "cearense", "manauara", "catarinense", "pernambuc", "paraense", "goian", "capixaba",
             "potiguar", "manezinho", "caipira", "fortaleza", "recife", "são paulo", "sao paulo")
PORTUGAL_KW = ("portugal", "lisboa", "europeu", "lusitan", "açor", "madeirense", "alentej", "minhot")


def es_region(accent: str) -> str:
    a = accent.lower()
    la, sp = any(k in a for k in LATAM_KW), any(k in a for k in SPAIN_KW)
    return "latam" if la and not sp else "spain" if sp and not la else "mixed" if (la and sp) else "none"


def pt_region(accent: str, variant: str) -> str:
    a, v = accent.lower(), (variant or "").lower()
    br = "brasil" in v or any(k in a for k in BRAZIL_KW)
    pt = "portugal" in v or any(k in a for k in PORTUGAL_KW)
    return "brazil" if br and not pt else "portugal" if pt and not br else "mixed" if (br and pt) else "none"


_GENERIC_NONNATIVE = re.compile(
    r"non[- ]?native|not (a )?native|second language|foreign|\bl2\b|\besl\b|learner|"
    r"nicht[- ]?muttersprach|fremdsprach|non natif|non-natif|langue seconde|étranger|etranger|"
    r"no nativ|extranjer|segunda lengua|non madrelingua|stranier|não nativ|nao nativ|estrangeir|"
    r"иностран|не носитель|неродн|외국인|외국어|非ネイティブ|外国人", re.I)
_L1_NAMES = ("german", "french", "dutch", "russian", "spanish", "italian", "polish", "portuguese",
             "brazilian", "chinese", "mandarin", "cantonese", "japanese", "korean", "arabic", "turkish",
             "hungarian", "swedish", "norwegian", "danish", "finnish", "greek", "czech", "romanian",
             "ukrainian", "persian", "farsi", "vietnamese", "thai", "indonesian", "hebrew", "israeli",
             "mexican", "argentin", "colombian", "slovak", "bulgarian", "serbian", "croatian", "latvian",
             "lithuanian", "estonian", "catalan", "basque", "swiss", "austrian", "belgian", "slovenian",
             "albanian", "bosnian", "macedonian", "belarus", "kazakh", "georgian", "armenian",
             "azerbaijan", "uzbek", "venezuelan", "chilean", "peruvian", "cuban", "egyptian",
             "moroccan", "tunisian", "algerian", "iranian", "kurdish", "icelandic")
EN_NATIVE = ("united states english", "england english", "canadian english", "australian english",
             "scottish english", "new zealand english", "irish english", "welsh english")
DE_NONNATIVE = ("amerikanisch", "niederländisch", "französisch", "tschechisch", "russisch", "britisch",
                "ungarisch", "polnisch", "italienisch", "türkisch", "spanisch", "brasilianisch",
                "kanadisch", "griechisch", "dänisch", "schwedisch", "norwegisch", "finnisch", "rumänisch",
                "bulgarisch", "ukrainisch", "slowakisch", "slowenisch", "kroatisch", "serbisch",
                "bosnisch", "litauisch", "lettisch", "estnisch", "chinesisch", "japanisch", "koreanisch",
                "arabisch", "persisch", "indisch", "portugiesisch", "hebräisch", "albanisch", "kurdisch",
                "australisch", "neuseeländisch", "irisch", "schottisch", "englisch", "isländisch")
DE_NATIVE = ("deutschland deutsch", "österreichisch", "schweizerdeutsch", "hochdeutsch")
FR_NONNATIVE = ("d’allemagne", "d'allemagne", "des états-unis", "des etats-unis", "du royaume-uni",
                "de russie", "d’italie", "d'italie", "d’espagne", "d'espagne", "des pays-bas", "de chine",
                "du brésil", "du bresil", "du japon", "de pologne", "du portugal", "de roumanie",
                "d’angleterre", "d'angleterre", "de turquie", "d’iran", "d'iran", "de corée", "d’inde",
                "d'inde", "du mexique", "d’argentine", "de grèce", "de suède", "d’ukraine", "d'ukraine")
FR_NATIVE = ("français de france", "français de belgique", "français de suisse", "français du canada",
             "québécois", "parisien")


def nativeness(locale: str, accent: str) -> str:
    """'nonnative' / 'native' / 'unknown' from a self-declared Common Voice accent string.
    Conservative: established national varieties (Indian, Singaporean, African French, ...) and
    regional dialect labels count as neither unless the locale has a native-variety list."""
    a = (accent or "").strip().lower()
    if not a:
        return "unknown"
    nn = bool(_GENERIC_NONNATIVE.search(a))
    if locale == "en":
        if re.search(r"cajun|american english|canadian|\bamerican\b", a):
            return "unknown"      # heritage / North-American labels: possibly native
        nn = nn or any(re.search(rf"\b{k}", a) for k in _L1_NAMES)
        nat = any(k in a for k in EN_NATIVE)
    elif locale == "de":
        nn = nn or any(f"{k} deutsch" in a or f"{k}es deutsch" in a for k in DE_NONNATIVE)
        nat = any(k in a for k in DE_NATIVE)
    elif locale == "fr":
        nn = nn or any(k in a for k in FR_NONNATIVE)
        nat = any(k in a for k in FR_NATIVE)
    elif locale == "pt":
        nn = nn or bool(re.search(r"nativo de (inglês|ingles|espanhol|alemão|francês)|native (english|spanish)", a))
        nat = pt_region(accent, "") == "brazil" or "nativ" in a and not nn
    else:
        nat = False
    if nn and not nat:
        return "nonnative"
    if nat and not nn:
        return "native"
    return "unknown"


# ----------------------------------------------------------------------------- selection

def cap_per_speaker(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["client_id"]].append(r)
    out = []
    for spk in sorted(by):
        rs = sorted(by[spk], key=lambda r: r["stem"])
        rng.shuffle(rs)
        out += rs[:k]
    return out


def make_disjoint(pos: list[dict], neg: list[dict], seed: int = SEED) -> tuple[list, list, Counter]:
    """No speaker and no normalised sentence on both sides: each shared speaker / sentence is
    given to one side by a seeded coin (neutral w.r.t. either system)."""
    cnt = Counter()
    shared_spk = {r["client_id"] for r in pos} & {r["client_id"] for r in neg}
    keep_pos = {s for s in shared_spk if _coin("spk" + s, seed)}
    p2 = [r for r in pos if r["client_id"] not in shared_spk or r["client_id"] in keep_pos]
    n2 = [r for r in neg if r["client_id"] not in shared_spk or r["client_id"] not in keep_pos]
    cnt["speaker_on_both_sides"] = (len(pos) - len(p2)) + (len(neg) - len(n2))
    shared_txt = {r["norm"] for r in p2} & {r["norm"] for r in n2}
    keep_pos_t = {t for t in shared_txt if _coin("txt" + t, seed)}
    p3 = [r for r in p2 if r["norm"] not in shared_txt or r["norm"] in keep_pos_t]
    n3 = [r for r in n2 if r["norm"] not in shared_txt or r["norm"] not in keep_pos_t]
    cnt["sentence_on_both_sides"] = (len(p2) - len(p3)) + (len(n2) - len(n3))
    return p3, n3, cnt


def _edges(vals: list[float], nb: int) -> np.ndarray:
    if not vals:
        return np.array([])
    q = np.quantile(np.asarray(vals, dtype=float), np.linspace(0, 1, nb + 1)[1:-1])
    return np.unique(q)


def stratum_of(r: dict, len_edges: np.ndarray, dur_edges: np.ndarray) -> tuple[int, int]:
    return int(np.searchsorted(len_edges, r["n_chars"], side="right")), \
        int(np.searchsorted(dur_edges, r["dur"], side="right"))


def allocate(pos_counts: dict, neg_counts: dict, target: int) -> dict:
    """Same quota per stratum on both sides: proportional to the positives' distribution, capped
    by min(pos, neg) available, deficit refilled from strata with spare capacity."""
    strata = sorted(set(pos_counts) | set(neg_counts))
    cap = {s: min(pos_counts.get(s, 0), neg_counts.get(s, 0)) for s in strata}
    total_pos = sum(pos_counts.values())
    target = min(target, sum(cap.values()))
    if target <= 0 or total_pos == 0:
        return {s: 0 for s in strata}
    raw = {s: target * pos_counts.get(s, 0) / total_pos for s in strata}
    q = {s: min(cap[s], int(np.floor(raw[s]))) for s in strata}
    while sum(q.values()) < target:
        spare = [s for s in strata if q[s] < cap[s]]
        if not spare:
            break
        s = max(spare, key=lambda s: (raw[s] - q[s], pos_counts.get(s, 0), s))
        q[s] += 1
    return q


def select_matched(pos: list[dict], neg: list[dict], target: int, seed: int, reserve: float = 1.0):
    """-> (quota per stratum, {stratum: ordered pos list}, {stratum: ordered neg list}, edges).
    Lists are length ceil(quota * reserve) (+ up to what exists) in a seeded random order; the
    first `quota` passing the audio check on each side form the final set."""
    len_edges = _edges([r["n_chars"] for r in pos], LEN_BINS)
    dur_edges = _edges([r["dur"] for r in pos], DUR_BINS)
    bp, bn = defaultdict(list), defaultdict(list)
    for r in pos:
        bp[stratum_of(r, len_edges, dur_edges)].append(r)
    for r in neg:
        bn[stratum_of(r, len_edges, dur_edges)].append(r)
    q = allocate({s: len(v) for s, v in bp.items()}, {s: len(v) for s, v in bn.items()}, target)
    rng = random.Random(seed)
    lp, ln = {}, {}
    for s in sorted(q):
        if q[s] == 0:
            continue
        n_take = int(np.ceil(q[s] * reserve)) + (2 if reserve > 1 else 0)
        a = sorted(bp[s], key=lambda r: r["stem"])
        b = sorted(bn[s], key=lambda r: r["stem"])
        rng.shuffle(a)
        rng.shuffle(b)
        lp[s], ln[s] = a[:n_take], b[:n_take]
    return q, lp, ln, (len_edges.tolist(), dur_edges.tolist())


def finalize_stratum(quota: int, pos_list: list[dict], neg_list: list[dict], ok) -> tuple[list, list]:
    """Take the first passing clips on each side, equal numbers (<= quota) on both sides."""
    p = [r for r in pos_list if ok(r)]
    n = [r for r in neg_list if ok(r)]
    k = min(quota, len(p), len(n))
    return p[:k], n[:k]


# ============================================================================ metadata I/O

def meta_dir(locale: str) -> Path:
    return work_dir("meta", locale)


def step_meta(locales) -> None:
    import requests

    jobs = [(l, f) for l in locales for f in ("invalidated", "dev", "test", "clip_durations")
            if not (meta_dir(l) / f"{f}.tsv").exists()]

    def one(job):
        l, f = job
        dst = meta_dir(l) / f"{f}.tsv"
        r = requests.get(f"{HF}/transcript/{l}/{f}.tsv", timeout=120)
        r.raise_for_status()
        tmp = dst.with_suffix(".tmp")
        tmp.write_bytes(r.content)
        os.replace(tmp, dst)
        return dst

    with ThreadPoolExecutor(8) as ex:
        for p in ex.map(one, jobs):
            print("got", p.parent.name, p.name, flush=True)


def read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def durations(locale: str) -> dict[str, float]:
    out = {}
    with open(meta_dir(locale) / "clip_durations.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1].strip().isdigit():
                out[p[0]] = int(p[1]) / 1000.0
    return out


def n_shards(locale: str) -> dict:
    p = work_dir("meta") / "n_shards.json"
    if not p.exists():
        import requests

        p.write_text(requests.get(f"{HF}/n_shards.json", timeout=60).text)
    return json.loads(p.read_text())[locale]


def shard_url(locale: str, split: str, k: int) -> str:
    return f"{HF}/audio/{locale}/{split}/{locale}_{split}_{k}.tar"


def first_member(url: str) -> str:
    import io
    import tarfile

    import requests

    r = requests.get(url, headers={"Range": "bytes=0-8191"}, timeout=60)
    r.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r|") as t:
        for m in t:
            if m.isfile():
                return m.name.rsplit("/", 1)[-1]
            continue
    raise RuntimeError(f"no member in first 8 KB of {url}")


def invalidated_shard0_limit(locale: str, rows: list[dict]) -> int:
    """Number of leading invalidated.tsv rows that are in shard 0 (the tar shards follow TSV
    order; checked again while streaming). Only shard 0 is read for multi-shard locales."""
    ns = n_shards(locale)["invalidated"]
    if ns == 1:
        return len(rows)
    cache = meta_dir(locale) / "invalidated_shard1_first.txt"
    if not cache.exists():
        cache.write_text(first_member(shard_url(locale, "invalidated", 1)))
    name = cache.read_text().strip()
    for i, r in enumerate(rows):
        if r["path"] == name:
            # shard 0 holds at most SHARD_ROWS rows (en: rows 40000.. are in neither shard 0 nor
            # the start of shard 1; found by streaming shard 0 and checked again on every fetch)
            return min(i, SHARD_ROWS)
    raise RuntimeError(f"{locale}: first member of invalidated shard 1 ({name}) not in invalidated.tsv")


# ============================================================================ select

def _row(r: dict, locale: str, split: str, shard: int, dur: float | None) -> dict:
    stem = r["path"].rsplit(".", 1)[0]
    return {"stem": stem, "client_id": r["client_id"], "sentence_id": r["sentence_id"],
            "sentence": r["sentence"], "norm": norm_sentence(r["sentence"]),
            "n_chars": n_chars(r["sentence"]), "dur": dur, "up_votes": int(r["up_votes"] or 0),
            "down_votes": int(r["down_votes"] or 0), "accents": r.get("accents") or "",
            "variant": r.get("variant") or "", "src_split": split, "src_shard": shard}


def candidate_pools(locale: str) -> tuple[list[dict], list[dict], dict]:
    """(rejected pool, accepted pool, cleaning counts) after vote, duration and text rules."""
    durs = durations(locale)
    counts: dict = {}
    inv = read_tsv(meta_dir(locale) / "invalidated.tsv")
    lim = invalidated_shard0_limit(locale, inv)
    counts["invalidated_rows"] = len(inv)
    counts["invalidated_rows_in_shards_read"] = lim
    rej, acc = [], []
    c = Counter()
    for r in inv[:lim]:
        if not is_rejected(int(r["up_votes"] or 0), int(r["down_votes"] or 0)):
            c["votes_not_clear_reject"] += 1
            continue
        rej.append(_row(r, locale, "invalidated", 0, durs.get(r["path"])))
    acc_rows = 0
    for split in ("dev", "test"):
        for r in read_tsv(meta_dir(locale) / f"{split}.tsv"):
            acc_rows += 1
            if not is_accepted(int(r["up_votes"] or 0), int(r["down_votes"] or 0)):
                c["votes_not_clear_accept"] += 1
                continue
            acc.append(_row(r, locale, split, 0, durs.get(r["path"])))
    counts["accepted_rows"] = acc_rows
    counts["after_votes"] = {"rejected": len(rej), "accepted": len(acc)}
    counts["votes"] = dict(c)
    for name, pool in (("rejected", rej), ("accepted", acc)):
        cc = Counter()
        keep = []
        for r in pool:
            why = dur_rule(r["dur"]) or text_rule(r["sentence"], locale)
            if why:
                cc[why] += 1
            else:
                keep.append(r)
        pool[:] = keep
        counts[f"rules_{name}"] = dict(cc)
    return rej, acc, counts


def weak_pool(locale: str, rej_strict: list[dict]) -> tuple[list[dict], dict]:
    """Test A-weak positives: the strict rejected candidates (flagged strict=True) plus other.tsv
    clips with is_weak_rejected (strict=False), after the same duration / text rules."""
    durs = durations(locale)
    rows = read_tsv(meta_dir(locale) / "other.tsv")
    c = Counter()
    votes = Counter((int(r["up_votes"] or 0), int(r["down_votes"] or 0)) for r in rows)
    weak = []
    for r in rows:
        if not is_weak_rejected(int(r["up_votes"] or 0), int(r["down_votes"] or 0)):
            c["votes_not_weak_reject"] += 1
            continue
        x = _row(r, locale, "other", 0, durs.get(r["path"]))
        why = dur_rule(x["dur"]) or text_rule(x["sentence"], locale)
        if why:
            c[why] += 1
            continue
        weak.append(dict(x, strict=False))
    pos = [dict(r, strict=True) for r in rej_strict] + weak
    info = {"other_rows": len(rows), "other_vote_table": {f"up={u},down={d}": n for (u, d), n in sorted(votes.items())},
            "other_weak_candidates": sum(1 for r in rows if is_weak_rejected(int(r["up_votes"] or 0), int(r["down_votes"] or 0))),
            "other_rules": dict(c), "positives_after_cleaning": len(pos),
            "positives_strict": len(rej_strict), "positives_weak": len(weak),
            "positive_speakers": len({r["client_id"] for r in pos})}
    return pos, info


def _apply_region(locale: str, rej: list, acc: list, counts: dict) -> tuple[list, list, str]:
    """es: Latin American speakers preferred; pt: Brazilian preferred."""
    if locale not in ("es", "pt"):
        return rej, acc, "all speakers"
    reg = (lambda r: es_region(r["accents"])) if locale == "es" else (lambda r: pt_region(r["accents"], r["variant"]))
    good, bad = ("latam", "spain") if locale == "es" else ("brazil", "portugal")
    rp = [r for r in rej if reg(r) == good]
    ap = [r for r in acc if reg(r) == good]
    spk = lambda rows: len({r["client_id"] for r in rows})
    # need enough labelled speakers on both sides of BOTH halves (a speaker gives <= 2 clips)
    if spk(rp) >= 2 * TARGET and spk(ap) >= 2 * TARGET:
        counts["region"] = {"rule": f"only {good}-labelled speakers", "removed_rejected": len(rej) - len(rp),
                            "removed_accepted": len(acc) - len(ap)}
        return rp, ap, f"only speakers whose accent label is {good}"
    rn = [r for r in rej if reg(r) != bad]
    an = [r for r in acc if reg(r) != bad]
    counts["region"] = {"rule": f"{bad}-labelled speakers removed; {good}-labelled and unlabelled kept "
                                f"(only {spk(rp)} / {spk(ap)} {good}-labelled speakers rejected / accepted)",
                        "removed_rejected": len(rej) - len(rn), "removed_accepted": len(acc) - len(an)}
    return rn, an, f"{bad}-labelled speakers excluded, {good}-labelled + unlabelled kept"


def build_test(pos: list[dict], neg: list[dict], seed: int, target: int = TARGET, reserve: float = RESERVE):
    """Split by speaker, de-duplicate across sides, cap per speaker, stratified matching.
    -> {half: {"quota", "pos", "neg", "edges", "counts"}}"""
    out = {}
    for half in ("test", "dev"):
        p = [r for r in pos if speaker_half(r["client_id"], seed) == half]
        n = [r for r in neg if speaker_half(r["client_id"], seed) == half]
        p, n, cnt = make_disjoint(p, n, seed)
        rng = random.Random(f"{seed}|{half}|cap")
        p2, n2 = cap_per_speaker(p, PER_SPEAKER, rng), cap_per_speaker(n, PER_SPEAKER, rng)
        cnt["over_speaker_cap"] = (len(p) - len(p2)) + (len(n) - len(n2))
        q, lp, ln, edges = select_matched(p2, n2, target, seed + (0 if half == "test" else 1), reserve)
        out[half] = {"quota": q, "pos": lp, "neg": ln, "edges": edges, "counts": dict(cnt),
                     "pool": {"pos": len(p2), "neg": len(n2), "pos_spk": len({r['client_id'] for r in p2}),
                              "neg_spk": len({r['client_id'] for r in n2})}}
    return out


def step_select(locales) -> None:
    for loc in locales:
        t0 = time.time()
        rej, acc, counts = candidate_pools(loc)
        rej, acc, region_note = _apply_region(loc, rej, acc, counts)
        sel = {"locale": loc, "counts": counts, "region": region_note, "tests": {}}
        sel["tests"]["A"] = build_test(rej, acc, SEED)
        # Test B: accepted clips only; native vs self-declared non-native
        nat = Counter()
        pb, nb = [], []
        for r in acc:
            k = nativeness(loc, r["accents"])
            nat[k] += 1
            (pb if k == "nonnative" else nb if k == "native" else []).append(r)
        rng = random.Random(f"{SEED}|B|elig")
        pb_cap = cap_per_speaker(pb, PER_SPEAKER, rng)
        elig = {"nonnative_clips": len(pb), "nonnative_clips_capped": len(pb_cap),
                "nonnative_speakers": len({r["client_id"] for r in pb}),
                "native_clips": len(nb), "native_speakers": len({r["client_id"] for r in nb}),
                "labels": dict(nat)}
        sel["testB_eligibility"] = elig
        if len(pb_cap) >= TESTB_MIN_CLIPS and elig["nonnative_speakers"] >= TESTB_MIN_SPK:
            sel["tests"]["B"] = build_test(pb, nb, SEED + 7)
        if loc in WEAK_LOCALES and (meta_dir(loc) / "other.tsv").exists():
            pw, winfo = weak_pool(loc, rej)
            sel["testAweak_eligibility"] = winfo
            if winfo["positives_after_cleaning"] >= WEAK_MIN_POS:
                sel["tests"]["Aweak"] = build_test(pw, acc, SEED)
            else:
                print(f"{loc}: A-weak NOT built: only {winfo['positives_after_cleaning']} positives", flush=True)
        p = work_dir("select") / f"{loc}.json"
        p.write_text(json.dumps(_jsonable(sel), ensure_ascii=False))
        qa = sel["tests"]["A"]["test"]["quota"]
        msg = f"{loc}: A test quota {sum(qa.values())} / side (pool {sel['tests']['A']['test']['pool']})"
        for tn in ("B", "Aweak"):
            if tn in sel["tests"]:
                msg += f"; {tn} test quota {sum(sel['tests'][tn]['test']['quota'].values())}"
        print(msg, f"B elig {elig['nonnative_clips_capped']} clips / {elig['nonnative_speakers']} spk",
              f"[{time.time() - t0:.0f}s]", flush=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {(",".join(map(str, k)) if isinstance(k, tuple) else k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def load_selection(loc: str) -> dict:
    return json.loads((work_dir("select") / f"{loc}.json").read_text())


def wanted(sel: dict) -> dict[tuple[str, int], set[str]]:
    """{(split, shard): stems} over all tests / halves / sides (reserves included)."""
    out = defaultdict(set)
    for t in sel["tests"].values():
        for h in t.values():
            for side in ("pos", "neg"):
                for rows in h[side].values():
                    for r in rows:
                        out[(r["src_split"], r["src_shard"])].add(r["stem"])
    return out


# ============================================================================ fetch

def _stream_shard(loc: str, split: str, shard: int, stems: set[str]) -> dict:
    import io
    import tarfile

    import requests
    import soundfile as sf

    from bench.data.common import resample, write_flac

    todo = {s for s in stems if not audio_path(loc, s).exists()}
    stat_p = work_dir("fetch", loc) / f"{split}_{shard}.json"
    stats = json.loads(stat_p.read_text()) if stat_p.exists() else {}
    if not todo:
        return {"split": split, "shard": shard, "fetched": 0, "missing": [], "bytes": 0, "cached": True}
    url = shard_url(loc, split, shard)
    n_bytes = 0
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = True

        class Counting(io.RawIOBase):
            def readable(self):
                return True

            def readinto(self, b):
                nonlocal n_bytes
                data = r.raw.read(len(b))
                n_bytes += len(data)
                b[:len(data)] = data
                return len(data)

        with tarfile.open(fileobj=io.BufferedReader(Counting(), 1 << 20), mode="r|") as t:
            for m in t:
                if not m.isfile():
                    continue
                stem = m.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                if stem not in todo:
                    continue
                data = t.extractfile(m).read()
                try:
                    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
                    x = resample(x.mean(axis=1), sr)
                except Exception as e:  # noqa: BLE001
                    stats[stem] = {"error": f"{type(e).__name__}"}
                    todo.discard(stem)
                    continue
                stats[stem] = {"active_s": round(active_seconds(x), 3), "dur": round(len(x) / 16000, 3),
                               "peak_db": round(float(20 * np.log10(np.abs(x).max() + 1e-12)), 1)}
                write_flac(audio_path(loc, stem), x, 16000)
                todo.discard(stem)
                if not todo:
                    break
    stat_p.write_text(json.dumps(stats))
    return {"split": split, "shard": shard, "fetched": len(stems) - len(todo), "missing": sorted(todo),
            "bytes": n_bytes}


def audio_stats(loc: str) -> dict:
    out = {}
    for p in work_dir("fetch", loc).glob("*.json"):
        if p.name != "log.json":
            out.update(json.loads(p.read_text()))
    return out


def step_fetch(locales, workers: int = 6) -> None:
    from bench.data.common import free_gb

    jobs = []
    for loc in locales:
        for (split, shard), stems in sorted(wanted(load_selection(loc)).items()):
            jobs.append((loc, split, shard, stems))
    print(f"{len(jobs)} shards to stream; {free_gb():.1f} GB free", flush=True)

    def one(j):
        loc, split, shard, stems = j
        t0 = time.time()
        res = _stream_shard(loc, split, shard, stems)
        res["seconds"] = round(time.time() - t0, 1)
        logp = work_dir("fetch", loc) / "log.json"
        log = json.loads(logp.read_text()) if logp.exists() else {}
        if not res.get("cached"):
            log[f"{split}_{shard}"] = res
            logp.write_text(json.dumps(log))
        print(f"{loc} {split}_{shard}: {res['fetched']} clips, {len(res['missing'])} missing, "
              f"{res['bytes'] / 1e9:.2f} GB streamed, {res['seconds']} s", flush=True)
        return res

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))


# ============================================================================ finalize

def step_finalize(locales) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bp = OUT_DIR / "build.json"
    build = json.loads(bp.read_text()) if bp.exists() else {}
    build["source"] = {"dataset": CV_VERSION, "license": LICENSE, "mirror": f"https://huggingface.co/datasets/{REPO}",
                       "revision": REV, "note": "mirror card states license cc0-1.0; TSVs from transcript/<locale>/, "
                                                "audio from audio/<locale>/<split>/ tar shards"}
    build["rules"] = {"rejected": "invalidated, down_votes >= 2 and up_votes <= down_votes - 2",
                      "accepted": "dev/test (validated), up_votes >= 2 and down_votes == 0",
                      "duration": f"{MIN_DUR} s <= clip_durations <= {MAX_DUR} s",
                      "text": "non-Latin-script locales (zh-CN, zh-TW, ja, ru, ko): no digits / Latin letters",
                      "silence": f"near-silent if < {SIL_MIN_ACTIVE_S} s of 25 ms frames above {SIL_DBFS} dBFS",
                      "per_speaker": PER_SPEAKER, "target_per_side_per_half": TARGET, "seed": SEED,
                      "strata": f"{LEN_BINS} character-length x {DUR_BINS} duration quantile bins of the positives"}
    build.setdefault("locales", {})
    for loc in locales:
        sel = load_selection(loc)
        st = audio_stats(loc)
        flog_p = work_dir("fetch", loc) / "log.json"
        flog = json.loads(flog_p.read_text()) if flog_p.exists() else {}
        rows_out = []
        info = {"counts": sel["counts"], "region": sel["region"], "testB_eligibility": sel["testB_eligibility"],
                **({"testAweak_eligibility": sel["testAweak_eligibility"]} if "testAweak_eligibility" in sel else {}),
                "shards_read": sorted(f"{loc}/{k}" for k in flog), "streamed_gb": round(sum(v["bytes"] for v in flog.values()) / 1e9, 2),
                "tests": {}}
        for tname, t in sel["tests"].items():
            info["tests"][tname] = {}
            for half, h in t.items():
                sil = Counter()

                def ok(r):
                    s = st.get(r["stem"])
                    if s is None or "error" in s or not audio_path(loc, r["stem"]).exists():
                        sil["audio_missing_or_undecodable"] += 1
                        return False
                    if s["active_s"] < SIL_MIN_ACTIVE_S:
                        sil["near_silent"] += 1
                        return False
                    return True

                n_pos = n_neg = 0
                for skey in sorted(h["quota"]):
                    qv = h["quota"][skey]
                    if qv == 0:
                        continue
                    p, n = finalize_stratum(qv, h["pos"].get(skey, []), h["neg"].get(skey, []), ok)
                    for side, rows in (("pos", p), ("neg", n)):
                        for r in rows:
                            rows_out.append(_manifest_row(loc, tname, half, side, skey, r, st[r["stem"]]))
                    n_pos += len(p)
                    n_neg += len(n)
                info["tests"][tname][half] = {"n_pos": n_pos, "n_neg": n_neg, "quota": sum(h["quota"].values()),
                                              "pool": h["pool"], "dedup_cap": h["counts"],
                                              "audio_rules_on_fetched": dict(sil), "edges": h["edges"]}
        rows_out.sort(key=lambda r: (r["test"], r["half"], -r["label"], r["id"]))
        mp = OUT_DIR / f"{loc}.jsonl"
        with open(mp, "w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        build["locales"][loc] = info
        print(loc, {t: {hh: (v["n_pos"], v["n_neg"]) for hh, v in d.items()} for t, d in info["tests"].items()},
              flush=True)
    bp.write_text(json.dumps(build, ensure_ascii=False, indent=1))


def _manifest_row(loc, test, half, side, skey, r, st) -> dict:
    pos_name = {"A": ("rejected", "accepted"), "Aweak": ("rejected", "accepted"),
                "B": ("nonnative", "native")}[test]
    return {"id": f"cv22-{loc}-{r['stem'].rsplit('_', 1)[-1]}", "locale": loc, "lang": SCORER_LANG[loc],
            "test": test, "half": half, "label": 1 if side == "pos" else 0,
            "side": pos_name[0] if side == "pos" else pos_name[1], "stratum": skey,
            "path": r["path"] if "path" in r else r["stem"] + ".mp3", "client_id": r["client_id"],
            "sentence_id": r["sentence_id"], "sentence": r["sentence"], "n_chars": r["n_chars"],
            "dur": round(st["dur"], 3), "up_votes": r["up_votes"], "down_votes": r["down_votes"],
            "accents": r["accents"], "variant": r["variant"],
            **({"strict": r["strict"]} if "strict" in r else {}),
            "source": f"{REPO}@{REV[:12]}:audio/{loc}/{r['src_split']}/{loc}_{r['src_split']}_{r['src_shard']}.tar"}


# ============================================================================ loading (for eval)

def load_manifest(locales=LOCALES, half: str | None = "test", test: str | None = None) -> list[dict]:
    out = []
    for loc in locales:
        p = OUT_DIR / f"{loc}.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if (half is None or r["half"] == half) and (test is None or r["test"] == test):
                r["audio"] = str(audio_path(loc, r["path"].rsplit(".", 1)[0]))
                out.append(r)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("meta", "select", "fetch", "finalize", "all"))
    ap.add_argument("--locales", default="all")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    locs = list(LOCALES) if a.locales == "all" else a.locales.split(",")
    for l in locs:
        if l not in LOCALES:
            sys.exit(f"unknown locale {l}")
    if a.step in ("meta", "all"):
        step_meta(locs)
    if a.step in ("select", "all"):
        step_select(locs)
    if a.step in ("fetch", "all"):
        step_fetch(locs, a.workers)
    if a.step in ("finalize", "all"):
        step_finalize(locs)


if __name__ == "__main__":
    main()
