"""L2-ARCTIC v5.0 (CC BY-NC 4.0, research use only): manually annotated subset only
(~150 utterances per speaker, 24 non-native English speakers, 6 L1s).

Source: HF mirror ``chikingsley/l2-arctic-release-v5.0`` (per-speaker zip archives). Only the
annotated members (annotation/*.TextGrid + the matching wav + transcript) are read from each
remote zip via HTTP range requests; the archives are never downloaded whole.

    python -m bench.data.l2arctic              # fetch
    python -m bench.data.l2arctic --relabel    # re-read the TextGrids, rewrite labels only

split: "test" for the standard test speakers NJS, TLV, TNI, TXHC, YKWK, ZHAA; "train" otherwise.

labels:
    l1: speaker's native language; gender: "m"/"f"
    words:  [{text, start, end}]   (silences dropped)
    phones: [{canonical, perceived, error, start, end, word}]
        canonical/perceived: ARPAbet (stress digit kept on canonical vowels when present);
        error: null (correct), "s" substitution, "d" deletion (perceived "sil"),
               "a" addition (canonical "sil"); word = index into words or null.
        Annotator comments that do not parse are kept in `raw`.
    phones_ipa: same structure from the IPA tier when present (else omitted)
"""
from __future__ import annotations

import re

from .common import (PartWriter, check_disk, data_dir, decode_audio, make_record, merge_parts,
                     summarize, write_flac)

NAME = "l2arctic"
URL = "https://huggingface.co/datasets/chikingsley/l2-arctic-release-v5.0/resolve/main/{spk}.zip"
SPEAKERS = {  # speaker: (L1, gender) from the L2-ARCTIC README
    "ABA": ("Arabic", "m"), "SKA": ("Arabic", "f"), "YBAA": ("Arabic", "m"), "ZHAA": ("Arabic", "f"),
    "BWC": ("Mandarin", "m"), "LXC": ("Mandarin", "f"), "NCC": ("Mandarin", "f"), "TXHC": ("Mandarin", "m"),
    "ASI": ("Hindi", "m"), "RRBI": ("Hindi", "m"), "SVBI": ("Hindi", "f"), "TNI": ("Hindi", "f"),
    "HJK": ("Korean", "f"), "HKK": ("Korean", "m"), "YDCK": ("Korean", "f"), "YKWK": ("Korean", "m"),
    "EBVS": ("Spanish", "m"), "ERMS": ("Spanish", "m"), "MBMPS": ("Spanish", "f"), "NJS": ("Spanish", "f"),
    "HQTV": ("Vietnamese", "m"), "PNV": ("Vietnamese", "f"), "THV": ("Vietnamese", "f"), "TLV": ("Vietnamese", "m"),
}
TEST_SPEAKERS = {"NJS", "TLV", "TNI", "TXHC", "YKWK", "ZHAA"}
SILENCE = {"", "sil", "sp", "spn"}


def parse_textgrid(txt: str) -> dict[str, list[tuple[float, float, str]]]:
    """Minimal Praat long-format TextGrid reader -> {tier name: [(xmin, xmax, text)]}."""
    tiers: dict[str, list] = {}
    cur = None
    xmin = xmax = None
    for line in txt.splitlines():
        s = line.strip()
        m = re.match(r'name = "(.*)"', s)
        if m:
            cur = m.group(1)
            tiers[cur] = []
            continue
        if cur is None:
            continue
        if s.startswith("intervals ["):
            xmin = xmax = None
        elif s.startswith("xmin ="):
            xmin = float(s.split("=", 1)[1])
        elif s.startswith("xmax ="):
            xmax = float(s.split("=", 1)[1])
        elif s.startswith("text =") and xmin is not None:
            tiers[cur].append((xmin, xmax, s.split("=", 1)[1].strip().strip('"')))
    return tiers


def _phone_entry(text: str, start: float, end: float, words: list[dict]) -> dict | None:
    parts = [p.strip() for p in text.split(",")]
    raw = None
    if len(parts) == 1:
        if parts[0] in SILENCE:
            return None
        can, per, err = parts[0], parts[0], None
    elif len(parts) >= 3 and parts[2] in ("s", "d", "a"):
        can, per, err = parts[0], parts[1], parts[2]
        if len(parts) > 3:
            raw = text
    else:
        can, per, err, raw = parts[0], (parts[1] if len(parts) > 1 else parts[0]), "?", text
    mid = (start + end) / 2
    wi = next((i for i, w in enumerate(words) if w["start"] <= mid < w["end"]), None)
    e = {"canonical": can, "perceived": per, "error": err, "start": round(start, 4),
         "end": round(end, 4), "word": wi}
    if raw:
        e["raw"] = raw
    return e


def labels_from_textgrid(txt: str, spk: str) -> dict:
    tiers = parse_textgrid(txt)
    names = {n.lower(): n for n in tiers}
    words = [{"text": t, "start": round(a, 4), "end": round(b, 4)}
             for a, b, t in tiers.get(names.get("words", ""), []) if t.strip().lower() not in SILENCE]
    out = {"l1": SPEAKERS[spk][0], "gender": SPEAKERS[spk][1], "words": words, "phones": []}
    phone_tiers = [n for n in tiers if n.lower() != "words"]
    # Tier selection. The tier NAMED "phones" (case-insensitive) is the full ARPAbet phone tier and
    # the one named "IPA" is the IPA tier. The IPA tier often holds only the few annotated errors
    # in ASCII-looking IPA ("d,t,s"), which a content-based guess takes for ARPAbet; the old code
    # let the LAST such tier overwrite `phones` and lost the full tier for 429 / 3599 files.
    # Files without these names fall back to the content check, and among several ARPAbet-looking
    # tiers the one with the most entries wins.
    cand: dict[str, list] = {"phones": [], "phones_ipa": []}
    for tn in phone_tiers:
        ents = [e for a, b, t in tiers[tn] if (e := _phone_entry(t, a, b, words))]
        if tn.lower() == "phones":
            kind, prio = "phones", 2
        elif tn.lower() == "ipa":
            kind, prio = "phones_ipa", 2
        else:
            is_ipa = any(not re.fullmatch(r"[A-Za-z]+[0-2]?|sil|err|\?", e["canonical"]) for e in ents)
            kind, prio = ("phones_ipa" if is_ipa else "phones"), 1
        cand[kind].append((prio, len(ents), ents))
    for kind, lst in cand.items():
        if lst:
            out[kind] = max(lst, key=lambda x: (x[0], x[1]))[2]
    return out


def _cdn_url(url: str) -> str:
    import time

    import requests

    for k in range(8):
        r = requests.head(url, allow_redirects=False, timeout=30)
        if r.status_code in (301, 302, 303, 307, 308):
            loc = r.headers["Location"]
            return loc if loc.startswith("http") else requests.compat.urljoin(url, loc)
        if r.status_code == 200:
            return url
        time.sleep(min(30 * (k + 1), 300))
    raise RuntimeError(f"could not resolve {url}")


def fetch_speaker(spk: str) -> int:
    import zipfile

    import fsspec

    root = data_dir(NAME)
    pw = PartWriter(root, spk)
    if pw.done:
        return len(pw.records)
    check_disk()
    split = "test" if spk in TEST_SPEAKERS else "train"
    # Resolve the HF redirect once (HF rate-limits "resolve" calls per IP); range reads then go
    # to the signed CDN URL directly.
    url = _cdn_url(URL.format(spk=spk))
    f = fsspec.filesystem("http").open(url, block_size=1 << 20, cache_type="readahead")
    z = zipfile.ZipFile(f)
    names = set(z.namelist())
    anns = sorted(n for n in names if "/annotation/" in n and n.endswith(".TextGrid"))
    for a in anns:
        utt = a.rsplit("/", 1)[1][:-len(".TextGrid")]
        uid = f"l2arctic-{spk}-{utt}"
        if uid in pw.ids:
            continue
        wav, tr = f"{spk}/wav/{utt}.wav", f"{spk}/transcript/{utt}.txt"
        if wav not in names:
            continue
        text = z.read(tr).decode("utf-8").strip() if tr in names else ""
        labels = labels_from_textgrid(z.read(a).decode("utf-8", "replace"), spk)
        if not text:
            text = " ".join(w["text"] for w in labels["words"])
        x, sr = decode_audio(z.read(wav))
        out = root / "audio" / spk / f"{utt}.flac"
        dur = write_flac(out, x, sr)
        pw.add(make_record(uid, out, "en", text, split, spk, dur, labels))
    pw.finish()
    print(f"{spk}: {len(pw.records)} annotated utterances", flush=True)
    return len(pw.records)


def relabel_speaker(spk: str) -> tuple[int, int]:
    """Re-read the annotation TextGrids of an already fetched speaker (range requests, audio is not
    re-downloaded) and rewrite `labels` of its records with the current tier selection.
    Returns (records, records whose labels changed)."""
    import json
    import os
    import zipfile

    import fsspec

    from .common import read_jsonl

    root = data_dir(NAME)
    path = root / "parts" / f"{spk}.jsonl"
    recs = list(read_jsonl(path))
    url = _cdn_url(URL.format(spk=spk))
    f = fsspec.filesystem("http").open(url, block_size=1 << 20, cache_type="readahead")
    z = zipfile.ZipFile(f)
    names = set(z.namelist())
    changed = 0
    for r in recs:
        utt = r["id"][len(f"l2arctic-{spk}-"):]
        a = f"{spk}/annotation/{utt}.TextGrid"
        if a not in names:
            continue
        lab = labels_from_textgrid(z.read(a).decode("utf-8", "replace"), spk)
        if lab != r["labels"]:
            changed += 1
            r["labels"] = lab
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    os.replace(tmp, path)
    print(f"{spk}: {len(recs)} records, {changed} relabelled", flush=True)
    return len(recs), changed


def main() -> None:
    import sys
    from concurrent.futures import ThreadPoolExecutor

    if "--relabel" in sys.argv[1:]:
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(relabel_speaker, SPEAKERS))
        merge_parts(data_dir(NAME), list(SPEAKERS))
        print(f"relabelled {sum(c for _, c in res)} of {sum(n for n, _ in res)} records")
        return
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(fetch_speaker, SPEAKERS))
    merge_parts(data_dir(NAME), list(SPEAKERS))
    from . import load

    print(summarize(load(NAME)))


if __name__ == "__main__":
    main()
