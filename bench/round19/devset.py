"""Round 19 (German): a dev set of planted reference-text edits on native German audio that is
NOT in the Lingolix benchmark.

    python -m bench.round19.devset speakers   # Lingolix clip -> Common Voice speaker (validated.tsv)
    python -m bench.round19.devset select     # choose CV clips (metadata only)
    python -m bench.round19.devset fetch      # stream the CV dev tar shard, keep only chosen clips
    python -m bench.round19.devset vocab      # candidate replacement words (isolated-word G2P)
    python -m bench.round19.devset build      # write the manifest

Sources
  * Common Voice 22.0 German (CC0), clips of CV's own dev split whose speaker is in the cv-real
    DEV half (bench.cv_build.speaker_half), accepted (up >= 2, down == 0), accent not marked
    non-native, 1-12 s, at most 2 clips per speaker. Excluded: every clip and every speaker of
    the Lingolix German items (speakers looked up in validated.tsv) and every Lingolix prompt.
  * FLEURS German validation (CC BY 4.0), clips <= 20 s.
Each clip gives one control item (its own text) and one changed item: one word of the text is
replaced the way the Lingolix benchmark does it (by a real word): sub_subtle / sub_moderate /
sub_gross (one or two sound substitutions, graded by articulatory distance; a far word for gross),
indel (one or two sounds added or dropped), word_omission (a real word inserted into the text that
the speaker did not say). Every edit is checked with the front-end (all other words keep their
phones). Items carry a fit / check fold by speaker (FLEURS: by clip; no speaker ids).

Files under $PRONUNCIADA_CACHE/round19/german/: dev_manifest.jsonl (Lingolix manifest layout:
id, target, label{bucket, category, word_index, ...}, plus audio, source, speaker, fold).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CACHE = Path(os.environ.get("PRONUNCIADA_CACHE") or Path.home() / ".cache" / "pronunciada")
WORK = CACHE / "round19" / "german"
CV_META = CACHE / "data" / "cv-real" / "meta" / "de"
CV_AUDIO_OLD = CACHE / "data" / "cv-real" / "audio" / "de"
CV_AUDIO = WORK / "cv_audio"
FLEURS_AUDIO = WORK / "fleurs_val_audio"
LINGOLIX_MANIFEST = CACHE / "third_party" / "lingolix" / "repo_v2026.10" / "src" / "data" / "tier1" / "manifest_de.jsonl"
SEED = 1919
N_CV = 460
N_FLEURS = 260
csv.field_size_limit(1 << 30)
_STRIP = re.compile(r"^\W+|\W+$")


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def read_tsv(p: Path):
    with open(p, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def lingolix_items() -> list[dict]:
    return [json.loads(l) for l in LINGOLIX_MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]


# ----------------------------------------------------------------------------- speakers

def cmd_speakers() -> None:
    clips = {x["cv_clip"] for x in lingolix_items()}
    spk = {}
    for name in ("validated.tsv",):
        for r in read_tsv(WORK / "cvmeta" / name):
            if r["path"] in clips:
                spk[r["path"]] = r["client_id"]
    for name in ("dev.tsv", "test.tsv", "invalidated.tsv"):
        for r in read_tsv(CV_META / name):
            if r["path"] in clips:
                spk.setdefault(r["path"], r["client_id"])
    out = {"clips": sorted(clips), "found": len(spk), "speakers": sorted(set(spk.values()))}
    (WORK / "lingolix_speakers.json").write_text(json.dumps(out))
    print(f"lingolix clips {len(clips)}, with speaker {len(spk)}, speakers {len(out['speakers'])}")


# ----------------------------------------------------------------------------- select

def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def cmd_select() -> None:
    from bench.cv_build import nativeness, speaker_half

    L = json.loads((WORK / "lingolix_speakers.json").read_text())
    bad_spk, bad_clip = set(L["speakers"]), set(L["clips"])
    bad_sent = {_norm(x["prompt"]) for x in lingolix_items()} | {_norm(x["target"]) for x in lingolix_items()}
    durs = {}
    with open(CV_META / "clip_durations.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1].strip().isdigit():
                durs[p[0]] = int(p[1]) / 1000
    have = {p.stem for p in CV_AUDIO_OLD.glob("*.flac")}
    cands = []
    for r in read_tsv(CV_META / "dev.tsv"):
        cid = r["client_id"]
        if cid in bad_spk or r["path"] in bad_clip or _norm(r["sentence"]) in bad_sent:
            continue
        if speaker_half(cid) != "dev":
            continue
        if int(r["up_votes"] or 0) < 2 or int(r["down_votes"] or 0) != 0:
            continue
        if nativeness("de", r.get("accents") or "") == "nonnative":
            continue
        d = durs.get(r["path"])
        if d is None or not 1.0 <= d <= 12.0:
            continue
        if len(r["sentence"].split()) < 3:
            continue
        stem = r["path"].rsplit(".", 1)[0]
        cands.append({"stem": stem, "speaker": cid, "sentence": r["sentence"], "dur": d,
                      "accents": r.get("accents") or "", "local": stem in have})
    # at most 2 clips per speaker; prefer clips already on disk, then a seeded order
    rng = random.Random(SEED)
    rng.shuffle(cands)
    cands.sort(key=lambda c: not c["local"])
    per, out, seen_sent = Counter(), [], set()
    for c in cands:
        if per[c["speaker"]] >= 2 or _norm(c["sentence"]) in seen_sent:
            continue
        per[c["speaker"]] += 1
        seen_sent.add(_norm(c["sentence"]))
        out.append(c)
        if len(out) >= int(N_CV * 1.15):           # reserve for clips that fail later rules
            break
    (WORK / "cv_select.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"candidates {len(cands)}, selected {len(out)} ({sum(c['local'] for c in out)} on disk), "
          f"speakers {len(per)}")


def cmd_fetch() -> None:
    import requests
    import soundfile as sf

    from bench.cv_build import shard_url
    from bench.data.common import resample, write_flac

    sel = json.loads((WORK / "cv_select.json").read_text())
    todo = {c["stem"] for c in sel if not c["local"] and not (CV_AUDIO / f"{c['stem']}.flac").exists()}
    print("to fetch", len(todo), flush=True)
    if not todo:
        return
    with requests.get(shard_url("de", "dev", 0), stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        import tarfile

        with tarfile.open(fileobj=io.BufferedReader(r.raw, 1 << 20), mode="r|") as t:
            for m in t:
                if not m.isfile():
                    continue
                stem = m.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                if stem not in todo:
                    continue
                data = t.extractfile(m).read()
                x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
                write_flac(CV_AUDIO / f"{stem}.flac", resample(x.mean(axis=1), sr), 16000)
                todo.discard(stem)
                if not todo:
                    break
    print("missing after fetch", len(todo))


def cv_audio(stem: str) -> Path | None:
    for p in (CV_AUDIO_OLD / f"{stem}.flac", CV_AUDIO / f"{stem}.flac"):
        if p.exists():
            return p
    return None


def fleurs_sources(n: int) -> list[dict]:
    """FLEURS validation clips <= 20 s, materialised as FLAC (read from the parquet once)."""
    from bench.data.common import write_flac
    from bench.pipeline.common import iter_parquet_audio, records

    recs = sorted([r for r in records("fleurs-validation", lang="de") if r["dur"] <= 20.0],
                  key=lambda r: _h("fleurs", r["id"]))[: int(n * 1.15)]
    want = {}
    for r in recs:
        _, path, row = r["audio"].rsplit(":", 2) if r["audio"].count(":") >= 2 else (None, None, None)
        want[int(row)] = r
        r["_flac"] = FLEURS_AUDIO / f"{r['id']}.flac"
    path = recs[0]["audio"].split(":", 1)[1].rsplit(":", 1)[0]
    if not all(r["_flac"].exists() for r in recs):
        for i, wav in iter_parquet_audio(path, set(want)):
            if not want[i]["_flac"].exists():
                write_flac(want[i]["_flac"], wav, 16000)
    return recs


# ----------------------------------------------------------------------------- vocabulary

def _word_counts() -> Counter:
    """Word frequencies from Common Voice German sentences (validated.tsv) and FLEURS transcripts,
    excluding every Lingolix prompt."""
    from bench.data import load

    bad = {_norm(x["prompt"]) for x in lingolix_items()}
    cnt = Counter()
    seen = set()
    for r in read_tsv(WORK / "cvmeta" / "validated.tsv"):
        s = r["sentence"]
        if s in seen or _norm(s) in bad:
            continue
        seen.add(s)
        for tok in s.split():
            t = _STRIP.sub("", tok)
            if t.isalpha() and len(t) >= 2:
                cnt[t] += 1
    for r in load("fleurs", lang="de"):
        for tok in r["text"].split():
            t = _STRIP.sub("", tok)
            if t.isalpha() and len(t) >= 2:
                cnt[t] += 1
    return cnt


def cmd_vocab(top: int = 40000) -> None:
    from bench import textswap as TS
    from pronunciada.text import get_g2p

    g2p = get_g2p("de")
    cnt = _word_counts()
    forms: dict[str, tuple[str, int]] = {}
    for w, c in cnt.most_common():
        if len(w) > 1 and w.isupper():
            continue
        k = w.lower()
        if k not in forms:
            forms[k] = (w, c)
        if len(forms) >= top:
            break
    out = []
    for k, (w, c) in forms.items():
        ref = g2p(w)
        if ref.warnings or len(ref.words) != 1:
            continue
        us = TS.units(ref, "de")
        if len(us) != 1 or us[0].text != w:
            continue
        out.append([w, list(us[0].phones), c])
    (WORK / "vocab.json").write_text(json.dumps(out, ensure_ascii=False))
    print("vocab entries", len(out))


class Vocab:
    def __init__(self):
        rows = json.loads((WORK / "vocab.json").read_text())
        self.by_ph: dict[tuple, list] = defaultdict(list)
        self.rows = []
        for w, ph, c in rows:
            ph = tuple(ph)
            self.by_ph[ph].append((w, c))
            self.rows.append((w, ph, c))
        self.inv = sorted({p for _, ph, _ in self.rows for p in ph})
        self.frequent = [(w, ph, c) for w, ph, c in self.rows if c >= 30]

    def neighbours(self, ph: tuple) -> dict[str, list]:
        """Real words at one edit: {"sub": [(word, ph, count, (i, a, b))], "ins": [...], "del": [...]}."""
        out = {"sub": [], "ins": [], "del": []}
        for i in range(len(ph)):
            for p in self.inv:
                if p != ph[i]:
                    q = ph[:i] + (p,) + ph[i + 1:]
                    out["sub"] += [(w, q, c, (i, ph[i], p)) for w, c in self.by_ph.get(q, [])]
            q = ph[:i] + ph[i + 1:]
            out["del"] += [(w, q, c, (i, ph[i], "")) for w, c in self.by_ph.get(q, [])]
        for i in range(len(ph) + 1):
            for p in self.inv:
                q = ph[:i] + (p,) + ph[i:]
                out["ins"] += [(w, q, c, (i, "", p)) for w, c in self.by_ph.get(q, [])]
        return out

    def two_edit(self, ph: tuple, kind: str) -> list:
        """Real words at exactly two edits; kind 'sub2' (same length) or 'indel2' (length +-1/2)."""
        res = []
        one = self.neighbours(ph)
        firsts = one["sub"] + one["ins"] + one["del"] if kind == "indel2" else one["sub"]
        seen = set()
        # second step from the phone strings at one edit, whether or not they are words
        for i in range(len(ph)):
            for p in self.inv:
                if p == ph[i]:
                    continue
                q1 = ph[:i] + (p,) + ph[i + 1:]
                if kind == "sub2":
                    for j in range(i + 1, len(ph)):
                        for p2 in self.inv:
                            if p2 != ph[j]:
                                q = q1[:j] + (p2,) + q1[j + 1:]
                                for w, c in self.by_ph.get(q, []):
                                    if (w, q) not in seen:
                                        seen.add((w, q))
                                        res.append((w, q, c, ("sub2",)))
        if kind == "indel2":
            from bench.textswap import edit_distance

            L = len(ph)
            for w, q, c in self.rows:
                if 1 <= abs(len(q) - L) <= 2 and q and q[0] in ph[:2] + (ph[-1],) and edit_distance(ph, q) == 2:
                    res.append((w, q, c, ("indel2",)))
        del firsts
        return res


def sub_grade(a: str, b: str) -> str:
    from pronunciada import articulatory as A

    try:
        d = len(A.diff(a, b))
    except KeyError:
        return "sub_moderate"
    if d == 0 or "type" in A.diff(a, b):
        return "sub_gross" if d else "sub_subtle"
    return "sub_subtle" if d == 1 else "sub_moderate"


# ----------------------------------------------------------------------------- build

KINDS = ("sub_subtle", "sub_moderate", "sub_gross", "indel", "indel", "word_omission")


def _units(text, g2p):
    from bench import textswap as TS

    ref = g2p(text)
    return ref, TS.units(ref, "de")


def _ok_target(u) -> bool:
    t = u.text
    return (not u.oov and len(u.phones) >= 2 and t.isalpha() and not (len(t) > 1 and t.isupper()))


def _replace(uid, text, ref, ou, k, cands, kind, g2p, max_tries=25):
    from bench import textswap as TS

    u = ou[k]
    rng = random.Random(_h(uid, kind, k))
    cands = sorted(cands, key=lambda e: (-min(e[2], 50), e[0]))
    head = cands[:12]
    rng.shuffle(head)
    for w, q, c, op in (head + cands[12:])[:max_tries]:
        if w.lower() == u.text.lower():
            continue
        nw = w[:1].upper() + w[1:] if u.text[:1].isupper() and k == 0 else w
        nt = text[:u.span[0]] + nw + text[u.span[1]:]
        r = TS.verify("de", g2p, ou, k, nt, nw, ref.warnings, lambda nu, q=q: nu.phones == q)
        if r:
            return nt, nw, q, op
    return None


def make_changed(uid, text, g2p, vocab: Vocab, kind_order: list[str]):
    from bench import textswap as TS

    ref, ou = _units(text, g2p)
    pos = [k for k in range(len(ou)) if _ok_target(ou[k])]
    rng = random.Random(_h(uid, "pos"))
    rng.shuffle(pos)
    for kind in kind_order:
        if kind == "word_omission":
            ks = list(range(1, len(ou)))
            rng.shuffle(ks)
            ins = [e for e in vocab.frequent if len(e[1]) >= 2 and e[0].isalpha()]
            r2 = random.Random(_h(uid, "ins"))
            for k in ks[:6]:
                for _ in range(15):
                    w, q, c = ins[r2.randrange(len(ins))]
                    a = ou[k].span[0]
                    nt = text[:a] + w + " " + text[a:]
                    ref2, nu = _units(nt, g2p)
                    if len(nu) != len(ou) + 1 or set(ref2.warnings) - set(ref.warnings):
                        continue
                    if nu[k].text != w or nu[k].phones != q:
                        continue
                    if all(nu[i + (i >= k)].phones == ou[i].phones for i in range(len(ou))):
                        return nt, {"category": "word_omission", "bucket": "word_omission", "added": w,
                                    "word_index": k, "position": _position(k, len(nu)),
                                    "word_level_scorable": True, "n_ops": len(q)}
            continue
        for k in pos:
            u = ou[k]
            if kind in ("sub_subtle", "sub_moderate"):
                one = vocab.neighbours(u.phones)["sub"]
                cands = [e for e in one if sub_grade(e[3][1], e[3][2]) == kind]
                if kind == "sub_moderate":
                    cands += [e for e in vocab.two_edit(u.phones, "sub2")][:200] if len(u.phones) >= 4 else []
            elif kind == "sub_gross":
                one = vocab.neighbours(u.phones)["sub"]
                cands = [e for e in one if sub_grade(e[3][1], e[3][2]) == "sub_gross"]
                if not cands:
                    cands = [(w, q, c, ("far",)) for w, q, c in vocab.frequent
                             if TS.is_far(u.phones, q) and len(q) >= 2][:400]
            else:   # indel
                nb = vocab.neighbours(u.phones)
                cands = nb["ins"] + nb["del"]
                if len(u.phones) >= 4:
                    cands += vocab.two_edit(u.phones, "indel2")[:200]
            if not cands:
                continue
            r = _replace(uid, text, ref, ou, k, cands, kind, g2p)
            if r:
                nt, nw, q, op = r
                n_ops = TS.edit_distance(u.phones, q)
                cat = "sub" if kind.startswith("sub") else ("insert" if len(q) > len(u.phones) else
                                                           "delete" if len(q) < len(u.phones) else "mixed")
                if kind == "indel" and n_ops == 2 and abs(len(q) - len(u.phones)) == 1:
                    cat = "mixed"
                return nt, {"category": cat, "bucket": kind, "word_index": k, "original": u.text,
                            "replacement": nw, "n_ops": n_ops, "position": _position(k, len(ou)),
                            "orig_phones": " ".join(u.phones), "new_phones": " ".join(q),
                            "word_level_scorable": True}
    return None


def _position(k, n):
    return "initial" if k == 0 else "final" if k == n - 1 else "medial"


def cmd_build() -> None:
    from pronunciada.text import get_g2p

    g2p = get_g2p("de")
    vocab = Vocab()
    srcs = []
    for c in json.loads((WORK / "cv_select.json").read_text()):
        p = cv_audio(c["stem"])
        if p is None:
            continue
        srcs.append({"id": f"cv-{c['stem'].rsplit('_', 1)[-1]}", "source": "cv", "audio": str(p),
                     "text": c["sentence"], "speaker": "cv:" + hashlib.sha1(c["speaker"].encode()).hexdigest()[:12],
                     "dur": c["dur"]})
    srcs = srcs[:N_CV]
    for r in fleurs_sources(N_FLEURS)[:N_FLEURS]:
        srcs.append({"id": f"fl-{r['id'].rsplit('-', 1)[-1]}", "source": "fleurs", "audio": str(r["_flac"]),
                     "text": r["text"], "speaker": "fl:" + r["id"], "dur": r["dur"]})
    out, cov = [], Counter()
    for i, s in enumerate(srcs):
        fold = "fit" if _h("fold", s["speaker"]) % 2 == 0 else "check"
        base = {"audio": s["audio"], "source": s["source"], "speaker": s["speaker"], "fold": fold,
                "prompt": s["text"], "dur": s["dur"], "lang": "de"}
        out.append({"id": f"{s['id']}__ctl", **base, "target": s["text"], "label": None})
        start = _h(s["id"], "kind") % len(KINDS)
        order = [KINDS[(start + j) % len(KINDS)] for j in range(len(KINDS))]
        r = make_changed(s["id"], s["text"], g2p, vocab, order)
        if r:
            nt, lab = r
            cov[(s["source"], lab["bucket"])] += 1
            out.append({"id": f"{s['id']}__chg", **base, "target": nt, "label": lab})
        else:
            cov[(s["source"], "none")] += 1
        if i % 50 == 0:
            print(i, len(srcs), flush=True)
    (WORK / "dev_manifest.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print({f"{a}/{b}": n for (a, b), n in sorted(cov.items())})
    print("items", len(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["speakers", "select", "fetch", "vocab", "build"])
    a = ap.parse_args(argv)
    WORK.mkdir(parents=True, exist_ok=True)
    {"speakers": cmd_speakers, "select": cmd_select, "fetch": cmd_fetch, "vocab": cmd_vocab,
     "build": cmd_build}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
