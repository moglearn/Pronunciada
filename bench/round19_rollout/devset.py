"""Round 19 rollout: a dev set of planted real-word changes on native audio, any space-separated
language (the German builder bench/round19/devset.py, generalised).

    python -m bench.round19_rollout.devset vocab --lang fr
    python -m bench.round19_rollout.devset build --lang fr

Clips: the "dev" Common Voice speakers of bench.round19_rollout.sources (CV DEV split, cv-real dev
half, no Lingolix clip / speaker / prompt) and FLEURS VALIDATION clips <= 20 s. Each clip gives a
control item (its own text) and one changed item, a real word put in the place of one word the way the
Lingolix benchmark does it: sub_subtle / sub_moderate / sub_gross (graded by articulatory distance; a
far word for gross), indel (one or two sounds added or dropped), word_omission (a real word inserted
that the speaker did not say). Every edit is checked with the front-end (all other words keep their
sounds). Folds fit / check by speaker (FLEURS: clip = speaker).

Replacement vocabulary: words of Common Voice dev.tsv / invalidated.tsv sentences and FLEURS train
and validation transcripts (never test), no Lingolix prompt.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict

from bench.round19_rollout.sources import _h, cv_meta, cv_rows, lingolix_items, norm, read_tsv, work

_STRIP = re.compile(r"^\W+|\W+$")
KINDS = ("sub_subtle", "sub_moderate", "sub_gross", "indel", "indel", "word_omission")


def _word_counts(lang: str) -> Counter:
    from bench.data import load
    from bench.pipeline.common import records

    bad = {norm(x["prompt"]) for x in lingolix_items(lang)}
    cnt, seen = Counter(), set()
    texts = []
    for name in ("dev.tsv", "invalidated.tsv"):
        p = cv_meta(lang) / name
        if p.exists():
            texts += [r["sentence"] for r in read_tsv(p)]
    texts += [r["text"] for r in load("fleurs", "train", lang)]
    texts += [r["text"] for r in records("fleurs-validation", lang=lang)]
    for s in texts:
        if s in seen or norm(s) in bad:
            continue
        seen.add(s)
        for tok in s.split():
            t = _STRIP.sub("", tok)
            if t.isalpha() and len(t) >= 2:
                cnt[t] += 1
    return cnt


def cmd_vocab(lang: str, top: int = 40000) -> None:
    from bench import textswap as TS
    from pronunciada.text import get_g2p

    g2p = get_g2p(lang)
    forms: dict[str, tuple[str, int]] = {}
    for w, c in _word_counts(lang).most_common():
        if len(w) > 1 and w.isupper():
            continue
        k = w.lower()
        if k not in forms:
            forms[k] = (w, c)
        if len(forms) >= top:
            break
    out = []
    for k, (w, c) in forms.items():
        try:
            ref = g2p(w)
        except Exception:  # noqa: BLE001
            continue
        if ref.warnings or len(ref.words) != 1:
            continue
        us = TS.units(ref, lang)
        if len(us) != 1 or us[0].text != w:
            continue
        out.append([w, list(us[0].phones), c])
    (work(lang) / "vocab.json").write_text(json.dumps(out, ensure_ascii=False))
    print(lang, "vocab entries", len(out))


class Vocab:
    def __init__(self, lang):
        rows = json.loads((work(lang) / "vocab.json").read_text())
        self.by_ph: dict[tuple, list] = defaultdict(list)
        self.rows = []
        for w, ph, c in rows:
            ph = tuple(ph)
            self.by_ph[ph].append((w, c))
            self.rows.append((w, ph, c))
        self.inv = sorted({p for _, ph, _ in self.rows for p in ph})
        self.frequent = [(w, ph, c) for w, ph, c in self.rows if c >= 30]

    def neighbours(self, ph: tuple) -> dict[str, list]:
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
        res, seen = [], set()
        if kind == "sub2":
            for i in range(len(ph)):
                for p in self.inv:
                    if p == ph[i]:
                        continue
                    q1 = ph[:i] + (p,) + ph[i + 1:]
                    for j in range(i + 1, len(ph)):
                        for p2 in self.inv:
                            if p2 != ph[j]:
                                q = q1[:j] + (p2,) + q1[j + 1:]
                                for w, c in self.by_ph.get(q, []):
                                    if (w, q) not in seen:
                                        seen.add((w, q))
                                        res.append((w, q, c, ("sub2",)))
        else:
            from bench.textswap import edit_distance

            L = len(ph)
            for w, q, c in self.rows:
                if 1 <= abs(len(q) - L) <= 2 and q and q[0] in ph[:2] + (ph[-1],) and edit_distance(ph, q) == 2:
                    res.append((w, q, c, ("indel2",)))
        return res


def sub_grade(a: str, b: str) -> str:
    from pronunciada import articulatory as A

    try:
        d = len(A.diff(a, b))
        typ = "type" in A.diff(a, b)
    except (KeyError, ValueError):
        return "sub_moderate"
    if d == 0 or typ:
        return "sub_gross" if d else "sub_subtle"
    return "sub_subtle" if d == 1 else "sub_moderate"


def _ok_target(u) -> bool:
    t = u.text
    return not u.oov and len(u.phones) >= 2 and t.isalpha() and not (len(t) > 1 and t.isupper())


def _position(k, n):
    return "initial" if k == 0 else "final" if k == n - 1 else "medial"


def make_changed(lang, uid, text, g2p, vocab: Vocab, kind_order):
    from bench import textswap as TS

    ref = g2p(text)
    ou = TS.units(ref, lang)
    pos = [k for k in range(len(ou)) if _ok_target(ou[k])]
    rng = random.Random(_h(uid, "pos"))
    rng.shuffle(pos)

    def replace(k, cands, kind, max_tries=25):
        u = ou[k]
        r2 = random.Random(_h(uid, kind, k))
        cands = sorted(cands, key=lambda e: (-min(e[2], 50), e[0]))
        head = cands[:12]
        r2.shuffle(head)
        for w, q, c, op in (head + cands[12:])[:max_tries]:
            if w.lower() == u.text.lower():
                continue
            nw = w[:1].upper() + w[1:] if u.text[:1].isupper() and k == 0 else w
            nt = text[:u.span[0]] + nw + text[u.span[1]:]
            try:
                ok = TS.verify(lang, g2p, ou, k, nt, nw, ref.warnings, lambda nu, q=q: nu.phones == q)
            except Exception:  # noqa: BLE001
                ok = None
            if ok:
                return nt, nw, q, op
        return None

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
                    try:
                        ref2 = g2p(nt)
                    except Exception:  # noqa: BLE001
                        continue
                    nu = TS.units(ref2, lang)
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
                cands = [e for e in vocab.neighbours(u.phones)["sub"] if sub_grade(e[3][1], e[3][2]) == kind]
                if kind == "sub_moderate" and len(u.phones) >= 4:
                    cands += vocab.two_edit(u.phones, "sub2")[:200]
            elif kind == "sub_gross":
                cands = [e for e in vocab.neighbours(u.phones)["sub"] if sub_grade(e[3][1], e[3][2]) == "sub_gross"]
                if not cands:
                    cands = [(w, q, c, ("far",)) for w, q, c in vocab.frequent if TS.is_far(u.phones, q) and len(q) >= 2][:400]
            else:
                nb = vocab.neighbours(u.phones)
                cands = nb["ins"] + nb["del"]
                if len(u.phones) >= 4:
                    cands += vocab.two_edit(u.phones, "indel2")[:200]
            if not cands:
                continue
            r = replace(k, cands, kind)
            if r:
                nt, nw, q, op = r
                n_ops = TS.edit_distance(u.phones, q)
                cat = "sub" if kind.startswith("sub") else ("insert" if len(q) > len(u.phones) else
                                                           "delete" if len(q) < len(u.phones) else "mixed")
                return nt, {"category": cat, "bucket": kind, "word_index": k, "original": u.text,
                            "replacement": nw, "n_ops": n_ops, "position": _position(k, len(ou)),
                            "orig_phones": " ".join(u.phones), "new_phones": " ".join(q),
                            "word_level_scorable": True}
    return None


def fleurs_dev_sources(lang: str, n: int) -> list[dict]:
    from bench.round19_rollout.sources import fleurs_val

    recs = sorted([r for r in fleurs_val(lang) if r["dur"] <= 20.0], key=lambda r: _h("fleurs", r["id"]))
    recs = recs[: int(n * 1.15)]
    materialise(lang, recs)
    return recs


def materialise(lang: str, recs: list[dict]) -> None:
    """FLEURS validation rows ("parquet:<path>:<row>") written once as 16 kHz FLAC; r["audio"] -> file."""
    from bench.data.common import write_flac
    from bench.pipeline.common import iter_parquet_audio

    d = work(lang) / "fleurs_val_audio"
    d.mkdir(exist_ok=True)
    want = {}
    for r in recs:
        if not str(r["audio"]).startswith("parquet:"):
            continue
        _, path, row = r["audio"].rsplit(":", 2)
        r["_flac"] = d / f"{r['id']}.flac"
        want.setdefault(path, {})[int(row)] = r
    for path, rows in want.items():
        todo = {i for i, r in rows.items() if not r["_flac"].exists()}
        if todo:
            for i, wav in iter_parquet_audio(path, todo):
                write_flac(rows[i]["_flac"], wav, 16000)
    for r in recs:
        if "_flac" in r:
            r["audio"] = str(r.pop("_flac"))


def cmd_build(lang: str) -> None:
    from bench.round19_rollout.sources import N_DEV_CV, N_DEV_FLEURS
    from pronunciada.text import get_g2p

    g2p = get_g2p(lang)
    vocab = Vocab(lang)
    srcs = [{"id": f"cv-{c['stem'].rsplit('_', 1)[-1]}", "source": "cv", "audio": c["audio"], "text": c["sentence"],
             "speaker": c["spk"], "dur": c["dur"]} for c in cv_rows(lang, "dev")]
    srcs = srcs[:N_DEV_CV]
    for r in fleurs_dev_sources(lang, N_DEV_FLEURS)[:N_DEV_FLEURS]:
        srcs.append({"id": f"fl-{r['id'].rsplit('-', 1)[-1]}", "source": "fleurs", "audio": r["audio"],
                     "text": r["text"], "speaker": "fl:" + r["id"], "dur": r["dur"]})
    out, cov = [], Counter()
    for i, s in enumerate(srcs):
        fold = "fit" if _h("fold", s["speaker"]) % 2 == 0 else "check"
        base = {"audio": s["audio"], "source": s["source"], "speaker": s["speaker"], "fold": fold,
                "prompt": s["text"], "dur": s["dur"], "lang": lang}
        try:
            g2p(s["text"])
        except Exception:  # noqa: BLE001
            continue
        out.append({"id": f"{s['id']}__ctl", **base, "target": s["text"], "label": None})
        start = _h(s["id"], "kind") % len(KINDS)
        order = [KINDS[(start + j) % len(KINDS)] for j in range(len(KINDS))]
        r = make_changed(lang, s["id"], s["text"], g2p, vocab, order)
        if r:
            nt, lab = r
            cov[(s["source"], lab["bucket"])] += 1
            out.append({"id": f"{s['id']}__chg", **base, "target": nt, "label": lab})
        else:
            cov[(s["source"], "none")] += 1
        if i % 100 == 0:
            print(i, len(srcs), flush=True)
    (work(lang) / "dev_manifest.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print({f"{a}/{b}": n for (a, b), n in sorted(cov.items())})
    print(lang, "items", len(out))


def manifest(lang: str) -> list[dict]:
    p = work(lang) / "dev_manifest.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["vocab", "build"])
    ap.add_argument("--lang", required=True)
    a = ap.parse_args(argv)
    {"vocab": cmd_vocab, "build": cmd_build}[a.cmd](a.lang)


if __name__ == "__main__":
    main()
