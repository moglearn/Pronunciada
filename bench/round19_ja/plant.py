"""Round-19 Japanese dev / fit sets: native clips whose reference text has one unit's core replaced
by another real word, the kinds of edit the Lingolix test-1 Japanese items use (planted here by
our own generator; nothing is read from those items except the exclusion lists).

    python -m bench.round19_ja.plant vocab              # -> <work>/vocab.json
    python -m bench.round19_ja.plant build --pool dev   # -> <work>/set_dev.jsonl
    python -m bench.round19_ja.plant build --pool fit   # -> <work>/set_fit.jsonl

Units are accent phrases of Open JTalk (a content word plus the particles / auxiliaries chained
to it, NJD chain_flag), close to the benchmark's bunsetsu units. A unit's *core* is the unit minus
trailing case / binding particles. Edits replace one core with a vocabulary core (real words: cores
seen in Common Voice ja / FLEURS ja sentences, the Lingolix prompts excluded) related to it by:

* length: the readings agree once length is removed (ー, っ, and a vowel kana after the same
  vowel: バアイ ~ バイ, シイ ~ シ), and differ;
* sub_subtle / sub_moderate / sub_gross: same phone count, one phone substituted (articulatory
  feature differences 1 / 2 / >= 3 or vowel <-> consonant), length keys differ;
* indel: one phone inserted or deleted, length keys differ.

The front-end is run on the edited text and the edit is kept only if every other unit's phones
are unchanged. Every clip is used once, as a control or as an edited item (target mix: length 40 %,
subtle 20 %, moderate 15 %, gross 10 %, indel 15 %, as the benchmark's Japanese test).
Output rows are dump.py inputs with "units", "label", "group" (speaker or clip) and "source".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from bench.round19_ja.cv_pool import WORK, excluded, norm_sentence, read_tsv

MIX = {"length": 0.40, "sub_subtle": 0.20, "sub_moderate": 0.15, "sub_gross": 0.10, "indel": 0.15}
_KANA = re.compile(r"^[ァ-ヴー]+$")
_JA = re.compile(r"^[ぁ-ゟ゠-ヿ一-鿿々ー]+$")
_TAIL_PARTICLES = {"格助詞", "係助詞", "副助詞", "並立助詞", "終助詞", "副助詞／並立助詞／終助詞", "連体化"}
VROW = {}
for v, row in (("a", "アカガサザタダナハバパマヤラワァャ"), ("i", "イキギシジチヂニヒビピミリィ"),
               ("u", "ウクグスズツヅヌフブプムユルゥュヴ"), ("e", "エケゲセゼテデネヘベペメレェ"),
               ("o", "オコゴソゾトドノホボポモヨロヲォョ")):
    for ch in row:
        VROW[ch] = v
VKANA = {"a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ"}


def len_key(pron: str) -> str:
    """Reading with length removed: ー and ッ dropped, a bare vowel kana after the same vowel dropped."""
    out = []
    last_v = None
    for ch in pron:
        if ch in "ーッ’'":
            continue
        if ch in "アイウエオ" and last_v is not None and VKANA[last_v] == ch:
            continue
        out.append(ch)
        last_v = VROW.get(ch, last_v if ch in "ャュョ" else None)
        if ch in "ャ":
            last_v = "a"
        elif ch in "ュ":
            last_v = "u"
        elif ch in "ョ":
            last_v = "o"
    return "".join(out)


# ----------------------------------------------------------------------------- units

_G2P = None


def g2p():
    global _G2P
    if _G2P is None:
        from pronunciada.text.ja import JaG2P

        _G2P = JaG2P()
    return _G2P


def njd_units(text: str):
    """[(start, end, core_end, pron_core, [njd words])] accent-phrase units with char spans."""
    import pyopenjtalk

    from pronunciada.text.ja import kana_moras

    njd = pyopenjtalk.run_frontend(text)
    units, cur = [], 0
    pos_text = 0
    for x in njd:
        s = x["string"]
        j = text.find(s, pos_text)
        if j < 0:
            return None
        pos_text = j + len(s)
        speak = x["mora_size"] > 0 and kana_moras(x["pron"])
        if not speak:
            cur = None
            continue
        if cur is None or x["chain_flag"] != 1 or not units:
            units.append({"start": j, "end": j + len(s), "words": [x]})
        else:
            units[-1]["end"] = j + len(s)
            units[-1]["words"].append(x)
        cur = 1
    out = []
    for u in units:
        ws = u["words"]
        k = len(ws)
        while k > 1 and ws[k - 1]["pos"] == "助詞" and ws[k - 1]["pos_group1"] in _TAIL_PARTICLES:
            k -= 1
        core_len = sum(len(w["string"]) for w in ws[:k])
        core_text = text[u["start"]:u["start"] + core_len]
        if core_text != "".join(w["string"] for w in ws[:k]):
            return None
        pron = "".join(w["pron"] for w in ws[:k]).replace("’", "")
        out.append({"start": u["start"], "end": u["end"], "core_end": u["start"] + core_len,
                     "text": text[u["start"]:u["end"]], "core": core_text, "pron": pron,
                     "pos": ws[0]["pos"]})
    return out


def unit_phones(ref, units):
    """Phones (tuple) of each unit's core and of the whole unit, from a Reference."""
    res = []
    for u in units:
        core, full = [], []
        for w in ref.words:
            a, b = w.char_span
            ph = [ref.phones[i].ipa for i in range(*w.phones)]
            if a >= u["start"] and b <= u["end"]:
                full += ph
                if b <= u["core_end"]:
                    core += ph
        res.append((tuple(core), tuple(full)))
    return res


# ----------------------------------------------------------------------------- vocabulary

def sentences():
    bad_spk, bad_prompts = excluded()
    seen = set()
    src = [WORK / "cvmeta" / "validated.tsv"]
    for p in src:
        for r in read_tsv(p):
            s = r["sentence"]
            n = norm_sentence(s)
            if n in seen or n in bad_prompts or r["client_id"] in bad_spk:
                continue
            seen.add(n)
            yield s
    from bench.pipeline.common import load_manifest

    for split in ("train",):
        for r in load_manifest("fleurs", split=split, lang="ja"):
            n = norm_sentence(r["text"])
            if n not in seen and n not in bad_prompts:
                seen.add(n)
                yield r["text"]


def build_vocab(limit: int | None = None) -> None:
    from bench.cv_build import text_rule

    cnt: dict[str, Counter] = defaultdict(Counter)
    n = 0
    for s in sentences():
        if text_rule(s, "ja"):
            continue
        us = njd_units(s)
        if not us:
            continue
        for u in us:
            if _JA.match(u["core"]) and _KANA.match(u["pron"]) and u["pos"] not in ("助詞", "助動詞", "記号"):
                cnt[u["core"]][u["pron"]] += 1
        n += 1
        if limit and n >= limit:
            break
    vocab = {}
    for w, c in cnt.items():
        pron, k = c.most_common(1)[0]
        if k / sum(c.values()) >= 0.8:
            vocab[w] = {"pron": pron, "n": sum(c.values())}
    (WORK / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    print(f"{n} sentences, {len(vocab)} cores", file=sys.stderr)


# ----------------------------------------------------------------------------- relations

def _feat_diff(a: str, b: str) -> int:
    from pronunciada.articulatory import covered, diff, features

    if not (covered(a) and covered(b)):
        return 9
    if features(a)["type"] != features(b)["type"]:
        return 9
    return len(diff(a, b))


def relation(pa: tuple, pb: tuple, ka: str, kb: str) -> str | None:
    """Kind of edit from phones pa (original core) to pb (replacement core); ka / kb: length keys."""
    if pa == pb:
        return None
    if ka == kb:
        return "length"
    if len(pa) == len(pb):
        d = [(x, y) for x, y in zip(pa, pb) if x != y]
        if len(d) != 1:
            return None
        x, y = d[0]
        if x.replace("ː", "") == y.replace("ː", "") or (len(x) == 2 and x[0] == x[1]) or (len(y) == 2 and y[0] == y[1]):
            return None
        f = _feat_diff(x, y)
        return "sub_subtle" if f <= 1 else "sub_moderate" if f == 2 else "sub_gross"
    if abs(len(pa) - len(pb)) == 1:
        s, t = (pa, pb) if len(pa) < len(pb) else (pb, pa)
        for i in range(len(t)):
            if t[:i] + t[i + 1:] == s:
                return "indel"
    return None


class Index:
    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.by_key = defaultdict(list)
        self.by_len = defaultdict(list)
        for w, d in vocab.items():
            k = len_key(d["pron"])
            self.by_key[k].append(w)
            self.by_len[len(d["pron"])].append(w)

    def candidates(self, core: str, pron: str, kind: str):
        if kind == "length":
            return [w for w in self.by_key[len_key(pron)] if self.vocab[w]["pron"] != pron and w != core]
        n = len(pron)
        out = []
        for m in (n - 1, n, n + 1):
            out += [w for w in self.by_len[m] if w != core]
        return out


_PH_CACHE: dict = {}


def iso_phones(word: str, pron: str):
    """Phones of a vocabulary core said on its own (cached)."""
    if word not in _PH_CACHE:
        r = g2p()(word)
        _PH_CACHE[word] = tuple(r.phones[i].ipa for i in range(len(r.phones))) if not r.warnings else None
    return _PH_CACHE[word]


def try_plant(text: str, kind: str, idx: Index, rng: random.Random, max_tries: int = 30):
    """(new_text, units, label) or None."""
    us = njd_units(text)
    if not us or len(us) < 3:
        return None
    ref0 = g2p()(text)
    if ref0.warnings:
        return None
    ph0 = unit_phones(ref0, us)
    order = list(range(len(us)))
    rng.shuffle(order)
    tries = 0
    for k in order:
        u = us[k]
        if not (_JA.match(u["core"]) and _KANA.match(u["pron"])) or u["pos"] in ("助詞", "助動詞"):
            continue
        core_ph = ph0[k][0]
        if len(core_ph) < 2:
            continue
        cands = idx.candidates(u["core"], u["pron"], kind)
        if kind != "length":
            ka = len_key(u["pron"])
            cands = [w for w in cands if len_key(idx.vocab[w]["pron"]) != ka]
            iso = [(w, iso_phones(w, idx.vocab[w]["pron"])) for w in cands]
            cands = [w for w, p in iso if p and relation(core_ph, p, ka, len_key(idx.vocab[w]["pron"])) == kind]
        cands.sort(key=lambda w: (-idx.vocab[w]["n"], w))
        top = cands[:8]
        rng.shuffle(top)
        for w in top:
            tries += 1
            if tries > max_tries:
                return None
            new = text[:u["start"]] + w + text[u["core_end"]:]
            nus = njd_units(new)
            if not nus or len(nus) != len(us):
                continue
            ref1 = g2p()(new)
            if ref1.warnings:
                continue
            ph1 = unit_phones(ref1, nus)
            ok = all(ph0[i][1] == ph1[i][1] and us[i]["text"] == nus[i]["text"] for i in range(len(us)) if i != k)
            if not ok or nus[k]["core"] != w:
                continue
            rel = relation(core_ph, ph1[k][0], len_key(u["pron"]), len_key(nus[k]["pron"]))
            if rel != kind:
                continue
            units = [{"text": x["text"], "start": x["start"], "end": x["end"]} for x in nus]
            label = {"bucket": kind, "category": "length" if kind == "length" else ("sub" if kind.startswith("sub") else "indel"),
                     "word_index": k, "original": u["core"], "replacement": w,
                     "orig_phones": " ".join(core_ph), "new_phones": " ".join(ph1[k][0]),
                     "word_level_scorable": True}
            return new, units, label
    return None


# ----------------------------------------------------------------------------- sets

def pool_rows(pool: str) -> list[dict]:
    """Native clips for a pool: CV (round19 cv_pool) plus FLEURS (dev: validation; fit: train)."""
    rows = []
    for x in open(WORK / f"cv_{pool}.jsonl", encoding="utf-8"):
        r = json.loads(x)
        rows.append({"id": r["id"], "audio": r["audio"], "text": r["text"], "group": r["speaker"],
                     "source": "cv"})
    from bench.pipeline.common import records

    if pool == "dev":
        fl = records("fleurs-validation", lang="ja")
    else:
        fl = records("fleurs", split="train", lang="ja")
    for r in fl:
        a = r["audio"].replace("/.cache/pronkit/", "/.cache/pronunciada/")
        # FLEURS has no speaker ids; a clip is its own group (validation speakers differ from train)
        rows.append({"id": r["id"], "audio": a, "text": r["text"], "group": r["id"], "source": "fleurs"})
    return rows


def _seed(*k) -> int:
    return int(hashlib.sha1("|".join(map(str, k)).encode()).hexdigest()[:12], 16)


def build(pool: str, n_fleurs: int | None = None) -> None:
    from bench.cv_build import text_rule

    idx = Index(json.loads((WORK / "vocab.json").read_text()))
    rows = [r for r in pool_rows(pool) if not text_rule(r["text"], "ja")]
    rng = random.Random(_seed("round19", pool))
    cv = [r for r in rows if r["source"] == "cv"]
    fl = [r for r in rows if r["source"] == "fleurs"]
    rng.shuffle(fl)
    if n_fleurs:
        fl = fl[:n_fleurs]
    out = []
    stats = Counter()
    for src, rs in (("cv", cv), ("fleurs", fl)):
        rng.shuffle(rs)
        n_edit = len(rs) // 2
        kinds = []
        for k, p in MIX.items():
            kinds += [k] * round(p * n_edit)
        kinds = kinds[:n_edit]
        kinds += ["length"] * (n_edit - len(kinds))
        rest = []
        ki = 0
        for r in rs:
            if ki < len(kinds):
                res = None
                for kind in [kinds[ki]] + [k for k in MIX if k != kinds[ki]]:
                    res = try_plant(r["text"], kind, idx, random.Random(_seed(r["id"], kind)))
                    if res:
                        break
                if res:
                    new, units, label = res
                    out.append({**r, "text": new, "prompt": r["text"], "units": units, "label": label})
                    stats[(src, label["bucket"])] += 1
                    ki += 1
                    continue
            rest.append(r)
        for r in rest:
            us = njd_units(r["text"])
            ref = g2p()(r["text"])
            if not us or ref.warnings or len(us) < 3:
                continue
            out.append({**r, "prompt": r["text"], "units": [{"text": x["text"], "start": x["start"], "end": x["end"]}
                                                             for x in us], "label": None})
            stats[(src, "control")] += 1
    with open(WORK / f"set_{pool}.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(pool, len(out), sorted(stats.items()), file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("vocab", "build"))
    ap.add_argument("--pool", default="dev")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--n-fleurs", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "vocab":
        build_vocab(a.limit)
    else:
        build(a.pool, a.n_fleurs)


if __name__ == "__main__":
    main()
