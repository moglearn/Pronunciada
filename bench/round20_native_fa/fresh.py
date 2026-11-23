"""Fresh confirmation sets: new clips from the same pools, same exclusion rules, same builder, none of the
clips or speakers of the diagnosed practice sets. Nobody reads their items; they are scored once at the end.

    python -m bench.round20_native_fa.fresh pool   --lang en,es,fr,de   # how many clips each pool can give
    python -m bench.round20_native_fa.fresh build  --lang en            # select, fetch, build the manifest

Common Voice (en, es, fr): the "dev" speakers of bench.round19_rollout.sources (CV dev split, cv-real dev half, no
Lingolix clip / speaker / prompt, no cv-real clip, accepted, not marked non-native, 1-12 s, >= 3 words, at most 2
clips per speaker) minus every speaker and sentence of the diagnosed set; speakers of the head-training roles
("train", "pool") are never used. German: the candidates of bench.round19.devset.cmd_select minus the diagnosed
speakers and sentences. FLEURS validation clips <= 20 s not in the diagnosed set (FLEURS test never read).
Items: bench.round19_rollout.devset.make_changed / bench.round19.devset.make_changed with the same kind rotation.
Korean: the round-19 pool (CV dev / test / other, 737 clips) and FLEURS ko validation (226) are fully used by the
diagnosed set; no fresh Korean clips without a new source.
Output: $CACHE/round20/nativefa/fresh/<lang>/dev_manifest.jsonl (same layout as the practice manifests).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter

from bench.round20_native_fa.rebuild import ROOT, manifest_path, patch_german, patch_rollout

N_CV, N_FL = 460, 260


def fresh_dir(lang):
    d = ROOT / "fresh" / lang
    d.mkdir(parents=True, exist_ok=True)
    return d


def diagnosed(lang):
    rows = [json.loads(l) for l in manifest_path(lang).read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["audio"] for r in rows}, {r["speaker"] for r in rows}, {r["prompt"] for r in rows}


def cv_candidates(lang):
    """All candidates of the original selection, in its seeded order (role dev only for the rollout builder)."""
    from bench.cv_build import nativeness, speaker_half

    if lang == "de":
        D = patch_german()
        L = json.loads((D.WORK / "lingolix_speakers.json").read_text())
        bad_spk, bad_clip = set(L["speakers"]), set(L["clips"])
        bad_sent = {D._norm(x["prompt"]) for x in D.lingolix_items()} | {D._norm(x["target"]) for x in D.lingolix_items()}
        durs = {}
        with open(D.CV_META / "clip_durations.tsv", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[1].strip().isdigit():
                    durs[p[0]] = int(p[1]) / 1000
        cands = []
        for r in D.read_tsv(D.CV_META / "dev.tsv"):
            cid = r["client_id"]
            if cid in bad_spk or r["path"] in bad_clip or D._norm(r["sentence"]) in bad_sent:
                continue
            if speaker_half(cid) != "dev" or int(r["up_votes"] or 0) < 2 or int(r["down_votes"] or 0) != 0:
                continue
            if nativeness("de", r.get("accents") or "") == "nonnative":
                continue
            d = durs.get(r["path"])
            if d is None or not 1.0 <= d <= 12.0 or len(r["sentence"].split()) < 3:
                continue
            cands.append({"stem": r["path"].rsplit(".", 1)[0], "speaker": cid, "sentence": r["sentence"], "dur": d})
        random.Random(D.SEED).shuffle(cands)
        return cands, D._norm
    S = patch_rollout()
    bad_clip, bad_spk, bad_sent = S.lingolix_exclusions(lang)
    bad_clip |= S.cvreal_clips(lang)
    durs = {}
    with open(S.cv_meta(lang) / "clip_durations.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1].strip().isdigit():
                durs[p[0]] = int(p[1]) / 1000
    cands = []
    for r in S.read_tsv(S.cv_meta(lang) / "dev.tsv"):
        cid = r["client_id"]
        if speaker_half(cid) != "dev" or cid in bad_spk or r["path"] in bad_clip or S.norm(r["sentence"]) in bad_sent:
            continue
        if int(r["up_votes"] or 0) < 2 or int(r["down_votes"] or 0) != 0:
            continue
        if nativeness(S.locale(lang), r.get("accents") or "") == "nonnative":
            continue
        d = durs.get(r["path"])
        if d is None or not 1.0 <= d <= 12.0 or len(r["sentence"].split()) < 3:
            continue
        if S.speaker_role(cid) != "dev":
            continue
        cands.append({"stem": r["path"].rsplit(".", 1)[0], "speaker": cid, "sentence": r["sentence"], "dur": d})
    random.Random(S.SEED).shuffle(cands)
    return cands, S.norm


def spk_tag(cid):
    return "cv:" + hashlib.sha1(cid.encode()).hexdigest()[:12]


def select(lang):
    used_audio, used_spk, used_prompt = diagnosed(lang)
    cands, norm = cv_candidates(lang)
    used_sent = {norm(p) for p in used_prompt}
    per, out, seen = Counter(), [], set()
    for c in cands:
        if spk_tag(c["speaker"]) in used_spk or norm(c["sentence"]) in used_sent or norm(c["sentence"]) in seen:
            continue
        if per[c["speaker"]] >= 2:
            continue
        per[c["speaker"]] += 1
        seen.add(norm(c["sentence"]))
        out.append(c)
    from bench.pipeline.common import records

    fl = [r for r in records("fleurs-validation", lang=lang) if r["dur"] <= 20.0]
    used_fl = {a.rsplit("/", 1)[-1].rsplit(".", 1)[0] for a in used_audio}
    fl = [r for r in fl if r["id"] not in used_fl and norm(r["text"]) not in used_sent]
    return out, fl


def cmd_pool(lang):
    cv, fl = select(lang)
    print(lang, "fresh CV clips available", len(cv), "speakers", len({c["speaker"] for c in cv}), "FLEURS val", len(fl))
    return {"cv": len(cv), "cv_speakers": len({c["speaker"] for c in cv}), "fleurs": len(fl)}


def fetch(lang, stems):
    import io
    import tarfile

    import requests
    import soundfile as sf

    from bench.cv_build import n_shards, shard_url
    from bench.data.common import resample, write_flac

    loc = {"zh": "zh-CN"}.get(lang, lang)
    d = fresh_dir(lang) / "cv_audio"
    d.mkdir(exist_ok=True)
    todo = {s for s in stems if not (d / f"{s}.flac").exists()}
    try:
        ns = int(n_shards(loc).get("dev", 1))
    except Exception:  # noqa: BLE001
        ns = 1
    for k in range(ns):
        if not todo:
            break
        with requests.get(shard_url(loc, "dev", k), stream=True, timeout=120) as r:
            r.raise_for_status()
            r.raw.decode_content = True
            with tarfile.open(fileobj=io.BufferedReader(r.raw, 1 << 20), mode="r|") as t:
                for m in t:
                    if not m.isfile():
                        continue
                    stem = m.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                    if stem not in todo:
                        continue
                    x, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32", always_2d=True)
                    write_flac(d / f"{stem}.flac", resample(x.mean(axis=1), sr), 16000)
                    todo.discard(stem)
                    if not todo:
                        break
        print(lang, "shard", k, "left", len(todo), flush=True)
    return d


def cmd_build(lang):
    from bench.round19_rollout.devset import materialise
    from pronunciada.text import get_g2p

    cv, fl = select(lang)
    n_cv = N_CV + N_FL - min(len(fl), N_FL)       # FLEURS validation is nearly used up: fill with CV clips
    cv = cv[: int(n_cv * 1.15)]
    d = fetch(lang, [c["stem"] for c in cv])
    fl = sorted(fl, key=lambda r: hashlib.sha256(("fresh|" + r["id"]).encode()).hexdigest())[: int(N_FL * 1.15)]
    S = patch_rollout()
    S.ROOT = ROOT / "fresh" / "_fl"           # FLEURS FLAC files of the fresh set: their own folder
    materialise(lang, fl)
    if lang == "de":
        D = patch_german()
        make, vocab, kinds, h = D.make_changed, D.Vocab(), D.KINDS, D._h
        mk = lambda uid, text, g2p, order: make(uid, text, g2p, vocab, order)  # noqa: E731
    else:
        import bench.round19_rollout.devset as V

        S.ROOT = ROOT / "rollout"                 # the practice vocabulary (same word list as the diagnosed set)
        vocab, kinds, h = V.Vocab(lang), V.KINDS, V._h
        mk = lambda uid, text, g2p, order: V.make_changed(lang, uid, text, g2p, vocab, order)  # noqa: E731
    g2p = get_g2p(lang)
    srcs = []
    for c in cv:
        p = d / f"{c['stem']}.flac"
        if p.exists():
            srcs.append({"id": f"cv-{c['stem'].rsplit('_', 1)[-1]}", "source": "cv", "audio": str(p), "text": c["sentence"],
                         "speaker": spk_tag(c["speaker"]), "dur": c["dur"]})
    srcs = srcs[:n_cv]
    for r in fl[:N_FL]:
        srcs.append({"id": f"fl-{r['id'].rsplit('-', 1)[-1]}", "source": "fleurs", "audio": r["audio"], "text": r["text"],
                     "speaker": "fl:" + r["id"], "dur": r["dur"]})
    out, cov = [], Counter()
    for s in srcs:
        fold = "fit" if h("fold", s["speaker"]) % 2 == 0 else "check"
        base = {"audio": s["audio"], "source": s["source"], "speaker": s["speaker"], "fold": fold, "prompt": s["text"],
                "dur": s["dur"], "lang": lang}
        try:
            g2p(s["text"])
        except Exception:  # noqa: BLE001
            continue
        out.append({"id": f"{s['id']}__ctl", **base, "target": s["text"], "label": None})
        start = h(s["id"], "kind") % len(kinds)
        order = [kinds[(start + j) % len(kinds)] for j in range(len(kinds))]
        r = mk(s["id"], s["text"], g2p, order)
        if r:
            nt, lab = r
            cov[(s["source"], lab["bucket"])] += 1
            out.append({"id": f"{s['id']}__chg", **base, "target": nt, "label": lab})
    (fresh_dir(lang) / "dev_manifest.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(lang, "fresh items", len(out), "clips", len(srcs), dict(Counter(s["source"] for s in srcs)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pool", "build"])
    ap.add_argument("--lang", required=True)
    a = ap.parse_args(argv)
    res = {}
    for l in a.lang.split(","):
        res[l] = (cmd_pool if a.cmd == "pool" else cmd_build)(l)
    if a.cmd == "pool":
        from bench.round20_native_fa import core as K

        K.save_json("fresh_pool.json", res)


if __name__ == "__main__":
    main()
