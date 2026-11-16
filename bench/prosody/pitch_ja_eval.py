"""Round 13: word-level Japanese pitch-accent judgement on dependable words, and the four bars.

    .venv/bin/python -m bench.prosody.pitch_ja_eval pitch_ja_v3r pitch_ja_v4 ... [heuristic]

Data (none used for training):
  val A / val B  FLEURS-ja VALIDATION natives split by sentence id (crc32 parity). A = dev for the
                 dependable-class filter and the threshold; B = report.
  hts            held-out HTS synthesis (validation texts, speed / half-tone outside the training
                 range): dictionary vs one edited accent phrase (bench.prosody.synth_ja)
  psola          natural validation recordings re-pitched (Praat PSOLA): unseen system and voices
Words judged: >= 2 morae, every mora labelled, accent class (RefSyllable.pitch_class, recomputed
from the text with the current front-end) in the dependable set chosen on val A: content-word
classes (noun, verb, adjective, adverb, adnominal; particles / auxiliaries never) with >= 15
words on val A whose native words exceed the provisional threshold (the score flagging 5 % of
val-A content words) no more often than 5 %, i.e. classes natives do not systematically depart
from. The class is pos / head-or-chained / single-or-compound accent phrase / accent type.
Word score (higher = more likely wrong):
  model      - mean over the word's morae of log P(expected high / low)
  heuristic  declination-corrected mora pitch (lf0 minus its local register, nucleus mean): the
             largest fall between adjacent morae of the word (and into the next mora) vs the
             fall at the expected nucleus: max_{i != k} fall_i - fall_k (k = none: - 0)
Threshold: the score flagging 2.5 % of judged words on val A (a margin under the 3 % bar).
Reports: val B flagged share; hts / psola: share of edited words caught and share of their
unedited twins' words flagged, AUC; coverage of running text (judged words / all words).
"""
from __future__ import annotations

import collections
import json
import os
import sys
import zlib

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")

from bench.prosody.common import OUT, load_prepped, save_json  # noqa: E402

MIN_N = 15          # occurrences of a class on val A to judge it
MAX_FLAG = 0.05     # a class is dependable if natives' words of it are flagged no more than average


def classes_for(u, g2p):
    ref = g2p(u["ref"].text)
    if len(ref.syllables) != len(u["ref"].syllables):
        return None
    return [s.pitch_class for s in ref.syllables]


from pronunciada.prosody.pitch_judge import class_key  # noqa: E402  (shared with the runtime)


def words(u, cls):
    """[(word index, syllable indices, class key)] for words with >= 2 labelled morae."""
    ref = u["ref"]
    out = []
    for wi, w in enumerate(ref.words):
        idx = list(range(*w.syllables))
        if len(idx) < 2 or any(ref.syllables[i].pitch_high is None for i in idx):
            continue
        out.append((wi, idx, class_key(cls[idx[0]]) if cls else None))
    return out


def model_probs(name, utts):
    from bench.prosody import prosody3 as P3

    net, cfg = P3.load_any("pitch", name)
    return [p for p, _, _ in P3.predict("pitch", net, [(u, 0) for u in utts], cfg)]


def heuristic_feats(utts):
    from bench.prosody.common import to_track
    from pronunciada.prosody.features import frame_features, register_features, syllable_windows

    out = []
    for u in utts:
        f = frame_features(to_track(u["track"]))
        r = register_features(f)[:, 0]
        wins = syllable_windows(u["ref"], u["spans"], len(f))
        m = np.array([r[w.nuc_start:w.nuc_end].mean() if w.nuc_end > w.nuc_start else r[w.start:w.end].mean()
                      for w in wins])
        out.append(m)
    return out


def word_scores(kind, P, u, ws):
    ref = u["ref"]
    out = []
    for wi, idx, key in ws:
        lab = np.array([ref.syllables[i].pitch_high for i in idx], bool)
        if kind == "model":
            p = np.clip(P[idx], 1e-4, 1 - 1e-4)
            out.append(float(-np.mean(np.log(np.where(lab, p, 1 - p)))))
        else:
            m = P
            ext = idx + ([idx[-1] + 1] if idx[-1] + 1 < len(m) else [])
            falls = {i: m[i] - m[i + 1] for i in ext[:-1]}
            k = next((i for i in idx[:-1] if lab[idx.index(i)] and not lab[idx.index(i) + 1]), None)
            if k is None and idx[-1] + 1 < len(ref.syllables) and lab[-1] and ref.syllables[idx[-1] + 1].pitch_high is False:
                k = idx[-1]
            fk = falls.get(k, 0.0) if k is not None else 0.0
            other = max([v for i, v in falls.items() if i != k] or [0.0])
            out.append(float(other - fk))
    return out


def run(names):
    from bench import metrics as M
    from pronunciada.text import get_g2p

    g2p = get_g2p("ja")
    sets = {"val": load_prepped("fleurs_val", "ja", "dev"), "hts": load_prepped("synja_hts", "ja", "heldout"),
            "psola": load_prepped("synja_psola", "ja", "heldout")}
    # the learner was asked for the DICTIONARY accent: score synthetic items against it (the
    # stored ref carries the accent actually synthesised, which is what training uses)
    import copy

    for k in ("hts", "psola"):
        for u in sets[k]:
            if u.get("wrong"):
                u["ref"] = copy.deepcopy(u["ref"])
                for s, d in zip(u["ref"].syllables, u["dict_pitch"]):
                    s.pitch_high = d
    cls = {k: [classes_for(u, g2p) for u in v] for k, v in sets.items()}
    half = ["A" if zlib.crc32(u["id"].encode()) % 2 == 0 else "B" for u in sets["val"]]
    res = {}
    for name in names:
        kind = "heuristic" if name == "heuristic" else "model"
        P = {k: (heuristic_feats(v) if kind == "heuristic" else model_probs(name, v)) for k, v in sets.items()}
        # word scores + class keys
        W = {}
        for k, utts in sets.items():
            rows = []
            for ui, u in enumerate(utts):
                ws = words(u, cls[k][ui])
                sc = word_scores(kind, P[k][ui], u, ws)
                ref = u["ref"]
                for (wi, idx, key), s in zip(ws, sc):
                    pred = (np.asarray(P[k][ui])[idx] > 0.5) if kind == "model" else None
                    agree = bool((pred == np.array([ref.syllables[i].pitch_high for i in idx])).all()) if pred is not None else None
                    rows.append({"u": ui, "w": wi, "key": key, "s": s, "agree": agree,
                                 "half": half[ui] if k == "val" else None, "wrong": bool(u.get("wrong")) and wi in u.get("wrong_words", []),
                                 "twin_of_wrong": (not u.get("wrong", False)) and wi in u.get("wrong_words", []),
                                 "n_words_utt": len(ref.words)})
            W[k] = rows
        # dependable classes on val A: native agreement of the word pattern (model) / low score (heuristic)
        A = [r for r in W["val"] if r["half"] == "A" and r["key"]]
        by = collections.defaultdict(list)
        for r in A:
            by[r["key"]].append(r)
        content = ("名詞", "動詞", "形容詞", "副詞", "連体詞")
        cand = [r for r in A if r["key"].split("/")[0] in content]
        thr0 = np.quantile([r["s"] for r in cand], 0.95)          # provisional: flags 5 % of content words
        stat = {key: (len(v), float(np.mean([r["s"] > thr0 for r in v]))) for key, v in by.items()}
        dep = sorted(k for k, (n, fl) in stat.items() if n >= MIN_N and fl <= MAX_FLAG and k.split("/")[0] in content)
        judged = lambda r: r["key"] in dep  # noqa: E731
        sa = np.array([r["s"] for r in A if judged(r)])
        thr = float(np.quantile(sa, 0.975)) if len(sa) else np.inf
        B = [r for r in W["val"] if r["half"] == "B" and judged(r)]
        nB_all = sum(len(sets["val"][ui]["ref"].words) for ui in set(range(len(sets["val"]))) if half[ui] == "B")
        sens = {"careful": float(np.quantile(sa, 0.99)), "balanced": thr, "eager": float(np.quantile(sa, 0.95))} \
            if len(sa) else {}
        out = {"dependable_classes": dep, "class_stats": stat, "threshold": thr, "thresholds": sens,
               "val_B": {"judged_words": len(B), "coverage_running_text": len(B) / max(nB_all, 1),
                         "flagged": float(np.mean([r["s"] > thr for r in B])) if B else None}}
        for k in ("hts", "psola"):
            pos = [r for r in W[k] if r["wrong"] and judged(r)]
            neg = [r for r in W[k] if r["twin_of_wrong"] and judged(r)]
            allneg = [r for r in W[k] if not r["wrong"] and judged(r) and not r["twin_of_wrong"]]
            s_p, s_n = np.array([r["s"] for r in pos]), np.array([r["s"] for r in neg + allneg])
            out[k] = {"n_wrong_judged": len(pos), "n_wrong_total": sum(r["wrong"] for r in W[k]),
                      "caught": float(np.mean(s_p > thr)) if len(s_p) else None,
                      "twin_flagged": float(np.mean([r["s"] > thr for r in neg])) if neg else None,
                      "other_words_flagged": float(np.mean([r["s"] > thr for r in allneg])) if allneg else None,
                      "auc": M.roc_auc(np.r_[np.ones(len(s_p)), np.zeros(len(s_n))].astype(int), np.r_[s_p, s_n])
                      if len(s_p) and len(s_n) else None}
        res[name] = out
        print(name, json.dumps({k: v for k, v in out.items() if k not in ("class_stats",)}, ensure_ascii=False,
                               default=lambda x: round(x, 4) if isinstance(x, float) else str(x)), flush=True)
    save_json(res, "r13_pitch_ja_eval.json")
    return res


if __name__ == "__main__":
    run(sys.argv[1:])
    _ = OUT
