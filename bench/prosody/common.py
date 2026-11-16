"""Shared helpers for prosody training / evaluation: load prepared utterances, build features,
syllable windows and network inputs, with optional augmentation."""
from __future__ import annotations

import json
import pickle
from dataclasses import replace

import numpy as np

from bench.prosody.prep import OUT, out_path
from pronunciada.prosody.f0 import F0Track
from pronunciada.prosody.features import (CTX_LEN, N_FEAT, SylWindow, context_input, contour_vector,
                                      frame_features, syllable_windows)


# experiment options (tone-ceiling diagnosis); defaults = the shipped configuration
OPTS = {"extra": "", "target": "syllable"}
_EXTRA_CACHE: dict = {}


def set_opts(**kw):
    OPTS.update(kw)


def n_extra() -> int:
    return {"": 0, "mfcc": 13, "lowspec": 16, "spec": 29, "post": 127, "post_only": 127}[OPTS["extra"]]


class _Unpickler(pickle.Unpickler):
    """Caches written before the package rename refer to `pronkit.*`."""

    def find_class(self, module, name):
        if module == "pronkit" or module.startswith("pronkit."):
            module = "pronunciada" + module[len("pronkit"):]
        return super().find_class(module, name)


def load_prepped(dataset: str, lang: str, split: str) -> list[dict]:
    with open(out_path(dataset, lang, split), "rb") as f:
        utts = _Unpickler(f).load()
    for u in utts:
        u["_src"] = (dataset, lang, split)
    return utts


def _extra_frames(u: dict, T: int):
    kind = OPTS["extra"]
    if not kind:
        return None
    ds, lang, split = u["_src"]
    if kind.startswith("post"):
        from bench.postcache import post_path

        lp = np.load(post_path("fleurs-validation" if ds == "fleurs_val" else ds, u["id"])).astype(np.float32)
        j = np.clip((np.arange(T) - 3) // 2, 0, len(lp) - 1)
        return np.clip(lp[j], -20, 0) / 10.0 + 1.0
    key = (ds, lang, split)
    if key not in _EXTRA_CACHE:
        _EXTRA_CACHE[key] = np.load(OUT / "extra" / f"{ds}-{lang}-{split}.npz")
    e = _EXTRA_CACHE[key][u["id"]].astype(np.float32)
    e = e[:T] if len(e) >= T else np.pad(e, ((0, T - len(e)), (0, 0)))
    return {"mfcc": e[:, :13], "lowspec": e[:, 13:], "spec": e}[kind]


def to_track(d: dict) -> F0Track:
    return F0Track(d["f0"].astype(np.float32), np.asarray(d["voiced"], bool),
                   d["voicing"].astype(np.float32), d["energy_db"].astype(np.float32))


def stretch_track(tr: F0Track, r: float) -> F0Track:
    """Resample a track in time by factor r (r < 1 = faster speech, fewer frames)."""
    T = len(tr.f0)
    T2 = max(2, int(round(T * r)))
    src = np.linspace(0, T - 1, T2)
    near = np.clip(np.round(src).astype(int), 0, T - 1)
    vo = tr.voiced[near]
    f0 = np.interp(src, np.arange(T), tr.f0)
    f0 = np.where(vo & (tr.f0[near] > 0), np.where(f0 > 0, f0, tr.f0[near]), 0.0)
    # interpolating across a voiced/unvoiced edge mixes in zeros: use the nearest voiced value
    lo, hi = np.floor(src).astype(int), np.minimum(np.ceil(src).astype(int), T - 1)
    edge = vo & ((tr.f0[lo] == 0) | (tr.f0[hi] == 0))
    f0[edge] = tr.f0[near][edge]
    return F0Track(f0.astype(np.float32), vo, np.interp(src, np.arange(T), tr.voicing).astype(np.float32),
                   np.interp(src, np.arange(T), tr.energy_db).astype(np.float32))


def scale_windows(wins: list[SylWindow], r: float, T: int) -> list[SylWindow]:
    out = []
    for w in wins:
        f = lambda x: int(np.clip(round(x * r), 0, T))  # noqa: E731
        s, e = f(w.start), max(f(w.end), f(w.start) + 1)
        ns, ne = max(f(w.nuc_start), s), min(max(f(w.nuc_end), f(w.nuc_start) + 1), e)
        if ne <= ns:
            ns, ne = s, e
        out.append(SylWindow(s, min(e, T), ns, ne, f(w.tok_start), f(w.tok_end)))
    return out


def jitter_windows(wins: list[SylWindow], rng, max_shift: int, T: int) -> list[SylWindow]:
    """Move every syllable start and end (and the nucleus) independently by up to +-max_shift
    frames (the aligner's resolution is 20 ms; +-4 frames = +-40 ms)."""
    if max_shift <= 0:
        return wins
    out = []
    for w in wins:
        d = rng.integers(-max_shift, max_shift + 1, size=3)
        s = int(np.clip(w.start + d[0], 0, max(T - 1, 0)))
        e = int(np.clip(w.end + d[1], s + 1, max(T, s + 1)))
        ns, ne = max(w.nuc_start + d[2], s), min(w.nuc_end + d[2], e)
        if ne <= ns:
            ns, ne = s, e
        out.append(replace(w, start=s, end=e, nuc_start=int(ns), nuc_end=int(ne)))
    return out


def utt_inputs(u: dict, rng=None, aug: bool = False, jitter: int = 0, tempo: float | None = None):
    """(feats [T, F], windows) for one prepared utterance, optionally augmented."""
    d = u["track"]
    if aug and "noisy" in u and rng.random() < 0.3:
        d = u["noisy"]
    tr = to_track(d)
    T0 = len(tr.f0)
    wins = syllable_windows(u["ref"], u["spans"], T0)
    r = tempo
    if aug and r is None:
        r = float(rng.uniform(0.85, 1.15))
    if r is not None and abs(r - 1) > 1e-3:
        tr = stretch_track(tr, r)
        wins = scale_windows(wins, len(tr.f0) / T0, len(tr.f0))
    feats = frame_features(tr)
    ex = _extra_frames(u, T0)
    if ex is not None:
        if len(feats) != T0:
            ex = ex[np.clip(np.round(np.linspace(0, T0 - 1, len(feats))).astype(int), 0, T0 - 1)]
        if OPTS["extra"] == "post_only":
            feats[:] = 0.0
        feats = np.concatenate([feats, ex], 1)
    if aug:
        s = float(rng.uniform(0.7, 1.3))
        feats[:, 0] *= s
        feats[:, 3] *= s
        feats[:, 0] += rng.normal(0, 0.05, len(feats)).astype(np.float32)
        feats[:, 2] += float(rng.normal(0, 0.2))
    if aug or jitter:
        wins = jitter_windows(wins, rng, jitter if jitter else 2, len(feats))
    return feats, wins


def tone_targets(u: dict, agree: dict | None = None):
    """Per syllable: (label 0..4 for tones 1..5, train mask, acceptable set)."""
    labs, mask, acc = [], [], []
    per = agree.get(u["id"]) if agree is not None else None
    for k, s in enumerate(u["ref"].syllables):
        t = s.tone
        labs.append((t - 1) if t else -1)
        ok = t is not None
        if per is not None:
            ok = ok and k < len(per) and per[k] in ("agree", "sandhi")
        mask.append(ok)
        acc.append({t, *s.tone_alts} if t else set())
    return np.array(labs), np.array(mask, bool), acc


def agree_map(u_list: list[dict]):
    """Per-syllable AISHELL agreement categories (from tone_agree.json, char-indexed) mapped to our
    syllables via prep's `ext`: computed directly here from ext for robustness."""
    from bench.prosody.tone_agree import categorize  # noqa: F401  (documented categories)

    out = {}
    for u in u_list:
        ext = u.get("ext")
        if ext is None:
            continue
        per = []
        for s, e in zip(u["ref"].syllables, ext):
            if e["erhua"] or e["tone"] is None or s.tone is None:
                per.append(None)
                continue
            if e["tone"] == s.tone:
                per.append("agree")
            elif e["tone"] in s.tone_alts:
                per.append("alt")
            else:
                ch = s.text[:1]
                lex = s.tone_lexical
                if ch in "一不" and {e["tone"], s.tone} <= {1, 2, 4}:
                    per.append("sandhi")
                elif {e["tone"], s.tone} == {2, 3} and lex == 3:
                    per.append("sandhi")
                else:
                    per.append("other")
        out[u["id"]] = per
    return out


def pitch_targets(u: dict):
    """Per mora: 1 high / 0 low / -1 unmarked."""
    labs = np.array([-1 if s.pitch_high is None else int(s.pitch_high) for s in u["ref"].syllables])
    return labs, labs >= 0, None


def build_arrays(utts, kind: str = "ctx", rng=None, aug=False, jitter=0, tempo=None, agree=None,
                 train_only=False, task: str = "tone"):
    """Stack network inputs over every labelled syllable. Returns X, y, meta list."""
    X, y, meta = [], [], []
    for ui, u in enumerate(utts):
        if not u["ref"].syllables:
            continue
        feats, wins = utt_inputs(u, rng, aug, jitter, tempo)
        labs, mask, acc = tone_targets(u, agree) if task == "tone" else pitch_targets(u)
        for i in range(len(wins)):
            if labs[i] < 0 or (train_only and not mask[i]):
                continue
            X.append(context_input(feats, wins, i, target=OPTS["target"]) if kind == "ctx"
                     else contour_vector(feats, wins, i))
            y.append(labs[i])
            meta.append((ui, i))
    if not X:
        shape = (0, CTX_LEN, N_FEAT + 5 + n_extra()) if kind == "ctx" else (0, 42)
        return np.zeros(shape, np.float32), np.zeros(0, int), []
    return np.stack(X), np.array(y), meta


def save_json(obj, name: str):
    p = OUT / "results" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(p, "w"), indent=1, ensure_ascii=False, default=float)
    return p
MODELS_DIR = OUT / "models"
