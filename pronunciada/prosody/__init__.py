"""Tone / word stress / pitch accent scoring (DESIGN section 5).

    scorer = ProsodyScorer("zh")
    out = scorer.score(wav, ref, evidence)     # one dict per RefSyllable

`evidence` is the pronunciada.align.Evidence for `ref` (only the phone spans are used; a list of
(start_frame, end_frame) per phone in ZIPA frames is accepted too).

Keys, as applicable:
  zh: tone_probs (5 floats, tones 1-5), tone_heard (1-5), tone_prob (calibrated P(the tone or one
      of its tone_alts was produced))
  en de es it pt ru: stress_prob on the syllable with primary stress of every word with >= 2
      syllables (calibrated P(the word's primary stress is where expected)); None elsewhere
  ja: pitch_prob (calibrated P(mora has the expected high/low pitch)), pitch_high_prob
Models: ONNX files + JSON metadata in pronunciada/data/prosody/ (run with onnxruntime).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .f0 import F0Track, track  # noqa: F401
from .f0 import track as _track
from .features import (N_STRESS_EXTRA, context_input, frame_features, stress_extra,  # noqa: F401
                       syllable_windows)

DATA = Path(__file__).resolve().parent.parent / "data" / "prosody"
# Mandarin tone model file stem in DATA. tone_zh = round 1 (one syllable + neighbours);
# tone_zh_v2 = utterance-level sequence model (sees all syllables of the utterance in order);
# tone_zh_v3 = v2 + local-register pitch + low-frequency log-mel; tone_zh_v3z = v3 + pooled ZIPA
# encoder activations per syllable (default since round 7: results/tone_v3.md, results/round7.md).
# PRONUNCIADA_TONE_MODEL overrides it; a non-default model uses calib/prosody/zh.<model>.json.
TONE_MODEL = "tone_zh_v3z"
# Word stress (en de es it pt ru) and Japanese pitch accent: shipped models and the environment
# variables that select an alternative (round 8: stress_v3 / pitch_ja_v3, results/round8_prosody.md).
STRESS_MODEL = "stress"
PITCH_MODEL = "pitch_ja"
STRESS_LANGS = ("en", "de", "es", "it", "pt", "ru")
_KIND_ENV = {"tone": ("PRONUNCIADA_TONE_MODEL", "TONE_MODEL"), "stress": ("PRONUNCIADA_STRESS_MODEL", "STRESS_MODEL"),
             "pitch": ("PRONUNCIADA_PITCH_MODEL", "PITCH_MODEL")}


def kind_of(lang: str) -> str | None:
    return "tone" if lang == "zh" else "pitch" if lang == "ja" else "stress" if lang in STRESS_LANGS else None


def default_model(lang: str) -> str | None:
    k = kind_of(lang)
    return globals()[_KIND_ENV[k][1]] if k else None


def selected_model(lang: str) -> str | None:
    """Stem of the prosody model in use for `lang` (its environment variable, else the default)."""
    k = kind_of(lang)
    if k is None:
        return None
    return Path(os.environ.get(_KIND_ENV[k][0], default_model(lang))).name


def needs_embeddings(lang: str) -> bool:
    """True when the selected prosody model of `lang` reads ZIPA encoder activations (meta
    inputs.zemb), so the caller must run the acoustic model with embeddings=True."""
    k = kind_of(lang)
    if k is None:
        return False
    stem = os.environ.get(_KIND_ENV[k][0], default_model(lang))
    p = Path(stem + ".json") if os.sep in stem else DATA / f"{stem}.json"
    try:
        return bool(json.loads(p.read_text()).get("inputs", {}).get("zemb"))
    except OSError:
        return False


def selected_tone_model() -> str:
    """Stem of the Mandarin tone model in use (PRONUNCIADA_TONE_MODEL or TONE_MODEL)."""
    return Path(os.environ.get("PRONUNCIADA_TONE_MODEL", TONE_MODEL)).name


def tone_needs_embeddings() -> bool:
    """True when the selected tone model reads ZIPA encoder activations (meta inputs.zemb), so the
    caller must run the acoustic model with embeddings=True and pass them to score()."""
    stem = os.environ.get("PRONUNCIADA_TONE_MODEL", TONE_MODEL)
    p = Path(stem + ".json") if os.sep in stem else DATA / f"{stem}.json"
    try:
        return bool(json.loads(p.read_text()).get("inputs", {}).get("zemb"))
    except OSError:
        return False


def _platt(p, ab):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(ab[0] * x + ab[1])))


def _session(path: Path, threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _phone_spans(evidence):
    if hasattr(evidence, "slots"):
        return [(s.start_frame, s.end_frame) for s in evidence.slots]
    return list(evidence)


class ProsodyScorer:
    def __init__(self, lang: str, data_dir: str | os.PathLike | None = None, threads: int = 2):
        from pronunciada.text.base import normalize_lang

        self.lang = normalize_lang(lang)
        d = Path(data_dir) if data_dir else DATA
        self.kind = None
        self.sess = None
        self.meta: dict = {}
        if self.lang == "zh":
            self.kind, stem = "tone", os.environ.get("PRONUNCIADA_TONE_MODEL", TONE_MODEL)
            if os.sep in stem:                        # a path stem outside the package (experiments)
                d, stem = Path(stem).parent, Path(stem).name
        elif self.lang == "ja":
            self.kind, stem = "pitch", os.environ.get("PRONUNCIADA_PITCH_MODEL", PITCH_MODEL)
        elif self.lang in STRESS_LANGS:
            self.kind, stem = "stress", os.environ.get("PRONUNCIADA_STRESS_MODEL", STRESS_MODEL)
        else:
            return                                    # fr, ko: nothing to score
        self.meta = json.load(open(d / f"{stem}.json"))
        self.sess = _session(d / f"{stem}.onnx", threads)
        self.inputs = self.meta.get("inputs", {})     # tone v3: extra frame inputs
        if self.inputs.get("zemb"):                   # ZIPA-activation variant: fixed PCA
            z = np.load(d / f"{stem}.zemb.npz")
            self.zemb = (z["mu"].astype(np.float32), z["W"].astype(np.float32), list(self.meta["zemb_layers"]))
        if self.kind == "tone":
            from .tone_leniency import enabled

            self.leniency = enabled(self.meta)
        if self.kind == "stress":
            langs = self.meta["langs"]
            self.lang_id = langs.index(self.lang) if self.lang in langs else 0
            self.platt = self.meta["platt"].get(self.lang, self.meta["platt"].get("all"))

    # ------------------------------------------------------------------ public
    def score(self, wav, ref, evidence, track=None, embeddings=None) -> list[dict]:
        """track: the F0Track of wav if the caller already computed it. embeddings: the acoustic
        model's {layer: [T', C]} activations (needed only by a tone model with inputs.zemb)."""
        out = [dict() for _ in ref.syllables]
        if self.sess is None or not ref.syllables:
            return out
        tr = track if track is not None else _track(np.asarray(wav, dtype=np.float32))
        feats = frame_features(tr)
        wins = syllable_windows(ref, _phone_spans(evidence), len(feats))
        if self.inputs:
            feats = self.extend_features(feats, wav)
        if self.inputs.get("zemb"):
            feats = np.concatenate([feats, self.zemb_frames(ref, _phone_spans(evidence), wins, embeddings,
                                                            len(feats))], 1)
        return self.score_features(feats, wins, ref, out)

    def zemb_frames(self, ref, spans, wins, embeddings, T):
        """[T, 33]: per syllable, the mean ZIPA activation (layers of meta zemb_layers) over its
        aligned phone frames, PCA-projected, broadcast over the syllable's window, + a flag."""
        from .features import pool_syllables

        mu, W, layers = self.zemb
        if embeddings is None or any(k not in embeddings for k in layers):
            # no activations given (direct ProsodyScorer use): the model was trained with this
            # stream dropped for 30 % of utterances (flag 0), so it falls back to pitch + low mel.
            # pronunciada.Scorer always passes them.
            return np.zeros((T, W.shape[1] + 1), np.float32)
        emb = np.concatenate([np.asarray(embeddings[k], np.float32) for k in layers], 1)
        Z = np.clip((pool_syllables(ref, spans, emb) - mu) @ W, -5, 5)
        out = np.zeros((T, W.shape[1] + 1), np.float32)
        for si, w in enumerate(wins):
            out[w.start:w.end, :-1] = Z[si]
            out[w.start:w.end, -1] = 1.0
        return out

    def extend_features(self, feats, wav):
        """Append the extra frame inputs a v3 tone model expects (meta "inputs"): local-register
        pitch features and / or the low-frequency log-mel patch (computed from `wav` here)."""
        from .features import lowmel_features, register_features

        cols = [feats]
        if self.inputs.get("reg"):
            cols.append(register_features(feats))
        if self.inputs.get("lowmel"):
            cols.append(lowmel_features(np.asarray(wav, dtype=np.float32), len(feats)))
        return np.concatenate(cols, 1)

    def score_features(self, feats, wins, ref, out=None) -> list[dict]:
        out = out if out is not None else [dict() for _ in ref.syllables]
        if self.kind == "tone":
            return self._tone(feats, wins, ref, out)
        if self.kind == "pitch":
            return self._pitch(feats, wins, ref, out)
        return self._stress(feats, wins, ref, out)

    # ------------------------------------------------------------------ kinds
    def _ctx(self, feats, wins, idx):
        return np.stack([context_input(feats, wins, i) for i in idx]).astype(np.float32)

    def _tone(self, feats, wins, ref, out):
        idx = [i for i, s in enumerate(ref.syllables) if s.tone is not None]
        if not idx:
            return out
        if self.meta.get("sequence"):     # utterance-level model: every syllable, in order
            z = self.sess.run(None, {"x": self._ctx(feats, wins, range(len(ref.syllables)))})[0][idx]
        else:
            z = self.sess.run(None, {"x": self._ctx(feats, wins, idx)})[0]
        z = z / self.meta["temperature"]
        z = z - z.max(1, keepdims=True)
        P = np.exp(z)
        P /= P.sum(1, keepdims=True)
        for k, i in enumerate(idx):
            s = ref.syllables[i]
            acc = {s.tone, *s.tone_alts}
            if self.leniency:
                from .tone_leniency import extra_tones

                acc |= extra_tones(ref, i, self.leniency)
            p_acc = float(sum(P[k, t - 1] for t in acc if 1 <= t <= 5))
            out[i].update(tone_probs=[float(v) for v in P[k]], tone_heard=int(P[k].argmax()) + 1,
                          tone_prob=float(_platt(p_acc, self.meta["platt"])))
        return out

    def _pitch(self, feats, wins, ref, out):
        idx = [i for i, s in enumerate(ref.syllables) if s.pitch_high is not None]
        if not idx:
            return out
        if self.meta.get("sequence"):     # utterance-level model: every mora, in order
            z = self.sess.run(None, {"x": self._ctx(feats, wins, range(len(ref.syllables)))})[0][idx, 0]
        else:
            z = self.sess.run(None, {"x": self._ctx(feats, wins, idx)})[0][:, 0]
        ph = 1 / (1 + np.exp(-z))
        for k, i in enumerate(idx):
            p = ph[k] if ref.syllables[i].pitch_high else 1 - ph[k]
            out[i].update(pitch_high_prob=float(ph[k]), pitch_prob=float(_platt(p, self.meta["platt"])))
        return out

    def _stress(self, feats, wins, ref, out):
        groups, X, E = [], [], []
        for w in ref.words:
            s0, s1 = w.syllables
            syl = ref.syllables[s0:s1]
            if s1 - s0 < 2 or not any(s.stress == 1 for s in syl):
                continue
            groups.append((s0, s1))
            X.append(self._ctx(feats, wins, range(s0, s1)))
            E.append(stress_extra(feats, wins, (s0, s1)))
        for i in range(len(ref.syllables)):
            out[i]["stress_prob"] = None
        if not groups:
            return out
        X = np.concatenate(X)
        E = np.concatenate(E).astype(np.float32)
        feeds = {"x": X, "extra": E}
        if self.meta.get("n_lang"):
            feeds["lang"] = np.full(len(X), self.lang_id, dtype=np.int64)
        z = self.sess.run(None, feeds)[0][:, 0]
        k = 0
        for s0, s1 in groups:
            zz = z[k:k + s1 - s0]
            k += s1 - s0
            p = np.exp(zz - zz.max())
            p /= p.sum()
            j = next(i for i in range(s0, s1) if ref.syllables[i].stress == 1)
            out[j]["stress_prob"] = float(_platt(p[j - s0], self.platt))
            out[j]["stress_dist"] = [float(v) for v in p]
        return out
