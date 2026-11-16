"""Public scoring API.

    scorer = pronunciada.Scorer("en")                    # loads ZIPA, adapter, calibrators once
    res = scorer.score("clip.wav", "The quick brown fox")      # -> SentenceResult
    res2 = scorer.rescore(res, strictness=1.0)       # no model re-run
    pronunciada.score(audio, text, "en")                 # convenience, caches Scorer objects

Pipeline: audio -> fbank -> ZIPA (ONNX) -> per-language output adapter -> text front-end ->
lattice evidence (pronunciada.align.score_reference) -> features (pronunciada.features) -> calibrators
(pronunciada.calibrate: sound -> word -> sentence) -> optional tone / stress (pronunciada.prosody) ->
SentenceResult. The sound probabilities shown come from the language's shipped sound model
(English: the English-specific model if it won on CV, else the universal one; every other
language: the universal model + its logit bias); word and sentence calibrators always consume
the universal sound model's log-odds, which is what they were trained on.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict

import numpy as np

from . import calibrate as C
from . import features as F
from .types import PhoneResult, Reference, SentenceResult, SyllableResult, WordResult

MIN_AUDIO_S = 0.25
SILENCE_PEAK = 1e-4


class Scorer:
    def __init__(self, lang: str, model: str = "zipa-small-crctc-500k", precision: str = "fp32",
                 strictness: float = 0.5, threads: int | None = None, *,
                 calib_dir: str | None = None, sound_model: str | None = None,
                 prosody: bool = True, cache_size: int = 4, explain_away: bool | None = None,
                 experimental: bool = False, delivery: bool | None = None):
        from .acoustic import AcousticModel
        from .adapter import Adapter
        from .text import get_g2p, normalize_lang
        from .variants import NativeStats

        self.lang = normalize_lang(lang)
        self.strictness = float(strictness)
        self.g2p = get_g2p(self.lang)
        self.adapter = Adapter.load(self.lang)
        self.native = NativeStats.load(self.lang)
        cfg = C.config(calib_dir)
        # sentence calibrator (round 5): per-language choice in config "sentence_model"
        # ({lang: packaged name}, default "sentence"); round-5 models also read the delivery
        # features of pronunciada.delivery (+ a projection of pooled encoder activations)
        self.sentence_name = cfg.get("sentence_model", {}).get(self.lang, "sentence")
        self.sentence = C.load_level(self.sentence_name, calib_dir)
        self.sentence_proj = C.load_projection(self.sentence_name, calib_dir)
        self.sentence_lin = C.load_linear(self.sentence_name, calib_dir)
        from . import delivery as D

        dcols = set(D.ALL) | set((self.sentence_proj or {}).get("names", []))
        uses = any(c in dcols for c in self.sentence.num)
        self.delivery = uses if delivery is None else bool(delivery)
        self._dcols = [c for c in self.sentence.num if c in dcols]
        want_emb = self.delivery and (self.sentence_proj is not None or delivery is True)
        self.am = AcousticModel(model, precision, threads, embeddings=want_emb)
        self.universal_name = cfg["universal_sound_model"]
        name, bias = C.sound_model_for(self.lang, calib_dir)
        if sound_model is not None:
            name = sound_model
            bias = float(cfg.get("bias", {}).get(self.lang, 0.0)) if name == self.universal_name else 0.0
        self.sound_name, self.sound_bias = name, bias
        self.universal_bias = float(cfg.get("bias", {}).get(self.lang, 0.0))
        self.sound = C.load_level(name, calib_dir)
        self.universal = C.load_level(self.universal_name, calib_dir)
        self.word = C.load_level("word", calib_dir)
        from pathlib import Path as _P

        _d = _P(calib_dir) if calib_dir else C.data_dir()
        # round 2: the word-accuracy regression is a separate model that also reads the word log-odds
        self.word_reg = C.load_level("word_reg", calib_dir) if (_d / "word_reg.npz").exists() else None
        # prosody channel (round 3): per-language word-level calibrator of the tone / stress /
        # pitch evidence, pronunciada/data/calib/prosody/<lang>.json (bench/calib/fit_prosody.py)
        pp = _d / "prosody" / f"{self.lang}.json"
        self.pros_cal = json.loads(pp.read_text()) if pp.exists() else None
        # A channel marked experimental (Japanese pitch accent: it costs more on correct speech
        # than it catches) only enters WordResult.prob with Scorer(..., experimental=True); the
        # per-syllable pitch_prob is reported either way.
        if self.pros_cal and self.pros_cal.get("experimental") and not experimental:
            self.pros_cal = None
        # omission channel (round 4): P(word was said at all), pooled over languages
        # (bench/calib/fit_omission.py); replaces the hard llr_del rule for WordResult.omitted
        po = _d / "omission.json"
        self.omit_cal = json.loads(po.read_text()) if po.exists() else None
        # explain-away second pass (round 3): config "explain_away" {threshold, max_passes, enabled}
        ea = {"threshold": 0.3, "max_passes": 2, **cfg.get("explain_away", {})}
        if explain_away is not None:
            ea["enabled"] = bool(explain_away)
        self.ea = ea if ea.get("enabled") else None
        self._calib_dir = calib_dir
        self.prosody = None
        self.prosody_error = None
        if prosody:
            try:
                from .prosody import ProsodyScorer

                ps = ProsodyScorer(self.lang, threads=min(2, threads or 2))
                self.prosody = ps if getattr(ps, "sess", None) is not None else None
            except Exception as e:  # noqa: BLE001  (models are built concurrently; optional)
                self.prosody_error = f"{type(e).__name__}: {e}"
        self._lp_cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size

    # ------------------------------------------------------------------ acoustic stage
    def _posteriors(self, wav: np.ndarray, timing: dict):
        """(adapted log-posteriors, fbank, encoder activations or None), cached per waveform."""
        key = hashlib.sha1(wav.tobytes()).hexdigest()
        if key in self._lp_cache:
            self._lp_cache.move_to_end(key)
            timing.update(fbank=0.0, model=0.0, adapter=0.0)
            return self._lp_cache[key]
        t = time.perf_counter()
        x = self.am._features(wav)
        timing["fbank"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        emb = None
        if self.am.embeddings:
            lp, emb = self.am.posteriors_from_features(x, return_embeddings=True)
        else:
            lp = self.am.posteriors_from_features(x)
        timing["model"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        lpa = self.adapter(lp) if self.adapter is not None else lp
        timing["adapter"] = (time.perf_counter() - t) * 1e3
        self._lp_cache[key] = (lpa, x, emb)
        while len(self._lp_cache) > self._cache_size:
            self._lp_cache.popitem(last=False)
        return self._lp_cache[key]

    # ------------------------------------------------------------------ public
    def score(self, audio, text, strictness: float | None = None) -> SentenceResult:
        """audio: path | bytes | (np.ndarray, sample_rate) | float32 16 kHz np.ndarray;
        text: the prompt (str) or a ready pronunciada.types.Reference."""
        from .align import score_reference
        from .audio import SAMPLE_RATE, load_audio

        t_all = time.perf_counter()
        timing: dict = {}
        warnings: list[str] = []
        if isinstance(text, Reference):
            ref = text
            t = time.perf_counter()
        else:
            if text is None or not str(text).strip():
                raise ValueError("text is empty")
            t = time.perf_counter()
            ref = self.g2p(str(text))
        ref.validate()
        if not ref.phones:
            raise ValueError(f"no scorable sounds in {getattr(ref, 'text', text)!r}")
        timing["g2p"] = (time.perf_counter() - t) * 1e3
        warnings += list(ref.warnings)
        t = time.perf_counter()
        wav = load_audio(audio)
        timing["load"] = (time.perf_counter() - t) * 1e3
        dur = len(wav) / SAMPLE_RATE
        if dur < MIN_AUDIO_S:
            warnings.append(f"audio is very short ({dur:.2f} s)")
        if len(wav) == 0 or float(np.max(np.abs(wav))) < SILENCE_PEAK:
            warnings.append("audio is silent")
        lpa, fb, emb = self._posteriors(wav, timing)
        t = time.perf_counter()
        ev = score_reference(lpa, ref)
        timing["align"] = (time.perf_counter() - t) * 1e3
        if not np.isfinite(ev.L_can) or ev.L_can <= -1e5:
            warnings.append("the text is too long for the audio (the expected sounds cannot all fit)")
        t = time.perf_counter()
        tab = F.phone_table(lpa, ref, ev, self.native, dur)
        timing["features"] = (time.perf_counter() - t) * 1e3
        trk = dv = pooled = None
        if self.delivery:
            from . import delivery as D
            from .prosody.f0 import track

            t = time.perf_counter()
            # the pitch track is shared with the prosody channel; skipped when neither needs it
            need_pitch = self.prosody is not None or any(k in D.PITCH for k in self._dcols) \
                or not self._dcols
            trk = track(wav) if need_pitch else None
            dv = D.features(wav, fb, lpa, ref, ev, trk, pitch=need_pitch)
            pooled = D.pool_embeddings(emb, ev)
            dv.update(D.project(pooled, self.sentence_proj))
            timing["delivery"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        cache = self._calibrate(tab, ref, dur, dv)
        cache.update(delivery=dv, pooled=pooled)
        timing["calibrate"] = (time.perf_counter() - t) * 1e3
        cache.update(ref=ref, ev=ev, tab=tab)
        if not np.isfinite(ev.L_can) or ev.L_can <= -1e5:
            # the expected sounds cannot all fit into the audio: nothing was said in a scorable
            # way, so every level gets the floor probability (the calibrators never saw the
            # llr = -1000 sentinel values and would extrapolate arbitrarily)
            lo = float(C.logit(C.P_CLIP))
            for k in ("s_zl", "s_zs", "w_zl", "w_zs"):
                cache[k] = np.full_like(np.asarray(cache[k], dtype=np.float64), lo)
            cache["sent_zl"] = cache["sent_zs"] = lo
            cache["S"] = dict(cache["S"], completeness=np.zeros(1))
        elif self.ea is not None:
            t = time.perf_counter()
            cache = self.explain_away(lpa, cache, dur, self.ea["threshold"], self.ea["max_passes"])
            timing["explain_away"] = (time.perf_counter() - t) * 1e3
        pros = None
        if self.prosody is not None:
            t = time.perf_counter()
            try:
                pros = self.prosody.score(wav, ref, ev, track=trk)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"prosody failed: {type(e).__name__}: {e}")
            timing["prosody"] = (time.perf_counter() - t) * 1e3
        if self.pros_cal and self.pros_cal.get("experimental"):
            warnings.append(f"{self.pros_cal['kind']} scoring is experimental: {self.pros_cal['experimental']}")
        cache.update(prosody=pros, dur=dur, warnings=warnings)
        res = self._assemble(cache, self.strictness if strictness is None else strictness)
        timing["total"] = (time.perf_counter() - t_all) * 1e3
        res.timing_ms = {k: round(v, 3) for k, v in timing.items()}
        return res

    def rescore(self, result: SentenceResult, strictness: float) -> SentenceResult:
        """Same audio and text at another strictness, from the evidence cached on `result`
        (no acoustic model, alignment or calibrator run)."""
        cache = getattr(result, "_pronunciada", None)
        if cache is None:
            raise ValueError("result has no cached evidence (was it produced by Scorer.score?)")
        t = time.perf_counter()
        res = self._assemble(cache, strictness)
        res.timing_ms = {"rescore": round((time.perf_counter() - t) * 1e3, 3)}
        return res

    def details(self, result: SentenceResult) -> dict:
        """Human-scale scores behind a result: per phone the 0-2 regression and both logits,
        per word the 0-10 accuracy regression, per sentence accuracy / fluency / prosodic /
        total on 0-10 (the speechocean762 scales)."""
        c = result._pronunciada  # noqa: SLF001
        if c["s_reg"] is None:
            c["s_reg"] = self.sound.predict(c["tab"], heads=("reg",))["reg"]
        return {"phone_score": c["s_reg"], "phone_z_len": c["s_zl"], "phone_z_str": c["s_zs"],
                "word_accuracy": c["w_reg"], "word_z_len": c["w_zl"], "word_z_str": c["w_zs"],
                "sentence": {k: float(c["sent"][f"reg_{k}"][0]) for k in ("accuracy", "fluency", "prosodic", "total")},
                "sentence_z_len": float(c["sent_zl"]), "sentence_z_str": float(c["sent_zs"]),
                "omitted": c["W"]["omitted_f"].astype(bool)}

    def sound_outputs(self, result: SentenceResult, model) -> dict:
        """Outputs of another sound calibrator (LevelModel or packaged name) on the cached
        features of `result` (benchmarks: feature sets A / B / C side by side)."""
        m = model if isinstance(model, C.LevelModel) else C.load_level(model, self._calib_dir)
        return m.predict(result._pronunciada["tab"])  # noqa: SLF001

    # ------------------------------------------------------------------ explain-away pass
    def explain_away(self, lpa, c: dict, dur: float, threshold: float, max_passes: int) -> dict:
        """Second pass(es): while the lowest not-yet-wildcarded word has sounds-based P < threshold
        (s = 0.5), wildcard its sounds in the base lattice (pronunciada.align wild_slots), re-run the
        evidence (no acoustic model), features and calibrators, at most `max_passes` times. Every
        other word / sound takes the last pass; a wildcarded word keeps the values of the pass in
        which it was selected (the first pass for the first one). The sentence level keeps the
        first pass. Returns the merged cache (c["explained"] = wildcarded word indices)."""
        from .align import score_reference

        ref = c["ref"]
        passes = [c]
        keep: dict[int, int] = {}                  # word -> index of the pass it is taken from
        for _ in range(int(max_passes)):
            cur = passes[-1]
            pw = C.sigmoid(C.blend_logit(cur["w_zl"], np.minimum(cur["w_zs"], cur["w_zl"]), 0.5))
            cand = [k for k in range(len(ref.words)) if k not in keep]
            if not cand:
                break
            wi = min(cand, key=lambda k: pw[k])
            if pw[wi] >= threshold:
                break
            keep[wi] = len(passes) - 1
            slots = [i for k in keep for i in range(*ref.words[k].phones)]
            ev2 = score_reference(lpa, ref, wild_slots=slots)
            tab2 = F.phone_table(lpa, ref, ev2, self.native, dur)
            c2 = self._calibrate(tab2, ref, dur)
            c2.update(ref=ref, ev=ev2, tab=tab2)
            passes.append(c2)
        if not keep:
            c["explained"] = []
            return c
        last = len(passes) - 1
        wsrc = np.array([keep.get(k, last) for k in range(len(ref.words))])
        psrc = np.array([wsrc[p.word] for p in ref.phones])
        out = dict(c)                               # sentence level stays first-pass
        for key in ("s_zl", "s_zs"):
            out[key] = np.array([passes[psrc[i]][key][i] for i in range(len(psrc))])
        out["s_reg"] = None if any(p["s_reg"] is None for p in passes) else \
            np.array([passes[psrc[i]]["s_reg"][i] for i in range(len(psrc))])
        for key in ("w_zl", "w_zs", "w_reg"):
            out[key] = np.array([passes[wsrc[k]][key][k] for k in range(len(wsrc))])
        out["W"] = {k: (np.array([passes[wsrc[j]]["W"][k][j] for j in range(len(wsrc))])
                        if isinstance(v, np.ndarray) and len(v) == len(wsrc) else v)
                    for k, v in c["W"].items()}
        out["tab"] = {k: ([passes[psrc[i]]["tab"][k][i] for i in range(len(psrc))] if isinstance(v, list)
                          else np.array([passes[psrc[i]]["tab"][k][i] for i in range(len(psrc))]))
                      if (isinstance(v, (list, np.ndarray)) and len(v) == len(psrc)) else v
                      for k, v in c["tab"].items()}
        ev = c["ev"]
        # spans too: a wildcarded word's from the pass it was selected in (not the wildcard)
        out["ev"] = type(ev)([passes[psrc[i]]["ev"].slots[i] for i in range(len(psrc))],
                             ev.L_can, ev.n_frames, ev.frame_shift, ev.free_tokens)
        out["explained"] = sorted(keep)
        out["n_passes"] = len(passes)
        out["first_pass"] = c                      # for benchmarks: the result without the pass
        return out

    # ------------------------------------------------------------------ calibration
    def _calibrate(self, tab: dict, ref, dur: float, dv: dict | None = None) -> dict:
        Xu = self.universal.matrix(tab)
        u = self.universal.logits(X=Xu)
        uzl, uzs = u["z_len"] + self.universal_bias, u["z_str"] + self.universal_bias
        ureg = u["reg"]
        if self.sound is self.universal:
            zl, zs, sreg = uzl, uzs, ureg
        else:   # probabilities shown from the language's own model; its 0-2 head on demand
            s = self.sound.logits(tab, heads=("len", "str"))
            zl, zs = s["z_len"] + self.sound_bias, s["z_str"] + self.sound_bias
            sreg = None
        W = F.word_table(tab, uzl, uzs, ureg)
        XW = np.column_stack([np.asarray(W[c], dtype=np.float64) for c in self.word.num])
        w = self.word.logits(X=XW)
        if self.word_reg is not None:
            W["word_z_len"], W["word_z_str"] = w["z_len"], w["z_str"]
            XR = np.column_stack([np.asarray(W[c], dtype=np.float64) for c in self.word_reg.num])
            w["reg"] = self.word_reg.predict(X=XR)["reg"]
        S = F.sentence_table(tab, uzl, uzs, ureg, W, w["z_len"], w["z_str"], w["reg"], dur)
        for k in self._dcols:            # round-5 delivery inputs (NaN when not computed)
            S[k] = np.array([np.nan if dv is None else float(dv.get(k, np.nan))])
        XS = np.column_stack([np.asarray(S[c], dtype=np.float64) for c in self.sentence.num])
        so = C.apply_linear_blend(self.sentence_lin, XS, self.sentence.logits(X=XS))
        return {"s_zl": zl, "s_zs": np.minimum(zs, zl), "s_reg": sreg, "W": W, "w_zl": w["z_len"],
                "w_zs": w["z_str"], "w_reg": w["reg"], "S": S, "sent": so,
                "sent_zl": so["z_len"][0], "sent_zs": so["z_str"][0]}

    # ------------------------------------------------------------------ result objects
    def _assemble(self, c: dict, strictness: float) -> SentenceResult:
        s = C.clip_strictness(strictness)
        ref, ev, tab = c["ref"], c["ev"], c["tab"]
        ps = C.sigmoid(C.blend_logit(c["s_zl"], c["s_zs"], s))
        pw = C.sigmoid(C.blend_logit(c["w_zl"], np.minimum(c["w_zs"], c["w_zl"]), s))
        pr = c.get("prosody") or [dict() for _ in ref.syllables]
        f2t = self.am.frame_to_time
        phones = []
        for i, p in enumerate(ref.phones):
            sl = ev.slots[i]
            st, en = float(f2t(sl.start_frame)), float(f2t(sl.end_frame))
            # what was heard instead: greedy tokens over the sound's span, "" = nothing (missing)
            heard = (sl.heard if sl.n_frames > 0 else "") if ps[i] < 0.5 else None
            feats = {k: _num(tab[k][i]) for k in ("llr_sub", "llr_del", "z_llr_sub", "n_frames", "peak_post")}
            feats.update(p_lenient=float(C.sigmoid(c["s_zl"][i])), p_strict=float(C.sigmoid(c["s_zs"][i])))
            phones.append(PhoneResult(p.ipa, float(ps[i]), round(st, 3), round(en, 3), heard, feats))
        words = []
        omitted = c["W"]["omitted_f"].astype(bool)
        for wi, w in enumerate(ref.words):
            ph = phones[w.phones[0]:w.phones[1]]
            spans = [(x.start, x.end) for x in ph if x.end > x.start]
            st = min(a for a, _ in spans) if spans else (ph[0].start if ph else 0.0)
            en = max(b for _, b in spans) if spans else st
            syls = []
            tone_p, stress_p, pitch_p = [], [], []
            for si in range(*w.syllables):
                sy = ref.syllables[si]
                d = pr[si] if si < len(pr) else {}
                a, b = sy.phones
                sp = [(x.start, x.end) for x in phones[a:b] if x.end > x.start]
                syls.append(SyllableResult(
                    sy.text, min(x for x, _ in sp) if sp else st, max(y for _, y in sp) if sp else st,
                    tone_expected=sy.tone, tone_heard=d.get("tone_heard"), tone_prob=d.get("tone_prob"),
                    stress_expected=sy.stress, stress_prob=d.get("stress_prob"),
                    pitch_prob=d.get("pitch_prob")))
                for lst, k in ((tone_p, "tone_prob"), (stress_p, "stress_prob"), (pitch_p, "pitch_prob")):
                    if d.get(k) is not None:
                        lst.append(float(d[k]))
            sounds = float(pw[wi])
            # Prosody channel (round 3): P_pros(word) from a word-level calibrator of the lowest /
            # second-lowest tone / stress / pitch logit over the word's scored syllables
            # (bench/calib/fit_prosody.py, FLEURS validation); WordResult.prob = sounds x P_pros.
            # Without a fitted channel the raw syllable minimum is reported and prob = sounds.
            raw = {"tone": tone_p, "stress": stress_p, "pitch": pitch_p}
            p_pros = None
            if self.pros_cal and raw.get(self.pros_cal["kind"]):
                p_pros = _pros_prob(self.pros_cal, raw[self.pros_cal["kind"]], s)
            tone = min(tone_p) if tone_p else None
            stress = min(stress_p) if stress_p else None
            pitch = min(pitch_p) if pitch_p else None
            if p_pros is not None:
                if self.pros_cal["kind"] == "stress":
                    stress = p_pros
                elif self.pros_cal["kind"] == "tone":
                    tone = p_pros
                else:
                    pitch = p_pros
            prob = sounds * (p_pros if p_pros is not None else 1.0)
            if self.omit_cal:
                present = _present_prob(self.omit_cal, tab["llr_del"][w.phones[0]:w.phones[1]])
                prob, om = prob * present, present < 0.5
            else:
                present, om = None, bool(omitted[wi])
            words.append(WordResult(w.text, prob, st, en, ph, syls, sounds_prob=sounds,
                                    stress_prob=stress, tone_prob=tone if tone is not None else pitch,
                                    present_prob=present, omitted=om))
        zl, zs = c["sent_zl"], min(c["sent_zs"], c["sent_zl"])
        sp = float(C.sigmoid(C.blend_logit(zl, zs, s)))
        flu = float(np.clip(c["sent"]["reg_fluency"][0] / 10.0, 0.0, 1.0))
        comp = (1.0 - float(np.mean([w.omitted for w in words]))) if (self.omit_cal and words) \
            else float(c["S"]["completeness"][0])
        scores = {k: float(np.clip(c["sent"][f"reg_{k}"][0] / 10.0, 0.0, 1.0))
                  for k in ("accuracy", "fluency", "prosodic", "total")}
        res = SentenceResult(self.lang, ref.text, sp, s, words, fluency=flu, completeness=comp,
                             warnings=list(c["warnings"]), scores=scores)
        res._pronunciada = c  # noqa: SLF001  (cached evidence for rescore / details)
        return res


def _pros_prob(cal: dict, probs: list[float], s: float) -> float:
    """P_pros(word) at strictness s: z = b0 + b1 * min logit + b2 * second-min logit + b3 * n;
    lenient / strict operating points = z + c_len / z + c_str, blended in log-odds."""
    q = np.sort(C.logit(np.asarray(probs, dtype=np.float64)))
    f = [q[0], q[1] if len(q) > 1 else q[0], float(len(q))]
    b = cal["coef"]
    z = b[0] + b[1] * f[0] + b[2] * f[1] + b[3] * f[2]
    zl, zs = z + cal["c_len"], z + cal["c_str"]
    return float(C.sigmoid(C.blend_logit(zl, min(zs, zl), s)))


def _present_prob(cal: dict, llr_del) -> float:
    """P(word was said): logistic on [max llr_del, mean llr_del, one-sound word, two-sound word]."""
    d = np.clip(np.nan_to_num(np.asarray(llr_del, dtype=np.float64), nan=0.0), -cal["clip"], cal["clip"])
    b = cal["coef"]
    z = b[0] + b[1] * d.max() + b[2] * d.mean() + b[3] * (len(d) == 1) + b[4] * (len(d) == 2)
    return float(C.sigmoid(z))


def _num(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) else x


# ----------------------------------------------------------------------------- convenience

_SCORERS: dict = {}


def score(audio, text, lang: str, strictness: float = 0.5, **kw) -> SentenceResult:
    """One-call scoring; Scorer objects are cached per (lang, options)."""
    key = (lang, tuple(sorted(kw.items())))
    if key not in _SCORERS:
        _SCORERS[key] = Scorer(lang, **kw)
    return _SCORERS[key].score(audio, text, strictness=strictness)
