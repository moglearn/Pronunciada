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
                 delivery: bool | None = None,
                 audio_check: bool = True, length: bool = True):
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
        # reported 0-10 word score (round 6): per-language choice in config "word_score_model";
        # reads the word-pooled ZIPA stack-1 activations. The sentence level keeps reading the
        # round-2 word regression (word_reg.npz), so sentence inputs do not change.
        ws_name = cfg.get("word_score_model", {}).get(self.lang)
        from pathlib import Path as _Pw

        _dw = _Pw(calib_dir) if calib_dir else C.data_dir()
        self.word_score = C.load_level(ws_name, calib_dir) if ws_name and (_dw / f"{ws_name}.npz").exists() else None
        self.word_score_proj = C.load_projection(ws_name, calib_dir) if self.word_score is not None else None
        uses = uses or self.word_score is not None
        self.delivery = uses if delivery is None else bool(delivery)
        self._delivery_forced = delivery is True        # training extraction: every feature
        self._dcols = [c for c in self.sentence.num if c in dcols]
        want_emb = self.delivery and (self.sentence_proj is not None or self.word_score is not None
                                      or delivery is True)
        if prosody:                              # a selected prosody model may read ZIPA activations
            from .prosody import needs_embeddings

            want_emb = want_emb or needs_embeddings(self.lang)
        self.am = AcousticModel(model, precision, threads, embeddings=want_emb)
        self._word_layers = tuple(sorted(set(
            (["enc1"] if delivery is True else []) + list((self.word_score_proj or {}).get("layers", [])))))
        self.universal_name = cfg["universal_sound_model"]
        name, bias = C.sound_model_for(self.lang, calib_dir)
        if sound_model is not None:
            name = sound_model
            bias = float(cfg.get("bias", {}).get(self.lang, 0.0)) if name == self.universal_name else 0.0
        self.sound_name, self.sound_bias = name, bias
        self.universal_bias = float(cfg.get("bias", {}).get(self.lang, 0.0))
        self.sound = C.load_level(name, calib_dir)
        # round 12b: a "detail" sound model (config "sound_detail_model", English: sound_E, which
        # also reads per-sound ZIPA-layer inputs, pronunciada/sound_emb.py) gives the reported 0-2
        # phone score and the coaching inputs; the probabilities that feed words / sentences and
        # PhoneResult.prob stay on self.sound.
        from . import sound_emb as _SE

        dname = cfg.get("sound_detail_model", {}).get(self.lang) if sound_model is None else None
        self.sound_detail = C.load_level(dname, calib_dir) if dname else None
        self.sound_aux = _SE.load(dname or name, calib_dir)
        self.universal = C.load_level(self.universal_name, calib_dir)
        self.word = C.load_level("word", calib_dir)
        from pathlib import Path as _P

        _d = _P(calib_dir) if calib_dir else C.data_dir()
        # round 2: the word-accuracy regression is a separate model that also reads the word log-odds
        self.word_reg = C.load_level("word_reg", calib_dir) if (_d / "word_reg.npz").exists() else None
        # prosody channel (round 3): per-language word-level calibrator of the tone / stress
        # evidence, pronunciada/data/calib/prosody/<lang>.json (bench/calib/fit_prosody.py)
        pp = _d / "prosody" / f"{self.lang}.json"
        from .prosody import default_model, selected_model

        if selected_model(self.lang) != default_model(self.lang):   # its own channel calibrator
            pm = _d / "prosody" / f"{self.lang}.{selected_model(self.lang)}.json"
            pp = pm if pm.exists() else pp
        self.pros_cal = json.loads(pp.read_text()) if pp.exists() else None
        # omission channel (round 4): P(word was said at all), pooled over languages
        # (bench/calib/fit_omission.py); replaces the hard llr_del rule for WordResult.omitted
        # round 7: Mandarin sentence accuracy / prosodic scores recalibrated on OMPAL DEV learners
        # (bench/calib/fit_sentence_zh.py); other languages and other scores unchanged
        pr7 = _d / f"sentence_{self.lang}_recal.json"
        self.sent_recal = json.loads(pr7.read_text()) if pr7.exists() else None
        if self.sent_recal and self.sent_recal.get("tone_model"):
            from .prosody import selected_tone_model

            if self.sent_recal["tone_model"] != selected_tone_model():
                self.sent_recal = None        # fitted on another tone model's outputs
        po = _d / "omission.json"
        self.omit_cal = json.loads(po.read_text()) if po.exists() else None
        # explain-away second pass (round 3): config "explain_away" {threshold, max_passes, enabled}
        ea = {"threshold": 0.3, "max_passes": 2, **cfg.get("explain_away", {})}
        if explain_away is not None:
            ea["enabled"] = bool(explain_away)
        self.ea = ea if ea.get("enabled") else None
        self._calib_dir = calib_dir
        # length channel (round 15): long / short sounds judged from alignment timing
        # (pronunciada/length.py, pronunciada/data/length/<lang>.json); None = not shipped / off
        from . import length as _L

        self.length_cal = _L.params(self.lang) if length else None
        from . import contrast_word as _CW                 # round 18: word-level contrast factor

        self.contrast_word = _CW.params(self.lang)
        self.audio_check = bool(audio_check)     # pronunciada.quality gate -> SentenceResult.audio
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
        aq = None
        if self.audio_check:
            from . import quality as Q

            t = time.perf_counter()
            aq = Q.check(wav, len(ref.syllables), self.lang)
            timing["audio_check"] = (time.perf_counter() - t) * 1e3
        lpa, fb, emb = self._posteriors(wav, timing)
        t = time.perf_counter()
        ev = score_reference(lpa, ref)
        timing["align"] = (time.perf_counter() - t) * 1e3
        if not np.isfinite(ev.L_can) or ev.L_can <= -1e5:
            warnings.append("the text is too long for the audio (the expected sounds cannot all fit)")
        t = time.perf_counter()
        tab = F.phone_table(lpa, ref, ev, self.native, dur)
        if self.sound_aux is not None:
            self.sound_aux.augment(tab, ref, ev, emb)
        phone_vec = None
        if self.lang == "zh" and emb:            # round 12b: Mandarin consonant / vowel coaching
            from . import coach_zh as _CZ
            from .sound_emb import phone_vectors as _pv

            if _CZ.params() is not None:
                phone_vec = _pv(emb, ev, len(ref.phones))
        contrast_ev = None
        if emb:                                  # round 17: head-based contrast evidence (ko / ru / zh)
            from . import contrast_heads as _CH

            if _CH.entries(self.lang):
                from .sound_emb import phone_vectors as _pv

                V17 = phone_vec if phone_vec is not None else _pv(emb, ev, len(ref.phones))
                contrast_ev = _CH.evidence(self.lang, [p.ipa for p in ref.phones], V17,
                                           tense=[getattr(p, "tense", None) for p in ref.phones])
                capz = _CH.cap_column(self.lang, contrast_ev)
                if capz is not None:
                    tab["head_cap_z"] = capz
        timing["features"] = (time.perf_counter() - t) * 1e3
        trk = dv = pooled = wpooled = spans = None
        if self.delivery:
            from . import delivery as D
            from .prosody.f0 import track

            t = time.perf_counter()
            # the pitch track is shared with the prosody channel; skipped when neither needs it
            need_pitch = self.prosody is not None or any(k in D.PITCH for k in self._dcols) \
                or self._delivery_forced
            trk = track(wav) if need_pitch else None
            dv = D.features(wav, fb, lpa, ref, ev, trk, pitch=need_pitch)
            pooled = D.pool_embeddings(emb, ev)
            dv.update(D.project(pooled, self.sentence_proj))
            if emb is not None:      # word-level pooled activations (round 6 word head / training)
                spans = D.word_frame_spans(ref, ev, len(lpa))
                wpooled = {k: D.pool_words(emb[k], spans) for k in self._word_layers}
            else:
                spans = wpooled = None
            timing["delivery"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        cache = self._calibrate(tab, ref, dur, dv, wpooled)
        cache.update(delivery=dv, pooled=pooled, word_pooled=wpooled, word_spans=spans)
        timing["calibrate"] = (time.perf_counter() - t) * 1e3
        cache["phone_vec"] = phone_vec
        cache["emb"] = emb              # round 4 coaching: attribute heads read the activations (no score uses it)
        cache["contrast_ev"] = contrast_ev  # round 17: per-contrast head evidence (coaching)
        cache.update(ref=ref, ev=ev, tab=tab)
        cache.update(lpa=lpa, audio=aq)          # read by pronunciada.coach; no score uses them
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
                if trk is None:     # the same track ProsodyScorer.score would compute; kept for coaching
                    from .prosody.f0 import track as _f0_track

                    trk = _f0_track(np.asarray(wav, dtype=np.float32))
                pros = self.prosody.score(wav, ref, ev, track=trk, embeddings=emb)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"prosody failed: {type(e).__name__}: {e}")
            timing["prosody"] = (time.perf_counter() - t) * 1e3
        lres = None
        if self.length_cal is not None and np.isfinite(ev.L_can) and ev.L_can > -1e5:
            from . import length as _L

            t = time.perf_counter()
            hs = None
            if emb and self.length_cal.get("head"):        # round 17: attribute-head length evidence
                from . import contrast_heads as _CH
                from .sound_emb import phone_vectors as _pv

                hs = _CH.length_evidence(self.lang, self.length_cal, [p.ipa for p in ref.phones],
                                         _pv(emb, ev, len(ref.phones)))
            lres = _L.score(ref, _L.frame_inputs(ref, ev, lpa, fb), self.length_cal, head_s=hs)
            timing["length"] = (time.perf_counter() - t) * 1e3
        cache.update(prosody=pros, dur=dur, warnings=warnings, f0=trk, wav=wav,   # f0 / wav: coaching only
                     length=lres)
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
            for k in ("emb_proj", "proto_cos", "proto_other", "proto_margin", "proto_has"):
                if k in c["tab"]:                 # round 12b inputs: kept from the first pass
                    tab2[k] = c["tab"][k]
            c2 = self._calibrate(tab2, ref, dur, c.get("delivery"), c.get("word_pooled"))
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
    def _calibrate(self, tab: dict, ref, dur: float, dv: dict | None = None,
                   wpooled: dict | None = None) -> dict:
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
        detail = {}
        if self.sound_detail is not None:
            d = self.sound_detail.logits(tab)
            sreg = d["reg"]
            detail = {"coach_zl": d["z_len"], "coach_zs": np.minimum(d["z_str"], d["z_len"])}
        caps = C.contrast_caps(self.lang, self._calib_dir) if hasattr(self, "_calib_dir") else []
        if caps:                       # round 16: contrast caps on the SHOWN sound log-odds only;
            detail.setdefault("coach_zl", zl)   # coaching and word / sentence inputs unchanged
            detail.setdefault("coach_zs", np.minimum(zs, zl))
            zl, zs = C.apply_contrast_caps(caps, tab, zl, zs)
        cz = C.contrast_cap_z(caps, tab) if caps else np.full(len(zl), np.nan)   # round 18: word factor input
        if "head_cap_z" in tab:
            hz0 = np.asarray(tab["head_cap_z"], np.float64)
            cz = np.where(np.isfinite(cz) & np.isfinite(hz0), np.minimum(cz, hz0), np.where(np.isfinite(cz), cz, hz0))
        if "head_cap_z" in tab:         # round 17: head-based contrast caps, composed by the minimum
            detail.setdefault("coach_zl", zl)
            detail.setdefault("coach_zs", np.minimum(zs, zl))
            hz = np.asarray(tab["head_cap_z"], np.float64)
            fire = np.isfinite(hz)
            zl, zs = np.array(zl, np.float64), np.array(zs, np.float64)
            zl[fire] = np.minimum(zl[fire], hz[fire])
            zs[fire] = np.minimum(zs[fire], hz[fire])
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
        w_rep = w["reg"]
        pj = self.word_score_proj
        if self.word_score is not None and wpooled and all(k in wpooled for k in pj["layers"]):
            P = np.concatenate([wpooled[k] for k in pj["layers"]], axis=1).astype(np.float64)
            z = ((P - pj["mu"]) / pj["sd"]) @ pj["W"] + pj["b"]
            for j, nm in enumerate(pj["names"]):
                W[nm] = z[:, j]
            XR2 = np.column_stack([np.asarray(W[c], dtype=np.float64) for c in self.word_score.num])
            w_rep = self.word_score.predict(X=XR2)["reg"]
        return {**detail, "s_zl": zl, "s_zs": np.minimum(zs, zl), "s_reg": sreg, "W": W, "w_zl": w["z_len"],
                "w_zs": w["z_str"], "w_reg": w_rep, "w_reg_r2": w["reg"], "S": S, "sent": so,
                "sent_zl": so["z_len"][0], "sent_zs": so["z_str"][0], "contrast_z": cz}

    # ------------------------------------------------------------------ result objects
    def _assemble(self, c: dict, strictness: float) -> SentenceResult:
        s = C.clip_strictness(strictness)
        ref, ev, tab = c["ref"], c["ev"], c["tab"]
        ps = C.sigmoid(C.blend_logit(c["s_zl"], c["s_zs"], s))
        pw = C.sigmoid(C.blend_logit(c["w_zl"], np.minimum(c["w_zs"], c["w_zl"]), s))
        pr = c.get("prosody") or [dict() for _ in ref.syllables]
        lres = c.get("length")
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
            if lres is not None and lres["prob"][i] is not None:
                phones[-1].length_expected = lres["expected"][i]
                phones[-1].length_prob = float(lres["prob"][i])
        words = []
        omitted = c["W"]["omitted_f"].astype(bool)
        for wi, w in enumerate(ref.words):
            ph = phones[w.phones[0]:w.phones[1]]
            spans = [(x.start, x.end) for x in ph if x.end > x.start]
            st = min(a for a, _ in spans) if spans else (ph[0].start if ph else 0.0)
            en = max(b for _, b in spans) if spans else st
            syls = []
            tone_p, stress_p = [], []
            for si in range(*w.syllables):
                sy = ref.syllables[si]
                d = pr[si] if si < len(pr) else {}
                a, b = sy.phones
                sp = [(x.start, x.end) for x in phones[a:b] if x.end > x.start]
                syls.append(SyllableResult(
                    sy.text, min(x for x, _ in sp) if sp else st, max(y for _, y in sp) if sp else st,
                    tone_expected=sy.tone, tone_heard=d.get("tone_heard"), tone_prob=d.get("tone_prob"),
                    stress_expected=sy.stress, stress_prob=d.get("stress_prob")))
                for lst, k in ((tone_p, "tone_prob"), (stress_p, "stress_prob")):
                    if d.get(k) is not None:
                        lst.append(float(d[k]))
            sounds = float(pw[wi])
            # Prosody channel (round 3): P_pros(word) from a word-level calibrator of the lowest /
            # second-lowest tone / stress logit over the word's scored syllables
            # (bench/calib/fit_prosody.py, FLEURS validation); WordResult.prob = sounds x P_pros.
            # Without a fitted channel the raw syllable minimum is reported and prob = sounds.
            raw = {"tone": tone_p, "stress": stress_p}
            p_pros = None
            if self.pros_cal and raw.get(self.pros_cal["kind"]):
                p_pros = _pros_prob(self.pros_cal, raw[self.pros_cal["kind"]], s)
            tone = min(tone_p) if tone_p else None
            stress = min(stress_p) if stress_p else None
            if p_pros is not None:
                if self.pros_cal["kind"] == "stress":
                    stress = p_pros
                else:
                    tone = p_pros
            prob = sounds * (p_pros if p_pros is not None else 1.0)
            # length channel (round 15): a factor that stays 1 unless the word's lowest length
            # logit is below a low native quantile (pronunciada.length.word_factor,
            # bench/length/fit_length.py)
            p_len = None
            if lres is not None:
                lv = [x.length_prob for x in ph if x.length_prob is not None]
                if lv:
                    from . import length as _L

                    p_len = _L.word_factor(self.length_cal, lv, s)
                    prob *= p_len
            p_con = None                 # round 18: word-level contrast factor (pronunciada.contrast_word)
            if self.contrast_word is not None and c.get("contrast_z") is not None:
                from . import contrast_word as _CW

                p_con = _CW.word_factor(self.contrast_word, c["contrast_z"][w.phones[0]:w.phones[1]], s)
                if p_con is not None:
                    prob *= p_con
            if self.omit_cal:
                present = _present_prob(self.omit_cal, tab["llr_del"][w.phones[0]:w.phones[1]])
                prob, om = prob * present, present < 0.5
            else:
                present, om = None, bool(omitted[wi])
            words.append(WordResult(w.text, prob, st, en, ph, syls, sounds_prob=sounds,
                                    stress_prob=stress, tone_prob=tone,
                                    present_prob=present, omitted=om, length_prob=p_len, contrast_prob=p_con))
        zl, zs = c["sent_zl"], min(c["sent_zs"], c["sent_zl"])
        sp = float(C.sigmoid(C.blend_logit(zl, zs, s)))
        flu = float(np.clip(c["sent"]["reg_fluency"][0] / 10.0, 0.0, 1.0))
        comp = (1.0 - float(np.mean([w.omitted for w in words]))) if (self.omit_cal and words) \
            else float(c["S"]["completeness"][0])
        scores = {k: float(np.clip(c["sent"][f"reg_{k}"][0] / 10.0, 0.0, 1.0))
                  for k in ("accuracy", "fluency", "prosodic", "total")}
        if self.sent_recal:
            scores = _sentence_recal(self.sent_recal, scores, [sy.tone_prob for w in words for sy in w.syllables])
        res = SentenceResult(self.lang, ref.text, sp, s, words, fluency=flu, completeness=comp,
                             warnings=list(c["warnings"]), scores=scores)
        res.audio = c.get("audio")
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


def _sentence_recal(cal: dict, scores: dict, tone_probs) -> dict:
    """Round 7 (zh): score_k = clip((ridge_k(features) - 1) / 4, 0, 1) for the recalibrated k;
    features = the four sentence heads + mean / mean-log / share-below-0.5 of syllable tone_prob
    (bench/calib/fit_sentence_zh.features)."""
    t = np.asarray([x for x in tone_probs if x is not None] or [1.0], np.float64)
    f = np.array([scores[k] for k in ("accuracy", "fluency", "prosodic", "total")]
                 + [t.mean(), np.log(np.clip(t, 1e-3, 1)).mean(), (t < 0.5).mean()], np.float64)
    out = dict(scores)
    for k, m in cal["scores"].items():
        c = np.asarray(m["coef"], np.float64)
        out[k] = float(np.clip((c[0] + c[1:] @ f - 1) / 4, 0.0, 1.0))
    return out


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
