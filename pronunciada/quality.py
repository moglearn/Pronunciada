"""Audio-quality gate: measured from the waveform alone, before scoring. numpy only.

    q = check(wav, n_syllables=12, lang="en")      # wav: float32 16 kHz in [-1, 1]
    q["issues"]             # subset of ISSUES, in that order
    q["retry_recommended"]  # bool(issues)

Measurements (20 ms non-overlapping frames, levels in dB relative to full scale of the frame
RMS):
  clipped_share   share of samples that sit at the clip level (|x| >= 0.99) or in a flat run of
                  >= 3 samples at the recording's own peak (clipped, then rescaled)
  noise_db        10th percentile frame level (the quiet floor)
  speech_db       mean level of the speech frames
  snr_db          p95 - p10 of the frame levels (loud speech over the quiet floor; an SNR
                  estimate that needs no voice-activity decision; white noise at 0 / 5 / 10 dB
                  SNR reads about 5 / 9 / 14 dB)
  speech_s        frames above the speech threshold (max(p95 - 30 dB, noise + 35 % of the
                  p95 - noise range)), with gaps < 100 ms bridged
  speech_share    speech_s / duration
  lead_sil_s / trail_sil_s   silence before the first / after the last speech frame
  expected_s      n_syllables x the language's median speech seconds per syllable (FLEURS
                  validation), None without a prompt
  speech_ratio    speech_s / expected_s
Thresholds (THRESHOLDS) were set on FLEURS validation (normal audio) and synthetic degradations
of it, checked on Common Voice dev clips (bench/coach/audio_gate.py, results/coaching.md).
"""
from __future__ import annotations

import numpy as np

SR = 16000
FRAME = 320                      # 20 ms
ISSUES = ("clipped", "too_noisy", "too_quiet", "too_short", "mostly_silence")

# median speech seconds per prompt syllable (Japanese: per mora) on FLEURS validation
SEC_PER_SYL = {"de": 0.183, "en": 0.192, "es": 0.154, "fr": 0.171, "it": 0.173, "ja": 0.122, "ko": 0.167, "pt": 0.167, "ru": 0.158, "zh": 0.185}

THRESHOLDS = {
    "clipped_share": 0.002,      # clipped:        share of clipped samples above
    "snr_db": 12.1,             # too_noisy:      SNR estimate (p95 - p10, dB) below
    "speech_db": -69.1,          # too_quiet:      speech level below (dBFS)
    "speech_ratio": 0.716,       # too_short:      speech shorter than this share of what the prompt needs
    "speech_share": 0.312,       # mostly_silence: speech share of the recording below ...
    "min_speech_s": 0.3,         # ... or less than this much speech at all
}


def _frame_db(wav: np.ndarray) -> np.ndarray:
    n = len(wav) // FRAME
    if n == 0:
        return np.zeros(0)
    x = wav[: n * FRAME].astype(np.float64).reshape(n, FRAME)
    x = x - x.mean(1, keepdims=True)
    return 10.0 * np.log10((x * x).mean(1) + 1e-12)


def _clipped_share(wav: np.ndarray) -> float:
    a = np.abs(wav)
    if len(a) == 0:
        return 0.0
    at_rail = a >= 0.99
    peak = float(a.max())
    flat = np.zeros_like(at_rail)
    if peak >= 0.1:
        near = a >= peak * 0.999
        # runs of >= 3 consecutive samples at the recording's own peak
        r = near[2:] & near[1:-1] & near[:-2]
        flat[2:] |= r
        flat[1:-1] |= r
        flat[:-2] |= r
    return float(np.mean(at_rail | flat))


def _bridge(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False runs shorter than max_gap frames lying between True frames."""
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return mask
    out = mask.copy()
    gaps = np.diff(idx)
    for k in np.flatnonzero((gaps > 1) & (gaps <= max_gap + 1)):
        out[idx[k] + 1: idx[k + 1]] = True
    return out


def measure(wav, n_syllables: int | None = None, lang: str | None = None) -> dict:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    dur = len(wav) / SR
    db = _frame_db(wav)
    out = {"duration_s": round(dur, 3), "clipped_share": _clipped_share(wav),
           "peak_dbfs": float(20 * np.log10(max(float(np.abs(wav).max()) if len(wav) else 0.0, 1e-9)))}
    if len(db) == 0:
        out.update(noise_db=None, speech_db=None, snr_db=None, speech_s=0.0, speech_share=0.0,
                   lead_sil_s=dur, trail_sil_s=0.0, expected_s=None, speech_ratio=None)
        return out
    p10, p95 = float(np.percentile(db, 10)), float(np.percentile(db, 95))
    thr = max(p95 - 30.0, p10 + 0.35 * (p95 - p10))
    sp = _bridge(db >= thr, 5)
    if p95 < -90:                                   # digital silence
        sp[:] = False
    n_sp = int(sp.sum())
    speech_db = float(db[sp].mean()) if n_sp else None
    snr = (p95 - p10) if n_sp else None
    idx = np.flatnonzero(sp)
    out.update(noise_db=p10, speech_db=speech_db, snr_db=snr, speech_s=n_sp * FRAME / SR,
               speech_share=n_sp / len(db),
               lead_sil_s=float(idx[0] * FRAME / SR) if n_sp else dur,
               trail_sil_s=float(dur - (idx[-1] + 1) * FRAME / SR) if n_sp else 0.0)
    exp = None
    if n_syllables:
        exp = n_syllables * SEC_PER_SYL.get(lang or "", 0.2)
    out["expected_s"] = exp
    out["speech_ratio"] = (out["speech_s"] / exp) if exp else None
    for k, v in list(out.items()):
        if isinstance(v, float):
            out[k] = round(v, 4)
    return out


def issues(m: dict, thresholds: dict | None = None) -> list[str]:
    t = {**THRESHOLDS, **(thresholds or {})}
    out = []
    if m["clipped_share"] > t["clipped_share"]:
        out.append("clipped")
    if m["snr_db"] is not None and m["snr_db"] < t["snr_db"]:
        out.append("too_noisy")
    if m["speech_db"] is not None and m["speech_db"] < t["speech_db"]:
        out.append("too_quiet")
    if m["speech_ratio"] is not None and m["speech_ratio"] < t["speech_ratio"]:
        out.append("too_short")
    if m["speech_share"] < t["speech_share"] or m["speech_s"] < t["min_speech_s"]:
        out.append("mostly_silence")
    return out


def check(wav, n_syllables: int | None = None, lang: str | None = None,
          thresholds: dict | None = None) -> dict:
    """{"schema_version", measured values, "issues": [...], "retry_recommended": bool}."""
    m = measure(wav, n_syllables, lang)
    iss = issues(m, thresholds)
    return {"schema_version": 1, **m, "issues": iss, "retry_recommended": bool(iss)}

