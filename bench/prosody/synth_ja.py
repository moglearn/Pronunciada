"""Round 13: accent-controlled Japanese speech with exact mora-level high/low labels.

    .venv/bin/python -m bench.prosody.synth_ja hts train      # FLEURS-ja TRAIN texts, training pairs
    .venv/bin/python -m bench.prosody.synth_ja hts heldout    # FLEURS-ja VALIDATION texts, unseen settings
    .venv/bin/python -m bench.prosody.synth_ja psola heldout  # FLEURS-ja VALIDATION natural audio, re-pitched
    .venv/bin/python -m bench.prosody.synth_ja psola train    # FLEURS-ja TRAIN natural audio, re-pitched (13b)

System 1 (training, + a held-out setting range): Open JTalk front-end + HTS engine through
pyopenjtalk 0.4.1 (BSD-3) with its bundled voice "Mei" (mei_normal.htsvoice, MMDAgent project,
CC BY 3.0: attribution required, commercial use and derivatives allowed). For every sentence two
versions are synthesised with the same speed / half-tone setting: the dictionary accent, and one
accent phrase (content-word head, >= 2 morae) with its nucleus moved or removed / added
(NJD `acc` of the phrase head edited before make_label). Labels come from the same Japanese
front-end the scorer uses (pronunciada.text.ja), fed the edited NJD, so mora segmentation and
Tokyo-rule high/low are exactly those of the synthesised accent.
  train:   speed U(0.9, 1.1), half-tone U(-3, +3)
  heldout: speed {0.82, 1.22}, half-tone {-6, +5} (outside the training range), validation texts

System 2 (held-out only): natural FLEURS-ja VALIDATION recordings re-pitched with Praat PSOLA
(parselmouth, offline). Per clip one content-word accent phrase gets a new nucleus; its pitch tier
is replaced by a contour through the speaker's local high / low levels following the new high/low
pattern (levels: the median voiced F0 over the phrase and +-3 morae, +2.5 / -2 semitones); the "correct" twin is the same recording PSOLA-resynthesised with its own contour, so
both carry the same processing artefacts. Unseen speakers, real voices, unseen system.

Output: ~/.cache/pronunciada/prosody/synja_<system>-ja-<split>.pkl, records as bench.prosody.prep
plus  pair: id of the twin, wrong: bool (this is the edited version), wrong_words: [word indices
whose pitch pattern differs from the dictionary], setting: {...}. Audio (16 kHz FLAC) of the
held-out sets under ~/.cache/pronunciada/data/synja/.
"""
from __future__ import annotations

import copy
import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402

from bench.prosody.prep import OUT, pack  # noqa: E402

CONTENT = ("名詞", "動詞", "形容詞", "副詞")
AUDIO = Path(os.path.expanduser(os.environ.get("PRONUNCIADA_CACHE", "~/.cache/pronunciada"))) / "data" / "synja"


# ----------------------------------------------------------------------------- NJD / labels

def phrases(njd):
    """[(head index, [entry indices], n morae, acc)] accent phrases (chain_flag 1 = continues)."""
    out, cur = [], None
    for i, x in enumerate(njd):
        if x["mora_size"] <= 0:
            cur = None
            continue
        if cur is None or x["chain_flag"] != 1:
            cur = [i, [i], 0, x["acc"]]
            out.append(cur)
        else:
            cur[1].append(i)
        cur[2] += x["mora_size"]
    return [tuple(p) for p in out]


def edit_accent(njd, rng):
    """Copy of njd with one content-word phrase's nucleus moved / removed / added, or None."""
    cand = [p for p in phrases(njd) if njd[p[0]]["pos"] in CONTENT and p[2] >= 2]
    if not cand:
        return None, None
    head, idx, n, acc = cand[int(rng.integers(len(cand)))]
    opts = [a for a in range(0, n + 1) if a != acc and not (acc == 0 and a == n) and not (acc == n and a == 0)]
    if not opts:
        return None, None
    new = int(rng.choice(opts))
    out = copy.deepcopy(njd)
    out[head]["acc"] = new
    return out, {"phrase_head": head, "moras": n, "acc_from": acc, "acc_to": new}


class _Shim:
    def __init__(self, pj, njd):
        self._pj, self._njd = pj, njd

    def run_frontend(self, seg):
        return copy.deepcopy(self._njd)

    def make_label(self, njd):
        return self._pj.make_label(njd)


def reference(g2p, text, njd):
    """The scorer's Reference for `text`, with the given NJD (accent) instead of the analyser's."""
    real = g2p._pj
    g2p._pj = _Shim(real, njd)
    try:
        return g2p(text)
    finally:
        g2p._pj = real


def pitch_words(ref):
    return [tuple(ref.syllables[i].pitch_high for i in range(*w.syllables)) for w in ref.words]


# ----------------------------------------------------------------------------- per clip

_W = {}


def _init():
    import pyopenjtalk

    from pronunciada.acoustic import AcousticModel
    from pronunciada.text import get_g2p

    _W.update(pj=pyopenjtalk, g2p=get_g2p("ja"), am=AcousticModel(threads=2))


def align(ref, wav):
    from pronunciada.align import _spans, emission_table, viterbi
    from pronunciada.lattice import build_lattice, slots_from_reference

    lp = _W["am"].posteriors(wav)
    slots = slots_from_reference(ref)
    lat = build_lattice(slots)
    tab = emission_table(lp)
    score, _ = viterbi(lat, tab)
    if not np.isfinite(score):
        return None
    return [(s, e) for s, e, _ in _spans(lat, slots, tab, len(lp), True)]


def item(uid, ref, wav, extra):
    from pronunciada.prosody.f0 import track

    spans = align(ref, wav)
    if spans is None:
        return None
    return {"id": uid, "speaker": extra.get("speaker"), "dur": len(wav) / 16000, "ref": ref, "spans": spans,
            "track": pack(track(wav)), **extra}


def hts_job(args):
    from scipy.signal import resample_poly

    uid, text, split = args
    pj, g2p = _W["pj"], _W["g2p"]
    rng = np.random.default_rng(zlib.crc32(uid.encode()))
    if any(c.isascii() and c.isalpha() for c in text):
        return []
    njd = pj.run_frontend(text)
    njd2, meta = edit_accent(njd, rng)
    if njd2 is None:
        return []
    if split == "train":
        speed, ht = float(rng.uniform(0.9, 1.1)), float(rng.uniform(-3, 3))
    else:
        speed, ht = float(rng.choice([0.82, 1.22])), float(rng.choice([-6.0, 5.0]))
    out = []
    try:
        r_ok, r_bad = reference(g2p, text, njd), reference(g2p, text, njd2)
    except Exception:  # noqa: BLE001
        return []
    if len(r_ok.syllables) != len(r_bad.syllables) or any("out of sync" in w for w in r_ok.warnings + r_bad.warnings):
        return []
    po, pb = pitch_words(r_ok), pitch_words(r_bad)
    wrong_words = [i for i, (a, b) in enumerate(zip(po, pb)) if a != b]
    if not wrong_words:
        return []
    for tag, nj, ref in (("ok", njd, r_ok), ("bad", njd2, r_bad)):
        w, sr = pj.synthesize(pj.make_label(nj), speed=speed, half_tone=ht)
        wav = resample_poly(w / 32768.0, 1, sr // 16000).astype(np.float32)
        wav = np.clip(wav / max(np.abs(wav).max(), 1e-6) * 0.7, -1, 1)
        it = item(f"{uid}-{tag}", ref, wav, {"pair": f"{uid}-{'bad' if tag == 'ok' else 'ok'}", "wrong": tag == "bad",
                                           "wrong_words": wrong_words, "setting": {"speed": speed, "half_tone": ht, **meta},
                                           "text": text, "dict_pitch": [s.pitch_high for s in r_ok.syllables]})
        if it is None:
            return []
        if split != "train":
            import soundfile as sf

            AUDIO.mkdir(parents=True, exist_ok=True)
            sf.write(AUDIO / f"{uid}-{tag}.flac", wav, 16000)
        out.append(it)
    return out


# ----------------------------------------------------------------------------- PSOLA (natural audio)

def psola_job(args):
    import parselmouth
    from parselmouth.praat import call

    from pronunciada.prosody.features import syllable_windows

    uid, text, wav = args
    pj, g2p = _W["pj"], _W["g2p"]
    rng = np.random.default_rng(zlib.crc32(uid.encode()))
    if any(c.isascii() and c.isalpha() for c in text):
        return []
    njd = pj.run_frontend(text)
    njd2, meta = edit_accent(njd, rng)
    if njd2 is None:
        return []
    try:
        r_ok, r_bad = reference(g2p, text, njd), reference(g2p, text, njd2)
    except Exception:  # noqa: BLE001
        return []
    if len(r_ok.syllables) != len(r_bad.syllables):
        return []
    po, pb = pitch_words(r_ok), pitch_words(r_bad)
    wrong_words = [i for i, (a, b) in enumerate(zip(po, pb)) if a != b]
    if not wrong_words:
        return []
    spans = align(r_ok, wav)
    if spans is None:
        return []
    T = (len(wav) + 80) // 160
    wins = syllable_windows(r_ok, spans, T)
    snd = parselmouth.Sound(wav.astype(np.float64), 16000)
    man = call(snd, "To Manipulation", 0.01, 60, 600)
    tier = call(man, "Extract pitch tier")
    f = snd.to_pitch_ac(0.01, 60, 600)
    ff, ts = f.selected_array["frequency"], f.xs()
    changed = [i for i in range(len(r_ok.syllables)) if r_ok.syllables[i].pitch_high != r_bad.syllables[i].pitch_high]
    lo_s = wins[max(0, changed[0] - 3)].start * 0.01
    hi_s = wins[min(len(wins) - 1, changed[-1] + 3)].end * 0.01
    v = ff[(ts >= lo_s) & (ts <= hi_s) & (ff > 0)]
    if len(v) < 10:
        return []
    med = float(np.median(v))                          # robust to octave errors of the tracker
    H, L = med * 2 ** (2.5 / 12), med * 2 ** (-2.0 / 12)   # ~4.5 semitones, a typical Tokyo fall
    # the edited phrase: every mora of the words whose pattern changed gets its new level
    syl_idx = [i for wi in wrong_words for i in range(*r_bad.words[wi].syllables)]
    a, b = wins[syl_idx[0]].start * 0.01, wins[syl_idx[-1]].end * 0.01
    out = []
    for tag, ref in (("ok", r_ok), ("bad", r_bad)):
        m2 = call(snd, "To Manipulation", 0.01, 60, 600)
        if tag == "bad":
            pt = call(m2, "Extract pitch tier")
            call(pt, "Remove points between", a, b)
            for i in syl_idx:
                w = wins[i]
                lvl = H if ref.syllables[i].pitch_high else L
                for t in np.linspace(w.start * 0.01 + 0.01, w.end * 0.01 - 0.01, 3):
                    call(pt, "Add point", float(t), lvl)
            call([pt, m2], "Replace pitch tier")
        y = call(m2, "Get resynthesis (overlap-add)")
        yy = np.asarray(y.values[0], np.float32)
        yy = yy[:len(wav)] if len(yy) >= len(wav) else np.pad(yy, (0, len(wav) - len(yy)))
        it = item(f"{uid}-{tag}", ref, np.clip(yy, -1, 1), {"pair": f"{uid}-{'bad' if tag == 'ok' else 'ok'}",
                                                            "wrong": tag == "bad", "wrong_words": wrong_words,
                                                            "setting": {"H": H, "L": L, **meta}, "text": text,
                                                            "dict_pitch": [s.pitch_high for s in r_ok.syllables]})
        if it is None:
            return []
        if "train" not in uid:
            import soundfile as sf

            AUDIO.mkdir(parents=True, exist_ok=True)
            sf.write(AUDIO / f"{uid}-{tag}.flac", np.clip(yy, -1, 1), 16000)
        out.append(it)
    _ = tier
    return out


def main(system, split, workers=2):
    if system == "hts":
        if split == "train":
            from bench.data import load

            jobs = [(f"synja-hts-{r['id']}", r["text"], split) for r in load("fleurs", split="train", lang="ja")]
        else:
            from bench.prosody.prep import fleurs_validation

            jobs = [(f"synja-hts-{r['id']}", r["text"], split) for r in fleurs_validation("ja")]
        fn = hts_job
    elif split == "train":           # round 13b: natural FLEURS-ja TRAIN speakers re-pitched (training only)
        from bench.data import load
        from pronunciada.audio import load_audio

        jobs = [(f"synja-psola-{r['id']}", r["text"], load_audio(r["audio"])) for r in load("fleurs", split="train", lang="ja")]
        fn = psola_job
    else:
        from bench.prosody.prep import fleurs_validation

        jobs = [(f"synja-psola-{r['id']}", r["text"], r["wav"]) for r in fleurs_validation("ja")]
        fn = psola_job
    out = []
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        for k, items in enumerate(ex.map(fn, jobs, chunksize=4)):
            out += items
            if k % 100 == 0:
                print(f"{system}/{split}: {k}/{len(jobs)} -> {len(out)} clips", flush=True)
    for u in out:
        u["_src"] = (f"synja_{system}", "ja", split)
    p = OUT / f"synja_{system}-ja-{split}.pkl"
    with open(p.with_suffix(".tmp"), "wb") as f:
        pickle.dump(out, f, protocol=5)
    os.replace(p.with_suffix(".tmp"), p)
    print("wrote", p, len(out), "clips", sum(u["wrong"] for u in out), "edited")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
