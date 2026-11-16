"""Does a tone model read the pitch, or does it guess the tone from the segments?

Native audio is resynthesised with a flat (monotone) pitch contour at the speaker's median F0
(Praat PSOLA via parselmouth: offline benchmark only, never in the runtime). Segments and timing
stay the same, so the same alignment spans are reused; ZIPA posteriors (bench.postcache dataset
"aishell3-monotone"), pitch track and spectral extras are recomputed from the new audio.

    .venv/bin/python -m bench.prosody.leak_test make          # 300 aishell3 test utterances
    .venv/bin/python -m bench.prosody.leak_test eval d_ref d_spec ...
    .venv/bin/python -m bench.prosody.leak_test eval tone_v2 t3_reg ...     # utterance-level models

Metrics per model, on syllables whose label tone is 2, 3 or 4 (a flat contour is wrong for them):
  acc_orig / acc_mono   accuracy vs the label on original / monotone audio
  still_label_mono      share of monotone syllables still classified as their (text) label tone
  mono_auc              detection AUC: negatives = P(acceptable) on the original audio,
                        positives = P(acceptable) on the monotone audio (higher = model hears it)
Plus the text-only prior: accuracy of the most frequent tone of each toneless syllable (from the
aishell3 train labels), i.e. what a model could get from segment identity alone.
"""
from __future__ import annotations

import collections
import copy
import os
import pickle
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "2")

from bench import metrics as M  # noqa: E402
from bench.prosody.common import OUT, build_arrays, load_prepped, save_json  # noqa: E402
from bench.prosody.prep import out_path, pack  # noqa: E402

MONO = OUT / "monotone"
N = 300


def monotone(wav: np.ndarray) -> np.ndarray:
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(wav.astype(np.float64), 16000)
    man = call(snd, "To Manipulation", 0.01, 60, 600)
    pt = call(man, "Extract pitch tier")
    f = snd.to_pitch_ac(0.01, 60, 600).selected_array["frequency"]
    med = float(np.median(f[f > 0])) if (f > 0).any() else 150.0
    call(pt, "Remove points between", 0, snd.duration)
    call(pt, "Add point", 0.0, med)
    call(pt, "Add point", snd.duration, med)
    call([pt, man], "Replace pitch tier")
    out = call(man, "Get resynthesis (overlap-add)")
    y = np.asarray(out.values[0], dtype=np.float32)
    y = y[:len(wav)] if len(y) >= len(wav) else np.pad(y, (0, len(wav) - len(y)))
    return np.clip(y, -1, 1)


def make():
    import soundfile as sf

    from bench import postcache
    from bench.data import load
    from bench.prosody.extra_feats import extra
    from pronunciada.prosody.f0 import track

    MONO.mkdir(parents=True, exist_ok=True)
    postcache.model(threads=1)        # stay within the CPU budget while training runs
    utts = load_prepped("aishell3", "zh", "test")[::800 // N][:N]
    recs = {r["id"]: r for r in load("aishell3", split="test")}
    from pronunciada.audio import load_audio

    new, ex = [], {}
    for u in utts:
        wav = load_audio(recs[u["id"]]["audio"])
        y = monotone(wav)
        sf.write(MONO / f"{u['id']}.flac", y, 16000)
        postcache.get_posteriors("aishell3-monotone", u["id"], y)
        v = copy.copy(u)
        v["track"] = pack(track(y))
        v.pop("noisy", None)
        new.append(v)
        ex[u["id"]] = extra(y)
    with open(out_path("aishell3mono", "zh", "test"), "wb") as f:
        pickle.dump(new, f)
    with open(out_path("aishell3orig", "zh", "test"), "wb") as f:
        pickle.dump(utts, f)
    np.savez(OUT / "extra" / "aishell3mono-zh-test.npz", **ex)
    full = np.load(OUT / "extra" / "aishell3-zh-test.npz")
    np.savez(OUT / "extra" / "aishell3orig-zh-test.npz", **{u["id"]: full[u["id"]] for u in utts})
    print("made", len(new))


def text_prior():
    from bench.prosody.tone_agree import norm  # noqa: F401

    def base(s):     # toneless syllable identity = its phone string
        return s.text, None

    cnt = collections.defaultdict(collections.Counter)
    for u in load_prepped("aishell3", "zh", "train"):
        r = u["ref"]
        for s in r.syllables:
            if s.tone:
                cnt["".join(r.phones[i].ipa for i in range(*s.phones))][s.tone] += 1
            if s.tone:
                cnt["char:" + s.text][s.tone] += 1
    ok_ph = ok_ch = n = 0
    for u in load_prepped("aishell3orig", "zh", "test"):
        r = u["ref"]
        for s in r.syllables:
            if not s.tone or s.tone == 1:
                continue
            ph = "".join(r.phones[i].ipa for i in range(*s.phones))
            n += 1
            ok_ph += cnt[ph].most_common(1)[0][0] == s.tone if cnt[ph] else 0
            ok_ch += cnt["char:" + s.text].most_common(1)[0][0] == s.tone if cnt["char:" + s.text] else 0
    return {"n": n, "acc_from_phones": ok_ph / n, "acc_from_character": ok_ch / n}


def eval_seq(name):
    """Utterance-level models (tone2 / tone3 checkpoints, e.g. tone_v2 and the v3 candidates):
    the same syllables (label 2/3/4), inputs rebuilt from the monotone audio (pitch track and,
    for spectral models, the low-frequency log-mel of the resynthesised audio)."""
    import json

    import torch

    from bench.prosody import tone2, tone3

    net, kind, ck = tone3.load3(name)
    T = json.load(open(OUT / "results" / f"{name}.json"))["temperature"]
    P = {}
    for cond in ("orig", "mono"):
        utts = load_prepped(f"aishell3{cond}", "zh", "test")
        ex, keep = tone2.examples(utts)
        P[cond] = ([tone2.softmax(z, T) for z in tone2.predict_utts(net, kind, [x for x, _, _, _ in ex])], keep, utts)
    (Po, keep, uo), (Pm, _, _) = P["orig"], P["mono"]
    pa_o, pa_m, ok_o, ok_m, pred1 = [], [], [], [], []
    for po, pm, ui in zip(Po, Pm, keep):
        for si, s in enumerate(uo[ui]["ref"].syllables):
            if s.tone in (2, 3, 4):
                acc = {s.tone, *s.tone_alts}
                pa_o.append(sum(po[si, t - 1] for t in acc))
                pa_m.append(sum(pm[si, t - 1] for t in acc))
                ok_o.append(int(po[si].argmax()) + 1 == s.tone)
                ok_m.append(int(pm[si].argmax()) + 1 == s.tone)
                pred1.append(int(pm[si].argmax()) == 0)
    lab = np.r_[np.zeros(len(pa_o)), np.ones(len(pa_m))]
    out = {"n": len(pa_o), "acc_orig": float(np.mean(ok_o)), "acc_mono": float(np.mean(ok_m)),
           "still_label_mono": float(np.mean(ok_m)), "pred_tone1_mono": float(np.mean(pred1)),
           "mono_auc": M.roc_auc(lab, 1 - np.r_[pa_o, pa_m]),
           "mean_p_acc_orig": float(np.mean(pa_o)), "mean_p_acc_mono": float(np.mean(pa_m))}
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}, flush=True)
    _ = torch
    return out


def eval_model(name):
    from bench.prosody.tone_eval import load_model, logits, softmax

    import torch

    if "model" in torch.load(OUT / "models" / f"{name}.pt", weights_only=False):
        return eval_seq(name)
    net, ck = load_model(name)       # also sets the feature options of the model
    T = __import__("json").load(open(OUT / "results" / f"{name}.json"))["temperature"]
    out = {}
    P = {}
    for cond in ("orig", "mono"):
        utts = load_prepped(f"aishell3{cond}", "zh", "test")
        # posteriors: original from the aishell3 cache, monotone from aishell3-monotone
        for u in utts:
            u["_src"] = ("aishell3" if cond == "orig" else "aishell3-monotone", "zh", "test") \
                if ck.get("opts", {}).get("extra", "").startswith("post") else (f"aishell3{cond}", "zh", "test")
        X, y, meta = build_arrays(utts, ck["kind"])
        P[cond] = (softmax(logits(net, X), T), y, meta, utts)
    (Po, y, meta, uo), (Pm, ym, _, _) = P["orig"], P["mono"]
    sel = np.isin(y, [1, 2, 3])                   # label tones 2, 3, 4
    acc_o = float((Po.argmax(1) == y)[sel].mean())
    acc_m = float((Pm.argmax(1) == y)[sel].mean())
    pa_o = np.array([sum(Po[k, t - 1] for t in {uo[a]["ref"].syllables[b].tone, *uo[a]["ref"].syllables[b].tone_alts})
                     for k, (a, b) in enumerate(meta)])
    pa_m = np.array([sum(Pm[k, t - 1] for t in {uo[a]["ref"].syllables[b].tone, *uo[a]["ref"].syllables[b].tone_alts})
                     for k, (a, b) in enumerate(meta)])
    lab = np.concatenate([np.zeros(sel.sum()), np.ones(sel.sum())])
    sc = np.concatenate([1 - pa_o[sel], 1 - pa_m[sel]])
    out = {"n": int(sel.sum()), "acc_orig": acc_o, "acc_mono": acc_m, "still_label_mono": acc_m,
           "pred_tone1_mono": float((Pm.argmax(1) == 0)[sel].mean()), "mono_auc": M.roc_auc(lab, sc),
           "mean_p_acc_orig": float(pa_o[sel].mean()), "mean_p_acc_mono": float(pa_m[sel].mean())}
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}, flush=True)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make()
    else:
        res = {"text_prior": text_prior()}
        print("text prior", res["text_prior"])
        for n in sys.argv[2:]:
            res[n] = eval_model(n)
        save_json(res, "leak_test.json" if all(not n.startswith(("t3", "tone_v")) for n in sys.argv[2:])
                  else "leak_test_v3.json")
