"""Dev-only (FLEURS validation) analysis: where do natives' heard tones differ from the reference?

    .venv/bin/python -m bench.prosody.tone_native_cats tone_v2 [tone_v3 ...]

Per category of syllable (neutral particles, 一/不, third-tone sandhi contexts, syllables with
tone_alts, polyphonic characters, position): n, agreement (argmax in {tone} + alts), mean
P(acceptable) and the heard-tone distribution. Used to decide reference-side leniency rules
(pronunciada.prosody.tone_leniency); a rule is only kept if dev natives support it.
"""
from __future__ import annotations

import collections
import sys

import numpy as np

from bench.prosody.common import load_prepped


def categories(ref, i):
    s = ref.syllables[i]
    out = ["all", f"tone{s.tone}"]
    if s.tone == 5:
        out.append("neutral:" + s.text)
    if s.text in "一不":
        out.append(f"yibu:{s.text}{s.tone}")
    if s.tone_lexical == 3 and s.tone == 2:
        out.append("t3sandhi->2")
    if s.tone == 3 and i + 1 < len(ref.syllables) and ref.syllables[i + 1].tone == 3:
        out.append("3before3(kept)")
    if s.tone_alts:
        out.append("has_alts")
    if s.tone_lexical != s.tone and s.tone != 5 and s.text not in "一不":
        out.append("surface!=lexical")
    if s.tone_lexical and s.tone == 5:
        out.append(f"neutral_lex{s.tone_lexical}")
    w = ref.words[s.word]
    a, b = w.syllables
    out.append("pos:mono" if b - a == 1 else ("pos:first" if i == a else ("pos:last" if i == b - 1 else "pos:mid")))
    if i == len(ref.syllables) - 1:
        out.append("utt_final")
    return out


def run(name):
    from bench.prosody.tone2 import examples, fit_temperature, load2, predict_utts, softmax

    net, kind, _ = load2(name)
    dev = load_prepped("fleurs_val", "zh", "dev")
    ex, keep = examples(dev)
    Z = predict_utts(net, kind, [x for x, _, _, _ in ex])
    y = np.concatenate([yy for _, yy, _, _ in ex])
    T = fit_temperature(np.concatenate(Z)[y >= 0], y[y >= 0])
    st = collections.defaultdict(lambda: [0, 0, 0.0, np.zeros(5)])
    for z, ui in zip(Z, keep):
        P = softmax(z, T)
        ref = dev[ui]["ref"]
        for i, s in enumerate(ref.syllables):
            if not s.tone:
                continue
            acc = {s.tone, *s.tone_alts}
            h = int(P[i].argmax()) + 1
            pa = float(sum(P[i, t - 1] for t in acc))
            for c in categories(ref, i):
                r = st[c]
                r[0] += 1
                r[1] += h in acc
                r[2] += pa
                r[3][h - 1] += 1
    print(f"== {name} (T={T:.3f})")
    rows = sorted(st.items(), key=lambda kv: (-kv[1][0]))
    for c, (n, ok, pa, hd) in rows:
        if n < 15 and not c.startswith(("yibu", "tone", "t3", "3before")):
            continue
        print(f"{c:22s} n={n:5d} agree={ok / n:.3f} P_acc={pa / n:.3f} heard 1-5: " +
              " ".join(f"{v / n:.2f}" for v in hd))


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
