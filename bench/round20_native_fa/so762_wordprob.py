"""Gate item 2 on the word score users see: speechocean762 TRAIN, WordResult.prob (what the Azure-format output
and the Lingolix comparison read), switches off vs B + C on. Prompt = the scored words joined by spaces.

    python -m bench.round20_native_fa.so762_wordprob rows.json

Prints the summary and writes it next to rows.json (copied to results/tables/round20_native_fa_r20_so762_wordprob.json).
Paired speaker-clustered bootstrap (2,000 resamples) for the change in word correlation.
"""
import json, os, sys
import numpy as np
from bench.data import load
from pronunciada import Scorer

ON = {"PRONUNCIADA_R20_UNSAID": "1", "PRONUNCIADA_R20_NUMALT": "1"}
out_path = sys.argv[1]
sc = Scorer("en", threads=4)
rows = []
recs = load("speechocean762", split="train")
for n, rec in enumerate(recs):
    hw = rec["labels"]["words"]
    text = " ".join(w["text"] for w in hw)
    for k in ON: os.environ.pop(k, None)
    try:
        r0 = sc.score(rec["audio"], text)
    except ValueError:
        continue
    os.environ.update(ON)
    r1 = sc.score(rec["audio"], text)
    for k in ON: os.environ.pop(k, None)
    if len(r0.words) != len(hw):
        continue
    rows.append({"id": rec["id"], "spk": rec["speaker"], "h": [w["accuracy"] for w in hw],
                 "p0": [w.prob for w in r0.words], "p1": [w.prob for w in r1.words],
                 "t": rec["labels"]["sentence"]["total"], "s0": r0.scores.get("total"), "s1": r1.scores.get("total"),
                 "pr0": r0.prob, "pr1": r1.prob, "info": {k: (v is not None) for k, v in (r1._pronunciada.get("r20") or {}).items()}})
    if n % 250 == 0: print(n, flush=True)
json.dump(rows, open(out_path, "w"))
h = np.concatenate([r["h"] for r in rows]); p0 = np.concatenate([r["p0"] for r in rows]); p1 = np.concatenate([r["p1"] for r in rows])
spk = np.concatenate([[r["spk"]] * len(r["h"]) for r in rows])
c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
res = {"clips": len(rows), "words": int(len(h)), "words_changed": int((p0 != p1).sum()),
       "clips_changed": int(sum(r["p0"] != r["p1"] for r in rows)),
       "word_pcc_off": c(p0, h), "word_pcc_on": c(p1, h),
       "sent_total_max_abs_diff": float(max(abs(r["s0"] - r["s1"]) for r in rows)),
       "sent_prob_max_abs_diff": float(max(abs(r["pr0"] - r["pr1"]) for r in rows))}
ch = p0 != p1
if ch.any():
    res["changed_words"] = {"mean_p_off": float(p0[ch].mean()), "mean_p_on": float(p1[ch].mean()), "mean_human": float(h[ch].mean()),
                            "human_below_7": int((h[ch] < 7).sum()), "raised": int((p1[ch] > p0[ch]).sum()), "lowered": int((p1[ch] < p0[ch]).sum())}
rng = np.random.default_rng(0); us = np.unique(spk); idx = {s: np.where(spk == s)[0] for s in us}; d = []
for _ in range(2000):
    ii = np.concatenate([idx[s] for s in rng.choice(us, len(us))])
    d.append(c(p1[ii], h[ii]) - c(p0[ii], h[ii]))
d = np.array(d); res["delta"] = res["word_pcc_on"] - res["word_pcc_off"]; res["delta_ci95"] = [float(np.quantile(d, .025)), float(np.quantile(d, .975))]
# mispronounced-word detection (human accuracy < 7 = flagged), AUC of 1-p
def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
y = (h < 7).astype(int); res["auc_low_words_off"] = auc(1 - p0, y); res["auc_low_words_on"] = auc(1 - p1, y)
print(json.dumps(res, indent=1))
json.dump(res, open(out_path.replace(".json", "_summary.json"), "w"), indent=1)
