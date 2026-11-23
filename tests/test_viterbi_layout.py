"""align.viterbi with predecessors laid out [K+1, S] equals the row-wise reduction it replaced."""
import numpy as np

from pronunciada import align as AL
from pronunciada.lattice import build_lattice, slots_from_reference
from pronunciada.text import get_g2p


def _viterbi_rows(lat, tab):
    """The previous implementation (max over the short last axis), kept as the reference."""
    T, S = len(tab), lat.n_states
    if T == 0:
        return -np.inf, np.zeros(0, dtype=np.int64)
    E = tab[:, AL._cols(lat.lab, tab.shape[1] - 2)]
    P, W = AL._padded_preds(lat)
    D = np.empty((T, S + 1))
    d = np.append(lat.init + E[0], -np.inf)
    D[0] = d
    for t in range(1, T):
        d[:S] = (d[P] + W).max(axis=1) + E[t]
        D[t] = d
    x = d[:S] + lat.fin
    s = int(x.argmax())
    score = float(x[s])
    if not np.isfinite(score):
        return -np.inf, np.zeros(0, dtype=np.int64)
    path = np.empty(T, dtype=np.int64)
    path[-1] = s
    for t in range(T - 1, 0, -1):
        s = int(P[s, int((D[t - 1, P[s]] + W[s]).argmax())])
        path[t - 1] = s
    return score, path


def test_viterbi_identical_to_row_reduction():
    rng = np.random.default_rng(3)
    g = get_g2p("en")
    for text, T in (("the quick brown fox", 90), ("hello", 12), ("a", 1), ("a cat sat on the mat", 400),
                    ("internationalization", 30), ("one two", 3)):
        lat = build_lattice(slots_from_reference(g(text)))
        for k in range(3):
            lp = rng.standard_normal((T, 127)) * (1 + 3 * k)
            lp -= np.log(np.exp(lp).sum(1, keepdims=True))
            if k == 2:
                lp = np.round(lp, 1)                    # many ties: argmax order must hold
            tab = AL.emission_table(lp.astype(np.float32))
            a, b = AL.viterbi(lat, tab), _viterbi_rows(lat, tab)
            assert a[0] == b[0] and np.array_equal(a[1], b[1])
