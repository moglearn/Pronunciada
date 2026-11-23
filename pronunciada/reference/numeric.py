"""Scalar arithmetic shared by the reference modules.

Every number is a Python float, i.e. an IEEE-754 binary64 ("double") value; every operation
below is one correctly rounded binary64 operation unless stated otherwise. exp, log and log1p
are the C library's functions (Python's math module). A port uses double everywhere and the
platform's exp / log / log1p; on the machine the tests were run on, NumPy's exp / log / log1p
returned the same bits as the C library.

Python raises exceptions where IEEE arithmetic returns inf or NaN (log(0), exp overflow, x / 0);
the helpers here return the IEEE results instead, so that the code reads like C.

Summation order. Production (NumPy) adds long vectors in one of three orders, and the last bits
of a sum depend on the order. To let a port reproduce production bit for bit, every sum in the
reference names its order:

* ``sum_seq``       left to right, starting from 0.0;
* ``sum_pairwise``  NumPy's pairwise summation of a contiguous vector (blocks of 128, eight
                    running partial sums inside a block); used where production sums a
                    contiguous array;
* ``sum_first_pw``  x[0] + sum_pairwise(x[1:]); NumPy's order for a reduction over a strided
                    slice (``np.add.reduceat``).
A port that only needs agreement to about 1e-12 may use any order.
"""
from __future__ import annotations

import math

INF = float("inf")
NEG_INF = float("-inf")
NAN = float("nan")
LOG2 = math.log(2.0)          # 0.6931471805599453


def is_nan(x: float) -> bool:
    return x != x


def is_finite(x: float) -> bool:
    return x == x and x != INF and x != NEG_INF


def exp(x: float) -> float:
    """e**x with IEEE overflow (inf) instead of an exception."""
    if x != x:
        return NAN
    if x > 709.782712893384:
        return INF
    return math.exp(x)


def log(x: float) -> float:
    """Natural log; log(0) = -inf, log(x < 0) = NaN, log(NaN) = NaN."""
    if x != x or x < 0.0:
        return NAN
    if x == 0.0:
        return NEG_INF
    if x == INF:
        return INF
    return math.log(x)


def log1p(x: float) -> float:
    if x != x or x < -1.0:
        return NAN
    if x == -1.0:
        return NEG_INF
    if x == INF:
        return INF
    return math.log1p(x)


def div(a: float, b: float) -> float:
    """a / b with IEEE semantics for b == 0 (inf with the sign of a, NaN for 0 / 0)."""
    if b == 0.0:
        if a != a or a == 0.0:
            return NAN
        neg = (a < 0.0) != (math.copysign(1.0, b) < 0.0)
        return NEG_INF if neg else INF
    return a / b


def logaddexp(x: float, y: float) -> float:
    """log(exp(x) + exp(y)), NumPy's formula (npy_logaddexp), bit for bit:
    x == y -> x + log 2; else with d = x - y: d > 0 -> x + log1p(exp(-d)),
    d <= 0 -> y + log1p(exp(d)); NaN in -> NaN out."""
    if x == y:
        return x + LOG2
    d = x - y
    if d > 0.0:
        return x + log1p(exp(-d))
    if d <= 0.0:
        return y + log1p(exp(d))
    return x + y                     # NaN


def maximum(a: float, b: float) -> float:
    """np.maximum: NaN if either input is NaN, else the larger value."""
    if a != a or b != b:
        return NAN
    return a if a >= b else b


def minimum(a: float, b: float) -> float:
    """np.minimum: NaN if either input is NaN, else the smaller value."""
    if a != a or b != b:
        return NAN
    return a if a <= b else b


def max_list(xs) -> float:
    """Largest value (np.max): NaN if any value is NaN; -inf for an empty list."""
    m = NEG_INF
    for x in xs:
        if x != x:
            return NAN
        if x > m:
            m = x
    return m


def argmax_list(xs) -> int:
    """Index of the first largest value (np.argmax); the first NaN wins if there is one."""
    best = 0
    for i in range(len(xs)):
        if xs[i] != xs[i]:
            return i
        if xs[i] > xs[best]:
            best = i
    return best


def sum_seq(xs) -> float:
    r = 0.0
    for x in xs:
        r += x
    return r


def _pairwise(a, lo: int, n: int) -> float:
    if n < 8:
        r = 0.0
        for i in range(n):
            r += a[lo + i]
        return r
    if n <= 128:
        r0 = a[lo]; r1 = a[lo + 1]; r2 = a[lo + 2]; r3 = a[lo + 3]
        r4 = a[lo + 4]; r5 = a[lo + 5]; r6 = a[lo + 6]; r7 = a[lo + 7]
        i = 8
        stop = n - (n % 8)
        while i < stop:
            r0 += a[lo + i]; r1 += a[lo + i + 1]; r2 += a[lo + i + 2]; r3 += a[lo + i + 3]
            r4 += a[lo + i + 4]; r5 += a[lo + i + 5]; r6 += a[lo + i + 6]; r7 += a[lo + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[lo + i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _pairwise(a, lo, n2) + _pairwise(a, lo + n2, n - n2)


def sum_pairwise(xs) -> float:
    """NumPy's pairwise summation (numpy/_core/src/umath/loops_utils.h.src, pairwise_sum)."""
    return _pairwise(xs, 0, len(xs))


def sum_first_pw(xs) -> float:
    """x[0] + pairwise(x[1:]) (order of a NumPy reduction over a strided slice)."""
    if not xs:
        return 0.0
    return xs[0] + _pairwise(xs, 1, len(xs) - 1)


def lse(xs, order: str = "seq") -> float:
    """log-sum-exp as production writes it: m = max (NaN-propagating), mm = m if finite else 0,
    mm + log(sum(exp(x - mm))) with the named summation order."""
    m = max_list(xs)
    mm = m if is_finite(m) else 0.0
    e = [exp(x - mm) for x in xs]
    s = sum_pairwise(e) if order == "pairwise" else sum_seq(e)
    return mm + log(s)


def fma(a: float, b: float, c: float) -> float:
    """a * b + c with a single rounding (C fma()). Computed exactly with rational arithmetic
    when the operands are finite."""
    if not (is_finite(a) and is_finite(b) and is_finite(c)):
        return a * b + c
    from fractions import Fraction

    return float(Fraction(a) * Fraction(b) + Fraction(c))


def float32(x: float) -> float:
    """Round a double to the nearest binary32 value (returned as a double)."""
    import struct

    if x != x or x == INF or x == NEG_INF:
        return x
    return struct.unpack("<f", struct.pack("<f", x))[0]
