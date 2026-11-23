"""Reader for NumPy .npz files (a zip archive of .npy arrays), standard library only.

.npy format (version 1.0 / 2.0 / 3.0): 6 bytes magic b"\\x93NUMPY", 1 byte major, 1 byte minor
version, header length (uint16 little-endian for 1.x, uint32 for 2.x / 3.x), then an ASCII
(3.x: UTF-8) Python dict literal {'descr': '<f8', 'fortran_order': False, 'shape': (n, m)}
padded with spaces, then the raw array data in C order. The calibration files use the types
below; strings ('<U') are stored as UTF-32 code units.
"""
from __future__ import annotations

import ast
import struct
import zipfile

_TYPES = {"<f8": ("d", 8), "<f4": ("f", 4), "<f2": ("e", 2), "<i8": ("q", 8), "<i4": ("i", 4),
          "<i2": ("h", 2), "|i1": ("b", 1), "<u4": ("I", 4), "<u2": ("H", 2), "|u1": ("B", 1),
          "|b1": ("?", 1)}


class Array:
    """A decoded array: `shape` (tuple) and `data` (flat list in C order)."""

    def __init__(self, shape, data):
        self.shape = tuple(shape)
        self.data = data

    def rows(self) -> list:
        """2-D array as a list of rows."""
        n, m = self.shape
        return [self.data[i * m:(i + 1) * m] for i in range(n)]

    def scalar(self):
        return self.data[0]


def parse_npy(buf: bytes) -> Array:
    if buf[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    major = buf[6]
    if major == 1:
        hlen = struct.unpack("<H", buf[8:10])[0]
        off = 10
    else:
        hlen = struct.unpack("<I", buf[8:12])[0]
        off = 12
    header = ast.literal_eval(buf[off:off + hlen].decode("utf-8"))
    off += hlen
    descr, shape = header["descr"], header["shape"]
    if header["fortran_order"] and len(shape) > 1:
        raise ValueError("Fortran-order arrays are not used by the packaged files")
    count = 1
    for s in shape:
        count *= s
    if descr.startswith("<U"):
        width = int(descr[2:])
        out = []
        for i in range(count):
            raw = buf[off + 4 * width * i: off + 4 * width * (i + 1)]
            out.append(raw.decode("utf-32-le").rstrip("\x00"))
        return Array(shape, out)
    code, size = _TYPES[descr]
    data = list(struct.unpack("<%d%s" % (count, code), buf[off:off + count * size]))
    return Array(shape, data)


def load_npz(path: str) -> dict:
    """{name: Array} for every member of an .npz file (names without the .npy suffix)."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            key = name[:-4] if name.endswith(".npy") else name
            out[key] = parse_npy(z.read(name))
    return out
