"""Environment overrides for measuring candidate models (size / speed work, bench/opt/).

Nothing here is read unless the variable is set; with all three unset every code path is the
default one and the outputs are unchanged.

  PRONUNCIADA_BACKBONE      path of an ONNX file used as the acoustic model (backbone) in place of
                            $PRONUNCIADA_CACHE/models/<name>/<precision file>, whatever precision
                            was asked for. The file's directory must be writable: the variant with
                            extra encoder outputs is built next to it, as for the cached model.
  PRONUNCIADA_ADAPTER_DIR   directory searched first for the per-language adapter <lang>.npz; a
                            language without a file there uses the packaged adapter.
  PRONUNCIADA_ORT_OPTIONS   JSON object of onnxruntime.SessionOptions attributes applied to every
                            ONNX Runtime session the scorer opens, after its own settings, e.g.
                            {"graph_optimization_level": "ORT_ENABLE_ALL", "enable_mem_pattern": true,
                             "config": {"session.intra_op.allow_spinning": "0"},
                             "sessions": ["backbone"]}
                            Enum values are given by name. "config" entries go through
                            add_session_config_entry; "sessions" (optional) limits the options to
                            some of "backbone", "sentence_head", "prosody".
"""
from __future__ import annotations

import json
import os
from pathlib import Path

SESSION_KINDS = ("backbone", "sentence_head", "prosody")
_ENUMS = {"graph_optimization_level": "GraphOptimizationLevel", "execution_mode": "ExecutionMode"}


def backbone_file() -> str | None:
    """The PRONUNCIADA_BACKBONE file, or None when unset."""
    p = os.environ.get("PRONUNCIADA_BACKBONE")
    if not p:
        return None
    if not os.path.isfile(p):
        raise FileNotFoundError(f"PRONUNCIADA_BACKBONE: no such file: {p}")
    return p


def adapter_file(lang: str) -> Path | None:
    """<PRONUNCIADA_ADAPTER_DIR>/<lang>.npz if the variable is set and the file exists, else None."""
    d = os.environ.get("PRONUNCIADA_ADAPTER_DIR")
    if not d:
        return None
    if not os.path.isdir(d):
        raise FileNotFoundError(f"PRONUNCIADA_ADAPTER_DIR: no such directory: {d}")
    p = Path(d) / f"{lang}.npz"
    return p if p.exists() else None


def ort_options() -> dict | None:
    s = os.environ.get("PRONUNCIADA_ORT_OPTIONS")
    if not s:
        return None
    o = json.loads(s)
    if not isinstance(o, dict):
        raise ValueError("PRONUNCIADA_ORT_OPTIONS must be a JSON object")
    bad = [k for k in o.get("sessions", []) if k not in SESSION_KINDS]
    if bad:
        raise ValueError(f"PRONUNCIADA_ORT_OPTIONS: unknown session kinds {bad}")
    return o


def apply_ort_options(so, kind: str):
    """Apply PRONUNCIADA_ORT_OPTIONS to SessionOptions `so` of a session of `kind`; returns `so`."""
    o = ort_options()
    if o is None or kind not in o.get("sessions", SESSION_KINDS):
        return so
    import onnxruntime as ort

    for k, v in o.items():
        if k == "sessions":
            continue
        if k == "config":
            for ck, cv in v.items():
                so.add_session_config_entry(str(ck), str(cv))
            continue
        if not hasattr(so, k):
            raise ValueError(f"PRONUNCIADA_ORT_OPTIONS: SessionOptions has no attribute {k!r}")
        if k in _ENUMS and isinstance(v, str):
            v = getattr(getattr(ort, _ENUMS[k]), v)
        setattr(so, k, v)
    return so
