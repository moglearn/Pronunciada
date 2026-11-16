"""pronunciada: on-device multilingual pronunciation scoring built on ZIPA.

    import pronunciada
    scorer = pronunciada.Scorer("en")
    res = scorer.score("clip.wav", "The quick brown fox", strictness=0.5)
    pronunciada.score("clip.wav", "The quick brown fox", "en")
"""
__version__ = "0.1.0"

from .types import (PhoneResult, Reference, SentenceResult, SyllableResult,  # noqa: E402,F401
                    WordResult)


def __getattr__(name):
    # lazy: importing pronunciada stays cheap (no onnxruntime / model load until a Scorer is built)
    if name in ("Scorer", "score"):
        from . import scorer

        return getattr(scorer, name)
    raise AttributeError(f"module 'pronunciada' has no attribute {name!r}")


__all__ = ["Scorer", "score", "SentenceResult", "WordResult", "SyllableResult", "PhoneResult",
           "Reference", "__version__"]
