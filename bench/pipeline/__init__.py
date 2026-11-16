"""Phase-2 evidence pipeline. Every stage runs as

    python -m bench.pipeline.<stage> --lang <code>|all [--dataset ...]

and is idempotent (results are cached under $PRONUNCIADA_CACHE/pipeline/ and
$PRONUNCIADA_CACHE/features/; pass --force to recompute).

Stages: warm (posterior cache), train_adapter, native_stats, features, analyze (results/).
"""
