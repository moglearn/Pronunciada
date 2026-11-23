"""Measuring and gating harness for the size / CPU-speed work (docs/optimization_and_mobile_report.md).

    bench/opt/profile.py   per-stage latency, cold start, peak RSS, disk footprint
    bench/opt/parity.py    numerical parity of a candidate against a reference
    bench/opt/gate.py      the zero-regression quality gate (held-out suite, cached Azure only)
    bench/opt/suite.py     runs the held-out suite behind bench/scoreboard.py for one configuration
"""
