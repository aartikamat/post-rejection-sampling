"""Independent oracle package for Phase B validation.

Deliberately implemented WITHOUT reusing prfs.coverage / prfs.reason_ledger.
Uses awk-style single-pass streaming aggregation over the two CSVs to
independently reproduce 1,455 / 1,641 / 263 / 253. If the oracle disagrees
with prfs, the disagreement is the failure signal, do not tweak either
side to force agreement.
"""
