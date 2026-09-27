"""tests/test_oracle_agreement.py, Phase B cross-check.

D14 rename: the module oracle/oracle_aggregate.py is a SECONDARY code path
that cross-checks the primary implementation. It uses a distinct
implementation (csv.reader + explicit column index + separate accumulator
structures) from prfs.coverage / prfs.reason_ledger (csv.DictReader + set
algebra) but shares the same logical set-intersection framework and the
same Python interpreter, so it is NOT a language-independent oracle. The
tests below verify agreement between the two implementations; they are not
evidence of external independence.
"""
from __future__ import annotations

from oracle.oracle_aggregate import aggregate
from prfs.coverage import per_filter_coverage, primary_coverage
from prfs.reason_ledger import build_ledger


def test_oracle_matches_primary_on_coverage(registry_csv, outcomes_csv):
    o = aggregate(registry_csv, outcomes_csv)
    s = primary_coverage(registry_csv, outcomes_csv)
    assert o["primary_matched"] == s.primary_matched == 1455
    assert o["primary_denominator"] == s.primary_denominator == 2997
    assert o["secondary_matched"] == s.secondary_matched == 1641
    assert o["mint_matched"] == s.mint_matched == 457
    assert o["sample_aligned"] == s.sample_aligned == 58407


def test_oracle_matches_primary_per_filter(registry_csv, outcomes_csv):
    o = aggregate(registry_csv, outcomes_csv)
    p = per_filter_coverage(registry_csv, outcomes_csv)
    p_by = {x.reason: (x.reg_events, x.matched) for x in p}
    o_by = {x["reason"]: (x["reg_events"], x["matched"]) for x in o["per_filter"]}
    assert p_by == o_by
    assert set(p_by.keys()) == {f"filter_{i}" for i in range(1, 8)}


def test_oracle_matches_primary_on_reason_ledger(registry_csv, outcomes_csv):
    o = aggregate(registry_csv, outcomes_csv)
    rows = build_ledger(registry_csv, outcomes_csv)
    assert o["reason_conflict_rows"] == len(rows) == 253
    assert o["reason_orphan_key_sum"] == sum(r.orphan_key_count for r in rows) == 263
