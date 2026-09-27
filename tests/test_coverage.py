"""tests/test_coverage.py, must reproduce the Phase 1 numbers exactly.

Expected against the frozen deposit at 10.5281/zenodo.20043516:
    primary  (mint, ts, reason) = 1,455 / 2,997 = 48.55%
    secondary (mint, ts)        = 1,641 / 2,997 = 54.75%
    mint-level                  = 457 / 457 = 100.00%
    sample alignment            = 58,407 / 67,000 = 87.17% (full denominator)
                                = 58,407 / 58,407 = 100.00% (eligible after excluding
                                  8,593 orphan-key rows)
    per-filter (all 7 rows)     - see FORMAL_SPEC section 4.3.

D14 additions verified here: canonical epoch-ms timestamp identity,
key-uniqueness assertion, dual sample-alignment denominators.
D17 addition verified here: per-filter Wilson 95% binomial CI.
"""
from __future__ import annotations

import pytest

from prfs.coverage import (
    KeyUniquenessError,
    assert_registry_key_uniqueness,
    per_filter_coverage,
    per_filter_coverage_with_ci,
    primary_coverage,
    secondary_coverage_diagnostic,
    wilson_ci,
)


def test_primary_event_coverage(registry_csv, outcomes_csv):
    s = primary_coverage(registry_csv, outcomes_csv)
    assert s.primary_matched == 1455
    assert s.primary_denominator == 2997
    assert round(s.primary_pct, 2) == 48.55


def test_secondary_diagnostic_coverage(registry_csv, outcomes_csv):
    matched, denom = secondary_coverage_diagnostic(registry_csv, outcomes_csv)
    assert matched == 1641
    assert denom == 2997
    assert round(matched / denom * 100.0, 2) == 54.75


def test_mint_level_coverage(registry_csv, outcomes_csv):
    s = primary_coverage(registry_csv, outcomes_csv)
    assert s.mint_matched == 457
    assert s.mint_denominator == 457
    assert s.mint_pct == 100.0


def test_sample_alignment_full_denominator(registry_csv, outcomes_csv):
    """% of ALL 67,000 outcome rows that attach to a matched 3-field key."""
    s = primary_coverage(registry_csv, outcomes_csv)
    assert s.sample_aligned == 58407
    assert s.sample_denominator_full == 67000
    assert round(s.sample_pct_full, 2) == 87.17


def test_sample_alignment_eligible_denominator(registry_csv, outcomes_csv):
    """D14: % of ELIGIBLE outcome rows (excluding the 8,593 orphan-key
    rows that share (mint, rejectTs) with a registry row under a different
    reason and therefore cannot align by construction). Under this
    denominator alignment is 58,407 / 58,407 = 100.00%."""
    s = primary_coverage(registry_csv, outcomes_csv)
    assert s.orphan_key_row_count == 67000 - 58407  # = 8,593
    assert s.sample_denominator_eligible == 58407
    assert round(s.sample_pct_eligible, 2) == 100.00


def test_per_filter_coverage_all_seven_rows(registry_csv, outcomes_csv):
    pf = per_filter_coverage(registry_csv, outcomes_csv)
    assert len(pf) == 7
    exp = {
        "filter_1": (579, 804, 72.01),
        "filter_2": (474, 1444, 32.83),
        "filter_3": (219, 313, 69.97),
        "filter_4": (117, 317, 36.91),
        "filter_5": (39, 62, 62.90),
        "filter_6": (9, 15, 60.00),
        "filter_7": (18, 42, 42.86),
    }
    for row in pf:
        exp_m, exp_t, exp_pct = exp[row.reason]
        assert row.matched == exp_m, f"{row.reason} matched"
        assert row.reg_events == exp_t, f"{row.reason} total"
        assert round(row.coverage_pct, 2) == exp_pct, f"{row.reason} pct"


def test_primary_and_secondary_delta_is_186(registry_csv, outcomes_csv):
    """The gap between secondary (2-field) and primary (3-field) coverage is
    186 registry events. This is not clock skew; it is reason-field drift
    between the two files (see FORMAL_SPEC section 4.2)."""
    s = primary_coverage(registry_csv, outcomes_csv)
    matched2, _ = secondary_coverage_diagnostic(registry_csv, outcomes_csv)
    assert matched2 - s.primary_matched == 186


def test_registry_key_uniqueness_asserted(registry_csv):
    """D14: the deposit is expected to have zero duplicate 3-field keys. The
    assertion must pass on the deposited file."""
    report = assert_registry_key_uniqueness(registry_csv)
    assert report.total_rows == 2997
    assert report.distinct_3f_keys == 2997
    assert report.duplicate_3f_key_rows == 0
    assert report.duplicate_3f_key_groups == 0
    assert report.conflicting_duplicate_2f_keys == 0
    assert report.exact_duplicate_3f_key_groups == 0


def test_registry_key_uniqueness_halts_on_duplicate(tmp_path):
    """D14: introducing a duplicate 3-field row must raise
    KeyUniquenessError. The pipeline is expected to HALT rather than infer
    uniqueness from a count coincidence."""
    p = tmp_path / "reg_dupe.csv"
    p.write_text(
        "timestamp,source,mint,symbol,reason,timeSlot\n"
        "2026-04-15T00:00:00.000Z,src,M1,S1,filter_1,normal\n"
        "2026-04-15T00:00:00.000Z,src,M1,S1,filter_1,normal\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyUniquenessError):
        assert_registry_key_uniqueness(p)


def test_wilson_ci_shape():
    """Wilson CI is a proper subset of [0, 1] and centred around the point."""
    lo, hi = wilson_ci(9, 15)  # filter_6-like small n
    assert 0.0 <= lo < 0.6 < hi <= 1.0
    # Small-n produces a wide CI; sparse-category warning threshold used
    # downstream is 0.30.
    assert (hi - lo) > 0.30
    # Edge cases: matched=0, matched=total.
    lo0, hi0 = wilson_ci(0, 10)
    assert lo0 == 0.0 and 0.0 < hi0 < 1.0
    lo1, hi1 = wilson_ci(10, 10)
    # Floating-point Wilson at matched=total gives an upper edge indistinguishable
    # from 1.0 to within 2 ulps; assert closeness rather than exact equality.
    assert 0.0 < lo1 < 1.0 and abs(hi1 - 1.0) < 1e-12
    lo_null, hi_null = wilson_ci(0, 0)
    assert lo_null == 0.0 and hi_null == 0.0


def test_per_filter_coverage_with_ci_sparse_warning(registry_csv, outcomes_csv):
    """D17: small-n categories carry a sparse-category warning; the wide-CI
    filter_6 (n=15) is expected to raise it, and the large-n filter_2 is not."""
    rows = per_filter_coverage_with_ci(registry_csv, outcomes_csv)
    by_reason = {r["reason"]: r for r in rows}
    assert by_reason["filter_6"]["sparse_category_warning"] is True
    assert by_reason["filter_2"]["sparse_category_warning"] is False
    # Every row carries lower, upper, width and centre.
    for r in rows:
        assert 0.0 <= r["wilson_ci_lower_pct"] <= r["coverage_pct"] <= r["wilson_ci_upper_pct"] <= 100.0
        assert r["wilson_ci_width_pct"] == pytest.approx(
            r["wilson_ci_upper_pct"] - r["wilson_ci_lower_pct"]
        )
