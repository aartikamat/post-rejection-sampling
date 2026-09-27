"""tests/test_cli_verify.py - end-to-end verify against real deposit."""
from __future__ import annotations

import json

from prfs.cli import verify


def test_verify_reproduces_headline_numbers(tmp_path, registry_csv, outcomes_csv):
    report_out = tmp_path / "report.json"
    ledger_out = tmp_path / "ledger.csv"
    rc = verify([
        "--registry-csv", str(registry_csv),
        "--outcomes-csv", str(outcomes_csv),
        "--report-out", str(report_out),
        "--ledger-out", str(ledger_out),
    ])
    assert rc == 0
    r = json.loads(report_out.read_text(encoding="utf-8"))
    assert r["coverage"]["primary_matched"] == 1455
    assert r["coverage"]["primary_denominator"] == 2997
    assert round(r["coverage"]["primary_pct"], 2) == 48.55
    assert r["coverage"]["secondary_matched"] == 1641
    assert round(r["coverage"]["secondary_pct"], 2) == 54.75
    # D14: dual denominators for sample alignment.
    assert r["coverage"]["sample_aligned"] == 58407
    assert r["coverage"]["sample_denominator_full"] == 67000
    assert round(r["coverage"]["sample_pct_full"], 2) == 87.17
    assert r["coverage"]["sample_denominator_eligible"] == 58407
    assert round(r["coverage"]["sample_pct_eligible"], 2) == 100.00
    assert r["coverage"]["orphan_key_row_count"] == 8593
    # D14: registry key uniqueness surfaced in the report.
    assert r["registry_key_uniqueness"]["distinct_3f_keys"] == 2997
    assert r["registry_key_uniqueness"]["duplicate_3f_key_rows"] == 0
    # D18: time window reconciliation surfaced explicitly.
    assert r["time_window"]["analytic_follow_up_window_ms"] == 743_040_000
    assert 8.62 < r["time_window"]["observed_calendar_span_days"] < 8.64
    # Reason ledger.
    assert r["reason_ledger"]["conflict_emission_rows"] == 253
    assert r["reason_ledger"]["sum_orphan_key_count"] == 263
    # D17: per-filter CI columns present on every row.
    assert len(r["per_filter"]) == 7
    for row in r["per_filter"]:
        assert "wilson_ci_lower_pct" in row
        assert "wilson_ci_upper_pct" in row
        assert "wilson_ci_width_pct" in row
        assert "sparse_category_warning" in row
    # Ledger written and re-readable
    text = ledger_out.read_text(encoding="utf-8")
    assert text.count("\n") >= 253
