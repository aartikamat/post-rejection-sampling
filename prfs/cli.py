"""prfs.cli - entrypoints.

prfs-run    - drive one PRFS session against a configured oracle.
prfs-verify - verify a deposited Zenodo dataset against the formal spec;
              reproduces 1,455 / 1,641 / 263 / 253 and writes a
              provenance-linked report to stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import __version__, __spec_version__
from .clock import DELTA_MAX_MS, span_days
from .coverage import (
    assert_registry_key_uniqueness,
    per_filter_coverage,
    per_filter_coverage_with_ci,
    primary_coverage,
)
from .reason_ledger import build_ledger, write_ledger


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prfs-verify")
    parser.add_argument("--registry-csv", required=True)
    parser.add_argument("--outcomes-csv", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--ledger-out", required=False, default=None)
    args = parser.parse_args(argv)
    reg = Path(args.registry_csv)
    out = Path(args.outcomes_csv)
    # D14: assert 3-field key uniqueness explicitly and surface the report.
    uniq = assert_registry_key_uniqueness(reg)
    summary = primary_coverage(reg, out)
    per_f_ci = per_filter_coverage_with_ci(reg, out)
    ledger = build_ledger(reg, out)
    orphan_total = sum(r.orphan_key_count for r in ledger)

    # D18: derive OBSERVED calendar span from the registry timestamps and
    # report it separately from the analytic follow-up window DELTA_MAX_MS.
    import csv as _csv
    with reg.open("r", encoding="utf-8", newline="") as fh:
        _reg_ts = [row["timestamp"] for row in _csv.DictReader(fh)]
    observed_span_days = span_days(_reg_ts)

    report = {
        "prfs_version": __version__,
        "spec_version": __spec_version__,
        "input_registry_sha256": _sha256(reg),
        "input_outcomes_sha256": _sha256(out),
        "registry_key_uniqueness": {
            "total_rows": uniq.total_rows,
            "distinct_3f_keys": uniq.distinct_3f_keys,
            "duplicate_3f_key_rows": uniq.duplicate_3f_key_rows,
            "duplicate_3f_key_groups": uniq.duplicate_3f_key_groups,
            "conflicting_duplicate_2f_keys": uniq.conflicting_duplicate_2f_keys,
            "exact_duplicate_3f_key_groups": uniq.exact_duplicate_3f_key_groups,
        },
        "coverage": {
            "primary_matched": summary.primary_matched,
            "primary_denominator": summary.primary_denominator,
            "primary_pct": summary.primary_pct,
            "secondary_matched": summary.secondary_matched,
            "secondary_denominator": summary.secondary_denominator,
            "secondary_pct": summary.secondary_pct,
            "mint_matched": summary.mint_matched,
            "mint_denominator": summary.mint_denominator,
            "mint_pct": summary.mint_pct,
            "sample_aligned": summary.sample_aligned,
            # D14: dual denominators (full and eligible)
            "sample_denominator_full": summary.sample_denominator_full,
            "sample_pct_full": summary.sample_pct_full,
            "sample_denominator_eligible": summary.sample_denominator_eligible,
            "sample_pct_eligible": summary.sample_pct_eligible,
            "orphan_key_row_count": summary.orphan_key_row_count,
        },
        "time_window": {
            "analytic_follow_up_window_ms": DELTA_MAX_MS,
            "analytic_follow_up_window_days_exact": DELTA_MAX_MS / 86_400_000.0,
            "observed_calendar_span_days": observed_span_days,
            "note": (
                "analytic_follow_up_window is the DELTA_MAX_MS constant from "
                "FORMAL_SPEC 2 (8.6 days exactly); observed_calendar_span_days "
                "is the max-minus-min of the registry timestamps (approximately "
                "8.63 days on this deposit). The two quantities are distinct."
            ),
        },
        # D17: per-filter coverage with Wilson 95% binomial CI on every row.
        "per_filter": per_f_ci,
        "reason_ledger": {
            "conflict_emission_rows": len(ledger),
            "sum_orphan_key_count": orphan_total,
        },
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.ledger_out:
        write_ledger(ledger, Path(args.ledger_out))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def run(argv: list[str] | None = None) -> int:
    """Live-run entrypoint stub. This reference implementation ships
    verify as the executable path; run is a documented no-op unless a
    real OracleAdapter is wired in via config."""
    parser = argparse.ArgumentParser(prog="prfs-run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(json.dumps({
        "status": "not_configured",
        "message": (
            "prfs-run requires a project-provided OracleAdapter binding. "
            "Use prfs-verify for the deposited-dataset validation path."
        ),
        "config_path": args.config,
        "out_path": args.out,
    }))
    return 2


if __name__ == "__main__":
    sys.exit(verify())
