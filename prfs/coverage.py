"""prfs.coverage, the coverage definitions.

Implements FORMAL_SPEC section 4. The primary and secondary numbers must
regenerate 1,455 / 2,997 = 48.55 percent and 1,641 / 2,997 = 54.75 percent
on the frozen Zenodo dataset (10.5281/zenodo.20043516).

Design changes documented in the D14 remediation:

1. Timestamp identity is decided on the CANONICAL epoch-millisecond form
   returned by prfs.clock.to_epoch_ms, not on the raw ISO literal. The raw
   literal is retained on every row for provenance. On the deposit both
   forms agree because every literal already has the canonical shape
   %Y-%m-%dT%H:%M:%S.%fZ; the canonical comparison guards against
   future mixed-precision inputs.

2. Registry three-field key uniqueness is asserted explicitly. The check
   reports total rows, distinct 3-field keys, duplicate 3-field key count
   (rows that share their 3-field key with at least one other row),
   conflicting duplicate count (2-field key shared but reasons differ), and
   the exact duplicate count (all three fields identical). If any duplicate
   is unexpectedly present the check raises KeyUniquenessError; the caller
   is expected to halt rather than infer uniqueness from a count coincidence.

3. Sample-level alignment is reported against BOTH denominators: the full
   67,000 outcome-row denominator and the eligible-after-excluding-orphan-
   keys denominator. Both are surfaced in the CoverageSummary; downstream
   consumers must state which is being cited.

4. The cross-check aggregator (formerly labelled "independent oracle")
   shares the same set-intersection logical framework as the primary code
   path in oracle/oracle_aggregate.py; it is a SECONDARY implementation
   with a distinct entry point and distinct accumulation strategy but is
   not language-independent. See PRFS_REFERENCE_IMPLEMENTATION_README.md
   for the honest independence statement.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

from prfs.clock import to_epoch_ms


@dataclass(frozen=True)
class CoverageSummary:
    primary_matched: int
    primary_denominator: int
    primary_pct: float
    secondary_matched: int
    secondary_denominator: int
    secondary_pct: float
    mint_matched: int
    mint_denominator: int
    mint_pct: float
    # Sample-level alignment reported against BOTH denominators (D14).
    sample_aligned: int
    sample_denominator_full: int          # full outcome row count (67,000)
    sample_pct_full: float                # % of all outcome rows
    sample_denominator_eligible: int      # excludes orphan-key rows
    sample_pct_eligible: float            # % of eligible outcome rows
    orphan_key_row_count: int             # outcome rows on orphan 3-field keys


@dataclass(frozen=True)
class PerFilterCoverage:
    reason: str
    reg_events: int
    matched: int
    coverage_pct: float


@dataclass(frozen=True)
class KeyUniquenessReport:
    total_rows: int
    distinct_3f_keys: int
    duplicate_3f_key_rows: int       # rows whose 3-field key appears more than once
    duplicate_3f_key_groups: int     # number of DISTINCT 3-field keys that appear more than once
    conflicting_duplicate_2f_keys: int  # (mint, ts) shared but reason differs
    exact_duplicate_3f_key_groups: int  # all three fields identical (exact duplicate row group)


class KeyUniquenessError(RuntimeError):
    """Raised when the registry violates the 3-field-key uniqueness contract."""


def _read_rej_keys(
    registry_csv: Path,
) -> tuple[
    set[tuple[str, int, str]],
    set[tuple[str, int]],
    dict[str, int],
    set[str],
    KeyUniquenessReport,
]:
    """Read rejections.csv. Return
    (canonical 3-field key set, canonical 2-field key set,
    per-reason denominator counts, mint set, key-uniqueness report).

    Canonicalisation: the timestamp is folded to its epoch-ms integer via
    prfs.clock.to_epoch_ms; the raw literal is dropped from the key but the
    canonical value is byte-identical to the raw literal on this deposit.
    """
    keys3: set[tuple[str, int, str]] = set()
    keys2: set[tuple[str, int]] = set()
    per_reason: dict[str, int] = {}
    mints: set[str] = set()

    # Uniqueness bookkeeping over the RAW (pre-set) triple stream.
    total_rows = 0
    triple_counts: dict[tuple[str, str, str], int] = {}
    reasons_by_pair: dict[tuple[str, str], set[str]] = {}

    with Path(registry_csv).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total_rows += 1
            mint = row["mint"]
            ts = row["timestamp"]
            reason = row["reason"]
            triple = (mint, ts, reason)
            triple_counts[triple] = triple_counts.get(triple, 0) + 1
            reasons_by_pair.setdefault((mint, ts), set()).add(reason)

            ts_ms = to_epoch_ms(ts)
            keys3.add((mint, ts_ms, reason))
            keys2.add((mint, ts_ms))
            per_reason[reason] = per_reason.get(reason, 0) + 1
            mints.add(mint)

    distinct_3f = len(triple_counts)
    duplicate_3f_key_groups = sum(1 for c in triple_counts.values() if c > 1)
    duplicate_3f_key_rows = sum(c for c in triple_counts.values() if c > 1)
    exact_duplicate_3f_key_groups = duplicate_3f_key_groups  # 3-field triple appearing >1 time
    conflicting_duplicate_2f_keys = sum(
        1 for rs in reasons_by_pair.values() if len(rs) > 1
    )

    report = KeyUniquenessReport(
        total_rows=total_rows,
        distinct_3f_keys=distinct_3f,
        duplicate_3f_key_rows=duplicate_3f_key_rows,
        duplicate_3f_key_groups=duplicate_3f_key_groups,
        conflicting_duplicate_2f_keys=conflicting_duplicate_2f_keys,
        exact_duplicate_3f_key_groups=exact_duplicate_3f_key_groups,
    )
    return keys3, keys2, per_reason, mints, report


def _read_out_keys(
    outcomes_csv: Path,
) -> tuple[
    set[tuple[str, int, str]],
    set[tuple[str, int]],
    set[str],
    list[tuple[str, int, str]],
]:
    """Read rejection_outcomes.csv. Return
    (canonical 3-field key set, canonical 2-field key set, mint set,
    list of canonical (mint, epoch_ms, reason) per sample row).
    """
    keys3: set[tuple[str, int, str]] = set()
    keys2: set[tuple[str, int]] = set()
    mints: set[str] = set()
    rows: list[tuple[str, int, str]] = []
    with Path(outcomes_csv).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mint = row["mint"]
            ts = row["rejectTs"]
            reason = row["rejectReason"]
            ts_ms = to_epoch_ms(ts)
            keys3.add((mint, ts_ms, reason))
            keys2.add((mint, ts_ms))
            mints.add(mint)
            rows.append((mint, ts_ms, reason))
    return keys3, keys2, mints, rows


def assert_registry_key_uniqueness(
    registry_csv: Path,
    *,
    allow_duplicate_3f_key_groups: int = 0,
) -> KeyUniquenessReport:
    """Read the registry, run the uniqueness audit, halt if the observed
    duplicate-3f-key-group count exceeds allow_duplicate_3f_key_groups.
    On the deposit the expected count is zero and this function raises
    KeyUniquenessError with the exact report if any duplicate is present.
    """
    _, _, _, _, report = _read_rej_keys(registry_csv)
    if report.duplicate_3f_key_groups > allow_duplicate_3f_key_groups:
        raise KeyUniquenessError(
            "registry 3-field key not unique: "
            f"total_rows={report.total_rows} distinct_3f_keys={report.distinct_3f_keys} "
            f"duplicate_3f_key_groups={report.duplicate_3f_key_groups} "
            f"duplicate_3f_key_rows={report.duplicate_3f_key_rows} "
            f"conflicting_duplicate_2f_keys={report.conflicting_duplicate_2f_keys} "
            f"exact_duplicate_3f_key_groups={report.exact_duplicate_3f_key_groups}"
        )
    return report


def primary_coverage(registry_csv: Path, outcomes_csv: Path) -> CoverageSummary:
    """Spec section 4.1 through 4.5 combined summary using the three-field
    canonical key as primary. Sample alignment is reported against both
    denominators (see FORMAL_SPEC 4.5 and the D14 remediation note)."""
    # Fail fast if the registry violates the 3-field-key uniqueness contract.
    assert_registry_key_uniqueness(registry_csv)

    rej_k3, rej_k2, _per_reason, rej_mints, _report = _read_rej_keys(registry_csv)
    out_k3, out_k2, out_mints, out_rows = _read_out_keys(outcomes_csv)
    primary_matched = len(rej_k3 & out_k3)
    secondary_matched = len(rej_k2 & out_k2)
    denom_events = len(rej_k3)
    denom_events2 = len(rej_k2)
    mint_matched = len(rej_mints & out_mints)
    mint_denominator = len(rej_mints)

    # Sample-level alignment:
    # numerator = outcome rows whose canonical 3-field key matches a
    # registry 3-field key.
    aligned_samples = sum(1 for k in out_rows if k in rej_k3)

    # Denominator A (D14 explicit): the FULL outcome row count. This is the
    # 67,000 headline. Reported as sample_pct_full.
    sample_denominator_full = len(out_rows)

    # Denominator B (D14 explicit): the ELIGIBLE denominator, defined as the
    # full outcome row count MINUS the rows that attach to an orphan
    # 3-field outcome key (an outcome key that has no matching registry
    # 3-field key). These orphan-key rows cannot align under the reason-
    # aware key by construction; excluding them gives the "of the outcome
    # rows that COULD have aligned, how many did" view.
    orphan_out_k3 = out_k3 - rej_k3
    orphan_key_row_count = sum(1 for k in out_rows if k in orphan_out_k3)
    sample_denominator_eligible = sample_denominator_full - orphan_key_row_count

    return CoverageSummary(
        primary_matched=primary_matched,
        primary_denominator=denom_events,
        primary_pct=(primary_matched / denom_events * 100.0) if denom_events else 0.0,
        secondary_matched=secondary_matched,
        secondary_denominator=denom_events2,
        secondary_pct=(secondary_matched / denom_events2 * 100.0) if denom_events2 else 0.0,
        mint_matched=mint_matched,
        mint_denominator=mint_denominator,
        mint_pct=(mint_matched / mint_denominator * 100.0) if mint_denominator else 0.0,
        sample_aligned=aligned_samples,
        sample_denominator_full=sample_denominator_full,
        sample_pct_full=(aligned_samples / sample_denominator_full * 100.0)
                        if sample_denominator_full else 0.0,
        sample_denominator_eligible=sample_denominator_eligible,
        sample_pct_eligible=(aligned_samples / sample_denominator_eligible * 100.0)
                            if sample_denominator_eligible else 0.0,
        orphan_key_row_count=orphan_key_row_count,
    )


def per_filter_coverage(
    registry_csv: Path, outcomes_csv: Path
) -> list[PerFilterCoverage]:
    """Spec section 4.3 per-filter coverage. Always returns all seven rows
    in ascending filter-label order."""
    assert_registry_key_uniqueness(registry_csv)
    rej_k3, _rej_k2, per_reason, _, _ = _read_rej_keys(registry_csv)
    out_k3, _out_k2, _out_mints, _out_rows = _read_out_keys(outcomes_csv)
    matched_by_reason: dict[str, int] = {r: 0 for r in per_reason}
    for k in rej_k3:
        if k in out_k3:
            matched_by_reason[k[2]] = matched_by_reason.get(k[2], 0) + 1
    out: list[PerFilterCoverage] = []
    for reason in sorted(per_reason.keys()):
        m = matched_by_reason.get(reason, 0)
        total = per_reason[reason]
        pct = (m / total * 100.0) if total else 0.0
        out.append(PerFilterCoverage(reason=reason, reg_events=total, matched=m, coverage_pct=pct))
    return out


def secondary_coverage_diagnostic(
    registry_csv: Path, outcomes_csv: Path
) -> tuple[int, int]:
    """Spec section 4.2 secondary integrity diagnostic; returns (matched_2f,
    registry_3f_key_denominator)."""
    assert_registry_key_uniqueness(registry_csv)
    rej_k3, rej_k2, _per_reason, _, _ = _read_rej_keys(registry_csv)
    _out_k3, out_k2, _om, _or = _read_out_keys(outcomes_csv)
    return len(rej_k2 & out_k2), len(rej_k3)


def wilson_ci(matched: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson-score binomial confidence interval for a per-filter coverage
    proportion. Default z = 1.959964 gives the two-sided 95 percent interval.
    Returns (lower, upper) as fractions in [0, 1]. Defined for total > 0."""
    if total <= 0:
        return (0.0, 0.0)
    p = matched / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denom
    half = (z * ((p * (1.0 - p) / total + z2 / (4.0 * total * total)) ** 0.5)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def per_filter_coverage_with_ci(
    registry_csv: Path, outcomes_csv: Path
) -> list[dict]:
    """Per-filter coverage with a Wilson 95 percent binomial CI on every row.
    Flags rows where the CI width exceeds 0.30 (sparse-category warning)."""
    base = per_filter_coverage(registry_csv, outcomes_csv)
    out: list[dict] = []
    for row in base:
        lo, hi = wilson_ci(row.matched, row.reg_events)
        width = hi - lo
        out.append({
            "reason": row.reason,
            "reg_events": row.reg_events,
            "matched": row.matched,
            "coverage_pct": row.coverage_pct,
            "wilson_ci_lower_pct": lo * 100.0,
            "wilson_ci_upper_pct": hi * 100.0,
            "wilson_ci_width_pct": width * 100.0,
            "sparse_category_warning": width > 0.30,
        })
    return out
