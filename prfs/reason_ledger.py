"""prfs.reason_ledger, build and query the REASON_CONFLICT_LEDGER.

Implements FORMAL_SPEC section 5. Reproduces the 253 conflict emissions and
sum(orphan_key_count) = 263 count computed in Phase 1.

Semantics (D15 decision, 2026-08-23):

Reason comparison at a given (mint, timestamp) key is a SET-of-distinct-
reasons comparison, not a multiset comparison. Rationale (documented in
decision_docs/SET_VS_MULTISET_DECISION.md):

  - On the registry side the deposit contains no repeated (mint, timestamp)
    keys at all, so on that side "multiset" reduces to "set" trivially.

  - On the outcome side each row is a SCHEDULED SAMPLE, not a rejection
    event. The reference schedule Schedule_fix = {5, 15, 60, 240, 1440}
    minutes generates up to five sample rows per (mint, rejectTs,
    rejectReason) triple, and every one of those sample rows carries a copy
    of the reason attributed at emission time. A reason that appears N
    times at a given (mint, rejectTs) key on the outcome side represents N
    scheduled samples of the SAME logical attribution, not N independent
    attribution decisions. Treating that as a multiplicity signal would
    falsely flag as a conflict every case where the outcome side has more
    than one sample row for a reason on which both sides agree.

The intent is enforced by test_reason_set_semantics in
tests/test_reason_ledger.py.

Timestamp comparison is on the canonical epoch-ms axis via prfs.clock,
consistent with prfs.coverage.

Confidence labels (D16 remediation, 2026-08-23): the code emits HIGH and
MEDIUM only; the Literal has been narrowed accordingly. The two branches
are the ones the FORMAL_SPEC 5.2 disposition rule actually distinguishes; a
prospective LOW category was not defined in the spec and is not added
post-hoc.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Iterable, Literal

from prfs.clock import to_epoch_ms

Disposition = Literal[
    "KEEP_REGISTRY_AS_PRIMARY",
    "KEEP_OUTCOME_AS_PRIMARY",
    "MULTI_FILTER_CO_FIRING",
    "ANONYMISATION_MAP_DRIFT",
    "UNRESOLVED",
]
Confidence = Literal["MEDIUM", "HIGH"]
ConflictType = Literal[
    "REG_ONLY_REASON",
    "OUT_ONLY_REASON",
    "SET_MISMATCH",
]

_HEADER = (
    "event_key,mint,timestamp_utc,conflict_type,registry_reasons,outcome_reasons,"
    "registry_row_count,outcome_sample_count,registry_only_reasons,outcome_only_reasons,"
    "shared_reasons,orphan_key_count,conflict_emission_flag,evidence,disposition,confidence,notes"
)


@dataclass(frozen=True)
class LedgerRow:
    event_key: str
    mint: str
    timestamp_utc: str  # ORIGINAL literal, preserved for provenance
    conflict_type: str
    registry_reasons: tuple
    outcome_reasons: tuple
    registry_row_count: int
    outcome_sample_count: int
    registry_only_reasons: tuple
    outcome_only_reasons: tuple
    shared_reasons: tuple
    orphan_key_count: int
    conflict_emission_flag: int
    evidence: str
    disposition: str
    confidence: str
    notes: str


def _read_registry(
    path: Path,
) -> tuple[
    dict[tuple[str, int], set[str]],
    dict[tuple[str, int], int],
    dict[tuple[str, int], str],
]:
    """Read the registry, indexing by the CANONICAL (mint, epoch_ms) key.
    Returns (distinct_reason_set_by_key, row_count_by_key,
    original_ts_literal_by_key) where the original literal is preserved for
    audit output."""
    reasons_by_key: dict[tuple[str, int], set[str]] = defaultdict(set)
    rows_by_key: dict[tuple[str, int], int] = defaultdict(int)
    original_ts: dict[tuple[str, int], str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ts_ms = to_epoch_ms(row["timestamp"])
            k = (row["mint"], ts_ms)
            reasons_by_key[k].add(row["reason"])
            rows_by_key[k] += 1
            original_ts.setdefault(k, row["timestamp"])
    return reasons_by_key, rows_by_key, original_ts


def _read_outcomes(
    path: Path,
) -> tuple[
    dict[tuple[str, int], set[str]],
    dict[tuple[str, int], int],
    dict[tuple[str, int], str],
]:
    """Read the outcomes, indexing by the CANONICAL (mint, epoch_ms) key.
    Distinct-reasons-only set semantics per D15 decision.
    """
    reasons_by_key: dict[tuple[str, int], set[str]] = defaultdict(set)
    rows_by_key: dict[tuple[str, int], int] = defaultdict(int)
    original_ts: dict[tuple[str, int], str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ts_ms = to_epoch_ms(row["rejectTs"])
            k = (row["mint"], ts_ms)
            reasons_by_key[k].add(row["rejectReason"])
            rows_by_key[k] += 1
            original_ts.setdefault(k, row["rejectTs"])
    return reasons_by_key, rows_by_key, original_ts


def build_ledger(
    registry_csv: Path, outcomes_csv: Path
) -> list[LedgerRow]:
    """Enumerate every (mint, ts) where the set of distinct reasons differs
    between the two files; emit one ledger row each."""
    reg_reasons, reg_rows, reg_orig_ts = _read_registry(registry_csv)
    out_reasons, out_rows, _out_orig_ts = _read_outcomes(outcomes_csv)
    rows: list[LedgerRow] = []
    # Only (mint, ts) keys where BOTH sides record data qualify as a
    # reason-attribution conflict per spec section 5.2. Registry-only
    # (no outcome observation at all) and outcome-only (no registry entry
    # at all) are separate diagnostics handled by coverage.py, not by the
    # reason ledger. The Phase 1 target of 253 conflict emissions is the
    # count under this rule.
    keys_both = sorted(set(reg_reasons.keys()) & set(out_reasons.keys()))
    for k in keys_both:
        r = reg_reasons.get(k, set())
        o = out_reasons.get(k, set())
        if r == o:
            continue
        reg_only = r - o
        out_only = o - r
        shared = r & o
        orphan = len(out_only)
        ctype = "SET_MISMATCH"
        evidence = (
            f"reg={sorted(r)} out={sorted(o)} "
            f"reg_only={sorted(reg_only)} out_only={sorted(out_only)}"
        )
        # Two-level confidence, matching what the code actually emits
        # (D16 remediation). HIGH when both sides have reasons the other
        # lacks (active contradiction); MEDIUM when one side is a proper
        # subset of the other (asymmetric absence).
        conf = "HIGH" if reg_only and out_only else "MEDIUM"
        ts_original = reg_orig_ts.get(k, "")
        rows.append(
            LedgerRow(
                event_key=f"{k[0]}|{ts_original}",
                mint=k[0],
                timestamp_utc=ts_original,
                conflict_type=ctype,
                registry_reasons=tuple(sorted(r)),
                outcome_reasons=tuple(sorted(o)),
                registry_row_count=reg_rows.get(k, 0),
                outcome_sample_count=out_rows.get(k, 0),
                registry_only_reasons=tuple(sorted(reg_only)),
                outcome_only_reasons=tuple(sorted(out_only)),
                shared_reasons=tuple(sorted(shared)),
                orphan_key_count=orphan,
                conflict_emission_flag=1,
                evidence=evidence,
                disposition="KEEP_REGISTRY_AS_PRIMARY",
                confidence=conf,
                notes="",
            )
        )
    return rows


def write_ledger(rows: Iterable[LedgerRow], out_csv: Path) -> None:
    """CSV writer with the frozen header from FORMAL_SPEC section 5."""
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_HEADER.split(","))
        for r in rows:
            w.writerow([
                r.event_key,
                r.mint,
                r.timestamp_utc,
                r.conflict_type,
                "|".join(r.registry_reasons),
                "|".join(r.outcome_reasons),
                r.registry_row_count,
                r.outcome_sample_count,
                "|".join(r.registry_only_reasons),
                "|".join(r.outcome_only_reasons),
                "|".join(r.shared_reasons),
                r.orphan_key_count,
                r.conflict_emission_flag,
                r.evidence,
                r.disposition,
                r.confidence,
                r.notes,
            ])


def partition_by_reason_source(
    rows: list, source: Literal["registry", "outcome", "union"]
) -> dict[str, int]:
    """Spec section 5.2 sensitivity: for a per-filter tally, partition by the
    chosen reason source. rows is an iterable of dicts with keys
    ('reason_reg', 'reason_out'). Returns {reason: count}."""
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        if source == "registry":
            for x in r.get("reason_reg", []):
                out[x] += 1
        elif source == "outcome":
            for x in r.get("reason_out", []):
                out[x] += 1
        elif source == "union":
            for x in set(r.get("reason_reg", [])) | set(r.get("reason_out", [])):
                out[x] += 1
        else:
            raise ValueError(f"unknown source {source}")
    return dict(out)
