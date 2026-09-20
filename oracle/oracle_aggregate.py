"""Cross-check aggregator (Phase B secondary implementation).

D14 rename: this module was previously described as an "independent oracle";
that phrasing overstated its independence. It uses a distinct code path
(csv.reader with explicit column-index lookup, single-pass streaming, and
its own accumulator structures) from prfs.coverage / prfs.reason_ledger and
does NOT import them, so it will catch code-path bugs in the primary path.
It nevertheless shares the same logical set-intersection framework and the
same Python runtime, so agreement with the primary implementation is
CROSS-VERIFICATION, not language- or framework-independent verification.

Timestamp identity is decided on the canonical epoch-ms form via
prfs.clock.to_epoch_ms, consistent with prfs.coverage (D14). The prfs.clock
module is imported here for the canonicalisation helper only; it does not
introduce any primary-code-path dependency.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prfs.clock import to_epoch_ms


def _index_columns(header: list[str], required: list[str]) -> list[int]:
    idx = []
    lower = [h.strip().lower() for h in header]
    for name in required:
        try:
            idx.append(lower.index(name.lower()))
        except ValueError:
            raise KeyError(f"required column {name!r} missing from {header!r}")
    return idx


def _stream(path: Path, required: list[str]):
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx = _index_columns(header, required)
        for row in reader:
            yield [row[i] for i in idx]


def aggregate(registry_csv: Path, outcomes_csv: Path) -> dict[str, Any]:
    """Independent single-pass aggregation of all headline numbers."""
    # ---- registry -------------------------------------------------------
    reg_keys3: set[tuple[str, int, str]] = set()
    reg_keys2: set[tuple[str, int]] = set()
    reg_mints: set[str] = set()
    reg_per_reason: Counter = Counter()
    reg_reasons_by_k2: dict[tuple[str, int], set[str]] = {}
    reg_rows_total = 0
    for mint, ts, reason in _stream(registry_csv, ["mint", "timestamp", "reason"]):
        reg_rows_total += 1
        ts_ms = to_epoch_ms(ts)
        reg_keys3.add((mint, ts_ms, reason))
        reg_keys2.add((mint, ts_ms))
        reg_mints.add(mint)
        reg_per_reason[reason] += 1
        reg_reasons_by_k2.setdefault((mint, ts_ms), set()).add(reason)

    # ---- outcomes -------------------------------------------------------
    out_keys3: set[tuple[str, int, str]] = set()
    out_keys2: set[tuple[str, int]] = set()
    out_mints: set[str] = set()
    out_reasons_by_k2: dict[tuple[str, int], set[str]] = {}
    aligned_samples = 0
    total_samples = 0
    for mint, ts, reason in _stream(
        outcomes_csv, ["mint", "rejectTs", "rejectReason"]
    ):
        total_samples += 1
        ts_ms = to_epoch_ms(ts)
        out_keys3.add((mint, ts_ms, reason))
        out_keys2.add((mint, ts_ms))
        out_mints.add(mint)
        out_reasons_by_k2.setdefault((mint, ts_ms), set()).add(reason)
        if (mint, ts_ms, reason) in reg_keys3:
            aligned_samples += 1

    # ---- coverage -------------------------------------------------------
    primary_matched = len(reg_keys3 & out_keys3)
    secondary_matched = len(reg_keys2 & out_keys2)
    denom = len(reg_keys3)
    denom2 = len(reg_keys2)

    # ---- per-filter -----------------------------------------------------
    per_filter_matched: Counter = Counter()
    for m, t, r in reg_keys3:
        if (m, t, r) in out_keys3:
            per_filter_matched[r] += 1
    per_filter = []
    for reason in sorted(reg_per_reason.keys()):
        total = reg_per_reason[reason]
        m = per_filter_matched[reason]
        per_filter.append({
            "reason": reason,
            "reg_events": total,
            "matched": m,
            "coverage_pct": (m / total * 100.0) if total else 0.0,
        })

    # ---- reason ledger --------------------------------------------------
    conflict_rows = 0
    orphan_sum = 0
    for k2 in sorted(set(reg_reasons_by_k2.keys()) & set(out_reasons_by_k2.keys())):
        rr = reg_reasons_by_k2[k2]
        oo = out_reasons_by_k2[k2]
        if rr == oo:
            continue
        conflict_rows += 1
        orphan_sum += len(oo - rr)

    return {
        "registry_rows_total": reg_rows_total,
        "outcome_rows_total": total_samples,
        "distinct_registry_keys_3f": len(reg_keys3),
        "distinct_registry_keys_2f": len(reg_keys2),
        "distinct_outcome_keys_3f": len(out_keys3),
        "distinct_outcome_keys_2f": len(out_keys2),
        "primary_matched": primary_matched,
        "primary_denominator": denom,
        "primary_pct": (primary_matched / denom * 100.0) if denom else 0.0,
        "secondary_matched": secondary_matched,
        "secondary_denominator": denom2,
        "secondary_pct": (secondary_matched / denom2 * 100.0) if denom2 else 0.0,
        "mint_matched": len(reg_mints & out_mints),
        "mint_denominator": len(reg_mints),
        "mint_pct": (len(reg_mints & out_mints) / len(reg_mints) * 100.0)
                    if reg_mints else 0.0,
        "sample_aligned": aligned_samples,
        "sample_denominator": total_samples,
        "sample_pct": (aligned_samples / total_samples * 100.0) if total_samples else 0.0,
        "per_filter": per_filter,
        "reason_conflict_rows": conflict_rows,
        "reason_orphan_key_sum": orphan_sum,
    }


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(prog="oracle-aggregate")
    p.add_argument("--registry-csv", required=True)
    p.add_argument("--outcomes-csv", required=True)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()
    result = aggregate(Path(args.registry_csv), Path(args.outcomes_csv))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
