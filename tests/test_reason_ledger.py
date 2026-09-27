"""tests/test_reason_ledger.py, reproduces the 253 / 263 numbers.

Includes D15 set-semantics test: within-key duplicate SAME-reason sample
rows on the outcome side must collapse and NOT trigger a conflict emission.
Truly disagreeing reasons at a shared (mint, timestamp) key MUST trigger
one.
"""
from __future__ import annotations

from prfs.reason_ledger import build_ledger, write_ledger


def test_conflict_emission_count_is_253(registry_csv, outcomes_csv):
    rows = build_ledger(registry_csv, outcomes_csv)
    assert len(rows) == 253


def test_orphan_key_count_sums_to_263(registry_csv, outcomes_csv):
    rows = build_ledger(registry_csv, outcomes_csv)
    assert sum(r.orphan_key_count for r in rows) == 263


def test_no_out_only_reason_rows(registry_csv, outcomes_csv):
    rows = build_ledger(registry_csv, outcomes_csv)
    assert sum(1 for r in rows if r.conflict_type == "OUT_ONLY_REASON") == 0


def test_default_disposition_is_keep_registry_as_primary(registry_csv, outcomes_csv):
    rows = build_ledger(registry_csv, outcomes_csv)
    dispositions = {r.disposition for r in rows}
    assert dispositions == {"KEEP_REGISTRY_AS_PRIMARY"}


def test_ledger_csv_write_and_reread(tmp_path, registry_csv, outcomes_csv):
    rows = build_ledger(registry_csv, outcomes_csv)
    out = tmp_path / "ledger.csv"
    write_ledger(rows, out)
    text = out.read_text(encoding="utf-8")
    # header + 253 rows = 254 lines (final newline preserved)
    non_empty = [l for l in text.splitlines() if l]
    assert len(non_empty) == 254
    assert non_empty[0].startswith("event_key,mint,timestamp_utc,conflict_type")


def test_registry_reasons_never_overwritten(registry_csv, outcomes_csv):
    """The ledger annotates disagreement, never rewrites source reasons."""
    rows = build_ledger(registry_csv, outcomes_csv)
    for r in rows:
        assert r.registry_reasons  # non-empty for SET_MISMATCH
        assert r.outcome_reasons   # non-empty for SET_MISMATCH
        # if any reg reason is also in outcome, it shouldn't be in reg_only
        assert set(r.shared_reasons) == set(r.registry_reasons) & set(r.outcome_reasons)
        assert set(r.registry_only_reasons) == set(r.registry_reasons) - set(r.outcome_reasons)
        assert set(r.outcome_only_reasons) == set(r.outcome_reasons) - set(r.registry_reasons)


def test_confidence_labels_are_two_level(registry_csv, outcomes_csv):
    """D16: the code emits only HIGH and MEDIUM. LOW is not defined."""
    rows = build_ledger(registry_csv, outcomes_csv)
    labels = {r.confidence for r in rows}
    assert labels.issubset({"HIGH", "MEDIUM"})
    assert "LOW" not in labels


def test_reason_set_semantics_within_key_duplicates_collapse(tmp_path):
    """D15 core assertion: multiple outcome sample rows carrying the SAME
    rejectReason at the SAME (mint, rejectTs) key MUST NOT be counted as a
    multiplicity signal. They collapse to a single distinct reason and, when
    the registry side agrees on that reason, MUST NOT emit a conflict row.
    """
    reg = tmp_path / "reg.csv"
    out = tmp_path / "out.csv"
    reg.write_text(
        "timestamp,source,mint,symbol,reason,timeSlot\n"
        "2026-04-15T00:00:00.000Z,src,M1,S1,filter_1,normal\n"
        "2026-04-15T00:01:00.000Z,src,M2,S2,filter_2,normal\n",
        encoding="utf-8",
    )
    # M1 shares filter_1 across 3 sample rows (scheduled repetitions) - should
    # collapse under set semantics, NO conflict.
    # M2 has DISAGREEING reasons across sample rows (filter_2 registry vs
    # filter_3 in one sample and filter_2 in another) - SHOULD emit a conflict.
    out.write_text(
        "sampleTs,mint,symbol,rejectReason,rejectTs,ageMin,priceUsd,liquidity,volume24h,dexId,pairAddress\n"
        "2026-04-15T00:05:00.000Z,M1,S1,filter_1,2026-04-15T00:00:00.000Z,5,1.0,10,10,dx,P\n"
        "2026-04-15T00:15:00.000Z,M1,S1,filter_1,2026-04-15T00:00:00.000Z,15,1.0,10,10,dx,P\n"
        "2026-04-15T01:00:00.000Z,M1,S1,filter_1,2026-04-15T00:00:00.000Z,60,1.0,10,10,dx,P\n"
        "2026-04-15T00:06:00.000Z,M2,S2,filter_3,2026-04-15T00:01:00.000Z,5,1.0,10,10,dx,P\n"
        "2026-04-15T00:16:00.000Z,M2,S2,filter_2,2026-04-15T00:01:00.000Z,15,1.0,10,10,dx,P\n",
        encoding="utf-8",
    )
    rows = build_ledger(reg, out)
    # Exactly one conflict emission (M2), none for M1.
    assert len(rows) == 1
    r = rows[0]
    assert r.mint == "M2"
    assert set(r.registry_reasons) == {"filter_2"}
    assert set(r.outcome_reasons) == {"filter_2", "filter_3"}
    assert set(r.outcome_only_reasons) == {"filter_3"}


def test_reason_set_semantics_disagreement_emits_conflict(tmp_path):
    """D15 complement: truly disagreeing reasons at a shared (mint, ts) key
    MUST emit exactly one conflict row per key."""
    reg = tmp_path / "reg.csv"
    out = tmp_path / "out.csv"
    reg.write_text(
        "timestamp,source,mint,symbol,reason,timeSlot\n"
        "2026-04-15T00:00:00.000Z,src,M1,S1,filter_1,normal\n",
        encoding="utf-8",
    )
    out.write_text(
        "sampleTs,mint,symbol,rejectReason,rejectTs,ageMin,priceUsd,liquidity,volume24h,dexId,pairAddress\n"
        "2026-04-15T00:05:00.000Z,M1,S1,filter_5,2026-04-15T00:00:00.000Z,5,1.0,10,10,dx,P\n",
        encoding="utf-8",
    )
    rows = build_ledger(reg, out)
    assert len(rows) == 1
    r = rows[0]
    assert set(r.registry_reasons) == {"filter_1"}
    assert set(r.outcome_reasons) == {"filter_5"}
    # active contradiction - both reg_only and out_only non-empty
    assert r.confidence == "HIGH"


def test_timestamp_normalisation_string_variability(tmp_path):
    """D14: mixed integer-second and millisecond ISO literals for the SAME
    UTC instant must be canonicalised and treated as the same key. On the
    deposit no variability exists, but the pipeline must handle it if
    introduced."""
    reg = tmp_path / "reg.csv"
    out = tmp_path / "out.csv"
    reg.write_text(
        "timestamp,source,mint,symbol,reason,timeSlot\n"
        "2026-04-15T00:00:00.000Z,src,M1,S1,filter_1,normal\n",
        encoding="utf-8",
    )
    # Outcome side uses the integer-second form; the same UTC instant.
    out.write_text(
        "sampleTs,mint,symbol,rejectReason,rejectTs,ageMin,priceUsd,liquidity,volume24h,dexId,pairAddress\n"
        "2026-04-15T00:05:00.000Z,M1,S1,filter_1,2026-04-15T00:00:00Z,5,1.0,10,10,dx,P\n",
        encoding="utf-8",
    )
    rows = build_ledger(reg, out)
    # Reason matches, so no conflict emission.
    assert len(rows) == 0
