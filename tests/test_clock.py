"""tests/test_clock.py, time model (spec section 2).

D14 additions: canonical epoch-ms comparison, parse-failure surfacing, and
mixed integer-second / millisecond equivalence at the canonical layer.
"""
from __future__ import annotations

import pytest

from prfs.clock import (
    DELTA_MAX_MS,
    ClockParseError,
    ParsedTs,
    add_minutes,
    add_ms,
    canonicalise,
    diff_ms,
    format_iso_ms,
    parse_iso_ms,
    raw_eq,
    span_days,
    to_epoch_ms,
    within_window,
)


def test_delta_max_is_exactly_8_6_days():
    """DELTA_MAX_MS is the analytic follow-up window from FORMAL_SPEC 2. It
    is 8.6 days expressed exactly in integer milliseconds. Distinct from the
    OBSERVED calendar span of the deposit (see span_days on the registry
    timestamps, approximately 8.63 days)."""
    assert DELTA_MAX_MS == 743_040_000
    assert DELTA_MAX_MS / 86_400_000 == 8.6


def test_parse_iso_ms_roundtrip():
    ts = "2026-04-15T12:34:56.789Z"
    dt = parse_iso_ms(ts)
    assert format_iso_ms(dt) == ts


def test_parse_iso_ms_rejects_missing_z():
    with pytest.raises(ClockParseError):
        parse_iso_ms("2026-04-15T12:34:56.789")


def test_parse_iso_ms_rejects_non_string():
    with pytest.raises(ClockParseError):
        parse_iso_ms(1234567890)  # type: ignore[arg-type]


def test_parse_iso_ms_rejects_malformed():
    with pytest.raises(ClockParseError):
        parse_iso_ms("not-a-timestampZ")


def test_add_minutes_preserves_ms_literal_shape():
    out = add_minutes("2026-04-15T00:00:00.000Z", 5)
    assert out == "2026-04-15T00:05:00.000Z"


def test_add_ms_preserves_ms_literal():
    out = add_ms("2026-04-15T00:00:00.000Z", 12345)
    assert out == "2026-04-15T00:00:12.345Z"


def test_diff_ms_signed():
    a = "2026-04-15T00:00:01.000Z"
    b = "2026-04-15T00:00:00.500Z"
    assert diff_ms(a, b) == 500
    assert diff_ms(b, a) == -500


def test_within_window_inclusive_bounds():
    t0 = "2026-04-15T00:00:00.000Z"
    assert within_window(t0, t0)
    assert within_window(t0, "2026-04-15T00:00:01.000Z")
    tend = add_ms(t0, DELTA_MAX_MS)
    assert within_window(t0, tend)
    assert not within_window(t0, add_ms(tend, 1))
    assert not within_window(t0, add_ms(t0, -1))


# ---- D14 canonicalisation tests --------------------------------------------

def test_to_epoch_ms_matches_string_equality_on_canonical_deposit():
    """When two literals are byte-identical their canonical epoch-ms values
    are equal."""
    a = "2026-04-15T12:34:56.789Z"
    assert to_epoch_ms(a) == to_epoch_ms(a)
    assert raw_eq(a, a)


def test_to_epoch_ms_treats_integer_second_and_millisecond_forms_equally():
    """Same UTC instant expressed in the two ISO shapes: canonical equality
    but NOT raw equality. This is the D14 protection: identity comparisons
    in the pipeline use the canonical form."""
    a = "2026-04-15T00:00:00.000Z"
    b = "2026-04-15T00:00:00Z"
    assert to_epoch_ms(a) == to_epoch_ms(b)
    assert not raw_eq(a, b)


def test_parsed_ts_equality_uses_epoch_only():
    a = ParsedTs(epoch_ms=1_000_000, original="X")
    b = ParsedTs(epoch_ms=1_000_000, original="Y")
    assert a == b
    assert hash(a) == hash(b)


def test_canonicalise_preserves_original_literal():
    a = "2026-04-15T12:34:56.789Z"
    p = canonicalise(a)
    assert p.original == a
    assert p.epoch_ms == to_epoch_ms(a)


def test_span_days_matches_observed_calendar_span_of_deposit(registry_csv):
    """D18: the observed calendar span of the registry timestamps is
    approximately 8.63 days (not 8.60, which is the analytic follow-up
    window)."""
    import csv
    with open(registry_csv, "r", encoding="utf-8", newline="") as fh:
        ts = [row["timestamp"] for row in csv.DictReader(fh)]
    d = span_days(ts)
    assert 8.62 < d < 8.64
    # And distinct from the analytic window:
    assert abs(d - (DELTA_MAX_MS / 86_400_000)) > 0.02
