"""prfs.clock, time model.

Implements FORMAL_SPEC section 2. Timestamps are stored on disk as ISO 8601
UTC strings with millisecond precision and a Z suffix. For any identity
comparison used in the coverage or reason-attribution pipeline the raw string
is FIRST canonicalised: parsed with parse_iso_ms, converted to a signed
integer of milliseconds since the Unix epoch via to_epoch_ms, and only then
compared to the canonicalised counterpart from the other file. String
equality on the raw literal is retained by a separate helper (raw_eq) for
byte-provenance audits and is never used as the identity check.

The primary key comparison in prfs.coverage and prfs.reason_ledger uses
canonical epoch-millisecond tuples, not raw strings; this makes the pipeline
resilient to timestamps that differ in surface form (fractional-second
padding, offset representation) while representing the same UTC instant.

Parse failures raise ClockParseError, which carries the offending literal
and the failure reason so that upstream code can surface the problem and
halt (per FORMAL_SPEC section 2.4) rather than silently drop the row.

DELTA_MAX_MS is the analytic follow-up window (8.6 days). The OBSERVED
calendar span of the deposit is different (see NUMBER_PROVENANCE_FINAL.csv
observation_calendar_span_days) and is derived separately from the min/max
rejectTs values.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Iterable

MS_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

# Analytic follow-up window (spec section 2). 8.6 days expressed exactly in
# integer milliseconds. This is NOT the observed calendar span of the deposit.
DELTA_MAX_MS: int = 743_040_000


class ClockParseError(ValueError):
    """Raised when an ISO literal cannot be parsed. Preserves the original
    literal on the .literal attribute for provenance."""

    def __init__(self, literal: object, reason: str):
        self.literal = literal
        self.reason = reason
        super().__init__(f"clock parse error ({reason}) for {literal!r}")


@dataclass(frozen=True)
class ParsedTs:
    """A canonical, sortable, comparable representation of a timestamp.

    epoch_ms is the signed integer millisecond offset from the Unix epoch in
    UTC. original preserves the raw string exactly as read from disk for
    provenance and audit. Two ParsedTs values are equal iff their epoch_ms
    values are equal; the original literal is a decorative field only.
    """

    epoch_ms: int
    original: str

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ParsedTs):
            return self.epoch_ms == other.epoch_ms
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.epoch_ms)


def parse_iso_ms(s: str) -> datetime:
    """Parse an ISO 8601 UTC millisecond timestamp with a Z suffix.

    Accepts either the "%Y-%m-%dT%H:%M:%S.%fZ" (millisecond) form or the
    "%Y-%m-%dT%H:%M:%SZ" (integer-second) form. Raises ClockParseError on
    any other input, carrying the raw literal.
    """
    if not isinstance(s, str):
        raise ClockParseError(s, "not-a-string")
    if not s.endswith("Z"):
        raise ClockParseError(s, "missing-Z-suffix")
    body = s[:-1]
    try:
        if "." in body:
            dt = datetime.strptime(s, MS_ISO_FMT)
        else:
            dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ClockParseError(s, f"strptime-{exc}") from None
    return dt.replace(tzinfo=timezone.utc)


def to_epoch_ms(s: str) -> int:
    """Return the signed integer millisecond offset since the Unix epoch for
    an ISO 8601 UTC literal. This is the CANONICAL comparison form used by
    the coverage and reason-attribution pipelines. Raw string equality on the
    literal is NEVER the identity check; canonicalised epoch-ms equality is.
    Any parse failure is surfaced as ClockParseError.
    """
    dt = parse_iso_ms(s)
    # datetime.timestamp on a timezone-aware datetime returns POSIX seconds
    # as a float. Multiply by 1000 and round to integer milliseconds. This
    # preserves the on-disk millisecond precision without introducing
    # sub-millisecond noise.
    return int(round(dt.timestamp() * 1000.0))


def canonicalise(s: str) -> ParsedTs:
    """Return the ParsedTs canonical form for s. Used by upstream key builders
    that need both the epoch and the original literal in one object."""
    return ParsedTs(epoch_ms=to_epoch_ms(s), original=s)


def raw_eq(a: str, b: str) -> bool:
    """True iff the two literals are byte-identical. This is a byte-provenance
    check only and is NOT the identity comparison for the coverage pipeline
    (see to_epoch_ms). The dataset was verified in Cycle A to consist entirely
    of a single canonical shape (%Y-%m-%dT%H:%M:%S.%fZ) so raw equality and
    canonical equality agree on this deposit; the pipeline nevertheless uses
    canonical equality for defence against future mixed-precision inputs.
    """
    return isinstance(a, str) and isinstance(b, str) and a == b


def format_iso_ms(dt: datetime) -> str:
    """Format a timezone-aware UTC datetime back to the ms literal with Z."""
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    dtu = dt.astimezone(timezone.utc)
    micro = dtu.microsecond
    ms = micro // 1000
    return f"{dtu.strftime('%Y-%m-%dT%H:%M:%S')}.{ms:03d}Z"


def add_minutes(ts_ms: str, minutes: float) -> str:
    """Add offset minutes, returning the raw-literal ms timestamp."""
    dt = parse_iso_ms(ts_ms)
    return format_iso_ms(dt + timedelta(minutes=minutes))


def add_ms(ts_ms: str, ms: int) -> str:
    """Add integer milliseconds to an ISO ms literal."""
    dt = parse_iso_ms(ts_ms)
    return format_iso_ms(dt + timedelta(milliseconds=ms))


def diff_ms(a_ms: str, b_ms: str) -> int:
    """Return (a - b) in signed integer milliseconds via canonical epoch."""
    return to_epoch_ms(a_ms) - to_epoch_ms(b_ms)


def within_window(t0_ms: str, sample_ms: str, delta_max_ms: int = DELTA_MAX_MS) -> bool:
    """True iff sample_ms is inside the closed window [t0, t0 + delta_max_ms]
    measured on the canonical epoch-ms axis."""
    d = diff_ms(sample_ms, t0_ms)
    return 0 <= d <= delta_max_ms


def span_days(literals: Iterable[str]) -> float:
    """Return (max - min) across a set of ISO literals expressed in days on
    the canonical epoch-ms axis. Used to derive the OBSERVED calendar span
    from the deposit; distinct from DELTA_MAX_MS which is the analytic
    follow-up window."""
    epochs = [to_epoch_ms(s) for s in literals]
    if not epochs:
        return 0.0
    return (max(epochs) - min(epochs)) / 86_400_000.0
