"""prfs.types, object model.

Implements FORMAL_SPEC §1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Candidate:
    """Candidate token observed by the scanner. Spec §1.1."""
    mint: str
    symbol: str
    first_seen_ts: str  # ISO-8601 UTC ms


@dataclass(frozen=True)
class Decision:
    """Filter-chain decision on one tick. Spec §1.4."""
    mint: str
    tick_ts: str
    accepted: bool
    reason: Optional[str] = None  # filter_1..filter_M if not accepted


@dataclass(frozen=True)
class RejectionEvent:
    """A durable rejection event. Spec §1.5 + §1.8.

    event_id is derived from (mint, ts, reason, tracker_epoch).
    p0 is the venue mid at tick_ts (spec §3.1), stored, not recomputed later.
    reason_reg is the emission-time reason ("registry reason"); the outcome
    side's rejectReason is a separate variable, adjudicated in reason_ledger.
    """
    event_id: str  # 128-bit hex
    mint: str
    ts: str        # emission timestamp
    reason_reg: str
    p0: float
    tracker_epoch: int


@dataclass(frozen=True)
class OracleResponse:
    """Result of an oracle query. Spec §1.7 + §3.4.

    ok=False + reason='absence' means Q returned ⊥.
    """
    ok: bool
    price_usd: Optional[float] = None
    liquidity: Optional[int] = None
    volume_24h: Optional[int] = None
    dex_id: Optional[str] = None
    pair_address: Optional[str] = None
    reason: Optional[str] = None  # 'absence', 'validation_failed', etc.


@dataclass(frozen=True)
class FollowupSample:
    """A single successful observation for an event. Spec §3.4 + §7.2.

    Append-only: retries and adaptive triggers each write their own row.
    """
    event_id: str
    tau_min: float                 # scheduled offset in minutes
    sample_ts: str                 # actual observation ts
    price_usd: float
    liquidity: int
    volume_24h: int
    dex_id: str
    pair_address: str
    is_adaptive: bool = False      # True if produced by ExtraTicks
    retry_index: int = 0           # 0 = first attempt


@dataclass(frozen=True)
class OracleAbsenceRecord:
    """Placeholder for a scheduled sample whose retries all failed. Spec §3.5."""
    event_id: str
    tau_min: float
    last_attempt_ts: str
    n_retries: int
    reason: str = "oracle_absence"
