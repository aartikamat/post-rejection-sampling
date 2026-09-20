"""prfs.censoring - right and interval censoring indicators.

Implements FORMAL_SPEC section 8.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .clock import DELTA_MAX_MS, add_ms, diff_ms
from .types import FollowupSample, OracleAbsenceRecord, RejectionEvent


@dataclass(frozen=True)
class CensoringIndicator:
    event_id: str
    tau_min: float
    right_censored: bool
    interval_censored: bool
    interval_lo_ms: int | None
    interval_hi_ms: int | None


def label_censoring(
    events: Iterable[RejectionEvent],
    samples: Iterable[FollowupSample],
    absences: Iterable[OracleAbsenceRecord],
    t_cut_ms: str,
    delta_probe_ms: int,
    schedule_min: tuple[int, ...] = (5, 15, 60, 240, 1440),
) -> list[CensoringIndicator]:
    """Emit one CensoringIndicator per (event, tau_min) scheduled sample."""
    events_l = list(events)
    samples_l = list(samples)
    absences_l = list(absences)

    got: set[tuple[str, float]] = {(s.event_id, float(s.tau_min)) for s in samples_l}
    absent: set[tuple[str, float]] = {(a.event_id, float(a.tau_min)) for a in absences_l}
    # sorted sample offsets per event for interval-neighbour lookup
    per_event_taus: dict[str, list[float]] = {}
    for s in samples_l:
        per_event_taus.setdefault(s.event_id, []).append(float(s.tau_min))
    for lst in per_event_taus.values():
        lst.sort()

    out: list[CensoringIndicator] = []
    for e in events_l:
        for tau in schedule_min:
            scheduled = add_ms(e.ts, int(tau) * 60_000)
            past_cut = diff_ms(scheduled, t_cut_ms) > 0
            right_cens = past_cut
            interval_lo = None
            interval_hi = None
            interval_cens = False
            key = (e.event_id, float(tau))
            if key not in got:
                # not observed at scheduled tau; look for neighbour tick within Delta_probe
                neighbours = per_event_taus.get(e.event_id, [])
                for other in neighbours:
                    d_ms = abs(int((other - tau) * 60_000))
                    if d_ms <= delta_probe_ms:
                        interval_cens = True
                        interval_lo = int((tau * 60_000) - delta_probe_ms)
                        interval_hi = int((tau * 60_000) + delta_probe_ms)
                        break
            out.append(
                CensoringIndicator(
                    event_id=e.event_id,
                    tau_min=float(tau),
                    right_censored=right_cens,
                    interval_censored=interval_cens,
                    interval_lo_ms=interval_lo,
                    interval_hi_ms=interval_hi,
                )
            )
    return out
