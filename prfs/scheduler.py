"""prfs.scheduler - fixed and adaptive schedulers.

Implements FORMAL_SPEC section 3.2 (schedule), section 3.3 (adaptive triggers).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

from .clock import DELTA_MAX_MS, add_ms, diff_ms, within_window
from .types import RejectionEvent

SCHEDULE_FIX_DEFAULT_MIN: tuple[int, ...] = (5, 15, 60, 240, 1440)


class Scheduler(ABC):
    @abstractmethod
    def next_ticks(self, event: RejectionEvent, now_ms: str) -> list[float]:
        """Return the sorted list of tau_min values whose scheduled time is
        <= now_ms and inside the event's observation window."""


@dataclass
class FixedScheduler(Scheduler):
    """Spec section 3.2 fixed-checkpoint scheduler; matches v1.0.0 default."""
    cadence: tuple[int, ...] = SCHEDULE_FIX_DEFAULT_MIN

    def next_ticks(self, event: RejectionEvent, now_ms: str) -> list[float]:
        due: list[float] = []
        for tau in self.cadence:
            scheduled = add_ms(event.ts, int(tau) * 60_000)
            if diff_ms(scheduled, now_ms) <= 0 and within_window(event.ts, scheduled):
                due.append(float(tau))
        return sorted(due)

    def all_ticks(self, event: RejectionEvent) -> list[float]:
        return sorted(float(x) for x in self.cadence)


@dataclass
class AdaptiveScheduler(Scheduler):
    """Spec section 3.3: adaptive scheduler adds ExtraTicks based on move triggers."""
    base: FixedScheduler
    delta_probe_ms: int = 60_000
    theta_price: float = 0.10  # ln-return threshold
    theta_liq: float = 0.10
    # per-event extra tick tracker; injected by simulator/tests.
    extras: dict[str, list[float]] = field(default_factory=dict)

    def next_ticks(self, event: RejectionEvent, now_ms: str) -> list[float]:
        base_due = set(self.base.next_ticks(event, now_ms))
        for tau in self.extra_ticks(event):
            scheduled = add_ms(event.ts, int(tau * 60_000))
            if diff_ms(scheduled, now_ms) <= 0 and within_window(event.ts, scheduled):
                base_due.add(float(tau))
        return sorted(base_due)

    def extra_ticks(self, event: RejectionEvent) -> list[float]:
        return list(self.extras.get(event.event_id, []))

    def register_extra_tick(self, event: RejectionEvent, tau_min: float) -> None:
        self.extras.setdefault(event.event_id, []).append(tau_min)

    def check_trigger(
        self, price_t: float, price_prev: float, liq_t: float, liq_prev: float
    ) -> bool:
        """Return True if either the price or liquidity ln-ratio breaches threshold."""
        try:
            if price_prev > 0 and price_t > 0:
                if abs(math.log(price_t / price_prev)) >= self.theta_price:
                    return True
        except (ValueError, ZeroDivisionError):
            pass
        try:
            if liq_prev > 0 and liq_t > 0:
                if abs(math.log(liq_t / liq_prev)) >= self.theta_liq:
                    return True
        except (ValueError, ZeroDivisionError):
            pass
        return False
