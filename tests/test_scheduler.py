"""tests/test_scheduler.py - schedule invariants (spec section 3.2, 3.3)."""
from __future__ import annotations

from prfs.clock import add_ms
from prfs.scheduler import AdaptiveScheduler, FixedScheduler, SCHEDULE_FIX_DEFAULT_MIN
from prfs.types import RejectionEvent


def _mk_event(ts="2026-04-15T00:00:00.000Z") -> RejectionEvent:
    return RejectionEvent(
        event_id="e0",
        mint="M0",
        ts=ts,
        reason_reg="filter_1",
        p0=1.0,
        tracker_epoch=0,
    )


def test_fixed_scheduler_default_cadence_5_15_60_240_1440():
    sched = FixedScheduler()
    assert sched.cadence == (5, 15, 60, 240, 1440)
    ev = _mk_event()
    # A year later - all ticks due, all inside 8.6-day window? no, 1440
    # minutes = 24h fits inside 8.6 days. Check that at t0 + 1440 min all
    # five are returned.
    later = add_ms(ev.ts, 1440 * 60_000)
    due = sched.next_ticks(ev, later)
    assert due == [5.0, 15.0, 60.0, 240.0, 1440.0]


def test_fixed_scheduler_returns_only_due_ticks():
    sched = FixedScheduler()
    ev = _mk_event()
    # at t0 + 10 min only tau=5 is due
    ts_10m = add_ms(ev.ts, 10 * 60_000)
    assert sched.next_ticks(ev, ts_10m) == [5.0]


def test_fixed_scheduler_returns_empty_before_first_tick():
    sched = FixedScheduler()
    ev = _mk_event()
    ts_1m = add_ms(ev.ts, 1 * 60_000)
    assert sched.next_ticks(ev, ts_1m) == []


def test_adaptive_scheduler_only_adds_ticks():
    """AdaptiveScheduler.next_ticks() must be a superset of
    FixedScheduler.next_ticks() at every now_ms."""
    fixed = FixedScheduler()
    adaptive = AdaptiveScheduler(base=fixed)
    ev = _mk_event()
    adaptive.register_extra_tick(ev, 30.0)
    ts_30m = add_ms(ev.ts, 30 * 60_000)
    a = set(adaptive.next_ticks(ev, ts_30m))
    f = set(fixed.next_ticks(ev, ts_30m))
    assert f.issubset(a)
    assert 30.0 in a


def test_adaptive_trigger_price_threshold():
    fixed = FixedScheduler()
    adaptive = AdaptiveScheduler(base=fixed, theta_price=0.10)
    # 10% jump = ln(1.1) ~= 0.0953 - just under. 15% jump = ln(1.15) ~= 0.14.
    assert not adaptive.check_trigger(1.10, 1.00, 100.0, 100.0)
    assert adaptive.check_trigger(1.15, 1.00, 100.0, 100.0)


def test_adaptive_trigger_liquidity_threshold():
    fixed = FixedScheduler()
    adaptive = AdaptiveScheduler(base=fixed, theta_liq=0.20)
    assert not adaptive.check_trigger(1.0, 1.0, 110.0, 100.0)
    assert adaptive.check_trigger(1.0, 1.0, 150.0, 100.0)
