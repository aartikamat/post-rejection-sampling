"""tests/test_restart.py - restart-recovery discipline (spec section 7.3)."""
from __future__ import annotations

import random

from prfs.clock import add_ms
from prfs.oracle_adapter import RetryPolicy
from prfs.persistence import WalEventStore, WalSampleLog, restart_recover
from prfs.sampler import Sampler
from prfs.scheduler import FixedScheduler
from prfs.simulator import DeterministicSimulator, SimulatorConfig
from prfs.types import RejectionEvent


def _mk_event(eid="e1", mint="M1", ts="2026-04-15T00:00:00.000Z"):
    return RejectionEvent(event_id=eid, mint=mint, ts=ts, reason_reg="filter_1", p0=1.0, tracker_epoch=0)


def test_kill_and_resume_mid_window_recovers_all_events(tmp_path):
    store = WalEventStore(tmp_path / "events.ndjson")
    events = [_mk_event(f"e{i}", f"M{i}") for i in range(4)]
    for e in events:
        store.append_event(e)
    log = WalSampleLog(tmp_path / "samples.ndjson")
    sched = FixedScheduler()
    sim = DeterministicSimulator(SimulatorConfig(seed=0))
    sampler = Sampler(sim, RetryPolicy(sleep=False), random.Random(0))
    ev = events[0]
    ts_60m = add_ms(ev.ts, 60 * 60_000)
    samples, _ = sampler.sample_due(ev, sched, ts_60m)
    for s in samples:
        log.append_sample(s)
    # simulate crash: re-open store from disk
    store2 = WalEventStore(tmp_path / "events.ndjson")
    log2 = WalSampleLog(tmp_path / "samples.ndjson")
    now = add_ms(ev.ts, 120 * 60_000)
    active = restart_recover(store2, log2, now)
    # all 4 events are still inside 8.6-day window at t0 + 2h
    assert len(active) == 4
    # replayed sample count = 3 (tau=5, 15, 60 due by t0 + 60m)
    assert len(list(log2.replay_samples())) == 3


def test_retry_state_survives_restart(tmp_path):
    """Restart re-reads the sample log so idempotency holds across restarts."""
    log = WalSampleLog(tmp_path / "samples.ndjson")
    sched = FixedScheduler()
    sim = DeterministicSimulator(SimulatorConfig(seed=0))
    sampler = Sampler(sim, RetryPolicy(sleep=False), random.Random(0))
    ev = _mk_event()
    ts = add_ms(ev.ts, 60 * 60_000)
    samples1, _ = sampler.sample_due(ev, sched, ts)
    for s in samples1:
        log.append_sample(s)
    # Restart: fresh log instance, fresh sampler with sample dedupe seeded
    # from replay.
    log2 = WalSampleLog(tmp_path / "samples.ndjson")
    sampler2 = Sampler(sim, RetryPolicy(sleep=False), random.Random(0))
    for prior in log2.replay_samples():
        sampler2._emitted.add((prior.event_id, float(prior.tau_min)))
    samples2, _ = sampler2.sample_due(ev, sched, ts)
    # Idempotent: no new samples emitted after restart.
    assert samples2 == []
    # And the log's write of any duplicate returns False.
    for s in samples1:
        assert log2.append_sample(s) is False


def test_terminal_state_locks_further_sample_writes(tmp_path):
    log = WalSampleLog(tmp_path / "samples.ndjson")
    log.mark_terminal("e1")
    sched = FixedScheduler()
    sim = DeterministicSimulator(SimulatorConfig(seed=0))
    sampler = Sampler(sim, RetryPolicy(sleep=False), random.Random(0))
    ev = _mk_event(eid="e1", mint="M1")
    ts = add_ms(ev.ts, 60 * 60_000)
    samples, _ = sampler.sample_due(ev, sched, ts)
    # sampler still returns them; the log enforces terminal state
    write_results = [log.append_sample(s) for s in samples]
    assert all(r is False for r in write_results)
