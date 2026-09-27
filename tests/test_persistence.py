"""tests/test_persistence.py - WAL, idempotency (spec section 7.1, 7.2)."""
from __future__ import annotations

from prfs.persistence import WalEventStore, WalSampleLog, restart_recover
from prfs.types import FollowupSample, OracleAbsenceRecord, RejectionEvent


def _mk_event(eid="e1", mint="M1", ts="2026-04-15T00:00:00.000Z", reason="filter_1"):
    return RejectionEvent(event_id=eid, mint=mint, ts=ts, reason_reg=reason, p0=1.0, tracker_epoch=0)


def test_append_event_is_idempotent_on_event_id(tmp_path):
    store = WalEventStore(tmp_path / "events.ndjson")
    e = _mk_event()
    assert store.append_event(e) is True
    assert store.append_event(e) is False   # duplicate id dropped
    assert store.append_event(_mk_event(eid="e2")) is True
    events = list(store.replay_events())
    assert [x.event_id for x in events] == ["e1", "e2"]


def test_append_sample_is_idempotent_on_event_id_and_tau(tmp_path):
    log = WalSampleLog(tmp_path / "samples.ndjson")
    s = FollowupSample(
        event_id="e1", tau_min=5.0, sample_ts="2026-04-15T00:05:00.000Z",
        price_usd=1.1, liquidity=32768, volume_24h=65536,
        dex_id="pumpswap", pair_address="P1",
    )
    assert log.append_sample(s) is True
    assert log.append_sample(s) is False    # duplicate (event_id, tau)
    s2 = FollowupSample(
        event_id="e1", tau_min=15.0, sample_ts="2026-04-15T00:15:00.000Z",
        price_usd=1.2, liquidity=32768, volume_24h=65536,
        dex_id="pumpswap", pair_address="P1",
    )
    assert log.append_sample(s2) is True
    samples = list(log.replay_samples())
    assert len(samples) == 2


def test_wal_replay_reconstructs_in_memory_state(tmp_path):
    store = WalEventStore(tmp_path / "events.ndjson")
    for i in range(5):
        store.append_event(_mk_event(eid=f"e{i}", mint=f"M{i}"))
    # simulate restart: fresh store instance
    store2 = WalEventStore(tmp_path / "events.ndjson")
    assert len(list(store2.replay_events())) == 5
    # further appends after restart must still be idempotent
    assert store2.append_event(_mk_event(eid="e0", mint="M0")) is False


def test_duplicate_event_id_logs_warning_and_drops(tmp_path, caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="prfs.persistence")
    store = WalEventStore(tmp_path / "events.ndjson")
    e = _mk_event()
    store.append_event(e)
    store.append_event(e)
    assert any("duplicate event_id" in r.message for r in caplog.records)


def test_absence_write_is_idempotent(tmp_path):
    log = WalSampleLog(tmp_path / "samples.ndjson")
    a = OracleAbsenceRecord(
        event_id="e1", tau_min=5.0, last_attempt_ts="2026-04-15T00:05:00.000Z",
        n_retries=4, reason="absence",
    )
    assert log.append_absence(a) is True
    assert log.append_absence(a) is False


def test_terminal_state_blocks_sample_writes(tmp_path):
    log = WalSampleLog(tmp_path / "samples.ndjson")
    log.mark_terminal("e1")
    s = FollowupSample(
        event_id="e1", tau_min=5.0, sample_ts="2026-04-15T00:05:00.000Z",
        price_usd=1.1, liquidity=32768, volume_24h=65536,
        dex_id="pumpswap", pair_address="P1",
    )
    assert log.append_sample(s) is False
