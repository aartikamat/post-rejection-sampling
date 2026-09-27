"""tests/test_censoring.py - spec section 8."""
from __future__ import annotations

from prfs.censoring import label_censoring
from prfs.types import FollowupSample, OracleAbsenceRecord, RejectionEvent


def _ev(eid="e1"):
    return RejectionEvent(event_id=eid, mint="M1", ts="2026-04-15T00:00:00.000Z",
                          reason_reg="filter_1", p0=1.0, tracker_epoch=0)


def test_right_censoring_marks_taus_past_cutoff():
    ev = _ev()
    t_cut = "2026-04-15T00:59:00.000Z"  # cuts off before tau=60 and later
    ind = label_censoring([ev], [], [], t_cut, delta_probe_ms=60_000)
    by_tau = {i.tau_min: i for i in ind}
    assert by_tau[5.0].right_censored is False
    assert by_tau[15.0].right_censored is False
    assert by_tau[60.0].right_censored is True
    assert by_tau[240.0].right_censored is True
    assert by_tau[1440.0].right_censored is True


def test_interval_censoring_from_neighbouring_adaptive_sample():
    ev = _ev()
    # observed only at tau=6.5 min, and only via an "adaptive" tick;
    # scheduled tau=5 should be interval-censored using delta_probe=120s
    samp = FollowupSample(event_id="e1", tau_min=6.5,
                          sample_ts="2026-04-15T00:06:30.000Z",
                          price_usd=1.0, liquidity=1, volume_24h=1,
                          dex_id="d", pair_address="P",
                          is_adaptive=True)
    ind = label_censoring([ev], [samp], [], "2026-04-16T00:00:00.000Z",
                          delta_probe_ms=120_000)
    by_tau = {i.tau_min: i for i in ind}
    assert by_tau[5.0].interval_censored is True
    assert by_tau[15.0].interval_censored is False


def test_no_censoring_when_all_scheduled_taus_observed():
    ev = _ev()
    samples = [
        FollowupSample(event_id="e1", tau_min=float(t),
                       sample_ts=f"2026-04-15T00:{int(t):02d}:00.000Z" if t <= 60 else "2026-04-15T04:00:00.000Z",
                       price_usd=1.0, liquidity=1, volume_24h=1,
                       dex_id="d", pair_address="P")
        for t in (5, 15, 60, 240, 1440)
    ]
    ind = label_censoring([ev], samples, [], "2026-04-30T00:00:00.000Z",
                          delta_probe_ms=60_000)
    for i in ind:
        assert i.interval_censored is False
        assert i.right_censored is False
