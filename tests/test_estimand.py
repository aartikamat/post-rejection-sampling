"""tests/test_estimand.py - spec section 6."""
from __future__ import annotations

import pytest

from prfs.estimand import (
    ExecutionModel,
    MissingExecutionModel,
    hypothetical_executed_pnl,
    observed_path_returns,
)
from prfs.types import FollowupSample, RejectionEvent


def _ev(eid="e1", p0=1.0):
    return RejectionEvent(event_id=eid, mint="M1", ts="2026-04-15T00:00:00.000Z",
                          reason_reg="filter_1", p0=p0, tracker_epoch=0)


def _sample(price, tau=5.0, eid="e1"):
    return FollowupSample(event_id=eid, tau_min=tau,
                          sample_ts="2026-04-15T00:05:00.000Z",
                          price_usd=price, liquidity=1, volume_24h=1,
                          dex_id="pumpswap", pair_address="P")


def test_observed_return_default_estimand():
    events = [_ev(p0=1.0)]
    samples = [_sample(1.2)]
    r = observed_path_returns(events, samples)
    assert len(r) == 1
    assert round(r[0].r_pct, 4) == 20.0


def test_hypothetical_pnl_raises_without_model():
    with pytest.raises(MissingExecutionModel):
        hypothetical_executed_pnl([_ev()], [_sample(1.2)], None)


def test_hypothetical_pnl_applies_frictions():
    model = ExecutionModel(
        entry_slippage_pct=0.5, exit_slippage_pct=0.5,
        fee_pct=0.3, exit_rule="horizon_close",
        portfolio_constraint="unit_position",
    )
    events = [_ev(p0=1.0)]
    samples = [_sample(1.2)]  # +20% gross
    pnls = hypothetical_executed_pnl(events, samples, model)
    # 20 - 0.5 - 0.5 - 2*0.3 = 18.4
    assert len(pnls) == 1
    assert round(pnls[0], 4) == 18.4


def test_hypothetical_pnl_rejects_incomplete_model():
    model = ExecutionModel(entry_slippage_pct=0.5, exit_slippage_pct=0.5,
                            fee_pct=0.3, exit_rule="", portfolio_constraint="")
    with pytest.raises(MissingExecutionModel):
        hypothetical_executed_pnl([_ev()], [_sample(1.2)], model)
