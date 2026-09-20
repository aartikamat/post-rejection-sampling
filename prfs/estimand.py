"""prfs.estimand - observed market path (section 6.1) vs hypothetical executed PnL (section 6.2).

Implements FORMAL_SPEC section 6. All hypothetical-PnL entrypoints require the
full exit-model configuration OR raise MissingExecutionModel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .types import FollowupSample, RejectionEvent


@dataclass(frozen=True)
class ObservedPathReturn:
    event_id: str
    tau_min: float
    r_pct: float  # (obs_price / p0 - 1) * 100


@dataclass(frozen=True)
class ExecutionModel:
    """Spec section 6.2: required to compute hypothetical executed PnL."""
    entry_slippage_pct: float
    exit_slippage_pct: float
    fee_pct: float
    exit_rule: str
    portfolio_constraint: str


class MissingExecutionModel(ValueError):
    """Raised when hypothetical PnL is requested without a full model."""


def observed_path_returns(
    events: Iterable[RejectionEvent],
    samples: Iterable[FollowupSample],
) -> list[ObservedPathReturn]:
    """Spec section 6.1 default estimand."""
    p0_by_event: dict[str, float] = {e.event_id: float(e.p0) for e in events}
    out: list[ObservedPathReturn] = []
    for s in samples:
        p0 = p0_by_event.get(s.event_id)
        if p0 is None or p0 <= 0:
            continue
        r = (float(s.price_usd) / p0 - 1.0) * 100.0
        out.append(ObservedPathReturn(event_id=s.event_id, tau_min=float(s.tau_min), r_pct=r))
    return out


def hypothetical_executed_pnl(
    events: Iterable[RejectionEvent],
    samples: Iterable[FollowupSample],
    model: Optional[ExecutionModel],
) -> list[float]:
    """Spec section 6.2. Raises MissingExecutionModel if model is None or incomplete."""
    if model is None:
        raise MissingExecutionModel(
            "Spec section 6.2 requires entry_slippage, exit_slippage, fees, "
            "exit_rule, portfolio_constraint. No ExecutionModel provided."
        )
    for name in ("entry_slippage_pct", "exit_slippage_pct", "fee_pct", "exit_rule", "portfolio_constraint"):
        val = getattr(model, name, None)
        if val is None or val == "":
            raise MissingExecutionModel(f"ExecutionModel missing required field: {name}")
    p0 = {e.event_id: float(e.p0) for e in events}
    pnls: list[float] = []
    for s in samples:
        base = p0.get(s.event_id)
        if base is None or base <= 0:
            continue
        gross = (float(s.price_usd) / base) - 1.0
        # apply slippage and fees (both sides)
        net = (
            gross
            - (model.entry_slippage_pct / 100.0)
            - (model.exit_slippage_pct / 100.0)
            - (2.0 * model.fee_pct / 100.0)
        )
        pnls.append(net * 100.0)
    return pnls
