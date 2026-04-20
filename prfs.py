"""
Post-Rejection Follow-up Sampling (PRFS)
Reference implementation.

Companion code for:
  Kamat, A. U. (2026). Post-Rejection Follow-up Sampling: A Methodology for
  Counterfactual Outcome Measurement in Algorithmic DEX Trading.
  Zenodo. https://doi.org/10.5281/zenodo.19671657

Author: Arati Uday Kamat (ORCID: 0009-0000-4781-312X)
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np


# Default follow-up checkpoints, in minutes after rejection.
DEFAULT_CHECKPOINTS_MIN = (5, 15, 60, 240, 1440)


@dataclass
class Candidate:
    """A single candidate trading signal observed by the system."""
    candidate_id: str
    symbol: str
    timestamp: float              # UNIX seconds
    reference_price: float        # price at decision time
    features: dict = field(default_factory=dict)


@dataclass
class Decision:
    """The accept/reject outcome emitted by a filter stack for one candidate."""
    candidate_id: str
    accepted: bool
    passed_gates: tuple = ()      # names of gates the candidate passed
    rejected_at: Optional[str] = None  # name of the gate that rejected (if any)


@dataclass
class AcceptedOutcome:
    """Realized PnL for an accepted trade over the holding window."""
    candidate_id: str
    realized_pnl_pct: float


@dataclass
class FollowupRecord:
    """Forward-price observations for a rejected candidate."""
    candidate_id: str
    reference_price: float
    prices_at_checkpoint: dict    # {minutes_after: price}


def pct_return(start: float, end: float) -> float:
    """Simple percent return."""
    if start <= 0:
        return 0.0
    return (end / start - 1.0) * 100.0


class RejectionTracker:
    """
    Records rejections and schedules forward-price follow-ups at the configured
    checkpoints. In production, `price_fn` would query a price oracle or DEX API.
    Here it is injected so the reference implementation stays infrastructure-free.
    """

    def __init__(
        self,
        price_fn: Callable[[str, float], float],
        checkpoints_min: Iterable[int] = DEFAULT_CHECKPOINTS_MIN,
    ) -> None:
        self.price_fn = price_fn
        self.checkpoints_min = tuple(sorted(set(checkpoints_min)))
        self._rejections: dict[str, Candidate] = {}

    def record(self, candidate: Candidate, decision: Decision) -> None:
        if decision.accepted:
            return
        self._rejections[candidate.candidate_id] = candidate

    def run_followup(self) -> list[FollowupRecord]:
        """
        For each recorded rejection, query forward prices at each checkpoint
        offset from the rejection timestamp. Returns one record per rejection.
        """
        records: list[FollowupRecord] = []
        for cid, cand in self._rejections.items():
            prices_at = {}
            for dt_min in self.checkpoints_min:
                t = cand.timestamp + dt_min * 60.0
                prices_at[dt_min] = self.price_fn(cand.symbol, t)
            records.append(
                FollowupRecord(
                    candidate_id=cid,
                    reference_price=cand.reference_price,
                    prices_at_checkpoint=prices_at,
                )
            )
        return records


class FollowupSampler:
    """
    Converts raw follow-up price observations into a counterfactual PnL
    distribution. Each rejection contributes one counterfactual PnL per
    checkpoint, computed as the percent return from rejection-price to
    checkpoint-price.
    """

    @staticmethod
    def counterfactual_pnl(records: Iterable[FollowupRecord]) -> dict[int, np.ndarray]:
        """
        Returns a dict {checkpoint_minutes: np.ndarray of percent PnL values}.
        One entry per rejection per checkpoint.
        """
        by_checkpoint: dict[int, list[float]] = {}
        for r in records:
            for dt_min, price in r.prices_at_checkpoint.items():
                by_checkpoint.setdefault(dt_min, []).append(
                    pct_return(r.reference_price, price)
                )
        return {k: np.asarray(v, dtype=float) for k, v in by_checkpoint.items()}


class FilterEvaluator:
    """
    Compares the realized PnL distribution of accepted trades to the
    counterfactual PnL distribution of rejected candidates at a chosen
    follow-up horizon. A filter is considered justified if the rejected
    distribution is dominated (in mean) by the accepted distribution under
    the operator's utility function.
    """

    def __init__(self, utility: Callable[[np.ndarray], float] | None = None) -> None:
        # Default utility is expected return. Operators with loss aversion can
        # pass a concave utility (e.g. CRRA) instead.
        self.utility = utility or (lambda x: float(np.mean(x)))

    def evaluate(
        self,
        accepted: Iterable[AcceptedOutcome],
        counterfactual_by_checkpoint: dict[int, np.ndarray],
        horizon_min: int,
    ) -> dict:
        accepted_arr = np.asarray([a.realized_pnl_pct for a in accepted], dtype=float)
        rejected_arr = counterfactual_by_checkpoint.get(horizon_min, np.asarray([]))

        if accepted_arr.size == 0 or rejected_arr.size == 0:
            return {
                "horizon_min": horizon_min,
                "n_accepted": int(accepted_arr.size),
                "n_rejected": int(rejected_arr.size),
                "note": "insufficient data for evaluation",
            }

        u_accepted = self.utility(accepted_arr)
        u_rejected = self.utility(rejected_arr)
        # Stochastic dominance check (first-order, empirical CDF comparison).
        # A filter strongly justified if accepted ECDF lies below rejected ECDF
        # everywhere — i.e. accepted trades are more likely to exceed any
        # given return threshold than rejected ones.
        grid = np.linspace(
            min(accepted_arr.min(), rejected_arr.min()),
            max(accepted_arr.max(), rejected_arr.max()),
            200,
        )
        ecdf_accepted = np.mean(accepted_arr[:, None] <= grid[None, :], axis=0)
        ecdf_rejected = np.mean(rejected_arr[:, None] <= grid[None, :], axis=0)
        dominance = float(np.mean(ecdf_accepted <= ecdf_rejected))

        return {
            "horizon_min": horizon_min,
            "n_accepted": int(accepted_arr.size),
            "n_rejected": int(rejected_arr.size),
            "mean_accepted_pct": float(np.mean(accepted_arr)),
            "mean_rejected_pct": float(np.mean(rejected_arr)),
            "utility_accepted": u_accepted,
            "utility_rejected": u_rejected,
            "utility_gap": u_accepted - u_rejected,
            "fraction_accepted_dominates": dominance,
            "filter_justified_by_mean": u_accepted > u_rejected,
        }
