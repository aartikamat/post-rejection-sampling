"""prfs.sampler - convert scheduled ticks into FollowupSample rows.

Implements FORMAL_SPEC section 3.4 (successful observation).
"""
from __future__ import annotations

import random
from typing import Iterable

from .clock import add_ms
from .oracle_adapter import OracleAdapter, RetryPolicy, validate_observation
from .scheduler import Scheduler
from .types import FollowupSample, OracleAbsenceRecord, RejectionEvent


class Sampler:
    def __init__(
        self,
        oracle: OracleAdapter,
        retry: RetryPolicy,
        rng: random.Random | None = None,
    ) -> None:
        self.oracle = oracle
        self.retry = retry
        self.rng = rng if rng is not None else random.Random(0)
        # (event_id, tau_min) -> True once persisted (see persistence for real dedupe)
        self._emitted: set[tuple[str, float]] = set()

    def sample_due(
        self, event: RejectionEvent, scheduler: Scheduler, now_ms: str
    ) -> tuple[list[FollowupSample], list[OracleAbsenceRecord]]:
        """Query the oracle for every due tick. Returns (samples, absences).
        Idempotent: a repeated call with the same (event, tau) does not re-emit.
        """
        samples: list[FollowupSample] = []
        absences: list[OracleAbsenceRecord] = []
        due = scheduler.next_ticks(event, now_ms)
        for tau in due:
            key = (event.event_id, float(tau))
            if key in self._emitted:
                continue
            scheduled_ts = add_ms(event.ts, int(tau * 60_000))
            resp, attempts = self.oracle.query_with_retry(
                event.mint, scheduled_ts, self.retry, self.rng
            )
            if resp.ok and validate_observation(resp):
                s = FollowupSample(
                    event_id=event.event_id,
                    tau_min=float(tau),
                    sample_ts=scheduled_ts,
                    price_usd=float(resp.price_usd or 0.0),
                    liquidity=int(resp.liquidity or 0),
                    volume_24h=int(resp.volume_24h or 0),
                    dex_id=resp.dex_id or "",
                    pair_address=resp.pair_address or "",
                    is_adaptive=False,
                    retry_index=attempts - 1,
                )
                samples.append(s)
            else:
                absences.append(
                    OracleAbsenceRecord(
                        event_id=event.event_id,
                        tau_min=float(tau),
                        last_attempt_ts=scheduled_ts,
                        n_retries=attempts,
                        reason=resp.reason or "absence",
                    )
                )
            self._emitted.add(key)
        return samples, absences
