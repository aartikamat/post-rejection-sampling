"""prfs.oracle_adapter - oracle contract with retry / absence handling.

Implements FORMAL_SPEC section 1.7 (oracle), 3.4 (successful observation
predicate), 3.5 (retry policy).
"""
from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .types import OracleResponse


@dataclass
class RetryPolicy:
    """Spec section 3.5 retry policy."""
    r_max: int = 3
    base_backoff_ms: int = 250
    jitter_lo: float = 0.5
    jitter_hi: float = 1.5
    sleep: bool = False  # tests override to skip real sleep


class OracleAdapter(ABC):
    """Abstract oracle. Spec section 1.7."""

    @abstractmethod
    def query(self, mint: str, ts_ms: str) -> OracleResponse:
        """Return an OracleResponse (ok=True) or an absence response (ok=False)."""

    def query_with_retry(
        self, mint: str, ts_ms: str, policy: RetryPolicy, rng: random.Random | None = None
    ) -> tuple[OracleResponse, int]:
        """Retry per spec section 3.5. Returns (final_response, n_attempts)."""
        r = rng if rng is not None else random.Random(0)
        resp = self.query(mint, ts_ms)
        attempts = 1
        while (not resp.ok) and attempts <= policy.r_max:
            wait_ms = policy.base_backoff_ms * (2 ** (attempts - 1))
            jitter = r.uniform(policy.jitter_lo, policy.jitter_hi)
            wait = wait_ms * jitter
            if policy.sleep:
                time.sleep(wait / 1000.0)
            resp = self.query(mint, ts_ms)
            attempts += 1
        return resp, attempts


def validate_observation(resp: OracleResponse) -> bool:
    """Spec section 3.4 validation predicate."""
    if not resp.ok:
        return False
    if resp.price_usd is None or resp.price_usd <= 0:
        return False
    if resp.liquidity is None or resp.liquidity < 0:
        return False
    if resp.volume_24h is None or resp.volume_24h < 0:
        return False
    if resp.pair_address is None:
        return False
    return True
