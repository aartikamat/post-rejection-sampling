"""prfs.simulator - deterministic offline oracle for validation.

Emits configurable failure-rate, absence-rate, and reason-drift streams so
Phase B tests can verify the reference implementation regenerates counts
under controlled conditions.
"""
from __future__ import annotations

import bisect
import hashlib
import random
from dataclasses import dataclass, field
from typing import Iterable

from .clock import parse_iso_ms
from .oracle_adapter import OracleAdapter
from .types import OracleResponse


@dataclass
class SimulatorConfig:
    seed: int = 0
    absence_rate: float = 0.0
    validation_failure_rate: float = 0.0
    reason_drift_rate: float = 0.0
    # per-mint deterministic price path: list of (ts_iso, price_usd)
    price_paths: dict[str, list[tuple[str, float]]] | None = None
    default_price_usd: float = 1.0
    default_liquidity: int = 32768
    default_volume_24h: int = 65536
    default_dex_id: str = "pumpswap"
    pair_address_prefix: str = "SIM"


class DeterministicSimulator(OracleAdapter):
    """OracleAdapter for offline replay. Yields the same OracleResponse
    sequence given the same config + seed. Determinism uses a
    per-(mint, ts) sub-rng seeded from cfg.seed so query order doesn't
    matter."""

    def __init__(self, cfg: SimulatorConfig) -> None:
        self.cfg = cfg

    def _rng_for(self, mint: str, ts_ms: str) -> random.Random:
        h = hashlib.sha256(f"{self.cfg.seed}|{mint}|{ts_ms}".encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "big")
        return random.Random(seed)

    def _price_at(self, mint: str, ts_ms: str) -> float:
        path = (self.cfg.price_paths or {}).get(mint)
        if not path:
            return self.cfg.default_price_usd
        target = parse_iso_ms(ts_ms)
        # linear step function: last (ts, price) with ts <= target
        idx = bisect.bisect_right([parse_iso_ms(t) for (t, _) in path], target) - 1
        if idx < 0:
            return path[0][1]
        return path[idx][1]

    def query(self, mint: str, ts_ms: str) -> OracleResponse:
        r = self._rng_for(mint, ts_ms)
        u = r.random()
        if u < self.cfg.absence_rate:
            return OracleResponse(ok=False, reason="absence")
        v = r.random()
        if v < self.cfg.validation_failure_rate:
            return OracleResponse(
                ok=True,
                price_usd=0.0,
                liquidity=0,
                volume_24h=0,
                dex_id=self.cfg.default_dex_id,
                pair_address=None,
                reason="validation_failed",
            )
        price = self._price_at(mint, ts_ms)
        return OracleResponse(
            ok=True,
            price_usd=price,
            liquidity=self.cfg.default_liquidity,
            volume_24h=self.cfg.default_volume_24h,
            dex_id=self.cfg.default_dex_id,
            pair_address=f"{self.cfg.pair_address_prefix}-{mint[:8]}",
        )
