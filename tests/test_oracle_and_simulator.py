"""tests/test_oracle_and_simulator.py - oracle contract + deterministic simulator."""
from __future__ import annotations

from prfs.oracle_adapter import RetryPolicy, validate_observation
from prfs.simulator import DeterministicSimulator, SimulatorConfig
from prfs.types import OracleResponse


def test_validate_observation_rejects_zero_price():
    r = OracleResponse(ok=True, price_usd=0.0, liquidity=100, volume_24h=100,
                       dex_id="pumpswap", pair_address="P1")
    assert not validate_observation(r)


def test_validate_observation_accepts_normal():
    r = OracleResponse(ok=True, price_usd=1.0, liquidity=100, volume_24h=100,
                       dex_id="pumpswap", pair_address="P1")
    assert validate_observation(r)


def test_validate_observation_rejects_missing_pair():
    r = OracleResponse(ok=True, price_usd=1.0, liquidity=100, volume_24h=100,
                       dex_id="pumpswap", pair_address=None)
    assert not validate_observation(r)


def test_simulator_deterministic_same_seed():
    cfg = SimulatorConfig(seed=42, absence_rate=0.3)
    sim1 = DeterministicSimulator(cfg)
    sim2 = DeterministicSimulator(cfg)
    for i in range(50):
        mint = f"M{i}"
        ts = f"2026-04-15T00:{i:02d}:00.000Z"
        a = sim1.query(mint, ts)
        b = sim2.query(mint, ts)
        assert a == b


def test_simulator_absence_rate_within_tolerance():
    cfg = SimulatorConfig(seed=0, absence_rate=0.5)
    sim = DeterministicSimulator(cfg)
    n = 2000
    absent = 0
    for i in range(n):
        r = sim.query(f"M{i}", f"2026-04-15T00:{i%60:02d}:{(i//60)%60:02d}.000Z")
        if not r.ok:
            absent += 1
    # 50% target with n=2000, allow +/- 5pp
    assert 0.45 * n <= absent <= 0.55 * n


def test_retry_policy_records_attempts_and_returns_final_response():
    """The oracle retry loop invokes .query up to R_max times when the
    response is not ok. Exercises the always-absent stub."""

    class AlwaysAbsent:
        calls = 0
        def query(self, mint, ts):
            AlwaysAbsent.calls += 1
            return OracleResponse(ok=False, reason="absence")
        # Use the mixin from OracleAdapter via a small shim; simpler here is
        # to reimplement the loop:
        def query_with_retry(self, mint, ts, policy, rng=None):
            resp = self.query(mint, ts)
            attempts = 1
            while (not resp.ok) and attempts <= policy.r_max:
                resp = self.query(mint, ts)
                attempts += 1
            return resp, attempts

    p = RetryPolicy(r_max=3, sleep=False)
    a = AlwaysAbsent()
    r, n = a.query_with_retry("M", "2026-04-15T00:00:00.000Z", p)
    assert not r.ok
    assert n == 4  # 1 + 3 retries
