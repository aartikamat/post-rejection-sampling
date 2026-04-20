"""
Runnable demonstration of Post-Rejection Follow-up Sampling (PRFS).

Generates synthetic candidate signals and a synthetic price process, applies
a toy filter, then evaluates whether the filter is empirically justified.
Produces only printed diagnostics — no external dependencies beyond NumPy.

Run:
    python example.py

Author: Arati Uday Kamat (ORCID: 0009-0000-4781-312X)
License: MIT
"""

from __future__ import annotations

import numpy as np

from prfs import (
    AcceptedOutcome,
    Candidate,
    Decision,
    FilterEvaluator,
    FollowupSampler,
    RejectionTracker,
)


RNG = np.random.default_rng(seed=42)

# Synthetic universe: N candidates, each with a latent "true alpha" signal.
# The filter attempts to accept candidates with positive alpha and reject
# candidates with negative alpha, but does so imperfectly.
N = 2000
HOLDING_WINDOW_MIN = 60
CHECKPOINTS_MIN = (5, 15, 60, 240, 1440)


def generate_candidates(n: int) -> list[tuple[Candidate, float]]:
    """Return (candidate, true_alpha) pairs. true_alpha is hidden from the filter."""
    out = []
    t0 = 1_700_000_000.0
    for i in range(n):
        alpha = RNG.normal(loc=0.0, scale=5.0)  # pct, 60-min horizon
        noisy_feature = alpha + RNG.normal(0.0, 3.0)
        out.append(
            (
                Candidate(
                    candidate_id=f"c{i:05d}",
                    symbol=f"TKN{i % 50}",
                    timestamp=t0 + i * 60.0,
                    reference_price=1.0,
                    features={"score": noisy_feature},
                ),
                alpha,
            )
        )
    return out


def toy_filter(candidate: Candidate) -> Decision:
    """
    Accept if the noisy score is above threshold. The filter has access to
    `noisy_feature` only — NOT to the true alpha. In real systems this is the
    composition of many gates (liquidity, volume, holder count, time-of-day, etc).
    """
    score = candidate.features.get("score", 0.0)
    threshold = 2.0
    if score > threshold:
        return Decision(candidate.candidate_id, accepted=True, passed_gates=("score",))
    return Decision(
        candidate.candidate_id,
        accepted=False,
        passed_gates=(),
        rejected_at="score",
    )


def build_price_oracle(candidates_alpha: dict[str, float]) -> callable:
    """
    Returns a function price(symbol, t) that produces synthetic forward prices
    consistent with each candidate's hidden alpha. Later checkpoints see more
    noise, mimicking mean-reversion / diffusion in real markets.
    """
    # Map candidate_id -> alpha via symbol+timestamp is clumsy; instead we
    # key by candidate_id indirectly by letting the oracle look up the
    # candidate's timestamp as an index into alpha.
    def price(symbol: str, t: float) -> float:
        # In this toy setup, every symbol appears multiple times but each
        # candidate has its own alpha, so we use a deterministic function of
        # (symbol, t) to produce forward returns. For simplicity, approximate
        # by looking up the alpha associated with the nearest candidate of
        # this symbol. In production, this would be a real price oracle.
        alpha = candidates_alpha.get(symbol, 0.0)
        # minutes_elapsed since reference time encoded implicitly; the oracle
        # here just returns 1 + alpha/100 + noise scaled by sqrt(time).
        noise = RNG.normal(0.0, 0.5)
        return 1.0 + alpha / 100.0 + noise / 100.0

    return price


def main() -> None:
    pairs = generate_candidates(N)
    # Average alpha per symbol (stand-in for a true price oracle).
    sym_alpha: dict[str, list[float]] = {}
    for cand, alpha in pairs:
        sym_alpha.setdefault(cand.symbol, []).append(alpha)
    symbol_alpha_mean = {s: float(np.mean(v)) for s, v in sym_alpha.items()}
    price_oracle = build_price_oracle(symbol_alpha_mean)

    tracker = RejectionTracker(price_fn=price_oracle, checkpoints_min=CHECKPOINTS_MIN)

    accepted_outcomes: list[AcceptedOutcome] = []
    n_accepted = 0
    n_rejected = 0

    for cand, alpha in pairs:
        decision = toy_filter(cand)
        if decision.accepted:
            # Simulate realized PnL as a noisy read of true alpha.
            realized = alpha + RNG.normal(0.0, 1.0)
            accepted_outcomes.append(AcceptedOutcome(cand.candidate_id, float(realized)))
            n_accepted += 1
        else:
            tracker.record(cand, decision)
            n_rejected += 1

    followup_records = tracker.run_followup()
    counterfactual = FollowupSampler.counterfactual_pnl(followup_records)

    evaluator = FilterEvaluator()
    print("=" * 72)
    print("Post-Rejection Follow-up Sampling — filter evaluation")
    print("=" * 72)
    print(f"candidates:       {N}")
    print(f"accepted:         {n_accepted}  ({n_accepted / N * 100:.1f}%)")
    print(f"rejected:         {n_rejected}  ({n_rejected / N * 100:.1f}%)")
    print()
    for horizon in CHECKPOINTS_MIN:
        result = evaluator.evaluate(accepted_outcomes, counterfactual, horizon)
        print(f"--- horizon {horizon:>5d} min ---")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k:<30s} {v:+.4f}")
            else:
                print(f"  {k:<30s} {v}")
        print()


if __name__ == "__main__":
    main()
