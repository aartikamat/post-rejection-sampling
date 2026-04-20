# Post-Rejection Follow-up Sampling (PRFS)

Reference implementation of **Post-Rejection Follow-up Sampling (PRFS)** — a counterfactual methodology for evaluating filter quality in algorithmic decentralized exchange (DEX) trading systems.

This repository accompanies the following peer-reviewed working papers:

- Kamat, A. U. (2026). *Post-Rejection Follow-up Sampling: A Methodology for Counterfactual Outcome Measurement in Algorithmic DEX Trading.* Zenodo. [https://doi.org/10.5281/zenodo.19671657](https://doi.org/10.5281/zenodo.19671657) — Also available on [SSRN (abstract 6607301)](http://ssrn.com/abstract=6607301).
- Kamat, A. U. (2026). *Hour-Aware Adaptive Risk Management for Autonomous Memecoin Trading: A Multi-Layer Intelligence Framework.* Zenodo. [https://doi.org/10.5281/zenodo.19670719](https://doi.org/10.5281/zenodo.19670719) — Also available on [SSRN (abstract 6564803)](http://ssrn.com/abstract=6564803).

## Motivation

Algorithmic trading systems reject the vast majority of candidate signals based on filters and thresholds. While accepted trades are rigorously measured, the counterfactual performance of *rejected* signals is rarely tracked. This omission creates a blind spot: a filter that rejects 90% of its input is only valuable if the rejected population systematically underperforms the accepted population.

**Post-Rejection Follow-up Sampling (PRFS)** addresses this by tracking the forward price trajectory of rejected tokens over a fixed follow-up window, producing a quantitative estimate of what the bot would have earned (or lost) had the rejection not occurred.

## Method summary

Given a live trading system that emits:

- A stream of **candidate signals** at time `t`
- A binary **accept/reject** decision for each candidate
- An observed **outcome** for accepted trades (realized PnL over a holding window `H`)

PRFS additionally records, for each rejected candidate:

- The symbol, mint address, and rejection timestamp
- The reference price at rejection time
- The set of signals that passed *up to the rejection gate* (for attribution)
- The forward prices at checkpoints `t + Δ` for `Δ ∈ {5m, 15m, 1h, 4h, 24h}`

From this record, the counterfactual PnL distribution of the rejected population is estimated, then compared to the realized PnL distribution of the accepted population. A filter is empirically justified if and only if its rejected distribution is dominated (stochastically or in mean) by the accepted distribution under the system's utility function.

## Repository contents

| File | Description |
| --- | --- |
| `prfs.py` | Core reference implementation: `RejectionTracker`, `FollowupSampler`, `FilterEvaluator` classes. |
| `example.py` | Runnable demonstration using synthetic signal data. Produces filter-quality diagnostics. |
| `LICENSE` | MIT License. |
| `CITATION.cff` | Machine-readable academic citation metadata. |

## Installation

Requires Python 3.9+ and NumPy. No external trading infrastructure is needed to run the reference implementation — synthetic data is generated inside `example.py`.

```bash
pip install numpy
python example.py
```

## Citation

If you use PRFS in academic work or production systems, please cite:

```bibtex
@misc{kamat2026prfs,
  author       = {Kamat, Arati Uday},
  title        = {Post-Rejection Follow-up Sampling: A Methodology for Counterfactual Outcome Measurement in Algorithmic DEX Trading},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19671657},
  url          = {https://doi.org/10.5281/zenodo.19671657}
}
```

A GitHub "Cite this repository" button is also available in the sidebar (generated from `CITATION.cff`).

## Scope and limitations

This repository contains a **clean reference implementation** of the PRFS methodology, intended for reproducibility and academic extension. It is not a production trading system and does not contain live market connectors, exchange credentials, or execution logic. The methodology is general — it applies to any algorithmic trading system in which rejections outnumber executions.

## Author

**Arati Uday Kamat** — Independent Researcher
ORCID: [0009-0000-4781-312X](https://orcid.org/0009-0000-4781-312X)
SSRN author page: [https://ssrn.com/author=11111069](https://ssrn.com/author=11111069)
Zenodo: [https://zenodo.org/me](https://zenodo.org/me)

## License

MIT License — see [LICENSE](LICENSE) for full text.
