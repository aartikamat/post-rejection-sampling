# Post-Rejection Follow-up Sampling (PRFS)

Reference implementation of **Post-Rejection Follow-up Sampling (PRFS)** — an **observational measurement** methodology for evaluating filter behaviour in algorithmic decentralized exchange (DEX) trading systems.

This repository accompanies the following working papers:

- Kamat, A. U. (2026). *Post-rejection follow-up sampling: Measuring outcomes of rejected decisions in algorithmic DEX trading.* arXiv: [2606.08228](https://arxiv.org/abs/2606.08228). SSRN: [abstract 6607301](https://ssrn.com/abstract=6607301). Companion dataset: Zenodo concept DOI [10.5281/zenodo.20043515](https://doi.org/10.5281/zenodo.20043515).
- Kamat, A. U. (2026). *Hour-Aware Adaptive Risk Management for Autonomous Memecoin Trading: A Multi-Layer Intelligence Framework.* Zenodo. [https://doi.org/10.5281/zenodo.19670719](https://doi.org/10.5281/zenodo.19670719) — Also available on [SSRN (abstract 6564803)](https://ssrn.com/abstract=6564803).

## Motivation

Filter-gated algorithmic trading systems on decentralised exchanges reject the majority of candidate signals evaluated. While accepted trades are directly measured on the execution path, the observed forward market trajectory of rejected candidates is rarely tracked on the same live venue that produced the rejection.

**Post-Rejection Follow-up Sampling (PRFS)** addresses this by observing the forward price and liquidity trajectory of rejected tokens over a fixed analytic horizon from the same live oracle path used by the rejecting scanner. The output is a per-event observational trajectory dataset that downstream analyses can use to characterise filter behaviour against realised market outcomes, rather than against synthetic backtest reconstructions.

The methodology is **observational, not counterfactual**: PRFS measures the subsequent market trajectory of the specific rejected mint on the specific venue where the rejection occurred. It does not compute or claim a hypothetical acceptance outcome; downstream PnL, precision, and outcome-classification analyses belong to separate companion papers.

## Method summary

Given a live trading system that emits:

- A stream of **candidate signals** at time `t`
- A binary **accept/reject** decision for each candidate
- An observed **outcome** for accepted trades (realized PnL over a holding window)

PRFS additionally records, for each rejected candidate:

- The symbol, mint address, and rejection timestamp
- The reference price and liquidity at rejection time
- The set of signals that passed *up to the rejection gate* (for attribution)
- Forward prices and liquidity at scheduled cadence checkpoints (reference-implementation default `Schedule_fix = {5, 15, 60, 240, 1440}` minutes from the rejection timestamp), up to a fixed 8.60-day analytic follow-up horizon (`DELTA_MAX_MS = 743,040,000 ms`)

The forward price and liquidity trajectories of rejected candidates form the raw observational data on which downstream filter-precision audits are conducted. This reference implementation captures the trajectories; downstream PnL, precision, and outcome-classification analyses are the subject of separate companion papers.

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
  title        = {Post-rejection follow-up sampling: Measuring outcomes of rejected decisions in algorithmic DEX trading},
  year         = {2026},
  eprint       = {2606.08228},
  archivePrefix = {arXiv},
  primaryClass = {q-fin.TR},
  url          = {https://arxiv.org/abs/2606.08228}
}
```

A GitHub "Cite this repository" button is also available in the sidebar (generated from `CITATION.cff`).

## Scope and limitations

This repository contains a **clean reference implementation** of the PRFS methodology, intended for reproducibility and academic extension. It is not a production trading system and does not contain live market connectors, exchange credentials, or execution logic. The methodology is general — it applies to any algorithmic trading system in which rejections outnumber executions.

## Competing Interests

The author declares the following competing interest: the trading system from which this reference implementation is drawn is the subject of a pending U.S. provisional patent application (application number 64/099,108, filed 2026-06-25, Micro Entity status, in the author's name). No further competing financial or personal interests are declared. The disclosed patent covers system-level architecture claims and does not restrict use of this reference implementation under its MIT licence. Operational parameters intentionally withheld from the companion manuscript (the production system's specific per-filter-category horizon mapping and threshold values, the mapping between anonymised filter labels and internal rule identifiers) are the subject of pending intellectual-property protection; the reference-implementation default schedule disclosed above is not withheld.

## Author

**Arati Uday Kamat** — Independent Researcher
ORCID: [0009-0000-4781-312X](https://orcid.org/0009-0000-4781-312X)
SSRN author page: [https://ssrn.com/author=11111069](https://ssrn.com/author=11111069)

## License

MIT License — see [LICENSE](LICENSE) for full text.
