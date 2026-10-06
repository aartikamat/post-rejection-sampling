# Post-Rejection Follow-up Sampling (PRFS)

Reference implementation of **Post-Rejection Follow-up Sampling (PRFS)** — an **observational measurement** design for recording what happens after a filter-gated system rejects a candidate — together with verification tools that audit a rejection registry against an outcome log.

[![tests](https://github.com/aartikamat/post-rejection-sampling/actions/workflows/tests.yml/badge.svg)](https://github.com/aartikamat/post-rejection-sampling/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

The design is specified in:

Kamat, A. U. (2026). *Post-Rejection Follow-up Sampling: Measuring Outcomes of Rejected Decisions in Algorithmic DEX Trading.* arXiv: [2606.08228](https://arxiv.org/abs/2606.08228). SSRN: [abstract 6607301](https://ssrn.com/abstract=6607301).

## Motivation

Filter-gated algorithmic trading systems on decentralised exchanges reject the majority of candidate signals evaluated. While accepted trades are directly measured on the execution path, the observed forward market trajectory of rejected candidates is rarely tracked on the same live venue that produced the rejection.

**PRFS** addresses this by observing the forward price and liquidity trajectory of each rejected token over a fixed analytic horizon, from the same live data feed used by the rejecting scanner. Follow-up state is keyed by the **decision** (token, rejection timestamp, reason), not by the token, so a second rejection of the same token opens its own schedule and leaves earlier schedules running.

The design is **observational, not counterfactual**: PRFS measures the subsequent market trajectory of the specific rejected token on the specific venue where the rejection occurred. It does not compute or claim a hypothetical acceptance outcome.

## What the package provides

- **Collector building blocks:** decision-keyed fixed and adaptive schedulers, a sampler with a retry policy and an explicit observation predicate, oracle-absence records for failed observations, append-only write-ahead logs with restart recovery, and right/interval censoring labels.
- **Verification tools:** primary (three-field) and diagnostic (two-field) coverage of an outcome log against a rejection registry, per-reason coverage with Wilson 95% intervals, a registry key-uniqueness audit, sample-level alignment against two denominators, and a reason-conflict ledger. The `prfs-verify` command writes all of these to a JSON report that records the SHA-256 hashes of its inputs.
- **Cross-check aggregator:** `oracle/oracle_aggregate.py` recomputes the headline counts through a separate code path.

The package contains no exchange connector, credentials or trading logic. To collect live data, subclass `prfs.oracle_adapter.OracleAdapter` for your price feed; `prfs.simulator.DeterministicSimulator` is a ready-made offline stand-in used by the tests.

## Installation

Requires Python 3.10, 3.11, or 3.12. The core package has no third-party dependencies.

```bash
git clone https://github.com/aartikamat/post-rejection-sampling.git
cd post-rejection-sampling
python -m pip install -e .
```

The distribution name is `post-rejection-sampling`; the import package is `prfs`. An unrelated project named `prfs` exists on PyPI, so do **not** run `pip install prfs`.

For running the test suite:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Package structure

| Module | Purpose |
| --- | --- |
| `prfs.types` | Typed records: rejection events, oracle responses, follow-up samples, absence records |
| `prfs.clock` | Timestamp parsing and canonical epoch-millisecond comparison |
| `prfs.scheduler` | Fixed and adaptive follow-up schedules (default `{5, 15, 60, 240, 1440}` minutes) |
| `prfs.oracle_adapter` | Abstract price-feed interface, retry policy and observation predicate |
| `prfs.sampler` | Turns due schedule offsets into follow-up samples or absence records |
| `prfs.persistence` | Append-only write-ahead logs, terminal markers and restart recovery |
| `prfs.censoring` | Right- and interval-censoring indicators per scheduled observation |
| `prfs.coverage` | Primary and diagnostic coverage, key-uniqueness audit, Wilson intervals |
| `prfs.reason_ledger` | Reason-conflict ledger between registry and outcome log |
| `prfs.estimand` | Observed forward returns; hypothetical PnL only with an explicit execution model |
| `prfs.provenance` | Run manifests and structured JSON-line logs |
| `prfs.simulator` | Deterministic offline oracle for testing |
| `prfs.cli` | `prfs-verify` and `prfs-run` command-line entry points |
| `oracle.oracle_aggregate` | Cross-check aggregator (separate code path) |

## Quickstart

Download `rejections.csv` and `rejection_outcomes.csv` from the example dataset (Zenodo concept DOI [10.5281/zenodo.20043515](https://doi.org/10.5281/zenodo.20043515)), then:

```python
from pathlib import Path
from prfs.coverage import primary_coverage, wilson_ci

s = primary_coverage(Path("rejections.csv"), Path("rejection_outcomes.csv"))
lo, hi = wilson_ci(s.primary_matched, s.primary_denominator)
print(f"Primary coverage: {s.primary_pct:.2f}%  95% CI: [{lo:.2%}, {hi:.2%}]")
```

Command-line usage (writes a JSON report with per-filter Wilson intervals):

```bash
prfs-verify --registry-csv rejections.csv --outcomes-csv rejection_outcomes.csv --report-out report.json
```

`prfs-run` is a placeholder entry point: it needs a project-provided `OracleAdapter` binding and is not needed to verify existing data.

## Example: verifying the archived dataset

The archived dataset is used here only as **example input**. It was **not** collected with `prfs` or under the PRFS schedule: a separate production tracker swept every tracked token every 10 minutes, followed each record for at most 24 hours and kept one record per token, so a new rejection of a tracked token restarted its record. Its first sample is at 2026-04-11T20:26:53Z, and rejections logged before then have no samples. These features of the tracker, not the design implemented in `prfs`, explain the coverage figures below.

Running the Quickstart against the archived files gives:

- Primary coverage (three-field key): **48.55%** (1,455 of 2,997; Wilson 95% CI [46.76%, 50.34%])
- Diagnostic coverage (two-field key): **54.75%** (1,641 of 2,997; Wilson 95% CI [52.97%, 56.53%])

The `checksums.txt` file in the Zenodo record pins the SHA-256 of both CSV files; `prfs-verify` records the same hashes in its report.

## Testing

```bash
python -m pytest -q                        # all 13 test modules
python -m pytest tests/test_estimand.py    # single module
python -m pytest -k "coverage"             # by keyword
```

Twenty-one tests use the example dataset: they reproduce its figures and check agreement between the two code paths. They run only when `rejections.csv` and `rejection_outcomes.csv` are placed in `data/`, and are skipped otherwise (as in CI).

CI runs the suite on Ubuntu, macOS, and Windows against Python 3.10/3.11/3.12.

## Citation

If you use PRFS in academic work, please cite:

```bibtex
@misc{kamat2026prfs,
  author        = {Kamat, Arati Uday},
  title         = {Post-Rejection Follow-up Sampling: Measuring Outcomes of Rejected Decisions in Algorithmic DEX Trading},
  year          = {2026},
  eprint        = {2606.08228},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.TR},
  url           = {https://arxiv.org/abs/2606.08228}
}
```

A GitHub "Cite this repository" button is also available in the sidebar (generated from `CITATION.cff`).

## Contributing and support

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug fixes, documentation improvements, tests, and portability fixes are welcome; methodology changes are out of scope and belong to the design paper.

Questions, bug reports and feature requests go to the [issue tracker](https://github.com/aartikamat/post-rejection-sampling/issues). The package is maintained by its author and issues are answered on a best-effort basis.

## Scope and limitations

This repository contains a **clean reference implementation** of the PRFS design, intended for reproducibility and academic extension. It is not a production trading system and does not contain live market connectors, exchange credentials, or execution logic. The design is general — it applies to any filter-gated system in which rejections outnumber executions and the rejected object remains observable.

## Competing Interests

The author declares the following competing interest: the trading system from which this reference implementation is drawn is the subject of a pending U.S. provisional patent application in the author's name. No further competing financial or personal interests are declared. The disclosed patent covers system-level architecture claims and does not restrict use of this reference implementation under its MIT licence. Operational parameters intentionally withheld from the companion manuscript (the production system's threshold values and the mapping between anonymised filter labels and internal rule identifiers) are the subject of pending intellectual-property protection; the reference-implementation default schedule disclosed above is not withheld.

## Author

**Arati Uday Kamat** — Independent Researcher
ORCID: [0009-0000-4781-312X](https://orcid.org/0009-0000-4781-312X)
SSRN author page: [https://ssrn.com/author=11111069](https://ssrn.com/author=11111069)

## License

MIT License — see [LICENSE](LICENSE) for full text.
