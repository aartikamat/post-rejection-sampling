# Contributing

Thanks for your interest in `prfs`. This project accompanies a research
manuscript on Post-Rejection Follow-up Sampling (PRFS), an observational
measurement methodology for algorithmic DEX trading. Contributions that
improve the reference implementation, its tests, its documentation, or its
reproducibility are welcome.

## Scope

`prfs` is a **reference implementation of a measurement methodology**,
not a trading system. Contributions that change the methodology itself
(coverage definition, censoring rules, estimand, reason-ledger schema) are
out of scope for this repository — those belong to the design paper
and require a formal amendment. Contributions that improve the code without
changing behavior are always welcome.

In-scope changes:

- Bug fixes in the reference implementation
- Additional tests, especially edge cases
- Documentation improvements (README, docstrings, examples)
- Portability fixes (Python 3.10-3.12 on Linux/macOS/Windows)
- CI improvements
- Type hints, style, refactors that preserve behavior

Out-of-scope changes (please open an issue to discuss first):

- Changes to the estimand or its confidence intervals
- Changes to the coverage or censoring definitions
- Additions of new trading strategies or exchanges
- Anything that reads live blockchain data

## Development setup

```bash
git clone https://github.com/aartikamat/post-rejection-sampling.git
cd post-rejection-sampling
python -m pip install -e ".[test]"
python -m pytest -q
```

Python 3.10, 3.11, and 3.12 are all supported.

## Testing

All contributions must pass the existing test suite. If you fix a bug,
add a test that fails before your fix and passes after. If you add a
feature, add tests that cover both the happy path and at least one
failure mode.

```bash
python -m pytest -q                       # all tests
python -m pytest tests/test_estimand.py   # single file
python -m pytest -k "coverage"            # by keyword
```

## Commit and PR style

- One logical change per commit; small commits are preferred to large ones.
- Commit messages: imperative present tense (`Fix off-by-one in censoring`
  rather than `Fixed off-by-one` or `Fixes off-by-one`).
- PRs should describe what changed, why, and how it was tested.
- Reference the issue number in the PR description if one exists.

## Reporting issues

Please use the issue templates. For bugs, include a minimal reproducer,
the exact Python and OS versions, and the full traceback. For enhancement
requests, describe the use case first and the proposed change second.

## Getting support

Questions about installing or using `prfs` are welcome as GitHub issues
(start the title with `[QUESTION]`). Please say what you tried, what you expected, and
what happened.

## Governance and maintenance

`prfs` is maintained by its author, who reviews and merges all changes.
Issues and pull requests are answered on a best-effort basis. Decisions
about scope follow the in-scope and out-of-scope lists above; changes to
the measurement design itself are made only through the design paper.
Releases follow semantic versioning and are listed in `CHANGELOG.md`.

## Code of conduct

Be respectful. This is a small research project; disagreements about
methodology or code should stay focused on the technical question.

## Licence

By contributing, you agree that your contributions will be licensed under
the MIT License (see `LICENSE`).
