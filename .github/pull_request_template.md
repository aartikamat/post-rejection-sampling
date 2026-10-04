## Summary

Describe what this PR changes and why.

## Scope check

- [ ] The change does not alter the estimand or its confidence intervals
- [ ] The change does not alter the coverage or censoring definitions
- [ ] With the example dataset in `data/`, `prfs-verify` still reports
      1,455 / 2,997 primary matches, 1,641 / 2,997 diagnostic matches,
      253 reason-conflict rows and 263 orphan outcome keys

## Testing

- [ ] Existing tests still pass locally (`python -m pytest -q`)
- [ ] New behaviour is covered by tests (if applicable)
- [ ] CI is green

## Documentation

- [ ] README updated if user-facing behaviour changed
- [ ] Docstrings updated for changed functions
- [ ] CHANGELOG.md updated under an "Unreleased" section

## Additional notes

Anything else the reviewer should know.
