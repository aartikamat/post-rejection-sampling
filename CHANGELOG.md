# CHANGELOG, Paper 2 / PRFS companion reproducibility package

## v2.1 (2026-09-15), PUBLIC-REPOSITORY REPACKAGING (concept-DOI convergence)

CSV data files, reference implementation code, tests, cross-check aggregator,
and computed results remain byte-identical to v2.0. The v2.1 release refreshes
the surrounding documentation so that every reference to a Zenodo record uses
the concept DOI (which auto-resolves to the latest version) rather than a
version-specific DOI:

- `README.md` and `CITATION.cff` now cite the companion dataset at concept
  DOI `10.5281/zenodo.20043515` (not the version DOI `20043516`), the
  reference implementation at concept DOI `10.5281/zenodo.19672363` (not
  version DOI `19672364`), and the companion paper preprint at concept DOI
  `10.5281/zenodo.20499924` (not version DOI `20499925`).
- The manuscript's canonical vector figures `Figure_2_sample_density.svg`
  and `Figure_3_filter_volume.svg` are shipped in `figures/`; the
  matplotlib regenerator `figures_gen.py` remains alongside them for
  numerical-content verification.
- `PACKAGE_MANIFEST.md` and `SHA256SUMS` at the package root name every
  shipped file with its SHA-256; the `data/checksums.txt` continues to
  cover the frozen CSVs against the byte-identical v1 hashes.
- Descriptive text throughout aligns with the seven-anonymised-filter
  layout (`filter_1` through `filter_7`) and states explicitly that the
  numeric ordering of the labels does not encode rejection volume or any
  other observable property of the underlying rules.

No scientific claim, no numerical result, and no test outcome changes in
this repackaging.

## v2.0 (2026-08-23), SCIENTIFIC CONSISTENCY REMEDIATION

CSV data files remain byte-identical to the original v1 deposit. Headline
numbers are unchanged. The v2 release resolves seven scientific and
manuscript-consistency defects identified in the D10, D13, D14, D15, D16,
D17, and D18 pre-audit findings:

- **D14 timestamp normalisation.** Timestamp identity in the coverage and
  reason-attribution pipeline is decided on the CANONICAL epoch-millisecond
  form (`prfs.clock.to_epoch_ms`). Raw literals are preserved on every row
  for byte-provenance audit. New helpers: `parse_iso_ms`, `to_epoch_ms`,
  `canonicalise`, `raw_eq`, `span_days`, and a `ClockParseError` that
  surfaces the offending literal instead of silently dropping the row. On
  the deposit both forms agree; the canonical comparison guards against
  future mixed-precision inputs. New tests in
  `tests/test_clock.py::test_to_epoch_ms_treats_integer_second_and_millisecond_forms_equally`
  and `tests/test_reason_ledger.py::test_timestamp_normalisation_string_variability`.

- **D14 registry key uniqueness.** `prfs.coverage.assert_registry_key_uniqueness`
  reports total rows, distinct 3-field keys, duplicate 3-field key groups
  and rows, conflicting duplicate 2-field keys, and exact-duplicate 3-field
  key groups. Called by every entry-point; halts with `KeyUniquenessError`
  on any unexpected duplicate. New tests in
  `tests/test_coverage.py::test_registry_key_uniqueness_asserted` and
  `test_registry_key_uniqueness_halts_on_duplicate`. Verify report now
  surfaces the report under `registry_key_uniqueness`.

- **D14 dual sample-alignment denominators.** `CoverageSummary` now carries
  `sample_denominator_full` (67,000) and `sample_denominator_eligible`
  (67,000 minus the 8,593 orphan-key rows) with matched percentages
  reported against each. Every downstream artefact states both denominators.

- **D14 honest oracle language.** `oracle/oracle_aggregate.py` renamed from
  "independent oracle" to "cross-check aggregator" in every occurrence.
  Docstrings clarify that the module shares the same logical
  set-intersection framework and same Python runtime as the primary
  implementation; agreement is CROSS-VERIFICATION, not language- or
  framework-independent verification. The aggregator also now canonicalises
  timestamps via `prfs.clock.to_epoch_ms` for parity with the primary path.

- **D15 SET vs MULTISET decision.** After inspection of the raw reason
  fields (registry: 2,997 rows, 2,997 distinct 3-field keys, zero within-
  key duplicates; outcomes: 800 within-key duplicate-same-reason cases from
  scheduled-sample repetition), the ledger continues to use SET semantics
  on the distinct reasons at each (mint, timestamp) key. Every use of
  "multiset" in the manuscript, README, and reference-implementation docs
  was replaced with "set of distinct reasons"; the collapse of within-key
  duplicate same-reason sample rows is stated explicitly. Design rationale
  is in `decision_docs/SET_VS_MULTISET_DECISION.md`; the intent is enforced
  by `tests/test_reason_ledger.py::test_reason_set_semantics_within_key_duplicates_collapse`
  and `test_reason_set_semantics_disagreement_emits_conflict`.

- **D16 confidence labels.** `prfs.reason_ledger.Confidence` is now
  `Literal["MEDIUM", "HIGH"]` (previously included LOW, which the code
  never emitted). The manuscript describes the two-level system explicitly.
  No post-hoc LOW threshold was added; the prospective LOW category was
  not defined in the specification. New test:
  `tests/test_reason_ledger.py::test_confidence_labels_are_two_level`.

- **D17 per-filter uncertainty.** `prfs.coverage.wilson_ci` and
  `per_filter_coverage_with_ci` provide two-sided 95 percent Wilson
  binomial confidence intervals on every per-filter row and flag
  categories whose CI width exceeds 30 percentage points with a
  sparse-category warning. The manuscript per-filter table now carries
  CI columns and hedges comparative claims on small-n categories. New
  tests: `test_wilson_ci_shape` and `test_per_filter_coverage_with_ci_sparse_warning`.

- **D18 time-window reconciliation.** The analytic follow-up window
  (`DELTA_MAX_MS = 743,040,000 ms = 8.60 days exactly`, `FORMAL_SPEC`
  section 2) and the observed calendar span of the registry timestamps
  (8.626 days, rounded 8.63) are now surfaced under distinct keys in every
  report. The manuscript states the two quantities as different things.
  `prfs.clock.span_days` provides the observed-span helper. New test:
  `tests/test_clock.py::test_span_days_matches_observed_calendar_span_of_deposit`.

- **D13 figures.** The two figures described in the paper are regenerated
  deterministically from the frozen data by `figures_gen.py` and shipped in
  `figures/` with a SHA-256 manifest (`figures/figures_checksums.txt`).
  Every plotted value traces to `prfs.coverage` / `prfs.clock` and the
  deposit CSVs; no hand-authored figure content. Both figures are provided
  as 300 dpi PNG raster and SVG vector.

- **D10 title and causal language.** The manuscript title is now
  "Post-Rejection Follow-up Sampling: An Observational Measurement
  Methodology for Algorithmically Rejected Trading Events". The words
  "counterfactual", "causal", "treatment", "effect", "impact", "leads to",
  and "caused by" occur only inside explicit disclaimers of identification.
  The prior "notice of correction" block that appeared before the abstract
  is removed from the manuscript; version history lives in this CHANGELOG,
  the reproducibility package, and the cover letter shipped only to the
  receiving editor.

## v1.0 (2026-08-23), initial reproducibility package

- Introduced the reference implementation `prfs` v2.0.0 with 54 pytest
  tests, all passing, zero skips.
- Added the secondary cross-check aggregator `oracle/oracle_aggregate.py`
  (labelled "independent oracle" in v1; corrected in v2).
- Added `REASON_CONFLICT_LEDGER.csv` (253 rows) with a two-field-and-
  reason-attribution disposition rule.
- CSV data files: byte-identical to the original 2026-04-19 v1 deposit.
